import torch
from torch import Tensor


SEQUENCE_MEAN = "sequence_mean"
TOKEN_MEAN = "token_mean"
GRPO_LOSSES = (SEQUENCE_MEAN, TOKEN_MEAN)


def _as_legal_mask(legal_vocab_mask: Tensor | None, shape: tuple[int, ...], device: torch.device) -> Tensor:
    vocab_size = shape[-1]
    if legal_vocab_mask is None:
        return torch.ones(*shape, dtype=torch.bool, device=device)
    mask = legal_vocab_mask.to(device=device, dtype=torch.bool)
    return torch.broadcast_to(mask, shape).clone()


def categorical_log_probs(
    logits: Tensor,
    legal_vocab_mask: Tensor | None = None,
) -> Tensor:
    if logits.shape[-1] == 0:
        raise ValueError("The vocabulary must be non-empty.")
    legal = _as_legal_mask(legal_vocab_mask, logits.shape, logits.device)
    if not legal.any(dim=-1).all():
        raise ValueError("Every distribution needs at least one legal action.")
    if not torch.isfinite(logits[legal]).all():
        raise ValueError("Logits for legal actions must be finite.")

    masked_logits = logits.masked_fill(~legal, torch.finfo(logits.dtype).min)
    log_probs = torch.log_softmax(masked_logits, dim=-1)
    legal_log_probs = log_probs[legal]
    if not torch.isfinite(legal_log_probs).all():
        raise ValueError("Legal log-probabilities must be finite.")

    probs = log_probs.exp()
    legal_probs = probs[legal]
    if not (legal_probs > 0).all():
        raise ValueError("All legal probabilities must be strictly positive.")
    return log_probs


def categorical_kl(
    policy_logits: Tensor,
    reference_logits: Tensor,
    legal_vocab_mask: Tensor | None = None,
) -> Tensor:
    legal = _as_legal_mask(legal_vocab_mask, policy_logits.shape, policy_logits.device)
    policy_log_probs = categorical_log_probs(policy_logits, legal)
    reference_log_probs = categorical_log_probs(reference_logits, legal)
    policy_probs = policy_log_probs.exp()
    difference = legal.to(dtype=policy_log_probs.dtype) * (policy_log_probs - reference_log_probs)
    return (policy_probs * difference).sum(dim=-1)


def _response_prefix_mask(response_mask: Tensor) -> Tensor:
    mask = response_mask.to(dtype=torch.bool)
    if mask.dim() != 2:
        raise ValueError("response_mask must have shape [batch, total_length].")
    if mask[:, 0].any():
        raise ValueError("A response token cannot appear before its prompt prefix.")
    return mask[:, 1:]


def _masked_reduction(values: Tensor, mask: Tensor, reduction: str) -> Tensor:
    if values.shape != mask.shape:
        raise ValueError("Values and mask must have the same shape.")
    counts = mask.sum(dim=1)
    if not torch.all(counts > 0):
        raise ValueError("Every completion must contain at least one response token.")

    mask_float = mask.to(dtype=values.dtype)
    sequence_values = (values * mask_float).sum(dim=1) / counts.to(values.dtype)
    if reduction == SEQUENCE_MEAN:
        return sequence_values.mean()
    if reduction == TOKEN_MEAN:
        return (values * mask_float).sum() / mask.sum().to(values.dtype)
    raise ValueError(f"Unknown reduction {reduction!r}; expected one of {GRPO_LOSSES}.")


def compute_group_advantages(rewards: Tensor, group_ids: Tensor) -> Tensor:
    if rewards.dim() != 1 or group_ids.shape != rewards.shape:
        raise ValueError("rewards and group_ids must be one-dimensional and have equal length.")
    if rewards.numel() == 0:
        raise ValueError("At least one completion is required.")

    with torch.no_grad():
        rewards = rewards.detach()
        group_ids = group_ids.detach()
        advantages = torch.empty_like(rewards)

        for group_id in torch.unique(group_ids):
            index = torch.nonzero(group_ids == group_id, as_tuple=False).squeeze(1)
            if index.numel() < 2:
                raise ValueError(f"Group {group_id.item()} has size {index.numel()}; groups require G >= 2.")
            group_rewards = rewards[index]
            mean = group_rewards.mean()
            population_std = torch.sqrt(((group_rewards - mean) ** 2).mean())
            if population_std == 0:
                advantages[index] = 0
            else:
                advantages[index] = (group_rewards - mean) / population_std

    return advantages


def grpo_objective(
    policy_logits: Tensor,
    old_logits: Tensor,
    reference_logits: Tensor,
    input_ids: Tensor,
    response_mask: Tensor,
    advantages: Tensor,
    clip_epsilon: float = 0.2,
    beta: float = 0.04,
    reduction: str = SEQUENCE_MEAN,
    legal_vocab_mask: Tensor | None = None,
) -> dict[str, Tensor]:
    if reduction not in GRPO_LOSSES:
        raise ValueError(f"Unknown reduction {reduction!r}; expected one of {GRPO_LOSSES}.")
    if not 0.0 < clip_epsilon <= 1.0:
        raise ValueError("clip_epsilon must be in the interval (0, 1].")
    if beta < 0.0:
        raise ValueError("beta must be non-negative.")

    batch_size, total_length, vocab_size = policy_logits.shape
    if input_ids.shape != (batch_size, total_length):
        raise ValueError("input_ids must have shape [batch, total_length].")
    if response_mask.shape != (batch_size, total_length):
        raise ValueError("response_mask must have shape [batch, total_length].")
    if advantages.shape != (batch_size,):
        raise ValueError("advantages must have shape [batch].")
    if old_logits.shape != policy_logits.shape or reference_logits.shape != policy_logits.shape:
        raise ValueError("Policy, old, and reference logits must have the same shape.")

    prefix_mask = _response_prefix_mask(response_mask)
    prefix_logits_shape = (batch_size, total_length - 1, vocab_size)
    legal = _as_legal_mask(legal_vocab_mask, prefix_logits_shape, policy_logits.device)

    policy_prefix_logits = policy_logits[:, :-1, :]
    old_prefix_logits = old_logits[:, :-1, :].detach()
    reference_prefix_logits = reference_logits[:, :-1, :].detach()
    response_tokens = input_ids[:, 1:].unsqueeze(-1)

    policy_log_probs = categorical_log_probs(policy_prefix_logits, legal).gather(-1, response_tokens).squeeze(-1)
    old_log_probs = categorical_log_probs(old_prefix_logits, legal).gather(-1, response_tokens).squeeze(-1)

    advantages = advantages.detach()
    ratio = torch.exp(policy_log_probs - old_log_probs)
    active_ratio = ratio[prefix_mask]
    if not torch.isfinite(active_ratio).all() or not (active_ratio > 0).all():
        raise ValueError("All sampled-token probability ratios must be finite and positive.")

    clipped_ratio = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon)
    unclipped_term = advantages[:, None] * ratio
    clipped_term = advantages[:, None] * clipped_ratio
    token_policy = torch.minimum(unclipped_term, clipped_term)

    flat_legal = _as_legal_mask(
        legal_vocab_mask,
        (batch_size * (total_length - 1), vocab_size),
        policy_logits.device,
    )
    token_kl = categorical_kl(
        policy_prefix_logits.reshape(-1, vocab_size),
        reference_prefix_logits.reshape(-1, vocab_size),
        flat_legal,
    ).reshape(batch_size, total_length - 1)

    policy_objective = _masked_reduction(token_policy, prefix_mask, reduction)
    kl_term = _masked_reduction(token_kl, prefix_mask, reduction)
    kl_penalty = beta * kl_term
    regularized_objective = policy_objective - kl_penalty

    return {
        "loss": -regularized_objective,
        "regularized_objective": regularized_objective,
        "policy_objective": policy_objective,
        "kl_term": kl_term,
        "kl_penalty": kl_penalty,
        "token_policy": token_policy,
        "token_kl": token_kl,
        "ratio": ratio,
        "clipped_ratio": clipped_ratio,
        "prefix_mask": prefix_mask,
    }
