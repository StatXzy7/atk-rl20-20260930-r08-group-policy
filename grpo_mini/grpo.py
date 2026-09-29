from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class GRPOOutput:
    loss: torch.Tensor
    objective: torch.Tensor
    policy_objective: torch.Tensor
    kl_penalty: torch.Tensor
    advantages: torch.Tensor
    ratios: torch.Tensor
    sequence_lengths: torch.Tensor
    sequence_policy: torch.Tensor
    sequence_kl: torch.Tensor
    token_policy: torch.Tensor
    token_kl: torch.Tensor
    response_mask: torch.Tensor


def group_advantages(rewards, group_ids):
    advantages = torch.zeros_like(rewards)
    unique_group_ids = torch.unique(group_ids)

    for group_id in unique_group_ids:
        indices = torch.nonzero(group_ids == group_id, as_tuple=False).squeeze(1)
        if indices.numel() < 2:
            raise ValueError(f"group {int(group_id.item())} must contain at least two sequences")
        group_rewards = rewards[indices]
        standard_deviation = torch.std(group_rewards, unbiased=False)
        if standard_deviation == 0.0:
            advantages[indices] = 0.0
        else:
            advantages[indices] = (group_rewards - group_rewards.mean()) / standard_deviation

    return advantages.detach()


def masked_log_softmax(logits, legal_mask):
    negative_infinity = torch.finfo(logits.dtype).min
    masked = logits.masked_fill(
        ~legal_mask.view(*([1] * (logits.ndim - 1)), -1),
        negative_infinity,
    )
    return F.log_softmax(masked, dim=-1)


def explicit_kl(current_logits, reference_logits, legal_mask):
    mask = legal_mask.to(device=current_logits.device, dtype=torch.bool)
    current_log_probabilities = masked_log_softmax(current_logits, mask)
    reference_log_probabilities = masked_log_softmax(reference_logits, mask)
    current_probabilities = current_log_probabilities.exp()
    reference_probabilities = reference_log_probabilities.exp()

    legal = mask.view(*([1] * (current_logits.ndim - 1)), -1)
    if not torch.all(current_probabilities.masked_select(legal) > 0.0):
        raise ValueError("every legal current probability must be positive")
    if not torch.all(reference_probabilities.masked_select(legal) > 0.0):
        raise ValueError("every legal reference probability must be positive")

    safe_current_log_probabilities = current_log_probabilities.masked_fill(~legal, 0.0)
    safe_reference_log_probabilities = reference_log_probabilities.masked_fill(~legal, 0.0)
    return (
        current_probabilities
        * (safe_current_log_probabilities - safe_reference_log_probabilities)
    ).sum(dim=-1)


def clipped_surrogate(advantage, ratio, epsilon):
    clipped_ratio = torch.clamp(ratio, 1.0 - epsilon, 1.0 + epsilon)
    return torch.minimum(ratio * advantage, clipped_ratio * advantage)


def sequence_masked_means(values, mask):
    lengths = mask.sum(dim=1)
    if torch.any(lengths == 0):
        raise ValueError("every sequence must contain at least one response token")
    return (values * mask).sum(dim=1) / lengths, lengths


def _assert_frozen(model, name):
    if model is not None and any(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError(f"{name} must be frozen")


def grpo_objective(
    *,
    model,
    tokens,
    response_mask,
    group_ids,
    rewards,
    legal_mask,
    epsilon=0.2,
    beta=0.01,
    reduction="sequence",
    old_model=None,
    old_logprobs=None,
    reference_model=None,
):
    if reduction not in {"sequence", "token"}:
        raise ValueError("reduction must be 'sequence' or 'token'")

    tokens = tokens.long()
    response_mask = response_mask.bool()
    group_ids = group_ids.long()
    rewards = rewards.detach().to(dtype=next(model.parameters()).dtype)
    legal_mask = legal_mask.to(device=tokens.device, dtype=torch.bool)

    if tokens.ndim != 2 or response_mask.shape != tokens.shape:
        raise ValueError("tokens and response_mask must have matching shape [batch, time]")
    if legal_mask.sum() < 2:
        raise ValueError("at least two legal vocabulary tokens are required")
    if tokens.shape[1] < 2:
        raise ValueError("each sequence must contain a prefix and one response token")

    batch_size, _ = tokens.shape
    if group_ids.shape != (batch_size,):
        raise ValueError("group_ids must have shape [batch]")
    if rewards.shape != (batch_size,):
        raise ValueError("rewards must have shape [batch]")

    target_response_mask = response_mask[:, 1:]
    if torch.any(target_response_mask.sum(dim=1) == 0):
        raise ValueError("every sequence must contain at least one response target")
    illegal_token_ids = torch.nonzero(~legal_mask, as_tuple=False).flatten()
    if torch.any(torch.isin(tokens, illegal_token_ids) & response_mask):
        raise ValueError("response tokens must be legal")

    advantages = group_advantages(rewards, group_ids).to(device=tokens.device)
    target_tokens = tokens[:, 1:]

    current_logits = model(tokens, legal_mask)[:, :-1]
    current_log_probabilities = masked_log_softmax(current_logits, legal_mask)
    current_logprobs = torch.gather(
        current_log_probabilities,
        dim=-1,
        index=target_tokens.unsqueeze(-1),
    ).squeeze(-1)

    if old_logprobs is None:
        if old_model is None:
            raise ValueError("old_model or old_logprobs is required")
        _assert_frozen(old_model, "old_model")
        with torch.no_grad():
            old_logits = old_model(tokens, legal_mask)[:, :-1]
            old_log_probabilities = masked_log_softmax(old_logits, legal_mask)
            old_logprobs = torch.gather(
                old_log_probabilities,
                dim=-1,
                index=target_tokens.unsqueeze(-1),
            ).squeeze(-1)
    else:
        old_logprobs = old_logprobs.detach()
        if old_logprobs.shape != target_tokens.shape:
            raise ValueError("old_logprobs must have shape [batch, time-1]")

    ratios = torch.exp(current_logprobs - old_logprobs)
    token_policy = clipped_surrogate(
        advantages.unsqueeze(1),
        ratios,
        epsilon,
    )

    if beta == 0.0:
        token_kl = torch.zeros_like(token_policy)
    else:
        if reference_model is None:
            raise ValueError("reference_model is required when beta is nonzero")
        _assert_frozen(reference_model, "reference_model")
        with torch.no_grad():
            reference_logits = reference_model(tokens, legal_mask)[:, :-1]
        token_kl = explicit_kl(current_logits, reference_logits, legal_mask)
        token_kl = token_kl * target_response_mask

    sequence_policy, sequence_lengths = sequence_masked_means(token_policy, target_response_mask)
    sequence_kl, _ = sequence_masked_means(token_kl, target_response_mask)

    if reduction == "sequence":
        policy_objective = sequence_policy.mean()
        kl_penalty = sequence_kl.mean()
    else:
        policy_objective = (token_policy * target_response_mask).sum() / sequence_lengths.sum()
        kl_penalty = (token_kl * target_response_mask).sum() / sequence_lengths.sum()

    objective = policy_objective - beta * kl_penalty
    loss = -objective

    return GRPOOutput(
        loss=loss,
        objective=objective,
        policy_objective=policy_objective,
        kl_penalty=kl_penalty,
        advantages=advantages,
        ratios=ratios,
        sequence_lengths=sequence_lengths,
        sequence_policy=sequence_policy,
        sequence_kl=sequence_kl,
        token_policy=token_policy,
        token_kl=token_kl,
        response_mask=target_response_mask,
    )
