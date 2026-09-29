import copy
import json

import torch

from grpo_lab import (
    TinyAutoregressiveRNN,
    compute_group_advantages,
    freeze_model,
    frozen_clone,
    grpo_objective,
    sample_completion_batch,
)


def reward_completion(tokens: torch.Tensor, mask: torch.Tensor, prompt_id: int) -> float:
    reward = 0.0
    for index, (token, active) in enumerate(zip(tokens.tolist(), mask.tolist())):
        if not active:
            continue
        expected = 1 + ((index + prompt_id) % 2)
        reward += 1.0 if token == expected else -0.25
    return reward


def branch_counts(result, advantages, mask):
    ratio = result["ratio"]
    active = mask
    broadcast_advantages = advantages[:, None].expand_as(ratio)
    lower = active & broadcast_advantages.lt(0) & ratio.lt(0.8)
    upper = active & broadcast_advantages.gt(0) & ratio.gt(1.2)
    return {
        "active_response_tokens": int(active.sum().item()),
        "lower_ratio_negative_advantage": int(lower.sum().item()),
        "upper_ratio_positive_advantage": int(upper.sum().item()),
    }


def parameter_summary(named_parameters):
    return {name: value.detach().norm().item() for name, value in named_parameters}


def one_update(policy, old_policy, reference_policy, batch, reduction, *, beta=0.04):
    old_policy.eval()
    reference_policy.eval()
    policy.train()
    with torch.no_grad():
        old_logits = old_policy(batch["input_ids"])
        reference_logits = reference_policy(batch["input_ids"])

    optimizer = torch.optim.SGD(policy.parameters(), lr=0.2)
    optimizer.zero_grad(set_to_none=True)
    result = grpo_objective(
        policy(batch["input_ids"]),
        old_logits,
        reference_logits,
        batch["input_ids"],
        batch["response_mask"],
        batch["advantages"],
        beta=beta,
        reduction=reduction,
        legal_vocab_mask=batch["legal_vocab_mask"],
    )
    before = parameter_summary(policy.named_parameters())
    result["loss"].backward()
    gradients = {}
    for name, parameter in policy.named_parameters():
        if parameter.grad is None:
            gradients[name] = None
            continue
        if not torch.isfinite(parameter.grad).all():
            raise RuntimeError(f"Non-finite gradient for {name}.")
        gradients[name] = {
            "l2_norm": parameter.grad.norm().item(),
            "max_abs": parameter.grad.abs().max().item(),
        }
    optimizer.step()
    after = parameter_summary(policy.named_parameters())
    deltas = {name: (after[name] - before[name]) for name in before}
    return result, gradients, deltas


def main():
    torch.manual_seed(1234)
    vocab_size, hidden_size, prompt_length = 5, 4, 2
    pad_id, eos_id = 0, 4
    prompts = torch.tensor([[1, 2], [2, 3]], dtype=torch.long)
    group_sizes = [2, 3]

    old_policy = freeze_model(TinyAutoregressiveRNN(vocab_size, hidden_size))
    reference_policy = frozen_clone(old_policy)
    generator = torch.Generator().manual_seed(77)
    legal_vocab_mask = torch.tensor([False, True, True, True, True])
    batch = sample_completion_batch(
        old_policy,
        prompts,
        group_sizes,
        max_new_tokens=6,
        pad_id=pad_id,
        eos_id=eos_id,
        generator=generator,
        legal_vocab_mask=legal_vocab_mask,
    )

    rewards = torch.tensor(
        [
            reward_completion(tokens, mask, int(group_id))
            for tokens, mask, group_id in zip(
                batch["response_tokens"], batch["response_tokens_mask"], batch["group_ids"]
            )
        ],
        dtype=torch.float32,
    )
    batch["rewards"] = rewards
    batch["advantages"] = compute_group_advantages(rewards, batch["group_ids"])
    batch["legal_vocab_mask"] = legal_vocab_mask

    perturbed_policy = TinyAutoregressiveRNN(vocab_size, hidden_size)
    perturbed_policy.load_state_dict(copy.deepcopy(old_policy.state_dict()))
    with torch.no_grad():
        scale = 0.75
        for parameter in perturbed_policy.parameters():
            parameter.add_(scale * torch.randn_like(parameter))

    outputs = {}
    for reduction in ("sequence_mean", "token_mean"):
        policy = TinyAutoregressiveRNN(vocab_size, hidden_size)
        policy.load_state_dict(copy.deepcopy(perturbed_policy.state_dict()))
        result, gradients, deltas = one_update(
            policy, old_policy, reference_policy, batch, reduction
        )
        outputs[reduction] = {
            "loss": result["loss"].item(),
            "regularized_objective": result["regularized_objective"].item(),
            "policy_objective": result["policy_objective"].item(),
            "kl_penalty": result["kl_penalty"].item(),
            "branches": branch_counts(
                result, batch["advantages"], result["prefix_mask"]
            ),
            "gradient_check": gradients,
            "parameter_norm_delta": deltas,
        }

    report = {
        "configuration": {
            "vocab_size": vocab_size,
            "hidden_size": hidden_size,
            "pad_id": pad_id,
            "eos_id": eos_id,
            "group_sizes": group_sizes,
            "max_new_tokens": 6,
            "beta": 0.04,
            "clip_epsilon": 0.2,
            "optimizer": "SGD(lr=0.2)",
        },
        "sampled_batch": {
            "input_ids": batch["input_ids"].tolist(),
            "response_mask": batch["response_mask"].to(torch.int8).tolist(),
            "group_ids": batch["group_ids"].tolist(),
            "response_lengths": batch["response_length"].tolist(),
            "rewards": rewards.tolist(),
            "advantages": batch["advantages"].tolist(),
            "legal_vocab_mask": legal_vocab_mask.to(torch.int8).tolist(),
        },
        "updates": outputs,
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
