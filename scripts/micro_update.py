import copy

import torch

from grpo_mini.grpo import grpo_objective
from grpo_mini.model import TinyAutoregressiveTransformer, freeze, make_legal_mask
from grpo_mini.sampling import sample_group


def collate_groups(samples):
    max_length = max(tokens.shape[1] for tokens, _ in samples)
    batch_tokens = []
    batch_masks = []
    group_ids = []

    for group_id, (tokens, response_mask) in enumerate(samples):
        padding_length = max_length - tokens.shape[1]
        if padding_length:
            tokens = torch.cat(
                [
                    tokens,
                    torch.zeros(
                        tokens.shape[0],
                        padding_length,
                        dtype=tokens.dtype,
                    ),
                ],
                dim=1,
            )
            response_mask = torch.cat(
                [
                    response_mask,
                    torch.zeros(
                        response_mask.shape[0],
                        padding_length,
                        dtype=response_mask.dtype,
                    ),
                ],
                dim=1,
            )
        batch_tokens.append(tokens)
        batch_masks.append(response_mask)
        group_ids.extend([group_id] * tokens.shape[0])

    return (
        torch.cat(batch_tokens, dim=0),
        torch.cat(batch_masks, dim=0),
        torch.tensor(group_ids, dtype=torch.long),
    )


def evaluate(model, batch, old_model, reference_model, legal_mask, reduction):
    tokens, response_mask, group_ids, rewards = batch
    with torch.no_grad():
        output = grpo_objective(
            model=model,
            tokens=tokens,
            response_mask=response_mask,
            group_ids=group_ids,
            rewards=rewards,
            legal_mask=legal_mask,
            epsilon=0.2,
            beta=0.1,
            reduction=reduction,
            old_model=old_model,
            reference_model=reference_model,
        )
    return output


def main():
    torch.manual_seed(20260930)
    vocab_size = 5
    pad_id = 0
    eos_id = 4
    legal_mask = make_legal_mask(vocab_size, pad_id)

    old_model = freeze(
        TinyAutoregressiveTransformer(
            vocab_size=vocab_size,
            pad_id=pad_id,
            d_model=8,
            nhead=2,
            dim_feedforward=16,
        ).double()
    )
    with torch.no_grad():
        old_model.head.bias[eos_id] = 2.0

    reference_model = freeze(copy.deepcopy(old_model))
    current_model = copy.deepcopy(old_model)
    current_model.train()
    for parameter in current_model.parameters():
        parameter.requires_grad_(True)
    with torch.no_grad():
        current_model.head.weight.add_(torch.randn_like(current_model.head.weight) * 0.02)
        current_model.head.bias.add_(torch.randn_like(current_model.head.bias) * 0.02)

    prompts = [
        torch.tensor([1, 2, 3], dtype=torch.long),
        torch.tensor([2, 3, 1], dtype=torch.long),
        torch.tensor([3, 1, 2], dtype=torch.long),
    ]
    group_sizes = [4, 4, 3]
    generator = torch.Generator().manual_seed(930)
    samples = [
        sample_group(
            old_model,
            prompt,
            group_size=group_size,
            max_new_tokens=5,
            eos_id=eos_id,
            legal_mask=legal_mask,
            generator=generator,
        )
        for prompt, group_size in zip(prompts, group_sizes)
    ]

    tokens, response_mask, group_ids = collate_groups(samples)
    rewards = torch.tensor(
        [
            1.0,
            1.0,
            1.0,
            1.0,
            0.0,
            1.0,
            2.0,
            3.0,
            0.0,
            0.0,
            4.0,
        ],
        dtype=torch.float64,
    )
    batch = (tokens, response_mask, group_ids, rewards)

    def loss_for(reduction):
        return grpo_objective(
            model=current_model,
            tokens=tokens,
            response_mask=response_mask,
            group_ids=group_ids,
            rewards=rewards,
            legal_mask=legal_mask,
            epsilon=0.2,
            beta=0.1,
            reduction=reduction,
            old_model=old_model,
            reference_model=reference_model,
        ).loss

    current_model.zero_grad(set_to_none=True)
    sequence_loss = loss_for("sequence")
    sequence_loss.backward()
    parameter = current_model.head.weight
    row, column = 1, 2
    analytical_gradient = parameter.grad[row, column].item()

    epsilon = 1e-6
    original_value = parameter.data[row, column].item()
    parameter.data[row, column] = original_value + epsilon
    positive_loss = loss_for("sequence").item()
    parameter.data[row, column] = original_value - epsilon
    negative_loss = loss_for("sequence").item()
    parameter.data[row, column] = original_value
    numerical_gradient = (positive_loss - negative_loss) / (2 * epsilon)

    before_sequence = evaluate(
        current_model,
        batch,
        old_model,
        reference_model,
        legal_mask,
        "sequence",
    )
    before_token = evaluate(
        current_model,
        batch,
        old_model,
        reference_model,
        legal_mask,
        "token",
    )

    gradient_norm = torch.nn.utils.clip_grad_norm_(current_model.parameters(), max_norm=10.0).item()
    optimizer = torch.optim.SGD(current_model.parameters(), lr=0.2)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    after_sequence = evaluate(
        current_model,
        batch,
        old_model,
        reference_model,
        legal_mask,
        "sequence",
    )
    after_token = evaluate(
        current_model,
        batch,
        old_model,
        reference_model,
        legal_mask,
        "token",
    )

    lengths = response_mask.sum(dim=1)
    print(f"group_sizes={group_sizes}")
    print(f"response_lengths={lengths.tolist()}")
    print(f"rewards={rewards.tolist()}")
    print(f"advantages={before_sequence.advantages.tolist()}")
    print(
        "before sequence "
        f"objective={before_sequence.objective.item():.8f} "
        f"policy={before_sequence.policy_objective.item():.8f} "
        f"kl={before_sequence.kl_penalty.item():.8f} "
        f"loss={before_sequence.loss.item():.8f}"
    )
    print(
        "before token    "
        f"objective={before_token.objective.item():.8f} "
        f"policy={before_token.policy_objective.item():.8f} "
        f"kl={before_token.kl_penalty.item():.8f} "
        f"loss={before_token.loss.item():.8f}"
    )
    print(
        "gradient_check "
        f"head.weight[{row},{column}] analytical={analytical_gradient:.8f} "
        f"numerical={numerical_gradient:.8f} "
        f"abs_error={abs(analytical_gradient - numerical_gradient):.3e}"
    )
    print(f"gradient_norm={gradient_norm:.8f} learning_rate=0.2 optimizer=SGD")
    print(
        "after sequence  "
        f"objective={after_sequence.objective.item():.8f} "
        f"policy={after_sequence.policy_objective.item():.8f} "
        f"kl={after_sequence.kl_penalty.item():.8f} "
        f"loss={after_sequence.loss.item():.8f}"
    )
    print(
        "after token     "
        f"objective={after_token.objective.item():.8f} "
        f"policy={after_token.policy_objective.item():.8f} "
        f"kl={after_token.kl_penalty.item():.8f} "
        f"loss={after_token.loss.item():.8f}"
    )


if __name__ == "__main__":
    main()
