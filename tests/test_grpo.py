import copy

import pytest
import torch
import torch.nn as nn

from grpo_mini.grpo import (
    clipped_surrogate,
    explicit_kl,
    grpo_objective,
    group_advantages,
)
from grpo_mini.model import TinyAutoregressiveTransformer, freeze, make_legal_mask
from grpo_mini.sampling import sample_group


class ConstantLogitsModel(nn.Module):
    def __init__(self, logits):
        super().__init__()
        self.logits = nn.Parameter(logits.clone().detach())

    def forward(self, tokens, legal_mask=None):
        return self.logits.expand(tokens.shape[0], tokens.shape[1], -1)


def test_group_advantages_are_population_normalized_and_detached():
    rewards = torch.tensor([0.0, 1.0, 2.0, 3.0])
    group_ids = torch.tensor([0, 0, 1, 1])
    advantages = group_advantages(rewards, group_ids)

    assert not advantages.requires_grad
    assert torch.allclose(advantages, torch.tensor([-1.0, 1.0, -1.0, 1.0]))


def test_zero_reward_variance_gives_strict_zero_advantages_and_policy_value():
    tokens = torch.tensor([[1, 2], [1, 3]], dtype=torch.long)
    response_mask = torch.tensor([[False, True], [False, True]])
    rewards = torch.tensor([2.0, 2.0])
    group_ids = torch.tensor([0, 0])
    legal_mask = torch.tensor([False, True, True, True])
    current_logits = torch.zeros(2, 2, 4)
    old_logits = torch.zeros(2, 2, 4)
    reference_logits = torch.zeros(2, 2, 4)
    current = ConstantLogitsModel(current_logits)
    old = freeze(ConstantLogitsModel(old_logits))
    reference = freeze(ConstantLogitsModel(reference_logits))

    output = grpo_objective(
        model=current,
        tokens=tokens,
        response_mask=response_mask,
        group_ids=group_ids,
        rewards=rewards,
        legal_mask=legal_mask,
        old_model=old,
        reference_model=reference,
        beta=0.1,
    )

    assert torch.equal(output.advantages, torch.zeros_like(output.advantages))
    assert output.policy_objective.item() == 0.0
    assert output.kl_penalty.item() == 0.0
    assert output.loss.item() == 0.0


def test_group_size_one_is_rejected_and_unequal_group_sizes_are_supported():
    rewards = torch.tensor([1.0, 2.0])
    group_ids = torch.tensor([0, 1])
    with pytest.raises(ValueError, match="at least two"):
        group_advantages(rewards, group_ids)

    tokens = torch.tensor([[1, 2]] * 5, dtype=torch.long)
    response_mask = torch.tensor([[False, True]] * 5)
    rewards = torch.tensor([1.0, 2.0, 0.0, 3.0, 6.0])
    group_ids = torch.tensor([0, 0, 1, 1, 1])
    legal_mask = torch.tensor([False, True, True, True])
    model = ConstantLogitsModel(torch.zeros(5, 2, 4))
    old = freeze(ConstantLogitsModel(torch.zeros(5, 2, 4)))

    output = grpo_objective(
        model=model,
        tokens=tokens,
        response_mask=response_mask,
        group_ids=group_ids,
        rewards=rewards,
        legal_mask=legal_mask,
        old_model=old,
        beta=0.0,
    )

    assert output.advantages.shape == (5,)
    assert torch.allclose(output.advantages[:2], torch.tensor([-1.0, 1.0]))
    assert torch.allclose(output.advantages[2:], torch.tensor([-1.224744871, 0.0, 1.224744871]), atol=1e-7)


def test_length_duplication_changes_sequence_and_token_reduction_weights():
    tokens = torch.tensor([[1, 2, 0], [1, 3, 3]], dtype=torch.long)
    response_mask = torch.tensor([[False, True, False], [False, True, True]])
    rewards = torch.tensor([0.0, 2.0])
    group_ids = torch.tensor([0, 0])
    legal_mask = torch.tensor([False, True, True, True])
    current_logits = torch.tensor(
        [
            [
                [0.0, torch.log(torch.tensor(5 / 12)), torch.log(torch.tensor(1 / 6)), torch.log(torch.tensor(5 / 12))],
                [0.0, torch.log(torch.tensor(5 / 12)), torch.log(torch.tensor(1 / 6)), torch.log(torch.tensor(5 / 12))],
                [0.0, torch.log(torch.tensor(5 / 12)), torch.log(torch.tensor(1 / 6)), torch.log(torch.tensor(5 / 12))],
            ],
            [
                [0.0, torch.log(torch.tensor(1 / 6)), torch.log(torch.tensor(1 / 6)), torch.log(torch.tensor(2 / 3))],
                [0.0, torch.log(torch.tensor(1 / 6)), torch.log(torch.tensor(1 / 6)), torch.log(torch.tensor(2 / 3))],
                [0.0, torch.log(torch.tensor(1 / 6)), torch.log(torch.tensor(1 / 6)), torch.log(torch.tensor(2 / 3))],
            ],
        ]
    )
    old_logits = torch.zeros_like(current_logits)
    current = ConstantLogitsModel(current_logits)
    old = freeze(ConstantLogitsModel(old_logits))
    common = dict(
        tokens=tokens,
        response_mask=response_mask,
        group_ids=group_ids,
        rewards=rewards,
        legal_mask=legal_mask,
        old_model=old,
        epsilon=0.2,
        beta=0.0,
    )

    sequence_output = grpo_objective(model=current, reduction="sequence", **common)
    token_output = grpo_objective(model=current, reduction="token", **common)

    assert torch.equal(sequence_output.sequence_lengths, torch.tensor([1, 2]))
    assert torch.allclose(sequence_output.sequence_policy, torch.tensor([-0.8, 1.2]))
    assert torch.isclose(sequence_output.policy_objective, torch.tensor(0.2))
    assert torch.isclose(token_output.policy_objective, torch.tensor(1.6 / 3.0))
    assert not torch.isclose(
        sequence_output.policy_objective,
        token_output.policy_objective,
    )


@pytest.mark.parametrize(
    ("advantage", "ratio", "expected"),
    [
        (1.0, 0.5, 0.5),
        (1.0, 2.0, 1.2),
        (-1.0, 0.5, -0.8),
        (-1.0, 2.0, -2.0),
    ],
)
def test_clipped_surrogate_covers_both_clip_branches(advantage, ratio, expected):
    value = clipped_surrogate(
        torch.tensor(advantage),
        torch.tensor(ratio),
        epsilon=0.2,
    )
    assert torch.isclose(value, torch.tensor(expected))


def test_explicit_kl_is_computed_over_the_full_legal_vocabulary_without_pad():
    legal_mask = torch.tensor([False, True, True, True])
    current_logits = torch.log(
        torch.tensor([[[0.0, 0.5, 0.25, 0.25]]])
    ).reshape(1, 1, 4)
    reference_logits = torch.zeros(1, 1, 4)

    kl = explicit_kl(current_logits, reference_logits, legal_mask)

    expected = (
        0.5 * torch.log(torch.tensor(1.5))
        + 0.25 * torch.log(torch.tensor(0.75))
        + 0.25 * torch.log(torch.tensor(0.75))
    )
    assert torch.isclose(kl, expected.reshape(()), atol=1e-7)


def test_kl_uses_same_reduction_and_remains_exactly_zero_for_equal_policies():
    tokens = torch.tensor([[1, 2, 0], [1, 3, 3]], dtype=torch.long)
    response_mask = torch.tensor([[False, True, False], [False, True, True]])
    rewards = torch.tensor([0.0, 2.0])
    group_ids = torch.tensor([0, 0])
    legal_mask = torch.tensor([False, True, True, True])
    current_logits = torch.zeros(2, 3, 4)
    current = ConstantLogitsModel(current_logits)
    old = freeze(ConstantLogitsModel(torch.zeros_like(current_logits)))
    reference = freeze(ConstantLogitsModel(torch.zeros_like(current_logits)))
    common = dict(
        tokens=tokens,
        response_mask=response_mask,
        group_ids=group_ids,
        rewards=rewards,
        legal_mask=legal_mask,
        old_model=old,
        reference_model=reference,
        epsilon=0.2,
        beta=0.5,
    )

    sequence_output = grpo_objective(model=current, reduction="sequence", **common)
    token_output = grpo_objective(model=current, reduction="token", **common)

    assert sequence_output.kl_penalty.item() == 0.0
    assert token_output.kl_penalty.item() == 0.0
    assert torch.allclose(sequence_output.ratios[response_mask[:, 1:]], torch.ones(3))
    assert sequence_output.policy_objective.item() == 0.0
    assert torch.isclose(token_output.policy_objective, torch.tensor(1.0 / 3.0))


def make_tiny_setup():
    torch.manual_seed(1729)
    vocab_size = 5
    legal_mask = make_legal_mask(vocab_size, pad_id=0)
    old_model = TinyAutoregressiveTransformer(
        vocab_size=vocab_size,
        pad_id=0,
        d_model=4,
        nhead=2,
        dim_feedforward=8,
    ).double()
    old_model = freeze(old_model)
    reference_model = freeze(copy.deepcopy(old_model))
    current_model = copy.deepcopy(old_model)
    for parameter in current_model.parameters():
        parameter.requires_grad_(True)
    current_model.train()
    with torch.no_grad():
        current_model.head.weight.add_(
            torch.randn_like(current_model.head.weight) * 0.03
        )
        current_model.head.bias.add_(
            torch.randn_like(current_model.head.bias) * 0.03
        )
    return old_model, reference_model, current_model, legal_mask


def uneven_length_batch():
    tokens = torch.tensor(
        [
            [1, 2, 3, 0, 0, 0, 0],
            [1, 2, 3, 3, 3, 3, 3],
            [1, 2, 4, 0, 0, 0, 0],
            [1, 2, 4, 3, 0, 0, 0],
        ],
        dtype=torch.long,
    )
    response_mask = torch.tensor(
        [
            [False, False, True, False, False, False, False],
            [False, False, True, True, True, True, True],
            [False, False, True, False, False, False, False],
            [False, False, True, True, False, False, False],
        ]
    )
    rewards = torch.tensor([0.0, 1.0, 0.0, 1.0])
    group_ids = torch.tensor([0, 0, 1, 1])
    return tokens, response_mask, rewards, group_ids


def test_tiny_update_and_finite_difference_gradient_with_extreme_lengths():
    old_model, reference_model, current_model, legal_mask = make_tiny_setup()
    tokens, response_mask, rewards, group_ids = uneven_length_batch()
    parameter = current_model.head.weight

    def evaluate():
        return grpo_objective(
            model=current_model,
            tokens=tokens,
            response_mask=response_mask,
            group_ids=group_ids,
            rewards=rewards,
            legal_mask=legal_mask,
            old_model=old_model,
            reference_model=reference_model,
            epsilon=0.2,
            beta=0.05,
            reduction="sequence",
        ).loss

    current_model.zero_grad(set_to_none=True)
    loss = evaluate()
    loss.backward()
    analytical_gradient = parameter.grad.detach().clone()

    delta = 1e-6
    row, column = 1, 2
    original_value = parameter.data[row, column].item()
    parameter.data[row, column] = original_value + delta
    positive_loss = evaluate().item()
    parameter.data[row, column] = original_value - delta
    negative_loss = evaluate().item()
    parameter.data[row, column] = original_value
    numerical_gradient = (positive_loss - negative_loss) / (2 * delta)

    assert analytical_gradient.abs().sum() > 0.0
    assert numerical_gradient != 0.0
    assert abs(analytical_gradient[row, column].item() - numerical_gradient) < 2e-5

    before = current_model.head.bias.detach().clone()
    optimizer = torch.optim.SGD(current_model.parameters(), lr=0.1)
    optimizer.step()
    assert not torch.allclose(current_model.head.bias.detach(), before)


def test_beta_zero_needs_no_reference_and_still_updates_policy():
    old_model, _, current_model, legal_mask = make_tiny_setup()
    tokens, response_mask, rewards, group_ids = uneven_length_batch()

    output = grpo_objective(
        model=current_model,
        tokens=tokens,
        response_mask=response_mask,
        group_ids=group_ids,
        rewards=rewards,
        legal_mask=legal_mask,
        old_model=old_model,
        beta=0.0,
        reduction="token",
    )
    output.loss.backward()

    assert output.kl_penalty.item() == 0.0
    assert output.loss.requires_grad
    assert current_model.head.weight.grad.abs().sum() > 0.0


def test_frozen_models_rewards_and_padding_are_excluded_from_optimization():
    old_model, reference_model, current_model, legal_mask = make_tiny_setup()
    tokens, response_mask, rewards, group_ids = uneven_length_batch()

    with pytest.raises(ValueError, match="old_model must be frozen"):
        trainable_old = copy.deepcopy(old_model)
        for parameter in trainable_old.parameters():
            parameter.requires_grad_(True)
        trainable_old.eval()
        grpo_objective(
            model=current_model,
            tokens=tokens,
            response_mask=response_mask,
            group_ids=group_ids,
            rewards=rewards,
            legal_mask=legal_mask,
            old_model=trainable_old,
            reference_model=reference_model,
            beta=0.01,
        )

    output = grpo_objective(
        model=current_model,
        tokens=tokens,
        response_mask=response_mask,
        group_ids=group_ids,
        rewards=rewards,
        legal_mask=legal_mask,
        old_model=old_model,
        reference_model=reference_model,
        beta=0.01,
        reduction="sequence",
    )

    assert not output.advantages.requires_grad
    assert not any(parameter.grad is not None for parameter in old_model.parameters())
    assert not any(parameter.grad is not None for parameter in reference_model.parameters())
    assert output.response_mask.shape == (4, 6)
    assert output.response_mask.sum().item() == 9
    assert output.token_kl[~output.response_mask].abs().max() == 0.0


def test_sampled_legal_probabilities_are_positive_and_right_pad_does_not_affect_prefixes():
    old_model, _, _, legal_mask = make_tiny_setup()
    prompt = torch.tensor([1, 2], dtype=torch.long)
    generator = torch.Generator().manual_seed(44)

    tokens, response_mask = sample_group(
        old_model,
        prompt,
        group_size=3,
        max_new_tokens=4,
        eos_id=4,
        legal_mask=legal_mask,
        generator=generator,
    )

    logits = old_model(tokens, legal_mask)
    probabilities = torch.softmax(logits[:, :-1][:, :, 1:], dim=-1)
    assert torch.isfinite(logits[:, :-1][:, :, 1:]).all()
    assert torch.all(probabilities > 0.0)
    assert response_mask[:, :2].sum() == 0
    assert torch.all(response_mask.sum(dim=1) >= 1)

    short_tokens = tokens[:, :4]
    short_mask = response_mask[:, :4]
    valid_length = short_mask[:, 1:].sum().item()
    assert short_mask[:, 1:].sum().item() == valid_length
    assert torch.equal(
        logits[:, :3].reshape(-1, logits.shape[-1]),
        old_model(short_tokens, legal_mask)[:, :3].reshape(-1, logits.shape[-1]),
    )
