import copy
import unittest

import torch

from grpo_lab import (
    TinyAutoregressiveRNN,
    compute_group_advantages,
    freeze_model,
    frozen_clone,
    grpo_objective,
    sample_completion_batch,
)
from grpo_lab.grpo import SEQUENCE_MEAN, TOKEN_MEAN, categorical_kl, categorical_log_probs


class AdvantageTests(unittest.TestCase):
    def test_group_mean_and_population_std_zero_variance_is_zero(self):
        rewards = torch.tensor([0.0, 2.0, 4.0, 1.0, 4.0, 7.0])
        groups = torch.tensor([0, 0, 0, 1, 1, 1])
        advantages = compute_group_advantages(rewards, groups)
        expected = torch.tensor([-1.2247449, 0.0, 1.2247449, -1.2247449, 0.0, 1.2247449])
        torch.testing.assert_close(advantages, expected, rtol=1e-6, atol=1e-6)

        constant = compute_group_advantages(torch.tensor([3.5, 3.5]), torch.tensor([0, 0]))
        torch.testing.assert_close(constant, torch.zeros_like(constant))
        self.assertFalse(constant.requires_grad)

    def test_group_smaller_than_two_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "G >= 2"):
            compute_group_advantages(torch.tensor([1.0, 2.0]), torch.tensor([0, 1]))

    def test_different_group_sizes_are_supported(self):
        rewards = torch.tensor([0.0, 2.0, 1.0, 4.0, 7.0])
        groups = torch.tensor([0, 0, 1, 1, 1])
        advantages = compute_group_advantages(rewards, groups)
        first = torch.tensor([-1.0, 1.0])
        second = torch.tensor([-1.2247449, 0.0, 1.2247449])
        torch.testing.assert_close(advantages[:2], first)
        torch.testing.assert_close(advantages[2:], second, rtol=1e-6, atol=1e-6)


class ObjectiveTests(unittest.TestCase):
    def make_logits(self, policy_values, other_values=None, vocab_size=4, batches=2, length=3):
        policy = torch.tensor(policy_values, dtype=torch.float64).reshape(batches, length, vocab_size)
        if other_values is None:
            other = torch.zeros(batches, length, vocab_size, dtype=torch.float64)
        else:
            other = torch.tensor(other_values, dtype=torch.float64).reshape(batches, length, vocab_size)
        return policy, other.clone(), other.clone()

    def objective(self, policy_values, advantages, response_mask, *, batches=2, length=3, **kwargs):
        vocab_size = len(policy_values) // (batches * length)
        policy_logits, old_logits, reference_logits = self.make_logits(
            policy_values, vocab_size=vocab_size, batches=batches, length=length
        )
        input_ids = torch.tensor(
            [[0, 1, 2], [0, 2, 1]], dtype=torch.long
        )[:batches, :length]
        mask = torch.tensor(response_mask, dtype=torch.bool)
        advantages = torch.tensor(advantages, dtype=torch.float64)
        return grpo_objective(
            policy_logits,
            old_logits,
            reference_logits,
            input_ids,
            mask,
            advantages,
            beta=0.0,
            **kwargs,
        )

    def test_sequence_and_token_reductions_give_hand_calculated_length_weights(self):
        log_ratio = torch.log(torch.tensor(33.0 / 29.0, dtype=torch.float64)).item()
        policy_values = [
            0.0, log_ratio, 0.0, 0.0,
            0.0, 0.0, log_ratio, 0.0,
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
        ]
        mask = [[False, True, True], [False, True, False]]
        result = self.objective(policy_values, [1.0, 1.0], mask, clip_epsilon=1.0)
        torch.testing.assert_close(result["policy_objective"], torch.tensor(1.05, dtype=torch.float64))

        result_token = self.objective(
            policy_values, [1.0, 1.0], mask, clip_epsilon=0.99, reduction=TOKEN_MEAN
        )
        torch.testing.assert_close(result_token["policy_objective"], torch.tensor(16.0 / 15.0, dtype=torch.float64))

        one_each = self.objective(
            policy_values,
            [1.0, 1.0],
            [[False, True, False], [False, True, False]],
            clip_epsilon=0.99,
            reduction=TOKEN_MEAN,
        )
        torch.testing.assert_close(one_each["policy_objective"], torch.tensor(1.05, dtype=torch.float64))

    def test_duplicating_length_shifts_only_token_mean_weight(self):
        log_ratio = torch.log(torch.tensor(33.0 / 29.0, dtype=torch.float64)).item()
        policy_values = [0.0] * 40
        for row in range(4):
            policy_values[row * 4 + 1] = log_ratio
        input_ids = torch.tensor([
            [0, 1, 1, 1, 1],
            [0, 1, 0, 0, 0],
        ])
        mask = torch.tensor([
            [False, True, True, True, True],
            [False, True, False, False, False],
        ])
        policy_logits, old_logits, reference_logits = self.make_logits(
            policy_values, vocab_size=4, batches=2, length=5
        )
        common = dict(
            clip_epsilon=1.0,
            beta=0.0,
        )
        sequence = grpo_objective(
            policy_logits, old_logits, reference_logits, input_ids, mask,
            torch.ones(2, dtype=torch.float64), reduction=SEQUENCE_MEAN, **common,
        )
        token = grpo_objective(
            policy_logits, old_logits, reference_logits, input_ids, mask,
            torch.ones(2, dtype=torch.float64), reduction=TOKEN_MEAN, **common,
        )
        torch.testing.assert_close(sequence["policy_objective"], torch.tensor(1.05, dtype=torch.float64))
        torch.testing.assert_close(token["policy_objective"], torch.tensor(1.08, dtype=torch.float64))

    def test_prompt_and_padding_tokens_do_not_affect_objective(self):
        ratio_two_logit = torch.log(torch.tensor(3.0, dtype=torch.float64)).item()
        policy_values = [0.0] * 24
        policy_values[1] = ratio_two_logit
        policy_values[14] = ratio_two_logit
        policy_values[8] = 10.0
        policy_values[20] = -10.0
        mask = [[False, True, False], [False, True, False]]
        result = self.objective(policy_values, [1.0, 1.0], mask, clip_epsilon=1.0)
        ratio_e = torch.tensor(2.0, dtype=torch.float64)
        torch.testing.assert_close(result["policy_objective"], ratio_e)
        torch.testing.assert_close(result["kl_penalty"], torch.tensor(0.0, dtype=torch.float64))

    def test_clip_branches_for_both_signs(self):
        cases = [
            ([0, 2, 0, 0, 0, 0, 0, 0], [1.0], 1.2),
            ([0, -2, 0, 0, 0, 0, 0, 0], [-1.0], 0.8),
        ]
        for values, advantage, expected_ratio in cases:
            result = self.objective(
                values, advantage, [[False, True]], batches=1, length=2
            )
            ratio = result["ratio"][0, 0].item()
            self.assertNotAlmostEqual(ratio, expected_ratio)
            torch.testing.assert_close(
                result["clipped_ratio"][0, 0],
                torch.tensor(expected_ratio, dtype=torch.float64),
            )
            torch.testing.assert_close(
                result["policy_objective"],
                torch.tensor(advantage[0] * expected_ratio, dtype=torch.float64),
            )

    def test_zero_advantage_stops_policy_and_kl_gradient(self):
        policy_logits = torch.randn(2, 3, 4, dtype=torch.float64, requires_grad=True)
        old_logits = torch.randn(2, 3, 4, dtype=torch.float64)
        reference_logits = old_logits.clone()
        input_ids = torch.tensor([[0, 1, 2], [0, 2, 1]])
        mask = torch.tensor([[False, True, True], [False, True, False]])
        result = grpo_objective(
            policy_logits,
            old_logits,
            reference_logits,
            input_ids,
            mask,
            torch.zeros(2, dtype=torch.float64),
            beta=0.0,
        )
        result["loss"].backward()
        torch.testing.assert_close(policy_logits.grad, torch.zeros_like(policy_logits))

    def test_identical_policy_has_unit_ratio_no_clipping_and_zero_kl(self):
        old_logits = torch.randn(2, 3, 4, dtype=torch.float64)
        reference_logits = old_logits.clone()
        input_ids = torch.tensor([[0, 1, 2], [0, 2, 1]])
        mask = torch.tensor([[False, True, True], [False, True, False]])
        identical = old_logits.clone().requires_grad_(True)
        result = grpo_objective(
            identical,
            old_logits,
            reference_logits,
            input_ids,
            mask,
            torch.tensor([1.0, -1.0], dtype=torch.float64),
            beta=0.04,
        )
        active = result["prefix_mask"]
        torch.testing.assert_close(result["ratio"][active], torch.ones(3, dtype=torch.float64))
        torch.testing.assert_close(result["clipped_ratio"][active], torch.ones(3, dtype=torch.float64))
        self.assertEqual(result["kl_term"].item(), 0.0)
        self.assertEqual(result["kl_penalty"].item(), 0.0)
        result["loss"].backward()
        self.assertTrue(torch.isfinite(identical.grad).all())

    def test_beta_zero_removes_kl_but_keeps_policy_gradient(self):
        policy_logits = torch.randn(1, 2, 4, dtype=torch.float64, requires_grad=True)
        old_logits = torch.randn(1, 2, 4, dtype=torch.float64)
        reference_logits = torch.randn(1, 2, 4, dtype=torch.float64)
        input_ids = torch.tensor([[0, 1]])
        mask = torch.tensor([[False, True]])
        result = grpo_objective(
            policy_logits,
            old_logits,
            reference_logits,
            input_ids,
            mask,
            torch.ones(1, dtype=torch.float64),
            beta=0.0,
        )
        result["loss"].backward()
        self.assertEqual(result["kl_penalty"].item(), 0.0)
        self.assertGreater(policy_logits.grad.abs().sum().item(), 0.0)

    def test_legal_probabilities_are_positive_and_explicit_kl_is_nonnegative(self):
        legal = torch.tensor([False, True, True, True])
        policy = torch.tensor([[2.0, -3.0, 0.0, 1.0]])
        reference = torch.tensor([[0.0, 3.0, -2.0, 0.0]])
        log_probs = categorical_log_probs(policy, legal)
        probs = log_probs.exp()[0, legal]
        self.assertTrue(torch.isfinite(log_probs[0, legal]).all())
        self.assertTrue((probs > 0).all())
        torch.testing.assert_close(probs.sum(), torch.tensor(1.0, dtype=torch.float32))
        self.assertGreaterEqual(categorical_kl(policy, reference, legal).item(), 0.0)

        with self.assertRaisesRegex(ValueError, "legal action"):
            categorical_log_probs(policy, torch.tensor([False, False, False, False]))

    def test_double_precision_gradcheck(self):
        torch.manual_seed(19)
        policy_logits = (torch.randn(2, 3, 4, dtype=torch.float64) * 0.01).requires_grad_(True)
        old_logits = torch.randn(2, 3, 4, dtype=torch.float64) * 0.01
        reference_logits = torch.randn(2, 3, 4, dtype=torch.float64) * 0.01
        input_ids = torch.tensor([[0, 1, 2], [0, 3, 1]])
        mask = torch.tensor([[False, True, True], [False, True, False]])
        legal = torch.tensor([False, True, True, True])

        def scalar_loss(logits, reduction):
            return grpo_objective(
                logits,
                old_logits,
                reference_logits,
                input_ids,
                mask,
                torch.tensor([0.7, -0.4], dtype=torch.float64),
                clip_epsilon=0.2,
                beta=0.03,
                reduction=reduction,
                legal_vocab_mask=legal,
            )["loss"]

        self.assertTrue(torch.autograd.gradcheck(lambda x: scalar_loss(x, SEQUENCE_MEAN), (policy_logits,)))
        self.assertTrue(torch.autograd.gradcheck(lambda x: scalar_loss(x, TOKEN_MEAN), (policy_logits,)))


class ModelTests(unittest.TestCase):
    def test_tiny_model_update_and_frozen_policies(self):
        torch.manual_seed(5)
        vocab_size, hidden_size = 5, 4
        prompts = torch.tensor([[1, 2], [2, 3]])
        old_policy = freeze_model(TinyAutoregressiveRNN(vocab_size, hidden_size))
        reference_policy = frozen_clone(old_policy)
        legal = torch.tensor([False, True, True, True, True])
        generator = torch.Generator().manual_seed(4)
        batch = sample_completion_batch(
            old_policy,
            prompts,
            [2, 2],
            max_new_tokens=4,
            generator=generator,
            legal_vocab_mask=legal,
        )
        self.assertEqual(batch["input_ids"].shape[0], 4)
        self.assertTrue((batch["response_length"] >= 1).all())
        self.assertFalse(batch["response_mask"][:, :2].any())

        with torch.no_grad():
            old_before = old_policy.embedding.weight.clone()
            reference_before = reference_policy.embedding.weight.clone()
            old_logits = old_policy(batch["input_ids"])
            reference_logits = reference_policy(batch["input_ids"])

        policy = TinyAutoregressiveRNN(vocab_size, hidden_size)
        policy.load_state_dict(copy.deepcopy(old_policy.state_dict()))
        rewards = torch.tensor([0.0, 1.0, 2.0, 3.0])
        advantages = compute_group_advantages(rewards, batch["group_ids"])
        result = grpo_objective(
            policy(batch["input_ids"]),
            old_logits,
            reference_logits,
            batch["input_ids"],
            batch["response_mask"],
            advantages,
            beta=0.02,
            legal_vocab_mask=legal,
        )
        result["loss"].backward()
        gradients = [parameter.grad for parameter in policy.parameters()]
        self.assertTrue(all(gradient is not None and torch.isfinite(gradient).all() for gradient in gradients))
        self.assertGreater(sum(gradient.abs().sum().item() for gradient in gradients), 0.0)
        self.assertTrue(torch.equal(old_policy.embedding.weight, old_before))
        self.assertTrue(torch.equal(reference_policy.embedding.weight, reference_before))

    def test_end_to_end_zero_reward_variance_beta_zero_has_no_policy_gradient(self):
        torch.manual_seed(9)
        prompts = torch.tensor([[1, 2], [3, 1]])
        old_policy = freeze_model(TinyAutoregressiveRNN(5, 4))
        reference_policy = frozen_clone(old_policy)
        legal = torch.tensor([False, True, True, True, True])
        batch = sample_completion_batch(
            old_policy,
            prompts,
            [2, 2],
            max_new_tokens=3,
            generator=torch.Generator().manual_seed(11),
            legal_vocab_mask=legal,
        )
        with torch.no_grad():
            old_logits = old_policy(batch["input_ids"])
            reference_logits = reference_policy(batch["input_ids"])
        policy = TinyAutoregressiveRNN(5, 4)
        advantages = compute_group_advantages(torch.ones(4), batch["group_ids"])
        result = grpo_objective(
            policy(batch["input_ids"]),
            old_logits,
            reference_logits,
            batch["input_ids"],
            batch["response_mask"],
            advantages,
            beta=0.0,
            legal_vocab_mask=legal,
        )
        result["loss"].backward()
        for parameter in policy.parameters():
            self.assertIsNotNone(parameter.grad)
            torch.testing.assert_close(parameter.grad, torch.zeros_like(parameter.grad))


if __name__ == "__main__":
    unittest.main()
