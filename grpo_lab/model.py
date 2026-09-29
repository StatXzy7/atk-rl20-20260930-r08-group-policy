import copy

import torch
from torch import Tensor, nn

from .grpo import categorical_log_probs


class TinyAutoregressiveRNN(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.input_layer = nn.Linear(hidden_size, hidden_size)
        self.hidden_layer = nn.Linear(hidden_size, hidden_size, bias=False)
        self.output_layer = nn.Linear(hidden_size, vocab_size)

    def step(self, token: Tensor, hidden: Tensor) -> Tensor:
        return torch.tanh(self.input_layer(self.embedding(token)) + self.hidden_layer(hidden))

    def forward(self, tokens: Tensor) -> Tensor:
        batch_size, total_length = tokens.shape
        hidden = tokens.new_zeros((batch_size, self.hidden_size), dtype=self.embedding.weight.dtype)
        logits = []
        for position in range(total_length):
            hidden = self.step(tokens[:, position], hidden)
            logits.append(self.output_layer(hidden))
        return torch.stack(logits, dim=1)

    def prompt_hidden(self, prompts: Tensor) -> Tensor:
        hidden = prompts.new_zeros((prompts.shape[0], self.hidden_size), dtype=self.embedding.weight.dtype)
        for position in range(prompts.shape[1]):
            hidden = self.step(prompts[:, position], hidden)
        return hidden


def freeze_model(model: nn.Module) -> nn.Module:
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def frozen_clone(model: nn.Module) -> nn.Module:
    return freeze_model(copy.deepcopy(model))


def sample_completion_batch(
    old_policy: TinyAutoregressiveRNN,
    prompts: Tensor,
    group_sizes: list[int],
    max_new_tokens: int,
    *,
    pad_id: int = 0,
    eos_id: int | None = None,
    generator: torch.Generator | None = None,
    legal_vocab_mask: Tensor | None = None,
) -> dict[str, Tensor]:
    if prompts.dim() != 2:
        raise ValueError("prompts must have shape [num_prompts, prompt_length].")
    if len(group_sizes) != prompts.shape[0] or any(size < 2 for size in group_sizes):
        raise ValueError("Each prompt needs a group size G >= 2.")
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be at least 1.")

    if eos_id is None:
        eos_id = old_policy.vocab_size - 1
    if legal_vocab_mask is None:
        legal_vocab_mask = torch.ones(old_policy.vocab_size, dtype=torch.bool, device=prompts.device)
        legal_vocab_mask[pad_id] = False
    if not legal_vocab_mask[eos_id]:
        raise ValueError("EOS must be a legal sampled action.")

    repeats = torch.as_tensor(group_sizes, dtype=torch.long, device=prompts.device)
    flat_prompts = torch.repeat_interleave(prompts, repeats, dim=0)
    group_ids = torch.arange(prompts.shape[0], device=prompts.device).repeat_interleave(repeats)

    was_training = old_policy.training
    old_policy.eval()
    with torch.no_grad():
        hidden = old_policy.prompt_hidden(flat_prompts)
        active = torch.ones(flat_prompts.shape[0], dtype=torch.bool, device=prompts.device)
        response_tokens = []
        response_mask = []

        for _ in range(max_new_tokens):
            logits = old_policy.output_layer(hidden)
            log_probs = categorical_log_probs(logits, legal_vocab_mask)
            sampled = torch.multinomial(log_probs.exp(), 1, generator=generator).squeeze(-1)
            token = torch.where(active, sampled, torch.full_like(sampled, pad_id))
            response_tokens.append(token)
            response_mask.append(active.clone())
            active = active & sampled.ne(eos_id)
            hidden = old_policy.step(token, hidden)

    if was_training:
        old_policy.train()

    response_tokens = torch.stack(response_tokens, dim=1)
    response_mask = torch.stack(response_mask, dim=1)
    input_ids = torch.concatenate([flat_prompts, response_tokens], dim=1)
    full_response_mask = torch.concatenate(
        [torch.zeros_like(flat_prompts, dtype=torch.bool), response_mask], dim=1
    )
    return {
        "input_ids": input_ids,
        "response_mask": full_response_mask,
        "group_ids": group_ids,
        "prompt_length": torch.full_like(group_ids, prompts.shape[1]),
        "response_length": response_mask.sum(dim=1),
        "response_tokens": response_tokens,
        "response_tokens_mask": response_mask,
    }
