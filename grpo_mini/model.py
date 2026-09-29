import torch
from torch import nn


def make_legal_mask(vocab_size, pad_id=None, device=None, dtype=torch.bool):
    mask = torch.ones(vocab_size, device=device, dtype=dtype)
    if pad_id is not None and pad_id >= 0:
        mask[pad_id] = False
    return mask


def legal_logits(logits, legal_mask):
    if legal_mask is None:
        return logits
    mask = legal_mask.to(device=logits.device)
    negative_infinity = torch.finfo(logits.dtype).min
    return logits.masked_fill(~mask.view(*([1] * (logits.ndim - 1)), -1), negative_infinity)


def freeze(model):
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    return model


class TinyAutoregressiveTransformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        pad_id=0,
        d_model=8,
        nhead=2,
        dim_feedforward=16,
    ):
        super().__init__()
        if vocab_size < 3:
            raise ValueError("vocab_size must include PAD and at least two legal tokens")
        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead")
        self.vocab_size = vocab_size
        self.pad_id = pad_id
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.position_embedding = nn.Embedding(64, d_model)
        self.layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=0.0,
            batch_first=True,
            activation="relu",
            norm_first=False,
        )
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, token_ids, legal_mask=None):
        batch_size, sequence_length = token_ids.shape
        if sequence_length > self.position_embedding.num_embeddings:
            raise ValueError("this tiny model supports at most 64 positions")

        positions = torch.arange(sequence_length, device=token_ids.device)
        hidden = self.token_embedding(token_ids) + self.position_embedding(positions)

        causal_mask = torch.triu(
            torch.ones(sequence_length, sequence_length, dtype=torch.bool, device=token_ids.device),
            diagonal=1,
        )
        hidden = self.layer(
            hidden,
            src_mask=causal_mask,
            is_causal=True,
        )
        logits = self.head(hidden)

        if legal_mask is None:
            legal_mask = make_legal_mask(
                self.vocab_size,
                self.pad_id,
                device=token_ids.device,
            )
        return legal_logits(logits, legal_mask)
