import torch


@torch.no_grad()
def sample_group(
    model,
    prompt_ids,
    group_size,
    max_new_tokens,
    eos_id,
    legal_mask=None,
    generator=None,
    pad_id=0,
):
    if group_size < 2:
        raise ValueError("group_size must be at least 2")
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be at least 1")

    prompt_ids = prompt_ids.long()
    if prompt_ids.ndim != 1:
        raise ValueError("prompt_ids must be one-dimensional")
    if torch.any(prompt_ids == pad_id):
        raise ValueError("prompts must not contain PAD")

    tokens = prompt_ids.view(1, -1).repeat(group_size, 1)
    response_mask = torch.zeros_like(tokens, dtype=torch.bool)
    finished = torch.zeros(group_size, dtype=torch.bool, device=tokens.device)

    for _ in range(max_new_tokens):
        logits = model(tokens, legal_mask)
        next_logits = logits[:, -1, :]
        next_token = torch.multinomial(
            torch.softmax(next_logits, dim=-1),
            num_samples=1,
            replacement=True,
            generator=generator,
        ).squeeze(1).to(tokens.device)
        next_token[finished] = pad_id
        response_position = ~finished
        if eos_id is not None and eos_id >= 0:
            finished = finished | (next_token == eos_id)

        tokens = torch.cat([tokens, next_token.unsqueeze(1)], dim=1)
        response_mask = torch.cat(
            [response_mask, response_position.unsqueeze(1)],
            dim=1,
        )

    return tokens, response_mask
