#!/usr/bin/python3
# -*- coding: utf-8 -*-

from bumblebee.core.utils import text_to_token_ids, token_ids_to_text
import torch


def predict_next_token_greedy(model, encoded_tokens):
    model.eval()
    with torch.no_grad():
        predicted_logits = model(encoded_tokens)
    # if encoded tokens do not have the batch size in the shape, add a unary
    # batch dim
    if len(predicted_logits.shape) == 2:
        context_size, vocab_size = predicted_logits.shape
        predicted_logits = predicted_logits.view(1, context_size, vocab_size)
    b, context_size, vocab_size = predicted_logits.shape
    # only the last token is the newly produced one
    new_tokens_logits = predicted_logits[:, -1, :]
    new_token_ids = torch.argmax(new_tokens_logits, dim=1, keepdim=True)
    return new_token_ids


def predict_next_token_prob(model, encoded_tokens, temperature=1.0):
    model.eval()
    with torch.no_grad():
        predicted_logits = model(encoded_tokens)
    # if encoded tokens do not have the batch size in the shape, add a unary
    # batch dim
    if len(predicted_logits.shape) == 2:
        context_size, vocab_size = predicted_logits.shape
        predicted_logits = predicted_logits.view(1, context_size, vocab_size)
    b, context_size, vocab_size = predicted_logits.shape
    # only the last token is the newly produced one
    predicted_last = predicted_logits[:, -1, :]  # shape (b, vocab_size)
    TOP_TOKEN_K = 5
    token_logits_topn, token_indeces_topn = torch.topk(
        predicted_last, k=TOP_TOKEN_K, dim=1)
    new_tokens_logits = torch.softmax(token_logits_topn / temperature, dim=1)
    new_token_ids = torch.multinomial(new_tokens_logits, num_samples=1)
    return token_indeces_topn[:, new_token_ids.flatten().tolist()]


def generate(
        model,
        tokenizer,
        device,
        start_context,
        max_new_tokens,
        temperature=1.0):
    context_size = model.get_context_size()
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    input_tokens = encoded
    next_ids = torch.tensor([[]], device=device, dtype=torch.int32)
    for _ in range(max_new_tokens):
        if input_tokens.shape[1] > context_size:
            input_tokens = input_tokens[:, -context_size:]
        if temperature > 0.0:
            next_id = predict_next_token_prob(
                model, input_tokens, temperature=temperature)
        else:
            next_id = predict_next_token_greedy(model, input_tokens)
        input_tokens = torch.cat((input_tokens, next_id), dim=-1)
        next_ids = torch.cat((next_ids, next_id), dim=-1)
    response = torch.cat((encoded, next_ids), dim=-1)
    decoded_response = token_ids_to_text(response, tokenizer)
    return decoded_response
