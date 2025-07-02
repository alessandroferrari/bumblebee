#!/usr/bin/python3
# -*- coding: utf-8 -*-

import torch


def text_to_token_ids(txt, tokenizer):
    encoded = tokenizer.encode(txt)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor


def token_ids_to_text(tokens, tokenizer):
    encoded_flat_list = tokens.squeeze(0).tolist()
    decoded_text = tokenizer.decode(encoded_flat_list)
    return decoded_text


def export_to_onnx(model, context_length, vocab_size, output_path, device):
    sample_input = torch.randint(
        0, vocab_size, (1, context_length), dtype=torch.int64, device=device)
    onnx_model = torch.onnx.export(
        model, sample_input, output_path, dynamo=False)
