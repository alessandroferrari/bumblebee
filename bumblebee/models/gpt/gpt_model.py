#!/usr/bin/python3
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from bumblebee.core.llm_blocks.blocks import LayerNormalization, TransformerBlock

GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False
}


class GPTModel(nn.Module):
    def __init__(self, vocab_size, num_embeddings, num_transformer_blocks, num_heads, context_length, dropout_rate, ff_embeddings_multiplier=4, qkv_bias=False):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.token_emb_layer = nn.Embedding(vocab_size, num_embeddings)
        self.pos_enc_layer = nn.Embedding(context_length, num_embeddings)
        self.dropout_layer = nn.Dropout(dropout_rate)
        self.trf_blocks = nn.Sequential(*[TransformerBlock(num_embeddings=num_embeddings,
                                                           num_heads=num_heads, context_length=context_length,
                                                           dropout_rate=dropout_rate,
                                                           ff_embeddings_multiplier=ff_embeddings_multiplier,
                                                           qkv_bias=qkv_bias)
                                          for _ in range(num_transformer_blocks)])
        self.final_layer_norm = LayerNormalization(num_embeddings)
        self.output_linear = nn.Linear(num_embeddings, vocab_size, bias=False)

    def forward(self, x):
        batch_size, seq_length = x.shape
        token_emb = self.token_emb_layer(x)
        pos_emb = self.pos_enc_layer(torch.arange(seq_length, device=x.device))
        x = token_emb + pos_emb
        x = self.dropout_layer(x)
        x = self.trf_blocks(x)
        x = self.final_layer_norm(x)
        logits = self.output_linear(x)
        return logits

    def get_context_size(self):
        return self.token_emb_layer.weight.shape[0]


def generate_text_simple(model, idx, max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        # include tokens up to the context size
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]  # last generate token from the model
        probas = torch.softmax(logits, dim=-1)
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=-1)
    return idx


if __name__ == "__main__":
    import tiktoken
    tokenizer = tiktoken.get_encoding("gpt2")
    start_context = "Hello, I am"
    encoded = tokenizer.encode(start_context)
    encoder_tensor = torch.tensor(encoded).unsqueeze(0)

    model = GPTModel(num_embeddings=GPT_CONFIG_124M['emb_dim'],
                     vocab_size=GPT_CONFIG_124M['vocab_size'],
                     num_transformer_blocks=GPT_CONFIG_124M['n_layers'],
                     num_heads=GPT_CONFIG_124M["n_heads"],
                     context_length=GPT_CONFIG_124M["context_length"],
                     dropout_rate=GPT_CONFIG_124M["drop_rate"])
    model.eval()
    out = generate_text_simple(model=model, idx=encoder_tensor,
                               max_new_tokens=6, context_size=GPT_CONFIG_124M["context_length"])
    out_txt = tokenizer.decode(out.squeeze(0).tolist())
    print(out_txt)
