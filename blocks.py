#!/usr/bin/python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from self_attention import MultiHeadSelfAttention


class LayerNormalization(nn.Module):
    def __init__(self, num_embeddings):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(num_embeddings))
        self.shift = nn.Parameter(torch.zeros(num_embeddings))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        variance = x.var(dim=-1, keepdim=True, unbiased=True)
        norm_x = (x - mean) / torch.sqrt(variance + self.eps)
        return self.scale * norm_x + self.shift


class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(torch.sqrt(torch.tensor(2.0 / torch.pi)))) * \
            (x + 0.044715 * torch.pow(x, 3))


class FeedForward(nn.Module):
    def __init__(self, num_embeddings, emb_mul=4.0):
        super().__init__()
        self.ffnet = nn.Sequential(nn.Linear(num_embeddings, emb_mul * num_embeddings),
                                   GELU(),
                                   nn.Linear(num_embeddings * emb_mul, num_embeddings))

    def forward(self, x):
        return self.ffnet(x)


class TransformerBlock(nn.Module):
    def __init__(self, num_embeddings, num_heads, context_length, dropout_rate, ff_embeddings_multiplier=4):
        super().__init__()
        self.layer_norm_mha = LayerNormalization(num_embeddings=num_embeddings)
        self.mh_self_attn = MultiHeadSelfAttention(d_in=num_embeddings, d_out=num_embeddings,
                                                   n_heads=num_heads, context_length=context_length,
                                                   dropout=dropout_rate)
        self.dropout_mha = nn.Dropout(p=dropout_rate)
        self.layer_norm_ff = LayerNormalization(num_embeddings=num_embeddings)
        self.ff = FeedForward(num_embeddings=num_embeddings,
                              emb_mul=ff_embeddings_multiplier)
        self.dropout_ff = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        # Multihead self attention sub block
        x_norm = self.layer_norm_ff(x)
        x_mha = self.mh_self_attn(x_norm)
        x_mha = self.dropout_mha(x_mha)
        x1 = x + x_mha  # skip connection

        # Feed forward sub block
        x1_norm = self.layer_norm_ff(x1)
        x1_ff = self.ff(x1_norm)
        x_out = self.dropout_ff(x1_ff) + x1  # skip connection

        return x_out
