#!/usr/bin/python3
# -*- coding: utf-8 -*-

import torch

from bumblebee.core.llm_blocks.normalization_blocks import RMSNormalization, LayerNormalization
from bumblebee.core.llm_blocks.positional_encodings import apply_rope

inputs = torch.tensor(
    [[0.43, 0.15, 0.89],  # Your    (x^1)
     [0.55, 0.87, 0.66],  # journey (x^2)
     [0.57, 0.85, 0.66],  # starts  (x^3)
     [0.22, 0.58, 0.33],  # with    (x^4)
     [0.77, 0.25, 0.11],  # one    (x^5)
     [0.05, 0.80, 0.55]   # step    (x^6)
     ]
)

# Simplified self-attention mechanism, without trainable weights


def simpliflied_self_attention(inputs):
    attn_weights = torch.matmul(inputs, inputs.T)
    attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)
    return torch.matmul(attn_weights, inputs)


D_IN = inputs.shape[1]
D_OUT = 2


def self_attention(inputs, W_query, W_key, W_value):
    # Compute query, key, and value matrices
    query = torch.matmul(inputs, W_query)
    key = torch.matmul(inputs, W_key)
    value = torch.matmul(inputs, W_value)
    # Compute attention weights
    attn_weights = torch.matmul(query, key.T)
    d_k = key.shape[-1]
    attn_weights = torch.nn.functional.softmax(attn_weights / d_k**0.5, dim=-1)
    return torch.matmul(attn_weights, value)


class SelfAttention(torch.nn.Module):
    def __init__(self, d_in, d_out, qkv_bias=False):
        super(SelfAttention, self).__init__()
        self.W_query = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = torch.nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, inputs):
        query = self.W_query(inputs)
        key = self.W_key(inputs)
        value = self.W_value(inputs)

        attn_weights = torch.matmul(query, key.T)
        attn_weights = torch.nn.functional.softmax(
            attn_weights / key.shape[-1]**0.5, dim=-1)
        return torch.matmul(attn_weights, value)


class CausalSelfAttention(torch.nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        super(CausalSelfAttention, self).__init__()
        self.W_query = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = torch.nn.Dropout(dropout)
        self.register_buffer(
            'mask',
            torch.triu(torch.ones((context_length, context_length)),
                       diagonal=1)
        )

    def forward(self, inputs):
        query = self.W_query(inputs)
        key = self.W_key(inputs)
        value = self.W_value(inputs)

        attn_weights = torch.matmul(query, key.transpose(-2, -1))
        attn_weights = attn_weights.masked_fill(self.mask.bool(), -torch.inf)
        attn_weights = torch.nn.functional.softmax(
            attn_weights / key.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)
        return torch.matmul(attn_weights, value)


class MultiHeadSelfAttention(torch.nn.Module):
    def __init__(
            self,
            d_in,
            d_out,
            n_heads,
            context_length,
            dropout,
            qkv_bias=False):
        super(MultiHeadSelfAttention, self).__init__()
        self.num_heads = n_heads
        self.head_dim = d_out // n_heads
        self.d_out = d_out
        self.W_query = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = torch.nn.Dropout(dropout)
        self.register_buffer(
            'mask',
            torch.triu(torch.ones((context_length, context_length)),
                       diagonal=1)
        )
        self.out_projection = torch.nn.Linear(d_out, d_out)

    def forward(self, inputs):
        b, num_tokens, d_in = inputs.shape
        self.query = self.W_query(inputs).view(
            b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        self.key = self.W_key(inputs).view(
            b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        self.value = self.W_value(inputs).view(
            b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)

        attn_weights = torch.matmul(self.query, self.key.transpose(-2, -1))
        attn_weights = attn_weights.masked_fill(
            self.mask[:num_tokens, :num_tokens].bool(), -torch.inf)
        attn_weights = torch.nn.functional.softmax(
            attn_weights / self.head_dim**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec = torch.matmul(attn_weights, self.value).transpose(1, 2)
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)

        context_vec = self.out_projection(context_vec)
        return context_vec


class GroupedQueryAttention(torch.nn.Module):
    def __init__(
            self,
            d_in,
            d_out,
            n_kv_groups,
            n_heads,
            context_length,
            dropout,
            qkv_bias=False,
            qwen_compatible=True):
        super(GroupedQueryAttention, self).__init__()
        self.n_heads = n_heads
        self.head_dim = d_out // n_heads
        self.n_kv_groups = n_kv_groups
        if d_out % n_heads != 0:
            raise Exception(f"In GroupedQueryAttention d_out {d_out} must be divisible by n_heads {n_heads}.")
        if d_out % n_kv_groups != 0:
            raise Exception(f"In GroupedQueryAttention d_out {d_out} must be divisible by n_kv_groups {n_kv_groups}.")
        self.d_out = d_out
        self.W_query = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = torch.nn.Linear(d_in, self.head_dim * n_kv_groups, bias=qkv_bias)
        self.W_value = torch.nn.Linear(d_in, self.head_dim * n_kv_groups, bias=qkv_bias)
        self.dropout = torch.nn.Dropout(dropout)
        self.out_projection = torch.nn.Linear(d_out, d_out)
        self.norm_q = RMSNormalization(self.head_dim, qwen_compatible=qwen_compatible)
        self.norm_k = RMSNormalization(self.head_dim, qwen_compatible=qwen_compatible)


    def forward(self, x, mask, cos, sin):
        b, num_tokens, d_out = x.shape

        w_query = self.W_query(x).view(b, num_tokens, self.n_heads, self.head_dim)
        w_query = self.norm_q(w_query)
        w_query = w_query.transpose(1,2)
        w_query = apply_rope(x, cos, sin, dtype=x.dtype)

        w_key = self.W_key(x).view(b, num_tokens, self.n_kv_groups, self.head_dim)
        w_key = self.norm_k(w_key)
        w_key = w_key.transpose(1,2)

        w_value = self.W_value(x).view(b, num_tokens, self.n_kv_groups, self.head_dim)
        w_value = w_value.transpose(1,2)

        w_key = torch.repeat_interleave(w_key, self.n_heads // self.n_kv_groups, dim=1)
        w_value = torch.repeat_interleave(w_value, self.n_heads // self.n_kv_groups, dim=1)

        attn_scores = torch.matmul(w_query, w_key.transpose(2,3))
        attn_scores = attn_scores.masked_fill(
            mask[:num_tokens, :num_tokens].bool(), -torch.inf)
        attn_scores = torch.nn.functional.softmax(attn_scores  / self.head_dim**0.5, dim=3)
        attn_scores = self.dropout(attn_scores)

        grouped_attn = torch.matmul(attn_scores, w_value).transpose(1,2)

        grouped_attn = grouped_attn.contiguous().view(b, num_tokens, d_out)

        return self.out_projection(grouped_attn)


if __name__ == "__main__":
    outputs = simpliflied_self_attention(inputs)
    print(outputs)

    W_query = torch.nn.Parameter(torch.randn(D_IN, D_OUT), requires_grad=False)
    W_key = torch.nn.Parameter(torch.randn(D_IN, D_OUT), requires_grad=False)
    W_value = torch.nn.Parameter(torch.randn(D_IN, D_OUT), requires_grad=False)

    outputs = self_attention(inputs, W_query, W_key, W_value)
    print(outputs)

    torch.manual_seed(1254)
    self_attention_layer = SelfAttention(D_IN, D_OUT)
    outputs = self_attention_layer(inputs)
    print(outputs)

    W_query = self_attention_layer.W_query.weight.T
    W_key = self_attention_layer.W_key.weight.T
    W_value = self_attention_layer.W_value.weight.T

    outputs = self_attention(inputs, W_query, W_key, W_value)
    print(outputs)

    torch.manual_seed(1254)
    causal_self_attention_layer = CausalSelfAttention(
        D_IN, D_OUT, inputs.shape[0], 0.2)
    outputs = causal_self_attention_layer(inputs)
    print(outputs)

    inputs = inputs.view(1, inputs.shape[0], inputs.shape[1])
    torch.manual_seed(1254)
    D_IN = 3
    D_OUT = 3
    N_HEADS = 3
    CONTEXT_LENGTH = inputs.shape[1]
    mha = MultiHeadSelfAttention(D_IN, D_OUT, N_HEADS, CONTEXT_LENGTH, 0.0)
    outputs = mha(inputs)
    print(outputs)

    torch.manual_seed(1254)
    D_IN = 3
    D_OUT = 3
    N_HEADS = 3
    CONTEXT_LENGTH = inputs.shape[1]
    mha = GroupedQueryAttention(d_in=D_IN, d_out=D_OUT, n_kv_groups=1,
                                n_heads = N_HEADS,
                                context_length = CONTEXT_LENGTH,
                                dropout = 0.0)
    outputs = mha(inputs)
    print(outputs)
