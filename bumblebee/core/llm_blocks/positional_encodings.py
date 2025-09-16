#!/usr/bin/python3
# -*- coding: utf-8 -*-

import torch

def compute_rope(head_dim, theta_base, context_length, dtype):

    inv_freq = 1.0 / (theta_base ** (torch.arange(start=0, end=head_dim, step=2).float() / head_dim))
    inv_freq = inv_freq.to(dtype)

    positions = torch.arange(context_length).to(dtype)

    angles = positions.view(context_length, 1) * inv_freq.T # context_length * head_dim / 2
    angles = angles.view(context_length, head_dim // 2, 1)
    angles = angles.repeat(1,1,2).view(context_length, head_dim)

    cos = torch.cos(angles)
    sin = torch.sin(angles)

    return cos, sin


def apply_rope(x, cos, sin):
    batch_size, num_heads, seq_length, head_dim = x.shape

    # sample x0, x2, x4, ...
    x1 = x[:,:,:,::2] # batch_size, num_heads, seq_length, head_dim // 2
    # sample x1, x3, x5, ...
    x2 = x[:,:,:,1::2] # batch_size, num_heads, seq_length, head_dim // 2
    cos = cos[:seq_length, :].view(1, 1, seq_length, head_dim)
    sin = sin[:seq_length, :].view(1, 1, seq_length, head_dim)

    # -x2, x1, -x4, x3, ...
    rotated = torch.cat((-x2, x1), dim=-1)
    x_rotated = x * cos + rotated * sin

    return x_rotated.to(dtype=x.dtype)