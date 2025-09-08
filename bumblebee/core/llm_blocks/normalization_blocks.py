#!/usr/bin/python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn


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


class RMSNormalization(nn.Module):
    def __init__(self, num_embeddings, qwen_compatible=True):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(num_embeddings))
        self.qwen_compatible=True

    def forward(self, x):
        input_dtype = x.dtype

        if self.qwen_compatible:
            x = x.to(torch.float32)

        x2 = x.pow(2).mean(dim=-1, keepdim=True)
        norm_x = x * torch.rsqrt(x2 + self.eps)
        norm_x = self.scale * x
        return  norm_x.to(input_dtype)