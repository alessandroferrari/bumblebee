#!/usr/bin/python3
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn


def cross_entropy_loss(predicted_logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    logits_flat = predicted_logits.flatten(0, 1)
    targets_flat = targets.flatten()
    loss = nn.functional.cross_entropy(logits_flat, targets_flat)
    return loss
