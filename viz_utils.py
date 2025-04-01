#!/usr/bin/python3
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt

def plot_losses(train_losses, eval_losses, eval_freq):
    fix, ax1 = plt.subplots(figsize=(5, 3))
    ax1.plot(range(len(train_losses)), train_losses, label="Training Loss")
    ax1.plot(range(len(eval_losses)), eval_losses, linestyle="-.", label="Validation Loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    plt.show()