#!/usr/bin/python3
# -*- coding: utf-8 -*-
from tokenizer import load_sample_book
from dataloader import create_dataloader
from gpt_model import GPTModel
from losses import cross_entropy_loss
import tiktoken
import torch
from trainer import Trainer
from viz_utils import plot_losses

TRAIN_EVAL_SPLIT = 0.9

GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 256,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False
}

def text_to_token_ids(txt, tokenizer):
    encoded = tokenizer.encode(txt, allow_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor

def token_ids_to_text(tokens, tokenizer):
    encoded_flat_list = tokens.squeeze(0).tolist()
    decoded_text = tokenizer.decode(encoded_flat_list)

def load_train_eval_text():
    raw_text = load_sample_book()
    train_txt = raw_text[:int(TRAIN_EVAL_SPLIT * len(raw_text))]
    eval_txt = raw_text[int(TRAIN_EVAL_SPLIT * len(raw_text)):]
    return train_txt, eval_txt


if __name__=="__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_txt, eval_txt = load_train_eval_text()

    tokenizer = tiktoken.get_encoding("gpt2")

    train_dataset = create_dataloader(txt=train_txt, tokenizer=tokenizer,
                                      batch_size=2, max_length=GPT_CONFIG_124M["context_length"],
                                      stride=GPT_CONFIG_124M["context_length"], shuffle=True, drop_last=True)
    eval_dataset = create_dataloader(txt=eval_txt, tokenizer=tokenizer,
                                     batch_size=2, max_length=GPT_CONFIG_124M["context_length"],
                                     stride=GPT_CONFIG_124M["context_length"], shuffle=False, drop_last=False)

    print(f"Number of batches training dataloader: {len(train_dataset)}")
    print(f"Number of batches eval dataloader: {len(eval_dataset)}")

    torch.manual_seed(123)
    model = GPTModel(num_embeddings=GPT_CONFIG_124M['emb_dim'],
                     vocab_size=GPT_CONFIG_124M['vocab_size'],
                     num_transformer_blocks=GPT_CONFIG_124M['n_layers'],
                     num_heads=GPT_CONFIG_124M["n_heads"],
                     context_length=GPT_CONFIG_124M["context_length"],
                     dropout_rate=GPT_CONFIG_124M["drop_rate"])
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)
    NUM_EPOCHS = 10
    EVAL_FREQ = 1
    trainer = Trainer(model, optimizer, train_dataset, eval_dataset, NUM_EPOCHS, device, eval_freq=EVAL_FREQ)
    train_losses, eval_losses = trainer.train()

    plot_losses(train_losses, eval_losses, EVAL_FREQ)