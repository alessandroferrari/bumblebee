#!/usr/bin/python3
# -*- coding: utf-8 -*-
from bumblebee.data.tokenizer import load_sample_book
from bumblebee.data.textbook_data.dataloader import create_dataloader
from bumblebee.models.gpt.gpt_model import GPTModel, generate_text_simple
from bumblebee.losses.losses import cross_entropy_loss
import tiktoken
import torch
from bumblebee.core.trainer import Trainer
from bumblebee.losses.viz_utils import plot_losses

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
    encoded = tokenizer.encode(txt)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor


def token_ids_to_text(tokens, tokenizer):
    encoded_flat_list = tokens.squeeze(0).tolist()
    decoded_text = tokenizer.decode(encoded_flat_list)
    return decoded_text


def load_train_eval_text():
    raw_text = load_sample_book()
    train_txt = raw_text[:int(TRAIN_EVAL_SPLIT * len(raw_text))]
    eval_txt = raw_text[int(TRAIN_EVAL_SPLIT * len(raw_text)):]
    return train_txt, eval_txt


def predict_next_token_greedy(model, encoded_tokens):
    model.eval()
    with torch.no_grad():
        predicted_logits = model(encoded_tokens)
    # if encoded tokens do not have the batch size in the shape, add a unary batch dim
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
    # if encoded tokens do not have the batch size in the shape, add a unary batch dim
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


def generate(model, tokenizer, device, start_context, max_new_tokens, temperature=1.0):
    context_size = model.get_context_size()
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    input_tokens = encoded
    next_ids = torch.tensor([[]], device=device, dtype=torch.int32)
    for _ in range(max_new_tokens):
        if input_tokens.shape[1] > context_size:
            input_tokens = input_tokens[:, -context_size:]
        if temperature>0.0:
            next_id = predict_next_token_prob(model, input_tokens, temperature=temperature)
        else:
            next_id = predict_next_token_greedy(model, input_tokens)
        input_tokens = torch.cat((input_tokens, next_id), dim=-1)
        next_ids = torch.cat((next_ids, next_id), dim=-1)
    response = torch.cat((encoded, next_ids), dim=-1)
    decoded_response = token_ids_to_text(response, tokenizer)
    return decoded_response


if __name__ == "__main__":

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
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.0004, weight_decay=0.1)
    NUM_EPOCHS = 10
    EVAL_FREQ = 1
    trainer = Trainer(model, optimizer, train_dataset,
                      eval_dataset, NUM_EPOCHS, device, eval_freq=EVAL_FREQ)
    train_losses, eval_losses = trainer.train()

    plot_losses(train_losses, eval_losses, EVAL_FREQ)

    for _ in range(10):
        response = generate(model, tokenizer, device,
                            start_context="Every effort moves you",
                            max_new_tokens=50,
                            temperature=1.0)
        print(f"Response: {response}")
