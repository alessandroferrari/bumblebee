#!/usr/bin/python3
# -*- coding: utf-8 -*-
import argparse
from bumblebee.data.tokenizer import load_sample_book
from bumblebee.data.textbook_data.dataloader import create_dataloader
from bumblebee.models.gpt.gpt_model import GPTModel, generate_text_simple
from bumblebee.losses.losses import cross_entropy_loss
import tiktoken
import torch
from bumblebee.core.infer import generate
from bumblebee.core.trainer import Trainer
from bumblebee.losses.viz_utils import plot_losses

def parse_args():
    parser = argparse.ArgumentParser(description="Pretrain GPT2 from scratch from a small sample text.")
    #Training params
    parser.add_argument("--train_eval_split", default=0.9, type=float, help="How to split between training and validation set. Number included between 0.7 and 0.9.")
    parser.add_argument("--device", "-d", type=str, default="auto", choices=["cpu", "cuda", "auto"], help="Force execution on a device.")
    parser.add_argument("--batch_size", "-b", type=int, default=2, help="Batch size to use for training. A larger batch size requires more memory.")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs to run training.")
    parser.add_argument("--eval_freq", type=int, default=1, help="How often to run evaluation.")
    parser.add_argument("--lr", type=float, default=0.0004, help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="Weight decay to use during training.")

    # Text eval sample params
    parser.add_argument("--input_prompt", "-i", default="Every effort moves you", help="Input prompt to provide to the network",
                        type=str)
    parser.add_argument("--temperature", "-t", default=0.0, type=float, help="Temperature for network decoding. 0 temperature is " + \
                        " deterministic greedy decoding. Higher temperatures correspond to more variety on network results generation.")
    parser.add_argument("--n_tokens", "-n", default=50, type=int, help="Number of tokens to generate per generation.")
    parser.add_argument("--n_repetitions", "-r", default=1, type=int, help="How many times repeat inference.")
    parser.add_argument("--export_to_onnx", "-e", action="store_true", help="Enables saving gpt2 to onnx format.")
    parser.add_argument("--onnx_output_path", "-o", help="Output file where to store the onnx file.")
    
    args = parser.parse_args()
    return args

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


def load_train_eval_text(train_eval_split=0.7):
    raw_text = load_sample_book()
    train_txt = raw_text[:int(train_eval_split * len(raw_text))]
    eval_txt = raw_text[int(train_eval_split * len(raw_text)):]
    return train_txt, eval_txt

if __name__ == "__main__":
    
    args = parse_args()
    
    if args.device!="auto":
        device = args.device
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_txt, eval_txt = load_train_eval_text(train_eval_split=args.train_eval_split)

    tokenizer = tiktoken.get_encoding("gpt2")

    train_dataset = create_dataloader(txt=train_txt, tokenizer=tokenizer,
                                      batch_size=args.batch_size, max_length=GPT_CONFIG_124M["context_length"],
                                      stride=GPT_CONFIG_124M["context_length"], shuffle=True, drop_last=True)
    eval_dataset = create_dataloader(txt=eval_txt, tokenizer=tokenizer,
                                     batch_size=args.batch_size, max_length=GPT_CONFIG_124M["context_length"],
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
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    trainer = Trainer(model, optimizer, train_dataset,
                      eval_dataset, args.num_epochs, device, eval_freq=args.eval_freq)
    train_losses, eval_losses = trainer.train()

    plot_losses(train_losses, eval_losses, args.eval_freq)

    model.eval()
    for _ in range(args.n_repetitions):
        response = generate(model, tokenizer, device,
                            start_context=args.input_prompt,
                            max_new_tokens=args.n_tokens,
                            temperature=args.temperature)
        print(f"Response: {response}")

    if args.export_to_onnx:
        export_to_onnx(model, PRETRAINED_GPT_MODEL_CONFIG["context_length"], PRETRAINED_GPT_MODEL_CONFIG["vocab_size"], args.onnx_output_path, device)
