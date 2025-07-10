#!/usr/bin/python3
# -*- coding: utf-8 -*-
import argparse
from bumblebee.data.instruct_sft.dataprep import download_and_load_instruct_sft_data
from bumblebee.data.instruct_sft.dataloader import create_dataloader, format_input
from bumblebee.models.gpt.gpt_model import GPTModel
from bumblebee.losses.losses import cross_entropy_loss
import json
import random
import os
import sys
import tiktoken
import torch
from bumblebee.core.infer import generate
from bumblebee.core.trainer import Trainer
from bumblebee.losses.viz_utils import plot_losses
from bumblebee.models.gpt.gpt_utils import download_and_load_gpt2, load_weights_into_gpt
from bumblebee.core.utils import export_to_onnx


def parse_args():
    parser = argparse.ArgumentParser(
        description="Pretrain GPT2 from scratch from a small sample text.")
    # Training params
    parser.add_argument(
        "--train_eval_split",
        default=0.85,
        type=float,
        help="How to split between training and validation set. Number included between 0.7 and 0.9.")
    parser.add_argument(
        "--device",
        "-d",
        type=str,
        default="auto",
        choices=[
            "cpu",
            "cuda",
            "auto"],
        help="Force execution on a device.")
    parser.add_argument(
        "--batch_size",
        "-b",
        type=int,
        default=2,
        help="Batch size to use for training. A larger batch size requires more memory.")
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=10,
        help="Number of epochs to run training.")
    parser.add_argument("--eval_freq", type=int, default=1,
                        help="How often to run evaluation.")
    parser.add_argument(
        "--lr",
        type=float,
        default=0.00005,
        help="Learning rate.")
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.1,
        help="Weight decay to use during training.")

    # Text eval sample params
    parser.add_argument(
        "--temperature",
        "-t",
        default=0.0,
        type=float,
        help="Temperature for network decoding. 0 temperature is " +
        " deterministic greedy decoding. Higher temperatures correspond to more variety on network results generation.")
    parser.add_argument(
        "--n_tokens",
        "-n",
        default=256,
        type=int,
        help="Number of tokens to generate per generation.")
    parser.add_argument(
        "--model",
        "-m",
        default="gpt2-small",
        type=str,
        choices=[
            "gpt2-small",
            "gpt2-medium",
            "gpt2-large",
            "gpt2-xl"],
        help="Select the gpt2 model to use for running inference.")
    parser.add_argument(
        "--export_to_onnx",
        "-e",
        action="store_true",
        help="Enables saving gpt2 to onnx format.")
    parser.add_argument(
        "--onnx_output_path",
        "-o",
        help="Output file where to store the onnx file.")
    parser.add_argument(
        "--checkpoint_path",
        "-c",
        default=None,
        help="Specify the path where to save the checkpoint after training."
    )
    parser.add_argument(
        "--load_from_checkpoint_path",
        default=None,
        help="Specify the path of a model checkpoint from where resuming training.")
    parser.add_argument(
        "--skip_training",
        action="store_true",
        help="Skip training and jump straightaway to evaluation."
    )
    parser.add_argument(
        "--answers_path",
        default=None,
        help="Specify the path where to save the answers from the model."
    )
    args = parser.parse_args()
    return args


GPT_CONFIG_SHARED = {
    "vocab_size": 50257,
    "context_length": 1024,
    "drop_rate": 0.0,
    "qkv_bias": True
}

gpt_model_configs = {
    "gpt2-small": {"emb_dim": 768, "n_layers": 12, "n_heads": 12, "size": "124M"},
    "gpt2-medium": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16, "size": "355M"},
    "gpt2-large": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20, "size": "774M"},
    "gpt2-xl": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25, "size": "1558M"}
}


def pretraining_batch_loss(logits_batch, target_batch, device):
    logits_batch = logits_batch.to(device)
    target_batch = target_batch.to(device)
    return cross_entropy_loss(logits_batch, target_batch.to(device))


def split_train_test(data, train_eval_split):
    random.shuffle(data)
    MIN_TRAIN_EVAL_SPLIT = 0.6
    MAX_TRAIN_EVAL_SPLIT = 0.9
    if not train_eval_split <= MAX_TRAIN_EVAL_SPLIT and not train_eval_split > MIN_TRAIN_EVAL_SPLIT:
        print(
            f"ERROR! train_eval_split {MIN_TRAIN_EVAL_SPLIT} < {train_eval_split} <= {MAX_TRAIN_EVAL_SPLIT} not verified.")
        sys.exit(2)
    split_idx = int(len(data) * train_eval_split)
    train_data = data[:split_idx]
    test_data = data[split_idx:]

    return train_data, test_data


if __name__ == "__main__":

    args = parse_args()

    settings, params = download_and_load_gpt2(
        gpt_model_configs[args.model]["size"], "models_zoo/"
    )

    if args.device != "auto":
        device = args.device
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = download_and_load_instruct_sft_data()

    train_data, test_data = split_train_test(data, args.train_eval_split)

    tokenizer = tiktoken.get_encoding("gpt2")

    train_dataset = create_dataloader(
        train_data,
        tokenizer,
        context_length=GPT_CONFIG_SHARED["context_length"],
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True)
    test_dataset = create_dataloader(
        test_data,
        tokenizer,
        context_length=GPT_CONFIG_SHARED["context_length"],
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False)

    print(f"Number of batches training dataloader: {len(train_dataset)}")
    print(f"Number of batches eval dataloader: {len(test_dataset)}")

    PRETRAINED_GPT_MODEL_CONFIG = GPT_CONFIG_SHARED.copy()
    PRETRAINED_GPT_MODEL_CONFIG.update(gpt_model_configs[args.model])

    torch.manual_seed(123)
    model = GPTModel(
        num_embeddings=PRETRAINED_GPT_MODEL_CONFIG['emb_dim'],
        vocab_size=PRETRAINED_GPT_MODEL_CONFIG['vocab_size'],
        num_transformer_blocks=PRETRAINED_GPT_MODEL_CONFIG['n_layers'],
        num_heads=PRETRAINED_GPT_MODEL_CONFIG["n_heads"],
        context_length=PRETRAINED_GPT_MODEL_CONFIG["context_length"],
        dropout_rate=PRETRAINED_GPT_MODEL_CONFIG["drop_rate"],
        qkv_bias=PRETRAINED_GPT_MODEL_CONFIG["qkv_bias"])
    if args.load_from_checkpoint_path:
        if not os.path.exists(args.load_from_checkpoint_path):
            print(
                "ERROR! {args.load_from_checkpoint_path} checkpoint path for resuming finetuning does not exists!")
            sys.exit(2)
        model.load_state_dict(
            torch.load(
                args.load_from_checkpoint_path,
                weights_only=True))
    else:
        load_weights_into_gpt(model, params)
    model.to(device)
    if not args.skip_training:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        trainer = Trainer(
            model,
            optimizer,
            train_dataset,
            test_dataset,
            pretraining_batch_loss,
            args.num_epochs,
            device,
            eval_freq=args.eval_freq)
        train_losses, eval_losses = trainer.train()

        plot_losses(train_losses, eval_losses, args.eval_freq)

    if args.export_to_onnx:
        export_to_onnx(
            model,
            PRETRAINED_GPT_MODEL_CONFIG["context_length"],
            PRETRAINED_GPT_MODEL_CONFIG["vocab_size"],
            args.onnx_output_path,
            device)

    if args.checkpoint_path:
        dirname = os.path.dirname(args.checkpoint_path)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        torch.save(model.state_dict(), args.checkpoint_path)

    model.eval()
    results = []
    for entry in test_data:
        RESPONSE_BEGIN = "\n\n### Response:\n"
        input_prompt = "{}{}".format(format_input(entry), RESPONSE_BEGIN)
        response = generate(model, tokenizer, device,
                            start_context=input_prompt,
                            max_new_tokens=args.n_tokens,
                            temperature=args.temperature)
        response_text = response[len(input_prompt):].replace(
            "### Response:", "").strip()
        print(f"Input prompt: {input_prompt}\n\n\n")
        print(f"Response: {response_text}\n\n\n")
        entry["model"] = response_text
        results.append(entry)
    if args.answers_path:
        dirname = os.path.dirname(args.answers_path)
        os.makedirs(dirname, exist_ok=True)
        with open(args.answers_path, "w") as f:
            json.dump(results, f, sort_keys=True, indent=4, ensure_ascii=False)
