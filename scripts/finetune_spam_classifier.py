#!/usr/bin/python3
# -*- coding: utf-8 -*-
import argparse
from bumblebee.data.spam_classification.dataloader import create_dataloader
from bumblebee.models.gpt.gpt_model import GPTModel, generate_text_simple
from bumblebee.losses.losses import cross_entropy_loss
import tiktoken
import torch
from bumblebee.core.infer import generate
from bumblebee.core.trainer import Trainer
from bumblebee.models.gpt.gpt_utils import download_and_load_gpt2, load_weights_into_gpt
from bumblebee.losses.viz_utils import plot_losses
from bumblebee.core.utils import export_to_onnx
from pathlib import Path
import os
import sys


def parse_args():
    parser = argparse.ArgumentParser(
        description="Finetune GPT2 to classify SMS spam.")
    # Training params
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
        "--export_to_onnx",
        "-e",
        action="store_true",
        help="Enables saving gpt2 to onnx format.")
    parser.add_argument(
        "--onnx_output_path",
        "-o",
        help="Output file where to store the onnx file.")

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

DATASET_BASEDIR = Path(__file__).parent.parent / \
    "datasets" / "spam_classification"
TRAIN_DATA_PATH = os.path.join(DATASET_BASEDIR, "train.csv")
VAL_DATA_PATH = os.path.join(DATASET_BASEDIR, "validation.csv")
TEST_DATA_PATH = os.path.join(DATASET_BASEDIR, "test.csv")


def spam_classification_batch_loss(activation_batch, target_batch, device):
    target_batch = target_batch.to(device)
    activation_batch = activation_batch.to(device)
    # only keep the logits of the last token output by gpt2
    loss = torch.nn.functional.cross_entropy(
        activation_batch[:, -1, :], target_batch)
    return loss


def classification_accuracy(data_loader, model):
    correct_samples = 0.0
    samples = 0.0
    for input_batch, target_batch in data_loader:
        input_batch = input_batch.to(device)
        target_batch = target_batch.to(device)
        with torch.no_grad():
            logits = model(input_batch)[:, -1, :]
        predicted_logits = torch.argmax(logits, dim=-1)
        samples += logits.shape[0]
        correct_samples += (predicted_logits == target_batch).sum().item()
    accuracy = correct_samples / samples
    return accuracy


if __name__ == "__main__":

    args = parse_args()

    settings, params = download_and_load_gpt2(
        gpt_model_configs[args.model]["size"], "models_zoo/"
    )

    if args.device != "auto":
        device = args.device
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = tiktoken.get_encoding("gpt2")

    train_dataloader = create_dataloader(csv_file=TRAIN_DATA_PATH,
                                         tokenizer=tokenizer,
                                         batch_size=args.batch_size,
                                         shuffle=True,
                                         drop_last=True)
    eval_dataloader = create_dataloader(csv_file=VAL_DATA_PATH,
                                        tokenizer=tokenizer,
                                        batch_size=args.batch_size,
                                        shuffle=False,
                                        drop_last=False)
    test_dataloader = create_dataloader(csv_file=TEST_DATA_PATH,
                                        tokenizer=tokenizer,
                                        batch_size=args.batch_size,
                                        shuffle=False,
                                        drop_last=False)

    print(f"Number of batches training dataloader: {len(train_dataloader)}")
    print(f"Number of batches eval dataloader: {len(test_dataloader)}")

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
    # Freezing the model layers of the pretrained transformer for finetuning
    for param in model.parameters():
        param.require_grad = False
    for param in model.trf_blocks[-1].parameters():
        param.requires_grad = True

    num_embeddings = model.output_linear.weight.shape[1]
    NUM_CLASSES = 2
    model.output_linear = torch.nn.Linear(
        num_embeddings, NUM_CLASSES, device=device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    trainer = Trainer(
        model,
        optimizer,
        train_dataloader,
        eval_dataloader,
        spam_classification_batch_loss,
        args.num_epochs,
        device,
        eval_freq=args.eval_freq)
    train_losses, eval_losses = trainer.train()

    accuracy = classification_accuracy(test_dataloader, model)
    print(f"Testing accuracy: {accuracy:.4f}")

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
