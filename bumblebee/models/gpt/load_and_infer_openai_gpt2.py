#!/usr/bin/python3
# -*- coding: utf-8 -*-
import argparse
from bumblebee.data.tokenizer import load_sample_book
from bumblebee.data.textbook_data.dataloader import create_dataloader
from bumblebee.models.gpt.gpt_model import GPTModel
import tiktoken
import torch
from bumblebee.core.infer import generate
from bumblebee.core.utils import export_to_onnx
from bumblebee.models.gpt.gpt_utils import download_and_load_gpt2, load_weights_into_gpt

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

def parse_args():
    parser = argparse.ArgumentParser(description="Load pretrained GPT2 weights from OpenAI and run inference on some samples.")
    parser.add_argument("--input_prompt", "-i", default="Every effort moves you", help="Input prompt to provide to the network",
                        type=str)
    parser.add_argument("--temperature", "-t", default=0.0, type=float, help="Temperature for network decoding. 0 temperature is " + \
                        " deterministic greedy decoding. Higher temperatures correspond to more variety on network results generation.")
    parser.add_argument("--n_tokens", "-n", default=50, type=int, help="Number of tokens to generate per generation.")
    parser.add_argument("--n_repetitions", "-r", default=1, type=int, help="How many times repeat inference.")
    parser.add_argument("--model", "-m", default="gpt2-small", type=str, choices=["gpt2-small", "gpt2-medium", "gpt2-large", "gpt2-xl"],
                        help="Select the gpt2 model to use for running inference.")
    parser.add_argument("--export_to_onnx", "-e", action="store_true", help="Enables saving gpt2 to onnx format.")
    parser.add_argument("--onnx_output_path", "-o", help="Output file where to store the onnx file.")
    parser.add_argument("--device", "-d", type=str, default="auto", choices=["cpu", "cuda", "auto"], help="Force execution on a device.")
    args = parser.parse_args()
    return args

if __name__ == "__main__":

    args = parse_args()

    settings, params = download_and_load_gpt2(gpt_model_configs[args.model]["size"], "models_zoo/")

    if args.device!="auto":
        device = args.device
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = tiktoken.get_encoding("gpt2")

    PRETRAINED_GPT_MODEL_CONFIG = GPT_CONFIG_SHARED.copy()
    PRETRAINED_GPT_MODEL_CONFIG.update(gpt_model_configs[args.model])

    torch.manual_seed(123)
    model = GPTModel(num_embeddings=PRETRAINED_GPT_MODEL_CONFIG['emb_dim'],
                     vocab_size=PRETRAINED_GPT_MODEL_CONFIG['vocab_size'],
                     num_transformer_blocks=PRETRAINED_GPT_MODEL_CONFIG['n_layers'],
                     num_heads=PRETRAINED_GPT_MODEL_CONFIG["n_heads"],
                     context_length=PRETRAINED_GPT_MODEL_CONFIG["context_length"],
                     dropout_rate=PRETRAINED_GPT_MODEL_CONFIG["drop_rate"],
                     qkv_bias=PRETRAINED_GPT_MODEL_CONFIG["qkv_bias"])
    load_weights_into_gpt(model, params)

    model.to(device)
    
    model.eval()
    for _ in range(args.n_repetitions):
        response = generate(model, tokenizer, device,
                            start_context=args.input_prompt,
                            max_new_tokens=args.n_tokens,
                            temperature=args.temperature)
        print(f"Response: {response}")

    if args.export_to_onnx:
        export_to_onnx(model, PRETRAINED_GPT_MODEL_CONFIG["context_length"], PRETRAINED_GPT_MODEL_CONFIG["vocab_size"], args.onnx_output_path, device)
   