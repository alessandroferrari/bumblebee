# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

import json
import numpy as np
import os
import urllib.request

# import requests
import tensorflow as tf
import tiktoken
import torch
from tqdm import tqdm

# Import from local files
from bumblebee.models.gpt.gpt_model import GPTModel


def download_and_load_gpt2(model_size, models_dir):
    # Validate model size
    allowed_sizes = ("124M", "355M", "774M", "1558M")
    if model_size not in allowed_sizes:
        raise ValueError(f"Model size not in {allowed_sizes}")

    # Define paths
    model_dir = os.path.join(models_dir, model_size)
    base_url = "https://openaipublic.blob.core.windows.net/gpt-2/models"
    filenames = [
        "checkpoint", "encoder.json", "hparams.json",
        "model.ckpt.data-00000-of-00001", "model.ckpt.index",
        "model.ckpt.meta", "vocab.bpe"
    ]

    # Download files
    os.makedirs(model_dir, exist_ok=True)
    for filename in filenames:
        file_url = os.path.join(base_url, model_size, filename)
        file_path = os.path.join(model_dir, filename)
        download_file(file_url, file_path)

    # Load settings and params
    tf_ckpt_path = tf.train.latest_checkpoint(model_dir)
    settings = json.load(open(os.path.join(model_dir, "hparams.json")))
    params = load_gpt2_params_from_tf_ckpt(tf_ckpt_path, settings)

    return settings, params


def download_file(url, destination):
    # Send a GET request to download the file
    with urllib.request.urlopen(url) as response:
        # Get the total file size from headers, defaulting to 0 if not present
        file_size = int(response.headers.get("Content-Length", 0))

        # Check if file exists and has the same size
        if os.path.exists(destination):
            file_size_local = os.path.getsize(destination)
            if file_size == file_size_local:
                print(f"File already exists and is up-to-date: {destination}")
                return

        # Define the block size for reading the file
        block_size = 1024  # 1 Kilobyte

        # Initialize the progress bar with total file size
        progress_bar_description = os.path.basename(
            url)  # Extract filename from URL
        with tqdm(total=file_size, unit="iB", unit_scale=True, desc=progress_bar_description) as progress_bar:
            # Open the destination file in binary write mode
            with open(destination, "wb") as file:
                # Read the file in chunks and write to destination
                while True:
                    chunk = response.read(block_size)
                    if not chunk:
                        break
                    file.write(chunk)
                    progress_bar.update(len(chunk))  # Update progress bar


def load_gpt2_params_from_tf_ckpt(ckpt_path, settings):
    # Initialize parameters dictionary with empty blocks for each layer
    params = {"blocks": [{} for _ in range(settings["n_layer"])]}

    # Iterate over each variable in the checkpoint
    for name, _ in tf.train.list_variables(ckpt_path):
        # Load the variable and remove singleton dimensions
        variable_array = np.squeeze(tf.train.load_variable(ckpt_path, name))

        # Process the variable name to extract relevant parts
        variable_name_parts = name.split("/")[1:]  # Skip the 'model/' prefix

        # Identify the target dictionary for the variable
        target_dict = params
        if variable_name_parts[0].startswith("h"):
            layer_number = int(variable_name_parts[0][1:])
            target_dict = params["blocks"][layer_number]

        # Recursively access or create nested dictionaries
        for key in variable_name_parts[1:-1]:
            target_dict = target_dict.setdefault(key, {})

        # Assign the variable array to the last key
        last_key = variable_name_parts[-1]
        target_dict[last_key] = variable_array

    return params


def assign(left, right):
    if left.shape != right.shape:
        raise ValueError(
            f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")
    return torch.nn.Parameter(torch.tensor(right))


def load_weights_into_gpt(gpt, params):
    gpt.pos_enc_layer.weight = assign(gpt.pos_enc_layer.weight, params["wpe"])
    gpt.token_emb_layer.weight = assign(
        gpt.token_emb_layer.weight, params["wte"])

    for b in range(len(params["blocks"])):
        q_w, k_w, v_w = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1)
        gpt.trf_blocks[b].mh_self_attn.W_query.weight = assign(
            gpt.trf_blocks[b].mh_self_attn.W_query.weight, q_w.T)
        gpt.trf_blocks[b].mh_self_attn.W_key.weight = assign(
            gpt.trf_blocks[b].mh_self_attn.W_key.weight, k_w.T)
        gpt.trf_blocks[b].mh_self_attn.W_value.weight = assign(
            gpt.trf_blocks[b].mh_self_attn.W_value.weight, v_w.T)

        q_b, k_b, v_b = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1)
        gpt.trf_blocks[b].mh_self_attn.W_query.bias = assign(
            gpt.trf_blocks[b].mh_self_attn.W_query.bias, q_b)
        gpt.trf_blocks[b].mh_self_attn.W_key.bias = assign(
            gpt.trf_blocks[b].mh_self_attn.W_key.bias, k_b)
        gpt.trf_blocks[b].mh_self_attn.W_value.bias = assign(
            gpt.trf_blocks[b].mh_self_attn.W_value.bias, v_b)

        gpt.trf_blocks[b].mh_self_attn.out_projection.weight = assign(
            gpt.trf_blocks[b].mh_self_attn.out_projection.weight,
            params["blocks"][b]["attn"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].mh_self_attn.out_projection.bias = assign(
            gpt.trf_blocks[b].mh_self_attn.out_projection.bias,
            params["blocks"][b]["attn"]["c_proj"]["b"])

        gpt.trf_blocks[b].ff.ffnet[0].weight = assign(
            gpt.trf_blocks[b].ff.ffnet[0].weight,
            params["blocks"][b]["mlp"]["c_fc"]["w"].T)
        gpt.trf_blocks[b].ff.ffnet[0].bias = assign(
            gpt.trf_blocks[b].ff.ffnet[0].bias,
            params["blocks"][b]["mlp"]["c_fc"]["b"])
        gpt.trf_blocks[b].ff.ffnet[2].weight = assign(
            gpt.trf_blocks[b].ff.ffnet[2].weight,
            params["blocks"][b]["mlp"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].ff.ffnet[2].bias = assign(
            gpt.trf_blocks[b].ff.ffnet[2].bias,
            params["blocks"][b]["mlp"]["c_proj"]["b"])

        gpt.trf_blocks[b].layer_norm_mha.scale = assign(
            gpt.trf_blocks[b].layer_norm_mha.scale,
            params["blocks"][b]["ln_1"]["g"])
        gpt.trf_blocks[b].layer_norm_mha.shift = assign(
            gpt.trf_blocks[b].layer_norm_mha.shift,
            params["blocks"][b]["ln_1"]["b"])
        gpt.trf_blocks[b].layer_norm_ff.scale = assign(
            gpt.trf_blocks[b].layer_norm_ff.scale,
            params["blocks"][b]["ln_2"]["g"])
        gpt.trf_blocks[b].layer_norm_ff.shift = assign(
            gpt.trf_blocks[b].layer_norm_ff.shift,
            params["blocks"][b]["ln_2"]["b"])

    gpt.final_layer_norm.scale = assign(
        gpt.final_layer_norm.scale, params["g"])
    gpt.final_layer_norm.shift = assign(
        gpt.final_layer_norm.shift, params["b"])
    gpt.output_linear.weight = assign(gpt.output_linear.weight, params["wte"])
