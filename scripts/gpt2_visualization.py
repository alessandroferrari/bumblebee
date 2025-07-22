#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Interactive GPT-2 Small Architecture Visualization
Shows the complete pipeline from input text to output tokens
"""

import torch
import tiktoken
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
import streamlit as st
from bumblebee.models.gpt.gpt_model import GPTModel
from bumblebee.models.gpt.gpt_utils import download_and_load_gpt2, load_weights_into_gpt
from bumblebee.core.infer import generate

# GPT-2 Small Configuration
GPT_CONFIG_SHARED = {
    "vocab_size": 50257,
    "context_length": 1024,
    "drop_rate": 0.0,
    "qkv_bias": True
}

GPT2_SMALL_CONFIG = {
    "emb_dim": 768,
    "n_layers": 12,
    "n_heads": 12,
    "size": "124M"
}

class GPT2Visualizer:
    def __init__(self):
        self.tokenizer = tiktoken.get_encoding("gpt2")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.load_model()

    def load_model(self):
        """Load the GPT-2 small model with pretrained weights"""
        try:
            settings, params = download_and_load_gpt2(GPT2_SMALL_CONFIG["size"], "models_zoo/")

            model_config = GPT_CONFIG_SHARED.copy()
            model_config.update(GPT2_SMALL_CONFIG)

            self.model = GPTModel(
                num_embeddings=model_config['emb_dim'],
                vocab_size=model_config['vocab_size'],
                num_transformer_blocks=model_config['n_layers'],
                num_heads=model_config['n_heads'],
                context_length=model_config['context_length'],
                dropout_rate=model_config['drop_rate'],
                qkv_bias=model_config['qkv_bias']
            )
            load_weights_into_gpt(self.model, params)
            self.model.to(self.device)
            self.model.eval()
        except Exception as e:
            st.error(f"Error loading model: {e}")
            # Create a dummy model for visualization
            self.model = GPTModel(
                num_embeddings=GPT2_SMALL_CONFIG['emb_dim'],
                vocab_size=GPT_CONFIG_SHARED['vocab_size'],
                num_transformer_blocks=GPT2_SMALL_CONFIG['n_layers'],
                num_heads=GPT2_SMALL_CONFIG['n_heads'],
                context_length=GPT_CONFIG_SHARED['context_length'],
                dropout_rate=GPT_CONFIG_SHARED['drop_rate'],
                qkv_bias=GPT_CONFIG_SHARED['qkv_bias']
            )

    def tokenize_text(self, text: str) -> Tuple[List[str], List[int]]:
        """Tokenize text and return tokens and their IDs"""
        tokens = self.tokenizer.encode(text)
        token_texts = [self.tokenizer.decode([t]) for t in tokens]
        return token_texts, tokens

    def get_embeddings(self, token_ids: List[int]) -> torch.Tensor:
        """Get token embeddings"""
        if self.model is None:
            return torch.randn(len(token_ids), GPT2_SMALL_CONFIG['emb_dim'])

        with torch.no_grad():
            token_tensor = torch.tensor(token_ids).unsqueeze(0).to(self.device)
            embeddings = self.model.token_emb_layer(token_tensor)
            return embeddings.squeeze(0)

    def get_positional_encodings(self, seq_length: int) -> torch.Tensor:
        """Get positional encodings"""
        if self.model is None:
            return torch.randn(seq_length, GPT2_SMALL_CONFIG['emb_dim'])

        with torch.no_grad():
            positions = torch.arange(seq_length, device=self.device)
            pos_embeddings = self.model.pos_enc_layer(positions)
            return pos_embeddings

    def visualize_tokenization(self, text: str) -> go.Figure:
        """Create tokenization visualization"""
        token_texts, token_ids = self.tokenize_text(text)

        fig = go.Figure()

        # Create token boxes
        for i, (token_text, token_id) in enumerate(zip(token_texts, token_ids)):
            fig.add_trace(go.Scatter(
                x=[i, i, i+0.8, i+0.8, i],
                y=[0, 1, 1, 0, 0],
                fill="toself",
                fillcolor="lightblue",
                line=dict(color="blue", width=2),
                showlegend=False,
                hoverinfo="text",
                text=f"Token: '{token_text}'<br>ID: {token_id}",
                mode="lines"
            ))

            # Add token text
            fig.add_annotation(
                x=i+0.4,
                y=0.5,
                text=f"'{token_text}'<br>{token_id}",
                showarrow=False,
                font=dict(size=10)
            )

        fig.update_layout(
            title="Step 1: Tokenization",
            xaxis_title="Token Position",
            yaxis_title="",
            xaxis=dict(range=[-0.2, len(token_texts)+0.2]),
            yaxis=dict(range=[-0.2, 1.2]),
            height=300
        )

        return fig

    def visualize_embeddings(self, token_ids: List[int]) -> go.Figure:
        """Create embedding visualization"""
        embeddings = self.get_embeddings(token_ids)

        # Sample a subset of embedding dimensions for visualization
        sample_dims = min(20, embeddings.shape[1])
        sample_embeddings = embeddings[:, :sample_dims].cpu().numpy()

        fig = px.imshow(
            sample_embeddings,
            title="Step 2: Token Embeddings",
            labels=dict(x="Embedding Dimension", y="Token Position"),
            color_continuous_scale="RdBu",
            aspect="auto"
        )

        fig.update_layout(height=400)
        return fig

    def visualize_positional_encoding(self, seq_length: int) -> go.Figure:
        """Create positional encoding visualization"""
        pos_embeddings = self.get_positional_encodings(seq_length)

        # Sample a subset for visualization
        sample_dims = min(20, pos_embeddings.shape[1])
        sample_pos_emb = pos_embeddings[:, :sample_dims].cpu().numpy()

        fig = px.imshow(
            sample_pos_emb,
            title="Step 3: Positional Encodings",
            labels=dict(x="Embedding Dimension", y="Position"),
            color_continuous_scale="Viridis",
            aspect="auto"
        )

        fig.update_layout(height=400)
        return fig

    def visualize_attention_weights(self, token_ids: List[int], layer_idx: int = 0) -> go.Figure:
        """Create attention weights visualization for a specific layer"""
        if self.model is None:
            # Create dummy attention weights
            seq_len = len(token_ids)
            attention_weights = np.random.rand(seq_len, seq_len)
            attention_weights = attention_weights / attention_weights.sum(axis=1, keepdims=True)
        else:
            with torch.no_grad():
                token_tensor = torch.tensor(token_ids).unsqueeze(0).to(self.device)

                # Get embeddings
                token_emb = self.model.token_emb_layer(token_tensor)
                pos_emb = self.model.pos_enc_layer(torch.arange(len(token_ids), device=self.device))
                x = token_emb + pos_emb

                # Get attention weights from the specified layer
                transformer_block = self.model.trf_blocks[layer_idx]
                x_norm = transformer_block.layer_norm_mha(x)

                # Get query, key, value
                mha = transformer_block.mh_self_attn
                query = mha.W_query(x_norm)
                key = mha.W_key(x_norm)
                value = mha.W_value(x_norm)

                # Calculate attention weights
                attn_weights = torch.matmul(query, key.transpose(-2, -1))
                attn_weights = attn_weights.masked_fill(
                    mha.mask[:len(token_ids), :len(token_ids)].bool(), -torch.inf)
                attention_weights = torch.softmax(attn_weights.reshape(attn_weights.shape[1], -1) / key.shape[-1]**0.5, dim=-1)
                attention_weights = attention_weights.cpu().numpy()

        token_texts, _ = self.tokenize_text(" ".join([str(t) for t in token_ids]))

        fig = px.imshow(
            attention_weights,
            title=f"Step 4: Self-Attention Weights (Layer {layer_idx + 1})",
            labels=dict(x="Key Tokens", y="Query Tokens"),
            color_continuous_scale="Blues",
            aspect="auto"
        )

        # Add token labels
        fig.update_xaxes(ticktext=token_texts, tickvals=list(range(len(token_texts))))
        fig.update_yaxes(ticktext=token_texts, tickvals=list(range(len(token_texts))))

        fig.update_layout(height=500)
        return fig

    def visualize_transformer_layers(self, token_ids: List[int]) -> go.Figure:
        """Create visualization of all transformer layers"""
        if self.model is None:
            n_layers = GPT2_SMALL_CONFIG['n_layers']
            layer_outputs = [torch.randn(len(token_ids), GPT2_SMALL_CONFIG['emb_dim']) for _ in range(n_layers)]
        else:
            with torch.no_grad():
                token_tensor = torch.tensor(token_ids).unsqueeze(0).to(self.device)

                # Get initial embeddings
                token_emb = self.model.token_emb_layer(token_tensor)
                pos_emb = self.model.pos_enc_layer(torch.arange(len(token_ids), device=self.device))
                x = token_emb + pos_emb

                layer_outputs = []
                for i, transformer_block in enumerate(self.model.trf_blocks):
                    x = transformer_block(x)
                    layer_outputs.append(x.squeeze(0).cpu())

        # Create subplot for each layer
        n_layers = len(layer_outputs)
        cols = 4
        rows = (n_layers + cols - 1) // cols

        fig = make_subplots(
            rows=rows, cols=cols,
            subplot_titles=[f"Layer {i+1}" for i in range(n_layers)],
            specs=[[{"secondary_y": False} for _ in range(cols)] for _ in range(rows)]
        )

        for i, layer_output in enumerate(layer_outputs):
            row = i // cols + 1
            col = i % cols + 1

            # Sample embedding dimensions for visualization
            sample_dims = min(10, layer_output.shape[1])
            sample_output = layer_output[:, :sample_dims].numpy()

            fig.add_trace(
                go.Heatmap(
                    z=sample_output,
                    colorscale="RdBu",
                    showscale=False
                ),
                row=row, col=col
            )

        fig.update_layout(
            title="Step 5: Transformer Layers Processing",
            height=200 * rows,
            showlegend=False
        )

        return fig

    def visualize_output_generation(self, text: str, max_new_tokens: int = 10) -> go.Figure:
        """Visualize the output generation process"""
        if self.model is None:
            # Create dummy generation
            generated_tokens = [100, 200, 300, 400, 500]
            generated_text = "dummy generated text"
        else:
            try:
                response = generate(self.model, self.tokenizer, self.device,
                                 start_context=text,
                                 max_new_tokens=max_new_tokens,
                                 temperature=0.0)

                # Extract generated tokens
                full_tokens = self.tokenizer.encode(response)
                input_tokens = self.tokenizer.encode(text)
                generated_tokens = full_tokens[len(input_tokens):]
                generated_text = self.tokenizer.decode(generated_tokens)
            except Exception as e:
                st.error(f"Error in generation: {e}")
                generated_tokens = []
                generated_text = "Error in generation"

        fig = go.Figure()

        # Input tokens
        input_token_texts, input_token_ids = self.tokenize_text(text)
        for i, (token_text, token_id) in enumerate(zip(input_token_texts, input_token_ids)):
            fig.add_trace(go.Scatter(
                x=[i, i, i+0.8, i+0.8, i],
                y=[0, 1, 1, 0, 0],
                fill="toself",
                fillcolor="lightgreen",
                line=dict(color="green", width=2),
                showlegend=False,
                hoverinfo="text",
                text=f"Input: '{token_text}'<br>ID: {token_id}",
                mode="lines"
            ))

        # Generated tokens
        for i, token_id in enumerate(generated_tokens):
            token_text = self.tokenizer.decode([token_id])
            fig.add_trace(go.Scatter(
                x=[len(input_token_ids) + i, len(input_token_ids) + i,
                   len(input_token_ids) + i + 0.8, len(input_token_ids) + i + 0.8,
                   len(input_token_ids) + i],
                y=[0, 1, 1, 0, 0],
                fill="toself",
                fillcolor="lightcoral",
                line=dict(color="red", width=2),
                showlegend=False,
                hoverinfo="text",
                text=f"Generated: '{token_text}'<br>ID: {token_id}",
                mode="lines"
            ))

        fig.update_layout(
            title=f"Step 6: Output Generation<br>Generated: '{generated_text}'",
            xaxis_title="Token Position",
            yaxis_title="",
            xaxis=dict(range=[-0.2, len(input_token_ids) + len(generated_tokens) + 0.2]),
            yaxis=dict(range=[-0.2, 1.2]),
            height=300
        )

        return fig

    def create_complete_visualization(self, text: str) -> None:
        """Create the complete interactive visualization"""
        st.title("🤖 GPT-2 Small Architecture Visualization")
        st.markdown("This visualization shows the complete pipeline from input text to generated output.")

        # Input section
        st.header("📝 Input Text")
        st.write(f"**Input:** {text}")

        # Step 1: Tokenization
        st.header("🔤 Step 1: Tokenization")
        st.write("The input text is tokenized using the GPT-2 tokenizer (tiktoken).")
        tokenization_fig = self.visualize_tokenization(text)
        st.plotly_chart(tokenization_fig, use_container_width=True)

        # Get token information
        token_texts, token_ids = self.tokenize_text(text)
        st.write(f"**Tokens:** {token_texts}")
        st.write(f"**Token IDs:** {token_ids}")

        # Step 2: Embeddings
        st.header("🎯 Step 2: Token Embeddings")
        st.write("Each token is converted to a 768-dimensional embedding vector.")
        embedding_fig = self.visualize_embeddings(token_ids)
        st.plotly_chart(embedding_fig, use_container_width=True)

        # Step 3: Positional Encoding
        st.header("📍 Step 3: Positional Encoding")
        st.write("Positional encodings are added to provide sequence order information.")
        pos_encoding_fig = self.visualize_positional_encoding(len(token_ids))
        st.plotly_chart(pos_encoding_fig, use_container_width=True)

        # Step 4: Self-Attention
        st.header("🧠 Step 4: Self-Attention Mechanism")
        st.write("Multi-head self-attention allows tokens to attend to all previous tokens.")

        # Layer selector
        layer_idx = st.slider("Select Attention Layer", 0, GPT2_SMALL_CONFIG['n_layers']-1, 0)
        attention_fig = self.visualize_attention_weights(token_ids, layer_idx)
        st.plotly_chart(attention_fig, use_container_width=True)

        # Step 5: Transformer Layers
        st.header("🏗️ Step 5: Transformer Layers")
        st.write(f"GPT-2 Small has {GPT2_SMALL_CONFIG['n_layers']} transformer layers, each containing:")
        st.markdown("- Multi-head self-attention")
        st.markdown("- Feed-forward network")
        st.markdown("- Layer normalization")
        st.markdown("- Residual connections")

        transformer_fig = self.visualize_transformer_layers(token_ids)
        st.plotly_chart(transformer_fig, use_container_width=True)

        # Step 6: Output Generation
        st.header("🎯 Step 6: Output Generation")
        st.write("The final layer projects embeddings back to vocabulary space for next token prediction.")

        max_tokens = st.slider("Number of tokens to generate", 1, 20, 5)
        output_fig = self.visualize_output_generation(text, max_tokens)
        st.plotly_chart(output_fig, use_container_width=True)

        # Model Architecture Summary
        st.header("📊 Model Architecture Summary")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Embedding Dimension", GPT2_SMALL_CONFIG['emb_dim'])
            st.metric("Number of Layers", GPT2_SMALL_CONFIG['n_layers'])

        with col2:
            st.metric("Number of Heads", GPT2_SMALL_CONFIG['n_heads'])
            st.metric("Vocabulary Size", GPT_CONFIG_SHARED['vocab_size'])

        with col3:
            st.metric("Context Length", GPT_CONFIG_SHARED['context_length'])
            st.metric("Model Size", GPT2_SMALL_CONFIG['size'])

def main():
    st.set_page_config(
        page_title="GPT-2 Small Visualization",
        page_icon="🤖",
        layout="wide"
    )

    # Initialize visualizer
    visualizer = GPT2Visualizer()

    # Sidebar for input
    st.sidebar.header("🎛️ Controls")
    input_text = st.sidebar.text_area(
        "Enter your input text:",
        value="The future of artificial intelligence",
        height=100
    )

    if st.sidebar.button("🚀 Generate Visualization"):
        with st.spinner("Creating visualization..."):
            visualizer.create_complete_visualization(input_text)

    # Default visualization
    if st.sidebar.button("📊 Show Default Example"):
        with st.spinner("Creating visualization..."):
            visualizer.create_complete_visualization("The future of artificial intelligence")

if __name__ == "__main__":
    main()