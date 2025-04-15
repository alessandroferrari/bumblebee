import torch
import torch.nn as nn
import tiktoken


class DummyLayerNorm(nn.Module):
    def __init__(self, num_embeddings):
        super().__init__()

    def forward(self, x):
        return x


class DummyTransformerBlock(nn.Module):
    def __init__(self, num_embeddings, num_heads, dropout_rate):
        super().__init__()

    def forward(self, x):
        return x


class DummyGPTModel(nn.Module):
    def __init__(self, vocab_size, num_embeddings=100, context_length=10, dropout_rate=0.1, num_layers=12, num_heads=4):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, num_embeddings)
        self.position_embedding = nn.Embedding(context_length, num_embeddings)
        self.dropout = nn.Dropout(dropout_rate)
        self.transformer_blocks = nn.Sequential(
            *[DummyTransformerBlock(num_embeddings, num_heads, dropout_rate) for _ in range(num_layers)])
        self.norm = DummyLayerNorm(num_embeddings)
        self.out_head = nn.Linear(num_embeddings, vocab_size, bias=False)

    def forward(self, in_idx):
        batch_size, context_legth = in_idx.shape
        tok_embds = self.token_embedding(in_idx)
        pos_embds = self.position_embedding(
            torch.arange(context_legth, device=in_idx.device))
        x = tok_embds + pos_embds
        x = self.dropout(x)
        x = self.transformer_blocks(x)
        x = self.norm(x)
        return self.out_head(x)


if __name__ == "__main__":
    tokenizer = tiktoken.get_encoding("gpt2")
    txt1 = "This is a test"
    txt2 = "What a test is"
    batch = []
    batch.append(torch.tensor(tokenizer.encode(txt1)))
    batch.append(torch.tensor(tokenizer.encode(txt2)))
    batch = torch.stack(batch, dim=0)
    dummy_gpt = DummyGPTModel(vocab_size=50257, num_embeddings=5,
                              context_length=4, dropout_rate=0.25, num_layers=6, num_heads=4)
    print(dummy_gpt(batch))
