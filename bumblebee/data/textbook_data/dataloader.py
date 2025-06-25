from bumblebee.data.tokenizer import load_sample_book, SimpleTokenizerV2, build_vocabulary
import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader


class GPTDatasetV1(Dataset):

    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(txt)

        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i: i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader(txt, tokenizer, batch_size=4, max_length=256, stride=128, shuffle=True,
                      drop_last=True, num_workers=0):

    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )
    return dataloader


if __name__ == "__main__":
    raw_text = load_sample_book()
    BATCH_SIZE = 8
    MAX_LENGTH = 4
    vocab = build_vocabulary(raw_text)
    tokenizer = SimpleTokenizerV2(vocab)
    dataloader = create_dataloader(raw_text, tokenizer, batch_size=BATCH_SIZE,
                                   max_length=MAX_LENGTH, stride=MAX_LENGTH, shuffle=False)
    data_iter = iter(dataloader)
    first_batch = next(data_iter)
    print(first_batch)

    EMBEDDING_SIZE = 256
    token_embedding_layer = torch.nn.Embedding(len(tokenizer), EMBEDDING_SIZE)
    token_embeddings = token_embedding_layer(first_batch[0])
    print("token_embeddings.shape:", token_embeddings.shape)

    pos_embedding_layer = torch.nn.Embedding(MAX_LENGTH, EMBEDDING_SIZE)
    pos_embeddings = pos_embedding_layer(torch.arange(MAX_LENGTH))

    input_embeddings = token_embeddings + pos_embeddings
