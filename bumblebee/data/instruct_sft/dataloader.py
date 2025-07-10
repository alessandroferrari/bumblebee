#!/usr/bin/python3
# -*- coding: utf-8 -*-

from functools import partial
import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader


def format_input(entry):
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )

    input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""

    return instruction_text + input_text


class InstructionDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data

        # Pre-tokenize texts
        self.encoded_texts = []
        for entry in data:
            instruction_plus_input = format_input(entry)
            response_text = f"\n\n### Response:\n{entry['output']}"
            full_text = instruction_plus_input + response_text
            self.encoded_texts.append(
                tokenizer.encode(full_text)
            )

    def __getitem__(self, index):
        return self.encoded_texts[index]

    def __len__(self):
        return len(self.data)


def instruct_sft_collate_fn(
    batch,
    pad_token_id=50256,
    loss_ignore_token=-100,
    device="cpu",
    force_max_length=None
):
    # check maximum length within the batch
    max_length = 0
    for entry in batch:
        if len(entry) > max_length:
            max_length = len(entry)

    inputs_list = []
    targets_list = []
    for entry in batch:
        # input sample: [356, 18, 35, ....., 56, 89, 50256, 50256, 50256]
        # target sample: [18, 35, ....., 56, 89, 50256, -100]
        input_entry = entry + [pad_token_id,] * (max_length - len(entry))
        target_entry = entry[1:] + [pad_token_id,] + \
            [loss_ignore_token,] * (max_length - len(entry))

        # force input lenght to be shorter than context
        if force_max_length is not None and force_max_length < len(
                input_entry):
            input_entry = input_entry[:force_max_length]
            target_entry = target_entry[:force_max_length]

        inputs_list.append(torch.tensor(input_entry, dtype=torch.int64))
        targets_list.append(torch.tensor(target_entry, dtype=torch.int64))

    input_batch = torch.stack(inputs_list).to(device)
    target_batch = torch.stack(targets_list).to(device)

    return input_batch, target_batch


def create_dataloader(
        data,
        tokenizer,
        context_length=1024,
        batch_size=8,
        num_workers=0,
        device="cpu",
        shuffle=True,
        drop_last=True):
    torch.manual_seed(123)

    instruct_sft_collate_fn_partial = partial(
        instruct_sft_collate_fn,
        device=device,
        force_max_length=context_length)

    dataset = InstructionDataset(data, tokenizer)
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=instruct_sft_collate_fn_partial,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )

    return data_loader


if __name__ == "__main__":
    entries = [{"instruction": "Evaluate the following phrase by transforming it into the spelling given.",
                "input": "freind --> friend",
                "output": "The spelling of the given phrase \"freind\" is incorrect, the correct spelling is \"friend\"."},
               {"instruction": "Edit the following sentence for grammar.",
                "input": "He go to the park every day.",
                "output": "He goes to the park every day."},
               {"instruction": "Convert 45 kilometers to meters.",
                "input": "",
                "output": "45 kilometers is 45000 meters."}]

    tokenizer = tiktoken.get_encoding("gpt2")

    dataloader = create_dataloader(entries,
                                   tokenizer,
                                   context_length=1024,
                                   batch_size=3,
                                   shuffle=False,
                                   drop_last=False)

    for input, target in dataloader:
        print("INPUT: ", input)
        print("TARGET: ", target)
