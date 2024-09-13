import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from model.streaming_llama_4k import StreamingLlamaForCausalLM4K
from paths import PROJECT_ROOT

HF_TOKEN = None
model_name = "meta-llama/Meta-Llama-3.1-8B"

NAME = "streaming_4k"
MODEL_CLASS = StreamingLlamaForCausalLM4K


def load_model():
    model = MODEL_CLASS.from_pretrained(model_name, token=HF_TOKEN, device_map="sequential",
                                        attn_implementation="sdpa", torch_dtype=torch.bfloat16)
    model.eval()
    return model


tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN, padding_side="right")
tokenizer.pad_token = tokenizer.eos_token

max_length = 16384
batch_size = 1
total_batches = 555

dataset = load_dataset(os.path.join(PROJECT_ROOT, "pg19", "pg19.py"), split="test", trust_remote_code=True)
samples = dataset["text"]
samples = tokenizer(samples)
input_ids = samples["input_ids"]
total = 0
for ids in input_ids:
    total += len(ids)
print("Total tokens: ", total)


def split_into_chunks(sequences, max_length, stride=None, use_partial=False):
    chunks = []
    if stride is None:
        stride = max_length

    for sequence in sequences:
        for i in range(0, len(sequence), stride):
            chunk = sequence[i:i + max_length]
            if len(chunk) == max_length or (use_partial and len(chunk) > 0):
                chunks.append(chunk)

    return chunks


chunks = split_into_chunks(input_ids, max_length=max_length)
print("Total chunks: ", len(chunks))

input_ids = torch.LongTensor(chunks)
generator = torch.Generator()
generator.manual_seed(42)
dataloader = DataLoader(input_ids, batch_size=batch_size, shuffle=True, generator=generator)
model = load_model()


def calculate_and_print_window_losses(losses, sample_counts, window_size=512, stride=512):
    losses = torch.stack(losses).sum(dim=0)
    sample_counts = torch.stack(sample_counts).sum(dim=0)
    losses = losses / sample_counts

    print("Losses shape: ", losses.shape)

    for i in range(0, len(losses), stride):
        window_average_losses = losses[i:i + window_size]
        if len(window_average_losses) > 0:
            average_loss = window_average_losses.nanmean()
            print(f"Loss for window {i}-{i + window_size}: {average_loss}")
        else:
            print(f"Loss for window {i}-{i + window_size}: No tokens")

    print(f"Total loss: {losses.nanmean()}")


def calculate_and_save_losses(losses, sample_counts, window_size=512, stride=512):
    losses = torch.stack(losses).sum(dim=0)
    sample_counts = torch.stack(sample_counts).sum(dim=0)
    losses = losses / sample_counts
    losses = losses.cpu().numpy()

    np.save(f"losses_{NAME}.npy", losses)


losses = []
sample_counts = []
criterion = nn.CrossEntropyLoss(reduction="none")
for i, batch in tqdm(zip(range(total_batches), dataloader), total=total_batches):
    input_ids = batch.cuda()
    bs = input_ids.shape[0]

    with torch.no_grad():
        outputs = model(input_ids=input_ids, labels=input_ids)
    logits = outputs["logits"]
    v = logits.size(-1)

    reshaped_logits = logits[:, :-1, :].reshape(-1, v)
    reshaped_input_ids = input_ids[:, 1:].reshape(-1)
    loss = criterion(reshaped_logits, reshaped_input_ids).view(bs, -1)

    pad_mask = input_ids[:, 1:] != tokenizer.pad_token_id
    loss = loss * pad_mask
    loss = loss.sum(dim=0)
    sample_count = pad_mask.sum(dim=0)

    # pad until max_length
    loss = F.pad(loss, (0, max_length - loss.shape[0]))
    sample_count = F.pad(sample_count, (0, max_length - sample_count.shape[0]))

    losses.append(loss.cpu())
    sample_counts.append(sample_count.cpu())

    if (i + 1) % 5 == 0:
        calculate_and_print_window_losses(losses, sample_counts)

calculate_and_save_losses(losses, sample_counts)
