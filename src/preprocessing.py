# importing all required packages
import os
import re
from collections import Counter

import tqdm
import pandas as pd
from datasets import load_dataset
import transformers
import torch
import torchvision
from torch.utils.data.dataset import Dataset

import config

# loading the cord-19 abstracts metadata
dataset = load_dataset("cord19", "metadata")

# finding all fields available for each row in dataset
for key in dataset["train"][0]:
    print(key, dataset["train"][0][key])

# preprocessing function that allows only characters and digits


def basicPreprocess(text):
    processed_text = text.lower()
    processed_text = re.sub(r"[^\-a-zA-Z0-9]+", ' ', processed_text)
    return processed_text


# cleaning all cord-19 abstracts
abstracts = []
for i in tqdm.tqdm(range(len(dataset["train"]))):
    abstracts.append(basicPreprocess(
        dataset["train"][i]["abstract"]).replace("\n", " "))

# combining all abstracts into single string
text = ''
for i in tqdm.tqdm(abstracts):
    text += i

# finding occurrence count of all words
counter = Counter(text.split())

# filtering only words with occurrence count greater than 450
vocab = []
for keys, values in counter.items():
    if(values > 450 and values < 10000):
        vocab.append(keys)

print("The length of the vocabulary is ", len(vocab))
print(vocab[:30])

# creating output directory
if not os.path.exists(config.RESULTS_PATH):
    os.mkdir(config.RESULTS_PATH)
if not os.path.exists(config.PRETRAINED_MODEL_PATH):
    os.mkdir(config.PRETRAINED_MODEL_PATH)

# download and save scibert scivocab uncased
tokenizer = transformers.AutoTokenizer.from_pretrained(
    'allenai/scibert_scivocab_uncased')
model = transformers.AutoModelWithLMHead.from_pretrained(
    'allenai/scibert_scivocab_uncased')

scibert_model_path = os.path.join(
    config.PRETRAINED_MODEL_PATH, "scibert_scivocab_uncased")
model.save_pretrained(scibert_model_path)
tokenizer.save_pretrained(scibert_model_path)

# adding new vocabulary to bert model
print("Old vocabulary length : ", len(tokenizer))
tokenizer.add_tokens(vocab)
model.resize_token_embeddings(len(tokenizer))
print("New vocabulary length : ", len(tokenizer))
del vocab
print(model.config)

# creating and saving new model tokenizer
covid_scibert_model_path = os.path.join(
    config.MODELS_PATH, 'COVID-scibert-latest')
if not os.path.exists(config.MODELS_PATH):
    os.mkdir(config.MODELS_PATH)
    os.mkdir(covid_scibert_model_path)
tokenizer.save_pretrained('models/COVID-scibert-latest')

# creating masked language data collator
data_collator = transformers.DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

# training arguments
training_args = transformers.TrainingArguments(
    output_dir=covid_scibert_model_path,
    overwrite_output_dir=True,
    num_train_epochs=config.CORD_FINETUNING_EPOCH,
    per_device_train_batch_size=config.CORD_FINETUNING_BATCH_SIZE,
    save_steps=10_000,
    save_total_limit=3,
    prediction_loss_only=True,
)

# creating the CORD-19 dataset reader which converts text to input_ids


class CORDDataset(Dataset):
    def __init__(self, tokenizer, abstracts, block_size):
        self.abstracts = abstracts
        self.block_size = block_size
        self.start = 0
        self.end = 1
        self.tokenizer = tokenizer
        self.batch_encoding = tokenizer(
            self.abstracts[self.start:self.end], add_special_tokens=True, truncation=False, max_length=block_size)
        self.examples = self.batch_encoding["input_ids"]
        self.examples = [
            {"input_ids": torch.IntTensor(e)} for e in self.examples]
        print(self.examples[0])

    def __len__(self):
        return len(self.abstracts)

    def __getitem__(self, idx):
        if idx < self.start or idx >= self.end:
            encodings = self.tokenizer(
                self.abstracts[idx], add_special_tokens=True, truncation=False, max_length=self.block_size)
            return {"input_ids": torch.tensor(encodings["input_ids"], dtype=torch.int32)}
        return self.examples[idx]


# loading the dataset
dataset = CORDDataset(
    tokenizer=tokenizer,
    abstracts=abstracts,
    block_size=config.CORD_19_BLOCK_SIZE,
)

# finetune and save the model
trainer = transformers.Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)

trainer.train()
trainer.save_model(covid_scibert_model_path)
