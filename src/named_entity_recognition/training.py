import sys

import torch
import transformers
import torch.nn as nn
from torchcrf import CRF
import pandas as pd
import numpy as np
import joblib
import torch.utils.data
from sklearn import preprocessing
from sklearn import model_selection
from tqdm import tqdm

import config
from feature_extraction import EntityDataset
from evaluate_conll import evaluate

# finding which dataset to use
if len(sys.argv) != 2:
    raise Exception("Specify the dataset to train (ncbi, chemdner, jnlpba)")

# setting parameters for the input dataset
if sys.argv[1] == "ncbi":
    EPOCHS = config.NER_NCBI_DISEASE_EPOCHS
    TRAIN_FILE = config.NER_NCBI_DISEASE_TRAIN_FILE
    TEST_FILE = config.NER_NCBI_DISEASE_TEST_FILE
    DEV_FILE = config.NER_NCBI_DISEASE_DEV_FILE
    CLASS_FILE = config.NER_NCBI_DISEASE_CLASS_FILE
    MODEL_PATH = config.NER_NCBI_DISEASE_MODEL_PATH
    META_PATH = config.NER_NCBI_DISEASE_META_PATH
elif sys.argv[1] == "chemdner":
    EPOCHS = config.NER_CHEMDNER_EPOCHS
    TRAIN_FILE = config.NER_CHEMDNER_TRAIN_FILE
    TEST_FILE = config.NER_CHEMDNER_TEST_FILE
    DEV_FILE = config.NER_CHEMDNER_DEV_FILE
    CLASS_FILE = config.NER_CHEMDNER_CLASS_FILE
    MODEL_PATH = config.NER_CHEMDNER_MODEL_PATH
    META_PATH = config.NER_CHEMDNER_META_PATH
elif sys.argv[1] == "jnlpba":
    EPOCHS = config.NER_JNLPBA_EPOCHS
    TRAIN_FILE = config.NER_JNLPBA_TRAIN_FILE
    TEST_FILE = config.NER_JNLPBA_TEST_FILE
    DEV_FILE = config.NER_JNLPBA_DEV_FILE
    CLASS_FILE = config.NER_JNLPBA_CLASS_FILE
    MODEL_PATH = config.NER_JNLPBA_MODEL_PATH
    META_PATH = config.NER_JNLPBA_META_PATH
else:
    raise Exception(
        "Specify exactly the dataset to train (ncbi, chemdner, jnlpba)")


# NERModel is BERT-BiLSTM-CRF
# BERT layer is CORD-SciBERT embedding output
# BiLSTM layer produced 1024 dimensional vectors
# Dropout layer to provide regularization
# Linear layer to map vectors to output labels
# CRF layer to find the best possible sequence
class NERModel(nn.Module):
    def __init__(self, num_tag):
        super(NERModel, self).__init__()
        self.num_tag = num_tag

        self.bert = transformers.BertModel.from_pretrained(
            config.CORD_SCIBERT_MODEL_PATH, return_dict=False)

        self.bilstm = nn.LSTM(768, 1024 // 2, num_layers=1,
                              bidirectional=True, batch_first=True)
        self.dropout_tag = nn.Dropout(0.3)

        self.hidden2tag_tag = nn.Linear(1024, self.num_tag)

        self.crf_tag = CRF(self.num_tag, batch_first=True)

    # return the loss only, not encode the tag
    def forward(self, ids, mask, token_type_ids, target_tag):
        x, _ = self.bert(ids, attention_mask=mask,
                         token_type_ids=token_type_ids)

        h, _ = self.bilstm(x)

        o_tag = self.dropout_tag(h)

        tag = self.hidden2tag_tag(o_tag)

        mask = torch.where(mask == 1, True, False)

        loss = - self.crf_tag(tag, target_tag,
                              mask=mask, reduction='token_mean')

        return loss

    # encode the tag, dont return loss
    def encode(self, ids, mask, token_type_ids, target_tag):
        # Bert - BiLSTM
        x, _ = self.bert(ids, attention_mask=mask,
                         token_type_ids=token_type_ids)
        h, _ = self.bilstm(x)

        # drop out
        o_tag = self.dropout_tag(h)

        # Hidden2Tag (Linear)
        tag = self.hidden2tag_tag(o_tag)

        # CRF Tag out
        mask = torch.where(mask == 1, True, False)
        tag = self.crf_tag.decode(tag, mask=mask)

        return tag


# label encoding the input dataset tags
total_tags = []
with open(CLASS_FILE) as f:
    for line in f.readlines():
        total_tags.append(line.strip())

enc_tag = preprocessing.LabelEncoder()
enc_tag.fit(list(total_tags))

# read data as sentences and tags


def process_data(data_path):
    sentences, tags = [], []
    sentence, tag = [], []
    i = 0

    for path in data_path:
        with open(path, "r") as f:
            for line in f:
                if i % 10000 == 0:
                    print(len(sentences))
                i += 1
                line = line.strip()
                if line.startswith("-DOCSTART-"):
                    continue
                elif len(line) == 0:
                    if sentence == [] and tag == []:
                        continue
                    sentences.append(sentence)
                    tags.append(tag)
                    sentence, tag = [], []
                else:
                    s, t = line.split("\t")
                    sentence.append(s)
                    tag.append(t)

    for i in range(len(tags)):
        tags[i] = enc_tag.transform(tags[i])

    return sentences, tags, enc_tag


# training function that backpropagates loss based on model output
def train_fn(data_loader, model, optimizer, device):
    model.train()
    final_loss = 0
    for data in tqdm(data_loader, total=len(data_loader)):
        for k, v in data.items():
            data[k] = v.to(device)
        optimizer.zero_grad()
        loss = model(**data)
        loss.backward()
        optimizer.step()
        final_loss += loss.item()
    return final_loss / len(data_loader)


# evaluation function to find the validation loss based on model output
def eval_fn(data_loader, model, device):
    model.eval()
    final_loss = 0

    for data in tqdm(data_loader, total=len(data_loader)):
        for k, v in data.items():
            data[k] = v.to(device)
        loss = model(**data)
        final_loss += loss.item()
    return final_loss / len(data_loader)


# test function to find metrics of the output on testing dataset
def test_fn(dataset, model, device, enc_tag):
    final_test = []
    final_pred = []
    O = enc_tag.transform(["O"])[0]

    with torch.no_grad():
        for data in tqdm(dataset):
            for k, v in data.items():
                data[k] = v.to(device).unsqueeze(0)

            tag = model.encode(**data)
            padded_pred = tag[0]
            test = data["target_tag"].cpu()[0][:len(padded_pred)]
            test = enc_tag.inverse_transform(test)
            padded_pred = enc_tag.inverse_transform(padded_pred)
            final_pred.extend(padded_pred[1:-1])
            final_test.extend(test[1:-1])

    print(evaluate(final_test, final_pred))


# function to load the model
def load_model(num_tag):
    path = MODEL_PATH
    device = torch.device(config.NER_DEVICE)
    model = NERModel(num_tag=num_tag)
    model.load_state_dict(torch.load(path))
    model.to(device)
    return model


# loading testing and training data
sentences, tag, enc_tag = process_data([TRAIN_FILE, DEV_FILE])
test_sentences, test_tag, _ = process_data([TEST_FILE])

# saving metadata for the model
meta_data = {
    "enc_tag": enc_tag
}
joblib.dump(meta_data, META_PATH)

num_tag = len(list(enc_tag.classes_))

# splitting training data into training and validation data
(
    train_sentences,
    valid_sentences,
    train_tag,
    valid_tag
) = model_selection.train_test_split(sentences, tag, random_state=42, test_size=0.1)

# creating dataloader for the dataset
train_dataset = EntityDataset(
    texts=train_sentences, tags=train_tag, enc_tag=enc_tag)
train_data_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=config.NER_BATCH_SIZE, num_workers=config.NER_NUM_WORKERS)

valid_dataset = EntityDataset(
    texts=valid_sentences, tags=valid_tag, enc_tag=enc_tag)
valid_data_loader = torch.utils.data.DataLoader(
    valid_dataset, batch_size=config.NER_BATCH_SIZE, num_workers=config.NER_NUM_WORKERS)

test_dataset = EntityDataset(
    texts=test_sentences, tags=test_tag, enc_tag=enc_tag)

# creating and loading model
device = torch.device(config.NER_DEVICE)
model = NERModel(num_tag=num_tag)
model.to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# running the model for multiple epochs
for epoch in range(EPOCHS):
    train_loss = train_fn(train_data_loader, model, optimizer, device)
    torch.cuda.empty_cache()

    valid_loss = eval_fn(valid_data_loader, model, device)
    torch.cuda.empty_cache()

    print(f"Train Loss = {train_loss}")
    print(f"Validation Loss = {valid_loss}")

    test_fn(test_dataset, model, device, enc_tag)
    torch.save(model.state_dict(), MODEL_PATH)
