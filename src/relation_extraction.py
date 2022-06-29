from collections import defaultdict, namedtuple
import sys
import xml.etree.ElementTree as ET
import random
import os
import random

import torch
import nltk
from transformers import BertTokenizerFast, BertForSequenceClassification
from transformers import Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from nltk.tokenize import sent_tokenize
import pandas

import config

# punkt data necessary for sentence tokenization
nltk.download("punkt")

# row type input dataset row
Row = namedtuple("Dataset", ["id", "text", "label", "entity1", "entity2"])


# function to return dataset rows


def processBC5CDRSplits(inFile):

    # randomly select chemicals and diseases for negative relation generation
    def randomSelectCID(cList, dList):
        c = cList[random.randint(0, len(cList) - 1)]
        d = dList[random.randint(0, len(dList) - 1)]
        return c + "\t" + d

    tree = ET.parse(inFile)
    root = tree.getroot()
    docNo = len(root)-3
    print("docNO = " + str(docNo))
    docNO = 0
    once = True
    dataset = []

    # Finding all the positive and negative relations from XML
    for child in root[3:]:
        docNO += 1
        id = child[0].text
        print(id)
        text = ""
        chemical, desease = {}, {}
        for passage in child.findall("passage"):
            text += str(passage[2].text).lower() + " "
            for annotation in passage.findall("annotation"):
                flag = ""
                s = str(annotation.find("text").text).lower()
                for infon in annotation.findall("infon"):
                    if infon.attrib['key'] == "type":
                        if infon.text == "Chemical":
                            flag = "Chemical"
                        elif infon.text == "Disease":
                            flag = "Disease"
                    if infon.attrib['key'] == "MESH":
                        for k in str(infon.text).split("|"):
                            if flag == "Chemical":
                                chemical[k] = s
                            elif flag == "Disease":
                                desease[k] = s

        symbols = [".", ":", ",", "%", ")", "("]
        for symbol in symbols:
            text = text.replace(symbol, " " + symbol+" ")
        text = text.rstrip()
        if once:
            print(chemical)
            print(desease)
            print("*" * 80)
            once = False

        relationNO = 0
        cidDict = {}
        cidList = []
        for relation in child.findall("relation"):
            relationNO += 1
            c = chemical[relation[1].text]
            d = desease[relation[2].text]
            dataset.append(Row(id+"_"+str(relationNO), text, 1, c, d))
            cidDict[c] = d
            cidList.append(c + "\t" + d)

        negRelationNO = relationNO
        maxTryNO = 100
        tryNO = 0
        for i in range(relationNO):
            negRelationNO += 1
            cid = randomSelectCID(list(chemical.values()),
                                  list(desease.values()))
            while cid in cidList and tryNO < maxTryNO:
                tryNO += 1
                cid = randomSelectCID(
                    list(chemical.values()), list(desease.values()))
            if tryNO < maxTryNO:
                c, d = cid.split("\t")
                dataset.append(Row(id+"_"+str(negRelationNO), text, 0, c, d))
            else:
                print("\n\n\n" + str(id))

    print("docNO = " + str(docNO))
    return dataset


# function to return CHEMPROT dataset rows


def processCHEMPROTSplits(foldername):

    # find the necessary files from input folder
    def findFiles():
        files = list(map(lambda x: os.path.join(
            foldername, x), os.listdir(foldername)))
        files = list(filter(lambda x: x.lower().find("readme") == -1
                            and x.lower().find("gold_standard") == -1, files))
        new_files = [None, None, None]

        for file in files:
            if file.find("abstract") != -1:
                new_files[0] = file
            if file.find("entities") != -1:
                new_files[1] = file
            if file.find("relations") != -1:
                new_files[2] = file

        return new_files

    abstractFile, entitiesFile, relationsFile = findFiles()
    entities = defaultdict(dict)
    entities_count = defaultdict(dict)
    relations = defaultdict(list)
    dataset = []

    # reading all the entities
    entitiesFile = pandas.read_csv(entitiesFile, sep="\t", header=None)
    for i in range(len(entitiesFile)):
        id, entID, entType, start, end, entName = entitiesFile.iloc[i]
        id = str(id)
        entID = str(entID)
        entName = str(entName)
        entType = str(entType)
        start = str(start)
        end = str(end)
        entities[id][entID] = (entID, entName, entType, start, end)
        if entName not in entities_count[id]:
            entities_count[id][entName] = []
        entities_count[id][entName].append(entID)

    # reading all the relations
    with open(relationsFile, "r") as f:
        for line in f.readlines():
            id, relType, _, relName, arg1, arg2 = line.strip().split("\t")
            key1 = arg1.split(":")[1]
            key2 = arg2.split(":")[1]
            entity1 = entities[id][key1]
            entity2 = entities[id][key2]
            relations[id].append((entity1, entity2, relType))

    count = 0
    abstractFile = pandas.read_csv(abstractFile, sep="\t", header=None)

    # reading all the abstracts and creating dataset rows
    for i in range(len(abstractFile)):
        if count % 100 == 0:
            print(count)
        id, title, abstract = abstractFile.iloc[i]
        id = str(id)

        sentences = sent_tokenize(title+"\t"+abstract)
        for rel in relations[id]:
            entity1, entity2, reltype = rel
            if reltype == "CPR:0" or reltype == "CPR:8":
                continue
            ent1IDs = entities_count[id][entity1[1]]
            ent2IDs = entities_count[id][entity2[1]]
            ent1ind = ent1IDs.index(entity1[0])+1
            ent2ind = ent2IDs.index(entity2[0])+1

            for sent in sentences:
                if ent1ind > 0 and sent.find(entity1[1]) != -1:
                    ent1ind -= 1
                if ent2ind > 0 and sent.find(entity2[1]) != -1:
                    ent2ind -= 1
                if (ent1ind, ent2ind) == (0, 0) and sent.find(entity1[1]) != -1 and sent.find(entity2[1]) != -1:
                    count += 1
                    dataset.append(
                        Row(count, sent, reltype, entity1[1], entity2[1]))

    return dataset


def processBC5CDR():
    tr_d = processBC5CDRSplits(config.RE_BC5CDR_TRAIN_FILE)
    te_d = processBC5CDRSplits(config.RE_BC5CDR_TEST_FILE)
    dev_d = processBC5CDRSplits(config.RE_BC5CDR_DEV_FILE)
    return tr_d, dev_d, te_d


def processCHEMPROT():
    tr_d = processCHEMPROTSplits(config.RE_CHEMPROT_TRAIN_FOLDER)
    te_d = processCHEMPROTSplits(config.RE_CHEMPROT_TEST_FOLDER)
    dev_d = processCHEMPROTSplits(config.RE_CHEMPROT_DEV_FOLDER)
    return tr_d, dev_d, te_d


# finding which dataset to use
if len(sys.argv) != 2:
    raise Exception("Specify the dataset to train (bc5cdr, chemprot)")

# setting parameters for the input dataset
if sys.argv[1] == "bc5cdr":
    tr_d, dev_d, te_d = processBC5CDR()
    LABELS = config.RE_BC5CDR_RELATIONS
    EPOCHS = config.RE_BC5CDR_EPOCHS
    MODEL_PATH = config.RE_BC5CDR_MODEL_PATH
elif sys.argv[1] == "chemprot":
    tr_d, dev_d, te_d = processCHEMPROT()
    LABELS = config.RE_CHEMPROT_RELATIONS
    EPOCHS = config.RE_CHEMPROT_EPOCHS
    MODEL_PATH = config.RE_CHEMPROT_MODEL_PATH
else:
    raise Exception("Specify exactly the dataset to train (bc5cdr, chemprot)")


print(tr_d[0].entity1+" "+tr_d[0].entity2+"\t"+tr_d[0].text)

model_name = "allenai/scibert_scivocab_uncased"
max_length = 512

tokenizer = BertTokenizerFast.from_pretrained(
    config.RE_PRETRAINED_MODEL, do_lower_case=True)


# Truncates a sequence pair in place to the maximum length


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


# text to features conversion class for each relation text
class RelationDataset:
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        row = self.dataset[index]
        tokens_a = tokenizer.tokenize(row.entity1+" "+row.entity2+"\t")

        tokens_b = None
        if row[0]:
            tokens_b = tokenizer.tokenize(row.text)

        if tokens_b:
          # Modifies `tokens_a` and `tokens_b` in place so that the total
          # length is less than the specified length.
          # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_length - 3)
        else:
          # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_length - 2:
                tokens_a = tokens_a[0:(max_length - 2)]

        tokens = []
        input_type_ids = []
        tokens.append("[CLS]")
        input_type_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            input_type_ids.append(0)
        tokens.append("[SEP]")
        input_type_ids.append(0)

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                input_type_ids.append(1)
            tokens.append("[SEP]")
            input_type_ids.append(1)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        input_mask = [1] * len(input_ids)
        # Zero-pad up to the sequence length.
        while len(input_ids) < max_length:
            input_ids.append(0)
            input_mask.append(0)
            input_type_ids.append(0)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(input_mask, dtype=torch.long),
            "token_type_ids": torch.tensor(input_type_ids, dtype=torch.long),
            "labels": torch.tensor([row.label], dtype=torch.long)
        }


train_dataset = RelationDataset(tr_d)
test_dataset = RelationDataset(te_d)
dev_dataset = RelationDataset(dev_d)
print(train_dataset[0])

# load the model and pass to Device
model = BertForSequenceClassification.from_pretrained(
    config.RE_PRETRAINED_MODEL, num_labels=len(LABELS)).to(config.RE_DEVICE)


def compute_metrics(pred):
    labels = pred.label_ids
    labels = labels[:, 0]
    preds = pred.predictions.argmax(-1)
    # calculate metrics using sklearn's function
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds)
    pre = precision_score(labels, preds)
    rec = recall_score(labels, preds)

    return {
        'accuracy': acc,
        'precision': pre,
        'recall': rec,
        'f1': f1
    }


training_args = TrainingArguments(
    output_dir=config.RE_RESULTS_PATH,          # output directory
    num_train_epochs=EPOCHS,              # total number of training epochs
    # batch size per device during training
    per_device_train_batch_size=config.RE_TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=config.RE_EVAL_BATCH_SIZE,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    # load the best model when finished training (default metric is loss)
    load_best_model_at_end=True,
    # but you can specify `metric_for_best_model` argument to change to accuracy or other metric
    evaluation_strategy="epoch",     # evaluate each `logging_steps`
    save_strategy="epoch",
    metric_for_best_model="f1"
)

trainer = Trainer(
    model=model,                         # the instantiated Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    eval_dataset=dev_dataset,          # evaluation dataset
    # the callback that computes metrics of interest
    compute_metrics=compute_metrics,
)

print(len(train_dataset))
print(len(test_dataset))
print(len(dev_dataset))

# train the model
trainer.train()

# evaluate the current model after training
trainer.evaluate()

# saving the fine tuned model & tokenizer
model.save_pretrained(MODEL_PATH)
tokenizer.save_pretrained(MODEL_PATH)

y_true = []
y_pred = []
length = 0

for row in test_dataset:
    if length % 1000 == 0:
        print(length)
    length += 1
    row["input_ids"] = row["input_ids"].unsqueeze(0).to("cuda")
    row["attention_mask"] = row["attention_mask"].unsqueeze(0).to("cuda")
    row["token_type_ids"] = row["token_type_ids"].unsqueeze(0).to("cuda")
    y_true.append(row["labels"][0])
    del row["labels"]
    outputs = model(**row)
    probs = outputs[0].cpu().softmax(1)
    y_pred.append(probs.cpu().argmax())


print(classification_report(y_true, y_pred, target_names=LABELS))
