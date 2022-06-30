import json
import os

from pymongo import MongoClient, UpdateOne
from transformers import AutoModel, AutoTokenizer, BertTokenizerFast, BertForSequenceClassification
import joblib
import torch
import transformers
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
import pandas as pd
from tqdm import tqdm

from named_entity_recognition.training import NERModel
from feature_extraction import EntityDataset
from relation_extraction import _truncate_seq_pair
import config

# punkt data needed for sent and word tokenize
nltk.download("punkt")

# get database from mongodb
client = MongoClient(
    config.GC_MONGO_URL
)
db = client.fyp
print(client.list_database_names())

# insert the found diseases, chemicals and proteins into entities collection


def insertIntoEntitiesCollection(disease, chemical, proteins):
    entities = disease+chemical+proteins
    entities_collection = db.get_collection("entities")

    def mapEntityToDBOperation(entity):
        return UpdateOne({"word": entity["word"]}, {"$set": entity}, upsert=True)

    operations = list(map(mapEntityToDBOperation, entities))
    if len(operations) > 0:
        with client.start_session() as session:
            with session.start_transaction():
                entities_collection.bulk_write(operations, session=session)

    return entities_collection.count_documents({})

# inserting the found relations into the relations collection


def insertIntoRelationsCollection(cidRelations, cprRelations):
    relations = cidRelations+cprRelations
    relations_collection = db.get_collection("relations")

    def mapRelationsToDBOperation(relation):
        return UpdateOne({
            "entity_1": relation["entity_1"],
            "entity_2": relation["entity_2"],
            "relation": relation["relation"]
        }, {"$set": relation}, upsert=True)

    operations = list(map(mapRelationsToDBOperation, relations))
    if len(operations) > 0:
        with client.start_session() as session:
            with session.start_transaction():
                relations_collection.bulk_write(operations, session=session)

    return relations_collection.count_documents({})


bert = AutoModel.from_pretrained(config.CORD_SCIBERT_MODEL_PATH)
TOKENIZER = transformers.BertTokenizer.from_pretrained(
    config.CORD_SCIBERT_MODEL_PATH, do_lower_case=True
)

# load the model based on their metapath and modelpath


def load_model(metaPath, modelPath):
    meta_data = joblib.load(metaPath)
    enc_tag = meta_data["enc_tag"]
    num_tag = len(list(enc_tag.classes_))

    device = torch.device(config.NER_DEVICE)
    model = NERModel(num_tag=num_tag)
    model.load_state_dict(torch.load(
        modelPath, map_location=torch.device(config.NER_DEVICE)))
    model.to(device)

    return model, enc_tag


# CORD19NERDataset that adds additional details to the original
# EntityDataset. It has words which has subword token to word mapping
# and get_entities returns all the found entities in a single text.
class CORD19NERDataset(EntityDataset):
    def __init__(self, texts, tags, enc_tag):
        super().__init__(texts, tags, enc_tag)
        self.words = [None for _ in range(len(texts))]
        self.tokenized = [[] for _ in range(len(texts))]

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        item = item
        text = self.texts[item]
        tags = self.tags[item]

        ids = []
        target_tag = []
        words = []
        tokenized = []

        for i, s in enumerate(text):
            inputs = TOKENIZER.encode(
                str(s),
                add_special_tokens=False
            )
            tokenized += TOKENIZER.tokenize(str(s))
            input_len = len(inputs)
            ids.extend(inputs)
            target_tag.extend([tags[i]] * input_len)
            words.extend([i]*input_len)

        ids = ids[:config.NER_MAX_LENGTH - 2]
        target_tag = target_tag[:config.NER_MAX_LENGTH - 2]

        ids = [102] + ids + [103]
        o_tag = self.enc_tag.transform(["O"])[0]
        target_tag = [o_tag] + target_tag + [o_tag]
        words = words

        mask = [1] * len(ids)
        token_type_ids = [0] * len(ids)

        padding_len = config.NER_MAX_LENGTH - len(ids)

        ids = ids + ([0] * padding_len)
        mask = mask + ([0] * padding_len)
        token_type_ids = token_type_ids + ([0] * padding_len)
        target_tag = target_tag + ([0] * padding_len)
        self.words[item] = words
        self.tokenized[item] = tokenized

        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "target_tag": torch.tensor(target_tag, dtype=torch.long),
        }

    def get_entities(self, index, tags):
        words = self.words[index].copy()
        tokenized = self.tokenized[index].copy()
        entities = []

        if len(tokenized) > len(tags[1:-1]):
            tokenized = tokenized[:254]
            words = words[:254]
        assert len(tokenized) == len(tags[1:-1])
        assert len(tokenized) == len(words)

        tags = tags[1:-1]
        cur_tag = None

        try:
            for i in range(len(tokenized)):
                tag = tags[i].split("-")[1] if len(tags[i]) > 1 else tags[i]
                bio = tags[i].split("-")[0]

                if tokenized[i][:2] == "##":
                    tokenized[i] = tokenized[i][2:]

                if bio == "B" and ((i-1 >= 0 and words[i-1] < words[i]) or i == 0):
                    entity = {}
                    entity["word"] = tokenized[i]
                    entity["type"] = tag
                    entities.append(entity)
                    cur_tag = tag
                elif bio == "B" and cur_tag is None:
                    entity = {}
                    entity["word"] = tokenized[i]
                    entity["type"] = tag
                    entities.append(entity)
                    cur_tag = tag
                elif bio == "B" and cur_tag is not None and (i-1 >= 0 and words[i-1] == words[i]):
                    entities[-1]["word"] += tokenized[i]
                    cur_tag = tag
                elif bio == "I" and cur_tag is None:
                    pass
                elif bio == "I" and (i-1 >= 0 and words[i-1] < words[i]):
                    entities[-1]["word"] += " "+tokenized[i]
                elif bio == "I" and (i-1 >= 0 and words[i-1] == words[i]):
                    entities[-1]["word"] += tokenized[i]
                elif bio == "O":
                    cur_tag = None
                else:
                    print(bio, tag, cur_tag)
        except:
            for token, tag, word in zip(tokenized, tags, words):
                print(token, tag, word)

        final_entities = []
        for entity in entities:
            if entity["word"].find("[UNK]") == -1:
                final_entities.append(entity)
        return final_entities


# to convert text to input features to bert
def getRelationalFormat(entity1, entity2, text, tokenizer):
    tokens_a = tokenizer.tokenize(entity1+" "+entity2+"\t")
    max_length = 512
    tokens_b = tokenizer.tokenize(" ".join(text))

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
        "input_ids": torch.tensor(input_ids, dtype=torch.long).unsqueeze(0),
        "attention_mask": torch.tensor(input_mask, dtype=torch.long).unsqueeze(0),
        "token_type_ids": torch.tensor(input_type_ids, dtype=torch.long).unsqueeze(0),
    }

# filter cell entities for relation extraction


def filter_cell_types(entities):
    final_entities = []
    for entity in entities:
        if not (entity["type"] == "cell_line" or entity["type"] == "cell_type"):
            final_entities.append(entity)

    return final_entities

# get inference returns all the entities and relations found from single text document


def getInference(text, enc_tag):
    device = torch.device(config.RE_DEVICE)
    dataset = CORD19NERDataset(
        texts=text,
        tags=[[0 for _ in range(len(text[i]))] for i in range(len(text))],
        enc_tag=enc_tag
    )
    document_diseases = []
    document_chemicals = []
    document_proteins = []
    document_cid_relations = []
    document_cpr_relations = []

    with torch.no_grad():
        for i in range(len(dataset)):
            data = dataset[i]
            for k, v in data.items():
                data[k] = v.to(device).unsqueeze(0)

            disease_tag = ncbi.encode(**data)
            disease_tag = ncbi_enc.inverse_transform(disease_tag[0])
            chemical_tag = chemdner.encode(**data)
            chemical_tag = chemdner_enc.inverse_transform(chemical_tag[0])
            protein_tag = jnlpba.encode(**data)
            protein_tag = jnlpba_enc.inverse_transform(protein_tag[0])

            disease = dataset.get_entities(i, disease_tag)
            chemical = dataset.get_entities(i, chemical_tag)
            protein = dataset.get_entities(i, protein_tag)

            document_diseases.extend(disease)
            document_chemicals.extend(chemical)
            document_proteins.extend(protein)

            final_proteins = filter_cell_types(protein)

            if len(chemical) > 0 and len(final_proteins) > 0:
                for prot in final_proteins:
                    for chem in chemical:
                        data = getRelationalFormat(
                            chem["word"], prot["word"], text[i], chemprot_tokenizer)
                        for k, v in data.items():
                            data[k] = v.to(device)
                        label = chemprot(**data)
                        label = label[0].cpu().softmax(1).argmax()
                        if label != 1:
                            document_cpr_relations.append({
                                "entity_1": chem["word"],
                                "entity_2": prot["word"],
                                "relation": config.RE_CHEMPROT_RELATIONS[label]
                            })

            if len(disease) > 0 and len(chemical) > 0:
                for dis in disease:
                    for chem in chemical:
                        data = getRelationalFormat(
                            dis["word"], chem["word"], text[i], bc5cdr_tokenizer)
                        for k, v in data.items():
                            data[k] = v.to(device)
                        label = bc5cdr(**data)
                        label = label[0].cpu().softmax(1).argmax()
                        if label != 0:
                            document_cid_relations.append({
                                "entity_1": chem["word"],
                                "entity_2": dis["word"],
                                "relation": config.RE_BC5CDR_RELATIONS[label]
                            })

    return document_diseases, document_chemicals, document_proteins, document_cid_relations, document_cpr_relations


# get text into array format from metadata and full_text files
def getFullTextIfExists(pdf_json, pmc_json):
    if type(pdf_json) != str and type(pmc_json) != str:
        return ""

    filename = pmc_json if type(pdf_json) != str else pdf_json
    filename = filename.split(";")[0]

    path = os.path.join(config.GC_CORD_19_PATH,
                        "document_parses", "document_parses", filename)

    with open(path) as f:
        m = json.load(f)
        fullText = ""
        for para in m["body_text"]:
            fullText += para["text"]+" "
        return fullText.strip()


# load all the NER models
ncbi, ncbi_enc = load_model(
    config.NER_NCBI_DISEASE_META_PATH, config.NER_NCBI_DISEASE_MODEL_PATH)
jnlpba, jnlpba_enc = load_model(
    config.NER_JNLPBA_META_PATH, config.NER_JNLPBA_MODEL_PATH)
chemdner, chemdner_enc = load_model(
    config.NER_CHEMDNER_META_PATH, config.NER_CHEMDNER_MODEL_PATH)

# load all the RE models
bc5cdr = BertForSequenceClassification.from_pretrained(
    config.RE_BC5CDR_MODEL_PATH, num_labels=len(config.RE_BC5CDR_RELATIONS)).to(config.RE_DEVICE)
bc5cdr_tokenizer = BertTokenizerFast.from_pretrained(
    config.RE_BC5CDR_MODEL_PATH)
chemprot = BertForSequenceClassification.from_pretrained(
    config.RE_CHEMPROT_MODEL_PATH, num_labels=len(config.RE_CHEMPROT_RELATIONS)).to(config.RE_DEVICE)
chemprot_tokenizer = BertTokenizerFast.from_pretrained(
    config.RE_CHEMPROT_MODEL_PATH)

# read the metadata file
df = pd.read_csv(os.path.join(config.GC_CORD_19_PATH, "metadata.csv"))

# track entities and relations
diseasesCount = 0
chemicalsCount = 0
proteinsCount = 0
cidRelationsCount = 0
cprRelationsCount = 0
entitiesCount = 0
relationsCount = 0

# finding and storing all entities and relations from the CORD19 dataset
for i in tqdm(range(0, len(df))):
    row = df.iloc[i]
    title = sent_tokenize(row["title"])
    text = title

    if type(row["abstract"]) == str:
        abstract = sent_tokenize(row["abstract"])
        text += abstract

    fullText = getFullTextIfExists(
        row["pdf_json_files"], row["pmc_json_files"])
    if len(fullText) > 0:
        fullText = sent_tokenize(fullText)
        text += fullText

    for j in range(len(text)):
        text[j] = word_tokenize(text[j])

    diseases, chemicals, proteins, cid_relations, cpr_relations = getInference(
        text, ncbi_enc)

    diseasesCount += len(diseases)
    chemicalsCount += len(chemicals)
    proteinsCount += len(proteins)
    cidRelationsCount += len(cid_relations)
    cprRelationsCount += len(cpr_relations)

    entitiesCount = insertIntoEntitiesCollection(diseases, chemicals, proteins)
    relationsCount = insertIntoRelationsCollection(
        cid_relations, cpr_relations)

    print(f"DOCUMENT {i} COMPLETED")
    print(
        f"diseases : {diseasesCount}, chemicals : {chemicalsCount}, proteins: {proteinsCount}")
    print(
        f"cid relations : {cidRelationsCount}, cpr relations : {cprRelationsCount}")
    print(f"entities : {entitiesCount}, relations : {relationsCount}")
