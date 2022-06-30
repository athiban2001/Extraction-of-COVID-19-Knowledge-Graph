from collections import Counter
import os

import pandas as pd
from tqdm import tqdm
import torch
from scipy import spatial

# fix based on relative paths
from OpenKE.openke.module.model import TransD
from OpenKE.openke.module.loss import MarginLoss
from OpenKE.openke.module.strategy import NegativeSampling
from OpenKE.openke.data import TrainDataLoader
from OpenKE.openke.config import Trainer

import config


# create entities to id mapping
def create_entity2id(df):
    entities = set()

    for i in range(len(df)):
        row = df.iloc[i]
        entities.add(row["entity_1"]+" chemical")
        if row["relation"] == "CID":
            entities.add(row["entity_2"]+" disease")
        else:
            entities.add(row["entity_2"]+" protein")

    entity2id = {entity: i for i, entity in enumerate(list(entities))}
    path = os.path.join(config.RL_DATASET_PATH, "entity2id.txt")
    with open(path, "w") as f:
        f.write(f"{len(entity2id)}\n")
        for entity in entity2id:
            f.write(f"{entity}\t{entity2id[entity]}\n")

    return entity2id


# create relation types to id mapping
def create_relation2id(df):
    relations = set()

    for i in range(len(df)):
        row = df.iloc[i]
        relations.add(row["relation"])

    relation2id = {relation: i for i, relation in enumerate(list(relations))}
    path = os.path.join(config.RL_DATASET_PATH, "relation2id.txt")
    with open(path, "w") as f:
        f.write(f"{len(relation2id)}\n")
        for relation in relation2id:
            f.write(f"{relation}\t{relation2id[relation]}\n")

    return relation2id


# create training relations to id mappings
def create_train2id(df):
    entity2id = create_entity2id(df)
    relation2id = create_relation2id(df)
    path = os.path.join(config.RL_DATASET_PATH, "train2id.txt")
    with open(path, "w") as f:
        f.write(f"{len(df)}\n")
        for i in range(len(df)):
            row = df.iloc[i]
            entity1_key = row["entity_1"]+" chemical"
            entity2_key = row["entity_2"] + \
                " protein" if row["relation"] != "CID" else row["entity_2"]+" disease"
            entity1 = entity2id[entity1_key]
            entity2 = entity2id[entity2_key]
            relation = relation2id[row["relation"]]
            f.write(f"{entity1}\t{entity2}\t{relation}\n")


def cosine_similiarity(embedding1, embedding2):
    return 1-spatial.distance.cosine(embedding1, embedding2)


def get_valid_entities(entities, threshold):
    valid_entities = []
    counter = Counter(entities)

    for entity in counter:
        if counter[entity] > threshold:
            valid_entities.append(entity)

    return valid_entities


def get_topn(entity_to_embedding, coronavirus, entity_type, n):
    similiarity_scores = []
    for entity in entity_to_embedding:
        if entity == "coronavirus disease" or entity_type not in entity:
            continue
        if entity_type == "disease":
            ent = entity[:-8]
            entities = valid_diseases
        elif entity_type == "chemical":
            ent = entity[:-9]
            entities = valid_chemicals
        else:
            ent = entity[:-8]
            entities = valid_proteins
        if ent not in entities:
            continue

        sim_dict = {}
        sim_dict["word"] = entity
        sim_dict["score"] = cosine_similiarity(
            coronavirus, entity_to_embedding[entity])
        similiarity_scores.append(sim_dict)

    similiarity_scores = sorted(
        similiarity_scores, reverse=True, key=lambda x: x["score"])
    return similiarity_scores[:n]

# print the results in table format


def print_table(result, entity_type):
    print("{:<50} {:<15}".format('Entity', 'Similiarity with Coronavirus'))
    print()
    for dic in result:
        print("{:<50} {:<15}".format(dic["word"].replace(
            f" {entity_type}", ""), dic["score"]))


df = pd.read_csv(config.GC_FINAL_RELATIONS_PATH)
create_train2id(df)

print("Created Dataset")


# dataloader for training
train_dataloader = TrainDataLoader(
    in_path=config.RL_DATASET_PATH,
    nbatches=100,
    threads=8,
    sampling_mode="normal",
    bern_flag=1,
    filter_flag=1,
    neg_ent=25,
    neg_rel=0)

# creating the model
transd = TransD(
    ent_tot=train_dataloader.get_ent_tot(),
    rel_tot=train_dataloader.get_rel_tot(),
    dim_e=400,
    dim_r=400,
    p_norm=1,
    norm_flag=True)

model = NegativeSampling(
    model=transd,
    loss=MarginLoss(margin=4.0),
    batch_size=train_dataloader.get_batch_size()
)

trainer = Trainer(model=model, data_loader=train_dataloader,
                  train_times=700, alpha=1.0, use_gpu=False)

print("Starting training process")
trainer.run()
transd.save_checkpoint(config.RL_MODEL_PATH)

# getting entity embeddings
results = transd.get_parameters("numpy")
entities_embedding = results["ent_embeddings.weight"]
entityid_to_embedding = {i: embedding for i,
                         embedding in enumerate(entities_embedding)}

entity_to_id = create_entity2id(df)

entity_to_embedding = {}
for key in entity_to_id:
    id = entity_to_id[key]
    embedding = entityid_to_embedding[id]
    entity_to_embedding[key] = embedding


coronavirus_embedding = entity_to_embedding["coronavirus disease"]


all_diseases = df.loc[df["relation"] == "CID"]["entity_2"].to_numpy()
all_chemicals = df["entity_1"].to_numpy()
all_proteins = df.loc[df["relation"] != "CID"]["entity_2"].to_numpy()


# finding valid entities which occur more than 5 times in the graph
valid_diseases = get_valid_entities(all_diseases, 5)
valid_proteins = get_valid_entities(all_proteins, 5)
valid_chemicals = get_valid_entities(all_chemicals, 5)

print(len(valid_diseases))
print(len(valid_proteins))
print(len(valid_chemicals))


# get top entities related to coronavirus based on cosine similarity
top_disease = get_topn(entity_to_embedding,
                       coronavirus_embedding, "disease", 25)
top_chemical = get_topn(entity_to_embedding,
                        coronavirus_embedding, "chemical", 25)
top_protein = get_topn(entity_to_embedding,
                       coronavirus_embedding, "protein", 25)


print("Top Diseases related to Corona virus")
print()
print()
print_table(top_disease, "disease")

print("Top Chemicals related to Corona virus")
print()
print()
print_table(top_chemical, "chemical")

print("Top Proteins related to Corona virus")
print()
print()
print_table(top_protein, "protein")
