from collections import Counter

import pandas as pd
from tqdm import tqdm

import config

df = pd.read_csv(config.GC_DB_RELATIONS_PATH)


def getValidCount(arr, threshold):
    counter = Counter(arr)
    results = []

    for key in counter:
        if counter[key] >= threshold:
            results.append(key)

    return results


chemicals = df["entity_1"].to_numpy()
proteins = df.loc[df["relation"] != "CID"]["entity_2"].to_numpy()
diseases = df.loc[df["relation"] == "CID"]["entity_2"].to_numpy()

valid_chemicals = getValidCount(chemicals, 5)
valid_proteins = getValidCount(proteins, 5)
valid_diseases = getValidCount(diseases, 5)
valid_entities = {key: True for key in valid_diseases +
                  valid_chemicals+valid_diseases}

final_df = pd.DataFrame(columns=["entity_1", "entity_2", "relation"])
count = 0

for i in tqdm(range(len(df))):
    row = df.iloc[i]
    if row["entity_1"] in valid_entities or row["entity_2"] in valid_entities:
        count += 1
        final_df.loc[len(final_df.index)] = [row["entity_1"],
                                             row["entity_2"], row["relation"]]

print("Initial no. of relations from the database : ", len(df))
final_df.to_csv(config.GC_FINAL_RELATIONS_PATH, index=False)

print("No. of entities which occurs more than 5 times : ", len(
    valid_diseases)+len(valid_proteins)+len(valid_chemicals))
print("No. of relations which contains atlease one 5-occurrence entity : ", count)
