from neo4j import GraphDatabase
import pandas as pd
import tqdm

import config

uri = config.GC_NEO4J_URL
driver = GraphDatabase.driver(uri, auth=("neo4j", "fyp"))


def find_entity_type(relation):
    if relation == "CID":
        return "Chemical", "Disease"
    else:
        return "Chemical", "Protein"


def getInsertQuery(entity_1_rel, entity_2_rel, relation):
    insert_query_1 = "MERGE (chem:Entity {name: $entity_1_name, type: $entity_1_rel})"
    insert_query_2 = "MERGE (disprot:Entity {name: $entity_2_name, type: $entity_2_rel})"
    insert_query = f"""
    {insert_query_1}
    {insert_query_2}
    """
    return insert_query


def enter_relation(tx, entity_1, entity_2, relation):
    entity_1_rel, entity_2_rel = find_entity_type(relation)
    tx.run(getInsertQuery(entity_1_rel, entity_2_rel, relation),
           entity_1_name=entity_1,
           entity_1_rel=entity_1_rel,
           entity_2_name=entity_2,
           entity_2_rel=entity_2_rel,
           )
    insert_query = f"""
    MATCH
    (a:Entity),
    (b:Entity)
    WHERE a.name = $entity_1_name AND a.type = $entity_1_rel AND b.name = $entity_2_name AND b.type = $entity_2_rel
    MERGE (a)-[:`{relation}`]->(b)
    """
    tx.run(insert_query,
           entity_1_name=entity_1,
           entity_1_rel=entity_1_rel,
           entity_2_name=entity_2,
           entity_2_rel=entity_2_rel,
           )


df = pd.read_csv(config.GC_FINAL_RELATIONS_PATH)

with driver.session() as session:
    for i in tqdm.tqdm(range(len(df))):
        row = df.iloc[i]
        session.write_transaction(
            enter_relation, row["entity_1"], row["entity_2"], row["relation"])
        break
