# Extraction of COVID-19 Knowledge Graph Through Mining of Unstructured Biomedical Corpora

The project contains the code and documentation of our Final Year Project.

## Abstract

The number of biomedical articles published is increasing rapidly over the years. Currently there are over 30 million publications in PubMed and over 25 million references in Medline. Among these concepts, Biomedical Named Entity Recognition (BioNER) and Biomedical Relation Extraction (BioRD) are the most important. Graphs are practical resources for defining relationships and are applicable in real-world scenarios. In the biomedical domain, Knowledge Graph is used to visualize the relationships between various entities such as proteins, chemicals and diseases. The system defines a biomedical knowledge graph as the following: a resource that integrates one or more expert-derived sources of information into a graph where nodes represent biomedical entities and edges represent relationships between two entities. The system uses Named Entity Recognition models for disease recognition, chemical recognition and protein recognition. Then the system uses the Chemical - Disease Relation Extraction and Chemical - Protein Relation Extraction models. And the system extracts the entities and relations from the CORD-19 dataset using the models. The system then creates a Knowledge Graph for the extracted relations and entities. The system performs Representation Learning on this KG to get the embeddings of all entities and get the  top related diseases, chemicals and proteins with respect to COVID-19.

## Architecture Diagram
![Architecture Diagram](/documentation/Review_3/architecture.png)

## Models

The models developed for this project are given [here.](https://github.com/athiban2001/Extraction-of-COVID-19-Knowledge-Graph/blob/master/models/README.md)

## Folder Structure

````
datasets/ ------------------ contains all the datasets used in the project
documentation/ ------------- contains all the documentation done for the review process
models/ -------------------- will contain the resultant models
results/ ------------------- contains the intermediate results between processes
src/ ----------------------- contains all the python code
````

## Procedures

1. Configure the project in the **src/config.py** file. Change all the necessary parameters accordingly and check whether all the paths are correct.
2. Check whether all the libraries are installed in the environment. The environment in which each file are run are documented in the **src/requirements.txt** file.
3. **Preprocessing** step can be done as follows, it stores the CORD_SCIBERT model in models folder.
    ````
    python src/preprocessing.py
    ````
4. **Feature Extraction** step can be done as follows, it creates the input features for Named Entity Recognition
    ````
    python src/feature_extraction.py
    ````
5. **Named Entity Recognition** step can be done as multiple steps.<br><br/>
   1. Create the Disease Named Entity Recognition model.
        ````
        python src/named_entity_recognition/training.py ncbi
        ````
   2. Create the Chemical Named Entity Recognition model.
        ````
        python src/named_entity_recognition/training.py chemdner
        ````
   3. Create the Protein Named Entity Recognition model.
        ````
        python src/named_entity_recognition/training.py jnlpba
        ````
6. **Relation Extraction** step can be done as multiple steps.<br><br/>
   1. Create the Chemical-Disease relation extraction model.
        ````
        python src/relation_extraction.py bc5cdr
        ````
   2. Create the Chemical-Protein relation extraction model.
        ````
        python src/relation_extraction.py chemprot
        ````
7. **Graph Construction** step can be done as multiple steps.<br><br/>
   1. Extract the entities and relations from the CORD-19 dataset.
        ````
        python src/graph_construction/extraction.py
        ````
   2. To filter the found entities based on entity occurrence count.
        ````
        python src/graph_construction/entities_filtering.py
        ````
   3. To create the knowledge graph based on the data.
        ````
        python src/graph_construction/neo4j.py
        ````
8. **Representation Learning** step can be donde as follows, to create embeddings of entities and using them to find most similar disease, chemicals and proteins to Coronavirus.
    ````
    python src/representation_learning.py
    ````