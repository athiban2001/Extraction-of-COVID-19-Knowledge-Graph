import os

# Folders
SRC_FOLDER_PATH = os.path.dirname(__file__)
RESULTS_PATH = os.path.join(SRC_FOLDER_PATH, "..", "results")
PRETRAINED_MODEL_PATH = os.path.join(RESULTS_PATH, "pretrained_model")
MODELS_PATH = os.path.join(SRC_FOLDER_PATH, "..", "models")

# Preprocessing and Feature Extraction
CORD_FINETUNING_EPOCH = 5
CORD_FINETUNING_BATCH_SIZE = 16
CORD_SCIBERT_MODEL_PATH = os.path.join(
    SRC_FOLDER_PATH, "..", "models", "COVID-scibert-latest")
CORD_19_BLOCK_SIZE = 256

# Named Entity Recognition (NER)
NER_MAX_LENGTH = 256
NER_DEVICE = "cpu"
NER_BATCH_SIZE = 8
NER_NUM_WORKERS = 5

# NER NCBI Disease
NER_NCBI_DISEASE_EPOCHS = 10
NER_NCBI_DISEASE_FOLDER = os.path.join(
    SRC_FOLDER_PATH, "..", "datasets", "NER", "NCBIDisease")
NER_NCBI_DISEASE_TRAIN_FILE = os.path.join(
    NER_NCBI_DISEASE_FOLDER, "NCBI_train.tsv")
NER_NCBI_DISEASE_TEST_FILE = os.path.join(
    NER_NCBI_DISEASE_FOLDER, "NCBI_test.tsv")
NER_NCBI_DISEASE_DEV_FILE = os.path.join(
    NER_NCBI_DISEASE_FOLDER, "NCBI_dev.tsv")
NER_NCBI_DISEASE_CLASS_FILE = os.path.join(
    NER_NCBI_DISEASE_FOLDER, "classes.txt")
NER_NCBI_DISEASE_MODEL_PATH = os.path.join(
    MODELS_PATH, "NER", "NCBIDisease.bin")
NER_NCBI_DISEASE_META_PATH = os.path.join(
    MODELS_PATH, "NER", "NCBIDisease_meta.bin")

# NER CHEMDNER
NER_CHEMDNER_EPOCHS = 6
NER_CHEMDNER_FOLDER = os.path.join(
    SRC_FOLDER_PATH, "..", "datasets", "NER", "CHEMDNER")
NER_CHEMDNER_TRAIN_FILE = os.path.join(NER_CHEMDNER_FOLDER, "train.tsv")
NER_CHEMDNER_TEST_FILE = os.path.join(NER_CHEMDNER_FOLDER, "test.tsv")
NER_CHEMDNER_DEV_FILE = os.path.join(NER_CHEMDNER_FOLDER, "devel.tsv")
NER_CHEMDNER_CLASS_FILE = os.path.join(NER_CHEMDNER_FOLDER, "classes.txt")
NER_CHEMDNER_MODEL_PATH = os.path.join(MODELS_PATH, "NER", "CHEMDNER.bin")
NER_CHEMDNER_META_PATH = os.path.join(MODELS_PATH, "NER", "CHEMDNER_meta.bin")

# NER JNLPBA
NER_JNLPBA_EPOCHS = 4
NER_JNLPBA_FOLDER = os.path.join(
    SRC_FOLDER_PATH, "..", "datasets", "NER", "JNLPBA")
NER_JNLPBA_TRAIN_FILE = os.path.join(NER_JNLPBA_FOLDER, "train.tsv")
NER_JNLPBA_TEST_FILE = os.path.join(NER_JNLPBA_FOLDER, "test.tsv")
NER_JNLPBA_DEV_FILE = os.path.join(NER_JNLPBA_FOLDER, "devel.tsv")
NER_JNLPBA_CLASS_FILE = os.path.join(NER_JNLPBA_FOLDER, "classes.txt")
NER_JNLPBA_MODEL_PATH = os.path.join(MODELS_PATH, "NER", "JNLPBA.bin")
NER_JNLPBA_META_PATH = os.path.join(MODELS_PATH, "NER", "JNLPBA_meta.bin")

# Relation Extraction (RE)
RE_MAX_LENGTH = 512
RE_PRETRAINED_MODEL = "allenai/scibert_scivocab_uncased"
RE_DEVICE = "cuda"
RE_TRAIN_BATCH_SIZE = 8
RE_EVAL_BATCH_SIZE = 20
RE_RESULTS_PATH = os.path.join(RESULTS_PATH, "RE")

# RE BC5CDR
RE_BC5CDR_EPOCHS = 3
RE_BC5CDR_RELATIONS = ["No Relation", "Chemical-Induced Disease"]
RE_BC5CDR_FOLDER = os.path.join(
    SRC_FOLDER_PATH, "..", "datasets", "RE", "BC5CDR", "CDR.Corpus.v010516")
RE_BC5CDR_TRAIN_FILE = os.path.join(
    RE_BC5CDR_FOLDER, "CDR_TrainingSet.BioC.xml")
RE_BC5CDR_DEV_FILE = os.path.join(
    RE_BC5CDR_FOLDER, "CDR_DevelopmentSet.BioC.xml")
RE_BC5CDR_TEST_FILE = os.path.join(RE_BC5CDR_FOLDER, "CDR_TestSet.BioC.xml")
RE_BC5CDR_MODEL_PATH = os.path.join(
    SRC_FOLDER_PATH, "..", "models", "RE", "BC5CDR_SciBERT")

# RE CHEMPROT
RE_CHEMPROT_EPOCHS = 4
RE_CHEMPROT_RELATIONS = ["CPR:1", "CPR:10", "CPR:2",
                         "CPR:3", "CPR:4", "CPR:5", "CPR:6", "CPR:7", "CPR:9"]
RE_CHEMPROT_FOLDER = os.path.join(
    SRC_FOLDER_PATH, "..", "datasets", "RE", "CHEMPROT")
RE_CHEMPROT_TRAIN_FOLDER = os.path.join(
    RE_CHEMPROT_FOLDER, "chemprot_training", "chemprot_training")
RE_CHEMPROT_TEST_FOLDER = os.path.join(
    RE_CHEMPROT_FOLDER, "chemprot_test_gs", "chemprot_test_gs")
RE_CHEMPROT_DEV_FOLDER = os.path.join(
    RE_CHEMPROT_FOLDER, "chemprot_development", "chemprot_development")
RE_CHEMPROT_MODEL_PATH = os.path.join(
    SRC_FOLDER_PATH, "..", "models", "RE", "CHEMPROT_SciBERT")

# Graph Construction
GC_MONGO_URL = "mongodb://127.0.0.1/"
GC_NEO4J_URL = "neo4j://localhost:7687/"
GC_CORD_19_PATH = os.path.join(SRC_FOLDER_PATH, "..", "datasets", "CORD19")
GC_FINAL_RELATIONS_PATH = os.path.join(
    RESULTS_PATH, "GC", "full_relations.csv")
GC_DB_RELATIONS_PATH = os.path.join(
    RESULTS_PATH, "GC", "database_dump", "relations.csv")
