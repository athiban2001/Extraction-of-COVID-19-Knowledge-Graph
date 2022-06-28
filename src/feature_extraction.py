import torch
import os
import config

import transformers

# loading the covid scibert model and tokenizer
covid_scibert_model_path = os.path.join(
    config.MODELS_PATH, config.CORD_SCIBERT_MODEL_NAME)

bert = transformers.AutoModel.from_pretrained(covid_scibert_model_path)
tokenizer = transformers.BertTokenizer.from_pretrained(
    covid_scibert_model_path,
    do_lower_case=True
)

# class that extracts features for each text in the input dataset


class EntityDataset:
    def __init__(self, texts, tags, enc_tag):
        # texts: [["hi", ",", "my", "name", "is", "abhishek"], ["hello".....]]
        # tags: [[1 2 3 4 1 5], [....].....]]
        # enc_tag: sklearn.LabelEncoder for the input dataset labels
        self.texts = texts
        self.tags = tags
        self.enc_tag = enc_tag

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = self.texts[item]
        tags = self.tags[item]

        ids = []
        target_tag = []

        for i, s in enumerate(text):
            inputs = tokenizer.encode(
                str(s),
                add_special_tokens=False
            )
            # abhishek: ab ##hi ##sh ##ek
            input_len = len(inputs)
            ids.extend(inputs)
            target_tag.extend([tags[i]] * input_len)

        ids = ids[:config.NER_MAX_LENGTH - 2]
        target_tag = target_tag[:config.NER_MAX_LENGTH - 2]

        ids = [102] + ids + [103]
        o_tag = self.enc_tag.transform(["O"])[0]
        target_tag = [o_tag] + target_tag + [o_tag]

        mask = [1] * len(ids)
        token_type_ids = [0] * len(ids)

        padding_len = config.NER_MAX_LENGTH - len(ids)

        ids = ids + ([0] * padding_len)
        mask = mask + ([0] * padding_len)
        token_type_ids = token_type_ids + ([0] * padding_len)
        target_tag = target_tag + ([0] * padding_len)

        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "target_tag": torch.tensor(target_tag, dtype=torch.long),
        }
