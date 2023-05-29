# rel_extractor
from transformers import BertTokenizerFast

from data_generator import load_tokenizer
from constants import ID2LABEL


class rel_extractor:
    def __init__(self):
        self.relations = []

    def extract_relations(self, dataset):
        self.combine_entities(dataset)

    def combine_entities(self, dataset):
        tokenizer = load_tokenizer()
        print(dataset)
        input_ids = dataset["train"]["input_ids"]
        predictions = dataset["train"]["labels"]
        all_entities = []
        all_labels = []

        for token, prediction in zip(input_ids, predictions):
            entities_in_batch = []
            labels_in_batch = []
            print(len(token))
            print(len(prediction))
            index = 0
            for tok, pred in zip(token, prediction):
                if pred != -100:
                    if ID2LABEL[pred] == "O":
                        entities_in_batch.append(tok)
                        labels_in_batch.append(pred)

                    if ID2LABEL[pred] == "B-CLASS":
                        print(ID2LABEL[pred], tokenizer.convert_ids_to_tokens(tok))
                        i = index
                        if i < len(prediction):
                            while ID2LABEL[prediction[i + 1]] == "I-CLASS":
                                print(
                                    ID2LABEL[prediction[i + 1]], tokenizer.convert_ids_to_tokens(token[i + 1])
                                )
                                if i > len(prediction) - 1:
                                    break
                                i += 1
                index += 1
            all_entities.append(entities_in_batch)
            all_labels.append(labels_in_batch)
