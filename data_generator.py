# Contains a wrapper class to generate the dataset with tokenization etc

import datasets
from datasets import load_dataset
from transformers import BertTokenizerFast

from constants import ID2LABEL
from utils import (
    remove_rejected_texts,
    add_token_labels,
    add_span_ner_labels,
    align_labels,
    split_texts,
    list_relations,
)


class data_generator:
    """
    Wrapper class for tokenizing and preprocessing the data.
    Contains tokenizer.
    """

    def __init__(self):
        self.tokenizer = load_tokenizer()

    def generate_dataset(self, remove_labels=False):
        """
        Generation and preprocessing of the dataset.
        Returns tokenized dataset in appropriate format.
        """
        dataset = load_data()

        # Perform preprocessing
        dataset = remove_rejected_texts(dataset)

        # Add labels
        dataset = dataset.map(add_token_labels, batched=True)
        dataset = dataset.map(add_span_ner_labels, batched=True)

        # Align labels
        dataset = dataset.map(align_labels, batched=True)

        # Remove some columns
        dataset = dataset.remove_columns(
            [
                "id",
                "text",
                "_input_hash",
                "_task_hash",
                "_is_binary",
                "spans",
                "_view_id",
                "answer",
                "_timestamp",
                "old_tokens",
            ]
        )

        # Split texts into batches of 400
        # dataset = dataset.map(split_texts, batched=True)

        # Tokenize and align labels
        dataset = dataset.map(self.tokenize_and_align_labels, batched=True)

        if remove_labels:
            # Remove some columns
            dataset = dataset.remove_columns(["tokens"])

        dataset = dataset.map(list_relations, batched=True, load_from_cache_file=False)
        dataset = dataset.map(self.tokenize_relations, batched=True)

        return dataset

    def tokenize_and_align_labels(self, texts, label_all_tokens=True):
        """
        Tokenizes the data, while making sure that subtokens are aligned with their label.
        """
        tokenized_inputs = self.tokenizer(
            texts["tokens"], truncation=True, is_split_into_words=True, padding="max_length"
        )

        labels = []
        for i, label in enumerate(texts["labels"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            # word_ids() => Return a list mapping the tokens
            # to their actual word in the initial sentence.
            # It Returns a list indicating the word corresponding to each token.
            previous_word_idx = None
            label_ids = []
            # Special tokens like `<s>` and `<\s>` are originally mapped to None
            # We need to set the label to -100 so they are automatically ignored in the loss function.
            for word_idx in word_ids:
                if word_idx is None:
                    # set –100 as the label for these special tokens
                    label_ids.append(-100)
                # For the other tokens in a word, we set the label to either the current label or -100, depending on
                # the label_all_tokens flag.
                elif word_idx != previous_word_idx:
                    # if current word_idx is != prev then its the most regular case
                    # and add the corresponding token
                    label_ids.append(label[word_idx])

                else:
                    # to take care of sub-words which have the same word_idx
                    # set -100 as well for them, but only if label_all_tokens == False
                    label_ids.append(label[word_idx] if label_all_tokens else -100)
                    # mask the subword representations after the first subword

                previous_word_idx = word_idx
            labels.append(label_ids)
        tokenized_inputs["labels"] = labels

        return {
            "labels": tokenized_inputs["labels"],
            "input_ids": tokenized_inputs["input_ids"],
            "token_type_ids": tokenized_inputs["token_type_ids"],
            "attention_mask": tokenized_inputs["attention_mask"],
        }

    def tokenize_relations(self, dataset):
        """
        Tokenizes the relations (true labels).
        Returns list of tuples (head span, child span, label).
        """
        all_relations = []

        # Loop through texts
        for relations in dataset["relations"]:
            rel_in_text = []

            # Tokenize the head and child span and form list of tuples
            for relation in relations:
                head, child, label = relation
                # Tokenize head
                head = self.tokenizer.convert_ids_to_tokens(
                    self.tokenizer(head, truncation=False)["input_ids"]
                )
                # Filter out irrelevant tokens
                head = filter(lambda token: token != "[CLS]" and token != "[SEP]", head)
                head = " ".join(head)
                # Tokenize child
                child = self.tokenizer.convert_ids_to_tokens(
                    self.tokenizer(child, truncation=False)["input_ids"]
                )
                # Filter out irrelevant spans
                child = filter(lambda token: token != "[CLS]" and token != "[SEP]", child)
                child = " ".join(child)

                # Append tuple to list
                rel_in_text.append((head, child, label))
            all_relations.append(rel_in_text)

        return {"relations": all_relations}

    def save_dataset(self, dataset, filename):
        dataset.to_json(filename)

    def load_tokenized_dataset(self, filename):
        return datasets.load_dataset("json", data_files=filename)


def load_data():
    return load_dataset("json", data_files="./dataset/small_test_set.json")


def load_tokenizer():
    return BertTokenizerFast.from_pretrained("bert-base-uncased")


def list_labels(tokens, labels, true_labels):
    """
    List tokens with their predicted and true label.
    For debugging purposes.
    """
    tokenizer = load_tokenizer()
    for text, label, truth in zip(tokens, labels, true_labels):
        if label > -100:
            label = ID2LABEL[label]
        if truth > -100:
            truth = ID2LABEL[truth]
        print(f"{tokenizer.convert_ids_to_tokens(text)}, {label}, {truth}")
