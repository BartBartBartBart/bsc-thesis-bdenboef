# Contains data pre-processing methods
# as well as some common utilities

import numpy as np
import evaluate
import random
import torch

from constants import (
    LABEL2ID,
    LABEL_LIST,
)


def remove_rejected_texts(dataset):
    """
    Filter the dataset and remove texts that were rejected during annotation.
    """
    indices_to_be_removed = []
    for index, texts in enumerate(dataset["train"]):
        if texts["answer"] == "reject":
            indices_to_be_removed.append(index)

    # create new train dataset
    dataset["train"] = dataset["train"].select(
        (i for i in range(len(dataset["train"])) if i not in indices_to_be_removed)
    )
    return dataset


def add_token_labels(text):
    """
    Set all NER-labels to "O"
    """
    for tokens in text["tokens"]:
        for token in tokens:
            token["label"] = LABEL2ID["O"]

    return text


def add_span_ner_labels(texts):
    """
    Add the corresponding NER+chunking label (BIO tagging scheme) to tokens
    """
    # Filter through all spans
    for span_nr, spans in enumerate(texts["spans"]):
        for span in spans:
            start = span["token_start"]
            if start is None:
                continue
            end = span["token_end"]
            label = span["label"]

            # Add the labels to the tokens
            for token_nr, tokens in enumerate(texts["tokens"]):
                for token in tokens:
                    if token_nr == span_nr and token["id"] >= start and token["id"] <= end:
                        if token["id"] == start:
                            token["label"] = LABEL2ID["B-" + label]
                        else:
                            token["label"] = LABEL2ID["I-" + label]

    return texts


def align_labels(texts):
    """
    Aligns labels with their tokens.
    """
    all_tokens = []
    all_labels = []
    all_ids = []

    for text_nr, tokens in enumerate(texts["tokens"]):
        data = {}
        data["tokens"] = []
        data["labels"] = []

        for token in tokens:
            data["labels"].append(token["label"])
            data["tokens"].append(token["text"])
        all_tokens.append(data["tokens"])
        all_labels.append(data["labels"])
        all_ids.append([text_nr])

    return {"tokens": all_tokens, "labels": all_labels, "id": all_ids, "old_tokens": texts["tokens"]}


def divide_chunks(text, n):
    """
    Return chunks of size n from text.
    """
    # looping till length l
    for i in range(0, len(text), n):
        yield text[i : i + n]


def split_texts(dataset):
    """
    Splits all the texts into batches of 400 tokens (this leaves room for subtokens, generated by tokenizing)
    """
    all_labels = []
    all_tokens = []

    for labels, tokens in zip(dataset["labels"], dataset["tokens"]):
        for label, token in zip(labels, tokens):
            all_tokens.append(token)
            all_labels.append(label)

    # Divide into batches of size 400 (leaving room for subwords)
    all_labels = divide_chunks(all_labels, 400)
    all_tokens = divide_chunks(all_tokens, 400)

    all_labels = list(all_labels)
    all_tokens = list(all_tokens)

    assert len(all_labels) == len(all_tokens)

    return {"labels": all_labels, "tokens": all_tokens}


def compute_metrics(eval_preds):
    """
    Custom metrics function for the transformers Trainer.
    Convert logits into predictions and compare against the true_labels.
    Returns the base metrics: prec, rec, f1 and accuracy.
    """
    metric = evaluate.load("seqeval")

    pred_logits, labels = eval_preds

    pred_logits = np.argmax(pred_logits, axis=2)

    # We remove all the values where the label is -100
    predictions = [
        [LABEL_LIST[eval_preds] for (eval_preds, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(pred_logits, labels)
    ]

    true_labels = [
        [LABEL_LIST[l] for (eval_preds, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(pred_logits, labels)
    ]

    # Left this in for debugging purposes
    # for pred, true in zip(predictions, true_labels):
    #     for pred_label, true_label in zip(pred,true):
    #         print(f"({pred_label}, {true_label})")

    results = metric.compute(predictions=predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }


def cv_split(dataset, fold):
    """
    Hardcoded cross-validation split.
    TODO: make this dynamic.
    """
    if fold == 0:
        # train_split = [0, 1, 2, 3, 4, 5]
        # eval_split = [6, 7, 8]
        train_split = [0, 1, 2, 3, 4, 5, 6]
        eval_split = [7, 8]
    elif fold == 1:
        # train_split = [7, 8, 0, 1, 2, 3]
        # eval_split = [4, 5, 6]
        train_split = [7, 8, 0, 1, 2, 3, 4]
        eval_split = [5, 6]
    elif fold == 2:
        # train_split = [5, 6, 7, 8, 0, 1]
        # eval_split = [2, 3, 4]
        train_split = [5, 6, 7, 8, 0, 1, 2]
        eval_split = [3, 4]
    elif fold == 3:
        # train_split = [3, 4, 5, 6, 7, 8]
        # eval_split = [0, 1, 2]
        train_split = [3, 4, 5, 6, 7, 8, 0]
        eval_split = [1, 2]

    # create new train dataset
    train_set = dataset["train"].select((i for i in range(len(dataset["train"])) if i in train_split))

    # create new eval dataset
    eval_set = dataset["train"].select((i for i in range(len(dataset["train"])) if i in eval_split))

    return train_set, eval_set


def list_relations(dataset):
    """
    Transforms the relations into an understandable format from the Prodigy JSON.
    """
    all_relations = []

    for batch_nr, relations in enumerate(dataset["relations"]):
        # contains all relations in tuple shape (head,child,label)
        rel_in_batch = []

        for relation in relations:
            head_begin = relation["head_span"]["token_start"]
            head_end = relation["head_span"]["token_end"]
            child_begin = relation["child_span"]["token_start"]
            child_end = relation["child_span"]["token_end"]

            head_entity = ""
            child_entity = ""
            for token_nr, token in enumerate(dataset["tokens"][batch_nr]):
                if token_nr >= head_begin and token_nr <= head_end:
                    if head_entity != "":
                        head_entity += " "
                    head_entity += token
                if token_nr >= child_begin and token_nr <= child_end:
                    if child_entity != "":
                        child_entity += " "
                    child_entity += token
            label = relation["label"]

            # For debugging purposes
            # print(f"({head_entity},{child_entity})--> REL: {label}")

            rel_in_batch.append((head_entity, child_entity, label))
        all_relations.append(rel_in_batch)

    return {"relations": all_relations}


def count_relations(dataset):
    """
    Counts the appearances of different relation types.
    """
    rel_counts = {
        "ASSOCIATION1": 0,
        "ASSOCIATION1..*": 0,
        "ASSOCIATION*": 0,
        "ATTRIBUTE": 0,
        "SUBTYPE": 0,
        "OPERATION": 0,
        "COMPOSITION": 0,
        "AGGREGATION": 0,
        "SPAN": 0,
        "COREF": 0,
    }
    for relations in dataset["train"]["relations"]:
        for rel in relations:
            rel_counts[rel[2]] += 1

    return rel_counts


def set_seeds(seed=1):
    """
    Set seeds for reproducibility.
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # multi-GPU


def calculate_score(true_rels, predictions):
    """
    Custom score function that counts correct, false and missed predictions per relation type.
    """
    scores_per_class = {
        "ASSOCIATION1": [0, 0, 0],
        "ASSOCIATION1..*": [0, 0, 0],
        "ASSOCIATION*": [0, 0, 0],
        "ATTRIBUTE": [0, 0, 0],
        "SUBTYPE": [0, 0, 0],
        "OPERATION": [0, 0, 0],
        "COMPOSITION": [0, 0, 0],
        "AGGREGATION": [0, 0, 0],
        "SPAN": [0, 0, 0],
        "COREF": [0, 0, 0],
    }

    for batch, prediction in enumerate(predictions):
        # wrong = 0
        # correct = 0
        for pred in prediction:

            # Correct prediction
            if pred in true_rels[batch]:
                scores_per_class[pred[2]][0] += 1
                true_rels[batch].remove(pred)
                # correct += 1

            # Wrong prediction
            else:
                print("Wrong prediction: ", pred)
                scores_per_class[pred[2]][1] += 1
                # wrong += 1

        # Missed predictions
        print("Missed", true_rels[batch])
        print(f"Missed {len(true_rels[batch])} relations")
        for missed_relation in true_rels[batch]:
            scores_per_class[missed_relation[2]][2] += 1
        # wrong += len(true_rels[batch])

    print("Relation  -  Correct pred  -  Wrong pred  -  Missed pred")
    print(scores_per_class)

    return scores_per_class


def print_reports(reports):
    """
    Print clasification reports
    """
    for report in reports:
        for x in report:
            print(x)


def parse_precomputed_preds(preds):
    """
    For debugging purposes.
    Parse saved output into appropriate input format for RE model.
    """
    precomputed_preds = []
    for pred in preds:
        batch = []
        for p in pred:
            x = p[0].replace(" ", ",")
            x = [int(num) for num in x.split(",")]
            batch.append(x)
        precomputed_preds.append(batch)

    return precomputed_preds
