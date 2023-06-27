# Contains data pre-processing methods

import numpy as np
import evaluate
import random
import torch

from constants import (
    ID2LABEL,
    LABEL2ID,
    LABEL_LIST,
)


def remove_rejected_texts(dataset):
    indices_to_be_removed = []
    for index, texts in enumerate(dataset["train"]):
        if texts["answer"] == "reject":
            indices_to_be_removed.append(index)

    # create new train dataset
    dataset["train"] = dataset["train"].select(
        (i for i in range(len(dataset["train"])) if i not in indices_to_be_removed)
    )
    return dataset


# Give all tokens the label "O"
def add_token_labels(text):
    for tokens in text["tokens"]:
        for token in tokens:
            token["label"] = LABEL2ID["O"]

    return text


# Give tokens that are part of a span the correct NER and chunking label
def add_span_ner_labels(texts):
    i = False
    for span_nr, spans in enumerate(texts["spans"]):
        for span in spans:
            start = span["token_start"]
            if start is None:
                print("a")
                print(span_nr)
                i = True
                print(span)
                continue
            if i:
                print(span)
                i = False
            end = span["token_end"]
            label = span["label"]

            for token_nr, tokens in enumerate(texts["tokens"]):
                for token in tokens:
                    if token_nr == span_nr and token["id"] >= start and token["id"] <= end:
                        if token["id"] == start:
                            if LABEL2ID["B-" + label] == []:
                                print("B-" + label)
                            token["label"] = LABEL2ID["B-" + label]
                        else:
                            if LABEL2ID["I-" + label] == []:
                                print("I-" + label)
                            token["label"] = LABEL2ID["I-" + label]

    return texts


# Align labels
def align_labels(texts):
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


# Yield successive n-sized
# chunks from l.
def divide_chunks(l, n):

    # looping till length l
    for i in range(0, len(l), n):
        yield l[i : i + n]


# Merge all texts and split into batches of 400 (leaves room for subtokens)
def split_texts(dataset):
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


# def check_tokens(dataset):
#     for token, label in zip(
#         tokenizer.convert_ids_to_tokens(dataset["train"][4]["input_ids"]), dataset["train"][4]["labels"]
#     ):
#         if label > -100:
#             label = ID2LABEL[label]
#         print(f"{token:_<40} {label}")


def compute_metrics(eval_preds):
    metric = evaluate.load("seqeval")

    pred_logits, labels = eval_preds

    pred_logits = np.argmax(pred_logits, axis=2)
    # the logits and the probabilities are in the same order,
    # so we don’t need to apply the softmax

    # We remove all the values where the label is -100
    predictions = [
        [LABEL_LIST[eval_preds] for (eval_preds, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(pred_logits, labels)
    ]

    true_labels = [
        [LABEL_LIST[l] for (eval_preds, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(pred_logits, labels)
    ]

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
    if fold == 0:
        train_split = [0, 1, 2, 3, 4, 5]
        eval_split = [6, 7, 8]
        # test_split = [7, 8]
    elif fold == 1:
        train_split = [7, 8, 0, 1, 2, 3]
        eval_split = [4, 5, 6]
        # test_split = [5, 6]
    elif fold == 2:
        train_split = [5, 6, 7, 8, 0, 1]
        eval_split = [2, 3, 4]
        # test_split = [3, 4]
    elif fold == 3:
        train_split = [3, 4, 5, 6, 7, 8]
        # eval_split = [6 ,8]
        eval_split = [0, 1, 2]
        # test_split = [0, 1]

    # create new train dataset
    train_set = dataset["train"].select((i for i in range(len(dataset["train"])) if i in train_split))

    # create new eval dataset
    eval_set = dataset["train"].select((i for i in range(len(dataset["train"])) if i in eval_split))

    # create net test dataset
    # test_set = dataset["train"].select((i for i in range(len(dataset["train"])) if i in test_split))

    return train_set, eval_set  # , test_set


def list_relations(dataset):
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


# def extract_relations(dataset):
#     dataset = dataset.map(list_relations, batched=True)
#     print(dataset["train"][0]["relations"])
#     return dataset


def count_relations(dataset):
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


def set_seeds(seed=1234):
    """Set seeds for reproducibility."""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # multi-GPU


def calculate_score(true_rels, predictions):
    all_correct = 0
    all_wrong = 0

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
        print(f"Batch {batch}")
        wrong = 0
        correct = 0
        for pred in prediction:
            if pred in true_rels[batch]:
                # index = true_rels[batch].index(pred)
                # print(pred, true_rels[batch][index])
                scores_per_class[pred[2]][0] += 1
                true_rels[batch].remove(pred)
                correct += 1
            else:
                print("Wrong prediction: ", pred)
                scores_per_class[pred[2]][1] += 1
                wrong += 1
        print("Missed", true_rels[batch])
        print(f"Missed {len(true_rels[batch])} relations")
        for missed_relation in true_rels[batch]:
            scores_per_class[missed_relation[2]][2] += 1
        wrong += len(true_rels[batch])
        print(f"correct: {correct}, wrong: {wrong}")
        all_correct += correct
        all_wrong += wrong

    print("Relation  -  Correct pred  -  Wrong pred  -  Missed pred")
    print(scores_per_class)

    return all_correct, all_wrong, scores_per_class


def print_reports(reports):
    for report in reports:
        for x in reports:
            print(x)
            # for y in reports[x]:
            #     print (y,':',reports[x][y])
