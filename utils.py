# Contains data pre-processing methods

import numpy as np
import evaluate

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


def list_labels(dataset, text_nr):
    for text, label in zip(dataset["train"][text_nr]["tokens"], dataset["train"][text_nr]["labels"]):
        if label > -100:
            label = ID2LABEL[label]
        print(f"{text}, {label}")


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


def clean_and_split_dataset(dataset):
    # create new train dataset
    train_set = dataset["train"].select((i for i in range(len(dataset["train"])) if i not in [4, 5, 6]))

    # create new eval dataset
    eval_set = dataset["train"].select((i for i in range(len(dataset["train"])) if i not in [0, 1, 2, 3]))
    return train_set, eval_set


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


def extract_relations(dataset):
    dataset = dataset.map(list_relations, batched=True)
    print(dataset["train"][0]["relations"])
    return dataset


# IMPLEMENTATION IDEA:
# list of spans (so can be single words) with ID
# list of relations of shape ("SPAN1", "SPAN2", "LABEL")
# Per sentence? When do I tokenize?
# Would this be a good input structure?
# "This is a sentence" -> [[rel1],[rel2]]
# and so on..
