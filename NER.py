import datasets
from datasets import Dataset
from datasets import load_dataset
from transformers import BertTokenizerFast 
from transformers import AutoTokenizer
import numpy as np
from transformers import DataCollatorWithPadding, AutoModelForTokenClassification, TrainingArguments, Trainer

id2label = {
    0: 'O',
    1: 'B-CLASS',
    2: 'I-CLASS',
    3: 'B-ATTRIBUTE',
    4: 'I-ATTRIBUTE',
    5: 'B-ASSOCIATION',
    6: 'I-ASSOCIATION',
    7: 'B-SYSTEM',
    8: 'I-SYSTEM',
    9: 'B-OPERATION',
    10: 'I-OPERATION'
}
label2id = {
    'O': 0,
    'B-CLASS': 1,
    'I-CLASS': 2,
    'B-ATTRIBUTE': 3,
    'I-ATTRIBUTE': 4,
    'B-ASSOCIATION': 5,
    'I-ASSOCIATION': 6,
    'B-SYSTEM': 7,
    'I-SYSTEM': 8,
    'B-OPERATION': 9,
    'I-OPERATION': 10
}

tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
label_list = ['O', 'B-CLASS', 'I-CLASS', 'B-ATTRIBUTE', 'I-ATTRIBUTE', 'B-ASSOCIATION', 'I-ASSOCIATION', 'B-SYSTEM', 'I-SYSTEM', 'B-OPERATION', 'I-OPERATION']
metric = datasets.load_metric("seqeval")

class NER_Model():
    def __init__(self):
        self.tokenizer = load_tokenizer()
        self.dataset = load_data()
        # Map id to label
        self.id2label = {
            0: 'O',
            1: 'B-CLASS',
            2: 'I-CLASS',
            3: 'B-ATTRIBUTE',
            4: 'I-ATTRIBUTE',
            5: 'B-ASSOCIATION',
            6: 'I-ASSOCIATION',
            7: 'B-SYSTEM',
            8: 'I-SYSTEM',
            9: 'B-OPERATION',
            10: 'I-OPERATION'
        }
        self.label2id = {
            'O': 0,
            'B-CLASS': 1,
            'I-CLASS': 2,
            'B-ATTRIBUTE': 3,
            'I-ATTRIBUTE': 4,
            'B-ASSOCIATION': 5,
            'I-ASSOCIATION': 6,
            'B-SYSTEM': 7,
            'I-SYSTEM': 8,
            'B-OPERATION': 9,
            'I-OPERATION': 10
        }
        self.label_list = ['O', 'B-CLASS', 'I-CLASS', 'B-ATTRIBUTE', 'I-ATTRIBUTE', 'B-ASSOCIATION', 'I-ASSOCIATION', 'B-SYSTEM', 'I-SYSTEM', 'B-OPERATION', 'I-OPERATION']

        # Load bert base model
        self.model = load_model()
        
        # Load data collater
        self.data_collator = load_data_collater(self.tokenizer)
        
        self.metric = load_seqeval()
        
        self.args = TrainingArguments( 
            "test-ner",
            save_strategy = "epoch",
            evaluation_strategy = "epoch", 
            learning_rate=9e-5, 
            per_device_train_batch_size=16, 
            per_device_eval_batch_size=16, 
            num_train_epochs=40, 
            weight_decay=0.01, 
            warmup_steps = 10,
            load_best_model_at_end=True,
            logging_steps=10
        ) 

def load_data():
    return load_dataset("json", data_files="./Dataset/small_test_set.json")

def load_tokenizer():
    return BertTokenizerFast.from_pretrained("bert-base-uncased")

def load_model():
    return AutoModelForTokenClassification.from_pretrained("bert-base-uncased", num_labels=11)

def load_data_collater(tokenizer):
    return DataCollatorWithPadding(tokenizer, padding=True)

def load_seqeval():
    return datasets.load_metric("seqeval")

def remove_rejected_texts(dataset):
    indices_to_be_removed = []
    for index, texts in enumerate(dataset["train"]):
        if texts["answer"] == 'reject':
            indices_to_be_removed.append(index)
            
    # create new train dataset
    dataset["train"] = dataset["train"].select(
        (
            i for i in range(len(dataset["train"])) 
            if i not in indices_to_be_removed
        )
    )
    return dataset

def list_labels(dataset, text_nr):
    for text, label in zip(dataset["train"][text_nr]["tokens"], dataset["train"][text_nr]["labels"]):
        if label > -100:
            label = id2label[label]
        print(f"{text}, {label}")
        
# Give all tokens the label "O"
def add_token_labels(text):
    for tokens in text["tokens"]:
        for token in tokens:
            token["label"] = label2id["O"]

    return text

# Give tokens that are part of a span the correct NER and chunking label
def add_span_ner_labels(texts):
    i = False
    for span_nr, spans in enumerate(texts["spans"]):
        for span in spans:
            start = span["token_start"]
            if start is None:
                print('a')
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
                            if label2id["B-" + label] == []:
                                print("B-" + label)
                            token["label"] = label2id["B-" + label]
                        else:
                            if label2id["I-" + label] == []:
                                print("I-" + label)
                            token["label"] = label2id["I-" + label]

    return texts

def align_labels(texts):
    all_tokens = []
    all_labels = []
    all_ids = []
    
    for text_nr, tokens in enumerate(texts["tokens"]):
        data = {}
        data['tokens'] = []
        data['labels'] = []
        
        for token in tokens:
            data['labels'].append(token["label"])
            data['tokens'].append(token["text"])
        all_tokens.append(data['tokens'])
        all_labels.append(data['labels'])
        all_ids.append([text_nr])

    return {"tokens": all_tokens, "labels": all_labels, "id": all_ids, "old_tokens": texts["tokens"]}
    
def tokenize_and_align_labels(texts, label_all_tokens=True): 
    tokenized_inputs = tokenizer(texts["tokens"], truncation=True, is_split_into_words=True, padding="max_length")     

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
        'input_ids': tokenized_inputs["input_ids"], 
        'token_type_ids': tokenized_inputs["token_type_ids"], 
        'attention_mask': tokenized_inputs["attention_mask"]
    }   

def check_tokens(dataset):
    for token, label in zip(tokenizer.convert_ids_to_tokens(dataset["train"][4]["input_ids"]),dataset["train"][4]["labels"]): 
        if label > -100:
            label = id2label[label]
        print(f"{token:_<40} {label}") 
    
def compute_metrics(eval_preds): 
    pred_logits, labels = eval_preds 
    
    pred_logits = np.argmax(pred_logits, axis=2) 
    # the logits and the probabilities are in the same order,
    # so we don’t need to apply the softmax
    
    # We remove all the values where the label is -100
    predictions = [ 
        [label_list[eval_preds] for (eval_preds, l) in zip(prediction, label) if l != -100] 
        for prediction, label in zip(pred_logits, labels) 
    ] 
    
    true_labels = [ 
      [label_list[l] for (eval_preds, l) in zip(prediction, label) if l != -100] 
       for prediction, label in zip(pred_logits, labels) 
    ] 
    
    for pred, true in zip(predictions, true_labels):
        print(f"Prediction: {pred}, true label: {true}")
        
    results = metric.compute(predictions=predictions, references=true_labels) 
    return { 
       "precision": results["overall_precision"], 
       "recall": results["overall_recall"], 
       "f1": results["overall_f1"], 
       "accuracy": results["overall_accuracy"], 
    }     

def clean_and_split_dataset(dataset):
    # Only take relevant columns
    ner_dataset = dataset.remove_columns(['tokens','text','_input_hash', '_task_hash', '_is_binary', 'spans', '_view_id', 'relations', 'answer','_timestamp', 'old_tokens'])

    # create new train dataset
    train_set = ner_dataset["train"].select(
        (
            i for i in range(len(ner_dataset["train"])) 
            if i not in [6,7,8]
        )
    )

    # create new eval dataset
    eval_set = ner_dataset["train"].select(
        (
            i for i in range(len(ner_dataset["train"])) 
            if i not in [0,1,2,3,4,5]
        )
    )
    return train_set, eval_set

def count_labels(dataset):
    label_counts = {
        'O': 0,
        'B-CLASS': 0,
        'I-CLASS': 0,
        'B-ATTRIBUTE': 0,
        'I-ATTRIBUTE': 0,
        'B-ASSOCIATION': 0,
        'I-ASSOCIATION': 0,
        'B-SYSTEM': 0,
        'I-SYSTEM': 0,
        'B-OPERATION': 0,
        'I-OPERATION': 0
    }
    for labels in dataset["train"]["labels"]: 
        for label in labels:
            if label != -100:
                label_counts[id2label[label]] += 1
    print(label_counts)
    
def train_ner_model(debug=False, count_labels_in_text=False):
    ner_model = NER_Model()
    
    # Remove rejected texts
    ner_model.dataset = remove_rejected_texts(ner_model.dataset)
    
    # Add token labels
    ner_model.dataset = ner_model.dataset.map(add_token_labels, batched=True)
    ner_model.dataset = ner_model.dataset.map(add_span_ner_labels, batched=True)
    
    # Tokenize and align labels
    ner_model.dataset = ner_model.dataset.map(align_labels, batched=True)
    ner_model.dataset = ner_model.dataset.map(tokenize_and_align_labels, batched=True)
    
    if debug:
        # Check labels are properly aligned 
        check_tokens(ner_model.dataset)
    
    if count_labels_in_text:
        count_labels(ner_model.dataset)
    
    train_set, eval_set = clean_and_split_dataset(ner_model.dataset)
    
    trainer = Trainer( 
        ner_model.model, 
        ner_model.args, 
        train_dataset=train_set, 
        eval_dataset=eval_set, 
        data_collator=ner_model.data_collator, 
        tokenizer=tokenizer, 
        compute_metrics=compute_metrics 
    ) 
    
    trainer.train()
    
if __name__ == "__main__":
    train_ner_model(count_labels_in_text=False)
    
    
    
    
    
