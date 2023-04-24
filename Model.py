# Contains the NER model
from datasets import load_dataset
from transformers import (
    TrainingArguments,
    BertTokenizerFast,
    AutoModelForTokenClassification,
    DataCollatorWithPadding,
)
import evaluate


class NER_Model:
    def __init__(
        self,
        learning_rate=9e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=40,
        weight_decay=0.01,
        warmup_steps=10,
        load_best_model_at_end=True,
        logging_steps=10,
    ):
        self.tokenizer = load_tokenizer()
        self.dataset = load_data()

        # Load bert base model
        self.model = load_model()

        # Load data collater
        self.data_collator = load_data_collater(self.tokenizer)

        self.metric = load_seqeval()

        self.args = TrainingArguments(
            "test-ner",
            save_strategy="epoch",
            evaluation_strategy="epoch",
            learning_rate=learning_rate,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            num_train_epochs=num_train_epochs,
            weight_decay=weight_decay,
            warmup_steps=warmup_steps,
            load_best_model_at_end=load_best_model_at_end,
            logging_steps=logging_steps,
        )


def load_data():
    return load_dataset("json", data_files="./dataset/small_test_set.json")


def load_tokenizer():
    return BertTokenizerFast.from_pretrained("bert-base-uncased")


def load_model():
    return AutoModelForTokenClassification.from_pretrained("bert-base-uncased", num_labels=11)


def load_data_collater(tokenizer):
    return DataCollatorWithPadding(tokenizer, padding=True)


def load_seqeval():
    return evaluate.load("seqeval")
