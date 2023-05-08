# Contains the NER model
from torch import nn
from transformers import (
    AutoModelForTokenClassification,
    BertForSequenceClassification,
    BertModel,
    DataCollatorWithPadding,
    TrainingArguments,
)

from data_generator import load_tokenizer


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

        # Load bert base model
        self.model = load_ner_model()

        # Load tokenizer
        self.tokenizer = load_tokenizer()

        # Load data collater
        self.data_collator = load_data_collator(self.tokenizer)

        # Define training arguments
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


# class GBM_BERT(BertForSequenceClassification):
#     def __init__(self, config):
#         super().__init__(config)
#         self.num_labels = config.num_labels
#         self.config = config

#         self.bert = BertModel(config)
#         classifier_dropout = (
#             config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
#         )
#         self.dropout = nn.Dropout(classifier_dropout)
#         # self.classifier = nn.Linear(config.hidden_size, config.num_labels)
#         # gradient boosting

#         # Initialize weights and apply final processing
#         self.post_init()


class RE_Model:
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
        self.model = load_re_model()

        self.tokenizer = load_tokenizer()

        self.data_collator = load_data_collator()

        # Define training arguments
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


def load_ner_model():
    return AutoModelForTokenClassification.from_pretrained("bert-base-uncased", num_labels=11)


def load_re_model():
    # TODO
    pass


def load_data_collator(tokenizer):
    return DataCollatorWithPadding(tokenizer, padding=True)
