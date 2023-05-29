# Contains the NER model
from torch import nn
from transformers import (
    AutoModelForTokenClassification,
    BertForSequenceClassification,
    BertModel,
    DataCollatorWithPadding,
    TrainingArguments,
)
import torch
import torch.nn.functional as F
from transformers import WEIGHTS_NAME, BertConfig, BertModel, BertPreTrainedModel, BertTokenizer
from torch.nn import MSELoss, CrossEntropyLoss

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


class BertForSequenceClassification(BertPreTrainedModel):
    r"""
        **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for computing the sequence classification/regression loss.
            Indices should be in ``[0, ..., config.num_labels - 1]``.
            If ``config.num_labels == 1`` a regression loss is computed (Mean-Square loss),
            If ``config.num_labels > 1`` a classification loss is computed (Cross-Entropy).

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Classification (or regression if config.num_labels==1) loss.
        **logits**: ``torch.FloatTensor`` of shape ``(batch_size, config.num_labels)``
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForSequenceClassification.from_pretrained(
            'bert-base-uncased')
        input_ids = torch.tensor(tokenizer.encode(
            "Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)
        loss, logits = outputs[:2]

    """

    def __init__(self, config):
        super(BertForSequenceClassification, self).__init__(config)
        self.num_labels = 10
        self.bert = BertModel(config)
        self.cls_dropout = nn.Dropout(0.1)  # dropout on CLS transformed token embedding
        self.ent_dropout = nn.Dropout(0.1)  # dropout on average entity embedding
        self.classifier = nn.Linear(config.hidden_size * 3, 10)
        self.init_weights()

    def forward(
        self,
        input_ids,
        token_type_ids=None,
        attention_mask=None,
        e1_mask=None,
        e2_mask=None,
        labels=None,
        position_ids=None,
        head_mask=None,
    ):
        outputs = self.bert(
            input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
        )
        # for details, see https://huggingface.co/transformers/model_doc/bert.html#bertmodel
        pooled_output = outputs[1]  # sequence of hidden-states at the output of the last layer of the model
        sequence_output = outputs[
            0
        ]  # last layer hidden-state of the first token of the sequence (classification token) further processed by a Linear layer and a Tanh activation function.

        def extract_entity(sequence_output, e_mask):
            extended_e_mask = e_mask.unsqueeze(1)
            extended_e_mask = torch.bmm(extended_e_mask.float(), sequence_output).squeeze(1)
            return extended_e_mask.float()

        e1_h = self.ent_dropout(extract_entity(sequence_output, e1_mask))
        e2_h = self.ent_dropout(extract_entity(sequence_output, e2_mask))

        # I extract entities with other model, how can I access them here?

        context = self.cls_dropout(pooled_output)
        pooled_output = torch.cat([context, e1_h, e2_h], dim=-1)

        logits = self.classifier(pooled_output)

        # add hidden states and attention if they are here
        outputs = (logits,) + outputs[2:]

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()

                # Maybe do weight normalization like with classes?

                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)


def load_ner_model():
    return AutoModelForTokenClassification.from_pretrained("bert-base-uncased", num_labels=11)


def load_data_collator(tokenizer):
    return DataCollatorWithPadding(tokenizer, padding=True)
