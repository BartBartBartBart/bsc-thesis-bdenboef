# Main file for training NER Bert

from Model import NER_Model
from NER_Trainer import NER_Trainer
from Utils import (
    remove_rejected_texts,
    add_token_labels,
    add_span_ner_labels,
    align_labels,
    split_texts,
    tokenize_and_align_labels,
    check_tokens,
    clean_and_split_dataset,
    compute_metrics,
    list_labels,
)


def train_ner_model(
    debug=False,
    count_labels_in_text=False,
    learning_rate=9e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=40,
    weight_decay=0.01,
    warmup_steps=10,
    load_best_model_at_end=True,
    logging_steps=10,
):
    ner_model = NER_Model(
        learning_rate,
        per_device_train_batch_size,
        per_device_eval_batch_size,
        num_train_epochs,
        weight_decay,
        warmup_steps,
        load_best_model_at_end,
        logging_steps,
    )

    # Remove rejected texts
    ner_model.dataset = remove_rejected_texts(ner_model.dataset)

    # Add token labels
    ner_model.dataset = ner_model.dataset.map(add_token_labels, batched=True)
    ner_model.dataset = ner_model.dataset.map(add_span_ner_labels, batched=True)

    # Align labels
    ner_model.dataset = ner_model.dataset.map(align_labels, batched=True)

    # Remove some columns
    ner_model.dataset = ner_model.dataset.remove_columns(
        [
            "id",
            "text",
            "_input_hash",
            "_task_hash",
            "_is_binary",
            "spans",
            "_view_id",
            "relations",
            "answer",
            "_timestamp",
            "old_tokens",
        ]
    )

    # Split texts into batches of 400
    ner_model.dataset = ner_model.dataset.map(split_texts, batched=True)

    # Tokenize and align labels
    ner_model.dataset = ner_model.dataset.map(tokenize_and_align_labels, batched=True)

    # Remove some columns
    ner_model.dataset = ner_model.dataset.remove_columns(["tokens"])

    if debug:
        # Check labels are properly aligned
        check_tokens(ner_model.dataset)

    # if count_labels_in_text:
    #     label_counts = count_labels(ner_model.dataset)
    #     print(label_counts)

    train_set, eval_set = clean_and_split_dataset(ner_model.dataset)

    trainer = NER_Trainer(
        model=ner_model.model,
        args=ner_model.args,
        train_dataset=train_set,
        eval_dataset=eval_set,
        data_collator=ner_model.data_collator,
        tokenizer=ner_model.tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()


if __name__ == "__main__":
    # with wandb.init(project='ner-bert'):

    #     config = wandb.config

    #     train_ner_model(
    #         debug=False,
    #         count_labels_in_text=False,
    #         learning_rate=config.learning_rate,
    #         per_device_train_batch_size=config.per_device_train_batch_size,
    #         per_device_eval_batch_size=config.per_device_eval_batch_size,
    #         num_train_epochs=config.num_train_epochs,
    #         weight_decay=config.weight_decay,
    #         warmup_steps=config.warmup_steps,
    #         load_best_model_at_end=True,
    #         logging_steps=10
    #     )

    # may need to return a metric for logging purposes
    # wandb.log({})

    # Normal Run
    #
    train_ner_model(
        debug=False,
        count_labels_in_text=False,
        learning_rate=9e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=40,
        weight_decay=0.01,
        warmup_steps=10,
        load_best_model_at_end=True,
        logging_steps=10,
    )
