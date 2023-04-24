# Main file for training NER Bert

import wandb

from model import NER_Model
from NER_Trainer import NER_Trainer
from data_generator import data_generator
from utils import (
    clean_and_split_dataset,
    compute_metrics,
)


def train_ner_model(
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

    generator = data_generator()
    dataset = generator.generate_dataset()
    # dataset = generator.save_dataset(dataset, 'tokenized_dataset')
    # dataset = generator.load_tokenized_dataset('tokenized_dataset/')

    train_set, eval_set = clean_and_split_dataset(dataset)

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
    with wandb.init(project="ner-bert"):

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

    # TODO: set seed
