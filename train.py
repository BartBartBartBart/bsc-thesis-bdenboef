# Main file for training NER Bert

import wandb
import argparse

from model import NER_Model
from NER_Trainer import NER_Trainer
from data_generator import data_generator
from rel_extractor import rel_extractor
from utils import cv_split, compute_metrics, set_seeds

# Parse arguments from command line
parser = argparse.ArgumentParser(description="Add hyperparameters.")
parser.add_argument(
    "--learning-rate", dest="learning_rate", default=9e-5, type=float, help="Learning rate for the model"
)
parser.add_argument(
    "--epochs", dest="epochs", default=40, type=int, help="Number of epochs the model will run"
)
parser.add_argument(
    "--train-batch-size",
    dest="per_device_train_batch_size",
    default=16,
    type=int,
    help="Number of batches in the train set",
)
parser.add_argument(
    "--eval-batch-size",
    dest="per_device_eval_batch_size",
    default=16,
    type=int,
    help="Number of batches in the eval set",
)
parser.add_argument(
    "--weight-decay",
    dest="train_batch_size",
    default=0.01,
    type=float,
    help="Number of batches in the train set",
)
parser.add_argument(
    "--warmup-steps", dest="warmup_steps", default=10, type=int, help="Number of warmup steps"
)
parser.add_argument(
    "--logging-steps", dest="logging_steps", default=10, type=int, help="Number of logging steps"
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
    set_seeds(seed=42)

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
    dataset = generator.generate_ner_dataset()
    # dataset = generator.save_dataset(dataset, 'tokenized_dataset')
    # dataset = generator.load_tokenized_dataset('tokenized_dataset/')

    for i in [0]:
        print(f"Fold {i}")
        # Cross-validation split
        train_set, eval_set = cv_split(dataset=dataset, fold=i)

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
    # with wandb.init(project="ner-bert"):

    # For hyperparameter optimization with wandb
    # config = wandb.config

    # train_ner_model(
    #     learning_rate=config.learning_rate,
    #     per_device_train_batch_size=config.per_device_train_batch_size,
    #     per_device_eval_batch_size=config.per_device_eval_batch_size,
    #     num_train_epochs=config.num_train_epochs,
    #     weight_decay=config.weight_decay,
    #     warmup_steps=config.warmup_steps,
    #     load_best_model_at_end=True,
    #     logging_steps=10
    # )

    # Normal Run

    # args = parser.parse_args()

    # train_ner_model(
    #     learning_rate=args.learning_rate,
    #     per_device_train_batch_size=args.per_device_train_batch_size,
    #     per_device_eval_batch_size=args.per_device_eval_batch_size,
    #     num_train_epochs=args.epochs,
    #     weight_decay=args.epochs,
    #     warmup_steps=args.warmup_steps,
    #     load_best_model_at_end=True,
    #     logging_steps=args.logging_steps,
    # )

    generator = data_generator()
    dataset = generator.generate_re_dataset()
    re = rel_extractor()
    train_set, eval_set = cv_split(dataset=dataset, fold=0)
    re.extract_relations(eval_set)
