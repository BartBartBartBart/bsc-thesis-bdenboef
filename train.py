# Main file for training NER Bert
# Can be run with best parameters using run_model.sh

import wandb
import argparse
import time
import numpy as np
from sklearn.metrics import classification_report

from model import NER_Model
from NER_Trainer import NER_Trainer
from data_generator import data_generator
from rel_extractor import rel_extractor
from utils import cv_split, compute_metrics, set_seeds, calculate_score, parse_precomputed_preds

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
    dest="weight_decay",
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

    generator = data_generator()
    dataset = generator.generate_dataset()

    ner_scores = []
    ner_reports = []
    re_scores = []
    avg_f1 = 0

    # precomputed_preds = parse_precomputed_preds(PREDICTIONS)

    for i in [0, 1, 2, 3]:
        print(f"Fold {i}")

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
        # NAMED ENTITY RECOGNITION

        # train model on train set and evaluate with eval
        trainer.train()

        # evaluate predictions on test set
        ner_predictions, label_ids, ner_metrics = trainer.predict(test_dataset=eval_set)

        # Transform probabilities into predictions
        ner_predictions = np.argmax(ner_predictions, axis=2)

        # We remove all the values where the label is -100
        predictions = [
            [eval_preds for (eval_preds, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(ner_predictions, label_ids)
        ]
        true_labels = [
            [l for (eval_preds, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(ner_predictions, label_ids)
        ]

        pred = []
        truth = []

        for labels, preds in zip(true_labels, predictions):
            pred += preds
            truth += labels

        results_per_class = classification_report(truth, pred, labels=np.unique(truth))
        ner_reports.append(results_per_class)

        ner_scores.append(ner_metrics)
        avg_f1 += ner_metrics["test_f1"]

        # # RELATION EXTRACTION
        re = rel_extractor()

        # Apply heuristics
        predicted_relations = re.extract_relations(eval_set["input_ids"], ner_predictions)

        # Calculate metrics
        scores_per_class = calculate_score(eval_set["relations"], predicted_relations)
        re_scores.append(scores_per_class)

    return ner_scores, ner_reports, re_scores, avg_f1 / 4


if __name__ == "__main__":
    # with wandb.init(project="ner-bert"):

    # #     # For hyperparameter optimization with wandb
    #     config = wandb.config

    #     train_ner_model(
    #         learning_rate=config.learning_rate,
    #         per_device_train_batch_size=config.per_device_train_batch_size,
    #         per_device_eval_batch_size=config.per_device_eval_batch_size,
    #         num_train_epochs=config.num_train_epochs,
    #         weight_decay=config.weight_decay,
    #         warmup_steps=config.warmup_steps,
    #         load_best_model_at_end=True,
    #         logging_steps=10
    #     )

    # Normal Run
    args = parser.parse_args()

    start = time.time()

    ner_scores, ner_reports, re_scores, avg_f1 = train_ner_model(
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        load_best_model_at_end=True,
        logging_steps=args.logging_steps,
    )

    end = time.time()

    print(f"Average F1 score: {avg_f1}")
    print(f"The run took {end-start} seconds.")

    print("NER results per fold:")
    print(ner_scores)
    for report in ner_reports:
        print(report)

    print("RE results per fold:")
    for re_score in re_scores:
        print(re_score)
