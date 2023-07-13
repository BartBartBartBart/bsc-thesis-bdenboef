# Contains a custom NER trainer, which implements weight normalization
# based on class occurrence in the loss function

from transformers import Trainer
from torch import nn
import torch
import numpy as np

from constants import ID2LABEL
from data_generator import data_generator


class NER_Trainer(Trainer):
    """
    Custom Trainer class that inherits from the original Trainer class.
    Can perform weighted CrossEntropyLoss to counter imbalanced classes.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.label_counts = count_labels()

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.
        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs = model(**inputs)

        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            logits = outputs.get("logits")
            class_weights = weight_normalization(n_classes=11, occurences_per_class=self.label_counts)
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100, weight=class_weights)
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss


def count_labels():
    """
    Counts the occurances of each class in the training data.
    This is used to perform weight normalization.
    """
    generator = data_generator()

    dataset = generator.generate_dataset()

    label_counts = {
        "O": 0,
        "B-CLASS": 0,
        "I-CLASS": 0,
        "B-ATTRIBUTE": 0,
        "I-ATTRIBUTE": 0,
        "B-ASSOCIATION": 0,
        "I-ASSOCIATION": 0,
        "B-SYSTEM": 0,
        "I-SYSTEM": 0,
        "B-OPERATION": 0,
        "I-OPERATION": 0,
    }
    for labels in dataset["train"]["labels"]:
        for label in labels:
            if label != -100:
                label_counts[ID2LABEL[label]] += 1

    return list(label_counts.values())


def weight_normalization(n_classes, occurences_per_class):
    """
    Calculates weights per class based on the occurances.
    Returns a tensor of these weights.
    """
    weights_per_class = 1.0 / np.array(np.power(occurences_per_class, 1))
    weights_per_class = weights_per_class / np.sum(weights_per_class) * n_classes
    return torch.tensor(weights_per_class)
