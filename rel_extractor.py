# Relation Extraction Rules
# In this file, raw tokens are combined into full entities
# based on the predicted NER labels.
# Then, the RE rules are applied

from data_generator import load_tokenizer
from constants import ID2LABEL, COREFERENCES


class rel_extractor:
    """
    Wrapper class for the rule-based RE method.
    Contains methods:
    - extract_relatiosn(input_ids, ner_predictions)
    - combine_entities(input_ids, predictions)
    - find_full_entity(label, index, tokenizer, prediction, tok, token)
    """

    def __init__(self):
        # Contains extracted relations
        self.relations = []

    def extract_relations(self, input_ids, ner_predictions):
        """
        Applies the RE rules on the combined entities.
        Returns the extracted relations.
        """

        # Combine broken up entities based on NER predictions
        all_entities, all_labels = self.combine_entities(input_ids, ner_predictions)

        for entities, labels in zip(all_entities, all_labels):
            relations_in_batch = []
            index = 0
            for entity, label in zip(entities, labels):
                # Apply heuristics

                # Do not use entities that are padding
                if entity == "[PAD]":
                    continue

                # Rule for extracting association relations
                if label == "ASSOCIATION":
                    skip_first = False

                    if index > 1:
                        i = index
                        if (
                            entities[i - 1] == "and" or entities[i - 1] == "or" or entities[i - 1] == ","
                        ) and labels[i - 2] != "ASSOCIATION":
                            skip_first = True

                        # Search for correct class on the left side
                        while i > 0 and entities[i - 1] != ".":
                            if labels[i - 1] == "CLASS":
                                if skip_first:
                                    i -= 1
                                    skip_first = False
                                    continue

                                # If "can be" is in front of association entity,
                                # classify as zero-or-more multiplicity
                                if entities[index - 2] == "can" and entities[index - 1] == "be":
                                    relations_in_batch.append([entities[i - 1], entity, "ASSOCIATION*"])
                                    break

                                # Check if last char is s for mulitplicity
                                elif entities[i - 1][-1] == "s":
                                    relations_in_batch.append([entities[i - 1], entity, "ASSOCIATION1..*"])
                                    break
                                else:
                                    relations_in_batch.append([entities[i - 1], entity, "ASSOCIATION1"])
                                    break
                            i -= 1

                        # Search for correct class on the right side
                        i = index
                        while i < len(labels) - 1 and entities[i + 1] != ".":
                            if labels[i + 1] == "CLASS":
                                # If the association entity is one of these
                                # words, it is likely zero-or-more
                                if (
                                    entity == "controls"
                                    or entity == "list"
                                    or entity == "score"
                                    or entity == "scores"
                                    or entity == "sends"
                                ):
                                    relations_in_batch.append([entity, entities[i + 1], "ASSOCIATION*"])
                                    break

                                # Check if last char is s for multiplicity
                                if entities[i + 1][-1] == "s":
                                    relations_in_batch.append([entity, entities[i + 1], "ASSOCIATION1..*"])
                                    break
                                else:
                                    relations_in_batch.append([entity, entities[i + 1], "ASSOCIATION1"])
                                    break
                            i += 1

                # Rule for extracting subtype and aggregation relations
                if label == "CLASS":
                    subtype = True
                    count = 0
                    i = index
                    potential_rels = []

                    # Check if this is a listing of 3 or more classes
                    while subtype and i < len(labels):
                        if labels[i + 1] == "CLASS":
                            potential_rels.append(entities[i + 1])
                            count += 1
                            i += 1
                        elif entities[i + 1] == "," or entities[i + 1] == "and":
                            i += 1
                            continue
                        elif count >= 3:
                            potential_rels.append(entity)
                            break
                        else:
                            subtype = False
                            potential_rels = []
                            count = 0

                    # If listing, add relations subtype or aggregation
                    # if the word in front of listing is "of"
                    if subtype and potential_rels:
                        aggregation = False
                        if index > 0:
                            i = index

                            # If the word before the listing is "of"
                            # classify as aggregation
                            if entities[index - 1] == "of":
                                aggregation = True

                            while i > 0 and entities[i - 1] != ".":
                                if labels[i - 1] == "CLASS":
                                    if i > 2 and entities[i - 3] == "in":
                                        i -= 1
                                        continue
                                    for ent in potential_rels:
                                        if aggregation:
                                            relations_in_batch.append([entities[i - 1], ent, "AGGREGATION"])
                                        else:
                                            relations_in_batch.append([ent, entities[i - 1], "SUBTYPE"])
                                    potential_rels = []
                                    break
                                i -= 1

                # RE rule for attribute relations
                if label == "ATTRIBUTE":
                    if index > 0:
                        i = index

                        # Search for correct class on the left side
                        while (
                            i > 0
                            and entities[i - 1] != "."
                            and entities[i - 1] != "!"
                            and entities[i - 1] != "?"
                        ):
                            if labels[i - 1] == "CLASS":
                                if i > 2 and entities[i - 3] == "in":
                                    i -= 1
                                    continue
                                relations_in_batch.append([entity, entities[i - 1], "ATTRIBUTE"])
                                break
                            i -= 1

                # RE rule for operation relations
                if label == "OPERATION":
                    if index > 0:
                        i = index

                        # Search for correct class on the left side
                        while (
                            i > 0
                            and entities[i - 1] != "."
                            and entities[i - 1] != "!"
                            and entities[i - 1] != "?"
                        ):
                            if labels[i - 1] == "CLASS":
                                if i > 2 and entities[i - 3] == "in":
                                    i -= 1
                                    continue
                                relations_in_batch.append([entities[i - 1], entity, "OPERATION"])
                                break
                            i -= 1

                # RE rule for coreference resolution
                if entity in COREFERENCES:
                    if index > 0:
                        i = index
                        first_class_found = False
                        while i > 0 and entities[i - 1] != ".":
                            if labels[i - 1] == "CLASS":
                                if not first_class_found:
                                    first_class_found = True
                                    i -= 1
                                    continue
                                if i > 2 and entities[i - 3] == "in":
                                    i -= 1
                                    continue
                                relations_in_batch.append([entity, entities[i - 1], "COREF"])
                                break
                            i -= 1

                if entity == "of" and labels[index - 1] == "CLASS" and labels[index + 1] == "CLASS":
                    relations_in_batch.append([entities[index - 1], entities[index + 1], "COMPOSITION"])

                index += 1
            self.relations.append(relations_in_batch)
        return self.relations

    def combine_entities(self, input_ids, predictions):
        """
        Turns list of raw subtokens with their predictions
        into full entities, using the BIO tagging scheme.
        Returns lists of entities and list of corresponding
        labels.
        """

        tokenizer = load_tokenizer()
        all_entities = []
        all_labels = []

        for token, prediction in zip(input_ids, predictions):
            entities_in_batch = []
            labels_in_batch = []
            index = 0

            # For each token, if it has a B tag,
            # search for the full entity and combine.
            for tok, pred in zip(token, prediction):
                if pred != -100 and tokenizer.convert_ids_to_tokens(tok)[0] != "#":
                    if ID2LABEL[pred] == "O":
                        entities_in_batch.append(tokenizer.convert_ids_to_tokens(tok))
                        labels_in_batch.append(ID2LABEL[pred])

                    if ID2LABEL[pred] == "B-CLASS":
                        entity = self.find_full_entity("CLASS", index, tokenizer, prediction, tok, token)
                        entities_in_batch.append(entity)
                        labels_in_batch.append("CLASS")

                    if ID2LABEL[pred] == "B-ATTRIBUTE":
                        entity = self.find_full_entity("ATTRIBUTE", index, tokenizer, prediction, tok, token)
                        entities_in_batch.append(entity)
                        labels_in_batch.append("ATTRIBUTE")

                    if ID2LABEL[pred] == "B-ASSOCIATION":
                        entity = self.find_full_entity(
                            "ASSOCIATION", index, tokenizer, prediction, tok, token
                        )
                        entities_in_batch.append(entity)
                        labels_in_batch.append("ASSOCIATION")

                    if ID2LABEL[pred] == "B-SYSTEM":
                        entity = self.find_full_entity("SYSTEM", index, tokenizer, prediction, tok, token)
                        entities_in_batch.append(entity)
                        labels_in_batch.append("SYSTEM")

                    if ID2LABEL[pred] == "B-OPERATION":
                        entity = self.find_full_entity("OPERATION", index, tokenizer, prediction, tok, token)
                        entities_in_batch.append(entity)
                        labels_in_batch.append("OPERATION")

                index += 1
            all_entities.append(entities_in_batch)
            all_labels.append(labels_in_batch)

        return all_entities, all_labels

    def find_full_entity(self, label, index, tokenizer, prediction, tok, token):
        """
        Look ahead in the NER predictions to combine
        tokens into entities, using the BIO tagging scheme.

        """
        entity = tokenizer.convert_ids_to_tokens(tok)

        # Edge case for 's associations
        if entity == "'":
            entity += tokenizer.convert_ids_to_tokens(token[index + 1])
            return entity

        # Check if next token is also part of the same entity
        i = index
        while (
            i < len(prediction) - 1
            and prediction[i + 1] != -100
            and (ID2LABEL[prediction[i + 1]] == f"I-{label}" or ID2LABEL[prediction[i + 1]] == f"B-{label}")
        ):
            # Sometimes a word is split into subtokens
            # In this case, the subtokens also start with a B tag
            # even though it is part of the same entity
            # Subtokens can be recognized, because they start with "#"
            if (
                ID2LABEL[prediction[i + 1]] == f"B-{label}"
                and tokenizer.convert_ids_to_tokens(token[i + 1])[0] != "#"
            ):
                break
            # Add token to entity
            entity += " " + tokenizer.convert_ids_to_tokens(token[i + 1])
            i += 1

        return entity
