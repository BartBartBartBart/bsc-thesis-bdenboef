# rel_extractor

from data_generator import load_tokenizer
from constants import ID2LABEL, COREFERENCES


class rel_extractor:
    def __init__(self):
        self.relations = []

    def extract_relations(self, input_ids, ner_predictions):
        # Combine broken up entities
        all_entities, all_labels = self.combine_entities(input_ids, ner_predictions)

        for entities, labels in zip(all_entities, all_labels):
            relations_in_batch = []
            index = 0
            subtype = True
            classes_in_sentence = []
            first_class = None
            for entity, label in zip(entities, labels):
                # print(entity, label)
                # Apply heuristics
                if entity == "[PAD]":
                    continue
                # Rule for extracting association relations
                if label == "ASSOCIATION":
                    subtype = False
                    skip_first = False
                    # print(entity)
                    if index > 1:
                        i = index
                        if (
                            entities[i - 1] == "and" or entities[i - 1] == "or" or entities[i - 1] == ","
                        ) and labels[i - 2] != "ASSOCIATION":
                            skip_first = True
                        while i > 0 and entities[i - 1] != ".":
                            if labels[i - 1] == "CLASS":
                                if skip_first:
                                    i -= 1
                                    skip_first = False
                                    continue
                                # Check if last char is s for mulitplicity
                                if entities[i - 1][-1] == "s":
                                    # print([entities[i - 1], entity, "ASSOCIATION1..*"])
                                    relations_in_batch.append([entities[i - 1], entity, "ASSOCIATION1..*"])
                                    break
                                else:
                                    # print([entities[i - 1], entity, "ASSOCIATION1"])
                                    relations_in_batch.append([entities[i - 1], entity, "ASSOCIATION1"])
                                    break
                            i -= 1
                        i = index
                        while i < len(labels) - 1 and entities[i + 1] != ".":
                            if labels[i + 1] == "CLASS":
                                # Check if last char is s for multiplicity
                                if entities[i + 1][-1] == "s":
                                    relations_in_batch.append([entity, entities[i + 1], "ASSOCIATION1..*"])
                                    break
                                else:
                                    relations_in_batch.append([entity, entities[i + 1], "ASSOCIATION1"])
                                    break
                            i += 1

                # For subtypes
                if label == "CLASS":
                    # classes_in_sentence.append(entity)
                    subtype2 = True
                    count = 0
                    i = index
                    potential_rels = []
                    # Check if this is a listing of 3 or more classes
                    while subtype2 and i < len(labels):
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
                            subtype2 = False
                            potential_rels = []
                            count = 0

                    # If listing, add relations subtype
                    if subtype2 and potential_rels:
                        if index > 0:
                            i = index
                            while i > 0 and entities[i - 1] != ".":
                                if labels[i - 1] == "CLASS":
                                    if i > 2 and entities[i - 3] == "in":
                                        i -= 1
                                        continue
                                    for sub in potential_rels:
                                        relations_in_batch.append([sub, entities[i - 1], "SUBTYPE"])
                                    potential_rels = []
                                    break
                                i -= 1

                # If no association within this sentence, create subtype relations
                # if entity == ".":
                #     if subtype:
                #         for class_entity in classes_in_sentence:
                #             if class_entity != classes_in_sentence[0]:
                #                 relations_in_batch.append([class_entity, classes_in_sentence[0], "SUBTYPE"])
                #     else:
                #         classes_in_sentence = []
                #         subtype = True

                # Attribute
                if label == "ATTRIBUTE":
                    if index > 0:
                        i = index
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

                        i = index
                        # while (
                        #     i < len(labels) - 1
                        #     and entities[i + 1] != "."
                        #     and entities[i + 1] != "!"
                        #     and entities[i + 1] != "?"
                        # ):
                        #     if labels[i + 1] == "CLASS":
                        #         if i > 2 and entities[i + 3] == "in":
                        #             i += 1
                        #             continue
                        #         relations_in_batch.append([entity, entities[i + 1], "ATTRIBUTE"])
                        #         break
                        #     i += 1

                # Operation
                if label == "OPERATION":
                    if index > 0:
                        i = index
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

                        i = index
                        # while (
                        #     i < len(labels) - 1
                        #     and entities[i + 1] != "."
                        #     and entities[i + 1] != "!"
                        #     and entities[i + 1] != "?"
                        # ):
                        #     if labels[i + 1] == "CLASS":
                        #         if i > 2 and entities[i + 3] == "in":
                        #             i += 1
                        #             continue
                        #         relations_in_batch.append([entity, entities[i + 1], "OPERATION"])
                        #         break
                        #     i += 1

                # Coreference resolution
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

                if label == "COMPOSITION":
                    pass

                if label == "AGGREGATION":
                    pass

                    # Span, aggregation?
                    # Associations lopen nu nog over de hele tekst in principe, grens zetten bij zin
                    # idem voor attribute

                index += 1
            self.relations.append(relations_in_batch)
        return self.relations

    # Combines the split up entities into the same entity
    # Returns a list of entities with corresponding list of labels
    def combine_entities(self, input_ids, predictions):
        tokenizer = load_tokenizer()
        # input_ids = dataset["input_ids"]
        # predictions = ner_predictions
        all_entities = []
        all_labels = []

        for token, prediction in zip(input_ids, predictions):
            entities_in_batch = []
            labels_in_batch = []
            index = 0
            for tok, pred in zip(token, prediction):
                if pred != -100:
                    if ID2LABEL[pred] == "O":
                        entities_in_batch.append(tokenizer.convert_ids_to_tokens(tok))
                        labels_in_batch.append(ID2LABEL[pred])

                    if ID2LABEL[pred] == "B-CLASS" and tokenizer.convert_ids_to_tokens(tok)[0] != "#":
                        entity = self.find_full_entity("CLASS", index, tokenizer, prediction, tok, token)
                        entities_in_batch.append(entity)
                        labels_in_batch.append("CLASS")

                    if ID2LABEL[pred] == "B-ATTRIBUTE" and tokenizer.convert_ids_to_tokens(tok)[0] != "#":
                        entity = self.find_full_entity("ATTRIBUTE", index, tokenizer, prediction, tok, token)
                        entities_in_batch.append(entity)
                        labels_in_batch.append("ATTRIBUTE")

                    if ID2LABEL[pred] == "B-ASSOCIATION" and tokenizer.convert_ids_to_tokens(tok)[0] != "#":
                        entity = self.find_full_entity(
                            "ASSOCIATION", index, tokenizer, prediction, tok, token
                        )
                        entities_in_batch.append(entity)
                        labels_in_batch.append("ASSOCIATION")

                    if ID2LABEL[pred] == "B-SYSTEM" and tokenizer.convert_ids_to_tokens(tok)[0] != "#":
                        entity = self.find_full_entity("SYSTEM", index, tokenizer, prediction, tok, token)
                        entities_in_batch.append(entity)
                        labels_in_batch.append("SYSTEM")

                    if ID2LABEL[pred] == "B-OPERATION" and tokenizer.convert_ids_to_tokens(tok)[0] != "#":
                        entity = self.find_full_entity("OPERATION", index, tokenizer, prediction, tok, token)
                        entities_in_batch.append(entity)
                        labels_in_batch.append("OPERATION")

                    if ID2LABEL[pred] == "B-ENUMERATION" and tokenizer.convert_ids_to_tokens(tok)[0] != "#":
                        entity = self.find_full_entity(
                            "ENUMERATION", index, tokenizer, prediction, tok, token
                        )
                        entities_in_batch.append(entity)
                        labels_in_batch.append("ENUMERATION")

                    if ID2LABEL[pred] == "B-NOTE" and tokenizer.convert_ids_to_tokens(tok)[0] != "#":
                        entity = self.find_full_entity("NOTE", index, tokenizer, prediction, tok, token)
                        entities_in_batch.append(entity)
                        labels_in_batch.append("NOTE")

                index += 1
            all_entities.append(entities_in_batch)
            all_labels.append(labels_in_batch)

        return all_entities, all_labels

    # Looks ahead in the labels to combine them into one entity
    def find_full_entity(self, label, index, tokenizer, prediction, tok, token):
        entity = tokenizer.convert_ids_to_tokens(tok)
        # Edge case for 's associations
        if entity == "'":
            entity += tokenizer.convert_ids_to_tokens(token[index + 1])
            return entity
        i = index
        if i < len(prediction):
            while (
                i < len(prediction) - 1
                and prediction[i + 1] != -100
                and (
                    ID2LABEL[prediction[i + 1]] == f"I-{label}" or ID2LABEL[prediction[i + 1]] == f"B-{label}"
                )
            ):
                # print(ID2LABEL[prediction[i + 1]], tokenizer.convert_ids_to_tokens(token[i + 1]))
                if ID2LABEL[prediction[i + 1]] == f"B-{label}":
                    if tokenizer.convert_ids_to_tokens(token[i + 1])[0] != "#":
                        break
                entity += " " + tokenizer.convert_ids_to_tokens(token[i + 1])
                if i > len(prediction) - 1:
                    break
                i += 1
        # print("entity: ", entity)
        return entity
