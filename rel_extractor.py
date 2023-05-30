# rel_extractor

from data_generator import load_tokenizer
from constants import ID2LABEL


class rel_extractor:
    def __init__(self):
        self.relations = []

    def extract_relations(self, dataset):
        all_entities, all_labels = self.combine_entities(dataset)
        for entities, labels in zip(all_entities, all_labels):
            relations_in_batch = []
            index = 0
            for entity, label in zip(entities, labels):
                # print(entity, label)
                # Apply heuristics

                # Rule for extracting association relations
                if label == "ASSOCIATION":
                    # print(label)
                    if index > 0:
                        i = index
                        while i > 0:
                            if labels[i - 1] == "CLASS":
                                # Check if last char is s for mulitplicity
                                if entities[i - 1][-1] == "s":
                                    relations_in_batch.append((entities[i - 1], entity, "ASSOCIATION1..*"))
                                    break
                                else:
                                    relations_in_batch.append((entities[i - 1], entity, "ASSOCIATION1"))
                                    break
                            i -= 1
                        i = index
                        while i < len(labels):
                            if labels[i + 1] == "CLASS":
                                # Check if last char is s for mulitplicity
                                if entities[i + 1][-1] == "s":
                                    relations_in_batch.append((entity, entities[i + 1], "ASSOCIATION1..*"))
                                    break
                                else:
                                    relations_in_batch.append((entity, entities[i + 1], "ASSOCIATION1"))
                                    break
                            i += 1

                    # Subtype relation
                    if label == "SUBTYPE":
                        pass

                    # Attribute
                    if label == "ATTRIBUTE":
                        pass

                    # Operation
                    if label == "OPERATION":
                        pass

                    if label == "Composition":
                        pass

                    # Span, coref, aggregation?

                index += 1
            self.relations.append(relations_in_batch)
        print(self.relations)
        return self.relations

    # Combines the split up entities into the same entity
    # Returns a list of entities with corresponding list of labels
    def combine_entities(self, dataset):
        tokenizer = load_tokenizer()
        input_ids = dataset["input_ids"]
        predictions = dataset["labels"]
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
            while prediction[i + 1] != -100 and (
                ID2LABEL[prediction[i + 1]] == f"I-{label}" or ID2LABEL[prediction[i + 1]] == f"B-{label}"
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
