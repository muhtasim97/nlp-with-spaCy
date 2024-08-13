
# ===========================================================
import spacy
import pickle
import json
import random
from spacy.training import Example


# with open("train_data.json", 'r') as json_file:
#         data = json.load(json_file)
    
#     # Save the data to a pickle file
# with open("wrong_train_data.pkl", 'wb') as pickle_file:
#         pickle.dump(data, pickle_file)


train_data = pickle.load(open('right_train_data.pkl', 'rb'))
nlp = spacy.blank('en')

def train_model(train_data):
    if 'ner' not in nlp.pipe_names:
        nlp.add_pipe('ner', last=True)
    ner = nlp.get_pipe('ner')

    # Add labels to the NER pipe
    for _, annotation in train_data:
        for ent in annotation['entities']:
            ner.add_label(ent[2])

    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
    with nlp.disable_pipes(*other_pipes):  # only train NER
        optimizer = nlp.begin_training()
        for itn in range(10):
            print(f"Starting iteration {itn}")
            random.shuffle(train_data)
            losses = {}
            for text, annotations in train_data:
                try:
                    # Check for overlapping entities and skip if found
                    entities = annotations['entities']
                    spans = [ent[:2] for ent in entities]
                    if len(spans) != len(set(spans)):
                        print(f"Skipping overlapping entities in text: {text}")
                        continue

                    # Convert to Example objects
                    examples = [Example.from_dict(nlp.make_doc(text), annotations)]
                    nlp.update(
                        examples,  # batch of Example objects
                        drop=0.2,  # dropout - make it harder to memorize data
                        sgd=optimizer,  # callable to update weights
                        losses=losses
                    )
                except Exception as e:
                    print(f"Error during training: {e}")
                
            print(losses)

train_model(train_data) 
nlp.to_disk('nlp_model')


# =====================================================
# import spacy
# import pickle
# import random
# from spacy.training import Example

# # Load the training data from the pickle file
# train_data = pickle.load(open('train_data.pkl', 'rb'))
# nlp = spacy.blank('en')

# def has_overlaps(entities):
#     """Check if there are overlapping entities."""
#     spans = [(start, end) for start, end, label in entities]
#     for i in range(len(spans)):
#         for j in range(i + 1, len(spans)):
#             if spans[i][0] < spans[j][1] and spans[j][0] < spans[i][1]:
#                 return True
#     return False

# def filter_overlapping_entities(train_data):
#     """Filter out any training examples with overlapping entities."""
#     filtered_data = []
#     for text, annotations in train_data:
#         if not has_overlaps(annotations['entities']):
#             filtered_data.append((text, annotations))
#         else:
#             print(f"Skipping overlapping entities in text: {text}")
#     return filtered_data

# def train_model(train_data):
#     # Filter out overlapping entities
#     train_data = filter_overlapping_entities(train_data)

#     # Add the NER pipeline to the blank model
#     if 'ner' not in nlp.pipe_names:
#         nlp.add_pipe('ner', last=True)
#     ner = nlp.get_pipe('ner')

#     # Add labels to the NER pipe
#     for _, annotation in train_data:
#         for ent in annotation['entities']:
#             ner.add_label(ent[2])

#     other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
#     with nlp.disable_pipes(*other_pipes):  # only train NER
#         optimizer = nlp.begin_training()
#         for itn in range(10):
#             print(f"Starting iteration {itn}")
#             random.shuffle(train_data)
#             losses = {}
#             for text, annotations in train_data:
#                 try:
#                     # Convert to Example objects
#                     examples = [Example.from_dict(nlp.make_doc(text), annotations)]
#                     nlp.update(
#                         examples,  # batch of Example objects
#                         drop=0.2,  # dropout - make it harder to memorize data
#                         sgd=optimizer,  # callable to update weights
#                         losses=losses
#                     )
#                 except Exception as e:
#                     print(f"Error during training: {e}")
                
#             print(f"Iteration {itn} losses: {losses}")

#     # Save the trained model to disk
#     nlp.to_disk('nlp_model')
#     print("Model training complete and saved to 'nlp_model'.")

# # Train the model
# train_model(train_data)