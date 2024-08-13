# CV Screening Using NLP with SpaCy

![spaCy](https://img.shields.io/badge/spaCy-v3.4.0-brightgreen) ![Python](https://img.shields.io/badge/Python-3.x-blue)

## Table of Contents

- [CV Screening Using NLP with SpaCy](#cv-screening-using-nlp-with-spacy)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Objective](#objective)
  - [Prerequisites](#prerequisites)
  - [Methodology](#methodology)
    - [Data Collection](#data-collection)
    - [Data Annotation](#data-annotation)
    - [Model Training](#model-training)
  - [Model Testing and Validation](#model-testing-and-validation)
  - [Challenges](#challenges)
  - [Future Work](#future-work)
  - [Conclusion](#conclusion)

## Introduction

This project automates the screening of CVs using Natural Language Processing (NLP) to extract key information such as names, contact details, education, and work experience. The project uses [spaCy](https://spacy.io/), a popular Python library for NLP tasks.

## Objective

The main objective is to create a CV screening tool that can automatically extract and classify various entities from resumes, including:

- **Name**
- **Designation**
- **Email Address**
- **Degree**
- **College Name**
- **Skills**
- **Companies Worked At**
- **Work Experience**

## Prerequisites

Make sure you have the following installed:

- **Python 3.x**
- **spaCy**
- **A spaCy model (e.g., `en_core_web_sm`)**

You can install the required packages with the following commands:

```bash
pip install spacy
python -m spacy download en_core_web_sm
```

## Methodology

### Data Collection

Gathered over 200 CVs from various sources to create a diverse training dataset. These CVs were used to train the NLP model.

### Data Annotation

Annotated the CVs manually to label the entities of interest. Here is an example of how the data is structured:

```bash
training_data = [
    (
        "Arun Prabhu Lead Software Engineer Chennai, Tamil Nadu - Email me on Indeed: indeed.com/r/Arun-Prabhu/a8de325d95905b67 Total IT experience 8 Years...",
        {"entities": [(0, 12, "Name"), (13, 34, "Designation"), (50, 83, "Location"), (101, 141, "Email Address"), ...]}
    ),
    ...
]
```

### Model Training

Trained the spaCy NLP model with the annotated data. The code below illustrates the basic steps for training:

```
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
```

## Model Testing and Validation

The trained model was tested on unseen CVs to evaluate its performance:

```
def parse_cv():
    # Get the CV text from the request
    data = request.json
    if not data or 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400

    cv_text = data['text']

    # Process the CV text with spaCy
    doc = nlp_model(cv_text)

    # Collect entities and their labels into a list of dictionaries
    parsed_data = [{'text': ent.text, 'start': ent.start_char, 'end': ent.end_char, 'label': ent.label_} for ent in doc.ents]

    # Return the parsed data as a JSON response
    return jsonify({'entities': parsed_data})
```

**Result for data from training set:**

```
{
    "entities": [
        {
            "label": "Name",
            "text": "Arun Prabhu"
        },
        {
            "label": "Location",
            "text": "Chennai"
        },
        {
            "label": "Years of Experience",
            "text": "6 Years"
        },
        {
            "label": "Designation",
            "text": "Lead Software Engineer"
        },
        {
            "label": "Companies worked at",
            "text": "TCS"
        },
        {
            "label": "Designation",
            "text": "Senior Consultant"
        },
        {
            "label": "Degree",
            "text": "B.Tech in Information Technology"
        },
        {
            "label": "College Name",
            "text": "Anna University"
        }
    ]
}
```

**Result for a unknown data:**

```
{
    "entities": [
        {
            "label": "Designation",
            "text": "SENIOR ENGINEERING MANAGER"
        },
        {
            "label": "Degree",
            "text": "Master of Business Administration : Decision Sciences"
        }
    ]
}
```

## Challenges

- **Data Diversity**: The CVs varied significantly in format, making it challenging to train a generalized model.
- **Entity Overlap**: Multiple instances of certain entities, such as company names and designations, required careful handling by the model.

## Future Work

- **Expand the Dataset**: Increasing the dataset's size and diversity could improve the model's accuracy.
- **Enhance Entity Recognition**: Further refinement of the model to handle overlapping entities and diverse CV formats better.
- **Integration with ATS**: Integrating the model with an Applicant Tracking System (ATS) for full automation of the screening process.

## Conclusion

This project demonstrates the feasibility of using NLP, specifically spaCy, to automate the extraction of key information from CVs. The results are promising if the dataset is big, and future enhancements could lead to more robust CV screening systems.
