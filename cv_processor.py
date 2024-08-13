from flask import Flask, request, jsonify
import spacy

app = Flask(__name__)
nlp_model = spacy.load('nlp_model')
nlp = spacy.load('en_core_web_sm')  # Load spaCy model

@app.route('/parse-cv', methods=['POST'])
def parse_cv():
    # Get the CV text from the request
    data = request.json
    if not data or 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400

    cv_text = data['text']

    # Process the CV text with spaCy
    doc = nlp_model(cv_text)

    # Collect entities and their labels into a list of dictionaries
    parsed_data = [{'text': ent.text, 'label': ent.label_} for ent in doc.ents]

    # Return the parsed data as a JSON response
    return jsonify({'entities': parsed_data})

if __name__ == '__main__':
    print("Starting Flask server on http://localhost:5000/")
    app.run(debug=True)