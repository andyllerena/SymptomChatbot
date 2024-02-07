from flask import Flask, render_template, request, jsonify
import json
import spacy
import random

app = Flask(__name__)

with open('patterns.json', 'r') as file:
    patterns = json.load(file)

nlp = spacy.load("en_core_web_md")

def generate_dynamic_response(intent, user_context):
    responses = patterns[intent]["responses"]
    if intent == "goodbye":
        return "Goodbye! Have a great day."
    else:
        return random.choice(responses)

user_context = {}

def update_user_context(user_input, intents):
    user_context['previous_input'] = user_input
    user_context['previous_intents'] = intents

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_user_input', methods=['POST'])
def process_user_input():
    user_message = request.json['user_message']

    best_match_intent = None
    best_similarity = 0.0

    for intent, details in patterns.items():
        phrases = details["phrases"]
        for phrase in phrases:
            phrase_doc = nlp(phrase.lower())
            similarity = nlp(user_message.lower()).similarity(phrase_doc)
            if similarity > best_similarity:
                best_similarity = similarity
                best_match_intent = intent

    if best_match_intent and best_similarity > 0.60:
        response = generate_dynamic_response(best_match_intent, user_context)
        update_user_context(user_message, [best_match_intent])
    else:
        response = "Bot: I'm sorry, I didn't understand that. Can you please provide more information?"

    return jsonify({"bot_response": response})

if __name__ == '__main__':
    app.run(debug=True)
