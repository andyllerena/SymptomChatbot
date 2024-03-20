import pandas as pd
import spacy
from spacy.matcher import PhraseMatcher
from joblib import load

def load_dataset(excel_path):
    df = pd.read_excel(excel_path)
    symptoms_list = df['Symptoms'].str.split(', ').explode().unique().tolist()
    symptoms_list.sort()  # Ensure the list is sorted for consistency
    return df, symptoms_list

nlp = spacy.load("en_core_web_lg")
matcher = PhraseMatcher(nlp.vocab, attr="LOWER")

def setup_phrase_matcher(symptoms_list):
    patterns = [nlp.make_doc(symptom.lower()) for symptom in symptoms_list]
    matcher.add("SYMPTOM", None, *patterns)

def extract_symptoms(text, symptoms_list):
    doc = nlp(text.lower())
    matches = matcher(doc)
    extracted_symptoms = set()
    for match_id, start, end in matches:
        matched_symptom = doc[start:end].text
        for symptom in symptoms_list:
            if symptom.lower() == matched_symptom:
                extracted_symptoms.add(symptom)
                break
    return extracted_symptoms

def symptoms_to_model_input(extracted_symptoms, symptoms_list, model):
    # Initialize a data frame with zeros for all features the model was trained on
    model_features = model.feature_names_in_
    model_input_df = pd.DataFrame(0, index=[0], columns=model_features)
    for symptom in extracted_symptoms:
        if symptom in model_features:
            model_input_df.at[0, symptom] = 1
    return model_input_df

def predict_disease(input_text, model, symptoms_list):
    extracted_symptoms = extract_symptoms(input_text, symptoms_list)
    model_input_df = symptoms_to_model_input(extracted_symptoms, symptoms_list, model)
    prediction = model.predict(model_input_df)
    return prediction[0]

if __name__ == "__main__":
    excel_path = 'C:\\Users\\aller\\Desktop\\chatbot3\\SymptomChatbot\\dataset.xlsx'
    model_path = 'C:\\Users\\aller\\Desktop\\chatbot3\\SymptomChatbot\\decision_tree_classifier.pkl'
    
    _, symptoms_list = load_dataset(excel_path)  # Only need the list of symptoms here
    setup_phrase_matcher(symptoms_list)
    
    clf_dt = load(model_path)  # Load the trained decision tree model
    
    user_input = "I have shortness of breath, nausea, vertigo, and vomiting."
    predicted_disease = predict_disease(user_input, clf_dt, symptoms_list)
    print(f"Based on your symptoms, you might have: {predicted_disease}")


