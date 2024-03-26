import streamlit as st
import pickle

model_path = "models/v2/random_forest-v2.pkl"
all_symptoms = ['itching', 'skin_rash', 'nodal_skin_eruptions', 'continuous_sneezing',
       'shivering', 'chills', 'joint_pain', 'stomach_pain', 'acidity',
       'ulcers_on_tongue', 'muscle_wasting', 'vomiting', 'burning_micturition',
       'spotting_ urination', 'fatigue', 'weight_gain', 'anxiety',
       'cold_hands_and_feets', 'mood_swings', 'weight_loss', 'restlessness',
       'lethargy', 'patches_in_throat', 'irregular_sugar_level', 'cough',
       'high_fever', 'sunken_eyes', 'breathlessness', 'sweating',
       'dehydration', 'indigestion', 'headache', 'yellowish_skin',
       'dark_urine', 'nausea', 'loss_of_appetite', 'pain_behind_the_eyes',
       'back_pain', 'constipation', 'abdominal_pain', 'diarrhoea',
       'mild_fever', 'yellow_urine', 'yellowing_of_eyes',
       'acute_liver_failure', 'fluid_overload', 'swelling_of_stomach',
       'swelled_lymph_nodes', 'malaise', 'blurred_and_distorted_vision',
       'phlegm', 'throat_irritation', 'redness_of_eyes', 'sinus_pressure',
       'runny_nose', 'congestion', 'chest_pain', 'weakness_in_limbs',
       'fast_heart_rate', 'pain_during_bowel_movements', 'pain_in_anal_region',
       'bloody_stool', 'irritation_in_anus', 'neck_pain', 'dizziness',
       'cramps', 'bruising', 'obesity', 'swollen_legs',
       'swollen_blood_vessels', 'puffy_face_and_eyes', 'enlarged_thyroid',
       'brittle_nails', 'swollen_extremeties', 'excessive_hunger',
       'extra_marital_contacts', 'drying_and_tingling_lips', 'slurred_speech',
       'knee_pain', 'hip_joint_pain', 'muscle_weakness', 'stiff_neck',
       'swelling_joints', 'movement_stiffness', 'spinning_movements',
       'loss_of_balance', 'unsteadiness', 'weakness_of_one_body_side',
       'loss_of_smell', 'bladder_discomfort', 'foul_smell_of urine',
       'continuous_feel_of_urine', 'passage_of_gases', 'internal_itching',
       'toxic_look_(typhos)', 'depression', 'irritability', 'muscle_pain',
       'altered_sensorium', 'red_spots_over_body', 'belly_pain',
       'abnormal_menstruation', 'dischromic _patches', 'watering_from_eyes',
       'increased_appetite', 'polyuria', 'family_history', 'mucoid_sputum',
       'rusty_sputum', 'lack_of_concentration', 'visual_disturbances',
       'receiving_blood_transfusion', 'receiving_unsterile_injections', 'coma',
       'stomach_bleeding', 'distention_of_abdomen',
       'history_of_alcohol_consumption', 'fluid_overload.1', 'blood_in_sputum',
       'prominent_veins_on_calf', 'palpitations', 'painful_walking',
       'pus_filled_pimples', 'blackheads', 'scurring', 'skin_peeling',
       'silver_like_dusting', 'small_dents_in_nails', 'inflammatory_nails',
       'blister', 'red_sore_around_nose', 'yellow_crust_ooze']
def load_model(path):
    with open(path, 'rb') as f:
        model = pickle.load(f)
    return model

def parse_symptoms(symptoms):
    sym = []
    for s in all_symptoms:
        if s in symptoms:
            sym.append(1)
        else:
            sym.append(0)
    return sym

class DiseaseDetectionApp:
    def __init__(self,symptoms):
        self.symptoms = symptoms
        self.selected_symptoms = []
        self.model = load_model(model_path)

    def add_symptom(self, symptom):
        if symptom not in self.selected_symptoms:
            self.selected_symptoms.append(symptom)

    def remove_symptom(self):
        self.selected_symptoms.clear()

    def detect_disease(self):
        if len(self.selected_symptoms) < 3:
            st.error("Please select at least 3 symptoms.")
        else:
            # Placeholder for disease detection logic
            symps = parse_symptoms(symptoms=self.selected_symptoms)
            dis = self.model.predict([symps])
            st.success(f"{dis[0]} detected based on selected symptoms.")

def main():
    app = DiseaseDetectionApp(all_symptoms)
    model = load_model(model_path)

    st.title("Disease Detection App")

    selected_symptoms = st.multiselect("Select Symptoms:", options=app.symptoms, default=app.selected_symptoms)
    if selected_symptoms:
        app.selected_symptoms = selected_symptoms

    if st.button("Detect Disease"):
        app.detect_disease()

    # if len(app.selected_symptoms) > 0:
    #     if st.button("Remove Selected Symptom"):
    #         app.remove_symptom()

if __name__ == "__main__":
    main()
