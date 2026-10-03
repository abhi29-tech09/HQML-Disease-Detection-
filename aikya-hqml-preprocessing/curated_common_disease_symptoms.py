"""
Tier 1: curated common/lung-disease symptom knowledge base for QuantaDx.

Built from general clinical knowledge (typical presentation patterns), NOT
derived from a patient cohort or a single citable source. This is a starting
point for a hackathon prototype, not a validated clinical resource -- it
needs review by a qualified clinician before any real-world use, exactly as
flagged in the AIKYA symptom-extraction module's README.

Each disease maps to a set of canonical_name symptoms using the SAME naming
convention as the earlier AIKYA chat-extraction schema (schema.py /
extractSymptoms.ts), so this dataset and the live chat-derived symptom
vectors share one vocabulary end-to-end.
"""

# ---------------------------------------------------------------- Respiratory / lung
RESPIRATORY = {
    "asthma": {
        "wheezing", "shortness_of_breath", "chest_tightness", "cough",
        "dry_cough", "shortness_of_breath_on_exertion", "difficulty_breathing_at_night"
    },
    "copd": {
        "shortness_of_breath", "chronic_cough", "productive_cough", "wheezing",
        "chest_tightness", "fatigue", "shortness_of_breath_on_exertion", "cyanosis"
    },
    "pneumonia": {
        "fever", "chills", "productive_cough", "chest_pain", "shortness_of_breath",
        "fatigue", "sweating", "rapid_breathing", "confusion"
    },
    "acute_bronchitis": {
        "productive_cough", "dry_cough", "chest_discomfort", "fatigue",
        "mild_fever", "sore_throat", "wheezing"
    },
    "chronic_bronchitis": {
        "chronic_cough", "productive_cough", "wheezing", "shortness_of_breath",
        "chest_tightness", "fatigue"
    },
    "tuberculosis": {
        "chronic_cough", "hemoptysis", "night_sweats", "weight_loss", "fever",
        "fatigue", "chest_pain", "loss_of_appetite"
    },
    "lung_cancer": {
        "chronic_cough", "hemoptysis", "chest_pain", "shortness_of_breath",
        "weight_loss", "fatigue", "hoarseness", "loss_of_appetite", "wheezing"
    },
    "pulmonary_embolism": {
        "sudden_shortness_of_breath", "chest_pain", "rapid_heart_rate",
        "cough", "hemoptysis", "dizziness", "leg_swelling"
    },
    "pleural_effusion": {
        "shortness_of_breath", "chest_pain", "dry_cough", "shortness_of_breath_on_exertion"
    },
    "pneumothorax": {
        "sudden_chest_pain", "sudden_shortness_of_breath", "rapid_heart_rate", "cyanosis"
    },
    "bronchiectasis": {
        "chronic_cough", "productive_cough", "hemoptysis", "shortness_of_breath",
        "fatigue", "recurrent_infections"
    },
    "cystic_fibrosis": {
        "chronic_cough", "productive_cough", "recurrent_infections", "weight_loss",
        "fatigue", "shortness_of_breath", "salty_tasting_skin"
    },
    "sarcoidosis": {
        "dry_cough", "shortness_of_breath", "fatigue", "chest_pain",
        "swollen_lymph_nodes", "skin_redness", "joint_pain"
    },
    "ards": {
        "severe_shortness_of_breath", "rapid_breathing", "cyanosis", "confusion", "fatigue"
    },
    "obstructive_sleep_apnea": {
        "snoring", "daytime_sleepiness", "morning_headache", "sleep_disturbance",
        "fatigue", "difficulty_concentrating"
    },
    "sinusitis": {
        "nasal_congestion", "facial_pain", "headache", "runny_nose", "sore_throat",
        "loss_of_smell", "mild_fever"
    },
    "laryngitis": {
        "hoarseness", "sore_throat", "dry_cough", "difficulty_swallowing", "loss_of_voice"
    },
    "pharyngitis": {
        "sore_throat", "difficulty_swallowing", "swollen_lymph_nodes", "mild_fever", "cough"
    },
    "common_cold": {
        "runny_nose", "nasal_congestion", "sneezing", "sore_throat", "mild_cough",
        "mild_headache", "fatigue"
    },
    "influenza": {
        "fever", "chills", "body_aches", "fatigue", "dry_cough", "sore_throat",
        "headache", "nasal_congestion"
    },
    "covid_19": {
        "fever", "dry_cough", "fatigue", "loss_of_smell", "loss_of_taste",
        "shortness_of_breath", "sore_throat", "body_aches", "headache"
    },
    "pulmonary_fibrosis": {
        "dry_cough", "shortness_of_breath_on_exertion", "fatigue", "clubbing_of_fingers"
    },
    "pulmonary_hypertension": {
        "shortness_of_breath_on_exertion", "fatigue", "chest_pain", "dizziness",
        "leg_swelling", "palpitations"
    },
}

# ---------------------------------------------------------------- General / common
GENERAL = {
    "gerd": {
        "heartburn", "acid_reflux", "chest_discomfort", "sour_taste", "difficulty_swallowing",
        "chronic_cough", "hoarseness"
    },
    "migraine": {
        "severe_headache", "sensitivity_to_light", "sensitivity_to_sound", "nausea",
        "vomiting", "blurred_vision", "throbbing_head_pain"
    },
    "hypertension": {
        "headache", "dizziness", "blurred_vision", "chest_pain", "shortness_of_breath",
        "nosebleed"
    },
    "type_2_diabetes": {
        "excessive_thirst", "frequent_urination", "fatigue", "blurred_vision",
        "slow_healing_wounds", "unexplained_weight_loss", "frequent_hunger"
    },
    "coronary_artery_disease": {
        "chest_pain", "chest_tightness", "shortness_of_breath_on_exertion",
        "fatigue", "palpitations", "dizziness"
    },
    "heart_failure": {
        "shortness_of_breath", "leg_swelling", "fatigue", "rapid_heart_rate",
        "shortness_of_breath_on_exertion", "difficulty_breathing_at_night", "weight_gain"
    },
    "stroke": {
        "sudden_numbness", "weakness_one_side", "slurred_speech", "confusion",
        "sudden_severe_headache", "difficulty_walking", "blurred_vision"
    },
    "urinary_tract_infection": {
        "painful_urination", "frequent_urination", "urgency_to_urinate",
        "cloudy_urine", "flank_pain", "mild_fever"
    },
    "gastroenteritis": {
        "diarrhea", "vomiting", "abdominal_pain", "nausea", "mild_fever", "fatigue"
    },
    "appendicitis": {
        "abdominal_pain", "loss_of_appetite", "nausea", "vomiting", "mild_fever",
        "abdominal_tenderness"
    },
    "kidney_stones": {
        "flank_pain", "blood_in_urine", "painful_urination", "nausea", "vomiting"
    },
    "iron_deficiency_anemia": {
        "fatigue", "pale_skin", "shortness_of_breath_on_exertion", "dizziness",
        "rapid_heart_rate", "cold_hands_and_feet"
    },
    "hypothyroidism": {
        "fatigue", "cold_intolerance", "weight_gain", "dry_skin", "hair_loss",
        "low_mood", "constipation"
    },
    "hyperthyroidism": {
        "weight_loss", "rapid_heart_rate", "heat_intolerance", "sweating",
        "tremor", "anxiety_symptoms", "sleep_disturbance"
    },
    "eczema": {
        "itching", "skin_redness", "dry_skin", "skin_scaling", "rash"
    },
    "psoriasis": {
        "skin_scaling", "skin_redness", "itching", "joint_pain", "rash"
    },
    "allergic_rhinitis": {
        "sneezing", "runny_nose", "nasal_congestion", "itching", "watery_eyes"
    },
    "conjunctivitis": {
        "eye_redness", "eye_discharge", "itching", "watery_eyes", "sensitivity_to_light"
    },
    "gout": {
        "joint_pain", "joint_swelling", "joint_redness", "sudden_joint_pain"
    },
    "osteoarthritis": {
        "joint_pain", "joint_stiffness", "joint_swelling", "reduced_range_of_motion"
    },
    "dengue": {
        "high_fever", "severe_headache", "joint_pain", "muscle_pain", "rash",
        "fatigue", "nausea"
    },
    "malaria": {
        "fever", "chills", "sweating", "headache", "fatigue", "nausea", "muscle_pain"
    },
    "typhoid_fever": {
        "high_fever", "fatigue", "abdominal_pain", "loss_of_appetite",
        "headache", "constipation"
    },
    "chickenpox": {
        "rash", "itching", "mild_fever", "fatigue", "loss_of_appetite"
    },
    "measles": {
        "high_fever", "rash", "runny_nose", "cough", "eye_redness", "sensitivity_to_light"
    },
    "anxiety_disorder": {
        "anxiety_symptoms", "rapid_heart_rate", "sweating", "difficulty_concentrating",
        "sleep_disturbance", "muscle_tension"
    },
    "depression": {
        "low_mood", "fatigue", "sleep_disturbance", "loss_of_appetite",
        "difficulty_concentrating", "low_energy"
    },
}

DISEASE_SYMPTOMS = {**RESPIRATORY, **GENERAL}

if __name__ == "__main__":
    all_symptoms = sorted(set().union(*DISEASE_SYMPTOMS.values()))
    print(f"Diseases: {len(DISEASE_SYMPTOMS)}")
    print(f"Unique canonical symptoms: {len(all_symptoms)}")
