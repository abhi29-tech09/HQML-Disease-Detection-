"""
QuantaDx -- Disease urgency tiers for risk stratification
SIH 2026 PS3, Deliverable #4 (Prediction & Decision Support Module)

Each of the 50 diseases from Deliverable #1 is assigned an urgency tier
based on general medical knowledge of how quickly each condition typically
needs attention if suspected. This is a simplifying judgment call for a
hackathon prototype, NOT a clinical triage protocol -- a qualified
clinician should review and refine this mapping before any real use, same
caveat as every other medical-knowledge component in this pipeline.

Tiers (fastest to slowest needed response):
  emergency  -- seek immediate/emergency care
  urgent     -- contact a doctor within 24-48 hours
  routine    -- schedule a doctor's visit when convenient
  self_care  -- usually manageable at home; see a doctor if it worsens
"""

URGENCY_TIER = {
    # --- Respiratory ---
    "asthma": "routine",
    "copd": "urgent",
    "pneumonia": "urgent",
    "acute_bronchitis": "self_care",
    "chronic_bronchitis": "routine",
    "tuberculosis": "urgent",
    "lung_cancer": "urgent",
    "pulmonary_embolism": "emergency",
    "pleural_effusion": "urgent",
    "pneumothorax": "emergency",
    "bronchiectasis": "routine",
    "cystic_fibrosis": "urgent",
    "sarcoidosis": "routine",
    "ards": "emergency",
    "obstructive_sleep_apnea": "routine",
    "sinusitis": "self_care",
    "laryngitis": "self_care",
    "pharyngitis": "self_care",
    "common_cold": "self_care",
    "influenza": "self_care",
    "covid_19": "routine",
    "pulmonary_fibrosis": "urgent",
    "pulmonary_hypertension": "urgent",
    # --- General ---
    "gerd": "self_care",
    "migraine": "self_care",
    "hypertension": "routine",
    "type_2_diabetes": "routine",
    "coronary_artery_disease": "urgent",
    "heart_failure": "urgent",
    "stroke": "emergency",
    "urinary_tract_infection": "routine",
    "gastroenteritis": "self_care",
    "appendicitis": "emergency",
    "kidney_stones": "urgent",
    "iron_deficiency_anemia": "routine",
    "hypothyroidism": "routine",
    "hyperthyroidism": "routine",
    "eczema": "self_care",
    "psoriasis": "routine",
    "allergic_rhinitis": "self_care",
    "conjunctivitis": "self_care",
    "gout": "urgent",
    "osteoarthritis": "routine",
    "dengue": "urgent",
    "malaria": "urgent",
    "typhoid_fever": "urgent",
    "chickenpox": "self_care",
    "measles": "urgent",
    "anxiety_disorder": "routine",
    "depression": "routine",
}

RECOMMENDED_ACTION = {
    "emergency": (
        "These symptoms match patterns associated with a medical emergency. "
        "Seek immediate care -- call emergency services or go to the nearest "
        "emergency room rather than waiting."
    ),
    "urgent": (
        "These symptoms warrant prompt medical attention. "
        "Contact a doctor within the next 24-48 hours."
    ),
    "routine": (
        "Consider scheduling a doctor's visit to discuss these symptoms "
        "when convenient."
    ),
    "self_care": (
        "These symptoms are often manageable with self-care, but see a doctor "
        "if they worsen or do not improve within a few days."
    ),
}

# Sensitivity bias per tier for threshold tuning (risk_thresholds.py):
# emergency/urgent conditions should be flagged even at lower confidence
# (prioritize catching true positives -- sensitivity); self_care conditions
# should require higher confidence before flagging (prioritize avoiding
# unnecessary alarm -- specificity).
TIER_SENSITIVITY_TARGET = {
    "emergency": 0.90,
    "urgent": 0.75,
    "routine": 0.60,
    "self_care": 0.50,
}

if __name__ == "__main__":
    from collections import Counter
    print("Diseases mapped:", len(URGENCY_TIER))
    print(Counter(URGENCY_TIER.values()))
