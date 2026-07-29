"""Neutral clinical configuration for dataset label extraction.

Extracted from the CheXpert adapter to allow MIMIC-CXR and other
CheXpert-label-compatible datasets to share the exact same
clinical stratification without direct coupling.
"""

# The 14 CheXpert observation columns, in release order.
FINDING_COLUMNS = (
    "No Finding",
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
)

# Hierarchy of clinical acuity for prioritizing primary finding in multi-label cases.
# When a patient has multiple positive findings, the label string orders them from most
# clinically acute (pneumothorax, pleural effusion) to least specific (fracture).
CLINICAL_HIERARCHY = (
    "Pneumothorax",
    "Pleural Effusion",
    "Pneumonia",
    "Consolidation",
    "Edema",
    "Atelectasis",
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Enlarged Cardiomediastinum",
    "Pleural Other",
    "Fracture",
)

_POSITIVE = "1.0"
