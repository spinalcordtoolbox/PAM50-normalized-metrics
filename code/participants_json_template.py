"""
Shared field definitions for participants.json files across all datasets.

Each create_<dataset>_participants.py script selects the subset of fields
relevant to its dataset and writes participants.json alongside participants.tsv.

Usage:
    from participants_json_template import FIELDS, make_participants_json

    json_content = make_participants_json(
        ['participant_id', 'session_id', 'sex', 'age', 'pathology'],
        pathology_levels={'HC': 'Healthy Control', 'MS': 'Multiple Sclerosis'}
    )
"""

FIELDS = {
    "participant_id": {
        "Description": "Unique Participant ID",
        "LongName": "Participant ID"
    },
    "session_id": {
        "Description": "Session ID",
        "LongName": "Session ID"
    },
    "sex": {
        "Description": "Sex of the participant as reported by the participant",
        "LongName": "Sex",
        "Levels": {"M": "male", "F": "female"}
    },
    "age": {
        "Description": "Participant age",
        "LongName": "Participant age",
        "Units": "years"
    },
    "pathology": {
        "Description": "The diagnosis of pathology of the participant",
        "LongName": "Pathology name",
        "Levels": {
            "HC":             "Healthy Control",
            "CN":             "Cognitively Normal",
            "MCI":            "Mild Cognitive Impairment",
            "dementia":       "Dementia",
            "MS":             "Multiple Sclerosis",
            "SCHZ":           "Schizophrenia",
            "bipolar":        "Bipolar Disorder",
            "ADHD":           "Attention-Deficit/Hyperactivity Disorder",
        }
    },
    "handedness": {
        "Description": "Dominant hand of the participant",
        "LongName": "Handedness"
    },
    "scanner": {
        "Description": "Scanner identifier",
        "LongName": "Scanner ID"
    },
    "race": {
        "Description": "Self-reported race of the participant",
        "LongName": "Race"
    },
    "height": {
        "Description": "Participant height",
        "LongName": "Height",
        "Units": "cm"
    },
    "weight": {
        "Description": "Participant weight",
        "LongName": "Weight",
        "Units": "kg"
    },
    "BMI": {
        "Description": "Body mass index",
        "LongName": "BMI",
        "Units": "kg/m²"
    },
}


def make_participants_json(columns: list[str], pathology_levels: dict | None = None) -> dict:
    """
    Build a participants.json dict for the given column list.

    Args:
        columns: ordered list of column names present in participants.tsv
        pathology_levels: if provided, overrides the default pathology Levels dict
                          to include only the levels present in this dataset

    Returns:
        dict ready to be serialised with json.dump
    """
    result = {}
    for col in columns:
        if col not in FIELDS:
            raise ValueError(f"Unknown field '{col}'. Add it to participants_json_template.py.")
        entry = dict(FIELDS[col])
        if col == "pathology" and pathology_levels is not None:
            entry = {**entry, "Levels": pathology_levels}
        result[col] = entry
    return result
