import pandas as pd

participants = [
    ("sub-002", "F", 22, 168),
    ("sub-003", "M", 22, 181),
    ("sub-004", "F", 23, 162),
    ("sub-005", "M", 22, 177),
    ("sub-006", "F", 23, 171),
    ("sub-007", "M", 23, 186),
    ("sub-008", "M", 26, 176),
    ("sub-009", "M", 22, 176),
    ("sub-010", "M", 23, 175),
    ("sub-011", "M", 23, 183),
]

sessions = [
    "ses-headDown",
    "ses-headNormal",
    "ses-headUp",
]

rows = []

for participant_id, sex, age, height in participants:
    for session_id in sessions:
        rows.append({
            "participant_id": participant_id,
            "session_id": session_id,
            "sex": sex,
            "age": age,
            "height": height,
        })

df = pd.DataFrame(rows)

output = r"spinal_cord\ds004507\participants.tsv"

df.to_csv(
    output,
    sep="\t",
    index=False
)

print(df)
print()
print(f"Rows written: {len(df)}")
print(f"File written: {output}")