import csv
import re
import os

def append_ocr_to_csv(ocr_text: str, filename: str = "results/students.csv"):
    """
    Parse OCR text into key-value pairs and append them as a new row to a CSV file.
    """
    # Step 1: Extract key-value pairs
    data = dict(re.findall(r"([\w\s]+):\s*([^\n,]+)", ocr_text))
    data = {k.strip(): v.strip() for k, v in data.items()}

    # Step 2: Determine fieldnames (existing or from current data)
    if os.path.exists(filename):
        with open(filename, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or list(data.keys())
    else:
        fieldnames = list(data.keys())

    # Step 3: Append to CSV
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if f.tell() == 0:  # empty file
            writer.writeheader()
        writer.writerow(data)

    print(f"Data appended to {filename}")

def append_marks_to_csv(student_id: str, marks: dict, filename: str = "marks.csv"):
    """
    Append or update per-question marks for a student identified by student_id.
    Ensures each row is uniquely identified by 'Student ID' and expands columns as needed.
    """
    key_field = "Student ID"
    existing_rows = []

    if os.path.exists(filename):
        with open(filename, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            existing_rows = list(reader)
            existing_fields = reader.fieldnames or []
    else:
        existing_fields = []

    # Build new/updated row
    target_index = None
    for i, row in enumerate(existing_rows):
        if row.get(key_field, "") == str(student_id):
            target_index = i
            break

    # Union of fields: existing + key + all question ids
    new_fields = set(existing_fields) | {key_field} | set(marks.keys())
    fieldnames = [key_field] + sorted([f for f in new_fields if f != key_field])

    if target_index is None:
        # New student row
        row = {f: "" for f in fieldnames}
        row[key_field] = str(student_id)
        for k, v in marks.items():
            row[str(k)] = v
        existing_rows.append(row)
    else:
        # Update existing row
        row = existing_rows[target_index]
        row[key_field] = str(student_id)
        for k, v in marks.items():
            row[str(k)] = v
        existing_rows[target_index] = row

    # Rewrite CSV with unified columns
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in existing_rows:
            # Ensure all fields present
            out = {f: r.get(f, "") for f in fieldnames}
            writer.writerow(out)

    print(f"Marks saved/updated in {filename} for student {student_id}")
