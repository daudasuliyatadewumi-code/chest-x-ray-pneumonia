"""
Patient Records & Dataset Directory Manager
Automated Chest X-ray Pneumonia Classification System
Student: Dauda Suliyat Adewumi (AI/HND/F24/0073)
Supervisor: Mr. Amari B.R
Institution: Federal Polytechnic Offa - Department of Artificial Intelligence

This module organizes chest X-ray scans and clinical diagnostic reports
into dedicated per-patient directories (e.g., patient_records/PID_XXXX/).
It allows instantaneous location, search, retrieval, and export of any patient's
radiograph, metadata, and diagnostic evaluation anytime requested.
"""

import os
import re
import json
import shutil
import datetime
from PIL import Image
import config

# Define default patient records root directory
PATIENT_RECORDS_DIR = os.path.join(config.BASE_DIR, "patient_records")
os.makedirs(PATIENT_RECORDS_DIR, exist_ok=True)


def validate_patient_id(patient_id):
    """
    Validates that the Patient ID adheres to the standardized clinical hospital format:
    Must start with 'PID-' followed by 3 to 6 digits (e.g., 'PID-001', 'PID-1001').
    
    Returns:
        tuple: (is_valid: bool, sanitized_id_or_error_message: str)
    """
    if not patient_id or not str(patient_id).strip():
        return False, "Patient ID cannot be blank. Please enter a valid ID (e.g., PID-001)."
    
    clean_id = str(patient_id).strip().upper()
    # Format: PID- followed by 3 to 6 digits
    pattern = r"^PID-\d{3,6}$"
    if not re.match(pattern, clean_id):
        return False, (
            "Patient ID must strictly follow the clinical format 'PID-XXXX' "
            "where 'PID-' is followed by 3 to 6 digits (e.g., PID-001, PID-1001). "
            "Arbitrary strings or numbers without 'PID-' prefix are not allowed."
        )
    return True, clean_id


def validate_patient_name(patient_name):
    """
    Validates that the Patient Name contains ONLY alphabetic letters (A-Z, a-z),
    spaces, and standard name punctuation (hyphens, apostrophes).
    Numbers and special symbols are strictly forbidden.
    
    Returns:
        tuple: (is_valid: bool, sanitized_name_or_error_message: str)
    """
    if not patient_name or not str(patient_name).strip():
        return False, "Patient Name cannot be blank. Please enter the full name."
    
    clean_name = str(patient_name).strip()
    
    # Check for any numerical digits
    if any(char.isdigit() for char in clean_name):
        return False, (
            "Patient Name must ONLY contain letters. Numbers are not permitted in a person's name."
        )
    
    # Must only contain letters, spaces, hyphens, and apostrophes
    pattern = r"^[A-Za-z\s\-\'\.]+$"
    if not re.match(pattern, clean_name):
        return False, (
            "Patient Name can only contain alphabetic letters (A-Z, a-z), spaces, "
            "and hyphens. Special symbols and numbers are not allowed."
        )
        
    if len(clean_name) < 2:
        return False, "Patient Name is too short. Please provide a full name of at least 2 letters."
        
    return True, clean_name


def get_patient_dir(patient_id):
    """
    Generate or get the dedicated directory path for a specific patient.
    Sanitizes patient ID for filesystem safety.
    """
    safe_id = "".join([c if c.isalnum() or c in ("-", "_") else "_" for c in str(patient_id).strip()])
    if not safe_id:
        safe_id = "UNKNOWN_PATIENT"
    return os.path.join(PATIENT_RECORDS_DIR, safe_id)


def save_patient_record(patient_id, patient_name, age, gender, image, prediction_result, clinical_notes=""):
    """
    Save a patient's chest X-ray image, diagnostic prediction, and metadata
    into their dedicated folder.
    
    Args:
        patient_id (str): Unique Patient Identifier (e.g. 'PID-1001')
        patient_name (str): Full patient name
        age (int or str): Patient's age
        gender (str): 'Male', 'Female', or 'Other'
        image (PIL.Image or str): Uploaded X-ray image object or source path
        prediction_result (dict): Output from model inference containing:
                                  class_name, confidence, raw_score
        clinical_notes (str): Optional clinical notes or doctor remarks
        
    Returns:
        dict: Summary of saved record and file paths
    """
    # Normalize ID and name if valid
    is_id_ok, clean_id = validate_patient_id(patient_id)
    if is_id_ok:
        patient_id = clean_id
    is_name_ok, clean_name = validate_patient_name(patient_name)
    if is_name_ok:
        patient_name = clean_name

    patient_dir = get_patient_dir(patient_id)
    os.makedirs(patient_dir, exist_ok=True)
    
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    file_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. Save Patient X-ray Image
    image_filename = f"xray_{file_timestamp}.jpeg"
    image_path = os.path.join(patient_dir, image_filename)
    
    if isinstance(image, str):
        # Image is a path to an existing file
        shutil.copyfile(image, image_path)
    elif hasattr(image, "save"):
        # Image is a PIL Image
        if image.mode != "RGB":
            image = image.convert("RGB")
        image.save(image_path, format="JPEG", quality=95)
    
    # Also maintain a symlink/copy of the latest scan as 'latest_xray.jpeg'
    latest_img_path = os.path.join(patient_dir, "latest_xray.jpeg")
    try:
        if isinstance(image, str):
            shutil.copyfile(image, latest_img_path)
        else:
            image.save(latest_img_path, format="JPEG", quality=95)
    except Exception:
        pass
        
    # 2. Compile Metadata & Diagnosis
    record_data = {
        "patient_id": str(patient_id).strip(),
        "patient_name": str(patient_name).strip(),
        "age": str(age).strip(),
        "gender": str(gender).strip(),
        "timestamp": timestamp,
        "primary_image_file": image_filename,
        "diagnosis": prediction_result.get("class_name", "UNKNOWN"),
        "confidence_percentage": round(float(prediction_result.get("confidence", 0.0)) * 100, 2),
        "raw_prediction_score": float(prediction_result.get("raw_score", 0.0)),
        "clinical_notes": clinical_notes,
        "directory_path": patient_dir
    }
    
    # Save JSON metadata
    json_path = os.path.join(patient_dir, "patient_data.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(record_data, f, indent=4)
        
    # 3. Generate Human-Readable Diagnostic Report Text File
    report_text = f"""================================================================================
FEDERAL POLYTECHNIC OFFA - DEPARTMENT OF ARTIFICIAL INTELLIGENCE
AUTOMATED CHEST X-RAY PNEUMONIA CLASSIFICATION CLINICAL REPORT
================================================================================
PATIENT ID:           {record_data['patient_id']}
PATIENT NAME:         {record_data['patient_name']}
AGE / GENDER:         {record_data['age']} yrs / {record_data['gender']}
DATE & TIME:          {record_data['timestamp']}
FOLDER LOCATION:      {record_data['directory_path']}
--------------------------------------------------------------------------------
AI DIAGNOSTIC EVALUATION (ResNet50 Deep CNN):
--------------------------------------------------------------------------------
DIAGNOSIS:            {record_data['diagnosis']}
CONFIDENCE SCORE:     {record_data['confidence_percentage']}%
SIGMOID PROBABILITY:  {record_data['raw_prediction_score']:.6f}
STATUS:               {'CRITICAL - REQUIRES CLINICAL ATTENTION' if record_data['diagnosis'] == 'PNEUMONIA' else 'CLEAR / NORMAL'}

CLINICAL NOTES:
{record_data['clinical_notes'] if record_data['clinical_notes'] else 'No clinical remarks recorded.'}

--------------------------------------------------------------------------------
DISCLAIMER: This automated diagnostic report is generated using deep learning 
(ResNet50 Transfer Learning) as part of an HND AI research project. It is intended
for clinical decision support and should be verified by a licensed radiologist.
================================================================================
"""
    report_path = os.path.join(patient_dir, "diagnostic_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
        
    record_data["report_path"] = report_path
    record_data["image_path"] = latest_img_path
    return record_data


def get_patient_record(patient_id):
    """
    Retrieve full record and files for a patient by Patient ID.
    Returns None if patient directory or record is not found.
    """
    patient_dir = get_patient_dir(patient_id)
    if not os.path.exists(patient_dir):
        return None
        
    json_path = os.path.join(patient_dir, "patient_data.json")
    if not os.path.exists(json_path):
        return None
        
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    # Check for latest image
    latest_img = os.path.join(patient_dir, "latest_xray.jpeg")
    if not os.path.exists(latest_img):
        primary_img = os.path.join(patient_dir, data.get("primary_image_file", ""))
        latest_img = primary_img if os.path.exists(primary_img) else None
        
    report_file = os.path.join(patient_dir, "diagnostic_report.txt")
    report_content = ""
    if os.path.exists(report_file):
        with open(report_file, "r", encoding="utf-8") as rf:
            report_content = rf.read()
            
    data["directory_path"] = patient_dir
    data["image_path"] = latest_img
    data["report_content"] = report_content
    return data


def list_all_patients():
    """
    List all patient directories and return a list of their metadata summaries.
    Sorted alphabetically by Patient ID.
    """
    if not os.path.exists(PATIENT_RECORDS_DIR):
        return []
        
    patients = []
    for entry in sorted(os.listdir(PATIENT_RECORDS_DIR)):
        entry_path = os.path.join(PATIENT_RECORDS_DIR, entry)
        if os.path.isdir(entry_path):
            json_path = os.path.join(entry_path, "patient_data.json")
            if os.path.exists(json_path):
                try:
                    with open(json_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        data["folder_name"] = entry
                        data["full_folder_path"] = entry_path
                        patients.append(data)
                except Exception:
                    continue
    return patients


def search_patients(query):
    """
    Search patients by ID, Name, or Diagnosis.
    Returns matched patient summaries.
    """
    query = str(query).strip().lower()
    all_patients = list_all_patients()
    if not query:
        return all_patients
        
    matched = []
    for p in all_patients:
        pid = p.get("patient_id", "").lower()
        pname = p.get("patient_name", "").lower()
        diag = p.get("diagnosis", "").lower()
        folder = p.get("folder_name", "").lower()
        if query in pid or query in pname or query in diag or query in folder:
            matched.append(p)
    return matched
