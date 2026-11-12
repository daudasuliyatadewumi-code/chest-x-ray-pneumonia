"""
Streamlit Web Application
A professional web interface for pneumonia classification from chest X-rays,
patient-wise directory management, and codebase architecture visualization.
Designed for defense demonstration:
Student: Dauda Suliyat Adewumi (AI/HND/F24/0073)
Supervisor: Mr. Amari B.R
Institution: Federal Polytechnic Offa - Department of Artificial Intelligence

Run with: streamlit run app.py
"""

import os
import re
import json
import numpy as np
import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import config
import patient_manager


# ============================================================
# Page Configuration
# ============================================================
st.set_page_config(
    page_title="Pneumonia Classifier & Patient Directory System",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# Custom CSS for Professional Medical Styling
# ============================================================
st.markdown("""
<style>
    /* Main styling */
    .main-header {
        text-align: center;
        padding: 1.2rem 1rem;
        background: linear-gradient(135deg, #0d47a1 0%, #1565c0 50%, #0288d1 100%);
        color: white;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.15);
    }
    .main-header h1 {
        font-size: 2.1rem;
        margin-bottom: 0.3rem;
        font-weight: 800;
        letter-spacing: -0.5px;
    }
    .main-header p {
        font-size: 1rem;
        opacity: 0.92;
        margin-bottom: 0;
    }
    
    /* Result cards */
    .result-card {
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
    }
    .result-normal {
        background: linear-gradient(135deg, #e8f5e9, #c8e6c9);
        border: 2px solid #4caf50;
    }
    .result-pneumonia {
        background: linear-gradient(135deg, #ffebee, #ffcdd2);
        border: 2px solid #f44336;
    }
    .result-label {
        font-size: 2.2rem;
        font-weight: bold;
        margin: 0.4rem 0;
    }
    .confidence-text {
        font-size: 1.25rem;
        font-weight: 600;
        opacity: 0.85;
    }
    
    /* Metric cards */
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        border: 1px solid #dee2e6;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: bold;
        color: #0d47a1;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #546e7a;
        font-weight: 500;
    }
    
    /* Info box */
    .info-box {
        background: #e3f2fd;
        color: #0d47a1;
        padding: 0.9rem 1.2rem;
        border-radius: 8px;
        border-left: 4px solid #1976d2;
        margin: 0.8rem 0;
        font-size: 0.95rem;
    }
    
    /* Patient Folder Badge */
    .folder-badge {
        background: #f3e5f5;
        border: 1px solid #ce93d8;
        border-left: 5px solid #7b1fa2;
        padding: 0.8rem 1rem;
        border-radius: 8px;
        color: #4a148c;
        font-family: monospace;
        font-size: 0.9rem;
        word-break: break-all;
        margin: 0.5rem 0;
    }
    
    /* Disclaimer */
    .disclaimer {
        background: #fff3e0;
        color: #e65100;
        padding: 0.8rem;
        border-radius: 8px;
        border-left: 4px solid #ff9800;
        font-size: 0.85rem;
        margin-top: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# Load Model (cached for performance)
# ============================================================
@st.cache_resource
def load_classification_model():
    """Load the trained model with caching."""
    model_path = os.path.join(config.MODEL_DIR, "pneumonia_classifier_final.keras")
    
    # Reassemble split model chunks if the main file is missing
    part_dir = os.path.dirname(model_path)
    if os.path.exists(part_dir):
        part_files = sorted([f for f in os.listdir(part_dir) if f.startswith("model_part_")])
        if not os.path.exists(model_path) and part_files:
            with st.spinner("Reassembling model from chunks (first run only)..."):
                with open(model_path, "wb") as outfile:
                    for part in part_files:
                        with open(os.path.join(part_dir, part), "rb") as infile:
                            outfile.write(infile.read())
                            
    if not os.path.exists(model_path):
        return None
        
    model = load_model(model_path)
    return model


@st.cache_data
def load_metrics():
    """Load evaluation metrics if available."""
    metrics_path = os.path.join(config.LOGS_DIR, "evaluation_metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            return json.load(f)
    return None


def preprocess_image(image):
    """Preprocess uploaded image for prediction."""
    img = image.resize(config.IMG_SIZE)
    if img.mode != "RGB":
        img = img.convert("RGB")
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def is_likely_xray(image):
    """
    Heuristic check to ensure image is a grayscale radiograph.
    Evaluates variance across color channels.
    """
    img_rgb = image.convert('RGB')
    arr = np.array(img_rgb)
    std_dev = np.std(arr, axis=2)
    mean_std_dev = np.mean(std_dev)
    if mean_std_dev > 10.0:
        return False
    return True


def make_prediction(model, img_array):
    """Make prediction and return class and confidence."""
    prediction = model.predict(img_array, verbose=0)[0][0]
    predicted_class = 1 if prediction > 0.5 else 0
    class_name = config.CLASS_NAMES[predicted_class]
    confidence = prediction if predicted_class == 1 else 1 - prediction
    
    return {
        "class_name": class_name,
        "confidence": float(confidence),
        "raw_score": float(prediction)
    }


# ============================================================
# Sidebar
# ============================================================
with st.sidebar:
    st.image("https://img.icons8.com/color/96/lungs.png", width=80)
    st.title("📋 Project Information")
    
    st.markdown("""
    **Project Title:**  
    Automated Pneumonia Classification from Chest Radiographs Using Deep CNNs
    
    **Student:**  
    Dauda Suliyat Adewumi  
    *(AI/HND/F24/0073)*
    
    **Supervisor:**  
    Mr. Amari B.R
    
    **Institution:**  
    Department of Artificial Intelligence  
    School of Computing  
    Federal Polytechnic Offa
    
    ---
    **Model:** ResNet50 (Transfer Learning)  
    **Backbone:** ImageNet Pre-trained  
    **Framework:** TensorFlow / Keras  
    **Classification:** Binary (Normal / Pneumonia)
    """)
    
    st.divider()
    metrics = load_metrics()
    if metrics:
        st.subheader("📊 Test Performance")
        mcol1, mcol2 = st.columns(2)
        with mcol1:
            st.metric("Accuracy", f"{metrics['accuracy']:.2%}")
            st.metric("Recall", f"{metrics['recall']:.2%}")
        with mcol2:
            st.metric("Precision", f"{metrics['precision']:.2%}")
            st.metric("F1-Score", f"{metrics['f1_score']:.2%}")
            
    st.divider()
    st.caption("Federal Polytechnic Offa - AI Final Project")


# ============================================================
# Main Header
# ============================================================
st.markdown("""
<div class="main-header">
    <h1>🫁 Automated Pneumonia Classification & Patient Record System</h1>
    <p>Deep Learning Diagnostic Pipeline & Patient File Directory Locator | Federal Polytechnic Offa</p>
</div>
""", unsafe_allow_html=True)

model = load_classification_model()
if model is None:
    st.error("⚠️ **Model weights file not found.** Please verify outputs/model/pneumonia_classifier_final.keras.")
    st.stop()


# ============================================================
# Navigation Tabs
# ============================================================
tab1, tab2, tab3 = st.tabs([
    "🔍 1. Live Diagnosis & Patient Intake",
    "📁 2. Patient Directory & File Locator",
    "🗺️ 3. System Flowchart & Codebase Architecture"
])


# ------------------------------------------------------------
# TAB 1: LIVE DIAGNOSIS & PATIENT INTAKE
# ------------------------------------------------------------
with tab1:
    st.markdown("### 📤 Patient Radiograph Analysis & Automatic Archival")
    
    col_input, col_result = st.columns([1, 1], gap="large")
    
    with col_input:
        st.markdown("#### Patient Details")
        p_col1, p_col2 = st.columns(2)
        with p_col1:
            patient_id = st.text_input(
                "Patient ID * (Format: PID-XXXX)",
                value="",
                placeholder="e.g., PID-001, PID-1001",
                help="Must start with 'PID-' followed by 3 to 6 digits (e.g., PID-001, PID-1001)"
            )
            # Immediate feedback if invalid ID format entered
            if patient_id.strip() and not re.match(r"^PID-\d{3,6}$", patient_id.strip().upper()):
                st.caption("⚠️ *Format required: 'PID-' followed by 3-6 digits (e.g., PID-001)*")

            patient_name = st.text_input(
                "Patient Full Name * (Letters Only)",
                value="",
                placeholder="e.g., Adebayo Temitope",
                help="Full name of the patient. Must contain ONLY letters (A-Z, a-z) and spaces."
            )
            # Immediate feedback if digits are entered in name
            if any(char.isdigit() for char in patient_name):
                st.caption("⚠️ *Numbers are not allowed in patient name. Letters only.*")
        with p_col2:
            patient_age = st.number_input("Age", min_value=1, max_value=120, value=30)
            patient_gender = st.selectbox("Gender", ["Male", "Female", "Other"])
            
        clinical_notes = st.text_area(
            "Clinical Indications / Notes (Optional)",
            value="",
            placeholder="Enter clinical symptoms, history, or physician remarks...",
            help="Clinical notes to be archived in the patient diagnostic report"
        )
        
        st.markdown("#### Upload Chest Radiograph")
        uploaded_file = st.file_uploader(
            "Upload Chest X-ray (JPEG, JPG, PNG)",
            type=["jpg", "jpeg", "png"],
            help="Upload frontal chest radiograph"
        )
        
        if uploaded_file is not None:
            uploaded_image = Image.open(uploaded_file)
            st.image(uploaded_image, caption="Uploaded Radiograph", use_container_width=True)
            classify_btn = st.button("🚀 Analyze & Archive Patient File", type="primary", use_container_width=True)
        else:
            classify_btn = False
            st.info("👆 Please upload a frontal chest X-ray to perform diagnostic classification.")

    with col_result:
        st.markdown("#### Diagnostic Evaluation")
        
        if uploaded_file is not None and classify_btn:
            # Validate Patient ID format (e.g., PID-001)
            is_id_ok, id_feedback = patient_manager.validate_patient_id(patient_id)
            if not is_id_ok:
                st.error(f"⚠️ **Invalid Patient ID:** {id_feedback}")
                st.stop()
                
            # Validate Patient Name (Letters only)
            is_name_ok, name_feedback = patient_manager.validate_patient_name(patient_name)
            if not is_name_ok:
                st.error(f"⚠️ **Invalid Patient Name:** {name_feedback}")
                st.stop()
                
            # Use normalized uppercase ID and cleaned name
            patient_id = id_feedback
            patient_name = name_feedback
                
            with st.spinner("Analyzing radiograph with ResNet50 CNN..."):
                if not is_likely_xray(uploaded_image):
                    st.error("⚠️ **Validation Failed:** The uploaded image does not match the grayscale optical profile of a medical chest radiograph. Please upload a genuine chest X-ray.")
                    st.stop()
                    
                img_array = preprocess_image(uploaded_image)
                pred_result = make_prediction(model, img_array)
                
                # Save into patient folder
                saved_record = patient_manager.save_patient_record(
                    patient_id=patient_id,
                    patient_name=patient_name,
                    age=patient_age,
                    gender=patient_gender,
                    image=uploaded_image,
                    prediction_result=pred_result,
                    clinical_notes=clinical_notes
                )
                
            # Display Diagnostic Result Card
            if pred_result["class_name"] == "NORMAL":
                st.markdown(f"""
                <div class="result-card result-normal">
                    <div style="font-size: 3rem;">✅</div>
                    <div class="result-label" style="color: #2e7d32;">NORMAL</div>
                    <div class="confidence-text" style="color: #1b5e20;">Confidence: {pred_result['confidence']:.1%}</div>
                    <p style="margin-top: 0.5rem; color: #2e7d32;">No focal consolidations or pneumonic infiltrates detected.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-card result-pneumonia">
                    <div style="font-size: 3rem;">⚠️</div>
                    <div class="result-label" style="color: #c62828;">PNEUMONIA DETECTED</div>
                    <div class="confidence-text" style="color: #b71c1c;">Confidence: {pred_result['confidence']:.1%}</div>
                    <p style="margin-top: 0.5rem; color: #c62828;">Patchy airspace consolidation/infiltrates identified.</p>
                </div>
                """, unsafe_allow_html=True)
                
            # Confidence Metrics
            m1, m2, m3 = st.columns(3)
            with m1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{pred_result['class_name']}</div>
                    <div class="metric-label">Diagnosis</div>
                </div>
                """, unsafe_allow_html=True)
            with m2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{pred_result['confidence']:.1%}</div>
                    <div class="metric-label">Confidence</div>
                </div>
                """, unsafe_allow_html=True)
            with m3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{pred_result['raw_score']:.4f}</div>
                    <div class="metric-label">Sigmoid Score</div>
                </div>
                """, unsafe_allow_html=True)
                
            st.progress(pred_result["confidence"])
            
            # File Location Confirmation
            st.success(f"✅ **Patient File Successfully Archived in Dedicated Directory!**")
            st.markdown(f"""
            <div class="folder-badge">
                📁 <b>Located Directory Path:</b><br>{saved_record['directory_path']}
            </div>
            """, unsafe_allow_html=True)
            
            st.caption("Files saved: `latest_xray.jpeg`, `patient_data.json`, `diagnostic_report.txt`")
            
        elif uploaded_file is None:
            st.markdown("""
            <div style="text-align: center; padding: 4rem 1rem; color: #90a4ae;">
                <div style="font-size: 4rem;">🫁</div>
                <p style="font-size: 1.15rem; font-weight: 500;">Enter patient information and upload an X-ray<br>to generate classification and archive to folder</p>
            </div>
            """, unsafe_allow_html=True)


# ------------------------------------------------------------
# TAB 2: PATIENT DIRECTORY & FILE LOCATOR
# ------------------------------------------------------------
with tab2:
    st.markdown("### 📁 Patient Dataset Directory & File Retrieval System")
    st.markdown("""
    <div class="info-box">
        Every patient has an isolated, structured folder on disk 
        (<code>patient_records/&lt;PATIENT_ID&gt;/</code>). Whenever a patient or requests their file, 
        simply enter their ID or Name below to immediately locate their folder path, radiograph scan, and full diagnostic report.
    </div>
    """, unsafe_allow_html=True)
    
    # Search and Filter
    search_col1, search_col2 = st.columns([2, 1])
    with search_col1:
        search_query = st.text_input("🔎 Search Patient by ID, Name, or Diagnosis:", placeholder="e.g. PID-1001, Adebayo, or Pneumonia")
    with search_col2:
        filter_diag = st.selectbox("Filter Diagnosis:", ["All", "NORMAL", "PNEUMONIA"])
        
    # Get patients
    matched_patients = patient_manager.search_patients(search_query)
    if filter_diag != "All":
        matched_patients = [p for p in matched_patients if p.get("diagnosis") == filter_diag]
        
    st.markdown(f"**Found {len(matched_patients)} Registered Patient Folder(s):**")
    
    if not matched_patients:
        st.info("ℹ️ **No patient records found.** Use **Tab 1: Live Diagnosis & Patient Intake** to enter your patient details, upload a chest X-ray, and archive their record here.")
    else:
        # Patient selector
        patient_options = [f"{p.get('patient_id')} — {p.get('patient_name')} ({p.get('diagnosis')})" for p in matched_patients]
        selected_option = st.selectbox("Select Patient to Retrieve File:", patient_options)
        
        selected_idx = patient_options.index(selected_option)
        sel_patient_id = matched_patients[selected_idx].get("patient_id")
        
        # Load complete patient record
        patient_data = patient_manager.get_patient_record(sel_patient_id)
        
        if patient_data:
            st.divider()
            
            # File Location Header
            st.markdown(f"#### 📍 File Location for Patient: `{patient_data['patient_id']}`")
            st.markdown(f"""
            <div class="folder-badge">
                📁 <b>Physical Storage Location on Disk:</b><br>{patient_data['directory_path']}
            </div>
            """, unsafe_allow_html=True)
            
            # 2 Columns: Image & Details
            pcol_img, pcol_info = st.columns([1, 1], gap="large")
            
            with pcol_img:
                st.markdown("##### 🩻 Stored Radiograph Scan")
                if patient_data.get("image_path") and os.path.exists(patient_data["image_path"]):
                    patient_xray = Image.open(patient_data["image_path"])
                    st.image(patient_xray, caption=f"Chest X-ray: {patient_data['patient_id']} - {patient_data['patient_name']}", use_container_width=True)
                else:
                    st.info("Radiograph file not found in folder.")
                    
            with pcol_info:
                st.markdown("##### 📑 Patient & Diagnostic Record")
                
                # Badges for status
                if patient_data.get("diagnosis") == "NORMAL":
                    st.success(f"**Diagnosis:** NORMAL (Clear) — Confidence: {patient_data.get('confidence_percentage')}%")
                else:
                    st.error(f"**Diagnosis:** PNEUMONIA DETECTED — Confidence: {patient_data.get('confidence_percentage')}%")
                    
                st.markdown(f"""
                - **Patient ID:** `{patient_data.get('patient_id')}`
                - **Full Name:** **{patient_data.get('patient_name')}**
                - **Age & Gender:** {patient_data.get('age')} years | {patient_data.get('gender')}
                - **Date/Time of Examination:** `{patient_data.get('timestamp')}`
                - **Sigmoid Score:** `{patient_data.get('raw_prediction_score', 0):.6f}`
                - **Clinical Indications:** {patient_data.get('clinical_notes', 'N/A')}
                """)
                
                st.markdown("##### 📂 Directory Contents:")
                if os.path.exists(patient_data["directory_path"]):
                    files_in_folder = os.listdir(patient_data["directory_path"])
                    for fname in files_in_folder:
                        fpath = os.path.join(patient_data["directory_path"], fname)
                        size_kb = os.path.getsize(fpath) / 1024
                        st.markdown(f"- 📄 `{fname}` *({size_kb:.1f} KB)*")
                        
                # Download Report Button
                if patient_data.get("report_content"):
                    st.download_button(
                        label="📥 Download Clinical Diagnostic Report (.txt)",
                        data=patient_data["report_content"],
                        file_name=f"Diagnostic_Report_{patient_data['patient_id']}.txt",
                        mime="text/plain",
                        use_container_width=True
                    )
                    
            # Expandable full clinical report
            with st.expander("📄 View Full Diagnostic Report Content"):
                st.text(patient_data.get("report_content", "No report text available."))


# ------------------------------------------------------------
# TAB 3: SYSTEM FLOWCHART & CODEBASE ARCHITECTURE
# ------------------------------------------------------------
with tab3:
    st.markdown("### 🗺️ System Flowchart & Codebase Architecture Map")
    st.markdown("""
    <div class="info-box">
        This comprehensive flowchart illustrates the entire technical pipeline, demonstrating how each Python module 
        (<code>download_dataset.py</code>, <code>patient_manager.py</code>, <code>data_preprocessing.py</code>, 
        <code>model.py</code>, <code>train.py</code>, <code>evaluate.py</code>, <code>app.py</code>) operates 
        together to achieve automated diagnosis and patient file organization.
    </div>
    """, unsafe_allow_html=True)
    
    # Display Flowchart Image
    flowchart_path = os.path.join(config.PLOTS_DIR, "system_flowchart.png")
    if os.path.exists(flowchart_path):
        st.image(flowchart_path, caption="Comprehensive System Flowchart & Codebase Mapping (Federal Polytechnic Offa)", use_container_width=True)
    else:
        st.info("Run `python generate_system_flowchart.py` to compile the architectural flowchart.")
        
    st.divider()
    
    # Publication Evaluation Figures
    st.markdown("### 📊 Model Evaluation Figures & Training Trajectory")
    
    fig_col1, fig_col2 = st.columns(2)
    
    with fig_col1:
        st.markdown("#### Figure 4.4: Confusion Matrix")
        cm_path = os.path.join(config.PLOTS_DIR, "Figure_4.4_Confusion_Matrix.png")
        if os.path.exists(cm_path):
            st.image(cm_path, caption="Confusion Matrix on Unseen Test Radiographs (N=624)", use_container_width=True)
            
    with fig_col2:
        st.markdown("#### Figure 4.5: Accuracy Curves")
        acc_path = os.path.join(config.PLOTS_DIR, "Figure_4.5_Training_and_Validation_Accuracy.png")
        if os.path.exists(acc_path):
            st.image(acc_path, caption="Training vs Validation Accuracy Across Phase 1 and Phase 2", use_container_width=True)
            
    loss_path = os.path.join(config.PLOTS_DIR, "Figure_4.6_Training_and_Validation_Loss.png")
    if os.path.exists(loss_path):
        st.markdown("#### Figure 4.6: Training and Validation Loss")
        st.image(loss_path, caption="Binary Cross-Entropy Loss Convergence (Phase 1 & Phase 2)", use_container_width=True)


# ============================================================
# Disclaimer & Footer
# ============================================================
st.markdown("""
<div class="disclaimer">
    <strong>⚠️ Clinical Disclaimer:</strong> This deep learning system is developed for academic research and educational demonstration 
    at the Department of Artificial Intelligence, Federal Polytechnic Offa. It provides decision-support insights and should not 
    replace standard clinical radiologist evaluation.
</div>
""", unsafe_allow_html=True)

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #546e7a; font-size: 0.88rem;">
    <b>Automated Pneumonia Classification & Patient Record System</b> | ResNet50 Transfer Learning<br>
    <b>Student:</b> Dauda Suliyat Adewumi (AI/HND/F24/0073) | <b>Supervisor:</b> Mr. Amari B.R<br>
    Department of Artificial Intelligence, School of Computing, Federal Polytechnic Offa
</div>
""", unsafe_allow_html=True)
