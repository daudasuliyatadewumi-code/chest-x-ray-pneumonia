"""
Simple & Clear Academic Flowchart Generator
Creates an intuitive, high-readability flowchart using standard shapes:
- Stadium / Oval: Start and End
- Parallelogram: Input / Output Operations
- Rectangle: Processing & Computational Steps
- Kite / Diamond: Decision Making Points
- Clear, large text with high contrast for easy reading and defense presentation.

Outputs:
- outputs/plots/system_flowchart.png
- outputs/plots/flowchart.png
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Polygon, FancyBboxPatch

# Output Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(BASE_DIR, "outputs", "plots")
ARTIFACT_DIR = r"C:\Users\DELL\.gemini\antigravity-ide\brain\c03fb3ec-902f-4af5-817e-bc195ef0f75d"
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(ARTIFACT_DIR, exist_ok=True)


def draw_diamond(ax, cx, cy, w, h, text, code_ref=None,
                 facecolor="#fff3cd", edgecolor="#d39e00", text_color="#212529"):
    """Draw kite / diamond for decision points with centered text."""
    pts = [
        [cx, cy + h / 2],      # Top
        [cx + w / 2, cy],      # Right
        [cx, cy - h / 2],      # Bottom
        [cx - w / 2, cy]       # Left
    ]
    poly = Polygon(pts, closed=True, facecolor=facecolor, edgecolor=edgecolor,
                   linewidth=2.5, zorder=3)
    ax.add_patch(poly)
    
    if code_ref:
        ax.text(cx, cy + h * 0.22, f"[{code_ref}]", ha="center", va="center",
                fontsize=9.5, fontweight="bold", color="#6c757d", zorder=4, family="monospace")
        ax.text(cx, cy - h * 0.08, text, ha="center", va="center",
                fontsize=11.5, fontweight="bold", color=text_color, zorder=4, linespacing=1.2)
    else:
        ax.text(cx, cy, text, ha="center", va="center",
                fontsize=12, fontweight="bold", color=text_color, zorder=4, linespacing=1.2)


def draw_parallelogram(ax, x, y, w, h, text, code_ref=None,
                       skew=0.035, facecolor="#e1f5fe", edgecolor="#0288d1",
                       text_color="#01579b"):
    """Draw slanted parallelogram for input/output."""
    pts = [
        [x + skew, y + h],        # Top-left
        [x + w, y + h],           # Top-right
        [x + w - skew, y],        # Bottom-right
        [x, y]                    # Bottom-left
    ]
    poly = Polygon(pts, closed=True, facecolor=facecolor, edgecolor=edgecolor,
                   linewidth=2.5, zorder=3)
    ax.add_patch(poly)
    
    cx = x + w / 2
    cy = y + h / 2
    if code_ref:
        ax.text(cx, cy + h * 0.22, f"[{code_ref}]", ha="center", va="center",
                fontsize=9.5, fontweight="bold", color="#455a64", zorder=4, family="monospace")
        ax.text(cx, cy - h * 0.12, text, ha="center", va="center",
                fontsize=11.5, fontweight="bold", color=text_color, zorder=4, linespacing=1.2)
    else:
        ax.text(cx, cy, text, ha="center", va="center",
                fontsize=12, fontweight="bold", color=text_color, zorder=4, linespacing=1.2)


def draw_process(ax, x, y, w, h, text, title=None, code_ref=None,
                 facecolor="#ffffff", edgecolor="#1565c0",
                 text_color="#212529", title_color="#0d47a1"):
    """Draw clean rectangle for process steps."""
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.008,rounding_size=0.015",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=2.2,
        zorder=3
    )
    ax.add_patch(box)
    
    cx = x + w / 2
    top_y = y + h - 0.022
    if title:
        ax.text(cx, top_y, title, ha="center", va="top",
                fontsize=12, fontweight="bold", color=title_color, zorder=4)
        top_y -= 0.026
        
    if code_ref:
        ax.text(cx, top_y, f"[Module: {code_ref}]", ha="center", va="top",
                fontsize=9.5, fontweight="bold", color="#546e7a", zorder=4, family="monospace")
        top_y -= 0.024
        
    ax.text(cx, (top_y + y + 0.008) / 2, text, ha="center", va="center",
            fontsize=10.5, color=text_color, zorder=4, linespacing=1.3)


def draw_stadium(ax, x, y, w, h, text, code_ref=None,
                 facecolor="#e8f5e9", edgecolor="#2e7d32",
                 text_color="#1b5e20"):
    """Draw stadium / pill shape for start and endpoints."""
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.01,rounding_size={h / 2.0}",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=2.5,
        zorder=3
    )
    ax.add_patch(box)
    
    cx = x + w / 2
    cy = y + h / 2
    if code_ref:
        ax.text(cx, cy + h * 0.20, f"[{code_ref}]", ha="center", va="center",
                fontsize=9.5, fontweight="bold", color="#37474f", zorder=4, family="monospace")
        ax.text(cx, cy - h * 0.12, text, ha="center", va="center",
                fontsize=12, fontweight="bold", color=text_color, zorder=4)
    else:
        ax.text(cx, cy, text, ha="center", va="center",
                fontsize=12.5, fontweight="bold", color=text_color, zorder=4)


def draw_arrow(ax, x1, y1, x2, y2, label=None, color="#37474f", width=2.2, label_offset=(0, 0)):
    """Draw connecting arrow with bold clear label."""
    ax.annotate(
        "",
        xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle="-|>",
            color=color,
            lw=width,
            mutation_scale=15,
            shrinkA=2,
            shrinkB=2
        ),
        zorder=2
    )
    if label:
        lx = (x1 + x2) / 2 + label_offset[0]
        ly = (y1 + y2) / 2 + label_offset[1]
        ax.text(lx, ly, label, ha="center", va="center",
                fontsize=10, fontweight="bold", color=color,
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor=color, lw=1.2, alpha=0.95),
                zorder=5)


def generate_simple_flowchart():
    # Dimensions: 12 x 18 inches at 300 DPI (crisp and clear)
    fig, ax = plt.subplots(figsize=(12, 18), dpi=300)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # =========================================================
    # 1. HEADER
    # =========================================================
    header = FancyBboxPatch(
        (0.06, 0.932), 0.88, 0.052,
        boxstyle="round,pad=0.008,rounding_size=0.012",
        facecolor="#0d47a1", edgecolor="#083170", linewidth=2, zorder=3
    )
    ax.add_patch(header)
    ax.text(0.5, 0.963, "PNEUMONIA CLASSIFICATION & PATIENT RECORD SYSTEM",
            ha="center", va="center", fontsize=15, fontweight="bold", color="white", zorder=4)
    ax.text(0.5, 0.943, "System Codebase & Decision Flowchart | Federal Polytechnic Offa | Dept. of Artificial Intelligence\nStudent: Dauda Suliyat Adewumi (AI/HND/F24/0073) | Supervisor: Mr. Amari B.R",
            ha="center", va="center", fontsize=9.5, color="#bbdefb", zorder=4)

    # =========================================================
    # 2. START TERMINAL (Oval/Stadium)
    # =========================================================
    draw_stadium(ax, 0.32, 0.865, 0.36, 0.046,
                 text="START: System Initialization",
                 code_ref="config.py",
                 facecolor="#e8eaf6", edgecolor="#283593", text_color="#1a237e")

    draw_arrow(ax, 0.50, 0.865, 0.50, 0.825)

    # =========================================================
    # 3. INPUT (Parallelogram)
    # =========================================================
    draw_parallelogram(ax, 0.20, 0.760, 0.60, 0.065,
                       text="Input Patient Details (ID, Name, Age) & Upload Frontal Chest X-ray",
                       code_ref="app.py / download_dataset.py",
                       skew=0.035, facecolor="#e1f5fe", edgecolor="#0288d1", text_color="#01579b")

    draw_arrow(ax, 0.50, 0.760, 0.50, 0.708)

    # =========================================================
    # 4. DECISION 1 (Kite / Diamond): Validation
    # =========================================================
    draw_diamond(ax, 0.50, 0.655, 0.36, 0.095,
                 text="Is Image a Valid\nGrayscale X-ray?\n(RGB variance <= 10.0)",
                 code_ref="app.py",
                 facecolor="#fff8e1", edgecolor="#f57f17", text_color="#e65100")

    # Branch NO: Error / Rejection
    draw_arrow(ax, 0.68, 0.655, 0.78, 0.655, label="NO (Color Photo)", color="#c62828", label_offset=(0, 0.015))
    draw_stadium(ax, 0.78, 0.635, 0.18, 0.042,
                 text="Reject Image &\nPrompt Valid Scan",
                 facecolor="#ffebee", edgecolor="#c62828", text_color="#b71c1c")

    # Branch YES: Proceed down
    draw_arrow(ax, 0.50, 0.607, 0.50, 0.550, label="YES (Valid)", color="#2e7d32", label_offset=(0.06, 0))

    # =========================================================
    # 5. PROCESS 1 (Rectangle): Preprocessing & Augmentation
    # =========================================================
    draw_process(ax, 0.16, 0.465, 0.68, 0.075,
                 title="Data Preprocessing & Class Weight Balancing",
                 code_ref="data_preprocessing.py",
                 text="• Normalize pixel intensities to [0, 1] (1/255.0) & resize to 224 x 224 x 3\n• Real-time augmentation: rotation (+/-20°), zoom, shear, and horizontal flips\n• Balanced class weights: Normal (1.94) vs. Pneumonia (0.67) to eliminate bias",
                 facecolor="#ffffff", edgecolor="#2e7d32", text_color="#212529", title_color="#1b5e20")

    draw_arrow(ax, 0.50, 0.465, 0.50, 0.415)

    # =========================================================
    # 6. PROCESS 2 (Rectangle): ResNet50 Inference
    # =========================================================
    draw_process(ax, 0.16, 0.330, 0.68, 0.075,
                 title="ResNet50 Deep CNN Feature Extraction & Classification",
                 code_ref="model.py / train.py / predict.py",
                 text="• Pre-trained ResNet50 backbone extracts high-level pulmonary features\n• Classification Head: GlobalAvgPool2D -> Dense(256) -> BatchNorm -> Dropout(0.5)\n• Output Neuron: Sigmoid activation outputs probability score P in range [0.0, 1.0]",
                 facecolor="#ffffff", edgecolor="#0d47a1", text_color="#212529", title_color="#0d47a1")

    draw_arrow(ax, 0.50, 0.330, 0.50, 0.278)

    # =========================================================
    # 7. DECISION 2 (Kite / Diamond): Diagnosis Decision Point
    # =========================================================
    draw_diamond(ax, 0.50, 0.230, 0.34, 0.090,
                 text="Is Sigmoid\nProbability (P)\n> 0.5?",
                 code_ref="model.py / predict.py",
                 facecolor="#fff3e0", edgecolor="#ff9800", text_color="#e65100")

    # Branch Left: P <= 0.5 -> NORMAL
    draw_arrow(ax, 0.33, 0.230, 0.20, 0.165, label="NO: P <= 0.5", color="#2e7d32", label_offset=(-0.03, 0.02))
    draw_stadium(ax, 0.08, 0.115, 0.26, 0.050,
                 text="NORMAL DIAGNOSIS\nConfidence: (1 - P) * 100%",
                 facecolor="#e8f5e9", edgecolor="#2e7d32", text_color="#1b5e20")

    # Branch Right: P > 0.5 -> PNEUMONIA
    draw_arrow(ax, 0.67, 0.230, 0.80, 0.165, label="YES: P > 0.5", color="#c62828", label_offset=(0.03, 0.02))
    draw_stadium(ax, 0.66, 0.115, 0.26, 0.050,
                 text="PNEUMONIA DETECTED\nConfidence: P * 100%",
                 facecolor="#ffebee", edgecolor="#c62828", text_color="#b71c1c")

    # =========================================================
    # 8. PATIENT FOLDER ARCHIVAL (Parallelogram - Output)
    # =========================================================
    # Connect both outcomes to patient folder
    draw_arrow(ax, 0.21, 0.115, 0.36, 0.065)
    draw_arrow(ax, 0.79, 0.115, 0.64, 0.065)

    draw_parallelogram(ax, 0.15, 0.015, 0.70, 0.058,
                       text="Save to Dedicated Folder: patient_records/<PATIENT_ID>/\nStores: xray.jpeg, patient_data.json, and diagnostic_report.txt\n-> Ready for Instant Retrieval Anytime Patient or Supervisor Requests File!",
                       code_ref="patient_manager.py",
                       skew=0.030, facecolor="#f3e5f5", edgecolor="#7b1fa2", text_color="#4a148c")

    # =========================================================
    # 9. SIMPLE SYMBOL LEGEND AT TOP-RIGHT
    # =========================================================
    leg_box = FancyBboxPatch(
        (0.72, 0.852), 0.22, 0.065,
        boxstyle="round,pad=0.005,rounding_size=0.008",
        facecolor="#f8f9fa", edgecolor="#b0bec5", linewidth=1.2, zorder=3
    )
    ax.add_patch(leg_box)
    ax.text(0.83, 0.905, "FLOWCHART KEY", ha="center", va="center",
            fontsize=8.5, fontweight="bold", color="#263238", zorder=4)
    ax.text(0.83, 0.873, "• Kite = Decision Point\n• Parallelogram = Input/Output\n• Rectangle = Processing Step\n• Oval = Start / Outcome",
            ha="center", va="center", fontsize=7.5, color="#37474f", zorder=4)

    # Save
    plt.tight_layout()
    sys_path = os.path.join(PLOTS_DIR, "system_flowchart.png")
    std_path = os.path.join(PLOTS_DIR, "flowchart.png")
    art_path = os.path.join(ARTIFACT_DIR, "system_flowchart.png")

    plt.savefig(sys_path, dpi=300, bbox_inches="tight")
    plt.savefig(std_path, dpi=300, bbox_inches="tight")
    plt.savefig(art_path, dpi=300, bbox_inches="tight")
    plt.close()

    print("[SUCCESS] Simple & clear flowchart generated successfully!")
    print(f"[OUTPUT] Saved to: {sys_path}")


if __name__ == "__main__":
    generate_simple_flowchart()
