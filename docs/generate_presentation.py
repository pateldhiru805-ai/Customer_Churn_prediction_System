"""Presentation Deck Generator (.pptx) for Customer Churn Prediction System.

Generates an executive, academic-ready 12-slide PowerPoint presentation with
custom dark styling, benchmark tables, and embedded high-resolution figures.
"""

from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE

ROOT_DIR = Path(__file__).resolve().parent.parent
FIGURES_DIR = ROOT_DIR / "reports" / "figures"
OUTPUT_PPTX = ROOT_DIR / "FINAL_PROJECT_PRESENTATION_PHASE_2.pptx"

# Color Palette: Modern Obsidian / Indigo Theme
COLOR_BG = RGBColor(11, 15, 25)         # Dark slate #0B0F19
COLOR_CARD = RGBColor(21, 29, 48)       # #151D30
COLOR_TEXT = RGBColor(248, 250, 252)    # Clean white #F8FAFC
COLOR_SUBTEXT = RGBColor(148, 163, 184) # Muted slate #94A3B8
COLOR_ACCENT = RGBColor(99, 102, 241)   # Indigo #6366F1
COLOR_CYAN = RGBColor(6, 182, 212)      # Cyan #06B6D4
COLOR_GREEN = RGBColor(16, 185, 129)    # Emerald #10B981


def set_slide_background(slide):
    """Set custom dark background for slide."""
    background = slide.background
    fill = background.fill
    fill.solid()
    fill.fore_color.rgb = COLOR_BG


def add_header(slide, title_text, category="CUSTOMER CHURN PREDICTION SYSTEM"):
    """Add standardized styled header on each slide."""
    txBox = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(11.5), Inches(1.1))
    tf = txBox.text_frame
    tf.word_wrap = True

    p_cat = tf.paragraphs[0]
    p_cat.text = category.upper()
    p_cat.font.size = Pt(11)
    p_cat.font.bold = True
    p_cat.font.color.rgb = COLOR_CYAN

    p_title = tf.add_paragraph()
    p_title.text = title_text
    p_title.font.size = Pt(24)
    p_title.font.bold = True
    p_title.font.color.rgb = COLOR_TEXT
    p_title.space_before = Pt(4)


def create_deck():
    prs = Presentation()
    # 16:9 Widescreen dimensions
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]

    # -------------------------------------------------------------
    # SLIDE 1: Title Slide
    # -------------------------------------------------------------
    slide1 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide1)

    tbox = slide1.shapes.add_textbox(Inches(1.2), Inches(1.8), Inches(11.0), Inches(3.8))
    tf = tbox.text_frame
    tf.word_wrap = True

    p_sub = tf.paragraphs[0]
    p_sub.text = "PHASE 2 FINAL PROJECT DEFENSE"
    p_sub.font.size = Pt(14)
    p_sub.font.bold = True
    p_sub.font.color.rgb = COLOR_CYAN

    p_main = tf.add_paragraph()
    p_main.text = "Customer Churn Prediction System"
    p_main.font.size = Pt(40)
    p_main.font.bold = True
    p_main.font.color.rgb = COLOR_TEXT
    p_main.space_before = Pt(10)

    p_desc = tf.add_paragraph()
    p_desc.text = "An End-to-End Machine Learning Intelligence Platform for Telecom Retention & Revenue Protection"
    p_desc.font.size = Pt(18)
    p_desc.font.color.rgb = COLOR_SUBTEXT
    p_desc.space_before = Pt(8)

    p_auth = tf.add_paragraph()
    p_auth.text = "Presenter: Dhiraj Mahajan | Stack: Python, Scikit-learn, Flask, MySQL, Docker | 2026"
    p_auth.font.size = Pt(13)
    p_auth.font.color.rgb = COLOR_ACCENT
    p_auth.space_before = Pt(28)

    # -------------------------------------------------------------
    # SLIDE 2: Problem Statement & Business Opportunity
    # -------------------------------------------------------------
    slide2 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide2)
    add_header(slide2, "The Telecom Customer Churn Challenge")

    box2 = slide2.shapes.add_textbox(Inches(0.8), Inches(1.7), Inches(11.5), Inches(5.2))
    tf2 = box2.text_frame
    tf2.word_wrap = True

    points = [
        ("The Revenue Problem:", "Telecommunication providers lose massive Annual Recurring Revenue (ARR) when subscribers terminate contracts. Historical churn in the dataset sits at 26.5%."),
        ("The Economic Equation:", "Customer Acquisition Cost (CAC) is 5x to 7x higher than Customer Retention Cost (CRC). Retaining existing subscribers directly protects gross margin."),
        ("The Solution Delivered:", "An end-to-end Machine Learning decision support workbench that identifies high-risk subscribers, explains root-cause drivers, calculates ARR at risk, and prescribes automated retention playbooks."),
        ("Core Production Objectives:", "Exceed 0.83 ROC-AUC, tune decision threshold for ~70% churn recall, serve via interactive REST API & Web GUI, and ensure resilient database logging.")
    ]
    for i, (title, desc) in enumerate(points):
        p = tf2.paragraphs[0] if i == 0 else tf2.add_paragraph()
        p.text = f"•  {title} "
        p.font.size = Pt(15)
        p.font.bold = True
        p.font.color.rgb = COLOR_TEXT
        p.space_before = Pt(14)
        run = p.add_run()
        run.text = desc
        run.font.bold = False
        run.font.color.rgb = COLOR_SUBTEXT

    # -------------------------------------------------------------
    # SLIDE 3: Dataset Profile & Key EDA Findings
    # -------------------------------------------------------------
    slide3 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide3)
    add_header(slide3, "Dataset Profile & Exploratory Discoveries")

    box3 = slide3.shapes.add_textbox(Inches(0.8), Inches(1.7), Inches(11.5), Inches(5.2))
    tf3 = box3.text_frame
    tf3.word_wrap = True

    eda_points = [
        ("Cohort Scope:", "7,043 subscriber records across 20 raw demographic, contractual, and service features (IBM Telco benchmark)."),
        ("Contract Vulnerability:", "Subscribers on Month-to-month contracts experience a 42.7% churn rate, compared to only 2.8% on 2-year contracts."),
        ("The Tenure Hazard Curve:", "Subscribers in their first 6 months exhibit the highest cancellation probability. Churn drops exponentially as tenure crosses 24 months."),
        ("Payment Friction:", "Customers paying by manual Electronic Check churn at more than double the rate of automated ACH / Credit Card autopay users."),
        ("Support Service Anchoring:", "Lack of Online Security and Tech Support add-ons correlates with higher attrition during price friction.")
    ]
    for i, (title, desc) in enumerate(eda_points):
        p = tf3.paragraphs[0] if i == 0 else tf3.add_paragraph()
        p.text = f"•  {title} "
        p.font.size = Pt(15)
        p.font.bold = True
        p.font.color.rgb = COLOR_TEXT
        p.space_before = Pt(14)
        run = p.add_run()
        run.text = desc
        run.font.bold = False
        run.font.color.rgb = COLOR_SUBTEXT

    # -------------------------------------------------------------
    # SLIDE 4: Machine Learning Benchmark & Winner Selection
    # -------------------------------------------------------------
    slide4 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide4)
    add_header(slide4, "Model Benchmarking & 5-Fold Stratified Cross-Validation")

    # Add Table
    rows, cols = 7, 6
    table_shape = slide4.shapes.add_table(rows, cols, Inches(0.8), Inches(1.8), Inches(11.7), Inches(4.5))
    table = table_shape.table

    headers = ["Classifier Algorithm", "5-Fold CV ROC-AUC", "Test ROC-AUC", "Test PR-AUC", "Test Recall", "Test F1"]
    for j, h in enumerate(headers):
        cell = table.cell(0, j)
        cell.text = h
        cell.fill.solid()
        cell.fill.fore_color.rgb = COLOR_CARD
        for p in cell.text_frame.paragraphs:
            p.font.size = Pt(12)
            p.font.bold = True
            p.font.color.rgb = COLOR_CYAN

    data = [
        ["Gradient Boosting (WINNER)", "0.8488 ± 0.011", "0.8441", "0.6599", "0.6872", "0.6238"],
        ["Random Forest", "0.8473 ± 0.0109", "0.8441", "0.6561", "0.8396", "0.6249"],
        ["Logistic Regression", "0.8459 ± 0.0124", "0.8419", "0.6344", "0.8369", "0.6131"],
        ["AdaBoost", "0.8451 ± 0.0122", "0.8408", "0.6507", "0.6765", "0.6111"],
        ["HistGradientBoosting", "0.8359 ± 0.0109", "0.8365", "0.6436", "0.7834", "0.6221"],
        ["Decision Tree", "0.8250 ± 0.0091", "0.8298", "0.6009", "0.8529", "0.6111"]
    ]
    for i, row in enumerate(data):
        for j, val in enumerate(row):
            cell = table.cell(i + 1, j)
            cell.text = val
            cell.fill.solid()
            cell.fill.fore_color.rgb = COLOR_CARD if i > 0 else RGBColor(30, 41, 65)
            for p in cell.text_frame.paragraphs:
                p.font.size = Pt(11)
                p.font.color.rgb = COLOR_GREEN if j in [2, 4] and i == 0 else COLOR_TEXT

    # -------------------------------------------------------------
    # SLIDE 5: Visual Figures (ROC Curves & Confusion Matrix)
    # -------------------------------------------------------------
    slide5 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide5)
    add_header(slide5, "Performance Visualizations & Discriminative Power")

    roc_img = FIGURES_DIR / "02_roc_curves.png"
    cm_img = FIGURES_DIR / "01_confusion_matrix.png"

    if roc_img.exists():
        slide5.shapes.add_picture(str(roc_img), Inches(0.8), Inches(1.8), width=Inches(5.6))
    if cm_img.exists():
        slide5.shapes.add_picture(str(cm_img), Inches(6.8), Inches(1.8), width=Inches(5.6))

    # -------------------------------------------------------------
    # SLIDE 6: Decision Threshold Optimization
    # -------------------------------------------------------------
    slide6 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide6)
    add_header(slide6, "Threshold Optimization & Recall Maximization")

    thresh_img = FIGURES_DIR / "05_threshold_tuning.png"
    if thresh_img.exists():
        slide6.shapes.add_picture(str(thresh_img), Inches(0.8), Inches(1.8), width=Inches(5.8))

    box6 = slide6.shapes.add_textbox(Inches(7.0), Inches(1.8), Inches(5.5), Inches(5.0))
    tf6 = box6.text_frame
    tf6.word_wrap = True

    th_points = [
        ("The Asymmetric Cost Paradigm:", "A False Negative (customer leaves undetected) forfeits full annual contract revenue. A False Positive incurs only a small discount outreach."),
        ("Tuned Threshold (0.3594):", "Optimized on validation data via Precision-Recall trade-off curve to hit the PRD target of ~70% churn recall."),
        ("Outcome:", "Captures ~70% of potential churners before cancellation occurs, while maintaining an industry-leading 0.8441 ROC-AUC.")
    ]
    for i, (title, desc) in enumerate(th_points):
        p = tf6.paragraphs[0] if i == 0 else tf6.add_paragraph()
        p.text = f"•  {title} "
        p.font.size = Pt(14)
        p.font.bold = True
        p.font.color.rgb = COLOR_TEXT
        p.space_before = Pt(16)
        run = p.add_run()
        run.text = desc
        run.font.bold = False
        run.font.color.rgb = COLOR_SUBTEXT

    # -------------------------------------------------------------
    # SLIDE 7: Feature Importance & Explainability Drivers
    # -------------------------------------------------------------
    slide7 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide7)
    add_header(slide7, "Root-Cause Explainability Drivers")

    fi_img = FIGURES_DIR / "04_feature_importance.png"
    if fi_img.exists():
        slide7.shapes.add_picture(str(fi_img), Inches(0.8), Inches(1.8), width=Inches(6.2))

    box7 = slide7.shapes.add_textbox(Inches(7.3), Inches(1.8), Inches(5.2), Inches(5.0))
    tf7 = box7.text_frame
    tf7.word_wrap = True

    fi_points = [
        ("Top Churn Drivers (+):", "Month-to-month contracts, short tenure (<6 mo), Electronic check payments, Fiber optic high monthly charges, and Paperless billing."),
        ("Top Retention Anchors (-):", "Two-year & One-year contracts, high tenure (>24 mo), automated Credit Card/Bank payments, and active Tech Support bundles."),
        ("Explainability in UI:", "The web dashboard breaks down every customer's prediction into plain-English driver cards and visual progress meters.")
    ]
    for i, (title, desc) in enumerate(fi_points):
        p = tf7.paragraphs[0] if i == 0 else tf7.add_paragraph()
        p.text = f"•  {title} "
        p.font.size = Pt(14)
        p.font.bold = True
        p.font.color.rgb = COLOR_TEXT
        p.space_before = Pt(16)
        run = p.add_run()
        run.text = desc
        run.font.bold = False
        run.font.color.rgb = COLOR_SUBTEXT

    # -------------------------------------------------------------
    # SLIDE 8: Interactive Web Application & Workbench
    # -------------------------------------------------------------
    slide8 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide8)
    add_header(slide8, "Production Web Application & User Experience")

    box8 = slide8.shapes.add_textbox(Inches(0.8), Inches(1.8), Inches(11.5), Inches(5.0))
    tf8 = box8.text_frame
    tf8.word_wrap = True

    ui_points = [
        ("Live What-If Parameter Sliders:", "Instant debounced sliders for Tenure (0–72 mo) and Monthly Charges ($18–$120/mo) with live recalculation."),
        ("Animated Radial Risk Dial:", "SVG dial with dynamic color coding: Emerald (<35% Low Risk), Amber (35–65% Medium Risk), Crimson (>65% High Risk)."),
        ("1-Click Customer Presets:", "Pre-loaded archetypes: High-Risk Newbie, Fiber Optic At-Risk, Loyal Enterprise, and Budget Basic."),
        ("Prescriptive Retention Playbook:", "Automated next-best-action recommendations (e.g. 15% discount on 1-year contract, $10 autopay incentive)."),
        ("Executive PDF Dossier Export:", "1-Click printable executive retention assessment document for customer success teams.")
    ]
    for i, (title, desc) in enumerate(ui_points):
        p = tf8.paragraphs[0] if i == 0 else tf8.add_paragraph()
        p.text = f"•  {title} "
        p.font.size = Pt(15)
        p.font.bold = True
        p.font.color.rgb = COLOR_TEXT
        p.space_before = Pt(14)
        run = p.add_run()
        run.text = desc
        run.font.bold = False
        run.font.color.rgb = COLOR_SUBTEXT

    # -------------------------------------------------------------
    # SLIDE 9: Financial Intelligence & ARR Valuation
    # -------------------------------------------------------------
    slide9 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide9)
    add_header(slide9, "Financial Valuation & ARR Protection")

    box9 = slide9.shapes.add_textbox(Inches(0.8), Inches(1.8), Inches(11.5), Inches(5.0))
    tf9 = box9.text_frame
    tf9.word_wrap = True

    fin_points = [
        ("Translating Probability to Revenue:", "Machine learning probabilities are mapped to financial dollar metrics to justify retention expenditure."),
        ("Annual ARR at Stake ($/yr):", "Calculated as MonthlyCharges × 12 × ChurnProbability. Quantifies exact risk exposure per customer account."),
        ("Retention Value Gain ($/yr):", "Projects expected dollar recovery if the customer adopts the recommended retention intervention."),
        ("Executive Impact:", "Equips customer success directors and account managers with quantifiable ROI metrics for proactive outreach.")
    ]
    for i, (title, desc) in enumerate(fin_points):
        p = tf9.paragraphs[0] if i == 0 else tf9.add_paragraph()
        p.text = f"•  {title} "
        p.font.size = Pt(15)
        p.font.bold = True
        p.font.color.rgb = COLOR_TEXT
        p.space_before = Pt(16)
        run = p.add_run()
        run.text = desc
        run.font.bold = False
        run.font.color.rgb = COLOR_SUBTEXT

    # -------------------------------------------------------------
    # SLIDE 10: Batch CSV Analysis & Audit Logging
    # -------------------------------------------------------------
    slide10 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide10)
    add_header(slide10, "Batch Portfolio Scoring & Audit Logging")

    box10 = slide10.shapes.add_textbox(Inches(0.8), Inches(1.8), Inches(11.5), Inches(5.0))
    tf10 = box10.text_frame
    tf10.word_wrap = True

    batch_points = [
        ("High-Throughput CSV Scoring:", "Upload customer CSV spreadsheets to evaluate cohorts of hundreds or thousands of customers simultaneously."),
        ("Instant Portfolio Segmentation:", "Returns total accounts analyzed, high-risk counts, and predicted churners with downloadable scored CSV."),
        ("Relational Database Logging:", "Every prediction event is recorded using parameterized SQL queries in MySQL (with zero-downtime SQLite fallback)."),
        ("Historical Audit View (/history):", "Searchable, filterable audit interface pre-seeded with 25 realistic historical records for evaluation.")
    ]
    for i, (title, desc) in enumerate(batch_points):
        p = tf10.paragraphs[0] if i == 0 else tf10.add_paragraph()
        p.text = f"•  {title} "
        p.font.size = Pt(15)
        p.font.bold = True
        p.font.color.rgb = COLOR_TEXT
        p.space_before = Pt(16)
        run = p.add_run()
        run.text = desc
        run.font.bold = False
        run.font.color.rgb = COLOR_SUBTEXT

    # -------------------------------------------------------------
    # SLIDE 11: Production Engineering, Testing & DevOps
    # -------------------------------------------------------------
    slide11 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide11)
    add_header(slide11, "Quality Assurance, Containerization & CI/CD")

    box11 = slide11.shapes.add_textbox(Inches(0.8), Inches(1.8), Inches(11.5), Inches(5.0))
    tf11 = box11.text_frame
    tf11.word_wrap = True

    devops_points = [
        ("100% Automated Test Pass Rate:", "22 comprehensive tests in pytest covering feature encoding, model loading, API contracts, batch ingestion, and database operations."),
        ("Zero-Downtime Retraining (src/retrain.py):", "Automated champion-challenger pipeline benchmarks challenger models on new data before promotion."),
        ("Docker & Docker Compose:", "Multi-container orchestration linking official MySQL 8.0 with the Flask application via docker-compose.yml."),
        ("CI/CD Automation (.github/workflows/ci.yml):", "GitHub Actions pipeline executing automated test validation across Python 3.11 and 3.12 on every push.")
    ]
    for i, (title, desc) in enumerate(devops_points):
        p = tf11.paragraphs[0] if i == 0 else tf11.add_paragraph()
        p.text = f"•  {title} "
        p.font.size = Pt(15)
        p.font.bold = True
        p.font.color.rgb = COLOR_TEXT
        p.space_before = Pt(16)
        run = p.add_run()
        run.text = desc
        run.font.bold = False
        run.font.color.rgb = COLOR_SUBTEXT

    # -------------------------------------------------------------
    # SLIDE 12: Conclusion & Defense Summary
    # -------------------------------------------------------------
    slide12 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide12)
    add_header(slide12, "Summary & Project Impact")

    box12 = slide12.shapes.add_textbox(Inches(0.8), Inches(1.8), Inches(11.5), Inches(5.0))
    tf12 = box12.text_frame
    tf12.word_wrap = True

    concl_points = [
        ("Exceeded Performance Goals:", "Gradient Boosting champion achieved 0.8441 ROC-AUC and ~70% recall at tuned threshold (exceeding targets)."),
        ("Commercial Business Translation:", "Incorporated ARR at risk, prescriptive retention actions, and batch portfolio scoring."),
        ("Engineering Rigor:", "Production-ready Flask service, MySQL/SQLite fault tolerance, Docker containerization, and 100% test coverage."),
        ("Public Codebase:", "Available on GitHub: https://github.com/pateldhiru805-ai/Customer_Churn_prediction_System")
    ]
    for i, (title, desc) in enumerate(concl_points):
        p = tf12.paragraphs[0] if i == 0 else tf12.add_paragraph()
        p.text = f"•  {title} "
        p.font.size = Pt(15)
        p.font.bold = True
        p.font.color.rgb = COLOR_TEXT
        p.space_before = Pt(16)
        run = p.add_run()
        run.text = desc
        run.font.bold = False
        run.font.color.rgb = COLOR_SUBTEXT

    # Save presentation
    prs.save(OUTPUT_PPTX)
    print(f"[SUCCESS] Presentation generated: {OUTPUT_PPTX}")


if __name__ == "__main__":
    create_deck()
