# MARS - Medical Analysis & Report Summarizer

> **AI-powered medical report understanding system** built with Google's MedGemma and a custom fine-tuned YOLO model for the [MedGemma Impact Challenge](https://www.kaggle.com/competitions/med-gemma-impact-challenge).

---

## Demo





https://github.com/user-attachments/assets/b1408c1b-15eb-4198-adba-bd5a999829cb





---

## Features

### Core Capabilities

| Feature | Description |
|---------|-------------|
| **PDF Processing** | Upload multi-page medical PDFs (lab reports, X-rays, prescriptions, discharge summaries) |
| **Medical Image Detection** | [Custom fine-tuned YOLO v8 model](https://github.com/Redluigi1/medical_image_bounding_box_dataset) trained on 300 manually annotated images to detect embedded X-rays, CT scans, MRIs, ultrasounds |
| **Structured Data Extraction** | Extracts patient info, doctor info, diagnoses, medications, lab results into structured JSON |
| **Lab Value Analysis** | Identifies abnormal values with patient-friendly explanations via tooltips |
| **Context-Aware Processing** | Detailed summary extracted first and used as context for image descriptions and lab analysis |
| **Interactive Image Q&A** | Draw bounding boxes on scans and ask specific questions about selected regions |
| **Context-Aware Chat** | Ask follow-up questions about the entire report |

### Multi-Agent Architecture

MARS uses **14 specialized MedGemma LLM agents** + **1 CNN model** orchestrated across 5 categories:

#### Classification Agents
| # | Agent | Method | Description |
|---|-------|--------|-------------|
| 1 | **Report Page Classifier** | `is_report_page()` | Classifies page type (lab report, prescription, bill, etc.) with confidence |
| 2 | **Medical Image Confirmer** | `confirm_medical_image()` | LLM second opinion on whether a YOLO-cropped sub-image is a valid medical scan |

#### Data Extraction Agents
| # | Agent | Method | Description |
|---|-------|--------|-------------|
| 3 | **Tabular Data Extractor** | `extract_report_tabular_data()` | Extracts structured tables from report pages into JSON |
| 4 | **Lab Value Deviation Analyzer** | `analyze_lab_value_deviations()` | Flags abnormal lab values (high/low) with patient-friendly explanations |
| 5 | **Detailed Summary Extractor** | `extract_detailed_summary()` | Generates a thorough report summary; extracted first and used as context for other agents |
| 6 | **Patient & Doctor Info Extractor** | `extract_patient_and_doctor_info()` | Extracts patient name/age/sex and doctor name/phone/email from images |
| 7 | **Patient History Extractor** | `extract_patient_history_from_text()` | Extracts past conditions, allergies, and previous treatments |
| 8 | **Report Summary Extractor** | `extract_report_summary()` | Extracts findings, diagnosis, patient explanation, and recommendations |
| 9 | **Medications & Appointments Extractor** | `extract_medications_and_appointments()` | Extracts prescribed meds, dosages, frequency, and follow-up dates |
| 10 | **Medications & Appointments (Text)** | `extract_medications_and_appointments_from_text()` | Extracts prescribed meds, dosages, frequency, and follow-up dates from summary text |
| 11 | **Patient & Doctor Info (Text)** | `extract_patient_and_doctor_info_from_text()` | Extracts patient name/age/sex and doctor name/phone/email from summary text |

#### Image Agent
| # | Agent | Method | Description |
|---|-------|--------|-------------|
| 12 | **Medical Image Describer** | `describe_medical_image()` | Generates short caption + patient-friendly description using summary as context |

#### Utility Agent
| # | Agent | Method | Description |
|---|-------|--------|-------------|
| 13 | **LLM Response Cleanup** | `_cleanup_llm_response()` | Second LLM call to extract valid JSON from a malformed first response (fallback) |

#### Interactive Agent
| # | Agent | Method | Description |
|---|-------|--------|-------------|
| 14 | **Image Region Query** | `query_image_region()` | Draws bounding box on user-selected region, sends annotated image to LLM for focused analysis |

#### CNN Model (Non-LLM)
| | Model | Method | Description |
|---|-------|--------|-------------|
| -- | **YOLO Medical Image Detector** | `get_bounding_boxes()` | Fine-tuned YOLO v8 model that detects medical image regions on each PDF page |

> See the full interactive agent reference at [agents.html](agents.html).

---



## Processing Pipeline

```mermaid
flowchart TB
    subgraph Input["INPUT"]
        PDF["Medical PDF(s)"]
    end

    subgraph Step1["STEP 1: Preprocessing"]
        PDF2IMG["PDF to PNG Converter<br/><i>Converts each page to an image</i>"]
    end

    subgraph Step2["STEP 2: Extract Context First"]
        DETAILED["#5 Detailed Summary Extractor<br/><i>Generates a detailed text summary<br/>of the entire report (used as context)</i>"]
        style DETAILED fill:#fff3cd,stroke:#ffc107
    end

    subgraph Step3["STEP 3: Per-Page Processing"]
        direction TB
        PAGECLASSIFY["#1 Report Page Classifier<br/><i>Classifies each page as<br/>lab report, prescription, imaging, etc.</i>"]
        
        PAGECLASSIFY -->|"Lab/Report Page"| TABULAR["#3 Tabular Data Extractor<br/><i>Extracts structured tables<br/>from report pages</i>"]
        
        YOLO["YOLO Medical Image Detector<br/><i>Fine-tuned YOLO v8 CNN<br/>detects embedded scans</i>"]
        YOLO --> CROP["Sub-Image Extractor<br/><i>Crops detected regions</i>"]
        CROP --> CONFIRM["#2 Medical Image Confirmer<br/><i>LLM verifies cropped region<br/>is a valid medical image</i>"]
        CONFIRM -->|"Confirmed"| DESCRIBE["#12 Medical Image Describer<br/><i>Generates patient-friendly<br/>caption + description</i>"]
    end

    subgraph Step4["STEP 4: Structured Data Extraction"]
        direction TB
        PATDOC["#6 Patient & Doctor<br/>Info Extractor<br/><i>Extracts names, age, contact details</i>"]
        HISTORY["#7 Patient History Extractor<br/><i>Extracts medical history<br/>from the summary text</i>"]
        MEDS["#9 Medications & Appointments<br/>Extractor<br/><i>Extracts prescribed drugs,<br/>dosages, follow-up dates</i>"]
        SUMMARY["#8 Report Summary Extractor<br/><i>Creates a concise summary with<br/>findings, diagnosis, recommendations</i>"]
        PATDOC_TEXT["#11 Patient & Doctor Info<br/>(from summary text)"]
        MEDS_TEXT["#10 Medications & Appointments<br/>(from summary text)"]
    end

    subgraph Step5["STEP 5: Lab Analysis"]
        LABANALYZE["#4 Lab Value Deviation Analyzer<br/><i>Flags abnormal values and<br/>generates patient-friendly explanations</i>"]
    end

    subgraph Utility["UTILITY"]
        CLEANUP["#13 LLM Response Cleanup<br/><i>Fallback: extracts valid JSON<br/>from malformed LLM responses</i>"]
    end

    subgraph Outputs["THREE JSON OUTPUTS"]
        direction LR
        JSON1["report_data.json<br/>patient_info | doctor_info<br/>patient_history | report_summary<br/>medications | next_appointment"]
        JSON2["image_gallery.json<br/>medical_images[]<br/>detailed_summary"]
        JSON3["tabular_reports.json<br/>tables[]<br/>value_explanations"]
    end

    subgraph Interactive["INTERACTIVE - Runtime"]
        REGION["#14 Image Region Query<br/><i>User draws a bounding box on a scan<br/>and asks a specific question</i>"]
        CHAT["Context-Aware Chat<br/><i>Ask follow-up questions<br/>about the entire report</i>"]
    end

    %% Main Flow
    PDF --> PDF2IMG
    PDF2IMG --> DETAILED
    PDF2IMG --> PAGECLASSIFY
    PDF2IMG --> YOLO

    %% Context: detailed_summary used by
    DETAILED -.->|"provides context"| DESCRIBE
    DETAILED -.->|"input text"| HISTORY
    DETAILED -.->|"input text"| MEDS
    DETAILED -.->|"input text"| MEDS_TEXT
    DETAILED -.->|"input text"| PATDOC_TEXT

    %% Text-only alternatives
    PATDOC -.->|"alternative"| PATDOC_TEXT
    MEDS -.->|"alternative"| MEDS_TEXT

    %% Context: report_summary used by
    SUMMARY -.->|"provides context"| LABANALYZE

    %% Tabular to Lab Analysis
    TABULAR --> LABANALYZE

    %% Cleanup utility
    CLEANUP -.->|"JSON repair fallback"| TABULAR
    CLEANUP -.->|"JSON repair fallback"| PATDOC
    CLEANUP -.->|"JSON repair fallback"| SUMMARY

    %% Outputs
    PATDOC --> JSON1
    HISTORY --> JSON1
    MEDS --> JSON1
    SUMMARY --> JSON1

    DESCRIBE --> JSON2
    DETAILED --> JSON2

    TABULAR --> JSON3
    LABANALYZE --> JSON3

    %% Interactive
    JSON2 -.->|"report context"| REGION
    JSON1 -.->|"report context"| CHAT

    %% Styling
    style Step2 fill:#fff3cd,stroke:#ffc107
    style Utility fill:#fef3c7,stroke:#f59e0b
    style Outputs fill:#d4edda,stroke:#28a745
    style Interactive fill:#e3f2fd,stroke:#2196f3
```

---

## Project Structure

```
MARS/
├── api.py                      # FastAPI backend with REST endpoints
├── medical_report_processor.py # Main orchestrator (multi-agent pipeline)
├── callables.py                # YOLO + MedGemma API wrappers
├── llm_logger.py               # LLM call logging utility
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Container configuration
├── docker-compose.yml          # Docker Compose setup
│
├── fine_tune_yolo/             # YOLO model training
│   └── runs/detect/train/weights/best.pt  # Fine-tuned weights
│
└── website/                    # React frontend (Vite)
    ├── src/
    │   ├── App.jsx             # Main application component
    │   └── App.css             # Glassmorphism styling
    ├── package.json            # Node dependencies
    └── index.html              # Entry point
```

---

## Getting Started

### Prerequisites

- Python 3.10+
- Node.js 18+ (for frontend)
- Google Cloud account with Vertex AI enabled
- Poppler (for PDF processing)

### 1. Clone the Repository

```bash
git clone https://github.com/Redluigi1/MARS.git
cd MARS
```

### 2. Set Up Python Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure Environment Variables

Create a `.env` file in the root directory:

```env
# Vertex AI Configuration
PROJECT_ID = your-gcp-project-id
REGION = us-central1
ENDPOINT_ID = your-medgemma-endpoint-id

# Google Cloud Credentials
GOOGLE_APPLICATION_CREDENTIALS = key.json
```

> **Important**: You need a `key.json` file with your Google Cloud service account credentials. See [Google Cloud Authentication](https://cloud.google.com/docs/authentication/getting-started).

### 4. Install Poppler (for PDF processing)

**Windows:**
```bash
# Download from: https://github.com/oschwartz10612/poppler-windows/releases
# Add to PATH
```

**macOS:**
```bash
brew install poppler
```

**Linux:**
```bash
sudo apt-get install poppler-utils
```

### 5. Set Up Frontend

```bash
cd website
npm install
```

---

## Running the Application

### Option A: Run Separately (Development)

**Terminal 1 - Backend:**
```bash
# From root directory
python api.py
# or
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

**Terminal 2 - Frontend:**
```bash
cd website
npm run dev
```

Access the app at: `http://localhost:5173`



---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check |
| `/process-pdf` | POST | Process PDF file(s) and extract all data |
| `/predict-text` | POST | Text-only LLM prediction |
| `/predict-multimodal` | POST | Image + text LLM prediction |
| `/get-bounding-boxes` | POST | Run YOLO on image |
| `/query-image-region` | POST | Query specific region of an image |

---

## Output Format

### report_data.json
```json
{
  "patient_info": { "name": "...", "age": "...", "sex": "..." },
  "doctor_info": { "name": "...", "phone": "...", "email": "..." },
  "patient_history": "...",
  "report_summary": {
    "main_findings": "...",
    "patient_explanation": "...",
    "diagnosis": "...",
    "recommendations": "..."
  },
  "medications": [{ "name": "...", "dosage": "...", "frequency": "..." }],
  "next_appointment": "..."
}
```

### image_gallery.json
```json
{
  "medical_images": [
    {
      "image_path": "...",
      "caption": "X-ray of left clavicle",
      "description": "This X-ray shows..."
    }
  ],
  "detailed_summary": "..."
}
```

### tabular_reports.json
```json
{
  "tabular_reports": [
    {
      "page_type": "lab_report",
      "tables": [
        {
          "table_name": "Blood Test Results",
          "columns": ["Test Name", "Value", "Reference Range", "Unit"],
          "rows": [["Hemoglobin", "13.5", "13.0-17.0", "g/dL"]],
          "value_explanations": {
            "0": {
              "test_name": "WBC Count",
              "status": "Low",
              "explanation": "..."
            }
          }
        }
      ]
    }
  ]
}
```



---

## Tech Stack

| Component | Technology |
|-----------|------------|
| **Backend** | Python, FastAPI, Uvicorn |
| **Frontend** | React, Vite, Three.js (3D background) |
| **AI Model** | MedGemma (via Vertex AI), Custom Fine-tuned YOLO v8 |
| **PDF Processing** | pdf2image, Poppler |
| **Styling** | Custom CSS (Glassmorphism) |
| **Deployment** | Docker, Docker Compose |

---


## Acknowledgments

- [Google Health AI Developer Foundations (HAI-DEF)](https://developers.google.com/health-ai-developer-foundations) for MedGemma
- [Ultralytics](https://ultralytics.com/) for YOLO v8
- Built for the [MedGemma Impact Challenge](https://www.kaggle.com/competitions/med-gemma-impact-challenge)
