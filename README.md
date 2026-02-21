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
| **Medical Image Detection** | Custom fine-tuned YOLO v8 model detects and extracts embedded X-rays, CT scans, MRIs |
| **Structured Data Extraction** | Extracts patient info, doctor info, diagnoses, medications, lab results |
| **Lab Value Analysis** | Identifies abnormal values with patient-friendly explanations |
| **Interactive Image Q&A** | Draw bounding boxes on scans and ask specific questions |
| **Context-Aware Chat** | Ask follow-up questions about the entire report |

### Multi-Agent Architecture

MARS uses **7 specialized MedGemma agents** orchestrated for different tasks:

1. **Page Classifier** - Determines page type (lab report, prescription, imaging, etc.)
2. **Tabular Extractor** - Extracts structured data from tables
3. **Lab Value Analyzer** - Identifies abnormal values and explains significance
4. **Image Describer** - Generates patient-friendly image descriptions
5. **Report Summarizer** - Creates comprehensive summaries
6. **Region Query Agent** - Answers questions about specific image regions
7. **Medical Image Detector** - Custom fine-tuned YOLO for detecting embedded scans

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
        DETAILED["Report Summarizer Agent<br/><i>Generates a detailed text summary<br/>of the entire report</i>"]
        style DETAILED fill:#fff3cd,stroke:#ffc107
    end

    subgraph Step3["STEP 3: Per-Page Processing"]
        direction TB
        PAGECLASSIFY["Page Classifier Agent<br/><i>Classifies each page as<br/>lab report, prescription, imaging, etc.</i>"]
        
        PAGECLASSIFY -->|"Lab/Report Page"| TABULAR["Tabular Extractor Agent<br/><i>Extracts structured tables<br/>from report pages</i>"]
        
        YOLO["Medical Image Detector<br/><i>Fine-tuned YOLO v8 model detects<br/>embedded X-rays, CT scans, MRIs</i>"]
        YOLO --> CROP["Sub-Image Extractor<br/><i>Crops detected regions<br/>from the page</i>"]
        CROP --> CONFIRM["Image Confirmation Agent<br/><i>Verifies cropped region is<br/>a valid medical image</i>"]
        CONFIRM -->|"Confirmed"| DESCRIBE["Image Describer Agent<br/><i>Generates patient-friendly<br/>description of the scan</i>"]
    end

    subgraph Step4["STEP 4: Structured Data Extraction"]
        direction TB
        PATDOC["Patient & Doctor<br/>Info Extractor<br/><i>Extracts names, age, contact details</i>"]
        HISTORY["Patient History Extractor<br/><i>Extracts medical history<br/>from the summary text</i>"]
        MEDS["Medications & Appointments<br/>Extractor<br/><i>Extracts prescribed drugs,<br/>dosages, follow-up dates</i>"]
        SUMMARY["Report Summary Generator<br/><i>Creates a concise summary with<br/>findings, diagnosis, recommendations</i>"]
    end

    subgraph Step5["STEP 5: Lab Analysis"]
        LABANALYZE["Lab Value Analyzer Agent<br/><i>Identifies abnormal values and<br/>generates patient-friendly explanations</i>"]
    end

    subgraph Outputs["THREE JSON OUTPUTS"]
        direction LR
        JSON1["report_data.json<br/>patient_info | doctor_info<br/>patient_history | report_summary<br/>medications | next_appointment"]
        JSON2["image_gallery.json<br/>medical_images[]<br/>detailed_summary"]
        JSON3["tabular_reports.json<br/>tables[]<br/>value_explanations"]
    end

    subgraph Interactive["INTERACTIVE - Runtime"]
        REGION["Region Query Agent<br/><i>User draws a bounding box on a scan<br/>and asks a specific question</i>"]
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

    %% Context: report_summary used by
    SUMMARY -.->|"provides context"| LABANALYZE

    %% Tabular to Lab Analysis
    TABULAR --> LABANALYZE

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

## Key Features Explained

### 1. Custom Fine-tuned YOLO for Medical Image Detection
- **Manually created dataset** of 300 annotated medical report images
- Trained to detect X-rays, CT scans, MRIs, ultrasounds embedded in PDFs
- Dataset and training code: [github.com/Redluigi1/medical_image_bounding_box_dataset](https://github.com/Redluigi1/medical_image_bounding_box_dataset)

### 2. Context-Aware Processing
- **detailed_summary** extracted first and used as context for image descriptions
- **report_summary** used as context for lab value analysis
- Ensures coherent, contextually relevant outputs

### 3. Interactive Region Query
- Click and drag on any medical image to select a region
- Ask specific questions like "What is this?" or "Is this normal?"
- Backend annotates the image and sends to MedGemma for focused analysis

### 4. Abnormal Value Tooltips
- Lab values outside reference ranges are highlighted
- Hover to see patient-friendly explanations
- Powered by MedGemma's medical understanding

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
