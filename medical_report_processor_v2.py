"""
Medical Report PDF Processor V2 - Parallel API Calls
Converts PDFs to images, extracts medical images and structured data, outputs to JSON.
This version uses asyncio and ThreadPoolExecutor for parallel API calls.
"""

import os
import json
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
import re

# PDF to image conversion
try:
    from pdf2image import convert_from_path
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    PDF2IMAGE_AVAILABLE = False
    print("WARNING: pdf2image not installed. Run: pip install pdf2image")

# Import from existing callables
from callables import get_bounding_boxes, predict_text_only, predict_multimodal
from llm_logger import get_logger, LLMLogger

# Default max workers for parallel API calls
DEFAULT_MAX_WORKERS = 4


class MedicalReportProcessor:
    """Main class to process medical PDFs and extract structured data with parallel API calls."""
    
    def __init__(self, output_dir: str = None, max_workers: int = DEFAULT_MAX_WORKERS):
        """
        Initialize the processor.
        
        Args:
            output_dir: Directory to save output files and images
            max_workers: Maximum number of parallel API calls (default: 4)
        """
        if output_dir is None:
            output_dir = Path(__file__).parent / "output"
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        self.images_dir = self.output_dir / "pdf_images"
        self.medical_images_dir = self.output_dir / "medical_images"
        self.images_dir.mkdir(exist_ok=True)
        self.medical_images_dir.mkdir(exist_ok=True)
        
        # Initialize logger
        self.logger = get_logger(str(self.output_dir / "logs"))
        
        # Store results
        self.report_data = {}
        self.image_gallery = {"medical_images": []}
        self.detailed_summary = ""  # Extracted first, used as context for image descriptions
        self.tabular_reports = []  # Store extracted tabular data from report pages
        self.page_image_map = {}  # Maps page_num to image_path for context
        
        # Parallel processing settings
        self.max_workers = max_workers
        self._lock = asyncio.Lock() if asyncio.get_event_loop().is_running() else None
        import threading
        self._thread_lock = threading.Lock()  # Thread-safe lock for concurrent access
    
    # ==================== PDF TO IMAGE CONVERSION ====================
    
    def convert_pdf_to_images(self, pdf_path: str) -> list:
        """
        Convert a PDF to images (one per page).
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of paths to generated images
        """
        if not PDF2IMAGE_AVAILABLE:
            raise ImportError("pdf2image is required. Install with: pip install pdf2image")
        
        pdf_path = Path(pdf_path)
        pdf_name = pdf_path.stem
        
        self.logger.logger.info(f"Converting PDF: {pdf_path}")
        
        # Convert PDF to images
        images = convert_from_path(str(pdf_path), dpi=200)
        
        image_paths = []
        for i, img in enumerate(images):
            img_path = self.images_dir / f"{pdf_name}_page_{i+1}.png"
            img.save(str(img_path), "PNG")
            image_paths.append(str(img_path))
            self.logger.logger.info(f"Saved page {i+1}: {img_path}")
        
        return image_paths
    
    # ==================== MEDICAL IMAGE DETECTION ====================
    
    def _call_llm_with_logging(self, prompt: str, image_paths: list = None, 
                                function_name: str = "generic_call",
                                system_prompt: str = "You are an expert medical imaging analyst."):
        """Helper to call LLM with logging."""
        start_time = time.time()
        
        try:
            if image_paths:
                result = predict_multimodal(prompt, image_paths, system_prompt=system_prompt)
            else:
                result = predict_text_only(prompt, system_prompt=system_prompt)
            
            duration = time.time() - start_time
            
            # Extract response text
            response_text = self._extract_response_text(result)
            
            self.logger.log_call(
                function_name=function_name,
                prompt=prompt,
                image_paths=image_paths,
                response=response_text,
                duration_seconds=duration,
                success=True
            )
            
            return response_text
            
        except Exception as e:
            duration = time.time() - start_time
            self.logger.log_call(
                function_name=function_name,
                prompt=prompt,
                image_paths=image_paths,
                duration_seconds=duration,
                success=False,
                error=str(e)
            )
            raise
    
    def _extract_response_text(self, result) -> str:
        """Extract text from LLM response."""
        if result is None:
            return ""
        if isinstance(result, str):
            return result
        
        # Handle dict response directly (Vertex AI format)
        if isinstance(result, dict):
            try:
                # Structure: {'choices': [{'message': {'content': 'actual text'}}]}
                if 'choices' in result:
                    return result['choices'][0]['message']['content']
                # Alternative structure
                if 'candidates' in result:
                    return result['candidates'][0]['content']['parts'][0]['text']
            except (KeyError, IndexError, TypeError):
                pass
            return str(result)
        
        # Handle list response
        if isinstance(result, list) and len(result) > 0:
            first = result[0]
            if isinstance(first, dict):
                # Handle the actual response structure from Vertex AI
                # Structure: {'choices': [{'message': {'content': 'actual text'}}]}
                try:
                    # Try the Vertex AI chat completions format first
                    if 'choices' in first:
                        return first['choices'][0]['message']['content']
                    # Fallback to candidates format
                    return first.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', str(first))
                except (KeyError, IndexError, TypeError):
                    return str(first)
            return str(first)
        return str(result)
    
    def is_full_page_medical_image(self, image_path: str) -> bool:
        """
        Ask LLM if >90% of the image is a medical scan (X-ray/CT/MRI).
        
        Args:
            image_path: Path to the PDF page image
            
        Returns:
            True if the image is primarily a medical scan
        """
        prompt = """You are analyzing a scanned PDF page. Your task is to determine if this page is PRIMARILY a medical imaging scan (like an X-ray, CT scan, MRI, or ultrasound image).

IMPORTANT: Look at the ENTIRE page carefully.

Answer 'YES' ONLY if:
- More than 90% of the page shows a greyscale medical scan (bones, organs, etc.)
- There is minimal text (just patient name/labels at corners/edges)
- The scan itself fills most of the page
- Examples: Full-page chest X-ray, brain MRI scan, CT slice

Answer 'NO' if:
- The page contains blood test results, lab reports, or tables with numbers
- The page has significant text content (paragraphs, forms, instructions)
- The page is a medical bill, invoice, or prescription
- The page has embedded small medical images within text/forms
- The page is mostly white/blank with some medical text with no scan image
- Examples: Lab report with test values, doctor's notes, medication list, billing statement

Think step-by-step:
1. What do I see? (Describe briefly)
2. Is it primarily a scan image or primarily text/forms?
3. Final answer: YES or NO

Format your response as:
ANALYSIS: [brief 1-sentence description]
ANSWER: YES or NO"""

        response = self._call_llm_with_logging(
            prompt=prompt,
            image_paths=[image_path],
            function_name="is_full_page_medical_image",
            system_prompt="You are a medical imaging classifier. Analyze carefully and provide accurate YES/NO answers."
        )
        
        # Extract the answer more robustly
        response_upper = response.upper()
        
        # Try to find explicit ANSWER: line first
        answer_match = re.search(r'ANSWER:\s*(YES|NO)', response_upper)
        if answer_match:
            return answer_match.group(1) == 'YES'
        
        # Fallback: Look for YES or NO but be more strict
        # Count occurrences to avoid false positives from analysis text
        yes_count = response_upper.count('YES')
        no_count = response_upper.count('NO')
        
        # If only one clear answer appears, use it
        if yes_count == 1 and no_count == 0:
            return True
        elif no_count >= 1 and yes_count == 0:
            return False
        
        # If ambiguous, look at the last line which should have the final answer
        lines = response_upper.strip().split('\n')
        if lines:
            last_line = lines[-1].strip()
            if 'YES' in last_line and 'NO' not in last_line:
                return True
            elif 'NO' in last_line:
                return False
        
        # Default to NO if unclear (conservative approach - avoid false positives)
        self.logger.logger.warning(f"Ambiguous response from is_full_page_medical_image: {response[:200]}")
        return False
    
    def has_embedded_medical_images(self, image_path: str) -> bool:
        """
        Ask LLM if the page contains any embedded medical images among text.
        
        Args:
            image_path: Path to the PDF page image
            
        Returns:
            True if the page contains embedded medical images
        """
        prompt = """Analyze this medical report page. Does this page contain any embedded medical imaging scans 
(such as X-rays, CT scans, MRI images, ultrasound images, or other diagnostic images) 
mixed with text or within a report format?

Answer with ONLY 'YES' or 'NO'.
- Answer 'YES' if there are any visible medical scans/images embedded in the document.
- Answer 'NO' if the page only contains text, tables, or forms without any medical imaging."""

        response = self._call_llm_with_logging(
            prompt=prompt,
            image_paths=[image_path],
            function_name="has_embedded_medical_images"
        )
        
        return "YES" in response.upper()
    
    def is_report_page(self, image_path: str) -> dict:
        """
        Classify if a page is a structured report (lab results, prescriptions, etc.).
        This is the FIRST stage of the two-stage report extraction process.
        
        Args:
            image_path: Path to the PDF page image
            
        Returns:
            Dict with 'is_report' (bool), 'report_type' (str), and 'confidence' (str)
        """
        prompt = """Analyze this medical document page and determine if it is a STRUCTURED REPORT containing tabular data.

A STRUCTURED REPORT includes:
- Lab test results with values in tables or lists
- medical bills/money receipts or invoices
- Blood work / pathology reports with test names and numerical values
- Prescription lists with medications, dosages
- Discharge summaries with structured sections
- Radiology/imaging reports with findings in structured format
- Vital signs or monitoring data in tabular form


Respond in this EXACT format:
IS_REPORT: YES or NO
REPORT_TYPE: [lab_report / prescription / radiology_report / discharge_summary / vitals_report / medical_bill / other_report]
CONFIDENCE: [high / medium / low]
REASON: [1 sentence explanation]"""

        response = self._call_llm_with_logging(
            prompt=prompt,
            image_paths=[image_path],
            function_name="is_report_page",
            system_prompt="You are a medical document classifier. Analyze carefully and classify accurately."
        )
        
        result = {
            "is_report": False,
            "report_type": "not_applicable",
            "confidence": "low",
            "reason": ""
        }
        
        response_upper = response.upper()
        
        # Parse IS_REPORT - more robust: check if YES/NO appears before REPORT_TYPE
        # This handles variations like ISMV:, IS_REPORT:, IS_STRUCTURED_REPORT:, etc.
        is_report_match = re.search(r'IS_REPORT:\s*(YES|NO)', response_upper)
        if is_report_match:
            result["is_report"] = is_report_match.group(1) == 'YES'
        else:
            # Fallback: check if YES appears before REPORT_TYPE (handles ISMV: YES, etc.)
            report_type_pos = response_upper.find('REPORT_TYPE')
            if report_type_pos > 0:
                text_before_report_type = response_upper[:report_type_pos]
                if 'YES' in text_before_report_type:
                    result["is_report"] = True
                elif 'NO' in text_before_report_type:
                    result["is_report"] = False
        
        # Parse REPORT_TYPE
        type_match = re.search(r'REPORT_TYPE:\s*(\S+)', response, re.IGNORECASE)
        if type_match:
            result["report_type"] = type_match.group(1).lower().strip()
        
        # Parse CONFIDENCE
        conf_match = re.search(r'CONFIDENCE:\s*(HIGH|MEDIUM|LOW)', response_upper)
        if conf_match:
            result["confidence"] = conf_match.group(1).lower()
        
        # Parse REASON
        reason_match = re.search(r'REASON:\s*(.+?)(?:\n|$)', response, re.IGNORECASE)
        if reason_match:
            result["reason"] = reason_match.group(1).strip()
        
        self.logger.logger.info(f"Page classified: is_report={result['is_report']}, type={result['report_type']}, confidence={result['confidence']}")
        return result
    
    def extract_report_tabular_data(self, image_path: str, report_type: str, source_pdf: str, page_num: int) -> dict:
        """
        Specialized agent to extract tabular data from confirmed report pages.
        This is the SECOND stage of the two-stage report extraction process.
        
        The output format uses explicit rows and columns for robust downstream processing.
        
        Args:
            image_path: Path to the PDF page image
            report_type: Type of report (lab_report, prescription, etc.)
            source_pdf: Name of source PDF
            page_num: Page number
            
        Returns:
            Dict with structured tabular data
        """
        self.logger.logger.info(f"Extracting tabular data from {report_type} (page {page_num})...")
        
        prompt = f"""You are a specialized medical report data extractor. This page is a {report_type}.

Extract ALL tabular/structured data from this page into a standardized format.

CRITICAL RULES:
1. Extract EVERY table, list, or structured section you can find
2. Preserve the exact values, units, and reference ranges
3. For each table, define the column headers and then list each row
4. Mark abnormal values with status "High", "Low", or "Abnormal" if indicated
5. If a field is not present, use null

Respond with ONLY valid JSON in this EXACT format:
{{
    "page_type": "{report_type}",
    "tables": [
        {{
            "table_name": "Name/title of the table or section",
            "columns": ["Column1", "Column2", "Column3", ...],
            "rows": [
                ["row1_col1", "row1_col2", "row1_col3", ...],
                ["row2_col1", "row2_col2", "row2_col3", ...]
            ]
        }}
    ],
    "metadata": {{
        "report_date": "date if found or null",
        "lab_name": "laboratory/hospital name or null",
        "patient_id": "patient ID if found or null",
        "doctor_name": "ordering doctor name or null",
        "sample_type": "blood/urine/etc or null",
        "collection_time": "sample collection time or null"
    }}
}}

COMMON COLUMN STRUCTURES by report type:
- Lab Report: ["Test Name", "Value", "Unit", "Reference Range", "Status"]
- Prescription: ["Medication", "Dosage", "Frequency", "Duration", "Instructions"]
- Vitals Report: ["Parameter", "Value", "Unit", "Time", "Status"]
- Radiology Report: ["Finding", "Location", "Description", "Impression"]

Output ONLY the JSON object, no other text or markdown."""

        response = self._call_llm_with_logging(
            prompt=prompt,
            image_paths=[image_path],
            function_name="extract_report_tabular_data",
            system_prompt="You are an expert medical data extraction system. Output ONLY valid JSON with precise tabular data."
        )
        
        # Define expected schema for cleanup fallback
        expected_schema = """{
    "page_type": "string",
    "tables": [
        {
            "table_name": "string",
            "columns": ["string", ...],
            "rows": [["string", ...], ...]
        }
    ],
    "metadata": {
        "report_date": "string or null",
        "lab_name": "string or null",
        "patient_id": "string or null",
        "doctor_name": "string or null",
        "sample_type": "string or null",
        "collection_time": "string or null"
    }
}"""
        
        # Parse the response with expected_schema to enable LLM cleanup fallback
        parsed_data = self._parse_json_from_llm_response(response, "extract_report_tabular_data", expected_schema)
        
        # Add source information
        parsed_data["source_pdf"] = source_pdf
        parsed_data["page_number"] = page_num
        parsed_data["image_path"] = image_path
        
        # Store in tabular_reports
        if "error" not in parsed_data:
            self.tabular_reports.append(parsed_data)
            self.logger.logger.info(f"Successfully extracted {len(parsed_data.get('tables', []))} table(s) from page {page_num}")
        else:
            self.logger.logger.warning(f"Error extracting tabular data from page {page_num}: {parsed_data.get('error')}")
        
        return parsed_data
    
    def analyze_lab_value_deviations(self, tabular_report: dict, report_summary: str = None) -> dict:
        """
        Analyze lab report tables and add explanations for abnormal values.
        Uses LLM to identify values outside reference ranges and explain their significance.
        
        Args:
            tabular_report: A tabular report dict containing tables with lab data
            report_summary: The report summary text to provide context
            
        Returns:
            Updated tabular_report with 'value_explanations' field added to each table
        """
        # Only process lab reports
        if tabular_report.get("page_type") != "lab_report":
            return tabular_report
        
        tables = tabular_report.get("tables", [])
        if not tables:
            return tabular_report
        
        self.logger.logger.info(f"Analyzing lab value deviations for {len(tables)} table(s)...")
        
        # Build context
        context_section = ""
        if report_summary:
            context_section = f"""
PATIENT CONTEXT (from medical report summary):
{report_summary}
"""
        
        # Process each table
        for table_idx, table in enumerate(tables):
            table_name = table.get("table_name", f"Table {table_idx + 1}")
            columns = table.get("columns", [])
            rows = table.get("rows", [])
            
            if not rows or not columns:
                continue
            
            # Format table data for LLM
            table_text = f"Table: {table_name}\n"
            table_text += "Columns: " + " | ".join(columns) + "\n"
            table_text += "Rows:\n"
            for row_idx, row in enumerate(rows):
                table_text += f"  Row {row_idx}: " + " | ".join(str(cell) for cell in row) + "\n"
            
            prompt = f"""You are a medical lab analyst. Analyze this lab report table and identify ANY values that are ABNORMAL (outside the reference range - either HIGH or LOW).
{context_section}
LAB DATA:
{table_text}

TASK: For each abnormal value, provide a brief patient-friendly explanation of:
1. Whether the value is HIGH or LOW
2. What this means for the patient's health (potential causes or implications)
3. Keep explanations concise (2-3 sentences max)

OUTPUT FORMAT - Return ONLY valid JSON:
{{
    "explanations": {{
        "row_index": {{
            "test_name": "Name of the test",
            "status": "High" or "Low",
            "value": "The actual value",
            "reference_range": "The reference range if available",
            "explanation": "Brief patient-friendly explanation"
        }}
    }}
}}

If ALL values are normal, return: {{"explanations": {{}}}}

IMPORTANT:
- Only include rows with ABNORMAL values
- Use row indices as keys (0-indexed)
- Output ONLY the JSON object, no other text"""

            response = self._call_llm_with_logging(
                prompt=prompt,
                image_paths=None,  # Text-only call
                function_name="analyze_lab_value_deviations",
                system_prompt="You are a medical lab analyst. Identify abnormal lab values and provide brief, accurate explanations. Output ONLY valid JSON."
            )
            
            expected_schema = """{
    "explanations": {
        "row_index": {
            "test_name": "string",
            "status": "High or Low",
            "value": "string",
            "reference_range": "string or null",
            "explanation": "string"
        }
    }
}"""
            
            parsed_result = self._parse_json_from_llm_response(response, "analyze_lab_value_deviations", expected_schema)
            
            # Add explanations to the table
            if "explanations" in parsed_result and parsed_result["explanations"]:
                table["value_explanations"] = parsed_result["explanations"]
                num_abnormal = len(parsed_result["explanations"])
                self.logger.logger.info(f"Found {num_abnormal} abnormal value(s) in {table_name}")
            else:
                table["value_explanations"] = {}
                self.logger.logger.info(f"All values normal in {table_name}")
        
        return tabular_report
    
    def confirm_medical_image(self, image_path: str) -> bool:
        """
        Confirm if a cropped sub-image is indeed a medical image.
        
        Args:
            image_path: Path to the cropped image
            
        Returns:
            True if confirmed as medical image
        """
        prompt = """Is this image a medical diagnostic scan (eg : X-ray, CT scan, MRI, ultrasound, etc.)?

Answer with ONLY 'YES' or 'NO'. Answer 'NO' if the image is not a medical diagnostic scan."""

        response = self._call_llm_with_logging(
            prompt=prompt,
            image_paths=[image_path],
            function_name="confirm_medical_image"
        )
        
        return "YES" in response.upper()
    
    def describe_medical_image(self, image_path: str, is_full_page: bool = False, 
                                page_image_path: str = None, max_retries: int = 2) -> dict:
        """
        Get a patient-friendly description and caption of a medical image.
        Uses the detailed summary as context, and for sub-images also uses the full page.
        Uses JSON output format for robust parsing with retry logic.
        
        Args:
            image_path: Path to the medical image
            is_full_page: Whether this is a full-page scan (or sub-image)
            page_image_path: Path to the original PDF page (for sub-image context)
            max_retries: Maximum number of retries if JSON parsing fails
            
        Returns:
            Dict with 'caption' and 'description' keys
        """
        # Build context-aware prompt
        context_section = ""
        if self.detailed_summary:
            context_section = f"""\nCONTEXT FROM MEDICAL REPORT:
{self.detailed_summary}\n"""
        
        prompt = f"""You are explaining this medical image to a patient.
{context_section}
Based on the image and the context above, provide the following:

1. caption: A very short image caption (maximum 8 words).
2. description: A detailed patient-friendly explanation (3-5 sentences) that identifies the scan type, body part shown, and what the image shows related to the patient's condition. Use simple language.

CRITICAL: Output ONLY valid JSON in this EXACT format:
{{
    "caption": "Your caption here (max 8 words)",
    "description": "Your detailed patient-friendly description here (3-5 sentences)."
}}

Do NOT include any thinking process, instructions, meta-commentary, or text outside the JSON object."""

        # For sub-images, include the full page as additional context
        image_paths = [image_path]
        if not is_full_page and page_image_path:
            image_paths.append(page_image_path)
        
        expected_schema = """{
    "caption": "string (max 8 words)",
    "description": "string (3-5 sentences, patient-friendly)"
}"""
        
        # Retry loop for robust parsing
        for attempt in range(max_retries + 1):
            response = self._call_llm_with_logging(
                prompt=prompt,
                image_paths=image_paths,
                function_name="describe_medical_image",
                system_prompt="You are a medical professional. Output ONLY valid JSON with the requested caption and description. Do not include any text outside the JSON object."
            )
            
            # Parse JSON response
            parsed_result = self._parse_json_from_llm_response(
                response, 
                "describe_medical_image", 
                expected_schema
            )
            
            # Check if parsing was successful
            if "error" not in parsed_result:
                # Validate that we have both required fields
                caption = parsed_result.get("caption", "").strip()
                description = parsed_result.get("description", "").strip()
                
                if caption and description:
                    # Clean up the caption (ensure max 8 words)
                    caption_words = caption.split()
                    if len(caption_words) > 8:
                        caption = ' '.join(caption_words[:8])
                    
                    # Ensure description ends with a period
                    if description and not description.endswith('.'):
                        description += '.'
                    
                    self.logger.logger.info(f"Successfully parsed image description (attempt {attempt + 1})")
                    return {
                        "caption": caption,
                        "description": description
                    }
                else:
                    self.logger.logger.warning(f"Parsed JSON missing required fields (attempt {attempt + 1})")
            else:
                self.logger.logger.warning(f"JSON parsing failed (attempt {attempt + 1}): {parsed_result.get('error', 'Unknown error')}")
            
            # If not the last attempt, log retry
            if attempt < max_retries:
                self.logger.logger.info(f"Retrying describe_medical_image (attempt {attempt + 2}/{max_retries + 1})...")
        
        # All retries failed - return fallback with error info
        self.logger.logger.error("All attempts to parse image description failed, returning fallback")
        return {
            "caption": "Medical scan image",
            "description": "This is a medical diagnostic image. Please consult with your healthcare provider for a detailed explanation of what this image shows."
        }
    
    def extract_sub_images(self, image_path: str, detections: list, source_pdf: str, page_num: int) -> list:
        """
        Crop sub-images using bounding boxes from YOLO.
        
        Args:
            image_path: Path to the PDF page image
            detections: List of detection dicts with 'bbox' [x1, y1, x2, y2] and 'confidence'
            source_pdf: Name of source PDF
            page_num: Page number
            
        Returns:
            List of paths to cropped images
        """
        img = Image.open(image_path)
        cropped_paths = []
        
        for i, detection in enumerate(detections):
            bbox = detection['bbox']
            confidence = detection.get('confidence', 0)
            x1, y1, x2, y2 = [int(coord) for coord in bbox]
            
            # Ensure coordinates are within image bounds
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(img.width, x2)
            y2 = min(img.height, y2)
            
            # Crop the image
            cropped = img.crop((x1, y1, x2, y2))
            
            # Save cropped image
            crop_filename = f"{source_pdf}_page{page_num}_crop{i+1}.png"
            crop_path = self.medical_images_dir / crop_filename
            cropped.save(str(crop_path), "PNG")
            cropped_paths.append(str(crop_path))
            
            self.logger.logger.info(f"Cropped sub-image: {crop_path} (confidence: {confidence:.2f})")
        
        return cropped_paths
    
    def process_page_for_medical_images(self, image_path: str, source_pdf: str, page_num: int):
        """
        Process a single PDF page for medical images AND structured reports.
        Directly runs CNN to detect medical images and confirms with LLM.
        
        Logic:
        1. Run CNN (YOLO) on every page to detect potential medical images
        2. Confirm each detected region with LLM using confirm_medical_image
        3. If structured report -> extract tabular data with specialized agent
        
        Args:
            image_path: Path to the PDF page image
            source_pdf: Name of source PDF
            page_num: Page number
        """
        self.logger.logger.info(f"Processing page {page_num} for medical images...")
        
        page_results = {"page_num": page_num, "medical_images": [], "tabular_data": None}
        
        # Store page image path for context when describing sub-images
        self.page_image_map[page_num] = image_path
        
        # === TWO-STAGE REPORT PROCESSING ===
        # Stage 1: Classify if this is a structured report page
        report_classification = self.is_report_page(image_path)
        
        if report_classification["is_report"]:
            # Stage 2: Extract tabular data using specialized agent
            self.logger.logger.info(f"Page {page_num}: Classified as {report_classification['report_type']} (confidence: {report_classification['confidence']})")
            self.extract_report_tabular_data(
                image_path, 
                report_classification["report_type"], 
                source_pdf, 
                page_num
            )
            # Note: We still continue to check for medical images in the same page
            # as some reports may contain embedded images (e.g., radiology reports with X-rays)
        
        # === MEDICAL IMAGE PROCESSING ===
        # Directly run CNN on every page to detect medical images
        self.logger.logger.info(f"Page {page_num}: Running CNN to detect medical images...")
        
        try:
            # Use YOLO to find bounding boxes (with overlapping box filtering enabled)
            detections = get_bounding_boxes(image_path)
            self.logger.logger.info(f"Found {len(detections)} potential medical image regions")
            
            if detections:
                # Extract sub-images using detection dicts
                cropped_paths = self.extract_sub_images(image_path, detections, source_pdf, page_num)
                
                # Confirm each sub-image is a medical image using LLM
                for crop_path in cropped_paths:
                    if self.confirm_medical_image(crop_path):
                        # Get description with context (sub-image = include page context)
                        desc_result = self.describe_medical_image(
                            crop_path, 
                            is_full_page=False, 
                            page_image_path=image_path
                        )
                        self.image_gallery["medical_images"].append({
                            "image_path": crop_path,
                            "type": "extracted_image",
                            "source_pdf": source_pdf,
                            "page_number": page_num,
                            "caption": desc_result["caption"],
                            "description": desc_result["description"]
                        })
                    else:
                        self.logger.logger.info(f"Cropped image not confirmed as medical: {crop_path}")
                        # Delete non-medical crops
                        os.remove(crop_path)
            else:
                self.logger.logger.info(f"Page {page_num}: No medical images detected by CNN")
        except Exception as e:
            self.logger.logger.error(f"Error running CNN on page {page_num}: {e}")
        
        return page_results
    
    def _process_single_crop(self, crop_path: str, image_path: str, source_pdf: str, page_num: int) -> dict:
        """
        Process a single cropped image - confirm and describe if medical.
        Thread-safe helper for parallel processing.
        
        Returns:
            Dict with image info if confirmed as medical, None otherwise
        """
        try:
            if self.confirm_medical_image(crop_path):
                desc_result = self.describe_medical_image(
                    crop_path, 
                    is_full_page=False, 
                    page_image_path=image_path
                )
                return {
                    "image_path": crop_path,
                    "type": "extracted_image",
                    "source_pdf": source_pdf,
                    "page_number": page_num,
                    "caption": desc_result["caption"],
                    "description": desc_result["description"]
                }
            else:
                self.logger.logger.info(f"Cropped image not confirmed as medical: {crop_path}")
                os.remove(crop_path)
                return None
        except Exception as e:
            self.logger.logger.error(f"Error processing crop {crop_path}: {e}")
            return None
    
    def process_page_for_medical_images_parallel(self, image_path: str, source_pdf: str, page_num: int) -> dict:
        """
        Process a single PDF page for medical images AND structured reports.
        PARALLEL VERSION: Uses ThreadPoolExecutor for concurrent crop processing.
        
        Logic:
        1. Run CNN (YOLO) on every page to detect potential medical images
        2. Confirm each detected region with LLM using confirm_medical_image IN PARALLEL
        3. If structured report -> extract tabular data with specialized agent
        
        Args:
            image_path: Path to the PDF page image
            source_pdf: Name of source PDF
            page_num: Page number
            
        Returns:
            Dict with page processing results
        """
        self.logger.logger.info(f"[PARALLEL] Processing page {page_num} for medical images...")
        
        page_results = {"page_num": page_num, "medical_images": [], "tabular_data": None}
        
        # Store page image path for context when describing sub-images
        with self._thread_lock:
            self.page_image_map[page_num] = image_path
        
        # === TWO-STAGE REPORT PROCESSING ===
        # Stage 1: Classify if this is a structured report page
        report_classification = self.is_report_page(image_path)
        
        if report_classification["is_report"]:
            # Stage 2: Extract tabular data using specialized agent
            self.logger.logger.info(f"Page {page_num}: Classified as {report_classification['report_type']} (confidence: {report_classification['confidence']})")
            tabular_data = self.extract_report_tabular_data(
                image_path, 
                report_classification["report_type"], 
                source_pdf, 
                page_num
            )
            page_results["tabular_data"] = tabular_data
        
        # === MEDICAL IMAGE PROCESSING (PARALLEL) ===
        self.logger.logger.info(f"Page {page_num}: Running CNN to detect medical images...")
        
        try:
            detections = get_bounding_boxes(image_path)
            self.logger.logger.info(f"Found {len(detections)} potential medical image regions")
            
            if detections:
                cropped_paths = self.extract_sub_images(image_path, detections, source_pdf, page_num)
                
                # Process all crops in parallel
                with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    futures = {
                        executor.submit(
                            self._process_single_crop, 
                            crop_path, image_path, source_pdf, page_num
                        ): crop_path 
                        for crop_path in cropped_paths
                    }
                    
                    for future in as_completed(futures):
                        result = future.result()
                        if result:
                            page_results["medical_images"].append(result)
                            with self._thread_lock:
                                self.image_gallery["medical_images"].append(result)
            else:
                self.logger.logger.info(f"Page {page_num}: No medical images detected by CNN")
        except Exception as e:
            self.logger.logger.error(f"Error running CNN on page {page_num}: {e}")
        
        return page_results
    
    # ==================== STRUCTURED DATA EXTRACTION ====================
    
    def extract_detailed_summary(self, image_paths: list):
        """
        Extract a detailed summary of the medical report.
        This is extracted FIRST and used as context for image descriptions.
        
        Args:
            image_paths: List of paths to PDF page images
        """
        self.logger.logger.info("Extracting detailed summary for context...")
        
        prompt = """Analyze these medical report pages and provide a DETAILED SUMMARY.

Include:
1. What medical condition/issue is being reported
2. What tests/procedures were done
3. What the findings show
4. Any treatments or surgeries that were performed
5. The current status and prognosis
6. Any other relevant information related to the patient

Be thorough but avoid excessive medical jargon.
"""

        self.detailed_summary = self._call_llm_with_logging(
            prompt=prompt,
            image_paths=image_paths,
            function_name="extract_detailed_summary",
            system_prompt="You are a medical professional explaining these reports."
        )
        
        self.logger.logger.info(f"Detailed summary extracted: {len(self.detailed_summary)} chars")
    
    def _parse_json_from_llm_response(self, response: str, function_name: str, expected_schema: str = None) -> dict:
        """
        Helper method to robustly parse JSON from LLM response.
        Handles both single JSON objects and arrays of objects.
        If parsing fails and expected_schema is provided, uses LLM cleanup as fallback.
        
        Args:
            response: Raw LLM response text
            function_name: Name of the calling function for logging
            expected_schema: Optional JSON schema string for LLM cleanup fallback
            
        Returns:
            Parsed JSON dict, or error dict if parsing failed
        """
        try:
            cleaned_response = response
            
            # First, strip any thinking tokens from the response
            cleaned_response = self._strip_thinking_tokens(cleaned_response)
            
            # Remove HTML-like tags (e.g., </br>, <br>)
            cleaned_response = re.sub(r'<[^>]+>', '', cleaned_response)
            
            # Remove markdown code blocks (support both {objects} and [arrays])
            json_candidates = []
            
            # Try to extract JSON from markdown code blocks first
            if "```json" in cleaned_response or "```" in cleaned_response:
                # Find code block boundaries and extract content between them
                code_block_pattern = re.compile(r'```(?:json)?\s*([\s\S]*?)\s*```', re.DOTALL)
                for match in code_block_pattern.finditer(cleaned_response):
                    block_content = match.group(1).strip()
                    # Use brace matching to extract complete JSON from the block
                    if block_content.startswith('{') or block_content.startswith('['):
                        # Find the complete JSON object using brace counting
                        brace_count = 0
                        bracket_count = 0
                        start_idx = 0
                        end_idx = len(block_content)
                        for i, char in enumerate(block_content):
                            if char == '{':
                                brace_count += 1
                            elif char == '}':
                                brace_count -= 1
                                if brace_count == 0 and bracket_count == 0:
                                    end_idx = i + 1
                                    break
                            elif char == '[':
                                bracket_count += 1
                            elif char == ']':
                                bracket_count -= 1
                                if bracket_count == 0 and brace_count == 0:
                                    end_idx = i + 1
                                    break
                        json_candidates.append(block_content[start_idx:end_idx])
            
            # If no code blocks found, try to find raw JSON structures using brace matching
            if not json_candidates:
                # Find the start of JSON objects or arrays and use brace counting
                for i, char in enumerate(cleaned_response):
                    if char in ('{', '['):
                        # Use brace matching to find the complete JSON
                        brace_count = 0
                        bracket_count = 0
                        for j in range(i, len(cleaned_response)):
                            c = cleaned_response[j]
                            if c == '{':
                                brace_count += 1
                            elif c == '}':
                                brace_count -= 1
                                if brace_count == 0 and bracket_count == 0:
                                    json_candidates.append(cleaned_response[i:j+1])
                                    break
                            elif c == '[':
                                bracket_count += 1
                            elif c == ']':
                                bracket_count -= 1
                                if bracket_count == 0 and brace_count == 0:
                                    json_candidates.append(cleaned_response[i:j+1])
                                    break
                        # Only try to find the first complete JSON structure
                        if json_candidates:
                            break
            
            # Remove comments (// ...) which are invalid in standard JSON
            cleaned_candidates = []
            for candidate in json_candidates:
                cleaned = re.sub(r'//.*$', '', candidate, flags=re.MULTILINE)
                cleaned_candidates.append(cleaned)
            
            # Try to parse each candidate
            parsed_json = None
            for candidate in cleaned_candidates:
                try:
                    # Try direct parsing
                    parsed = json.loads(candidate)
                    
                    # If LLM returned an array, extract the first element
                    if isinstance(parsed, list):
                        if len(parsed) > 0:
                            self.logger.logger.warning(f"{function_name}: LLM returned array instead of object, using first element")
                            if isinstance(parsed[0], dict):
                                parsed_json = parsed[0]
                                break
                        else:
                            continue
                    else:
                        parsed_json = parsed
                        break
                        
                except json.JSONDecodeError as e:
                    # Try to repair common JSON errors
                    repaired = self._repair_json(candidate)
                    if repaired:
                        try:
                            parsed = json.loads(repaired)
                            if isinstance(parsed, list) and len(parsed) > 0 and isinstance(parsed[0], dict):
                                parsed_json = parsed[0]
                            else:
                                parsed_json = parsed
                            self.logger.logger.info(f"{function_name}: Successfully repaired malformed JSON")
                            break
                        except json.JSONDecodeError:
                            continue
            
            # Check if we got a valid, non-empty result
            if parsed_json and isinstance(parsed_json, dict):
                # Check if the result is meaningful (not just {} or error-like)
                # For extraction functions, we expect certain keys
                is_empty_or_trivial = len(parsed_json) == 0
                has_only_error = list(parsed_json.keys()) == ["error"] or list(parsed_json.keys()) == ["raw_response"]
                
                if not is_empty_or_trivial and not has_only_error:
                    return parsed_json
                else:
                    self.logger.logger.warning(f"{function_name}: Parsed JSON is empty or trivial, attempting cleanup...")
            
            # Fallback: Use LLM cleanup if we have expected_schema
            if expected_schema:
                self.logger.logger.info(f"{function_name}: Primary parsing failed, attempting LLM cleanup...")
                cleanup_result = self._cleanup_llm_response(response, expected_schema, function_name)
                if cleanup_result and isinstance(cleanup_result, dict) and len(cleanup_result) > 0:
                    return cleanup_result
            
            self.logger.logger.error(f"{function_name}: Could not find valid JSON in response")
            return {"error": "No valid JSON found in response", "raw_response": response[:500]}
                
        except Exception as e:
            self.logger.logger.error(f"{function_name}: Unexpected error during JSON parsing: {e}")
            error_context = response[:500] if len(response) > 500 else response
            return {"error": f"Parsing error: {str(e)}", "raw_response": error_context}
    
    def _repair_json(self, json_str: str) -> str:
        """
        Attempt to repair common JSON syntax errors.
        
        Args:
            json_str: Potentially malformed JSON string
            
        Returns:
            Repaired JSON string, or None if unrepairable
        """
        try:
            # Remove trailing commas before closing braces/brackets
            repaired = re.sub(r',\s*([}\]])', r'\1', json_str)
            
            # Fix missing commas between properties (common LLM error)
            # Pattern: "value"\n    "key" should be "value",\n    "key"
            repaired = re.sub(r'"\s*\n\s*"', '",\n    "', repaired)
            
            # Fix missing commas after closing braces
            # Pattern: }\n    { should be },\n    {
            repaired = re.sub(r'}\s*\n\s*{', '},\n    {', repaired)
            
            # Fix unquoted null/true/false/numbers before a quoted string
            repaired = re.sub(r'(null|true|false|\d+)\s*\n\s*"', r'\1,\n    "', repaired)
            
            return repaired
        except Exception:
            return None
    
    def _strip_thinking_tokens(self, text: str) -> str:
        """
        Remove common thinking tokens and markers from LLM response.
        These tokens appear when the LLM outputs its internal reasoning.
        
        Args:
            text: Raw LLM response text
            
        Returns:
            Text with thinking tokens stripped
        """
        cleaned = text
        
        # Remove <unusedXX>thought...content patterns (common in some models)
        cleaned = re.sub(r'<unused\d+>thought.*?(?=\{|$)', '', cleaned, flags=re.DOTALL | re.IGNORECASE)
        cleaned = re.sub(r'<unused\d+>.*?<unused\d+>', '', cleaned, flags=re.DOTALL)
        
        # Remove thinking block markers
        cleaned = re.sub(r'<\|thinking\|>.*?<\|/thinking\|>', '', cleaned, flags=re.DOTALL)
        cleaned = re.sub(r'<thinking>.*?</thinking>', '', cleaned, flags=re.DOTALL | re.IGNORECASE)
        
        # Remove **thought or *thought patterns
        cleaned = re.sub(r'\*+thought.*?(?=\{|```|$)', '', cleaned, flags=re.DOTALL | re.IGNORECASE)
        
        # Remove lines that start with numbered thinking steps (1. Identify..., 2. Extract...)
        cleaned = re.sub(r'^\s*\d+\.\s+\*\*[^*]+\*\*:.*$', '', cleaned, flags=re.MULTILINE)
        
        return cleaned.strip()
    
    def _cleanup_llm_response(self, raw_response: str, expected_schema: str, function_name: str) -> dict:
        """
        Use a second LLM call to clean up and extract JSON from a messy first response.
        This handles cases where the LLM returns errors, refusals, or malformed responses
        but the actual data might still be extractable from the text.
        
        Args:
            raw_response: The raw response from the first LLM call
            expected_schema: Description of the expected JSON schema
            function_name: Name of the calling function for logging
            
        Returns:
            Parsed JSON dict, or None if cleanup failed
        """
        self.logger.logger.info(f"{function_name}: Attempting LLM cleanup of response...")
        
        # Truncate extremely long or garbled responses to avoid token limits
        truncated_response = raw_response[:3000] if len(raw_response) > 3000 else raw_response
        
        cleanup_prompt = f"""CRITICAL: Output ONLY a valid JSON object. Do NOT include any thinking, explanations, or text before or after the JSON.

The text below is a messy AI response that was supposed to extract medical data. Extract ALL medical information you can find and output it as clean JSON.

REQUIRED OUTPUT FORMAT (output EXACTLY this structure):
{expected_schema}

RULES:
1. Output ONLY the JSON object - nothing else, no markdown, no explanation
2. Extract ALL medical data you can find (diagnoses, findings, treatments, etc.)
3. If a field is not found, use null
4. Do NOT refuse or explain - just output the JSON
5. Do not leave out any information from the text

TEXT TO EXTRACT FROM:
{truncated_response}

JSON OUTPUT:"""

        try:
            cleaned_response = self._call_llm_with_logging(
                prompt=cleanup_prompt,
                image_paths=[],  # Text-only cleanup
                function_name=f"{function_name}_cleanup",
                system_prompt="Output ONLY valid JSON. No thinking, no explanation, no markdown code blocks - just the raw JSON object starting with { and ending with }."
            )
            
            # First, strip any thinking tokens that might have leaked through
            stripped_response = self._strip_thinking_tokens(cleaned_response)
            
            # Try to parse the cleaned response
            # First try: direct parse if response is pure JSON
            try:
                if stripped_response.strip().startswith('{'):
                    parsed = json.loads(stripped_response.strip())
                    if isinstance(parsed, dict) and len(parsed) > 0:
                        self.logger.logger.info(f"{function_name}: LLM cleanup successful (direct parse)!")
                        return parsed
            except json.JSONDecodeError:
                pass
            
            # Second try: extract JSON from markdown code blocks
            json_code_blocks = re.findall(r'```(?:json)?\s*(\{[\s\S]*?\})\s*```', stripped_response)
            for block in json_code_blocks:
                try:
                    parsed = json.loads(block)
                    if isinstance(parsed, dict) and len(parsed) > 0:
                        self.logger.logger.info(f"{function_name}: LLM cleanup successful (code block)!")
                        return parsed
                except json.JSONDecodeError:
                    continue
            
            # Third try: find the outermost JSON object using brace matching
            brace_count = 0
            start_idx = None
            for i, char in enumerate(stripped_response):
                if char == '{':
                    if brace_count == 0:
                        start_idx = i
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0 and start_idx is not None:
                        json_str = stripped_response[start_idx:i+1]
                        try:
                            parsed = json.loads(json_str)
                            if isinstance(parsed, dict) and len(parsed) > 0:
                                self.logger.logger.info(f"{function_name}: LLM cleanup successful (brace matching)!")
                                return parsed
                        except json.JSONDecodeError:
                            pass
                        break
            
            # Fourth try: regex extraction as last resort
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', stripped_response)
            if json_match:
                try:
                    parsed = json.loads(json_match.group())
                    if isinstance(parsed, dict) and len(parsed) > 0:
                        self.logger.logger.info(f"{function_name}: LLM cleanup successful (regex)!")
                        return parsed
                except json.JSONDecodeError:
                    pass
            
            self.logger.logger.warning(f"{function_name}: LLM cleanup did not produce valid JSON")
            return None
            
        except Exception as e:
            self.logger.logger.error(f"{function_name}: LLM cleanup failed with error: {e}")
            return None
    
    def extract_patient_and_doctor_info(self, image_paths: list) -> dict:
        """
        Extract patient info and doctor info.
        Focused call for demographic and contact information.
        Note: patient_history is extracted separately using extract_patient_history_from_text.
        
        Args:
            image_paths: List of paths to PDF page images
            
        Returns:
            Dictionary with patient_info, doctor_info
        """
        self.logger.logger.info("Extracting patient and doctor information...")
        
        prompt = """You are extracting DEMOGRAPHIC INFORMATION from medical documents.

TASK: Find and extract ONLY the patient's personal details and doctor's contact information.

WHAT TO EXTRACT:
- patient_info.name: The PATIENT's full name (the person receiving medical care)
- patient_info.age: The patient's age in years 
- patient_info.sex: Male, Female, or Other
- doctor_info.name: The DOCTOR's name (usually has "Dr." prefix)
- doctor_info.phone: Doctor's phone number
- doctor_info.email: Doctor's email address

WHAT TO IGNORE:
- Hospital/clinic names and addresses (these are NOT the patient or doctor name)
- Bill amounts, invoice numbers, bank details
- Test results and findings (those go elsewhere)
- Past medical history (that is extracted separately)

Respond with ONLY this JSON (no other text):
{
    "patient_info": {
        "name": "patient's full name or null",
        "age": "age in years or null",
        "sex": "Male/Female/Other or null"
    },
    "doctor_info": {
        "name": "doctor's name (with Dr. prefix) or null",
        "phone": "phone number or null",
        "email": "email or null"
    }
}"""

        response = self._call_llm_with_logging(
            prompt=prompt,
            image_paths=image_paths,
            function_name="extract_patient_and_doctor_info",
            system_prompt="You are a medical records specialist. Extract ONLY demographic data. Do NOT include addresses, hospital names, or financial information."
        )
        
        expected_schema = """{
    "patient_info": {"name": "string or null", "age": "string or null", "sex": "string or null"},
    "doctor_info": {"name": "string or null", "phone": "string or null", "email": "string or null"}
}"""
        
        return self._parse_json_from_llm_response(response, "extract_patient_and_doctor_info", expected_schema)
    
    def extract_patient_history_from_text(self, text: str) -> dict:
        """
        Extract patient history (past medical conditions, allergies, previous treatments)
        from the detailed summary text.
        This is a dedicated LLM call for patient history extraction.
        
        Args:
            text: The detailed summary text extracted from the medical report
            
        Returns:
            Dictionary with patient_history
        """
        self.logger.logger.info("Extracting patient history from detailed summary...")
        
        prompt = f"""You are extracting PATIENT MEDICAL HISTORY from the following medical document summary.

DOCUMENT SUMMARY:
{text}

TASK: Find and extract ALL information about the patient's medical history.

WHAT TO EXTRACT (include ALL that apply):
- Past medical conditions (e.g., diabetes, hypertension, heart disease, asthma)
- Known allergies (medications, food, environmental)
- Previous surgeries or procedures
- Previous treatments for any conditions
- Chronic conditions
- Family medical history if mentioned
- Previous hospitalizations
- Pre-existing conditions

WHAT TO IGNORE:
- Current diagnosis (that's the new condition being reported)
- Current treatment plan (that's the new treatment)
- Test results and findings (those go elsewhere)
- Doctor/patient demographic info

Respond with ONLY this JSON (no other text):
{{
    "patient_history": "comprehensive description of past medical conditions, allergies, and previous treatments, or null if none mentioned"
}}"""

        response = self._call_llm_with_logging(
            prompt=prompt,
            image_paths=None,  # Text-only call
            function_name="extract_patient_history_from_text",
            system_prompt="You are a medical records specialist. Extract ONLY past medical history information. Be thorough - include all allergies, previous conditions, and prior treatments mentioned."
        )
        
        expected_schema = """{
    "patient_history": "string or null"
}"""
        
        return self._parse_json_from_llm_response(response, "extract_patient_history_from_text", expected_schema)
    
    def extract_report_summary(self, image_paths: list) -> dict:
        """
        Extract report summary with findings, diagnosis, and recommendations.
        Focused call for medical content summarization.
        
        Args:
            image_paths: List of paths to PDF page images
            
        Returns:
            Dictionary with report_summary
        """
        self.logger.logger.info("Extracting report summary...")
        
        prompt = """CRITICAL: Output ONLY a valid JSON object. Do NOT include any text before or after the JSON.

The images are of a patients medical report. Read the medical documents and create a clear, helpful summary for a patient.

FIELD DEFINITIONS:

1. main_findings: A brief medical summary (2-3 sentences about what was found).

2. patient_explanation: A detailed explanation covering:
   - What condition/injury was found
   - What tests or imaging were done (X-ray, CT scan, blood test, etc.)
   - What the test results showed
   - What treatment was done (surgery, medication, etc.)
   - Current status and expected recovery time
   Use simple, non-medical language a patient can understand.

3. diagnosis: The specific medical condition or diagnosis 

4. recommendations: What the doctor advises for follow-up.

WHAT TO IGNORE: Hospital names, addresses, bill amounts, invoice numbers.

Output ONLY this JSON structure (nothing else - no explanations, no thinking):
{
    "report_summary": {
        "main_findings": "brief medical findings summary",
        "patient_explanation": "detailed patient-friendly explanation",
        "diagnosis": "specific medical diagnosis",
        "recommendations": "doctor's advice for follow-up"
    }
}

REMEMBER: Output ONLY the JSON object above. Start with { and end with }. No other text."""

        response = self._call_llm_with_logging(
            prompt=prompt,
            image_paths=image_paths,
            function_name="extract_report_summary",
            system_prompt="Output ONLY valid JSON. No thinking, no explanations. Extract medical findings into the exact JSON format requested. Start your response with { and end with }."
        )
        
        expected_schema = """{
    "report_summary": {"main_findings": "string or null", "patient_explanation": "string or null", "diagnosis": "string or null", "recommendations": "string or null"}
}"""
        
        return self._parse_json_from_llm_response(response, "extract_report_summary", expected_schema)
    
    def extract_medications_and_appointments(self, image_paths: list) -> dict:
        """
        Extract medications and next appointment information.
        Focused call for prescription and scheduling data.
        
        Args:
            image_paths: List of paths to PDF page images
            
        Returns:
            Dictionary with medications and next_appointment
        """
        self.logger.logger.info("Extracting medications and appointments...")
        
        prompt = """You are extracting PRESCRIPTION and APPOINTMENT information from medical documents.

TASK: Find any prescribed medications and scheduled follow-up appointments.

WHAT TO EXTRACT:

1. medications: List of prescribed drugs/medicines
   - name: The medicine/drug name (e.g., "Paracetamol", "Amoxicillin", "Calcium supplement")
   - dosage: Amount per dose (e.g., "500mg", "10ml")
   - frequency: How often to take (e.g., "twice daily", "every 8 hours", "after meals")
   - duration: How long to take (e.g., "7 days", "2 weeks", "till next visit")

2. next_appointment: When to return for follow-up
   GOOD EXAMPLE: "15th February 2024" or "After 6 weeks" or "In 2 months"

WHAT TO IGNORE:
- Diagnosis or condition names (those are NOT medications)
- Medical equipment or implants (plates, screws, etc. are NOT medications)
- Bill items and costs
- Test names

If no medications are prescribed, return an empty array [].

Respond with ONLY this JSON (no other text):
{
    "medications": [
        {
            "name": "medicine name",
            "dosage": "amount per dose or null",
            "frequency": "how often or null",
            "duration": "how long or null"
        }
    ],
    "next_appointment": "follow-up date/time or null"
}"""

        response = self._call_llm_with_logging(
            prompt=prompt,
            image_paths=image_paths,
            function_name="extract_medications_and_appointments",
            system_prompt="You are a pharmacist extracting prescription details. Only include actual medications (drugs/medicines), not diagnoses or medical devices."
        )
        
        expected_schema = """{
    "medications": [{"name": "string", "dosage": "string or null", "frequency": "string or null", "duration": "string or null"}],
    "next_appointment": "string or null"
}"""
        
        return self._parse_json_from_llm_response(response, "extract_medications_and_appointments", expected_schema)
    
    def extract_medications_and_appointments_from_text(self, text: str) -> dict:
        """
        Extract medications and appointments from pre-extracted text.
        This is a cost-saving alternative that uses text-only LLM call.
        
        Args:
            text: Combined text extracted from all PDF pages
            
        Returns:
            Dictionary with medications and next_appointment
        """
        self.logger.logger.info("Extracting medications and appointments from text...")
        
        prompt = f"""You are extracting PRESCRIPTION and APPOINTMENT information from the following medical document text.

DOCUMENT TEXT:
{text}

TASK: Find any prescribed medications and scheduled follow-up appointments.

WHAT TO EXTRACT:

1. medications: List of prescribed drugs/medicines
   - name: The medicine/drug name (e.g., "Paracetamol", "Amoxicillin", "Calcium supplement")
   - dosage: Amount per dose (e.g., "500mg", "10ml")
   - frequency: How often to take (e.g., "twice daily", "every 8 hours", "after meals")
   - duration: How long to take (e.g., "7 days", "2 weeks", "till next visit")

2. next_appointment: When to return for follow-up
   GOOD EXAMPLE: "15th February 2024" or "After 6 weeks" or "In 2 months"

WHAT TO IGNORE:
- Diagnosis or condition names (those are NOT medications)
- Medical equipment or implants (plates, screws, etc. are NOT medications)
- Bill items and costs
- Test names

If no medications are prescribed, return an empty array [].

Respond with ONLY this JSON (no other text):
{{
    "medications": [
        {{
            "name": "medicine name",
            "dosage": "amount per dose or null",
            "frequency": "how often or null",
            "duration": "how long or null"
        }}
    ],
    "next_appointment": "follow-up date/time or null"
}}"""

        response = self._call_llm_with_logging(
            prompt=prompt,
            image_paths=None,  # Text-only call
            function_name="extract_medications_and_appointments_from_text",
            system_prompt="You are a pharmacist extracting prescription details. Only include actual medications (drugs/medicines), not diagnoses or medical devices."
        )
        
        expected_schema = """{
    "medications": [{"name": "string", "dosage": "string or null", "frequency": "string or null", "duration": "string or null"}],
    "next_appointment": "string or null"
}"""
        
        return self._parse_json_from_llm_response(response, "extract_medications_and_appointments_from_text", expected_schema)
    
    def extract_patient_and_doctor_info_from_text(self, text: str) -> dict:
        """
        Extract patient and doctor info from pre-extracted text.
        This is a cost-saving alternative that uses text-only LLM call.
        
        Args:
            text: Combined text extracted from all PDF pages
            
        Returns:
            Dictionary with patient_info, doctor_info, patient_history
        """
        self.logger.logger.info("Extracting patient and doctor information from text...")
        
        prompt = f"""You are extracting DEMOGRAPHIC INFORMATION from the following medical document text.

DOCUMENT TEXT:
{text}

TASK: Find and extract ONLY the patient's personal details and doctor's contact information.

WHAT TO EXTRACT:
- patient_info.name: The PATIENT's full name (the person receiving medical care)
- patient_info.age: The patient's age in years
- patient_info.sex: Male, Female, or Other
- doctor_info.name: The DOCTOR's name (usually has "Dr." prefix)
- doctor_info.phone: Doctor's phone number
- doctor_info.email: Doctor's email address
- patient_history: Any mentioned past medical conditions, allergies, or previous treatments

WHAT TO IGNORE:
- Hospital/clinic names and addresses (these are NOT the patient or doctor name)
- Bill amounts, invoice numbers, bank details
- Test results and findings (those go elsewhere)
- Dates of visits (unless it's medical history)

Respond with ONLY this JSON (no other text):
{{
    "patient_info": {{
        "name": "patient's full name or null",
        "age": "age in years or null",
        "sex": "Male/Female/Other or null"
    }},
    "doctor_info": {{
        "name": "doctor's name (with Dr. prefix) or null",
        "phone": "phone number or null",
        "email": "email or null"
    }},
    "patient_history": "past medical conditions/allergies mentioned or null"
}}"""

        response = self._call_llm_with_logging(
            prompt=prompt,
            image_paths=None,  # Text-only call
            function_name="extract_patient_and_doctor_info_from_text",
            system_prompt="You are a medical records specialist. Extract ONLY demographic data. Do NOT include addresses, hospital names, or financial information."
        )
        
        expected_schema = """{
    "patient_info": {"name": "string or null", "age": "string or null", "sex": "string or null"},
    "doctor_info": {"name": "string or null", "phone": "string or null", "email": "string or null"},
    "patient_history": "string or null"
}"""
        
        return self._parse_json_from_llm_response(response, "extract_patient_and_doctor_info_from_text", expected_schema)
    
    def extract_all_structured_data(self, image_paths: list, first_page_images: list = None):
        """
        Extract all structured data from PDF.
        
        Uses:
        - Image-based extraction for patient/doctor info (from ALL page images)
        - Detailed summary text for medications/appointments extraction
        
        Note: Lab results are extracted via the two-stage report extraction process
        and saved to tabular_reports.json, so they are not duplicated here.
        
        Args:
            image_paths: List of paths to PDF page images
            first_page_images: Optional list of first page images from each PDF (deprecated, now uses all images)
        """
        # Use ALL images for patient/doctor info extraction (not just first page)
        # This provides more comprehensive extraction from multi-page reports
        
        # Extract patient/doctor info from ALL page images
        self.logger.logger.info("Starting patient/doctor info extraction from ALL PAGE IMAGES...")
        patient_doctor_data = self.extract_patient_and_doctor_info(image_paths)
        
        # Retry once if patient name is null (to improve reliability)
        patient_info = patient_doctor_data.get("patient_info", {})
        if patient_info is None or patient_info.get("name") is None:
            self.logger.logger.info("Patient name is null, retrying extraction once more...")
            patient_doctor_data = self.extract_patient_and_doctor_info(image_paths)
        
        # Extract patient history from the detailed summary (dedicated LLM call)
        patient_history_data = {"patient_history": None}
        if self.detailed_summary:
            self.logger.logger.info(f"Using detailed_summary for patient history extraction ({len(self.detailed_summary)} chars)")
            patient_history_data = self.extract_patient_history_from_text(self.detailed_summary)
        else:
            self.logger.logger.warning("No detailed summary available for patient history extraction")
        
        # Extract medications and appointments from the detailed summary
        if self.detailed_summary:
            self.logger.logger.info(f"Using detailed_summary for medications extraction ({len(self.detailed_summary)} chars)")
            meds_data = self.extract_medications_and_appointments_from_text(self.detailed_summary)
        else:
            # Fallback: Use image-based extraction if no detailed summary available
            self.logger.logger.warning("No detailed summary available, falling back to image-based extraction for medications...")
            meds_data = self.extract_medications_and_appointments(image_paths)
        
        # Extract report summary (still uses images for visual context)
        summary_data = self.extract_report_summary(image_paths)
        
        # Note: Lab results are NOT extracted here - they are captured via the
        # two-stage report extraction (is_report_page -> extract_report_tabular_data)
        # and saved to tabular_reports.json
        
        # Merge all the data together
        self.report_data = {
            "patient_info": patient_doctor_data.get("patient_info", None),
            "doctor_info": patient_doctor_data.get("doctor_info", None),
            "patient_history": patient_history_data.get("patient_history", None),
            "report_summary": summary_data.get("report_summary", None),
            "medications": meds_data.get("medications", []),
            "next_appointment": meds_data.get("next_appointment", None)
        }
        
        # Log if any errors occurred
        if "error" in patient_doctor_data:
            self.logger.logger.warning(f"Error in patient/doctor info extraction: {patient_doctor_data.get('error')}")
            self.report_data["extraction_errors"] = self.report_data.get("extraction_errors", {})
            self.report_data["extraction_errors"]["patient_doctor_info"] = patient_doctor_data.get("error")
        
        if "error" in summary_data:
            self.logger.logger.warning(f"Error in report summary extraction: {summary_data.get('error')}")
            self.report_data["extraction_errors"] = self.report_data.get("extraction_errors", {})
            self.report_data["extraction_errors"]["report_summary"] = summary_data.get("error")
        
        if "error" in meds_data:
            self.logger.logger.warning(f"Error in medications extraction: {meds_data.get('error')}")
            self.report_data["extraction_errors"] = self.report_data.get("extraction_errors", {})
            self.report_data["extraction_errors"]["medications"] = meds_data.get("error")
        
        self.logger.logger.info("Structured data extraction complete")
    
    def extract_all_structured_data_parallel(self, image_paths: list, first_page_images: list = None):
        """
        Extract all structured data from PDF using PARALLEL API calls.
        
        This version runs independent extractions concurrently:
        - Patient/doctor info extraction (from ALL page images)
        - Patient history extraction (from detailed_summary)
        - Medications extraction (from detailed_summary)
        - Report summary extraction
        
        Args:
            image_paths: List of paths to PDF page images
            first_page_images: Optional list of first page images from each PDF (deprecated, now uses all images)
        """
        self.logger.logger.info("[PARALLEL] Starting parallel structured data extraction...")
        
        # Use ALL images for patient/doctor info extraction (not just first page)
        # This provides more comprehensive extraction from multi-page reports
        
        # Define extraction tasks
        def extract_patient_doctor():
            result = self.extract_patient_and_doctor_info(image_paths)
            # Retry once if patient name is null
            patient_info = result.get("patient_info", {})
            if patient_info is None or patient_info.get("name") is None:
                self.logger.logger.info("[PARALLEL] Patient name null, retrying...")
                result = self.extract_patient_and_doctor_info(image_paths)
            return ("patient_doctor", result)
        
        def extract_history():
            if self.detailed_summary:
                return ("history", self.extract_patient_history_from_text(self.detailed_summary))
            return ("history", {"patient_history": None})
        
        def extract_meds():
            if self.detailed_summary:
                return ("meds", self.extract_medications_and_appointments_from_text(self.detailed_summary))
            return ("meds", self.extract_medications_and_appointments(image_paths))
        
        def extract_summary():
            return ("summary", self.extract_report_summary(image_paths))
        
        # Run all extractions in parallel
        results = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [
                executor.submit(extract_patient_doctor),
                executor.submit(extract_history),
                executor.submit(extract_meds),
                executor.submit(extract_summary)
            ]
            
            for future in as_completed(futures):
                try:
                    key, value = future.result()
                    results[key] = value
                    self.logger.logger.info(f"[PARALLEL] Completed extraction: {key}")
                except Exception as e:
                    self.logger.logger.error(f"[PARALLEL] Extraction failed: {e}")
        
        # Merge results
        patient_doctor_data = results.get("patient_doctor", {})
        patient_history_data = results.get("history", {})
        meds_data = results.get("meds", {})
        summary_data = results.get("summary", {})
        
        self.report_data = {
            "patient_info": patient_doctor_data.get("patient_info", None),
            "doctor_info": patient_doctor_data.get("doctor_info", None),
            "patient_history": patient_history_data.get("patient_history", None),
            "report_summary": summary_data.get("report_summary", None),
            "medications": meds_data.get("medications", []),
            "next_appointment": meds_data.get("next_appointment", None)
        }
        
        # Log errors
        for name, data in [("patient_doctor_info", patient_doctor_data), 
                           ("report_summary", summary_data), 
                           ("medications", meds_data)]:
            if "error" in data:
                self.logger.logger.warning(f"Error in {name}: {data.get('error')}")
                self.report_data.setdefault("extraction_errors", {})[name] = data.get("error")
        
        self.logger.logger.info("[PARALLEL] Structured data extraction complete")
    
    # ==================== MAIN PIPELINE ====================
    
    def process_pdfs(self, pdf_paths: list) -> dict:
        """
        Main pipeline to process multiple PDFs.
        
        Args:
            pdf_paths: List of paths to PDF files
            
        Returns:
            Dictionary with report_data and image_gallery paths
        """
        all_page_images = []
        first_page_images = []  # Store first page from each PDF for patient/doctor info
        pdf_to_pages = {}  # Store pages per PDF for processing
        
        # Step 1: Convert all PDFs to images first
        for pdf_path in pdf_paths:
            self.logger.logger.info(f"Processing PDF: {pdf_path}")
            pdf_name = Path(pdf_path).stem
            
            page_images = self.convert_pdf_to_images(pdf_path)
            pdf_to_pages[pdf_name] = page_images
            all_page_images.extend(page_images)
            
            # Collect first page from each PDF for patient/doctor info extraction
            if page_images:
                first_page_images.append(page_images[0])
        
        # Step 2: Extract detailed summary FIRST (needed as context for image descriptions)
        self.extract_detailed_summary(all_page_images)
        
        # Step 3: Now process each page for medical images (with summary context available)
        for pdf_name, page_images in pdf_to_pages.items():
            for i, img_path in enumerate(page_images):
                self.process_page_for_medical_images(img_path, pdf_name, i + 1)
        
        # Step 4: Extract structured data (uses first pages only for patient/doctor info)
        self.extract_all_structured_data(all_page_images, first_page_images)
        
        # Step 5: Analyze lab value deviations and add explanations
        # Use report_summary as context for the LLM
        report_summary_text = None
        if self.report_data.get("report_summary"):
            rs = self.report_data["report_summary"]
            # Combine relevant summary fields for context
            summary_parts = []
            if rs.get("main_findings"):
                summary_parts.append(f"Main findings: {rs['main_findings']}")
            if rs.get("diagnosis"):
                summary_parts.append(f"Diagnosis: {rs['diagnosis']}")
            if rs.get("patient_explanation"):
                summary_parts.append(f"Overview: {rs['patient_explanation']}")
            report_summary_text = "\n".join(summary_parts) if summary_parts else None
        
        self.logger.logger.info(f"Analyzing {len(self.tabular_reports)} tabular report(s) for abnormal lab values...")
        for tabular_report in self.tabular_reports:
            self.analyze_lab_value_deviations(tabular_report, report_summary_text)
        
        # Step 6: Save outputs
        report_path = self.output_dir / "report_data.json"
        gallery_path = self.output_dir / "image_gallery.json"
        tabular_path = self.output_dir / "tabular_reports.json"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(self.report_data, f, indent=2, ensure_ascii=False)
        
        # Add detailed summary to gallery output
        self.image_gallery["detailed_summary"] = self.detailed_summary
        
        with open(gallery_path, 'w', encoding='utf-8') as f:
            json.dump(self.image_gallery, f, indent=2, ensure_ascii=False)
        
        # Save tabular reports from two-stage extraction
        with open(tabular_path, 'w', encoding='utf-8') as f:
            json.dump({"tabular_reports": self.tabular_reports}, f, indent=2, ensure_ascii=False)
        
        self.logger.logger.info(f"Report data saved to: {report_path}")
        self.logger.logger.info(f"Image gallery saved to: {gallery_path}")
        self.logger.logger.info(f"Tabular reports saved to: {tabular_path}")
        
        # Print summary
        summary = self.logger.get_summary()
        self.logger.logger.info(f"Processing complete! LLM call summary: {summary}")
        
        return {
            "report_data_path": str(report_path),
            "image_gallery_path": str(gallery_path),
            "tabular_reports_path": str(tabular_path),
            "report_data": self.report_data,
            "image_gallery": self.image_gallery,
            "tabular_reports": self.tabular_reports,
            "llm_summary": summary
        }
    
    def analyze_lab_value_deviations_parallel(self, report_summary: str = None):
        """
        Analyze all lab reports for deviations IN PARALLEL.
        
        Args:
            report_summary: Report summary text for context
        """
        lab_reports = [r for r in self.tabular_reports if r.get("page_type") == "lab_report"]
        
        if not lab_reports:
            return
        
        self.logger.logger.info(f"[PARALLEL] Analyzing {len(lab_reports)} lab report(s) for deviations...")
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self.analyze_lab_value_deviations, report, report_summary): report
                for report in lab_reports
            }
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    self.logger.logger.error(f"[PARALLEL] Lab analysis failed: {e}")
    
    def process_pdfs_parallel(self, pdf_paths: list) -> dict:
        """
        Main pipeline to process multiple PDFs with PARALLEL API calls.
        
        Parallelization points:
        1. Pages processed in parallel (each page independently)
        2. Structured data extractions run in parallel
        3. Lab value deviation analysis runs in parallel
        
        Args:
            pdf_paths: List of paths to PDF files
            
        Returns:
            Dictionary with report_data and image_gallery paths
        """
        self.logger.logger.info(f"[PARALLEL] Starting parallel processing of {len(pdf_paths)} PDF(s)...")
        
        all_page_images = []
        first_page_images = []
        page_tasks = []  # List of (image_path, pdf_name, page_num)
        
        # Step 1: Convert all PDFs to images first (sequential - I/O bound)
        for pdf_path in pdf_paths:
            self.logger.logger.info(f"Converting PDF: {pdf_path}")
            pdf_name = Path(pdf_path).stem
            page_images = self.convert_pdf_to_images(pdf_path)
            all_page_images.extend(page_images)
            
            if page_images:
                first_page_images.append(page_images[0])
            
            for i, img_path in enumerate(page_images):
                page_tasks.append((img_path, pdf_name, i + 1))
        
        # Step 2: Extract detailed summary FIRST (needs to complete before image descriptions)
        self.extract_detailed_summary(all_page_images)
        
        # Step 3: Process all pages for medical images IN PARALLEL
        self.logger.logger.info(f"[PARALLEL] Processing {len(page_tasks)} pages in parallel...")
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(
                    self.process_page_for_medical_images_parallel,
                    img_path, pdf_name, page_num
                ): (pdf_name, page_num)
                for img_path, pdf_name, page_num in page_tasks
            }
            for future in as_completed(futures):
                pdf_name, page_num = futures[future]
                try:
                    result = future.result()
                    self.logger.logger.info(f"[PARALLEL] Completed page {page_num} of {pdf_name}")
                except Exception as e:
                    self.logger.logger.error(f"[PARALLEL] Failed page {page_num} of {pdf_name}: {e}")
        
        # Step 4: Extract structured data IN PARALLEL
        self.extract_all_structured_data_parallel(all_page_images, first_page_images)
        
        # Step 5: Analyze lab value deviations IN PARALLEL
        report_summary_text = None
        if self.report_data.get("report_summary"):
            rs = self.report_data["report_summary"]
            summary_parts = []
            if rs.get("main_findings"):
                summary_parts.append(f"Main findings: {rs['main_findings']}")
            if rs.get("diagnosis"):
                summary_parts.append(f"Diagnosis: {rs['diagnosis']}")
            if rs.get("patient_explanation"):
                summary_parts.append(f"Overview: {rs['patient_explanation']}")
            report_summary_text = "\n".join(summary_parts) if summary_parts else None
        
        self.analyze_lab_value_deviations_parallel(report_summary_text)
        
        # Step 6: Save outputs
        report_path = self.output_dir / "report_data.json"
        gallery_path = self.output_dir / "image_gallery.json"
        tabular_path = self.output_dir / "tabular_reports.json"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(self.report_data, f, indent=2, ensure_ascii=False)
        
        self.image_gallery["detailed_summary"] = self.detailed_summary
        
        with open(gallery_path, 'w', encoding='utf-8') as f:
            json.dump(self.image_gallery, f, indent=2, ensure_ascii=False)
        
        with open(tabular_path, 'w', encoding='utf-8') as f:
            json.dump({"tabular_reports": self.tabular_reports}, f, indent=2, ensure_ascii=False)
        
        self.logger.logger.info(f"Report data saved to: {report_path}")
        self.logger.logger.info(f"Image gallery saved to: {gallery_path}")
        self.logger.logger.info(f"Tabular reports saved to: {tabular_path}")
        
        summary = self.logger.get_summary()
        self.logger.logger.info(f"[PARALLEL] Processing complete! LLM call summary: {summary}")
        
        return {
            "report_data_path": str(report_path),
            "image_gallery_path": str(gallery_path),
            "tabular_reports_path": str(tabular_path),
            "report_data": self.report_data,
            "image_gallery": self.image_gallery,
            "tabular_reports": self.tabular_reports,
            "llm_summary": summary
        }

# ==================== IMAGE REGION QUERY FUNCTIONS ====================

def annotate_image_with_bounding_box(image_path: str, bbox: dict, output_path: str = None) -> str:
    """
    Annotate an image with a red bounding box highlighting the region of interest.
    
    Args:
        image_path: Path to the original image
        bbox: Dictionary with keys 'x', 'y', 'width', 'height' (in pixels, relative to display size)
               and optionally 'scale_x', 'scale_y' for scaling to natural dimensions
        output_path: Optional path to save annotated image. If None, saves to temp file.
    
    Returns:
        Path to the annotated image
    """
    # Open the image
    img = Image.open(image_path)
    draw = ImageDraw.Draw(img, 'RGBA')
    
    # Get bounding box coordinates
    x = bbox.get('x', 0)
    y = bbox.get('y', 0)
    width = bbox.get('width', 100)
    height = bbox.get('height', 100)
    
    # Scale if needed (frontend sends coordinates relative to displayed size)
    scale_x = bbox.get('scale_x', 1.0)
    scale_y = bbox.get('scale_y', 1.0)
    
    x = int(x * scale_x)
    y = int(y * scale_y)
    width = int(width * scale_x)
    height = int(height * scale_y)
    
    # Draw semi-transparent red fill
    draw.rectangle(
        [x, y, x + width, y + height],
        fill=(255, 0, 0, 77)  # Semi-transparent red (30% opacity)
    )
    
    # Draw red border (6px thick)
    border_width = 6
    draw.rectangle(
        [x, y, x + width, y + height],
        outline=(255, 0, 0, 255),
        width=border_width
    )
    
    # Draw corner markers (circles at corners)
    marker_radius = 12
    corners = [
        (x, y),
        (x + width, y),
        (x, y + height),
        (x + width, y + height)
    ]
    for cx, cy in corners:
        draw.ellipse(
            [cx - marker_radius, cy - marker_radius, cx + marker_radius, cy + marker_radius],
            fill=(255, 0, 0, 255)
        )
    
    # Add "REGION OF INTEREST" text above the box
    try:
        # Try to use a larger font
        font = ImageFont.truetype("arial.ttf", 24)
    except:
        # Fallback to default font
        font = ImageFont.load_default()
    
    text = "REGION OF INTEREST"
    text_y = max(0, y - 35)
    draw.text((x, text_y), text, fill=(255, 0, 0, 255), font=font)
    
    # Save the annotated image
    if output_path is None:
        import tempfile
        output_path = tempfile.mktemp(suffix='.png')
    
    # Convert to RGB if necessary (remove alpha for PNG compatibility with some viewers)
    if img.mode == 'RGBA':
        # Create a white background
        background = Image.new('RGB', img.size, (255, 255, 255))
        background.paste(img, mask=img.split()[3])  # Use alpha channel as mask
        img = background
    
    img.save(output_path, 'PNG')
    return output_path


def query_image_region(
    image_path: str,
    bbox: dict,
    question: str,
    context: str = None,
    system_prompt: str = "You are an expert medical imaging specialist. Focus your analysis on the region highlighted with the red bounding box in the image.",
    max_tokens: int = 2048
) -> dict:
    """
    Query the LLM about a specific region of an image.
    
    This function:
    1. Annotates the image with a red bounding box at the specified region
    2. Builds a prompt that tells the LLM to focus on the highlighted region
    3. Sends the annotated image + prompt to the multimodal LLM
    
    Args:
        image_path: Path to the original image
        bbox: Dictionary with 'x', 'y', 'width', 'height' keys
        question: The user's question about the region
        context: Optional context (e.g., AI summary from report analysis)
        system_prompt: System prompt for the LLM
        max_tokens: Maximum tokens for response
    
    Returns:
        dict with 'success', 'response' or 'error'
    """
    import tempfile
    import shutil
    from callables import predict_multimodal
    
    try:
        # Create annotated image
        temp_dir = tempfile.mkdtemp()
        annotated_path = os.path.join(temp_dir, 'annotated_image.png')
        annotate_image_with_bounding_box(image_path, bbox, annotated_path)
        
        # Build the prompt
        context_part = ""
        if context:
            context_part = f"Context from the medical report analysis:\n{context}\n\n"
        
        full_prompt = (
            f"{context_part}"
            f"The following image is a medical scan with a RED BOUNDING BOX marking the region of interest. "
            f"The user is asking about this specific highlighted region.\n\n"
            f"User question: {question}\n\n"
            f"Please analyze the highlighted region and answer the question."
        )
        
        # Call multimodal prediction
        result = predict_multimodal(
            prompt=full_prompt,
            image_paths=[annotated_path],
            system_prompt=system_prompt,
            max_tokens=max_tokens
        )
        
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)
        
        if result:
            return {"success": True, "response": result}
        else:
            return {"success": False, "error": "No response from LLM"}
            
    except Exception as e:
        return {"success": False, "error": str(e)}


def main():
    """Example usage of the processor with parallel API calls."""
    import sys
    
    # Check for --sequential flag
    use_parallel = "--sequential" not in sys.argv
    args = [arg for arg in sys.argv[1:] if arg != "--sequential"]
    
    # Get PDF paths from command line or use default
    if args:
        pdf_paths = args
    else:
        # Default to example.pdf in the same directory
        base_dir = Path(__file__).parent
        pdf_paths = [str(base_dir / "docs.pdf"), str(base_dir / "blood.pdf")]
        pdf_paths = [str(base_dir / "dengue.pdf")]
    
    # Check if PDFs exist
    for pdf_path in pdf_paths:
        if not Path(pdf_path).exists():
            print(f"ERROR: PDF not found: {pdf_path}")
            return
    
    mode = "PARALLEL" if use_parallel else "SEQUENTIAL"
    print(f"Processing {len(pdf_paths)} PDF(s) in {mode} mode...")
    
    # Create processor and run
    processor = MedicalReportProcessor()
    
    if use_parallel:
        result = processor.process_pdfs_parallel(pdf_paths)
    else:
        result = processor.process_pdfs(pdf_paths)
    
    print("\n" + "="*50)
    print(f"PROCESSING COMPLETE ({mode})")
    print("="*50)
    print(f"Report data: {result['report_data_path']}")
    print(f"Image gallery: {result['image_gallery_path']}")
    print(f"Tabular reports: {result['tabular_reports_path']}")
    print(f"LLM calls summary: {result['llm_summary']}")
    
    print("\n--- Report Data Preview ---")
    print(json.dumps(result['report_data'], indent=2)[:2000])
    
    print("\n--- Image Gallery Preview ---")
    print(json.dumps(result['image_gallery'], indent=2)[:1000])
    
    print("\n--- Tabular Reports Preview ---")
    print(json.dumps(result['tabular_reports'], indent=2)[:1500])


if __name__ == "__main__":
    main()