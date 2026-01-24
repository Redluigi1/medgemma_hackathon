"""
Medical Report PDF Processor
Converts PDFs to images, extracts medical images and structured data, outputs to JSON.
"""

import os
import json
import time
from pathlib import Path
from datetime import datetime
from PIL import Image
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


class MedicalReportProcessor:
    """Main class to process medical PDFs and extract structured data."""
    
    def __init__(self, output_dir: str = None):
        """
        Initialize the processor.
        
        Args:
            output_dir: Directory to save output files and images
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
- Blood work / pathology reports with test names and numerical values
- Prescription lists with medications, dosages
- Discharge summaries with structured sections
- Radiology/imaging reports with findings in structured format
- Vital signs or monitoring data in tabular form

NOT a structured report:
- Full-page medical images (X-rays, CT scans, MRI)
- Bills, invoices, or payment receipts
- Handwritten doctor's notes without structure
- Blank or mostly empty pages
- Cover pages or headers only

Respond in this EXACT format:
IS_REPORT: YES or NO
REPORT_TYPE: [lab_report / prescription / radiology_report / discharge_summary / vitals_report / other_report / not_applicable/ medical_bill]
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
        
        # Parse IS_REPORT
        is_report_match = re.search(r'IS_REPORT:\s*(YES|NO)', response_upper)
        if is_report_match:
            result["is_report"] = is_report_match.group(1) == 'YES'
        
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
        
        # Parse the response
        parsed_data = self._parse_json_from_llm_response(response, "extract_report_tabular_data")
        
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
    
    def confirm_medical_image(self, image_path: str) -> bool:
        """
        Confirm if a cropped sub-image is indeed a medical image.
        
        Args:
            image_path: Path to the cropped image
            
        Returns:
            True if confirmed as medical image
        """
        prompt = """Is this image a medical diagnostic scan (X-ray, CT scan, MRI, ultrasound, etc.)?

Answer with ONLY 'YES' or 'NO'."""

        response = self._call_llm_with_logging(
            prompt=prompt,
            image_paths=[image_path],
            function_name="confirm_medical_image"
        )
        
        return "YES" in response.upper()
    
    def describe_medical_image(self, image_path: str, is_full_page: bool = False, 
                                page_image_path: str = None) -> dict:
        """
        Get a patient-friendly description and caption of a medical image.
        Uses the detailed summary as context, and for sub-images also uses the full page.
        
        Args:
            image_path: Path to the medical image
            is_full_page: Whether this is a full-page scan (or sub-image)
            page_image_path: Path to the original PDF page (for sub-image context)
            
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
Based on the image and the context above, provide ONLY the following two things:

1. CAPTION: A very short image caption (maximum 8 words).

2. DESCRIPTION: A detailed patient-friendly explanation  that identifies the scan type, body part shown, and what the image shows related to the patient's condition. Use simple language.

CRITICAL: Output ONLY the caption and description text below. Do NOT include any thinking process, instructions, or meta-commentary. Just provide the actual content that will be shown to the patient.

Format your response EXACTLY as:
CAPTION: [your actual caption text here - max 8 words]
DESCRIPTION: [your actual description text here - 3-5 sentences for the patient to read]"""

        # For sub-images, include the full page as additional context
        image_paths = [image_path]
        if not is_full_page and page_image_path:
            image_paths.append(page_image_path)
        
        response = self._call_llm_with_logging(
            prompt=prompt,
            image_paths=image_paths,
            function_name="describe_medical_image",
            system_prompt="You are a medical professional. Provide ONLY the requested caption and description. Do not include thinking process or instructions."
        )
        
        # Clean response: remove thinking tokens and extract clean text
        cleaned_response = response
        
        # Remove thinking tokens (like <unused94>thought, <|thinking|>, etc.)
        cleaned_response = re.sub(r'<unused\d+>.*?(?=CAPTION:|DESCRIPTION:|$)', '', cleaned_response, flags=re.DOTALL)
        cleaned_response = re.sub(r'<\|thinking\|>.*?<\|/thinking\|>', '', cleaned_response, flags=re.DOTALL)
        cleaned_response = re.sub(r'\*\*thought.*?(?=CAPTION:|DESCRIPTION:|$)', '', cleaned_response, flags=re.DOTALL | re.IGNORECASE)
        
        # Parse caption and description from response
        caption = ""
        description = ""
        
        if "CAPTION:" in cleaned_response and "DESCRIPTION:" in cleaned_response:
            # Extract caption
            caption_match = re.search(r'CAPTION:\s*(.+?)(?=DESCRIPTION:|$)', cleaned_response, re.DOTALL)
            if caption_match:
                caption = caption_match.group(1).strip()
                # Remove any asterisks, bullet points, or formatting
                caption = re.sub(r'^\*+\s*', '', caption)
                caption = re.sub(r'\*+$', '', caption)
                # Take only the first line if multi-line
                caption = caption.split('\n')[0].strip()
            
            # Extract description
            desc_match = re.search(r'DESCRIPTION:\s*(.+?)$', cleaned_response, re.DOTALL)
            if desc_match:
                description = desc_match.group(1).strip()
                # Remove any leading asterisks or formatting
                description = re.sub(r'^\*+\s*', '', description)
                description = re.sub(r'\*+$', '', description)
                # Remove any meta-instructions that might have slipped through
                # Look for sentences that are instructions rather than content
                sentences = description.split('.')
                clean_sentences = []
                for sent in sentences:
                    sent = sent.strip()
                    # Skip sentences that look like instructions
                    if sent and not any(phrase in sent.lower() for phrase in [
                        'avoid', 'uses simple', 'stick with', 'combine into', 
                        'check if', 'review and', 'formulate the', 'identify the'
                    ]):
                        clean_sentences.append(sent)
                if clean_sentences:
                    description = '. '.join(clean_sentences)
                    if not description.endswith('.'):
                        description += '.'
        else:
            # Fallback: use the whole response if format not followed
            self.logger.logger.warning("Image description response didn't follow expected format")
            description = cleaned_response
        
        return {
            "caption": caption,
            "description": description
        }
    
    def extract_sub_images(self, image_path: str, bboxes: list, source_pdf: str, page_num: int) -> list:
        """
        Crop sub-images using bounding boxes from YOLO.
        
        Args:
            image_path: Path to the PDF page image
            bboxes: List of bounding boxes [x1, y1, x2, y2]
            source_pdf: Name of source PDF
            page_num: Page number
            
        Returns:
            List of paths to cropped images
        """
        img = Image.open(image_path)
        cropped_paths = []
        
        for i, bbox in enumerate(bboxes):
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
            
            self.logger.logger.info(f"Cropped sub-image: {crop_path}")
        
        return cropped_paths
    
    def process_page_for_medical_images(self, image_path: str, source_pdf: str, page_num: int):
        """
        Process a single PDF page for medical images AND structured reports.
        
        Logic (two-stage for reports):
        1. FIRST: Classify if this is a structured report -> extract tabular data with specialized agent
        2. Check if full-page medical image -> save it, skip CNN
        3. Otherwise check for embedded images -> run CNN to extract sub-images
        
        Args:
            image_path: Path to the PDF page image
            source_pdf: Name of source PDF
            page_num: Page number
        """
        self.logger.logger.info(f"Processing page {page_num} for medical images...")
        
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
        # Step 1: Is this a full-page medical image?
        is_full_page = self.is_full_page_medical_image(image_path)
        
        if is_full_page:
            self.logger.logger.info(f"Page {page_num}: FULL-PAGE medical image detected - skipping embedded check & CNN")
            
            # Copy to medical images directory
            dest_path = self.medical_images_dir / f"{source_pdf}_page{page_num}_full.png"
            Image.open(image_path).save(str(dest_path))
            
            # Get description with context (full-page = no page context needed, just summary)
            desc_result = self.describe_medical_image(image_path, is_full_page=True)
            
            # Add to gallery with caption and description
            self.image_gallery["medical_images"].append({
                "image_path": str(dest_path),
                "type": "full_page",
                "source_pdf": source_pdf,
                "page_number": page_num,
                "caption": desc_result["caption"],
                "description": desc_result["description"]
            })
            # Return early - do NOT check for embedded images or run CNN
            return
        
        # Step 2: NOT a full-page scan. Check if it has embedded medical images among text
        if self.has_embedded_medical_images(image_path):
            self.logger.logger.info(f"Page {page_num}: Has embedded medical images, running CNN...")
            
            # Use YOLO to find bounding boxes
            try:
                bboxes = get_bounding_boxes(image_path)
                self.logger.logger.info(f"Found {len(bboxes)} potential regions")
                
                if bboxes:
                    # Extract sub-images
                    cropped_paths = self.extract_sub_images(image_path, bboxes, source_pdf, page_num)
                    
                    # Confirm each sub-image is a medical image
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
                                "type": "sub_image",
                                "source_pdf": source_pdf,
                                "page_number": page_num,
                                "caption": desc_result["caption"],
                                "description": desc_result["description"]
                            })
                        else:
                            self.logger.logger.info(f"Cropped image not confirmed as medical: {crop_path}")
                            # Delete non-medical crops
                            os.remove(crop_path)
            except Exception as e:
                self.logger.logger.error(f"Error running CNN on page {page_num}: {e}")
        else:
            self.logger.logger.info(f"Page {page_num}: No medical images detected")
    
    # ==================== STRUCTURED DATA EXTRACTION ====================
    
    def extract_detailed_summary(self, image_paths: list):
        """
        Extract a detailed summary of the medical report.
        This is extracted FIRST and used as context for image descriptions.
        
        Args:
            image_paths: List of paths to PDF page images
        """
        self.logger.logger.info("Extracting detailed summary for context...")
        
        prompt = """Analyze these medical report pages and provide a DETAILED SUMMARY suitable for a patient to understand.

Include:
1. What medical condition/issue is being reported
2. What tests/procedures were done
3. What the findings show
4. Any treatments or surgeries that were performed
5. The current status and prognosis

Write in clear, simple language a patient can understand. Be thorough but avoid excessive medical jargon.
Keep the summary to 150-200 words."""

        self.detailed_summary = self._call_llm_with_logging(
            prompt=prompt,
            image_paths=image_paths,
            function_name="extract_detailed_summary",
            system_prompt="You are a medical professional explaining reports to patients in simple terms."
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
                # Find all JSON code blocks
                json_blocks = re.findall(r'```(?:json)?\s*([\{\[][\s\S]*?[\}\]])\s*```', cleaned_response, re.DOTALL)
                if json_blocks:
                    json_candidates.extend(json_blocks)
            
            # If no code blocks found, try to find raw JSON structures
            if not json_candidates:
                # Find all JSON-like structures (objects or arrays)
                raw_jsons = re.findall(r'([\{\[][\s\S]*?[\}\]])', cleaned_response)
                json_candidates.extend(raw_jsons)
            
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
        Extract patient info, doctor info, and patient history.
        Focused call for demographic and contact information.
        
        Args:
            image_paths: List of paths to PDF page images
            
        Returns:
            Dictionary with patient_info, doctor_info, patient_history
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
- patient_history: Any mentioned past medical conditions, allergies, or previous treatments

WHAT TO IGNORE:
- Hospital/clinic names and addresses (these are NOT the patient or doctor name)
- Bill amounts, invoice numbers, bank details
- Test results and findings (those go elsewhere)
- Dates of visits (unless it's medical history)

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
    },
    "patient_history": "past medical conditions/allergies mentioned or null"
}"""

        response = self._call_llm_with_logging(
            prompt=prompt,
            image_paths=image_paths,
            function_name="extract_patient_and_doctor_info",
            system_prompt="You are a medical records specialist. Extract ONLY demographic data. Do NOT include addresses, hospital names, or financial information."
        )
        
        expected_schema = """{
    "patient_info": {"name": "string or null", "age": "string or null", "sex": "string or null"},
    "doctor_info": {"name": "string or null", "phone": "string or null", "email": "string or null"},
    "patient_history": "string or null"
}"""
        
        return self._parse_json_from_llm_response(response, "extract_patient_and_doctor_info", expected_schema)
    
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

Read the medical documents and create a clear, helpful summary for a patient.

FIELD DEFINITIONS:

1. main_findings: A brief medical summary (2-3 sentences about what was found).

2. patient_explanation: A detailed explanation (5-8 sentences) covering:
   - What condition/injury was found
   - What tests or imaging were done (X-ray, CT scan, blood test, etc.)
   - What the test results showed
   - What treatment was done (surgery, medication, etc.)
   - Current status and expected recovery time
   Use simple, non-medical language a patient can understand.

3. diagnosis: The specific medical condition or diagnosis (e.g., "Fracture of left clavicle").

4. recommendations: What the doctor advises for follow-up (e.g., "Follow-up X-ray in 6 weeks").

WHAT TO IGNORE: Hospital names, addresses, bill amounts, invoice numbers.

Output ONLY this JSON structure (nothing else - no explanations, no thinking):
{
    "report_summary": {
        "main_findings": "brief medical findings summary",
        "patient_explanation": "detailed 5-8 sentence patient-friendly explanation",
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
    
    def extract_all_structured_data(self, image_paths: list):
        """
        Extract all structured data from PDF images using multiple focused calls.
        This method orchestrates three separate extraction calls for robustness.
        
        Note: Lab results are extracted via the two-stage report extraction process
        and saved to tabular_reports.json, so they are not duplicated here.
        
        Args:
            image_paths: List of paths to PDF page images
        """
        self.logger.logger.info("Starting structured data extraction (3 separate calls)...")
        
        # Call 1: Patient info, doctor info, and patient history
        patient_doctor_data = self.extract_patient_and_doctor_info(image_paths)
        
        # Call 2: Report summary
        summary_data = self.extract_report_summary(image_paths)
        
        # Call 3: Medications and next appointment
        meds_data = self.extract_medications_and_appointments(image_paths)
        
        # Note: Lab results are NOT extracted here - they are captured via the
        # two-stage report extraction (is_report_page -> extract_report_tabular_data)
        # and saved to tabular_reports.json
        
        # Merge all the data together
        self.report_data = {
            "patient_info": patient_doctor_data.get("patient_info", None),
            "doctor_info": patient_doctor_data.get("doctor_info", None),
            "patient_history": patient_doctor_data.get("patient_history", None),
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
        pdf_to_pages = {}  # Store pages per PDF for processing
        
        # Step 1: Convert all PDFs to images first
        for pdf_path in pdf_paths:
            self.logger.logger.info(f"Processing PDF: {pdf_path}")
            pdf_name = Path(pdf_path).stem
            
            page_images = self.convert_pdf_to_images(pdf_path)
            pdf_to_pages[pdf_name] = page_images
            all_page_images.extend(page_images)
        
        # Step 2: Extract detailed summary FIRST (needed as context for image descriptions)
        self.extract_detailed_summary(all_page_images)
        
        # Step 3: Now process each page for medical images (with summary context available)
        for pdf_name, page_images in pdf_to_pages.items():
            for i, img_path in enumerate(page_images):
                self.process_page_for_medical_images(img_path, pdf_name, i + 1)
        
        # Step 4: Extract structured data from all pages
        self.extract_all_structured_data(all_page_images)
        
        # Step 5: Save outputs
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


def main():
    """Example usage of the processor."""
    import sys
    
    # Get PDF paths from command line or use default
    if len(sys.argv) > 1:
        pdf_paths = sys.argv[1:]
    else:
        # Default to example.pdf in the same directory
        base_dir = Path(__file__).parent
    
        pdf_paths = [str(base_dir / "no.pdf")]
        pdf_paths = [str(base_dir / "docs.pdf"), str(base_dir / "blood.pdf")]
    
    # Check if PDFs exist
    for pdf_path in pdf_paths:
        if not Path(pdf_path).exists():
            print(f"ERROR: PDF not found: {pdf_path}")
            return
    
    print(f"Processing {len(pdf_paths)} PDF(s)...")
    
    # Create processor and run
    processor = MedicalReportProcessor()
    result = processor.process_pdfs(pdf_paths)
    
    print("\n" + "="*50)
    print("PROCESSING COMPLETE")
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
