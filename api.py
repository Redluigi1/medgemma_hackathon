"""
FastAPI Application for Medical Report Processor
Exposes REST API endpoints for medical report processing and LLM predictions.
"""

import os
import shutil
import tempfile
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from callables import predict_text_only, predict_multimodal, get_bounding_boxes
from medical_report_processor_v2 import MedicalReportProcessor, query_image_region

app = FastAPI(
    title="Medical Report Processor API",
    description="API for processing medical PDFs and making LLM predictions",
    version="1.0.0"
)

# Enable CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================== Request/Response Models ====================

class TextPredictionRequest(BaseModel):
    prompt: str
    system_prompt: str = "You are an expert medical doctor."
    max_tokens: int = 1024


class TextPredictionResponse(BaseModel):
    success: bool
    response: Optional[dict] = None
    error: Optional[str] = None


class ProcessPDFResponse(BaseModel):
    success: bool
    report_data: Optional[dict] = None
    image_gallery: Optional[dict] = None
    tabular_reports: Optional[list] = None
    llm_summary: Optional[dict] = None
    error: Optional[str] = None


# ==================== API Endpoints ====================

@app.get("/")
async def root():
    """Health check endpoint."""
    return {"status": "healthy", "service": "Medical Report Processor API"}


@app.get("/health")
async def health_check():
    """Health check endpoint for container orchestration."""
    return {"status": "healthy"}


@app.post("/process-pdf", response_model=ProcessPDFResponse)
async def process_pdf(files: List[UploadFile] = File(...)):
    """
    Process one or more PDF files and extract medical information.
    
    This endpoint:
    1. Accepts PDF file(s) via multipart form upload
    2. Processes them using MedicalReportProcessor
    3. Returns structured JSON with report_data, image_gallery, and tabular_reports
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    
    # Validate file types
    for file in files:
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid file type: {file.filename}. Only PDF files are accepted."
            )
    
    # Create temp directory for processing
    temp_dir = tempfile.mkdtemp()
    pdf_paths = []
    
    try:
        # Save uploaded files to temp directory
        for file in files:
            file_path = os.path.join(temp_dir, file.filename)
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            pdf_paths.append(file_path)
        
        # Process PDFs
        processor = MedicalReportProcessor(output_dir=temp_dir)
        result = processor.process_pdfs(pdf_paths)
        #print output also 
        print(result)
        
        # Convert image paths to base64 before cleanup
        image_gallery = result.get("image_gallery", {})
        if image_gallery and "medical_images" in image_gallery:
            import base64
            for img in image_gallery.get("medical_images", []):
                img_path = img.get("image_path")
                if img_path and os.path.exists(img_path):
                    try:
                        with open(img_path, "rb") as f:
                            img_data = f.read()
                            # Determine image type
                            ext = os.path.splitext(img_path)[1].lower()
                            mime_type = "image/png" if ext == ".png" else "image/jpeg"
                            img["image_base64"] = f"data:{mime_type};base64,{base64.b64encode(img_data).decode('utf-8')}"
                    except Exception as e:
                        print(f"Error encoding image {img_path}: {e}")
        
        return ProcessPDFResponse(
            success=True,
            report_data=result.get("report_data"),
            image_gallery=image_gallery,
            tabular_reports=result.get("tabular_reports"),
            llm_summary=result.get("llm_summary")
        )
        
    except Exception as e:
        return ProcessPDFResponse(
            success=False,
            error=str(e)
        )
    finally:
        # Cleanup temp directory
        shutil.rmtree(temp_dir, ignore_errors=True)


@app.post("/predict-text", response_model=TextPredictionResponse)
async def predict_text(request: TextPredictionRequest):
    """
    Make a text-only LLM prediction.
    
    This endpoint calls the Vertex AI endpoint with a text prompt only (no images).
    """
    try:
        result = predict_text_only(
            prompt=request.prompt,
            system_prompt=request.system_prompt,
            max_tokens=request.max_tokens
        )
        
        return TextPredictionResponse(
            success=True,
            response=result
        )
        
    except Exception as e:
        return TextPredictionResponse(
            success=False,
            error=str(e)
        )


@app.post("/predict-multimodal")
async def predict_multimodal_endpoint(
    prompt: str = Form(...),
    system_prompt: str = Form(default="You are an expert medical doctor."),
    max_tokens: int = Form(default=2048),
    images: List[UploadFile] = File(...)
):
    """
    Make a multimodal (text + images) LLM prediction.
    
    This endpoint accepts:
    - prompt: Text prompt (form field)
    - system_prompt: System prompt (form field, optional)
    - max_tokens: Max tokens (form field, optional)
    - images: One or more image files
    """
    if not images:
        raise HTTPException(status_code=400, detail="No images provided")
    
    # Create temp directory for images
    temp_dir = tempfile.mkdtemp()
    image_paths = []
    
    try:
        # Save uploaded images
        for img in images:
            # Validate image type
            if not img.content_type or not img.content_type.startswith('image/'):
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid file type: {img.filename}. Only image files are accepted."
                )
            
            file_path = os.path.join(temp_dir, img.filename)
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(img.file, buffer)
            image_paths.append(file_path)
        
        # Call multimodal prediction
        result = predict_multimodal(
            prompt=prompt,
            image_paths=image_paths,
            system_prompt=system_prompt,
            max_tokens=max_tokens
        )

        #print the output json as well
        
        
        return JSONResponse(content={
            "success": True,
            "response": result
        })
    
    except Exception as e:
        return JSONResponse(content={
            "success": False,
            "error": str(e)
        })
    finally:
        # Cleanup temp directory
        shutil.rmtree(temp_dir, ignore_errors=True)


@app.post("/get-bounding-boxes")
async def get_bounding_boxes_endpoint(
    image: UploadFile = File(...),
    confidence_threshold: float = Form(default=0.0)
):
    """
    Get bounding boxes for medical images using YOLO model.
    
    This endpoint runs the fine-tuned YOLO model on an image
    and returns detected bounding boxes with confidence scores.
    """
    # Create temp directory
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Save uploaded image
        file_path = os.path.join(temp_dir, image.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)
        
        # Get bounding boxes
        result = get_bounding_boxes(
            image_path=file_path,
            confidence_threshold=confidence_threshold
        )
        
        return JSONResponse(content={
            "success": True,
            "detections": result
        })
        
    except Exception as e:
        return JSONResponse(content={
            "success": False,
            "error": str(e)
        })
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


@app.post("/query-image-region")
async def query_image_region_endpoint(
    image: UploadFile = File(...),
    bbox_x: float = Form(...),
    bbox_y: float = Form(...),
    bbox_width: float = Form(...),
    bbox_height: float = Form(...),
    scale_x: float = Form(default=1.0),
    scale_y: float = Form(default=1.0),
    question: str = Form(...),
    context: str = Form(default=""),
    max_tokens: int = Form(default=2048)
):
    """
    Query the LLM about a specific region of an image.
    
    This endpoint:
    1. Receives the original image and bounding box coordinates
    2. Annotates the image with a red bounding box on the backend
    3. Sends the annotated image to the LLM with a prompt focusing on the region
    
    Args:
        image: The original image file
        bbox_x, bbox_y: Top-left corner of bounding box (in display pixels)
        bbox_width, bbox_height: Size of bounding box (in display pixels)
        scale_x, scale_y: Scale factors to convert display coords to natural image coords
        question: The user's question about the selected region
        context: Optional context (e.g., AI analysis summary)
        max_tokens: Maximum tokens for response
    """
    # Create temp directory
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Validate image type
        if not image.content_type or not image.content_type.startswith('image/'):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type: {image.filename}. Only image files are accepted."
            )
        
        # Save uploaded image
        file_path = os.path.join(temp_dir, image.filename or "image.png")
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)
        
        # Build bounding box dict
        bbox = {
            'x': bbox_x,
            'y': bbox_y,
            'width': bbox_width,
            'height': bbox_height,
            'scale_x': scale_x,
            'scale_y': scale_y
        }
        
        # Call the query function
        result = query_image_region(
            image_path=file_path,
            bbox=bbox,
            question=question,
            context=context if context else None,
            max_tokens=max_tokens
        )
        
        return JSONResponse(content=result)
        
    except Exception as e:
        return JSONResponse(content={
            "success": False,
            "error": str(e)
        })
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
