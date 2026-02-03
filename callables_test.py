from ultralytics import YOLO
import os
from google.cloud import aiplatform
import base64
from dotenv import load_dotenv
from google.protobuf import json_format
from google.protobuf.struct_pb2 import Value

# 1. Load configuration from .env first
load_dotenv()

# Set Google credentials from environment variable
google_creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
if google_creds_path:
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = google_creds_path

def get_bounding_boxes(image_path, confidence_threshold=0.0, filter_overlapping=True, overlap_conf_threshold=0.5, overlap_ratio_threshold=0.7):
    """
    Performs inference and returns a list of bounding box coordinates with confidence scores.
    
    Args:
        image_path: Path to the image file
        confidence_threshold: Minimum confidence to include a detection (default 0.0)
        filter_overlapping: If True, filter out smaller low-confidence boxes inside larger boxes
        overlap_conf_threshold: Confidence threshold below which smaller overlapping boxes are removed (default 0.5)
        overlap_ratio_threshold: How much of the smaller box must be inside the larger box to be filtered (default 0.7 = 70%)
    
    Returns:
        list: A list of dicts, where each dict has 'bbox' [x1, y1, x2, y2] and 'confidence' (float)
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Construct the path relative to the script location
    model_path = os.path.join(base_dir, 'fine_tune_yolo/runs/detect/train4/weights/best.pt')
    # Load the model
    model = YOLO(model_path)
    
    # Run inference
    results = model(image_path)[0]
    
    # Extract boxes (xyxy format: top-left and bottom-right corners) and confidence scores
    boxes = results.boxes.xyxy.cpu().numpy().tolist()
    confidences = results.boxes.conf.cpu().numpy().tolist()
    
    # Combine into list of dicts
    detections = []
    for box, conf in zip(boxes, confidences):
        if conf >= confidence_threshold:
            detections.append({
                'bbox': box,
                'confidence': conf
            })
    
    # Filter overlapping boxes if enabled
    if filter_overlapping and len(detections) > 1:
        detections = filter_overlapping_boxes(detections, overlap_conf_threshold, overlap_ratio_threshold)
    
    return detections


def filter_overlapping_boxes(detections, conf_threshold=0.5, overlap_ratio=0.7):
    """
    Filter out smaller bounding boxes that are significantly inside larger ones
    and have low confidence.
    
    If a smaller box is more than overlap_ratio (e.g., 70%) inside a larger box
    and the smaller box has confidence below conf_threshold (e.g., 0.5),
    the smaller box is removed and only the larger one is kept.
    
    Args:
        detections: List of dicts with 'bbox' [x1, y1, x2, y2] and 'confidence'
        conf_threshold: Confidence threshold below which to consider removing a box
        overlap_ratio: Fraction of smaller box that must be inside larger box to remove it
    
    Returns:
        list: Filtered list of detections
    """
    if len(detections) <= 1:
        return detections
    
    def box_area(box):
        """Calculate area of a bounding box."""
        x1, y1, x2, y2 = box
        return max(0, x2 - x1) * max(0, y2 - y1)
    
    def intersection_area(box1, box2):
        """Calculate intersection area between two boxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0
        return (x2 - x1) * (y2 - y1)
    
    # Sort by area (largest first)
    sorted_dets = sorted(detections, key=lambda d: box_area(d['bbox']), reverse=True)
    
    # Track which indices to keep
    keep = [True] * len(sorted_dets)
    
    for i in range(len(sorted_dets)):
        if not keep[i]:
            continue
        
        box_i = sorted_dets[i]['bbox']
        area_i = box_area(box_i)
        
        for j in range(i + 1, len(sorted_dets)):
            if not keep[j]:
                continue
            
            box_j = sorted_dets[j]['bbox']
            area_j = box_area(box_j)
            conf_j = sorted_dets[j]['confidence']
            
            # Check if smaller box (j) is significantly inside larger box (i)
            intersection = intersection_area(box_i, box_j)
            
            if area_j > 0:
                overlap_fraction = intersection / area_j
                
                # If smaller box is mostly inside larger box AND has low confidence, remove it
                if overlap_fraction >= overlap_ratio and conf_j < conf_threshold:
                    keep[j] = False
    
    # Return only the kept detections
    return [det for det, k in zip(sorted_dets, keep) if k]

# Example usage:
# base_dir = os.path.dirname(os.path.abspath(__file__))
# image_path = os.path.join(base_dir, 'img5.png')
# coords = get_bounding_boxes(image_path)
# print(f"Detected {len(coords)} objects. Coordinates: {coords}")





PROJECT_ID = os.getenv("PROJECT_ID")
REGION = os.getenv("REGION")
ENDPOINT_ID = os.getenv("ENDPOINT_ID")

# 2. Initialize Vertex AI
aiplatform.init(project=PROJECT_ID, location=REGION)
endpoint_name = f"projects/{PROJECT_ID}/locations/{REGION}/endpoints/{ENDPOINT_ID}"
endpoint = aiplatform.Endpoint(endpoint_name)

def encode_image(image_path):
    """Encodes a local image to base64 string with the proper data prefix."""
    with open(image_path, "rb") as image_file:
        b64_string = base64.b64encode(image_file.read()).decode("utf-8")
        # Standard OpenAI format for base64 images
        return f"data:image/jpeg;base64,{b64_string}"

def predict_text_only(prompt: str, system_prompt: str = "You are an expert medical doctor.", max_tokens: int = 1024):
    """Function 1: Text-only prediction."""
    instances = [{
        "@requestFormat": "chatCompletions",
        "messages": [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": [{"type": "text", "text": prompt}]}
        ],
        "max_tokens": max_tokens
    }]
    try:
        response = endpoint.predict(instances=instances)
        return response.predictions if response.predictions else None
    except Exception as e:
        print(f"Error: {e}")
        return None

def predict_multimodal(prompt: str, image_paths: list, system_prompt: str = "You are an expert medical doctor.", max_tokens: int = 2048):
    """Function 2: Text + Multiple Images prediction."""
    
    # Create the user content list starting with the text prompt
    user_content = [{"type": "text", "text": prompt}]
    
    # Append each image to the user content
    for path in image_paths:
        user_content.append({
            "type": "image_url",
            "image_url": {"url": encode_image(path)}
        })

    instances = [{
        "@requestFormat": "chatCompletions",
        "messages": [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": user_content}
        ],
        "max_tokens": max_tokens
    }]

    try:
        response = endpoint.predict(instances=instances)
        return response.predictions if response.predictions else None
    except Exception as e:
        print(f"Error: {e}")
        return None

# # --- Example Usage ---
# images = ["img4.png"]
# result = predict_multimodal("Summarize these reports.", images)
# print(result)











from google import genai
from google.genai import types
from PIL import Image


client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    

# Usage






def predict_text_only(prompt: str, system_prompt: str = "You are an expert medical doctor.", max_tokens: int = 1024):
    """Function 1: Text-only prediction using Gemini 3 Flash."""
    try:
        response = client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=[prompt],
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                max_output_tokens=max_tokens,
                thinking_config=types.ThinkingConfig(include_thoughts=True)
            )
        )
        # Returns the text content of the first candidate
        return response.text
    except Exception as e:
        print(f"Error: {e}")
        return None

def predict_multimodal(prompt: str, image_paths: list, system_prompt: str = "You are an expert medical doctor.", max_tokens: int = 2048):
    """Function 2: Text + Multiple Images prediction using Gemini 3 Flash."""
    
    # Build the contents list starting with the prompt
    contents = [prompt]
    
    # Add all images in the list
    for path in image_paths:
        try:
            img = Image.open(path)
            contents.append(img)
        except Exception as e:
            print(f"Could not load image at {path}: {e}")

    try:
        response = client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                max_output_tokens=max_tokens,
                thinking_config=types.ThinkingConfig(include_thoughts=True)
            )
        )
        return response.text
    except Exception as e:
        print(f"Error: {e}")
        return None

# # --- Example Usage (Matching your previous structure) ---
# if __name__ == "__main__":
#     # Example 1: YOLO + Multimodal
#     img_path = "fine_tune_yolo/img1.png"
#     coords = get_bounding_boxes(img_path)
    
#     analysis_prompt = f"I detected objects at {coords}. Can you explain what these represent in the context of this medical image?"
#     result = predict_multimodal(analysis_prompt, [img_path])
#     print(f"Multimodal Result: {result}")

#     # Example 2: Text Only
#     text_result = predict_text_only("What are the primary symptoms of a Stage 2 pressure ulcer?")
#     print(f"Text Result: {text_result}")