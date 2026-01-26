from ultralytics import YOLO
import os
from google.cloud import aiplatform
import base64
from dotenv import load_dotenv
from google.cloud import aiplatform
from google.protobuf import json_format
from google.protobuf.struct_pb2 import Value
from dotenv import load_dotenv
from google.cloud import aiplatform
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "key.json"
# 1. Load configuration
load_dotenv()

def get_bounding_boxes(image_path):
    """
    Performs inference and returns a list of bounding box coordinates.
    
    Returns:
        list: A list of lists, where each inner list is [x1, y1, x2, y2]
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Construct the path relative to the script location
    model_path = os.path.join(base_dir, 'fine_tune_yolo/runs/detect/train3/weights/best.pt')
    # Load the model
    model = YOLO(model_path)
    
    # Run inference
    results = model(image_path)[0]
    
    # Extract boxes (xyxy format: top-left and bottom-right corners)
    # .tolist() converts the tensor to a standard Python list
    boxes = results.boxes.xyxy.cpu().numpy().tolist()
    
    return boxes

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


