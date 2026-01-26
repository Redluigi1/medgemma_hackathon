import os
from PIL import Image
from ultralytics import YOLO
from dotenv import load_dotenv
from google import genai
from google.genai import types

# 1. Load configuration
load_dotenv()

# Setup Client
# Ensure GEMINI_API_KEY is in your .env file
API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=API_KEY)

def get_bounding_boxes(image_path):
    """
    Performs inference and returns a list of bounding box coordinates.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, 'fine_tune_yolo/runs/detect/train3/weights/best.pt')
    
    model = YOLO(model_path)
    results = model(image_path)[0]
    boxes = results.boxes.xyxy.cpu().numpy().tolist()
    
    return boxes

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