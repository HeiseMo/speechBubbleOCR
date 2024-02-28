import os
import cv2
import numpy as np
import easyocr
from deep_translator import GoogleTranslator
from ultralytics import YOLO

# Directories setup
input_folder_path = 'rawImg'
masked_folder_path = 'masked'  # For inpainted images
output_folder_path = 'output'  # For text files

# Load the YOLO model for speech bubble detection
model = YOLO("comic-speech-bubble-detector.pt")

# Ensure output and masked directories exist
os.makedirs(output_folder_path, exist_ok=True)
os.makedirs(masked_folder_path, exist_ok=True)

# Initialize the OCR reader for Korean
reader = easyocr.Reader(['ko'])

def crop_and_ocr(image_path, bboxes):
    """Crop the speech bubbles from the image based on bboxes and perform OCR."""
    image = cv2.imread(image_path)
    texts = []
    for bbox in bboxes:
        # Coordinates from the bounding box
        x1, y1, x2, y2 = [int(coord) for coord in bbox.xyxy[0].tolist()]
        # Crop the speech bubble area
        cropped_image = image[y1:y2, x1:x2]
        # Perform OCR on the cropped area
        result = reader.readtext(cropped_image, detail=1)
        text = " ".join([text for (_, text, _) in result])
        texts.append(text)
    return texts

def process_images(input_folder, masked_folder, output_folder):
    for image_name in os.listdir(input_folder):
        image_path = os.path.join(input_folder, image_name)
        if os.path.isfile(image_path):
            print(f"Processing {image_name}...")

            # Detect speech bubbles
            results = model.predict(image_path)
            result = results[0]  # Assuming one image is processed

            # Crop and OCR speech bubbles
            texts = crop_and_ocr(image_path, result.boxes)

            # Concatenate all detected texts for raw and translation purposes
            raw_text = '\n'.join(texts)

            # Save the raw text
            raw_text_path = os.path.join(output_folder, os.path.splitext(image_name)[0] + '_raw.txt')
            with open(raw_text_path, 'w', encoding='utf-8') as f:
                f.write(raw_text)

            # Translate the text to English
            translated_text = GoogleTranslator(source='auto', target='en').translate(raw_text)

            # Save the translated text
            translated_text_path = os.path.join(output_folder, os.path.splitext(image_name)[0] + '_translated.txt')
            with open(translated_text_path, 'w', encoding='utf-8') as f:
                f.write(translated_text)

            print(f"Finished processing {image_name}.")

# Correct the function call with proper arguments
process_images(input_folder_path, masked_folder_path, output_folder_path)
