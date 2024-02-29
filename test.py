import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import easyocr
from deep_translator import GoogleTranslator
from ultralytics import YOLO

# Directories setup
input_folder_path = 'rawImg'
masked_folder_path = 'masked'  # For inpainted images
output_folder_path = 'output'  # For final images with translated text
font_path = 'T:\Projects\speechBubbleOCR\ComicNeue\ComicNeue-Regular.ttf'  # Specify the path to a legible font

# Load the YOLO model for speech bubble detection
model = YOLO("comic-speech-bubble-detector.pt")
model = model.to('cpu')  # Use CPU for inference
text_segmentation = YOLO('comic-text-segmenter.pt')

# Ensure directories exist
os.makedirs(output_folder_path, exist_ok=True)
os.makedirs(masked_folder_path, exist_ok=True)

# Initialize the OCR reader for Korean
reader = easyocr.Reader(['ko'])

def draw_inpaint(image_path, bboxes):
    image = cv2.imread(image_path)
    
    for bbox in bboxes:
        # Convert bbox coordinates to integer
        x1, y1, x2, y2 = [int(coord) for coord in bbox.xyxy[0].tolist()]
        # Create a mask for the speech bubble area
        mask = np.zeros_like(image)
        cv2.rectangle(mask, (x1, y1), (x2, y2), (255, 255, 255), -1)
        # cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2) #change back to line up
        # Inpaint the image to white within the speech bubble area
        image = cv2.inpaint(image, mask[:, :, 0], 3, cv2.INPAINT_TELEA)

    return image

def crop_and_ocr(image_path, bboxes, output_file):
    """Crop the speech bubbles from the image based on bboxes, perform OCR, and save the text into a file."""
    image = cv2.imread(image_path)
    texts = []
    for i, bbox in enumerate(bboxes):
        # Coordinates from the bounding box
        x1, y1, x2, y2 = [int(coord) for coord in bbox.xyxy[0].tolist()]
        # Crop the speech bubble area
        cropped_image = image[y1:y2, x1:x2]
        text_results = text_segmentation.predict(cropped_image)
        text_result = text_results[0]
        # Perform OCR on the cropped area
        x1, y1, x2, y2 = [int(coord) for coord in text_result.boxes[0].xyxy[0].tolist()]

        # Padding for box coords
        x1 -= 8
        y1 -= 8
        x2 += 8
        y2 += 8

        cropped_text_img = cropped_image[y1:y2, x1:x2]
        result = reader.readtext(cropped_text_img, detail=1)
        text = " ".join([text for (_, text, _) in result])
        texts.append(text)
    
        # Save the cropped image with a unique identifier
        cropped_image_path = os.path.join(output_file, f"{os.path.splitext(os.path.basename(image_path))[0]}_{i}.png")
        cv2.imwrite(cropped_image_path, cropped_text_img)
    
    # Save the texts into a file
    raw_text_path = os.path.join(output_file, f"{os.path.splitext(os.path.basename(image_path))[0]}_raw.txt")
    with open(raw_text_path, 'w', encoding='utf-8') as file:
        file.write('\n'.join(texts))
    
    return texts

def process_images(input_folder, masked_folder, output_folder):
    for image_name in os.listdir(input_folder):
        image_path = os.path.join(input_folder, image_name)
        if os.path.isfile(image_path):
            print(f"Processing {image_name}...")

            # Detect speech bubbles
            bubble_results = model.predict(image_path)
            bubble_result = bubble_results[0]  # Assuming one image is processed

            text_results = text_segmentation.predict(image_path)
            # Assuming one image is processed
            text_result = text_results[0]

            # Crop and OCR speech bubbles
            texts = crop_and_ocr(image_path, bubble_result.boxes, output_folder)

            # Translate the text to English
            translated_texts = [GoogleTranslator(source='auto', target='en').translate(text) for text in texts]
            translated_text_path = os.path.join(output_folder, os.path.splitext(image_name)[0] + '_translated.txt')

            with open(translated_text_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(translated_texts))

            # Inpaint and overlay translated text
            final_image = draw_inpaint(image_path, text_result.boxes)

            # Save the final image
            final_path = os.path.join(os.path.abspath(masked_folder), os.path.splitext(image_name)[0] + '_inpainted.png')
            cv2.imwrite(final_path, final_image)
            
            print(f"Finished processing {image_name}.")

# Correct the function call with proper arguments
process_images(input_folder_path, masked_folder_path, output_folder_path)
