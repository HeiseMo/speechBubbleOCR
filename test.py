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

def draw_green_boxes(image_path, bboxes):
    """Draw a green line around the speech bubbles based on the bounding boxes."""
    image = cv2.imread(image_path)
    
    for bbox in bboxes:
        # Convert bbox coordinates to integer
        x1, y1, x2, y2 = [int(coord) for coord in bbox.xyxy[0].tolist()]
        
        # Draw a green rectangle around the text area
        # cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 255), 2)  # 2 is the thickness of the line
        # Create a mask for the speech bubble area
        mask = np.zeros_like(image)
        cv2.rectangle(mask, (x1, y1), (x2, y2), (255, 255, 255), -1)
        # Inpaint the image to white within the speech bubble area
        image = cv2.inpaint(image, mask[:, :, 0], 3, cv2.INPAINT_TELEA)

    # Save or display the image
    cv2.imwrite('output_with_green_boxes.png', image)
    # If you want to display the image uncomment the next line
    # cv2.imshow('Image with Green Boxes', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

# Assuming bboxes is a list of bounding boxes and each bbox has a .xyxy attribute
# Here, you would pass the actual bounding boxes and image path to the function


def crop_and_ocr(image_path, bboxes, output_file):
    """Crop the speech bubbles from the image based on bboxes, perform OCR, and save the text into a file."""
    image = cv2.imread(image_path)
    texts = []
    for i, bbox in enumerate(bboxes):
        # Coordinates from the bounding box
        x1, y1, x2, y2 = [int(coord) for coord in bbox.xyxy[0].tolist()]
        # Crop the speech bubble area
        cropped_image = image[y1:y2, x1:x2]
        # cv2.imshow('Cropped Image', cropped_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # Perform OCR on the cropped area
        result = reader.readtext(cropped_image, detail=1)
        text = " ".join([text for (_, text, _) in result])
        texts.append(text)
    
        # Save the cropped image with a unique identifier
        cropped_image_path = os.path.join(output_file, f"{os.path.splitext('crop')[0]}_{i}.png")
        cv2.imwrite(cropped_image_path, cropped_image)
    
    # Save the texts into a file
    raw_text_path = os.path.join(output_file, f"{os.path.splitext('2_img')[0]}_raw.txt")
    with open(raw_text_path, 'w', encoding='utf-8') as file:
        file.write('\n'.join(texts))
    
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
            texts = crop_and_ocr(image_path, result.boxes, output_folder)

            # Translate the text to English
            translated_texts = [GoogleTranslator(source='auto', target='en').translate(text) for text in texts]
            bubble_results = model.predict(image_path)
            bubble_result = bubble_results[0]  # Assuming one image is processed

            text_results = text_segmentation.predict(image_path)
            text_result = text_results[0]  # Assuming one image is processed

            # Inpaint and overlay translated text
            final_image =draw_green_boxes(image_path, text_result.boxes)

            # Save the final image
            final_image_path = os.path.join(output_folder, image_name)
            cv2.imwrite(final_image_path, final_image)

            print(f"Finished processing {image_name}.")

# Correct the function call with proper arguments
process_images(input_folder_path, masked_folder_path, output_folder_path)
