import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
import easyocr
from deep_translator import GoogleTranslator
from ultralytics import YOLO
from CAE import extract_comic_archive, create_cbz_from_directory

# Global Variables
# Directories setup
input_folder_path = 'rawImg\\Chapter'
masked_folder_path = 'masked'  # For inpainted images
output_folder_path = 'output'  # For final images with translated text
raw_output_folder_path = 'rawOutput'

# Load the YOLO model for speech bubble detection
model = YOLO("comic-speech-bubble-detector.pt")
model = model.to('cpu')  # Use CPU for inference
text_segmentation = YOLO('comic-text-segmenter.pt')

# Ensure directories exist
os.makedirs(output_folder_path, exist_ok=True)
os.makedirs(masked_folder_path, exist_ok=True)
os.makedirs(raw_output_folder_path, exist_ok=True)

# Initialize the OCR reader for Korean
reader = easyocr.Reader(['ko'])

def convert_to_black_and_white(image_path):
    img = Image.open(image_path)
    # image = Image.open(image_path)
    bw_image = img.convert('L')  # Converts the image to black and white
    equalized_image = ImageOps.equalize(bw_image, mask=None)
    equalized_image = ImageOps.autocontrast(equalized_image, cutoff=3, ignore=None, mask=None, preserve_tone=True)  # Perform histogram equalization
    
    equalized_image_path = os.path.join(raw_output_folder_path, f"{os.path.splitext(os.path.basename(image_path))[0]}_equalized.png")
    equalized_image.save(equalized_image_path)

    return equalized_image_path

def text_split_lines(text, max_width, font):
    words = text.split()
    lines = []
    current_line = []
    current_length = 0

    # Decide how many words you want per line
    words_per_line = 4
    for word in words:
        # Calculate the width of the word when rendered in the specific font
        word_width = font.getlength(word + ' ')  # add space to word width
        if current_length + word_width <= max_width or len(current_line) < words_per_line:
            # Add the word to the current line
            current_line.append(word)
            current_length += word_width
        else:
            # Start a new line
            lines.append(' '.join(current_line))
            current_line = [word]
            current_length = word_width
    
    if current_line:
        lines.append(' '.join(current_line))
    # Split the words into lines
    lines = [' '.join(words[i:i+words_per_line]) for i in range(0, len(words), words_per_line)]
    # Join the lines with newline characters
    text = '\n'.join(lines)

    return text

def draw_translation(image_path, texts, text_bboxes):
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)

    # Specify the font and size
    # Make sure this path is correct
    font_path = 'ComicNeue\ComicNeue-Bold.ttf'  
    # Adjust the size as needed
    font_size = 42  
    font = ImageFont.truetype(font_path, font_size)

    # Sort boxes based on the y-coord of the top-left corner
    text_bboxes = sorted(text_bboxes, key=lambda bbox: bbox.xyxy[0][1])

    for i, bbox in enumerate(text_bboxes):
        # Convert bbox coordinates to integer
        x1, y1, x2, y2 = [int(coord) for coord in bbox.xyxy[0].tolist()]
        # Define text
        if i < len(texts):
            text = texts[i]
            max_width = x2 - x1
            text = text_split_lines(text, max_width, font)
            bbox_text = draw.multiline_textbbox((x1, y1), text, font=font)
            text_width = bbox_text[2] - bbox_text[0]
            text_height = bbox_text[3] - bbox_text[1]
            # Calculate x, y position for the text
            x = x1 + (x2 - x1 - text_width) / 2
            y = y1 + (y2 - y1 - text_height) / 2
            # Draw text
            draw.multiline_text((x, y), text, fill='black', font=font)
            
    filename = os.path.splitext(os.path.basename(image_path))[0].replace('_inpainted', '')
    final_image_path = os.path.join(os.path.abspath(output_folder_path), f"{filename}_final.png")
    image.save(final_image_path)

    return image

def draw_inpaint(image_path, bboxes):
    image = cv2.imread(image_path)
    for bbox in bboxes:
        # Convert bbox coordinates to integer
        x1, y1, x2, y2 = [int(coord) for coord in bbox.xyxy[0].tolist()]
        # Create a mask for the speech bubble area
        mask = np.zeros_like(image)
        cv2.rectangle(mask, (x1, y1), (x2, y2), (255, 255, 255), -1)
        # cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2) # line to draw green box
        # Inpaint the image to white within the speech bubble area
        image = cv2.inpaint(image, mask[:, :, 0], 3, cv2.INPAINT_TELEA)

    return image

def crop_and_ocr(image_path, bboxes):
    """Crop the speech bubbles from the image based on bboxes, perform OCR, and save the text into a file."""
    image = cv2.imread(image_path)
    texts = []
    for i, bbox in enumerate(bboxes):
        # Coordinates from the bounding box
        x1, y1, x2, y2 = [int(coord) for coord in bbox.xyxy[0].tolist()]
        # Crop the speech bubble area
        cropped_image = image[y1:y2, x1:x2]
        # Perform text recognition
        text_results = text_segmentation.predict(cropped_image)
        text_result = text_results[0]
        
        # Perform OCR on the cropped area
        x1, y1, x2, y2 = [int(coord) for coord in text_result.boxes[0].xyxy[0].tolist()]

        # Padding for box coords
        x1 -= 35
        y1 -= 25
        x2 += 55
        y2 += 8

         # Save the cropped image with a unique identifier
        cropped_image_path = os.path.join(raw_output_folder_path, f"{os.path.splitext(os.path.basename(image_path))[0]}_{i}.png")
        cv2.imwrite(cropped_image_path, cropped_image)

        bwImage = convert_to_black_and_white(cropped_image_path)
        bwImage = cv2.imread(bwImage)
        cropped_text_image = bwImage[y1:y2, x1:x2]
        
        result = reader.readtext(cropped_text_image, detail=1)
        text = " ".join([text for (_, text, _) in result])
        texts.append(text)

    # Save the texts into a file
    raw_text_path = os.path.join(raw_output_folder_path, f"{os.path.splitext(os.path.basename(image_path))[0]}_raw.txt")
    with open(raw_text_path, 'w', encoding='utf-8') as file:
        file.write('\n'.join(texts))
    
    return texts

def process_images():
    for image_name in os.listdir(input_folder_path):
        image_path = os.path.join(input_folder_path, image_name)
        if os.path.isfile(image_path):
            print(f"Processing {image_name}...")

            # Detect speech bubbles
            bubble_results = model.predict(image_path)
            bubble_result = bubble_results[0]  # Assuming one image is processed

            text_results = text_segmentation.predict(image_path)
            # Assuming one image is processed
            text_result = text_results[0]

            # Crop and OCR speech bubbles
            texts = crop_and_ocr(image_path, bubble_result.boxes)

            # Translate the text to English
            translated_texts = [GoogleTranslator(source='auto', target='en').translate(text) for text in texts]
            translated_text_path = os.path.join(raw_output_folder_path, os.path.splitext(image_name)[0] + '_translated.txt')
            
            with open(translated_text_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(translated_texts))

            # Inpaint and overlay translated text
            inpainted_image = draw_inpaint(image_path, text_result.boxes)
            inpainted_image_path = os.path.join(os.path.abspath(masked_folder_path), os.path.splitext(image_name)[0] + '_inpainted.png')
            cv2.imwrite(inpainted_image_path, inpainted_image)
            
            # Draw translated text onto image and save file
            draw_translation(inpainted_image_path, translated_texts, text_result.boxes)
            
            print(f"Finished processing {image_name}.")

# Correct the function call with proper arguments
process_images()
