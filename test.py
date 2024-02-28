import os
import cv2
from cv2 import inpaint, INPAINT_TELEA
import numpy as np
import easyocr
from deep_translator import GoogleTranslator

# Directories setup
input_folder_path = 'rawImg'
masked_folder_path = 'masked'  # For inpainted images
output_folder_path = 'output'  # For text files

# Ensure output and masked directories exist
os.makedirs(output_folder_path, exist_ok=True)
os.makedirs(masked_folder_path, exist_ok=True)

# Initialize the OCR reader for Korean
reader = easyocr.Reader(['ko'])


def reduce_noise(image_path):
    """Apply Gaussian Blur to reduce image noise."""
    image = cv2.imread(image_path)
    blurred_image = cv2.GaussianBlur(image, (5, 5), 0)
    cv2.imwrite(image_path.replace('.png', '_blurred.png'), blurred_image)
    return blurred_image

def apply_threshold(image_path):
    """Convert image to binary using Otsu's thresholding."""
    image = cv2.imread(image_path, 0)  # 0 to read image in grayscale mode
    _, binary_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.imwrite(image_path.replace('.png', '_thresholded.png'), binary_image)
    return binary_image

def correct_skew(image_path):
    """Correct image skew."""
    image = cv2.imread(image_path, 0)
    coords = np.column_stack(np.where(image > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    cv2.imwrite(image_path.replace('.png', '_deskewed.png'), rotated)
    return rotated

def remove_text_with_inpainting(image_path, bboxes, masked_folder):
    """Remove text from an image using inpainting, given bounding boxes, and save to a separate folder."""
    image = cv2.imread(image_path)
    mask = np.zeros(image.shape[:2], dtype=np.uint8)  # Create a mask with the same dimensions as the image, but single channel

    padding = 10  # Padding to expand the bounding box, can be adjusted as needed

    for bbox in bboxes:
        # Extend the bounding box by the defined padding
        top_left = (max(int(bbox[0][0]) - padding, 0), max(int(bbox[0][1]) - padding, 0))
        bottom_right = (min(int(bbox[2][0]) + padding, image.shape[1]), min(int(bbox[2][1]) + padding, image.shape[0]))
        cv2.rectangle(mask, top_left, bottom_right, 255, -1)  # Fill rectangle in mask

    inpainted_image = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)

    # Get the file name without the extension and add the '_inpainted' suffix
    file_root = os.path.splitext(os.path.basename(image_path))[0]
    file_ext = os.path.splitext(image_path)[1]
    inpainted_image_name = f"{file_root}_inpainted{file_ext}"
    inpainted_image_path = os.path.join(masked_folder, inpainted_image_name)
    cv2.imwrite(inpainted_image_path, inpainted_image)


def process_images(input_folder, masked_folder, output_folder):
    for image_name in os.listdir(input_folder):
        image_path = os.path.join(input_folder, image_name)
        if os.path.isfile(image_path):
            print(f"Processing {image_name}...")

            # Perform OCR on the image
            result = reader.readtext(image_path, detail=1)

            bboxes = []
            raw_text = ''
            for (bbox, text, prob) in result:
                if prob >= 0.4:
                    raw_text += text + '\n'
                    bboxes.append(bbox)

            # Remove text and save inpainted images in the 'masked' folder
            remove_text_with_inpainting(image_path, bboxes, masked_folder)

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