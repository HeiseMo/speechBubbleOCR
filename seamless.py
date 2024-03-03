from PIL import Image, ImageDraw
import natsort
import os

def combine_image_batch(folder_path, output_folder, output_filename_prefix, batch_size, max_dimension):
    images = []
    
    # Load and sort images from the specified folder
    image_files = natsort.natsorted([filename for filename in os.listdir(folder_path) if filename.endswith('.jpg')])
    for filename in image_files:
        image_path = os.path.join(folder_path, filename)
        images.append(Image.open(image_path))

    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Create batches and save combined images
    for i in range(0, len(images), batch_size):
        batch_images = images[i:i + batch_size]
        canvas_width = max(img.width for img in batch_images) * 1  # Single column layout
        canvas_height = sum(img.height for img in batch_images)
        canvas = Image.new('RGB', (canvas_width, canvas_height), (255, 255, 255))
        draw = ImageDraw.Draw(canvas)

        x_offset, y_offset = 0, 0

        for img in batch_images:
            canvas.paste(img, (x_offset, y_offset))
            y_offset += img.height

        # Save the resulting image in the output folder
        output_filename = f"{output_filename_prefix}_{i // batch_size + 1}.jpg"
        output_path = os.path.join(output_folder, output_filename)
        canvas.save(output_path)

if __name__ == "__main__":
    folder_path = "downloaded_images"
    output_folder = "combined_comic_folder"
    output_filename_prefix = "combined_comic_batch"
    batch_size = 80
    max_dimension = 65500

    combine_image_batch(folder_path, output_folder, output_filename_prefix, batch_size, max_dimension)
