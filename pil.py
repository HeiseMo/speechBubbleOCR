from PIL import Image, ImageDraw, ImageFont

# Load your image
image = Image.open('T:\Projects\speechBubbleOCR\\1708914227_53842016081709.jpg')
draw = ImageDraw.Draw(image)

# Specify the font and size
font_path = 'ComicNeue\ComicNeue-Regular.ttf'  # Make sure this path is correct
font_size = 24  # Adjust the size as needed
font = ImageFont.truetype(font_path, font_size)

# Use the textbbox method to get text width and height
text = "Your text here"
bbox = draw.textbbox((0, 0), text, font=font)
text_width = bbox[2] - bbox[0]
text_height = bbox[3] - bbox[1]

# Now, you can use text_width and text_height to position your text
# For example, to center the text on the image
image_width, image_height = image.size
x = (image_width - text_width) / 2
y = (image_height - text_height) / 2

# Draw the text on the image at the calculated position
draw.text((x, y), text, font=font, fill=(0, 0, 0))  # Adjust color as needed

# Save or show the image
image.show()  # Or use image.save('path_to_save_image.jpg')
