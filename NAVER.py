from bs4 import BeautifulSoup
import requests
import os

# Load the HTML content from your file
html_file_path = 'T:\\Projects\\speechBubbleOCR\\신의 탑 - 3부 188화 _ 네이버 웹툰.html'
with open(html_file_path, 'r', encoding='utf-8') as file:
    html_content = file.read()

# Parse the HTML content
soup = BeautifulSoup(html_content, 'html.parser')

# Find the 'wt_viewer' class and then find all <img> tags within it
wt_viewer_section = soup.find(class_="wt_viewer")
image_urls = [img['src'] for img in wt_viewer_section.find_all('img') if img.get('src')]

# Directory to save images
save_dir = 'T:\\Projects\\speechBubbleOCR\\rawImg\\Chapter'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Function to download and save images
def download_images(image_urls, save_dir):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3',
        'Referer': 'https://www.naver.com'
    }
    for index, url in enumerate(image_urls):
        try:
            response = requests.get(url, headers=headers)
            if response.headers['Content-Type'].startswith('image/'):
                image_name = url.split('/')[-1]
                file_path = os.path.join(save_dir, image_name)
                with open(file_path, "wb") as file:
                    file.write(response.content)
                print(f"Downloaded {file_path}")
            else:
                print(f"Skipped non-image URL: {url}")
        except Exception as e:
            print(f"Failed to download {url}: {e}")

# Call the function to download images
download_images(image_urls, save_dir)
