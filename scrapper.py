import os
import requests
from bs4 import BeautifulSoup

def fetch_image_urls(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    img_tags = soup.find('div', {'id': 'bo_v_con'}).find_all('img')

    image_urls = []
    for img in img_tags:
        src = img.get('src')
        if src:
            image_urls.append(src)

    return image_urls

def download_images(image_urls, folder_name='downloaded_images'):
    os.makedirs(folder_name, exist_ok=True)
    for i, url in enumerate(image_urls):
        try:
            image_data = requests.get(url).content
            filename = os.path.join(folder_name, f'image_{i + 1}.jpg')
            with open(filename, 'wb') as f:
                f.write(image_data)
            print(f'Downloaded: {filename}')
        except Exception as e:
            print(f"Error downloading image {i + 1}: {e}")

if __name__ == '__main__':
    target_url = ''  # Replace with the actual website URL
    image_urls = fetch_image_urls(target_url)
    download_images(image_urls)

