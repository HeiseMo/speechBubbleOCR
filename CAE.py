import os
import sys
import zipfile
import rarfile
import patoolib
import re

# Function to extract comic book archive files
def extract_comic_archive(file_path, output_folder=None):
    file_name = os.path.basename(file_path)
    base_name, file_extension = os.path.splitext(file_name)
    if output_folder is None:
        output_folder = os.path.join('rawImg', base_name)
    os.makedirs(output_folder, exist_ok=True)

    if file_extension.lower() == '.cbz':
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(output_folder)
    elif file_extension.lower() == '.cbr':
        rarfile.UNRAR_TOOL = "unrar"  # Ensure this points to the correct unrar executable path
        with rarfile.RarFile(file_path, 'r') as rar_ref:
            rar_ref.extractall(output_folder)
    else:
        raise ValueError(f"Unsupported file extension: {file_extension}")

    print(f"Extraction complete. Files are in: {output_folder}")

# Function to create a .cbz file from a directory of images
def create_cbz_from_directory(directory_path, output_cbz_name):
    with zipfile.ZipFile(output_cbz_name, 'w', zipfile.ZIP_DEFLATED) as cbz_file:
        for root, dirs, files in os.walk(directory_path):
            files.sort(key=lambda x: [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', x)])
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
                    file_path = os.path.join(root, file)
                    cbz_file.write(file_path, arcname=os.path.relpath(file_path, directory_path))
    print(f"{output_cbz_name} created from images in {directory_path}.")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python cbz.py <command> <path> [output_path]")
        sys.exit(1)

    command = sys.argv[1]
    path = sys.argv[2]
    output_path = sys.argv[3] if len(sys.argv) > 3 else None

    if command == "extract":
        extract_comic_archive(path, output_path)
    elif command == "package":
        if output_path is None:
            print("Output path is required for packaging.")
            sys.exit(1)
        create_cbz_from_directory(path, output_path)
    else:
        print("Unknown command. Use 'extract' or 'package'.")
