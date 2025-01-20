import os
import urllib.request

def download_file():
    # Directory and URL setup
    directory = "checkpoints"
    file_url = "https://ml-site.cdn-apple.com/models/depth-pro/depth_pro.pt"

    # Create directory
    os.makedirs(directory, exist_ok=True)
    print(f"Directory '{directory}' created.")

    # Download the file
    file_path = os.path.join(directory, os.path.basename(file_url))
    print(f"Downloading {file_url} to {file_path}...")
    urllib.request.urlretrieve(file_url, file_path)
    print("Download completed.")

