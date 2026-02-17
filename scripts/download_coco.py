import requests 
from tqdm import tqdm
import os
import zipfile

# Download URLs
url_ann = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
url_data = "http://images.cocodataset.org/zips/val2017.zip"

# DOWNLOAD ANNOTATIONS

target_dir = "./data"
os.makedirs(target_dir, exist_ok=True) # Make sure the directory exists

zip_path = os.path.join(target_dir, "stuff_annotations_trainval2017.zip")

response = requests.get(url_ann, stream=True)
response.raise_for_status()

# Get file size for the progress bar
total_size = int(response.headers.get('content-length', 0))
block_size = 8192 # 8KB chunks

progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True, desc="Downloading MSCoco Annotations")

with open(zip_path, 'wb') as f:
    for chunk in response.iter_content(chunk_size=block_size):
        progress_bar.update(len(chunk))
        f.write(chunk)

progress_bar.close()

print("Succesfully downloaded Annotations for COCO.\nUnzipping the the dir...")

target_dir="./data/coco/"
if not os.path.exists(target_dir):
    os.mkdir(target_dir)

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(target_dir)

if os.path.exists(zip_path):
    os.remove(zip_path)


# DOWNLOAD IMAGES
target_dir = "./data"
zip_path = os.path.join(target_dir, "val2017.zip")

response = requests.get(url_data, stream=True)
response.raise_for_status()

# Get file size for the progress bar
total_size = int(response.headers.get('content-length', 0))
block_size = 8192 # 8KB chunks

progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True, desc="Downloading MSCoco Val Images")

with open(zip_path, 'wb') as f:
    for chunk in response.iter_content(chunk_size=block_size):
        progress_bar.update(len(chunk))
        f.write(chunk)

progress_bar.close()

print("Succesfully downloaded Val Images for COCO.\nUnzipping the the dir...")
target_dir="./data/coco/"

if not os.path.exists(target_dir):
    os.mkdir(target_dir)

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(target_dir)

if os.path.exists(zip_path):
    os.remove(zip_path)

print("Successfully downloaded the MSCOCO")

