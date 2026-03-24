import gdown
import os
import zipfile
import shutil
import subprocess
import time

# Run this script from root!

# PLEASE, before installing double check URLs from https://github.com/matanr/Memories_of_Forgotten_Concepts/
ablated_models_object_url= "https://drive.google.com/uc?id=1e5aX8gkC34YaHGR0S1-EQwBmUXiAPvpE" #original link
ablated_models_object_url2= "https://drive.google.com/uc?id=1NsxoV7B8z8UVxC9-BboD0hUzpkuhMJxX"
ablated_models_others_url = "https://drive.google.com/uc?id=1yeZNJ8MoHsisdZmt5lbnG_kSgl5xned0"
vangogh_styleclassifier_url = "https://drive.google.com/uc?id=1me_MOrXip1Xa-XaUrPZZY7i49pgFe1po"

def download_and_extract(url, output_dir, filename, label=""):
  print(f"Downloading {label}")

  output_file = os.path.join(output_dir, filename)

  if not os.path.exists(output_file):
    # Doesn't work in Colab:
    gdown.download_folder(url, output=output_dir)
    # But the following works:
    #response = subprocess.run(["gdown", "--fuzzy","-O", output_file, url], check=True, capture_output=True)

  with zipfile.ZipFile(output_file, 'r') as zip_ref:
      zip_ref.extractall(path=output_dir)
  
  print(f"Successfully download and extracted {label} under {output_dir}")

# Download ablated models
target_dir = "./data/ablated"
os.makedirs(target_dir, exist_ok=True)
download_and_extract(ablated_models_object_url2, target_dir, "unlearned_ckpt_object.zip", "Ablated Models 1/2")
time.sleep(5)
download_and_extract(ablated_models_others_url, target_dir, "esd_fmn_models.zip", "Ablated Models 2/2")

# Download Style Classifier for Van Gogh
target_dir="./data/"

download_and_extract(vangogh_styleclassifier_url, target_dir, "vangogh_classifier.zip", "Vangogh Classifier")
