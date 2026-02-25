import os
import shutil
import glob


ablated_dir = "./data/ablated"

# Move ESD and FMN checkpoints
old_object_dir = os.path.join(ablated_dir, "unlearned_ckpt_object")

# Delete the now-empty directory
shutil.move(os.path.join(old_object_dir, "ESD_ckpt"), os.path.join(ablated_dir, "ESD_ckpt"))
shutil.move(os.path.join(old_object_dir, "FMN_ckpt"), os.path.join(ablated_dir, "FMN_ckpt"))

# Handle others folder creation and wildcars
others_dir = os.path.join(ablated_dir, "ESD_ckpt/others")
os.makedirs(others_dir, exist_ok=True)

# Mv equivalent
source_wildcard = os.path.join(ablated_dir, "files/pretrained/SD-1-4/ESD_ckpt/*")
for file_path in glob.glob(source_wildcard):
    shutil.move(file_path, others_dir)


# Move forget_me_ckpt
shutil.move(os.path.join(ablated_dir, "files/pretrained/SD-1-4/forget_me_ckpt"), 
            os.path.join(ablated_dir, "forget_me_ckpt"))

# Delete the files directory
shutil.rmtree(os.path.join(ablated_dir, "files"))

# Move style classifier folder
shutil.move(os.path.join(ablated_dir, "results/checkpoint-2800"), 
            os.path.join(ablated_dir, "style_classifier/"))

# Delete the results directory
shutil.rmtree(os.path.join(ablated_dir, "results"))

print("Operations completed successfully!")
