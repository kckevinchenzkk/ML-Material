import os
import shutil

# Define the source directory
source_dir = "backup"

# Iterate through all groups in the source directory
for group in os.listdir(source_dir):
    group_path = os.path.join(source_dir, group)
    
    # Ensure we are processing directories
    if os.path.isdir(group_path):
        # Iterate through all images in the group directory
        for image_file in os.listdir(group_path):
            if image_file.endswith(".jpg"):  # Process only .jpg files
                # Construct folder name and paths
                folder_name = f"{group.replace('第', '').replace('组', '')}-{image_file.split('-')[0]}"
                new_folder_path = os.path.join(source_dir, folder_name)
                os.makedirs(new_folder_path, exist_ok=True)  # Create the new folder
                
                # Copy the image 20 times into the new folder
                for i in range(1, 21):
                    destination_file = os.path.join(new_folder_path, f"{image_file.split('.')[0]}_{i}.jpg")
                    shutil.copy(os.path.join(group_path, image_file), destination_file)

print("Folders and images have been created successfully.")