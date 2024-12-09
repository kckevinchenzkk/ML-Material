import os

# Define the source directory
source_dir = "Materials_data"  # Replace with your actual path

# Define the specific mappings for folder renaming
specific_mappings = {
    "10四": "14",
    "10五": "15",
    "10二": "12",
    "10一": "11",
    "10三": "13",
}

# Iterate through all items in the source directory
for folder in os.listdir(source_dir):
    folder_path = os.path.join(source_dir, folder)

    # Process only directories
    if os.path.isdir(folder_path):
        # Check if the folder name starts with any of the specific mappings
        for prefix, new_prefix in specific_mappings.items():
            if folder.startswith(prefix):
                # Replace the prefix with the new value
                new_folder_name = folder.replace(prefix, new_prefix, 1)
                new_folder_path = os.path.join(source_dir, new_folder_name)
                
                # Rename the folder
                os.rename(folder_path, new_folder_path)
                print(f"Renamed: {folder} -> {new_folder_name}")
                break

print("Folder renaming completed.")