import os
import shutil

def find_and_duplicate_images(base_dir):
    # Iterate through directories starting with "copy-1", "copy-2", etc.
    for root, dirs, files in os.walk(base_dir):
        base_dir_name = os.path.basename(root)
        if base_dir_name.startswith("copy-") and base_dir_name[5:].isdigit():
            prefix_number = base_dir_name.split('-')[1]  # Extract the number after "copy-"
            for file in files:
                if file.endswith(".png") or file.endswith(".jpg"):
                    file_path = os.path.join(root, file)

                    # Create folder name based on image name and prefix number
                    file_name, file_ext = os.path.splitext(file)
                    new_folder_name = f"{prefix_number}-{file_name}"
                    new_folder_path = os.path.join(root, new_folder_name)

                    # Create the folder if it doesn't exist
                    os.makedirs(new_folder_path, exist_ok=True)

                    # Copy the image 20 times into the new folder
                    for i in range(1, 21):
                        new_file_name = f"{file_name}-{i}{file_ext}"
                        new_file_path = os.path.join(new_folder_path, new_file_name)
                        shutil.copy(file_path, new_file_path)
                    print(f"Processed: {file_path} -> {new_folder_path}")

if __name__ == "__main__":
    base_directory = "./Materials_data"
    if os.path.exists(base_directory):
        find_and_duplicate_images(base_directory)
    else:
        print("The specified directory does not exist.")
