import shutil
import os

# Function to copy the contents of a directory to another directory
def copy_directory_contents(src_dir, dest_dir):
    try:
        for item in os.listdir(src_dir):
            src_item = os.path.join(src_dir, item)
            dest_item = os.path.join(dest_dir, item)

            if os.path.isdir(src_item):
                if not os.path.exists(dest_item):
                    os.makedirs(dest_item)
                copy_directory_contents(src_item, dest_item)
            else:
                if not os.path.exists(dest_item):
                    shutil.copy2(src_item, dest_item)
        print(f"Successfully copied contents from '{src_dir}' to '{dest_dir}'")
    except Exception as e:
        print(f"Error copying contents from '{src_dir}' to '{dest_dir}': {str(e)}")

# List of source directories
source_directories = [
    r'C:\Users\bardi\OneDrive\Documents\CST_Sem3\Nebula\Nebula\dataset\Additions\ElectricGuitar',
    r'C:\Users\bardi\OneDrive\Documents\CST_Sem3\Nebula\Nebula\dataset\Additions\AcousticGuitar',
]

# Destination directory
destination_directory = r'C:\Users\bardi\OneDrive\Documents\CST_Sem3\Nebula\Nebula\dataset\BIGDATA'

# Create the destination directory if it doesn't exist
if not os.path.exists(destination_directory):
    os.makedirs(destination_directory)

# Copy contents of each source directory to the destination directory
for src_dir in source_directories:
    copy_directory_contents(src_dir, destination_directory)