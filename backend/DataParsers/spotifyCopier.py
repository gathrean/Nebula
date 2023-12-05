import os
import shutil

# Source directory containing files and subdirectories with songs
source_directory = r'C:\Users\bardi\OneDrive\Documents\CST_Sem3\All data\openmic-2018-v1.0.0\openmic-2018\audio'

# Destination folder where all songs will be copied
destination_folder = r'C:\Users\bardi\OneDrive\Documents\CST_Sem3\Nebula\Nebula\dataset\Spotify'

# Ensure the destination folder exists; create it if it doesn't
os.makedirs(destination_folder, exist_ok=True)

# Walk through the source directory and its subdirectories
for root, _, files in os.walk(source_directory):
    for file in files:
        source_file_path = os.path.join(root, file)
        
        # You can specify criteria to determine if a file is a song (e.g., file extension)
        # For example, let's copy only .mp3 files as songs
        if file.endswith('.ogg'):
            # Construct the destination path for the song file
            destination_file_path = os.path.join(destination_folder, file)
            
            # Copy the song file to the destination folder
            shutil.copy2(source_file_path, destination_file_path)
            print(f"Copied: {source_file_path} -> {destination_file_path}")

print("All songs have been copied to the destination folder.")
