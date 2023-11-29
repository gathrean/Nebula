import os

# Specify the directory containing the music files
directory = r"C:\Users\bardi\OneDrive\Documents\CST_Sem3\Nebula\Nebula\dataset\MultiSongs"

# List all files in the directory
files = os.listdir(directory)

# Iterate through the files and rename them, removing spaces
for filename in files:
    # Check if the file is a music file (you can customize this check based on file extensions)
    if filename.endswith(('.mp3', '.wav', '.flac', '.ogg', '.m4a', '.wma')):
        # Remove spaces from the filename
        new_name = filename.replace(" ", "")
        
        # Join the directory path with the new filename
        new_path = os.path.join(directory, new_name)
        
        # Rename the file
        os.rename(os.path.join(directory, filename), new_path)
        
        print(f"Renamed: {filename} -> {new_name}")
    else:
        print(f"Skipped: {filename} (not a supported music file)")

print("All supported music files renamed successfully.")
