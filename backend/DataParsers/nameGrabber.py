import os
import csv

# Define the directory containing the files
directory = r"C:\Users\bardi\OneDrive\Documents\CST_Sem3\Nebula\Nebula\dataset\Additions\ElectricGuitar"

# Define the instrument name to be added to the beginning of each file
instrument_name = "[gel]scale"

# Create a list to store the new file names
new_file_names = []

# Iterate over the files in the directory
for filename in os.listdir(directory):
    # Check if the item is a file (not a directory)
    if os.path.isfile(os.path.join(directory, filename)):
        # Add the instrument name to the beginning of the file name
        new_filename = f"{instrument_name}_{filename}"
        new_file_names.append(new_filename)

        # Rename the file
        os.rename(os.path.join(directory, filename), os.path.join(directory, new_filename))

# Create a new CSV file and write the new file names to a column
csv_filename = "GEL.csv"
with open(csv_filename, mode='w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(["Original File Name", "New File Name"])  # Header row

    # Write the original and new file names to the CSV
    for original, new in zip(os.listdir(directory), new_file_names):
        writer.writerow([original, new])

print(f"File names updated and saved to {csv_filename}")
