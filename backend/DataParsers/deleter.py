import csv
import os

# Directory path where you want to delete files
directory_path = r"C:\Users\bardi\OneDrive\Documents\CST_Sem3\Nebula\Nebula\dataset\BIGDATA"

# CSV file path
csv_file_path = r"C:\Users\bardi\OneDrive\Documents\CST_Sem3\Nebula\Nebula\dataset\MultiTraining.csv"

# Function to read valid file names from the CSV
def get_valid_file_names(csv_path):
    valid_file_names = set()
    with open(csv_path, 'r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            valid_file_names.add(row['FileName'])
    return valid_file_names

# Get the set of valid file names from the CSV
valid_file_names_set = get_valid_file_names(csv_file_path)

# Iterate through files in the directory and delete those not in the CSV
for filename in os.listdir(directory_path):
    file_path = os.path.join(directory_path, filename)
    if os.path.isfile(file_path) and filename not in valid_file_names_set:
        print(f"Deleting {filename}")
        os.remove(file_path)
        print(f"{filename} deleted")

print("Done")
