import os
import csv

# Define the directory containing the text files
directory_path = r'C:\Users\bardi\OneDrive\Documents\CST_Sem3\Nebula\Nebula\dataset\Multi'

# Define the instrument abbreviations
instrument_abbreviations = ["cel", "cla", "flu", "gac", "gel", "org", "pia", "sax", "tru", "vio", "voi"]

# Initialize the data list with the header row
data = [['FileName'] + instrument_abbreviations]

# Iterate through the text files in the directory
for filename in os.listdir(directory_path):
    if filename.endswith(".txt"):
        # Read the content of the text file
        with open(os.path.join(directory_path, filename), 'r') as file:
            content = file.read().strip()

        # Replace the '.txt' extension with '.wav' in the filename
        filename_wav = filename.replace('.txt', '.wav')

        file_data = [filename_wav]

        # Check each instrument abbreviation and see if it is present in the file content
        for instrument in instrument_abbreviations:
            if instrument in content:
                file_data.append('1')
            else:
                file_data.append('0')

        # Append the file data to the main data list
        data.append(file_data)

# Remove spaces from the titles in the data list
data = [[item.replace(" ", "") for item in row] for row in data]

# Create a CSV file for writing with spaces removed from the title
output_file = 'MultiTest.csv'

# Write the data to the CSV file with UTF-8 encoding
with open(output_file, 'w', newline='', encoding='utf-8') as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerows(data)

print(f"CSV file '{output_file}' has been created.")
