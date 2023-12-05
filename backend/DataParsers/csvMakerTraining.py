import os
import csv
import re

# Define the directory containing the song files
directory_path = r'C:\Users\bardi\OneDrive\Documents\CST_Sem3\Nebula\Nebula\dataset\BIGDATA'

# Define the instrument abbreviations to check (excluding 'dru' for special handling)
instrument_abbreviations = ["cel", "cla", "flu", "gac", "gel", "pia", "sax", "tru", "vio", "dru"]

# Initialize the data list with the header row
data = [['FileName'] + instrument_abbreviations + ['dru']]

# Iterate through the song files in the directory
for filename in os.listdir(directory_path):
    # Check if it's a song file
    if filename.lower().endswith((".wav", ".mp3", ".flac", ".aac")):
        # Use regular expression to extract instruments from the file name
        instruments = re.findall(r'\[([^\]]*)\]', filename)

        file_data = [filename]

        # Check each instrument abbreviation and see if it is present in the file name
        for instrument in instrument_abbreviations:
            if instrument in instruments:
                file_data.append('1')
            else:
                file_data.append('0')

        # Append the file data to the main data list
        data.append(file_data)

# Create a CSV file for writing
output_file = 'MultiTraining.csv'

# Write the data to the CSV file with UTF-8 encoding
with open(output_file, 'w', newline='', encoding='utf-8') as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerows(data)

print(f"CSV file '{output_file}' has been created.")
