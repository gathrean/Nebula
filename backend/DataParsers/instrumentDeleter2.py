import csv

# Define the input and output file paths
input_file = r'C:\Users\bardi\OneDrive\Documents\CST_Sem3\Nebula\Nebula\output.csv'
output_file = 'newFinalData.csv'

# Define the list of instruments you want to check
instruments_to_check = ['accordion', 'banjo', 'mallet_percussion', 'mandolin',
                        'organ', 'synthesizer', 'ukulele', 'saxophone', 'trombone', 'cymbals']

# Initialize an empty list to store the filtered data
filtered_data = []

# Read the input CSV file
with open(input_file, mode='r', newline='') as file:
    reader = csv.DictReader(file)
    
    # Create a new header that includes only the instruments not in the check list
    new_header = ['FileName'] + [instrument for instrument in reader.fieldnames if instrument not in instruments_to_check]
    filtered_data.append(new_header)
    
    for row in reader:
        # Check if any of the chosen instruments have a '1' in the row
        should_keep = not any(row[instrument] == '1' for instrument in instruments_to_check)
        
        if should_keep or any(row[instrument] == '1' for instrument in new_header[1:]):
            # Create a new row with 1s and 0s for the remaining instruments
            new_row = [row['FileName']] + [row[instrument] for instrument in new_header[1:]]
            filtered_data.append(new_row)

# Write the filtered data to the output CSV file
with open(output_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(filtered_data)

print("Filtered data has been written to", output_file)
