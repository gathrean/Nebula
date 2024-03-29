import csv

# Define the list of instruments and their corresponding numerical identifiers
instruments = {
    "accordion": 0, "banjo": 1, "bass": 2, "cello": 3, "clarinet": 4,
    "cymbals": 5, "drums": 6, "flute": 7, "guitar": 8, "mallet_percussion": 9,
    "mandolin": 10, "organ": 11, "piano": 12, "saxophone": 13, "synthesizer": 14,
    "trombone": 15, "trumpet": 16, "ukulele": 17, "violin": 18, "voice": 19
}

# Input and output file paths
input_file_path = r'C:\Users\bardi\OneDrive\Documents\CST_Sem3\Nebula\Nebula\dataset\openmic-2018-aggregated-labels.csv'
second_csv_path = r'C:\Users\bardi\OneDrive\Documents\CST_Sem3\All data\openmic-2018-v1.0.0\openmic-2018\partitions\split01_test.csv'
output_file_path = 'SpotifyTrain.csv'

# Read the second CSV file and store sample keys in a set
sample_keys_set = set()
with open(second_csv_path, mode='r', newline='') as second_csv_file:
    csv_reader = csv.reader(second_csv_file)
    for row in csv_reader:
        sample_key = row[0]
        sample_keys_set.add(sample_key)

# Initialize a dictionary to store the instrument usage for each sample key
sample_key_instruments = {}

# Read and process the input CSV file
with open(input_file_path, mode='r', newline='') as input_file:
    csv_reader = csv.reader(input_file)
    next(csv_reader)  # Skip the header row

    for row in csv_reader:
        sample_key = row[0]
        if sample_key in sample_keys_set:  # Check if sample key is in the set
            instrument = row[1]
            relevance = float(row[2])

            if sample_key not in sample_key_instruments:
                sample_key_instruments[sample_key] = [0] * len(instruments)

            if relevance > 0:
                instrument_index = instruments.get(instrument)
                if instrument_index is not None:
                    sample_key_instruments[sample_key][instrument_index] = 1

# Write the results to the output CSV file
with open(output_file_path, mode='w', newline='') as output_file:
    csv_writer = csv.writer(output_file)
    
    # Write the header row
    header_row = ["FileName"] + list(instruments.keys())
    csv_writer.writerow(header_row)
    
    # Write data rows
    for sample_key, instrument_usage in sample_key_instruments.items():
        file_name = f"{sample_key}.ogg"
        csv_row = [file_name] + instrument_usage
        csv_writer.writerow(csv_row)

print("Output CSV file has been created.")
