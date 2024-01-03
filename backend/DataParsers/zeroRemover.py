import csv

# Input and output file names
input_file = r"C:\Users\bardi\OneDrive\Documents\CST_Sem3\Nebula\Nebula\dataset\FinalData3.csv"
output_file = "FinalData3.csv"

# Open the input and output files
with open(input_file, mode='r') as input_csv, open(output_file, mode='w', newline='') as output_csv:
    # Create CSV reader and writer objects
    csv_reader = csv.DictReader(input_csv)
    fieldnames = csv_reader.fieldnames
    csv_writer = csv.DictWriter(output_csv, fieldnames=fieldnames)
    
    # Write the header to the output file
    csv_writer.writeheader()
    
    # Iterate through each row in the input CSV
    for row in csv_reader:
        # Check if any instrument column has a value other than 0
        all_zeros = all(row[instrument] == '0' for instrument in fieldnames[1:])
        
        # If not all zeros, write the row to the output CSV
        if not all_zeros:
            csv_writer.writerow(row)

print("Rows with all 0s in instrument columns removed. Output written to", output_file)
