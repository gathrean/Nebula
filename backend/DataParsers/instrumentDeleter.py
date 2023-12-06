import csv

# Function to read the note file and extract song names
def read_note_file(note_file):
    with open(note_file, 'r') as file:
        return [line.strip() for line in file.readlines()]

# Function to create a new CSV file without the removed songs
def create_filtered_csv(input_csv, output_csv, songs_to_remove):
    with open(input_csv, 'r') as input_file, open(output_csv, 'w', newline='') as output_file:
        csv_reader = csv.reader(input_file)
        csv_writer = csv.writer(output_file)

        # Write the header to the new CSV file
        header = next(csv_reader)
        csv_writer.writerow(header)

        # Write rows that do not correspond to songs in the note file
        for row in csv_reader:
            if row[0] not in songs_to_remove:
                csv_writer.writerow(row)

if __name__ == "__main__":
    note_file = r"C:\Users\bardi\Downloads\songs_to_remove.txt"
    input_csv = r"C:\Users\bardi\OneDrive\Documents\CST_Sem3\Nebula\Nebula\dataset\SpotifyTest.csv"
    output_csv = "FinalDataTest.csv"

    songs_to_remove = read_note_file(note_file)
    create_filtered_csv(input_csv, output_csv, songs_to_remove)

    print(f"Lines for songs in the note file have been removed, and the result is saved in '{output_csv}'.")
