import json
import csv

# Define CSV file name
csv_filename = "explication_data2.csv"

with open("data/data_20250222090435.json", "r", encoding="utf-8") as file:
    data=json.load(file)

# Open CSV file for writing
with open(csv_filename, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    
    # Write header
    writer.writerow(["Word", "Word Sense", "Examples", "Explication 1", "Explication 2"])
    
    # Process JSON data
    for entry in data["data"]:
        word = entry["word"]
        for sense in entry["senses"]:
            definition = sense["definition"]
            examples = "\n".join([f"{ex["sentence"]} (Source: {ex["source"]})" for ex in sense["examples"]])
            explication_1 = sense["responses"][0] if len(sense["responses"]) > 0 else ""
            explication_2 = sense["responses"][1] if len(sense["responses"]) > 1 else ""
            
            # Write row to CSV
            writer.writerow([word, definition, examples, explication_1, explication_2])

print(f"CSV file '{csv_filename}' has been created successfully.")