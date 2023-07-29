import csv
from typing import List, Dict, Any

def write_to_csv(filename: str, data: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    # Open the CSV file
    with open(filename, 'w', newline='') as csvfile:
        # Create a CSV writer
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Write the header
        writer.writeheader()

        # Write the data
        writer.writerows(data)
