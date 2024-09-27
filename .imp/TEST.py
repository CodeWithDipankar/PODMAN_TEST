import pandas as pd
import os

print("Hi I am Dipankar Mandal")

# Read the CSV file from the /app/input directory inside the container
input_file_path = '/app/input/test.csv'  # Fixed path for the input CSV file
data = pd.read_csv(input_file_path)

# Print the original data without index
print("\nOriginal Data:\n", data.to_string(index=False))

# Example of modifying the DataFrame
data['Loyalty Status'] = data['Purchase Amount ($)'].apply(lambda x: 'Gold' if x > 150 else 'Silver')

# Ensure the output directory exists inside the container (/app/output is where the local OUTPUT folder is mounted)
output_dir = '/app/output'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Save the modified DataFrame to a new CSV file in the output directory
output_file_path = os.path.join(output_dir, 'modified_test.csv')
data.to_csv(output_file_path, index=False)

print(f"\nModified Data saved as '{output_file_path}'.")
