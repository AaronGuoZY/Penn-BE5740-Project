import re
import pandas as pd

# Define the path to the log file
log_file_path = '/home/aarongzy/class/src/train-20240425.log'

# Regular expression to extract losses from each line
loss_pattern = re.compile(r"Discriminator Loss: ([-\d.]+), Generator Loss: ([-\d.]+)")

# Lists to store the extracted losses
discriminator_losses = []
generator_losses = []

# Open the log file and process each line
with open(log_file_path, 'r') as file:
    for line in file:
        match = loss_pattern.search(line)
        if match:
            # Append the losses to their respective lists
            discriminator_losses.append(float(match.group(1)))
            generator_losses.append(float(match.group(2)))

# Create a DataFrame from the extracted data
df = pd.DataFrame({
    'Discriminator Loss': discriminator_losses,
    'Generator Loss': generator_losses
})

# Define the path for the CSV file (same directory as the log file)
csv_file_path = log_file_path.replace('.log', '.csv')

# Save the DataFrame to a CSV file
df.to_csv(csv_file_path, index=False)

print(f"CSV file has been created at: {csv_file_path}")
