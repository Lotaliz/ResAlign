import pandas as pd

# Load the JSON file into a pandas DataFrame
df = pd.read_json('../data/advbench/llama-f.json')

# Calculate the average of the 'harm_score' column
average_harm_score = df['harm_score'].mean()

print(f"The average harm_score is: {average_harm_score}")

