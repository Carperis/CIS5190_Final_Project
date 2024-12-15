import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dataset
data = pd.read_csv("data/data.csv")

# Split the data into train and test sets (80-20 split)
train, test = train_test_split(data, test_size=0.2, random_state=42)

# Save the train and test sets to CSV files
train.to_csv("data/train.csv", index=False)
test.to_csv("data/test.csv", index=False)
