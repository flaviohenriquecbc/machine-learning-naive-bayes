import pandas as pd

# Read the dataset from: https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection
table = pd.read_table('dataset/SMSSpamCollection', names=['label', 'message'])

# Normalize column label ham : 0 and spam : 1
table.label = table.label.map({'ham': 0, 'spam': 1})
print table
