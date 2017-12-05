import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Read the dataset from: https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection
table = pd.read_table('dataset/SMSSpamCollection', names=['label', 'message'])

# Normalize column label ham : 0 and spam : 1
table.label = table.label.map({'ham': 0, 'spam': 1})

# split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(table['message'], 
                                                    table['label'], 
                                                    random_state=1)

print('Number of rows in the total set: {}'.format(table.shape[0]))
print('Number of rows in the training set: {}'.format(X_train.shape[0]))
print('Number of rows in the test set: {}'.format(X_test.shape[0]))

# Instantiate the CountVectorizer method
count_vector = CountVectorizer()

# Fit the training data and then return the matrix
training_data = count_vector.fit_transform(X_train)

# Transform testing data and return the matrix. Note we are not fitting the testing data into the CountVectorizer()
testing_data = count_vector.transform(X_test)

naive_bayes = MultinomialNB()
naive_bayes.fit(training_data, y_train)

# Make the predictions
predictions = naive_bayes.predict(testing_data)

print('Accuracy score: ', format(accuracy_score(y_test, predictions)))
print('Precision score: ', format(precision_score(y_test, predictions)))
print('Recall score: ', format(recall_score(y_test, predictions)))
print('F1 score: ', format(f1_score(y_test, predictions)))
