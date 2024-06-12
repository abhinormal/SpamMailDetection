import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle

# Load data from CSV file into a pandas dataframe
raw_mail_data = pd.read_csv('mail_data.csv')

# Replace null values with empty strings
mail_data = raw_mail_data.where((pd.notnull(raw_mail_data)), '')

# Replace 'spam' with 0 and 'ham' with 1
mail_data.loc[mail_data['Category'] == 'spam', 'Category'] = 0
mail_data.loc[mail_data['Category'] == 'ham', 'Category'] = 1

# Separate data into texts (X) and labels (Y)
X = mail_data['Message']
Y = mail_data['Category']

# Split data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)

# Transform the text data into feature vectors
feature_extraction = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)
X_train_features = feature_extraction.fit_transform(X_train)
X_test_features = feature_extraction.transform(X_test)

# Convert Y_train and Y_test values to integers
Y_train = Y_train.astype(int)
Y_test = Y_test.astype(int)

# Train the logistic regression model
model = LogisticRegression()
model.fit(X_train_features, Y_train)

# Evaluate the model on training data
prediction_on_training_data = model.predict(X_train_features)
accuracy_on_training_data = accuracy_score(Y_train, prediction_on_training_data)
print(f'Accuracy on training data: {accuracy_on_training_data:.4f}')

# Evaluate the model on test data
prediction_on_test_data = model.predict(X_test_features)
accuracy_on_test_data = accuracy_score(Y_test, prediction_on_test_data)
print(f'Accuracy on test data: {accuracy_on_test_data:.4f}')

# Save the model and vectorizer to disk
with open('spam_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

with open('vectorizer.pkl', 'wb') as vec_file:
    pickle.dump(feature_extraction, vec_file)
