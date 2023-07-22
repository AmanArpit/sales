import pickle
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, KFold

# Read the dataset
data = pd.read_csv(r"C:\Users\91700\OneDrive\Desktop\vgsales.csv")
df = pd.DataFrame(data)

# Drop 'Name' and 'Platform' columns as they are not needed
df.drop(['Name', 'Platform'], axis=1, inplace=True)

# Fill missing values in the 'Year' and 'Publisher' columns
df['Year'].fillna(df['Year'].mean(), inplace=True)
df['Publisher'].fillna(df['Publisher'].mode().iloc[0], inplace=True)

# Preprocess categorical columns using LabelEncoder
le = LabelEncoder()
df['Genre'] = le.fit_transform(df['Genre'])
df['Publisher'] = le.fit_transform(df['Publisher'])

# Define the input and target columns for training the model
input_columns = ['NA_Sales', 'JP_Sales', 'EU_Sales', 'Other_Sales']
target_column = 'Global_Sales'

# Select the input features and target variable
X = df[input_columns].values
y = df[target_column].values

# Perform data preprocessing - scale features using StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Initialize the Ridge regression model with regularization (alpha=0.1)
ridge_model = Ridge(alpha=0.1)

# Define the number of folds for cross-validation
num_folds = 5

# Initialize the KFold object
kf = KFold(n_splits=num_folds, shuffle=True, random_state=8)

# Lists to store the evaluation metrics for each fold
train_scores = []
test_scores = []

# Perform k-fold cross-validation
for train_index, test_index in kf.split(X_scaled):
    X_train, X_test = X_scaled[train_index], X_scaled[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Train the model on the training data
    ridge_model.fit(X_train, y_train)

    # Evaluate the model on the training and test data
    train_score = ridge_model.score(X_train, y_train)
    test_score = ridge_model.score(X_test, y_test)

    train_scores.append(train_score)
    test_scores.append(test_score)

# Calculate the mean and standard deviation of the evaluation metrics
mean_train_score = np.mean(train_scores)
mean_test_score = np.mean(test_scores)
std_train_score = np.std(train_scores)
std_test_score = np.std(test_scores)

print("Mean train score:", mean_train_score)
print("Mean test score:", mean_test_score)
print("Standard deviation of train scores:", std_train_score)
print("Standard deviation of test scores:", std_test_score)
