from re import DEBUG
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score, roc_auc_score
)
import numpy as np

DEBUG_MODE = True

def convert_labels(dataframe: pd.DataFrame, label_name: str) -> pd.DataFrame:
    """
    Converts the categorical string label column of the given pandas DataFrame into a numerical category.
    """
    if dataframe[label_name].dtype == "object":
        dataframe[label_name] = dataframe[label_name].astype("category").cat.codes
    return dataframe

# Load dataset
TRAIN_DATASET_PATH = "./sample_data/synthetic_insider_threat.csv"
df = pd.read_csv(TRAIN_DATASET_PATH)

# Filter the DataFrame to keep only the rows where the category is 'malicious', 'normal', or 'medical'
valid_categories = ['Malicious', 'Normal', 'Medical']
df = df[df['Category'].isin(valid_categories)]

if DEBUG_MODE: 
  # Print unique values in 'col1'
  unique_values = df['Category'].unique()
  print("Unique values in 'Category' column:")
  print(unique_values)

df = convert_labels(df, "Category")  # Convert labels to numerical categories

if DEBUG_MODE: 
  # Print unique values in 'col1'
  unique_values = df['Category'].unique()
  print("Unique values in 'Category' column after label numerization:")
  print(unique_values)

# Split dataset into train and test sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Convert text data into numerical features using TF-IDF vectorization
vectorizer = TfidfVectorizer(max_features=5000)  # Limit features to avoid overfitting
X_train = vectorizer.fit_transform(train_df["Tweet"])
X_test = vectorizer.transform(test_df["Tweet"])

# Define the target variable (labels)
y_train = train_df["Category"]
y_test = test_df["Category"]

# Train logistic regression model for multiclass classification
model = LogisticRegression(max_iter=1000, multi_class="multinomial", solver="lbfgs")
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)  # Get class probabilities

# Compute confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", conf_matrix)

# Compute evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average="weighted")  # Weighted average for multiclass

# Ensure labels are ordered before passing them to roc_auc_score
ordered_labels = np.sort(df["Category"].unique())
auc = roc_auc_score(y_test, y_pred_proba, multi_class="ovr", labels=ordered_labels)

# Display results
print(f"Accuracy: {accuracy:.4f}")
print(f"F1-score: {f1:.4f}")
print(f"AUC (macro-averaged): {auc:.4f}\n")

# Classification report for detailed metrics per class
print("\nClassification Report:")
print(classification_report(y_test, y_pred))