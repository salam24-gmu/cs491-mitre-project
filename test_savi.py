from re import DEBUG
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score, roc_auc_score
)
import numpy as np

DEBUG_MODE = True

# Load dataset
TRAIN_DATASET_PATH = r"C:/Users/salam/OneDrive/Documents/School/Spring 2025/CS491 Industry Design/cs491-mitre-project/synthetic_insider_threat.csv"
df = pd.read_csv(TRAIN_DATASET_PATH)
df = pd.read_csv(TRAIN_DATASET_PATH)

# Filter dataset to include only specific categories
valid_categories = ['Malicious', 'Normal', 'Medical']
df = df[df['Category'].isin(valid_categories)]

def convert_labels(dataframe: pd.DataFrame, label_name: str) -> pd.DataFrame:
    """
    Converts the categorical string label column of the given pandas DataFrame into a numerical category.
    """
    if dataframe[label_name].dtype == "object":
        dataframe[label_name] = dataframe[label_name].astype("category").cat.codes
    return dataframe

df = convert_labels(df, "Category")

# Vectorize text data using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df["Tweet"])
y = df["Category"]

# Perform cross-validation
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
model = LogisticRegression(max_iter=1000, multi_class="multinomial", solver="lbfgs")

y_pred = cross_val_predict(model, X, y, cv=kf, method='predict')
y_pred_proba = cross_val_predict(model, X, y, cv=kf, method='predict_proba')

# Compute confusion matrix
conf_matrix = confusion_matrix(y, y_pred)
print("Confusion Matrix:\n", conf_matrix)

# Compute evaluation metrics
accuracy = accuracy_score(y, y_pred)
f1 = f1_score(y, y_pred, average="weighted")  # Weighted average for multiclass
ordered_labels = np.sort(df["Category"].unique())
auc = roc_auc_score(y, y_pred_proba, multi_class="ovr", labels=ordered_labels)

# Display results
print(f"Accuracy: {accuracy:.4f}")
print(f"F1-score: {f1:.4f}")
print(f"AUC (macro-averaged): {auc:.4f}\n")

# Classification report for detailed metrics per class
print("\nClassification Report:")
print(classification_report(y, y_pred))