import nltk
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from empath import Empath
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

DEFAULT_INPUT_DATA = "./finetune/data/main_training_dataset/augmented_synthetic_data_V3.csv"
DEFAULT_OUTPUT_PATH = "./out/personality_profile/"

def display_personality_chart(user_id, ocean_scores_arr, show_plot=False, save_plot=False):

    # Create a list of the traits and their values
    categories = {"Openness": 0, "Conscientiousness": 1, "Extraversion": 2, "Agreeableness": 3, "Neuroticism": 4}
    values = ocean_scores_arr

    # Number of categories
    N = len(categories)

    # Compute angle for each axis (evenly spaced for radar chart)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]

    # Make the plot a circular (close the plot by repeating the first value at the end)
    values += values[:1]
    angles += angles[:1]

    # Create the radar chart
    fig, ax = plt.subplots(figsize=(6, 6), dpi=100, subplot_kw=dict(polar=True))

    # Draw one axe per trait
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    # Set the labels for each trait
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)

    # Set the radial axis limits
    ax.set_rlabel_position(0)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_ylim(0, 1)

    # Plot the data for the user
    ax.plot(angles, values, linewidth=2, linestyle='solid', label='User')
    ax.fill(angles, values, alpha=0.25)

    # Add a title
    plt.title(f"Personality Radar Chart for {user_id}", size=14, color='blue', fontweight='bold')

    # Show the chart
    if show_plot:
        plt.show()

    if save_plot:
        plt.savefig(f"{DEFAULT_OUTPUT_PATH}/personality_chart_{user_id}.png", bbox_inches='tight')
    # Close the plot
    plt.close()

# Setup
analyzer = SentimentIntensityAnalyzer()
lexicon = Empath()

# Aggregate features
sentiment_scores = []
empath_categories = []

# dictionary of user-ids to their personality profile
ocean_scores = {}

# -- CONSTRUCTING THE USER PROFILES -- 
# 1. Read in Tweets from the CSV 
print("Reading in Tweets from the CSV...")
df = pd.read_csv(DEFAULT_INPUT_DATA)
df_origin = df.copy() # keep a copy to reference later 

df = df[['user_id', 'tweet']].sort_values(by='user_id')
df = df[df['user_id'] != '        '] # remove the null key 

# Get a unique list of user_ids
user_ids = df['user_id'].unique()

user_profiles = {}
for user_id in user_ids:
    user_profiles[user_id] = df[df['user_id'] == user_id]['tweet'].tolist()


print("Computing personality scores...")
for user_id, tweets in user_profiles.items():

  for tweet in tweets:
      sentiment = analyzer.polarity_scores(tweet)
      sentiment_scores.append(sentiment['compound'])

      categories = lexicon.analyze(tweet, normalize=True)
      empath_categories.append(categories)

  # Average features
  avg_sentiment = np.mean(sentiment_scores)
  avg_empath = {}
  for category in empath_categories:
      for key, val in category.items():
          avg_empath[key] = avg_empath.get(key, 0) + val / len(empath_categories)

  # Map features to OCEAN scores (simplified heuristic)
  ocean_inferred = {
      'Openness': np.mean([avg_empath.get('philosophy', 0), avg_empath.get('reading', 0), avg_empath.get('art', 0), avg_empath.get('curious', 0)]),
      'Conscientiousness': np.mean([avg_empath.get('work', 0), avg_empath.get('achievement', 0), avg_empath.get('responsible',0), avg_empath.get('ethical',0)]), # TODO: make this more granular
      'Extraversion': np.mean([avg_empath.get('social_media', 0), avg_empath.get('friends', 0), avg_empath.get('fun', 0), avg_empath.get('happy',0), avg_empath.get('party',0)]),
      'Agreeableness': np.mean([avg_empath.get('love', 0), avg_empath.get('help', 0), avg_empath.get('dependable',0), avg_empath.get('trust',0)]),
      'Neuroticism': np.mean([avg_empath.get('nervousness', 0), avg_empath.get('fear', 0), avg_empath.get('hurt', 0), avg_empath.get('virus',0), avg_empath.get('download',0)])
  }


  # Normalize inferred values (0–1 range)
  max_vals = max(ocean_inferred.values())
  for trait in ocean_inferred:
      ocean_inferred[trait] = min(ocean_inferred[trait] / max_vals, 1.0)
  ocean_scores[user_id] = ocean_inferred

ocean_scores_arr = []

for k, v in ocean_scores.items():
    arr = [k]

    for k1, v1 in v.items():
        arr.append(float(v1))

    ocean_scores_arr.append(arr)

# Convert to dict: user_id -> list of OCEAN scores
user_ocean_scores = {
    row[0]: row[1:]
    for row in ocean_scores_arr
}

ocean_labels = ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]

# Ensure those columns exist in the DataFrame, if not, initialize them with NA
for col in ocean_labels:
    if col not in df_origin.columns:
        df_origin[col] = pd.NA

# Get the unique user_ids from the DataFrame
user_id_list = df_origin['user_id'].unique()

# Loop through the user_id_list and update the corresponding OCEAN scores
for user_id in user_id_list:
    if user_id in user_ocean_scores:
        # Get the OCEAN scores for the current user
        traits = user_ocean_scores[user_id]
        # Update the OCEAN scores in the DataFrame for the current user_id
        for label, score in zip(ocean_labels, traits):
            df_origin.loc[df_origin['user_id'] == user_id, label] = score


print("Displaying the first 5 rows of the DataFrame with OCEAN scores...")
print(df_origin[['user_id',"Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism",'classification']].head())
print("Possible classification values are {}".format(df_origin['classification'].unique()))

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt


df = df_origin

# Omit Conscientiousness since its values are always 1 for whatever reason 

# Remove rows with any missing values
df = df.dropna(subset=['Openness', 'Extraversion', 'Agreeableness', 'Neuroticism'])


# Step 1: Prepare the features (X) and target variable (Y)
print("Preparing the features and target variable...")
X = df[['Openness', 'Extraversion', 'Agreeableness', 'Neuroticism']]
y = df['classification']

# Step 2: Encode the target variable (y) into numeric labels
print("Running a label encoder...")
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)  # This will convert 'malicious', 'non-malicious', 'medical' to numeric values

# Step 3: Split the data into training and testing sets (80% train, 20% test)
print("Splitting the data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Step 4: Initialize and train the logistic regression model
print("Training a logistic regression model...")
model = LogisticRegression(max_iter=10000,class_weight='balanced')  # Increase max_iter if necessary
model.fit(X_train, y_train)

# Step 5: Predict the test data
print("Predicting the test data...")
y_pred = model.predict(X_test)

# Step 6: Evaluate the model
print("Evaluating the model...")
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# 2. Predict probabilities (not classes!)
y_probs = model.predict_proba(X_test)[:, 1]  # probability for positive class

# 3. Compute AUC
auc_score = roc_auc_score(y_test, y_probs)
print("AUC score:", auc_score)

# 4. Plot ROC curve (optional but helpful)
fpr, tpr, thresholds = roc_curve(y_test, y_probs)
plt.plot(fpr, tpr, label=f"AUC = {auc_score:.2f}")
plt.plot([0, 1], [0, 1], linestyle='--')  # random guess line
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid()
plt.show()

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Get raw confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=[0,1,2])

# Convert to percentages
cm_percent = cm.astype('float') / cm.sum() * 100

# Optional: plot with percentage labels
labels = [f"{value:.1f}%" for value in cm_percent.flatten()]
labels = np.array(labels).reshape(3, 3)

disp = ConfusionMatrixDisplay(confusion_matrix=cm_percent, display_labels=["malicious", "non-malicious", "potentially malicious"])
disp.plot(cmap='Blues', values_format=".1f")
plt.title("Confusion Matrix (% of Total)")
plt.grid(False)
plt.show()
