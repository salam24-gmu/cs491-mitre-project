from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from empath import Empath
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt


DEFAULT_INPUT_DATA = "./finetune/data/main_training_dataset/augmented_synthetic_data_V3.csv"
DEFAULT_OUTPUT_PATH = "./out/personality_profile/"

def display_personality_chart(user_id, ocean_scores_by_user, show_plot=False, save_plot=False):

    # Create a list of the traits and their values
    categories = {"Openness": 0, "Conscientiousness": 1, "Extraversion": 2, "Agreeableness": 3, "Neuroticism": 4}
    values = list(ocean_scores_by_user[user_id].values())

    # Number of categories
    N = len(categories)

    # Compute angle for each axis (evenly spaced for radar chart)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]

    # Make the plot a circular (close the plot by repeating the first value at the end)
    values.append(values[0])
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
    ax.set_ylim(0, max(values) * 1.1)

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

def get_average_value(dictionary: dict) -> float:
    """
    Calculate the average value of a dictionary.
    :param dictionary: Dictionary with numeric values.
    :return: Average value.
    """
    if not dictionary:
        return 0.0
    return sum(dictionary.values()) / len(dictionary)

def calculate_ocean_scores(user_profiles, analyzer, lexicon):
    """
    Calculate OCEAN scores for each user based on their tweets.
    :param: user_profiles: Dictionary of user profiles with user_id as keys and list of tweets as values.
    :param: analyzer: SentimentIntensityAnalyzer instance.
    :param: lexicon: Empath instance.
    :return: Dictionary with user_id as keys and OCEAN scores as values.
    """

    ocean_scores_by_user = {}
    for user_id, tweets in user_profiles.items():
        sentiment_scores = []
        empath_categories = []

        ocean_score_of_user = {
            'Openness': [],
            'Conscientiousness': [],
            'Extraversion': [],
            'Agreeableness': [],
            'Neuroticism': []
        }

        for tweet in tweets:
            sentiment = analyzer.polarity_scores(tweet)
            sentiment_scores.append(sentiment['compound'])

            categories = lexicon.analyze(tweet, normalize=True)
            empath_categories.append(categories)

            ocean_score_of_tweet = {
                'Openness': get_average_value(lexicon.analyze(tweet, categories=['philosophy', 'art', 'literature', 'music', 'reading', 'fantasy'], normalize=True)),
                'Conscientiousness': get_average_value(lexicon.analyze(tweet, categories=['achievement', 'work', 'ethical', 'education', 'planning'], normalize=True)),
                'Extraversion': get_average_value(lexicon.analyze(tweet, categories=['social_media', 'friends', 'fun', 'happy', 'party', 'nightlife'], normalize=True)),
                'Agreeableness': get_average_value(lexicon.analyze(tweet, categories=['love', 'help', 'dependable', 'trust', 'sympathy', 'gratitude'], normalize=True)),
                'Neuroticism': get_average_value(lexicon.analyze(tweet, categories=['nervousness', 'fear', 'hurt', 'sadness', 'pain', 'anger', 'virus', 'download'], normalize=True))
            }

            ocean_score_of_user['Openness'].append(ocean_score_of_tweet['Openness'])
            ocean_score_of_user['Conscientiousness'].append(ocean_score_of_tweet['Conscientiousness'])
            ocean_score_of_user['Extraversion'].append(ocean_score_of_tweet['Extraversion'])
            ocean_score_of_user['Agreeableness'].append(ocean_score_of_tweet['Agreeableness'])
            ocean_score_of_user['Neuroticism'].append(ocean_score_of_tweet['Neuroticism'])

        for trait in ocean_score_of_user:
            ocean_score_of_user[trait] = float(np.mean(ocean_score_of_user[trait]))
        
        ocean_scores_by_user[user_id] = ocean_score_of_user
    return ocean_scores_by_user
    
def parse_user_profiles(df):
    """
    Parse user profiles from the DataFrame.
    :param df: DataFrame containing tweets and user IDs.
    :return: Dictionary with user_id as keys and list of tweets as values.
    """

    df = df[['user_id', 'tweet']].sort_values(by='user_id')
    df = df[df['user_id'] != '        '] # remove the null key 

    user_ids = df['user_id'].unique()

    user_profiles = {}
    for user_id in user_ids:
        user_profiles[user_id] = df[df['user_id'] == user_id]['tweet'].tolist()

    return user_profiles

def populate_with_labels(df, labels):
    """
    Populate the DataFrame with labels.
    :param df: DataFrame to populate.
    :param labels: List of labels to add to the DataFrame.
    """
    for label in labels:
        if label not in df.columns:
            df[label] = pd.NA
        else:
            df[label] = df[label].astype(float)  # Ensure the column is of type float

def update_ocean_scores(user_id_list, ocean_scores_by_user, df_copy):
    """
    Update the DataFrame with OCEAN scores for each user_id.
    :param user_id_list: List of user IDs to update.
    :param ocean_scores_by_user: Dictionary with user_id as keys and OCEAN scores as values.
    :param df_copy: Original DataFrame to update.
    """
    # Loop through the user_id_list and update the corresponding OCEAN scores
    for user_id in user_id_list:
        if user_id in ocean_scores_by_user:
            # Get the OCEAN scores for the current user
            traits = ocean_scores_by_user[user_id].values()
            # Update the OCEAN scores in the DataFrame for the current user_id
            for label, score in zip(ocean_labels, traits):
                df_copy.loc[df_copy['user_id'] == user_id, label] = score

def plot_auc_curve(y_test, y_probs, auc_score, show_plot=False, save_plot=False):
    """
    Plot the AUC curve.
    :param y_test: True labels.
    :param y_probs: Predicted probabilities.
    """
    print("Plotting the AUC curve...")

    # 4. Plot ROC curve (optional but helpful)
    fpr, tpr, thresholds = roc_curve(y_test, y_probs)
    plt.plot(fpr, tpr, label=f"AUC = {auc_score:.2f}")
    plt.plot([0, 1], [0, 1], linestyle='--')  # random guess line
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid()

    if show_plot:
        plt.show()
    if save_plot:
        plt.savefig(f"{DEFAULT_OUTPUT_PATH}auc-roc-curves/roc_curve_{user_id}.png", bbox_inches='tight')

def plot_confusion_matrix(y_test, y_pred, show_plot=False, save_plot=False):
    """
    Plot the confusion matrix.
    """
    print("Plotting the confusion matrix...")
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    import matplotlib.pyplot as plt

    # Get raw confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=[0,1])

    # Convert to percentages
    cm_percent = cm.astype('float') / cm.sum() * 100

    # Optional: plot with percentage labels
    labels = [f"{value:.1f}%" for value in cm_percent.flatten()]
    labels = np.array(labels).reshape(2, 2)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm_percent, display_labels=["malicious", "non-malicious"])
    disp.plot(cmap='Blues', values_format=".1f")
    plt.title("Confusion Matrix (% of Total)")
    plt.grid(False)

    if show_plot:
        plt.show()

    if save_plot:
        plt.savefig(f"{DEFAULT_OUTPUT_PATH}/confusion_matricies/confusion_matrix_{user_id}.png", bbox_inches='tight')

# Setup
analyzer = SentimentIntensityAnalyzer()
lexicon = Empath()
ocean_labels = ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]
sentiment_scores = []
empath_categories = []
ocean_scores = {}

print("Reading in Tweets from the CSV...")
df = pd.read_csv(DEFAULT_INPUT_DATA)
user_profiles = parse_user_profiles(df)
df = df[df['classification'] != 'potentially malicious']
df = df[df['user_id'] != '        '] #remove null key 

print("Computing personality scores...")
ocean_scores_by_user = calculate_ocean_scores(user_profiles, analyzer, lexicon)
for user_id in ocean_scores_by_user.keys():
    display_personality_chart(user_id, ocean_scores_by_user, show_plot=False, save_plot=True)
user_id_list = df['user_id'].unique()
df_copy = df.copy() # keep a copy to reference later 
populate_with_labels(df_copy, ocean_labels)


update_ocean_scores(user_id_list, ocean_scores_by_user, df_copy)

df_copy = df_copy.dropna(subset=['Openness', 'Conscientiousness', 'Extraversion', 'Agreeableness', 'Neuroticism'])

print("Preparing the features and target variable...")
X = df_copy[['Openness', 'Conscientiousness', 'Extraversion', 'Agreeableness', 'Neuroticism']].astype(float)
y = df_copy['classification']

print("Running a label encoder...")
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)


print("Splitting the data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.1, random_state=42)

from imblearn.over_sampling import SMOTE

# Apply SMOTE to the training data only
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

print("Training a model...")
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score

model = XGBClassifier(eval_metric='logloss')

params = {
    "n_estimators": [100, 200, 300, 400, 500],
    "max_depth": [3, 4, 5, 6, 7, 8, 10],
    "learning_rate": [0.001, 0.01, 0.05, 0.1, 0.2, 0.3],
    "subsample": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    "gamma": [0, 0.1, 0.2, 0.5, 1, 5, 10],
    "reg_lambda": [0.01, 0.1, 1, 10, 100],
    "reg_alpha": [0.01, 0.1, 1, 10, 100],
}


from sklearn.model_selection import RandomizedSearchCV
search = RandomizedSearchCV(model, params, scoring='roc_auc', cv=3, n_iter=30)
search.fit(X_train_sm, y_train_sm)

# Extract best parameters
best_params = search.best_params_

# Create a new model with the best parameters
best_model = XGBClassifier(**best_params, eval_metric='logloss')

best_model.fit(X_train_sm, y_train_sm)

print("Predicting the test data...")
y_pred = best_model.predict(X_test)

print("Evaluating the model...")
y_probs = best_model.predict_proba(X_test)[:, 1]  # probability for positive class
scores = cross_val_score(best_model, X, y_encoded, cv=5, scoring='roc_auc', error_score='raise')
avg_auc = scores.mean()
print(f"Average AUC: {avg_auc:.3f}")

plot_auc, plot_cm = False, False
if plot_auc:
   plot_auc_curve(y_test, y_probs, avg_auc, show_plot=True, save_plot=True)

if plot_cm:   
   plot_confusion_matrix(y_test, y_pred, show_plot=True, save_plot=True)

print("Done!")
