import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from matplotlib.ticker import MaxNLocator
import numpy as np

DATASET_PATH = "./finetune/data/main_training_dataset/augmented_synthetic_data_V3.csv"

def run(display_graph=False):
    '''
    Returns a labeled dataframe with temporal anomalies labeled.'''

    df = pd.read_csv(DATASET_PATH)

    # Convert the timestamp column to datetime
    pd.set_option("display.max_columns", None)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Extract the hour and minute from the timestamp
    df['date'] = df['timestamp'].dt.strftime("%Y-%m-%d")
    df['hour'] = df['timestamp'].dt.hour
    df['minute'] = df['timestamp'].dt.minute

    # Create a 'time_of_day' column in the HH:MM format
    df['time_of_day_str'] = df['timestamp'].dt.strftime('%H:%M')

    # Scaling the time_of_day for temporal anomaly detection
    df['time_of_day'] = df['hour'] * 60 + df['minute']  # Time in minutes (for temporal anomaly detection)
    scaler = StandardScaler()
    df['time_scaled'] = scaler.fit_transform(df[['time_of_day']])

    # Use Isolation Forest to detect anomalies in the time of day
    model = IsolationForest(contamination='auto')  # Contamination is the expected percentage of anomalies
    df['temporal anomaly'] = model.fit_predict(df[['time_scaled']])

    # Convert -1 (anomalous) to True and 1 (normal) to False for better interpretation
    df['temporal anomaly'] = df['temporal anomaly'] == -1

    # Sort the entire dataframe by date and then by HH:MM timestamp (time_of_day_str)
    df = df.sort_values(by=['date', 'time_of_day'])

    if display_graph:

        # Visualize the distribution of tweet times
        plt.figure(figsize=(10, 6))

        NON_MALICIOUS_POINT_SIZE = 10
        MALICIOUS_POINT_SIZE = 50

        # Scatter plot for Malicious and Anomalous
        plt.scatter(df[(df['classification'] == 'malicious') & (df['temporal anomaly'] == True)]['date'], 
                    df[(df['classification'] == 'malicious') & (df['temporal anomaly'] == True)]['hour'], 
                    color='r', label='Malicious and Anomalous', marker='x', s=MALICIOUS_POINT_SIZE)

        # Scatter plot for Malicious and Non-Anomalous
        plt.scatter(df[(df['classification'] == 'malicious') & (df['temporal anomaly'] == False)]['date'], 
                    df[(df['classification'] == 'malicious') & (df['temporal anomaly'] == False)]['hour'], 
                    color='g', label='Malicious and Non-Anomalous', marker='o', s=MALICIOUS_POINT_SIZE)

        # Scatter plot for Non-Malicious and Anomalous
        plt.scatter(df[(df['classification'] == 'non-malicious') & (df['temporal anomaly'] == True)]['date'], 
                    df[(df['classification'] == 'non-malicious') & (df['temporal anomaly'] == True)]['hour'], 
                    color='b', label='Non-Malicious and Anomalous', marker='^', s=NON_MALICIOUS_POINT_SIZE)

        # Scatter plot for Non-Malicious and Non-Anomalous
        plt.scatter(df[(df['classification'] == 'non-malicious') & (df['temporal anomaly'] == False)]['date'], 
                    df[(df['classification'] == 'non-malicious') & (df['temporal anomaly'] == False)]['hour'], 
                    color='y', label='Non-Malicious and Non-Anomalous', marker='s', s=NON_MALICIOUS_POINT_SIZE)

        # Label the axes and add title
        plt.xlabel('Date')
        plt.ylabel('Hour of Posting')
        plt.title('Distribution of Tweets by Maliciousness and Temporal Anomaly (Alam, Darla, Erickson, and Ha 2025)')
        plt.suptitle("Uses Isolation Forest with auto contamination setting.",fontsize=8,y=-.95)

        # Rotate the x-axis labels for better readability
        plt.xticks(rotation=45)

        # Set the y-ticks to represent the time in 24-hour format (HH:MM)
        plt.yticks(df['hour'].unique())

        # Show grid, legend, and adjust layout
        plt.legend()
        plt.tight_layout()
        plt.grid(True)

        # Reduce the number of major horizontal gridlines using MaxNLocator
        plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True, prune='both', nbins=24))  # Set nbins to control gridlines
        plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.25), ncol=2)

        # Show the plot
        plt.show()
    return df 

def get_nonanomolous_temporal_interval_range():
    '''
    Returns the temporal interval range of the median first and last hour of non-anomalous tweets.
    '''
    
    df = run()
    pd.set_option("display.max_rows", None)

    dates = df['date'].unique()

    first_hours = []
    last_hours = []
    for i in range(len(dates)):
        filtered_df = df[df['date'] == dates[i]]


        first_hour_of_temp_anomaly = filtered_df[filtered_df['temporal anomaly'] == False]['hour'].min()
        last_hour_of_temp_anomaly = filtered_df[filtered_df['temporal anomaly'] == False]['hour'].max()

        first_hours.append(first_hour_of_temp_anomaly)
        last_hours.append(last_hour_of_temp_anomaly)

    first_hours_median = np.median(first_hours)
    last_hours_median = np.median(last_hours)

    return first_hours_median, last_hours_median

if __name__ == "__main__":

    first_hours_median, last_hours_median = get_nonanomolous_temporal_interval_range()

    print(f"Median first hour of temporal anomaly: {first_hours_median}")
    print(f"Median last hour of temporal anomaly: {last_hours_median}")

