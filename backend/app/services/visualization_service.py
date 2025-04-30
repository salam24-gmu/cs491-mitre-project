import os
import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
import uuid # Use UUID for request ID

logger = logging.getLogger(__name__)

PALETTE = sns.color_palette("viridis", n_colors=10)
ENTITY_COLOR_MAP = {
    "SUSPICIOUS_BEHAVIOR": PALETTE[0],
    "SENSITIVE_INFO": PALETTE[1],
    "TIME_ANOMALY": PALETTE[2],
    "TECH_ASSET": PALETTE[3],
    "MEDICAL_CONDITION": PALETTE[4],
    "SENTIMENT_INDICATOR": PALETTE[5],
    "PERSON": PALETTE[6],
    "ORG": PALETTE[7],
    "LOC": PALETTE[8],
}

def _save_plot(figure, output_path: Path, filename: str):
    """Helper function to save plots."""
    try:
        output_path.mkdir(parents=True, exist_ok=True)
        full_path = output_path / filename
        figure.savefig(full_path, bbox_inches='tight')
        plt.close(figure) # Close the figure to free memory
        logger.info(f"Saved plot to {full_path}")
        # --- Return only the filename --- 
        return filename
    except Exception as e:
        logger.error(f"Failed to save plot {filename}: {e}")
        plt.close(figure)
        return None

def plot_entity_distribution(user_data: pd.DataFrame, user_id: str, output_path: Path) -> Optional[str]:
    """
    Generates a bar chart showing the distribution of detected entity types for a user.

    Args:
        user_data (pd.DataFrame): DataFrame containing analysis results for a single user.
                                  Expected columns: 'entities' (list of dicts).
        user_id (str): The ID of the user.
        output_path (Path): Directory to save the plot.

    Returns:
        Optional[str]: Path to the saved plot file, or None if failed.
    """
    logger.info(f"Generating entity distribution plot for user: {user_id}")
    try:
        all_entities = []
        if 'entities' in user_data.columns:
            for entity_list in user_data['entities'].dropna():
                if isinstance(entity_list, list):
                    all_entities.extend([entity.get('entity_group', 'UNKNOWN') for entity in entity_list])

        if not all_entities:
            logger.warning(f"No entities found for user {user_id} to plot.")
            return None

        entity_counts = pd.Series(all_entities).value_counts()

        if entity_counts.empty:
            logger.warning(f"Entity counts are empty for user {user_id}.")
            return None

        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.barh(entity_counts.index, entity_counts.values, color=[ENTITY_COLOR_MAP.get(ent, PALETTE[9]) for ent in entity_counts.index])
        ax.set_xlabel("Count")
        ax.set_ylabel("Entity Type")
        ax.set_title(f"Entity Distribution for User {user_id}")
        ax.invert_yaxis() # Display the highest count at the top

        # Add counts at the end of the bars
        for bar in bars:
            ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                    f'{int(bar.get_width())}', va='center', ha='left')

        return _save_plot(fig, output_path, f"entity_distribution_{user_id}.png")

    except Exception as e:
        logger.error(f"Error generating entity distribution plot for {user_id}: {e}", exc_info=True)
        return None


def plot_risk_profile(user_data: pd.DataFrame, user_id: str, output_path: Path) -> Optional[str]:
    """
    Generates a radar chart visualizing key risk metrics per tweet for a user.

    Args:
        user_data (pd.DataFrame): DataFrame containing analysis results for a single user.
                                  Expected columns: 'risk_metrics' (dict).
        user_id (str): The ID of the user.
        output_path (Path): Directory to save the plot.

    Returns:
        Optional[str]: Path to the saved plot file, or None if failed.
    """
    logger.info(f"Generating risk profile plot for user: {user_id}")
    try:
        if 'risk_metrics' not in user_data.columns or user_data['risk_metrics'].isnull().all():
            logger.warning(f"No risk metrics data found for user {user_id}.")
            return None

        metrics_to_plot = [
            'total_risk', 'base_risk', 'context_risk', 'semantic_risk',
            'risk_density', 'high_risk_combinations'
        ]

        # Extract and average metrics per tweet
        num_tweets = len(user_data)
        avg_metrics = {}
        valid_metrics_count = 0
        for metric in metrics_to_plot:
            total_value = 0
            count = 0
            for metrics_dict in user_data['risk_metrics'].dropna():
                if isinstance(metrics_dict, dict) and metric in metrics_dict:
                    value = metrics_dict.get(metric, 0)
                    # Ensure value is numeric before summing
                    if isinstance(value, (int, float)):
                        total_value += value
                        count += 1
                elif isinstance(metrics_dict, dict):
                    # Check for entity specific risks or counts if metric isn't top-level
                    pass # Add logic here if needed for nested metrics

            if count > 0:
                avg_metrics[metric] = total_value / count # Average over tweets where metric exists
                valid_metrics_count += 1
            else:
                avg_metrics[metric] = 0 # Default to 0 if metric not found in any tweet

        if valid_metrics_count == 0:
            logger.warning(f"No valid risk metrics found to plot for user {user_id} among specified metrics: {metrics_to_plot}")
            return None


        labels = list(avg_metrics.keys())
        stats = list(avg_metrics.values())

        # Ensure labels and stats are not empty
        if not labels or not stats:
            logger.warning(f"Could not extract labels or stats for risk profile plot for user {user_id}.")
            return None

        # Complete the loop for the radar chart
        labels += labels[:1]
        stats += stats[:1]

        angles = np.linspace(0, 2 * np.pi, len(labels) -1, endpoint=False).tolist()
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        ax.plot(angles, stats, linewidth=2, linestyle='solid', label=f"User {user_id} Avg Metrics")
        ax.fill(angles, stats, 'skyblue', alpha=0.4)

        # Improve readability of labels and ticks
        ax.set_thetagrids(np.degrees(angles[:-1]), labels[:-1])
        ax.tick_params(axis='y', labelsize=8)
        ax.tick_params(axis='x', pad=10) # Add padding to labels

        # Set reasonable limits based on data, maybe normalize scores 0-1?
        # For now, let matplotlib decide the scale, but this might need adjustment
        # Example: ax.set_yticks(np.linspace(0, max(stats) if stats else 1, 5))

        plt.title(f"Risk Profile (Avg per Tweet) - User {user_id}", size=14, y=1.1)
        # ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1)) # Legend might overlap

        return _save_plot(fig, output_path, f"risk_profile_{user_id}.png")

    except Exception as e:
        logger.error(f"Error generating risk profile plot for {user_id}: {e}", exc_info=True)
        return None

def plot_temporal_distribution(all_data: pd.DataFrame, output_path: Path) -> Optional[str]:
    """
    Generates a scatter plot showing tweet distribution over time, colored by
    maliciousness and marked by temporal anomaly (if available).

    Args:
        all_data (pd.DataFrame): DataFrame containing analysis results for all users.
                                 Expected columns: 'timestamp', 'predicted_class', 'risk_metrics' (dict with 'off_hours').
        output_path (Path): Directory to save the plot.

    Returns:
        Optional[str]: Path to the saved plot file, or None if failed.
    """
    logger.info(f"Generating temporal distribution plot for {len(all_data)} tweets.")
    try:
        if 'timestamp' not in all_data.columns:
            logger.warning("Timestamp column missing, cannot generate temporal plot.")
            return None

        # Convert timestamp to datetime objects, coercing errors
        all_data['datetime'] = pd.to_datetime(all_data['timestamp'], errors='coerce')
        valid_data = all_data.dropna(subset=['datetime'])

        if valid_data.empty:
            logger.warning("No valid timestamps found after conversion.")
            return None

        # Extract hour and date for plotting
        valid_data['hour'] = valid_data['datetime'].dt.hour
        valid_data['date'] = valid_data['datetime'].dt.date

        # Determine Malicious/Non-Malicious and Anomalous/Non-Anomalous
        valid_data['Malicious'] = valid_data['predicted_class'] == 'malicious'
        # Check for 'off_hours' within 'risk_metrics' dictionary
        valid_data['Anomalous (Off-Hours)'] = valid_data['risk_metrics'].apply(
            lambda x: isinstance(x, dict) and x.get('off_hours', False)
        )

        # Create categories for plotting similar to the example
        def get_category(row):
            if row['Malicious'] and row['Anomalous (Off-Hours)']:
                return 'Malicious and Anomalous'
            elif row['Malicious'] and not row['Anomalous (Off-Hours)']:
                return 'Malicious and Non-Anomalous'
            elif not row['Malicious'] and row['Anomalous (Off-Hours)']:
                return 'Non-Malicious and Anomalous'
            else:
                return 'Non-Malicious and Non-Anomalous'

        valid_data['Category'] = valid_data.apply(get_category, axis=1)

        category_styles = {
            'Malicious and Anomalous': {'marker': 'x', 'color': 'red', 'label': 'Malicious and Anomalous'},
            'Malicious and Non-Anomalous': {'marker': 'o', 'color': 'green', 'edgecolors': 'darkgreen', 'label': 'Malicious and Non-Anomalous'},
            'Non-Malicious and Anomalous': {'marker': '^', 'color': 'blue', 'label': 'Non-Malicious and Anomalous'},
            'Non-Malicious and Non-Anomalous': {'marker': 's', 'color': 'yellowgreen', 'edgecolors': 'darkgreen', 'label': 'Non-Malicious and Non-Anomalous'}
        }

        fig, ax = plt.subplots(figsize=(16, 8))

        dates = sorted(valid_data['date'].unique())

        for category, style in category_styles.items():
            subset = valid_data[valid_data['Category'] == category]
            if not subset.empty:
                # Use date index for x-axis positioning
                date_map = {date: i for i, date in enumerate(dates)}
                x_positions = subset['date'].map(date_map)
                ax.scatter(x_positions, subset['hour'],
                            marker=style['marker'],
                            color=style['color'],
                            edgecolors=style.get('edgecolors'),
                            label=style['label'], s=50) # s controls marker size

        # Format X-axis to show dates
        ax.set_xticks(range(len(dates)))
        ax.set_xticklabels([d.strftime('%Y-%m-%d') for d in dates], rotation=45, ha='right')

        # Format Y-axis for hours
        ax.set_yticks(range(0, 25, 2))
        ax.set_ylabel("Hour of Posting")
        ax.set_ylim(-1, 24)

        ax.set_title("Distribution of Tweets by Maliciousness and Temporal Anomaly")
        ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.3), ncol=4, title="Category") # Adjust legend position
        ax.grid(True)
        plt.tight_layout(rect=[0, 0.05, 1, 1]) # Adjust layout to prevent label overlap

        return _save_plot(fig, output_path, "temporal_distribution.png")

    except Exception as e:
        logger.error(f"Error generating temporal distribution plot: {e}", exc_info=True)
        return None


def plot_user_risk_trend(user_data: pd.DataFrame, user_id: str, output_path: Path) -> Optional[str]:
    """
    Generates a line graph showing the user's risk level over time.

    Args:
        user_data (pd.DataFrame): DataFrame containing analysis results for a single user.
                                  Expected columns: 'timestamp', 'risk_metrics' (dict with 'total_risk').
        user_id (str): The ID of the user.
        output_path (Path): Directory to save the plot.

    Returns:
        Optional[str]: Path to the saved plot file, or None if failed.
    """
    logger.info(f"Generating risk trend plot for user: {user_id}")
    try:
        if 'timestamp' not in user_data.columns or 'risk_metrics' not in user_data.columns:
            logger.warning(f"Timestamp or risk_metrics column missing for user {user_id}, cannot generate risk trend plot.")
            return None

        # Convert timestamp and extract total_risk
        user_data['datetime'] = pd.to_datetime(user_data['timestamp'], errors='coerce')
        user_data['total_risk'] = user_data['risk_metrics'].apply(
            lambda x: x.get('total_risk', None) if isinstance(x, dict) else None
        )

        # Filter out rows with invalid dates or risk scores
        valid_data = user_data.dropna(subset=['datetime', 'total_risk']).sort_values('datetime')

        if valid_data.empty:
            logger.warning(f"No valid timestamp/risk data found for user {user_id} to plot trend.")
            return None

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(valid_data['datetime'], valid_data['total_risk'], marker='o', linestyle='-', label='Total Risk Score')

        # Formatting
        ax.set_xlabel("Time")
        ax.set_ylabel("Total Risk Score")
        ax.set_title(f"Risk Level Trend for User {user_id}")
        ax.legend()
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.xticks(rotation=45)
        plt.tight_layout()

        return _save_plot(fig, output_path, f"risk_trend_{user_id}.png")

    except Exception as e:
        logger.error(f"Error generating risk trend plot for {user_id}: {e}", exc_info=True)
        return None

async def generate_visualizations(analysis_results: List[Dict[str, Any]], output_base_dir: str = "generated_plots") -> Dict[str, Any]:
    """
    Main function to orchestrate the generation of all plots based on file analysis results.

    Args:
        analysis_results (List[Dict[str, Any]]): The list of 'FileRowAnalysisResult' like dictionaries
                                                  from the file processing endpoint.
        output_base_dir (str): The base directory to save plots. A unique subdirectory will be created.

    Returns:
        Dict[str, Any]: A dictionary containing the request_id and paths (filenames) to the generated plots.
                        Example:
                        {
                            "request_id": "uuid_string",
                            "base_path": "generated_plots/uuid_string", // Optional: Informative path
                            "overall": {
                                "temporal_distribution": "temporal_distribution.png"
                            },
                            "user_plots": {
                                "user_id_1": {
                                    "entity_distribution": "entity_distribution_user_id_1.png",
                                    "risk_profile": "risk_profile_user_id_1.png",
                                    "risk_trend": "risk_trend_user_id_1.png"
                                },
                                "user_id_2": { ... }
                            }
                        }
    """
    if not analysis_results:
        logger.warning("No analysis results provided, skipping visualization generation.")
        return {}

    # --- Create a unique output directory using UUID --- 
    request_id = str(uuid.uuid4())
    output_path = Path(output_base_dir) / request_id
    logger.info(f"Preparing to save plots to: {output_path} (Request ID: {request_id})")

    # --- Data Preparation ---
    # Convert the list of dicts/FileRowAnalysisResult objects into a pandas DataFrame
    plot_data = []
    for result in analysis_results:
        # Skip rows with errors or no analysis data
        if result.get('error') or not result.get('analysis_result'):
            continue

        # Extract core fields and analysis results safely
        row_data = {
            'user_id': result.get('user_id', 'UNKNOWN_USER'),
            'timestamp': result.get('timestamp'),
            'tweet_id': result.get('tweet_id'),
            'text': result.get('text'),
            'predicted_class': result.get('predicted_class'),
            'malicious_probability': result.get('malicious_probability'),
            # Extract from the nested 'analysis_result' dict
            'entities': result['analysis_result'].get('entities', []),
            'risk_metrics': result['analysis_result'].get('risk_metrics', {})
        }
        plot_data.append(row_data)

    if not plot_data:
        logger.warning("No valid data extracted from analysis results for plotting.")
        return {"request_id": request_id, "error": "No valid data for plotting"}

    df = pd.DataFrame(plot_data)
    logger.info(f"Created DataFrame with {len(df)} rows for plotting.")

    # --- Plot Generation ---
    # Initialize with request_id and base_path
    generated_plots = {
        "request_id": request_id, 
        "base_path": str(output_path), 
        "overall": {}, 
        "user_plots": {}
    }

    # Overall plots
    temporal_plot_filename = plot_temporal_distribution(df.copy(), output_path)
    if temporal_plot_filename:
        generated_plots["overall"]["temporal_distribution"] = temporal_plot_filename

    # Per-user plots
    if 'user_id' in df.columns:
        for user_id, user_data in df.groupby('user_id'):
            if user_id is None or pd.isna(user_id) or str(user_id).strip() == "":
                user_id_str = "UNKNOWN_USER"
            else:
                user_id_str = str(user_id)

            logger.info(f"--- Generating plots for User: {user_id_str} ({len(user_data)} tweets) ---")
            user_plots = {}

            entity_filename = plot_entity_distribution(user_data.copy(), user_id_str, output_path)
            if entity_filename: user_plots["entity_distribution"] = entity_filename

            risk_profile_filename = plot_risk_profile(user_data.copy(), user_id_str, output_path)
            if risk_profile_filename: user_plots["risk_profile"] = risk_profile_filename

            risk_trend_filename = plot_user_risk_trend(user_data.copy(), user_id_str, output_path)
            if risk_trend_filename: user_plots["risk_trend"] = risk_trend_filename

            if user_plots:
                generated_plots["user_plots"][user_id_str] = user_plots
    else:
        logger.warning("No 'user_id' column found in data, cannot generate per-user plots.")

    logger.info(f"Finished generating plots. Results: {generated_plots}")
    return generated_plots 