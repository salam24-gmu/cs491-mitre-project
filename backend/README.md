# Insider Threat Detection API

This FastAPI backend provides a RESTful API for detecting potential insider threats using a fine-tuned NER (Named Entity Recognition) model combined with an advanced risk profiling pipeline.

## Features

- **Named Entity Recognition (NER)**: Identifies entities like SUSPICIOUS_BEHAVIOR, SENSITIVE_INFO, TIME_ANOMALY, etc. in text
- **Risk Profiling**: Analyzes text content to generate comprehensive risk assessments
- **User Profiles**: Aggregates individual text analyses into user-level risk profiles
- **Multi-Dimensional Risk**: Evaluates multiple risk factors including behavior patterns, time anomalies, and combinations
- **CPU Optimized**: Configured to run efficiently on CPU for demonstrations and testing
- **Asynchronous Processing**: Uses FastAPI's async features and thread pools for non-blocking operations
- **Batch Processing**: Process multiple texts in a single API call for efficiency

## Getting Started

### Prerequisites

- Python 3.8+ installed
- Fine-tuned NER model files in appropriate location (`../finetune/data/insider-threat-ner_model/model-4n4ekgnp/`)

### Installation

1. Clone this repository
2. Run the setup script:

```bash
cd backend
chmod +x setup.sh
./setup.sh
```

### Running the Server

The server can be started using the provided run script:

```bash
./run.sh
```

Additional options:
- Test model loading only: `./run.sh --test-model`
- Run without auto-reload (for production): `./run.sh --no-reload`

The API will be available at http://localhost:8000 with the Swagger UI documentation at http://localhost:8000/docs.

## API Endpoints

### System Information

#### 1. Get System Info
**GET** `/`

Returns information about the system, including model paths and CPU configuration.

#### 2. Get Model Details
**GET** `/model-info`

Returns detailed information about the loaded model.

### Basic Entity Detection

#### 3. Detect Threats
**POST** `/detect-threats`

Process a single text for entity detection and basic threat analysis.

**Request Body:**
```json
{
  "text": "I'm going to download the customer database tonight after everyone has gone home.",
  "timestamp": "2023-04-15T22:10:00Z"  // Optional
}
```

#### 4. Batch Detect Threats
**POST** `/batch-detect-threats`

Process multiple texts for basic entity detection in a single request.

**Request Body:**
```json
{
  "texts": [
    {
      "text": "I'm going to download the customer database tonight after everyone has gone home.",
      "timestamp": "2023-04-15T22:10:00Z"
    },
    {
      "text": "I need the admin password for the server room door."
    }
  ]
}
```

### Advanced Risk Profiling

#### 5. Analyze Risk
**POST** `/analyze-risk`

Analyze a single text with comprehensive risk profiling.

**Request Body:**
```json
{
  "text": "I'm going to download the customer database tonight after everyone has gone home.",
  "timestamp": "2023-04-15T22:10:00Z",
  "user_id": "user_123",  // Optional
  "tweet_id": "tweet_456"  // Optional
}
```

#### 6. Batch Analyze Risk
**POST** `/batch-analyze-risk`

Process multiple texts with comprehensive risk profiling and generate aggregated user profiles.

**Request Body:**
```json
{
  "texts": [
    {
      "text": "I'm going to download the customer database tonight after everyone has gone home.",
      "timestamp": "2023-04-15T22:10:00Z",
      "user_id": "user_123",
      "tweet_id": "tweet_456"
    },
    {
      "text": "I need the admin password for the server room door.",
      "user_id": "user_123",
      "tweet_id": "tweet_789"
    }
  ]
}
```

#### 7. User Risk Profile
**POST** `/user-risk-profile`

Generate a comprehensive risk profile for a specific user based on their texts.

**Request Body:**
```json
{
  "user_id": "user_123",
  "texts": [
    {
      "text": "I'm going to download the customer database tonight after everyone has gone home.",
      "timestamp": "2023-04-15T22:10:00Z",
      "tweet_id": "tweet_456"
    },
    {
      "text": "I need the admin password for the server room door.",
      "tweet_id": "tweet_789"
    }
  ]
}
```

## Risk Profiling Pipeline

The risk profiling pipeline analyzes texts in multiple stages:

1. **Entity-Level Analysis**: Identifies named entities and assigns risk weights based on entity type

2. **Tweet-Level Assessment**: Combines entity risks with contextual analysis to calculate a risk score that accounts for:
   - Entity combinations (e.g., SUSPICIOUS_BEHAVIOR + SENSITIVE_INFO)
   - Time patterns (off-hours activity)
   - Semantic indicators (threatening language, personal grievances)

3. **User-Level Aggregation**: Combines multiple tweet analyses into comprehensive user profiles with:
   - Multiple risk dimensions (total risk, density, volatility)
   - Behavioral pattern detection (risk spikes, increasing trends)
   - Entity distribution analysis
   - Anomaly detection

## Risk Metrics

The system generates several important risk metrics:

### Aggregate Risk Measures
- **Total Risk Score**: Sum of all risk contributions
- **Max Tweet Risk**: Risk score of the most concerning tweet

### Normalized Metrics  
- **Average Risk Per Tweet**: Risk normalized by activity volume
- **Entity Density**: Entity mentions per tweet

### Behavioral Patterns  
- **High-Risk Combination Density**: Frequency of dangerous entity combinations
- **Off-Hours Percentage**: Proportion of activity outside normal hours

### Volatility Indicators  
- **Risk Standard Deviation**: Variability of risk across tweets
- **Outlier Score**: Statistical measure of how anomalous a user is

## Performance Optimization

The API is configured to run efficiently on CPU with the following optimizations:

1. **Asynchronous Processing**: API endpoints run asynchronously and don't block the server
2. **Thread Pool**: CPU-intensive operations run in a thread pool to prevent blocking the event loop
3. **Model Optimizations**: TorchScript, low memory usage, and CPU-specific optimizations
4. **Optional Batch Processing**: Process multiple texts in a single request for better throughput

## Model Information

The NER model was fine-tuned specifically for insider threat detection and can identify the following entity types:

- `PERSON`: Names of individuals
- `ORG`: Organization names
- `LOC`: Location names
- `TIME_ANOMALY`: Suspicious timing (after hours, weekends, etc.)
- `SENSITIVE_INFO`: References to sensitive or confidential information
- `TECH_ASSET`: Technology assets (databases, servers, etc.)
- `MEDICAL_CONDITION`: References to health conditions
- `SUSPICIOUS_BEHAVIOR`: Actions that may indicate malicious intent
- `SENTIMENT_INDICATOR`: Emotional indicators (anger, frustration, etc.)

## Example Usage

Using curl to analyze a single text with comprehensive risk profiling:

```bash
curl -X 'POST' \
  'http://localhost:8000/analyze-risk' \
  -H 'Content-Type: application/json' \
  -d '{
  "text": "I'm going to download the customer database tonight after everyone has gone home.",
  "user_id": "user_123"
}'
```

Using Python requests to generate a user risk profile:

```python
import requests

response = requests.post(
    "http://localhost:8000/user-risk-profile",
    json={
        "user_id": "user_123",
        "texts": [
            {
                "text": "I'm going to download the customer database tonight after everyone has gone home.",
                "timestamp": "2023-04-15T22:10:00Z"
            },
            {
                "text": "I need the admin password for the server room door."
            }
        ]
    }
)
print(response.json())
```

## Troubleshooting

If you encounter issues:

1. Check if the model files exist in the expected location
2. Run `./run.sh --test-model` to test model loading separately
3. Check the logs for specific error messages
4. Ensure all dependencies are installed correctly 