
# Price-Predictor using AWS-SageMaker

This project aims to predict flight prices using machine learning techniques. The model is trained on a dataset that includes various features affecting flight prices and utilizes AWS SageMaker for model training and deployment.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Technologies Used](#technologies-used)
- [File Descriptions](#file-descriptions)
- [Installation Instructions](#installation-instructions)
- [Usage](#usage)
- [Model Training](#model-training)
- [Deployment](#deployment)
- [Contributions](#contributions)

## Project Overview
The project aims to accurately forecast flight prices using a dataset that includes various relevant features. Utilizing the XGBoost algorithm, known for its efficiency in regression tasks, the project involved thorough data cleaning, preprocessing, and exploratory data analysis to extract meaningful insights. Feature engineering techniques were employed to transform raw data into informative inputs, while hyperparameter tuning optimized model performance. The final model was deployed via a Streamlit application, enabling users to input flight details and receive real-time price predictions.

## Dataset
The dataset used for training the model consists of flight price information along with several features:
- **airline**
- **date_of_journey**
- **source**
- **destination**
- **dep_time**
- **arrival_time**
- **duration**
- **total_stops**
- **additional_info**
- **price** (target variable)

## Technologies Used
- Python
- Streamlit
- XGBoost
- Scikit-learn
- Pandas
- Feature-engine
- AWS SageMaker
- Boto3

## File Descriptions
- **data_cleaning.py**: Contains functions to clean and preprocess the dataset.
- **eda.py**: Explores the dataset and visualizes relationships between features and the target variable.
- **feature_engineering.py**: Implements feature extraction and transformation techniques.
- **model_tuning.py**: Fine-tunes the XGBoost model using hyperparameter optimization.
- **app.py**: The main Streamlit application that allows users to input flight details and receive price predictions.

## AWS Configuration
- The model training was conducted on an AWS SageMaker EC2 instance of type `ml.t2.medium`.
- Data was stored and accessed via AWS S3 buckets.
- Boto3 was used for interaction with AWS services.

## Installation Instructions
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd flight-price-predictor
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
Start the Streamlit application:

```bash
streamlit run app.py
```

Open the web browser and navigate to `http://localhost:8501`.

Input the required details (airline, date of journey, source, destination, etc.) and click "Predict" to get the predicted flight price.

## Model Training
The model is trained on an EC2 instance using AWS SageMaker. The process includes data cleaning, exploratory data analysis (EDA), feature engineering, and model tuning to achieve the best performance.

## Deployment
The application is deployed using Streamlit, allowing users to interact with the model via a user-friendly web interface.

## Contributions
Contributions to this project are welcome. Please fork the repository and submit a pull request.


