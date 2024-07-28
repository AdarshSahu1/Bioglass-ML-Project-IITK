import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample

# Load the dataset
file_path = 'C:\\Users\\29200\\Downloads\\Dataset\\dropkarungaColumn.xlsx'
data = pd.read_excel(file_path)

# Drop columns with too many NaNs and non-numerical columns
data_cleaned = data.drop(columns=['RESEARCH PAPER/ ARTICLE', 'VEGF'])

# Encode the 'Class' column
label_encoder = LabelEncoder()
data_cleaned['Class'] = label_encoder.fit_transform(data_cleaned['Class'])

# Separate features and target variables
features = data_cleaned.drop(columns=['Cell Viability 24', 'Cell Viability 48', 'Cell Viability 72', 'Cell Viability 96', 'Cell Viability 120', 'ALP 7', 'ALP 14', 'ALP 21'])
targets = data_cleaned[['Cell Viability 24', 'Cell Viability 48', 'Cell Viability 72', 'Cell Viability 96', 'Cell Viability 120', 'ALP 7', 'ALP 14', 'ALP 21']]

# Random Oversampling
X_resampled, y_resampled = resample(features, targets, replace=True, n_samples=len(features) * 2, random_state=42)

# Parameters for Gaussian noise
noise_level = 0.1  # Adjust the noise level as needed
n_samples_to_generate = 500  # Number of new samples to generate

# Generate synthetic samples by adding Gaussian noise
X_resampled_noise = np.vstack([features.values] + [features.values + noise_level * np.random.normal(size=features.values.shape) for _ in range(n_samples_to_generate)])
y_resampled_noise = np.vstack([targets.values] + [targets.values + noise_level * np.random.normal(size=targets.values.shape) for _ in range(n_samples_to_generate)])

# Convert resampled data to DataFrame
features_resampled = pd.DataFrame(X_resampled, columns=features.columns)
targets_resampled = pd.DataFrame(y_resampled, columns=targets.columns)

features_noise = pd.DataFrame(X_resampled_noise, columns=features.columns)
targets_noise = pd.DataFrame(y_resampled_noise, columns=targets.columns)

# Combine features and targets
oversampled_resampled = pd.concat([features_resampled, targets_resampled], axis=1)
oversampled_noise = pd.concat([features_noise, targets_noise], axis=1)

# Save to Excel files
oversampled_resampled.to_excel('C:\\Users\\29200\\Downloads\\Dataset\\oversampled_resampled.xlsx', index=False)
oversampled_noise.to_excel('C:\\Users\\29200\\Downloads\\Dataset\\oversampled_noise.xlsx', index=False)

