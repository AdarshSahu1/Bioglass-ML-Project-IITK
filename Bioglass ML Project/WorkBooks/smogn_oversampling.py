import pandas as pd
import smogn
import numpy as np

# Load the dataset
file_path = "C:\\Users\\29200\\Downloads\\Dataset\\dropkarungaColumn.xlsx"
df = pd.read_excel(file_path)

# Drop the specified columns
df_dropped = df.drop(columns=['RESEARCH PAPER/ ARTICLE', 'Class'])

# Define the target variables
target_columns = ['SiO2', 'B2O3', 'CaO', 'Na2O', 'P2O5', 'Ce', 'Ce2O3', 'CeO2']

# Separate the features and target variables
X = df_dropped.drop(columns=target_columns)
y = df_dropped[target_columns]

# Define a custom relevance function
def custom_relevance(y):
    relevance = {
        "lower": y.min(),
        "upper": y.max(),
        "lower_quantile": np.percentile(y, 25),
        "upper_quantile": np.percentile(y, 75)
    }
    return relevance

# Apply SMOGN for regression
def smogn_apply(X, y, target_column):
    data = pd.concat([X, y[target_column]], axis=1)
    relevance = custom_relevance(y[target_column])
    data_smogn = smogn.smoter(
        data, 
        y=target_column, 
        rel_method='range', 
        rel_ctrl_pts_rg=[[relevance['lower'], 0], [relevance['lower_quantile'], 0.5], [relevance['upper_quantile'], 0.5], [relevance['upper'], 1]]
    )
    return data_smogn.drop(columns=[target_column]), data_smogn[target_column]

X_resampled_list = []
y_resampled_list = []

for col in target_columns:
    X_res, y_res = smogn_apply(X, y, col)
    X_resampled_list.append(X_res)
    y_resampled_list.append(y_res)

# Combine all resampled data
X_resampled = pd.concat(X_resampled_list).drop_duplicates().reset_index(drop=True)
y_resampled = pd.concat(y_resampled_list).drop_duplicates().reset_index(drop=True)
df_resampled = pd.concat([X_resampled, y_resampled], axis=1)

# Save the resampled DataFrame to a new Excel file
output_file_path = "C:\\Users\\29200\\Downloads\\Dataset\\resampled_dataset.xlsx"
df_resampled.to_excel(output_file_path, index=False)

print("SMOGN oversampling completed and saved to:", output_file_path)
