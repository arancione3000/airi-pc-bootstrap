import pandas as pd
from sklearn.preprocessing import StandardScaler

file_path_demos = "data/DEMoS_summary.csv"

# Read data
data = pd.read_csv(file_path_demos) 
print("Data Loaded!")

# Initialize the StandardScaler
scaler = StandardScaler()
# Fit and transform the selected columns
num_columns = data.select_dtypes(exclude=['object']).columns[1:]
data[num_columns] = scaler.fit_transform(data[num_columns])

# Set the path to the CSV file
csv_file = "data/DEMoS_summary_scaled.csv"
# Save the data summary DataFrame to a CSV file
data.to_csv(csv_file, index=False)