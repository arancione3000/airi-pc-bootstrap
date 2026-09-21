import os
import pandas as pd
import opensmile
from tqdm import tqdm

# Set the path to the DEMOS folder
folder_path = "data/DEMOS"

# Set the path to the CSV file
csv_file = "data/DEMoS_summary.csv"

# Initialize the OpenSMILE instance
smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.ComParE_2016,
    feature_level=opensmile.FeatureLevel.Functionals,
)

# Create an empty list to store the data summary DataFrames
data_summary_list = []

# Get the total number of WAV files in the folder
num_files = len([filename for filename in os.listdir(folder_path) if filename.endswith(".wav")])

# Iterate over the WAV files in the folder with a progress bar
for filename in tqdm(os.listdir(folder_path), total=num_files, desc="Processing files"):
    if filename.endswith(".wav"):
        # Get the file's name and extract relevant information
        file_path = os.path.join(folder_path, filename)
        name = os.path.splitext(filename)[0]
        file_info = name.split("_")

        # Extract emotion, gender, and speaker ID from the file name
        emotion = file_info[-1][:3]
        gender = file_info[1][0]
        speaker_id = file_info[2]
        selection = file_info[0]

        # Process the audio file using OpenSMILE
        features = smile.process_file(file_path)

        # Add additional columns to the features DataFrame
        features["File"] = filename
        features["Selection"] = selection
        features["Emotion"] = emotion
        features["Gender"] = gender
        features["Speaker ID"] = speaker_id

        # Append the features DataFrame to the data summary list
        data_summary_list.append(features)

# Concatenate the data summary DataFrames into a single DataFrame
data_summary = pd.concat(data_summary_list, ignore_index=True)

# Reorder the columns
columns_order = ["File", "Selection", "Emotion", "Gender", "Speaker ID"] + list(data_summary.columns[:-5])
data_summary = data_summary.reindex(columns=columns_order)


# Save the data summary DataFrame to a CSV file
data_summary.to_csv(csv_file, index=False)

print("Data summary saved to:", csv_file)