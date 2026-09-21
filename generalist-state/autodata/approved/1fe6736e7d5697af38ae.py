import pandas as pd
import numpy as np
import os

from sklearn.model_selection import KFold
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, recall_score
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import json
from pathlib import Path


import warnings

# Silence all warnings
warnings.filterwarnings("ignore")

def filter_dataset(data, speaker_id: int):
    # Load the dataset from file and return feature matrix and target values
    # Use pandas or suitable library to load the dataset    
    data = data[data['Speaker ID']==speaker_id].reset_index(drop=True)
    X = data.drop(columns=['Emotion']) # Feature matrix (6373 features)
    y = data[['Emotion']]  # Target values
    # Convert target values to numeric labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    # columns to drop when training
    columns_to_drop = X.columns[:4] # ['index', 'File', 'Selection', 'Gender', 'Speaker ID']
    return X, y, label_encoder, columns_to_drop

def split_dataset(X, y, test_size=0.2, random_state=42):
    # Split the dataset into training, development, and test sets
    # Ensure balanced samples per emotion 

    # Split the dataset into training and test sets based on gender and emotion
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

    # Further split the training set into training and development sets
    X_train, X_dev, y_train, y_dev = train_test_split(X_train, y_train, test_size=test_size, random_state=random_state, stratify=y_train)

    # Return the split datasets
    return X_train, X_dev, X_test, y_train, y_dev, y_test

def optimize_complexity(X_train, X_dev, y_train, y_dev, file_path, columns_to_drop, speaker_id):
    # Optimize the SVM complexity hyperparameter using the development set and save it in json
    
    # Define the range of complexity levels to evaluate
    complexity_levels = [2**i for i in range(30, 0, -1)] # 30 different values
    
    # Initialize variables to store the maximum UAR and the corresponding complexity level
    max_uar = 0.0
    optimized_complexity = None
    
    # Iterate over the complexity levels
    for complexity in tqdm(complexity_levels, total=len(complexity_levels), desc="Testing better C"):
        # Create an SVM classifier with the current complexity level and a linear kernel
        svm = SVC(kernel='linear', C=complexity, max_iter=200, cache_size=600)
        
        # Fit the SVM model to the development data
        svm.fit(X_train.drop(columns=columns_to_drop), y_train)
        
        # Predict the labels for the development data
        y_pred = svm.predict(X_dev.drop(columns=columns_to_drop))
        
        # Generate the classification report
        report = classification_report(y_dev, y_pred, output_dict=True)
        
        # Calculate the UAR (Unweighted Average Recall) score
        uar = report['macro avg']['recall']
        
        # Check if the current UAR is higher than the maximum UAR
        if uar > max_uar:
            max_uar = uar
            optimized_complexity = complexity
    
    # Read SON file
    with open(file_path, "r") as json_file:
        result = json.load(json_file)
    
    # Save the updated dictionary as a JSON file
    with open(file_path, 'w') as json_file:
        result[f'C_speakerid_{speaker_id}'] = optimized_complexity
        json.dump(result, json_file)
    
    return optimized_complexity

    

def train_final_model(X_train_final, y_train_final, X_dev_final, y_dev_final, complexity, columns_to_drop):
    # Combine the training set and development set to create a final training set
    X_final = pd.concat([X_train_final, X_dev_final])
    y_final = np.concatenate((y_train_final, y_dev_final), axis=0)

    # Train an SVM model with the optimized complexity level
    svm = SVC(kernel='linear', C=complexity, max_iter=200, cache_size=600)
    svm.fit(X_final.drop(columns=columns_to_drop), y_final)

    # Return the trained final model
    return svm

def save_final_test(label_encoder, complexity, columns_to_drop,
                    X_train_final, y_train_final, X_dev_final, y_dev_final, X_test, y_test,
                    file_path):
    
    model = train_final_model(X_train_final, y_train_final, X_dev_final, y_dev_final, complexity, columns_to_drop)

    # Concatenate X_train_final, X_dev_final, and X_test
    X_all = pd.concat([X_train_final, X_dev_final, X_test])

    # Create dataset labels for X_train_final, X_dev_final, and X_test
    dataset_labels = ['train'] * len(X_train_final) + ['dev'] * len(X_dev_final) + ['test'] * len(X_test)

    # Predict y for X_all
    y_pred = model.predict(X_all.drop(columns=columns_to_drop))

    # Convert numeric labels to original string labels using label_encoder
    y_true_labels = label_encoder.inverse_transform(np.concatenate((y_train_final, y_dev_final, y_test), axis=0))
    y_pred_labels = label_encoder.inverse_transform(y_pred)

    # Create a DataFrame with X, y_true, y_pred, and dataset labels
    df = X_all.copy()[columns_to_drop]
    df['y_true'] = y_true_labels
    df['y_pred'] = y_pred_labels
    df['Dataset'] = dataset_labels

    # Save the DataFrame as a CSV file
    df.to_csv(file_path, index=False)

def cross_validation(X, y, optimized_complexity, label_encoder, file_path, columns_to_drop):
    # Perform cross-validation by considering all six possible permutations of the folds
    kf = KFold(n_splits=6, shuffle=True, random_state=42)

    fold = 0
    for train_index, test_index in kf.split(X):
        # Split the data into training and test sets
        X_train, X_test = X.iloc[train_index,:], X.iloc[test_index,:]
        y_train, y_test = y[train_index], y[test_index]

        # Train the final model on the combined training and development sets
        model = SVC(kernel='linear', C=optimized_complexity, max_iter=200, cache_size=600)

        # Fit the model
        model.fit(X_train.drop(columns=columns_to_drop),y_train)

        # Concatenate X_train_final, X_dev_final, and X_test
        X_all = pd.concat([X_train, X_test])

        # Create dataset labels for X_train_final, X_dev_final, and X_test
        dataset_labels = [1] * len(X_train) + [0] * len(X_test)

        # Predict y for X_all
        y_pred = model.predict(X_all.drop(columns=columns_to_drop))

        # Convert numeric labels to original string labels using label_encoder
        y_true_labels = label_encoder.inverse_transform(np.concatenate((y_train, y_test), axis=0))
        y_pred_labels = label_encoder.inverse_transform(y_pred)

        # Create a DataFrame with X, y_true, y_pred, and dataset labels
        df = X_all.copy()[columns_to_drop]
        df['y_true'] = y_true_labels
        df['y_pred'] = y_pred_labels
        df['Train'] = dataset_labels
        df['Fold'] = fold
        fold+=1

        #Check if file_path exists
        if os.path.exists(file_path):
            df_old = pd.read_csv(file_path)
            df_new = pd.concat([df,df_old])
        else:
            df_new = df.copy()

        # Save the DataFrame as a CSV file
        df_new.to_csv(file_path, index=False)

def createAggCVResult(folder_path):

    # Initialize an empty list to store the results
    results_list = []

    # Iterate over the speaker IDs
    for speaker_id in range(1, 70):
        # Construct the file path
        file_path = folder_path + f"CE_SpeakerID_{speaker_id}_TrainDevTest_Framework.csv"
        
        # Check if the file exists
        if Path(file_path).is_file():
            # Read the CSV file
            df = pd.read_csv(file_path)
            
            # Filter the "Dataset" column to "test"
            df_test = df[df["Dataset"] == "test"]
            
            # Get the unique classes from y_true and y_pred
            classes = df_test["y_true"].unique()
            
            # Initialize a dictionary to store the recall for each class
            recall_dict = {"speaker_id": speaker_id}
            
            # Calculate the recall for each class
            for cls in classes:
                y_true = df_test["y_true"] == cls
                y_pred = df_test["y_pred"] == cls
                recall = recall_score(y_true, y_pred)
                recall_dict[cls] = recall
            
            # Append the recall dictionary to the results list
            results_list.append(recall_dict)

    # Convert the results list to a DataFrame
    result_df = pd.DataFrame(results_list)
    result_df.to_csv(folder_path + 'AggCVResult.csv')


def main():
    
    # Path to the DEMoS dataset
    file_path_demos = "data\DEMoS_summary_scaled.csv"

    # Read data
    data = pd.read_csv(file_path_demos) 
    print("Data Loaded!")

    unique_speakers_id = np.sort(data['Speaker ID'].unique())

    for speaker_id in unique_speakers_id:
        
        # Read the csv file and filter by speaker_id
        X, y, label_encoder, columns_to_drop = filter_dataset(data = data, speaker_id=speaker_id)

        # Split in train, dev and test
        X_train, X_dev, X_test, y_train, y_dev, y_test = split_dataset(X, y, test_size=0.2, random_state=42)
        print("Data Splitted")
        
        # Find the best complexity and save it
        file_path_complexity_file = "results/corpus_evaluation/complexity.json"
        complexity = optimize_complexity(X_train, X_dev, y_train, y_dev, file_path_complexity_file, columns_to_drop, speaker_id)
        print("Data Optimized")

        # Save final results
        file_path_result = f"results/corpus_evaluation/CE_SpeakerID_{speaker_id}_TrainDevTest_Framework.csv"
        save_final_test(label_encoder, complexity, columns_to_drop,
                    X_train, y_train, X_dev, y_dev, X_test, y_test,
                    file_path_result)
        
        # Save cross_validation_results results
        file_path_result_cv = f"results/corpus_evaluation/CE_SpeakerID_{speaker_id}_CV_Framework.csv"
        cross_validation(X, y, complexity, label_encoder, file_path_result_cv, columns_to_drop)
    
    
    # Create AGG Result from Cross Validation Analysis
    folder_path = "results/corpus_evaluation/"
    createAggCVResult(folder_path)

if __name__ == '__main__':
    main()