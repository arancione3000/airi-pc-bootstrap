import pandas as pd
import io

# This example demonstrates how to load datasets into a Pandas DataFrame,
# a core concept discussed in the article "Jupyter Notebook'ta Pandas Kullanarak Veri Seti Yükleme Rehberi".

# --- Example 1: Loading data from a CSV string ---

# Simulate CSV file content as a multi-line string.
# In a real scenario, this data would typically come from a file (e.g., 'data.csv').
csv_data = """
name,age,city
Alice,30,New York
Bob,24,London
Charlie,35,Paris
David,29,Berlin
"""

# Use io.StringIO to treat the string data as a file-like object.
# This allows Pandas to read from an in-memory string as if it were a physical file.
csv_file_like_object = io.StringIO(csv_data)

# Load the CSV data into a Pandas DataFrame.
# This is a primary method for loading tabular datasets, as highlighted in the article.
df_csv = pd.read_csv(csv_file_like_object)

print("--- Loaded CSV Dataset ---")
print("First 5 rows:")
print(df_csv.head())
print("\nDataFrame Info:")
df_csv.info()
print("\n" + "="*40 + "\n")


# --- Example 2: Loading data from a JSON string ---

# Simulate JSON file content as a multi-line string.
# Pandas also supports loading other common formats like JSON, Excel, SQL databases, etc.
json_data = """
[
    {"name": "Eve", "age": 28, "city": "Rome"},
    {"name": "Frank", "age": 42, "city": "Madrid"},
    {"name": "Grace", "age": 22, "city": "Tokyo"}
]
"""

# Use io.StringIO for JSON data as well, treating the string as a file.
json_file_like_object = io.StringIO(json_data)

# Load the JSON data into a Pandas DataFrame.
# This demonstrates loading a different common data format, expanding on the article's scope.
df_json = pd.read_json(json_file_like_object)

print("--- Loaded JSON Dataset ---")
print("First 5 rows:")
print(df_json.head())
print("\nDataFrame Info:")
df_json.info()
print("\n" + "="*40 + "\n")

# To load from actual files, you would replace io.StringIO(data_string)
# with the file path, e.g., pd.read_csv('path/to/your/file.csv') or pd.read_json('path/to/your/file.json')
