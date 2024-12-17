import streamlit as st
import pandas as pd
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import re
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset

# Load Spacy's English model for better tokenization
nlp = spacy.load('en_core_web_sm')

# Advanced Tokenizer for better query understanding
class AdvancedTokenizer:
    def __init__(self):
        self.operations = [
            "max", "min", "sum", "difference", "top", "mean", "average", 
            "groupby", "filter", "sort", "median", "count", "pivot", 
            "join", "merge", "null", "range", "date", "trend", "normalize", 
            "bin", "aggregate", "unique", "standardize"
        ]

    def tokenize(self, query):
        query = query.lower()
        doc = nlp(query)  # Use spacy's NLP model to process the query
        tokens = [token.text for token in doc if token.is_alpha or token.is_digit or token.text in ['>', '<', '=', '+', '-', '.', '/', '%']]  # Include operators
        operation = None
        columns = []
        conditions = []
        for token in tokens:
            if token in self.operations:
                operation = token
            elif token.isalpha():
                columns.append(token)
            elif token.isdigit():
                conditions.append(token)
            elif token in ['>', '<', '=', '+', '-', '%']:  # Include operators
                conditions.append(token)
        return operation, columns, conditions

# Initialize tokenizer instance
tokenizer = AdvancedTokenizer()

# Enhanced Neural Network without the Embedding layer
class EnhancedNLPModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(EnhancedNLPModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  # Fully connected layer
        self.fc2 = nn.Linear(hidden_size, output_size)  # Output layer
        self.bn1 = nn.BatchNorm1d(hidden_size)  # Batch normalization
        self.bn2 = nn.BatchNorm1d(output_size)  # Batch normalization
        self.dropout = nn.Dropout(0.5)  # Dropout for regularization

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # Pass through fully connected layer
        x = self.bn1(x)
        x = self.dropout(x)  # Apply dropout
        x = torch.relu(self.fc2(x))
        x = self.bn2(x)
        return x

# Function to predict operation from query
def predict_operation(query):
    operation, _, _ = tokenizer.tokenize(query)
    input_data = tfidf_vectorizer.transform([query]).toarray()  # Convert query to tf-idf vector
    input_tensor = torch.tensor(input_data, dtype=torch.float32)  # Convert to tensor
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
    predicted_label = torch.argmax(output, dim=1).item()
    return predicted_label

# Dataset for queries (example queries)
queries = [
    "What is the max sales?", 
    "Show top 3 products by sales", 
    "What is the sum of profit?", 
    "Find the difference in sales of A and B",
    "Group by Product and sum sales",
    "Filter Product by sales greater than 1000",
    "Sort products by sales",
    "Pivot by Category and sum sales",
    "Merge sales and products on ProductID",
    "Check for null values",
    "Find the range of sales",
    "Show trends in sales over time",
    "Normalize sales data",
    "Bin sales data into 5 categories",
    "Standardize sales data",
    "Show unique product categories"
]
labels = ["max", "top", "sum", "difference", "groupby", "filter", "sort", "pivot", "merge", "null", "range", "trend", "normalize", "bin", "standardize", "unique"]

# Preprocessing the queries: Use TF-IDF vectorization
tfidf_vectorizer = TfidfVectorizer()
X_tfidf = tfidf_vectorizer.fit_transform(queries)
y_labels = [i for i in range(len(labels))]  # Numeric labels for operations

# Train/test split (for example)
X_train = torch.tensor(X_tfidf.toarray(), dtype=torch.float32)
y_train = torch.tensor(y_labels, dtype=torch.long)

# Dataset class for the queries
class QueryDataset(Dataset):
    def __init__(self, queries, labels):
        self.queries = queries
        self.labels = labels

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        return self.queries[idx], self.labels[idx]

# Create DataLoader for batching
train_dataset = QueryDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)

# Neural network setup
input_size = X_train.shape[1]  # Feature size (number of words after TF-IDF)
hidden_size = 128
output_size = len(labels)

model = EnhancedNLPModel(input_size, hidden_size, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
epochs = 10
for epoch in range(epochs):
    for batch_data in train_loader:
        inputs, targets = batch_data
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

# Function to execute queries on the DataFrame and display action
def execute_query(query, df):
    # Get predicted operation based on the query
    operation = predict_operation(query)
    action = ''
    result = None

    if operation == 0:  # Max operation
        result = df.max().to_dict()
        action = 'Performed Max Operation on Data'
    elif operation == 1:  # Top N operation
        top_n_matches = re.findall(r'\d+', query)
        if not top_n_matches:
            result = {"error": "No valid number found for 'top N' operation."}
        else:
            top_n = int(top_n_matches[0])  # Extract number from query
            result = df.nlargest(top_n, df.columns[0]).to_dict(orient='records')
            action = f'Fetched Top {top_n} Records by {df.columns[0]}'
    elif operation == 2:  # Sum operation
        result = df.sum().to_dict()
        action = 'Calculated Sum of All Columns'
    elif operation == 3:  # Difference operation
        products = re.findall(r"Product [A-E]", query)  # Extract product names
        if len(products) == 2:
            diff = df[df['Product'] == products[0]]['Sales'].values[0] - df[df['Product'] == products[1]]['Sales'].values[0]
            result = {"difference": diff}
            action = f'Calculated Difference in Sales for Products {products[0]} and {products[1]}'
    elif operation == 4:  # Groupby operation (example)
        grouped = df.groupby('Product').sum()
        result = grouped.to_dict()
        action = 'Grouped Data by Product and Calculated Sum'
    elif operation == 5:  # Filter operation (example)
        condition_matches = re.findall(r'\d+', query)
        if not condition_matches:
            result = {"error": "No valid number found for 'filter' condition."}
        else:
            condition_value = int(condition_matches[0])  # Extract condition
            filtered = df[df['Sales'] > condition_value]
            result = filtered.to_dict(orient='records')
            action = f'Filtered Data where Sales are Greater than {condition_value}'
    elif operation == 6:  # Sort operation
        sorted_df = df.sort_values(by='Sales', ascending=False)
        result = sorted_df.to_dict(orient='records')
        action = 'Sorted Data by Sales'
    elif operation == 7:  # Mean operation (average)
        result = df.mean().to_dict()  # Calculate mean of all columns
        action = 'Calculated Mean of All Columns'

    return result, action

# Streamlit UI
st.title("Advanced Query Analysis with Real-World Dataset")
st.write("""This application lets you upload a CSV or Excel file, view it in its raw format, 
    convert the data to JSON format, and then perform complex queries on the data.""")

# Upload file
uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file is not None:
    # Read the uploaded file
    if uploaded_file.name.endswith("csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    # Show the raw dataset (CSV/Excel file)
    st.write("### Raw Dataset:")
    st.dataframe(df)

    # Convert the dataset to JSON format
    json_data = df.to_json(orient="records")
    st.write("### Dataset in JSON format:")
    st.json(json_data)

    # Extract columns from the dataset and display them
    extracted_columns = df.columns.tolist()
    st.write("### Extracted Columns in JSON format:")
    st.json(extracted_columns)

    # User input for the query
    user_query = st.text_input("Enter your query (e.g., 'What is the max sales?'): ")

    if user_query:
        result, action = execute_query(user_query, df)
        st.write(f"### Action Performed: {action}")
        st.write("### Query Result:")
        st.json(result)

        # Display result in a chart based on the action
        if isinstance(result, dict):
            st.write("### Data Visualization:")
            if 'difference' in result:
                st.bar_chart([result['difference']])
            else:
                st.line_chart(result)
