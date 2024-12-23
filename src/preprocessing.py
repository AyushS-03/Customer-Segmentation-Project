import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def load_data(file_path):
    # Load the dataset with a limited number of rows to save memory
    data = pd.read_excel(file_path, nrows=10000)  # Adjust nrows as needed
    return data

def clean_data(df):
    # Remove duplicates
    df = df.drop_duplicates()
    
    # Handle missing values
    df.dropna(subset=['CustomerID'], inplace=True)
    df.fillna(0, inplace=True)
    
    return df

def normalize_data(df):
    scaler = MinMaxScaler()
    
    # Normalize numerical columns only
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    
    return df

def feature_engineering(df):
    # Example feature engineering: create a total purchase value column
    if 'TotalPurchaseValue' not in df.columns:
        df['TotalPurchaseValue'] = df['Quantity'] * df['UnitPrice']
    
    return df

def preprocess_data(file_path):
    # Load data
    df = load_data(file_path)
    
    # Clean data
    df = clean_data(df)
    
    # Convert InvoiceDate to datetime
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    
    # Normalize data
    df = normalize_data(df)
    
    # Feature engineering
    df = feature_engineering(df)
    
    return df

if __name__ == "__main__":
    # Example usage
    raw_data_path = '../data/raw/Online_Retail.xlsx'
    processed_data = preprocess_data(raw_data_path)
    processed_data.to_csv('../data/processed/processed_data.csv', index=False)