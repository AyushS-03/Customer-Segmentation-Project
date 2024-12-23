# Customer Segmentation Project

## Project Overview
This project aims to segment customers using unsupervised machine learning techniques to provide actionable business insights.

## Progress
- **Data Collection**: Collected raw data from `Online_Retail.xlsx`.
- **Data Preprocessing**: 
  - Handled missing values.
  - Converted `InvoiceDate` to datetime.
  - Created `TotalPurchaseValue` feature.
  - Normalized numerical features.
- **Exploratory Data Analysis**: Completed.
- **Model Development**: 
  - Implemented K-means clustering with autoencoder-based feature extraction.
  - Determined optimal number of clusters using Elbow Method.
  - Trained the model and evaluated using Silhouette Score and Davies-Bouldin Index.
- **Visualization and Insights**: Completed.
- **Interactive Dashboard**: In progress.

## How to Run
1. Ensure all dependencies are installed:
    ```sh
    pip install -r requirements.txt
    ```
2. Preprocess the data:
    ```sh
    python src/preprocessing.py
    ```
3. Follow the notebooks for EDA, clustering, and visualization.
4. Run the Streamlit dashboard:
    ```sh
    streamlit run dashboard/app.py
    ```

## Notes for Copilot
- Ensure memory-efficient operations for low-end PCs.
- Optimize code for performance.