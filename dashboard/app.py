import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from keras.models import Model
from keras.layers import Input, Dense
import plotly.express as px

# Load the processed data
data = pd.read_csv('data/processed/processed_data.csv')

# Convert InvoiceDate to datetime
data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'])

# Sidebar filters
st.sidebar.header('Filter Options')
country_filter = st.sidebar.multiselect('Select Country', options=data['Country'].unique(), default=data['Country'].unique())
date_filter = st.sidebar.date_input('Select Date Range', [data['InvoiceDate'].min().date(), data['InvoiceDate'].max().date()])

# Convert date_filter to datetime64[ns]
start_date = pd.to_datetime(date_filter[0])
end_date = pd.to_datetime(date_filter[1])

# Filter data based on user input
filtered_data = data[(data['Country'].isin(country_filter)) & (data['InvoiceDate'].between(start_date, end_date))]

# Sample a smaller subset of data for visualization to reduce memory usage
sample_data = filtered_data.sample(frac=0.1, random_state=42)  # Adjust the fraction as needed

# Select features for clustering
features = ['Quantity', 'UnitPrice', 'TotalPurchaseValue']
X = sample_data[features]

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Define the autoencoder model
input_dim = X_scaled.shape[1]
encoding_dim = 2  # Dimension of the encoded representation

input_layer = Input(shape=(input_dim,))
encoder = Dense(encoding_dim, activation="relu")(input_layer)
decoder = Dense(input_dim, activation="sigmoid")(encoder)

autoencoder = Model(inputs=input_layer, outputs=decoder)
autoencoder.compile(optimizer='adam', loss='mse')

# Train the autoencoder
autoencoder.fit(X_scaled, X_scaled, epochs=50, batch_size=256, shuffle=True, validation_split=0.2)

# Extract the encoded features
encoder_model = Model(inputs=input_layer, outputs=encoder)
X_encoded = encoder_model.predict(X_scaled)

# Apply K-means clustering on encoded features
kmeans = KMeans(n_clusters=5, random_state=42)
sample_data['segment'] = kmeans.fit_predict(X_encoded)

# Main Header
st.title('Customer Segmentation Dashboard')
st.markdown('Identify and analyze customer segments for targeted marketing strategies.')

# Overview Panel
st.header('Overview')
total_customers = filtered_data['CustomerID'].nunique()
total_revenue = filtered_data['TotalPurchaseValue'].sum()
num_segments = sample_data['segment'].nunique()
avg_purchase_value = filtered_data['TotalPurchaseValue'].mean()

st.metric('Total Customers', total_customers)
st.metric('Total Revenue', f"${total_revenue:,.2f}")
st.metric('Number of Segments', num_segments)
st.metric('Average Purchase Value', f"${avg_purchase_value:,.2f}")

# Pie Chart: Proportion of customers in each segment
st.subheader('Customer Segment Distribution')
fig, ax = plt.subplots()
segment_counts = sample_data['segment'].value_counts()
ax.pie(segment_counts, labels=segment_counts.index, autopct='%1.1f%%', startangle=90)
ax.axis('equal')
st.pyplot(fig)

# Bar Chart: Revenue contribution by each segment
st.subheader('Revenue Contribution by Segment')
fig, ax = plt.subplots()
segment_revenue = sample_data.groupby('segment')['TotalPurchaseValue'].sum()
sns.barplot(x=segment_revenue.index, y=segment_revenue.values, ax=ax)
ax.set_xlabel('Segment')
ax.set_ylabel('Total Revenue')
st.pyplot(fig)

# Segment Insights Panel
st.header('Segment Insights')
segment_summary = sample_data.groupby('segment').agg({
    'CustomerID': 'nunique',
    'Quantity': 'mean',
    'TotalPurchaseValue': 'mean'
}).reset_index()
segment_summary.columns = ['Segment', 'Number of Customers', 'Average Quantity', 'Average Purchase Value']
st.dataframe(segment_summary)

# Radar Chart: Comparing RFM values across segments
st.subheader('RFM Comparison Across Segments')
# Assuming RFM values are calculated and added to the sample_data
# For demonstration, we'll use Quantity, UnitPrice, and TotalPurchaseValue as proxies for RFM
fig = px.line_polar(sample_data, r='Quantity', theta='segment', line_shape='linear')
st.plotly_chart(fig)

# Correlation Heatmap
st.subheader('Correlation Heatmap')
fig, ax = plt.subplots(figsize=(10, 6))
numeric_cols = sample_data.select_dtypes(include=['float64', 'int64']).columns
corr = sample_data[numeric_cols].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
ax.set_title('Correlation Heatmap')
st.pyplot(fig)

# Filtering and Customization Panel
st.sidebar.header('Custom Segmentation')
num_clusters = st.sidebar.slider('Select Number of Clusters', min_value=2, max_value=10, value=5)
selected_features = st.sidebar.multiselect('Select Features for Clustering', options=features, default=features)

# Reapply K-means clustering with user-selected parameters
X_custom = sample_data[selected_features]
X_custom_scaled = scaler.fit_transform(X_custom)
X_custom_encoded = encoder_model.predict(X_custom_scaled)
kmeans_custom = KMeans(n_clusters=num_clusters, random_state=42)
sample_data['custom_segment'] = kmeans_custom.fit_predict(X_custom_encoded)

# Customer Segmentation Map
st.header('Customer Segmentation Map')
fig = px.scatter_geo(filtered_data, locations="Country", locationmode='country names', color="TotalPurchaseValue",
                     hover_name="Country", size="TotalPurchaseValue", projection="natural earth")
st.plotly_chart(fig)

# Segment Explorer Panel
st.header('Segment Explorer')
selected_segment = st.selectbox('Select Segment', options=sample_data['segment'].unique())
segment_data = sample_data[sample_data['segment'] == selected_segment]
st.write(f"Top Customers in Segment {selected_segment}")
top_customers = segment_data.groupby('CustomerID')['TotalPurchaseValue'].sum().sort_values(ascending=False).head(10)
st.dataframe(top_customers)

st.write(f"Top Products in Segment {selected_segment}")
top_products = segment_data.groupby('Description')['Quantity'].sum().sort_values(ascending=False).head(10)
st.dataframe(top_products)