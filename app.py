import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import f1_score
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler, LabelEncoder
import holidays
import pickle
import io

# Function to load and prepare data
@st.cache_data
def load_data():
    df = pd.read_csv('lastfinal.csv')
    df = df.drop(['invoice_no'], axis=1)
    df['invoice_date'] = pd.to_datetime(df['invoice_date'], format='mixed')
    df['gender'] = df['gender'].map({'Male': 0, 'Female': 1})
    df['total_price'] = df['price'] * df['quantity']
    df['day_of_week'] = df['invoice_date'].dt.dayofweek
    df['is_holiday'] = df['invoice_date'].apply(lambda x: 1 if x in holidays.Turkey() else 0)
    df['invoice_month'] = df['invoice_date'].dt.month
    df['invoice_year'] = df['invoice_date'].dt.year
    
    # Label encoding for categorical variables
    label_encoder_category = LabelEncoder()
    df['category_encoded'] = label_encoder_category.fit_transform(df['category'])
    df['shopping_mall_encoded'] = label_encoder_category.fit_transform(df['shopping_mall'])
    
    # Feature scaling
    scaler = StandardScaler()
    df['age_scaled'] = scaler.fit_transform(df[['age']])
    
    return df

# Function to oversample data
def oversample_data(df):
    max_count = df['shopping_mall'].value_counts().max()
    resampled_dfs = []
    
    for mall in df['shopping_mall'].unique():
        mall_df = df[df['shopping_mall'] == mall]
        resampled_mall_df = resample(mall_df, replace=True, n_samples=max_count, random_state=42)
        resampled_dfs.append(resampled_mall_df)

    resampled_df = pd.concat(resampled_dfs)
    resampled_df = resampled_df.sample(frac=1, random_state=42).reset_index(drop=True)
    return resampled_df

# Streamlit App Interface
st.set_page_config(page_title='Location Prediction with XGBoost', layout='wide')
st.title('Retail Location Prediction & Sales Forecasting')
st.markdown("""
    ## Overview:
    This web app uses machine learning to predict retail outlet preferences and sales forecasts. 
    It uses customer demographics, product purchases, and seasonal data to train an XGBoost model.
""")

# Sidebar for interaction
st.sidebar.header('Model Settings')
st.sidebar.markdown("""
    This app allows you to upload your dataset and get predictions on retail outlet preferences.
""")

# Load the dataset
df = load_data()
st.write("### Original Data")
st.write(df.head())

# Perform Oversampling if selected
if st.sidebar.checkbox('Perform Oversampling'):
    df = oversample_data(df)
    st.write("### Oversampled Data")
    st.write(df.head())

# Features and Target
X = df[['gender', 'age_scaled', 'total_price', 'category_encoded', 'day_of_week', 'is_holiday', 'invoice_month', 'invoice_year']]
y = df[['shopping_mall_encoded']]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the XGBoost model
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

# Make Predictions
y_pred = model.predict(X_test)

# Calculate F1 Score
f1 = f1_score(y_test, y_pred, average='micro')

# Display Results
st.markdown("### Model Performance")
st.write(f"**F1 Score**: {f1:.4f}")

# Model Visualization
st.markdown("### Feature Importance")
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")

# Plotting Feature Importance
fig, ax = plt.subplots(figsize=(10, 6))
plot_data = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values(by='Importance', ascending=False)

sns.barplot(x='Importance', y='Feature', data=plot_data, ax=ax, palette='viridis')
st.pyplot(fig)

# Option to download the trained model
@st.cache
def save_model(model):
    model_file = io.BytesIO()
    pickle.dump(model, model_file)
    model_file.seek(0)
    return model_file

st.markdown("### Download Model")
st.download_button('Download Trained Model', save_model(model), file_name="xgboost_model.pkl")

# App Footer
st.markdown("""
    --- 
    Created with ❤️ by [Your Name].
    For business analytics and machine learning.
""")
