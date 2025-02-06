# ShopCastML

This repository contains two machine learning models:  
1. **Customer Recommendation Model** - Predicts products customers are likely to purchase based on past purchases and demographics.  
2. **Location Prediction Model** - Identifies which shopping mall a customer is likely to visit based on transaction history and external factors.  

## Features
- **Customer Recommendation Model**  
  - Predicts product preferences using **Random Forest Classifier**.  
  - Uses **feature engineering** to extract insights from customer purchase history.  
  - Handles categorical variables with **Label Encoding**.  

- **Location Prediction Model**  
  - Forecasts customer visits to shopping malls using **ensemble models (Random Forest, AdaBoost, Gradient Boosting)**.  
  - Incorporates **holiday data & day-of-week effects**.  
  - Balances dataset using **SMOTE & resampling techniques**.  

## Tech Stack
- **Python** (pandas, NumPy, scikit-learn, imbalanced-learn, holidays)  
- **Machine Learning** (Random Forest, Gradient Boosting, AdaBoost, SMOTE)  
- **Data Preprocessing** (Label Encoding, Standard Scaling)  
- **Visualization** (Matplotlib, Seaborn)  

