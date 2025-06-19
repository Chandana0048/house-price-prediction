# house-price-prediction
# 🏠 House Price Prediction using scikit-learn

A machine learning regression project built using Python and scikit-learn to predict house prices based on various features such as population, income, house age, and location-based variables. This project uses the California Housing dataset and demonstrates end-to-end regression workflow.


## 📌 Project Overview

Predicting house prices is a classic regression problem in data science. In this project, I developed a machine learning model to estimate house prices in California based on socio-economic and property attributes using regression algorithms.

The project walks through the complete ML pipeline including data preprocessing, model training, evaluation, and cross-validation.


## 🎯 Objectives

- Build a supervised **regression model** using scikit-learn
- Apply **data cleaning**, feature scaling, and preprocessing techniques
- Train and evaluate models using **R² Score** and **RMSE**
- Perform **cross-validation** to check model robustness
- Compare **Linear Regression** and **Random Forest Regressor**


## 🗂️ Dataset

- **Source:** Built-in scikit-learn **California Housing dataset**
- **Records:** 20,640 housing records
- **Features:** 8 numerical attributes (e.g., Population, HouseAge, MedInc)
- **Target:** Median house price in California districts


## 📚 Technologies Used

- Python  
- scikit-learn  
- Pandas, NumPy  
- Matplotlib, Seaborn  
- Google Colab / VS Code  


## 📈 Project Workflow

1. **Data Collection**  
   Loaded the California Housing dataset using scikit-learn’s `fetch_california_housing()` function.

2. **Exploratory Data Analysis (EDA)**  
   Inspected the data structure, summary statistics, and visualized feature relationships.

3. **Data Cleaning**  
   Verified null values, removed inconsistencies, and checked for outliers.

4. **Feature Scaling**  
   Applied **StandardScaler** for scaling numerical variables to improve model performance.

5. **Model Building**  
   Trained both **Linear Regression** and **Random Forest Regressor** models.

6. **Model Evaluation**  
   Assessed models using **R² Score**, **Mean Squared Error (MSE)**, and **Root Mean Squared Error (RMSE)**.

7. **Cross-Validation**  
   Performed 5-fold cross-validation to validate model stability.

8. **Results Summary**  
   Compared model performance and selected the best one based on metrics.


## 🔍 Key Results

- **Linear Regression R² Score:** ~0.57  
- **Random Forest R² Score:** Higher and better generalization  
- Computed **RMSE manually** due to a minor version-related issue with the scikit-learn `mean_squared_error()` function.


## 📝 Key Learnings

- Understood the complete regression pipeline from **data preprocessing to model evaluation**
- Learned to handle **scikit-learn version compatibility issues** during evaluation
- Compared linear and non-linear regression models
- Applied **cross-validation to check model robustness**
- Gained hands-on practice with **feature scaling** and evaluation metrics


## 📌 Future Improvements

- Apply **hyperparameter tuning** using GridSearchCV  
- Integrate **XGBoost or Gradient Boosting models** for performance optimization  
- Build a **Flask web app** for deploying the trained model  
- Visualize feature importance for Random Forest


## 📎 License

This project is open-source and available under the [MIT License](LICENSE).


## 📬 Contact

**Chandana K P**  

[LinkedIn](https://www.linkedin.com/in/chandana-puttanagappa)


## 🚀 Acknowledgements

Dataset from scikit-learn's open-source repository.  
Inspired by classic regression problems in data science.
