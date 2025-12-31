#  Car Price Prediction using Machine Learning

This project is an **end-to-end Machine Learning pipeline** to predict the **selling price of a used car** based on its features such as year, present price, fuel type, transmission, kilometers driven, etc.

The project is built with a **production mindset**, moving from a Jupyter notebook to modular Python scripts with proper preprocessing, training, and inference workflows.

---

## 📌Problem Statement

Predict the **Selling Price** of a car using historical car data by learning patterns from features like:

    - Year of manufacture
    - Present market price
    - Kilometers driven
    - Fuel type
    - Seller type
    - Transmission
    - Number of previous owners

---
## 💡Solution Overview

- Performed **Exploratory Data Analysis (EDA)** and experimentation in a Jupyter Notebook
- Selected **RandomForestRegressor** as the final model based on performance
- Implemented **robust preprocessing** to handle categorical variables safely
- Built **production-ready scripts** for training and prediction
- Ensured **feature consistency** between training and inference

---

## 📂Project Structure

```text
Car_Price_Prediction/
│
├── data/
│   └── CarPrice_Assignment.csv
│
├── notebooks/
│   └── Car_Price_Prediction.ipynb
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── preprocess.py
│   ├── train.py
│   └── predict.py
│
├── model.pkl
├── scaler.pkl
├── features.pkl
├── requirements.txt
└── README.md
```

## ⚙️Tech Stack

    - Python
    - Pandas, NumPy
    - Scikit-learn
    - Jupyter Notebook
    - Git & GitHub

## 🤖Machine Learning Pipeline
    1. Data Loading
       - CSV data loaded from data/

    2. Preprocessing
       - Drop irrelevant columns (Car_Name)
       - One-hot encode categorical variables
       - Scale numerical features
       - Save feature columns for inference consistency

    3. Model Training
       - Algorithm: RandomForestRegressor
       - Train-test split
       - Performance evaluated using R² Score

    4. Model Persistence
      - Trained model saved as model.pkl
      - Scaler saved as scaler.pkl
      - Feature columns saved as features.pkl

    5. Prediction
      - CLI-based prediction using predict.py
      - Safe handling of categorical and missing features

## 🚀 How to Run the Project
 **1. Clone the repository**

    git clone <my-repo-url>
    cd Car_Price_Prediction

**2. Create & activate virtual environment**

    python -m venv venv
    venv\Scripts\activate

**3. Install dependencies**

    pip install -r requirements.txt

**4. Train the model**

    python -m src.train

**Expected output:**

    Training completed | R2 Score: 0.96
    Model & scaler saved successfully

**5. Run prediction**

    python -m src.predict

**Example output:**

    Predicted Selling Price: 2.85

## 🌟Model Performance

    - Model Used: RandomForestRegressor
    - Metric: R² Score
    - Achieved R²: ~0.96

## 🧪 Sample Input Used for Prediction

    {
        "Year": 2017,
        "Present_Price": 3.6,
        "Kms_Driven": 2135,
        "Fuel_Type": "Petrol",
        "Seller_Type": "Dealer",
        "Transmission": "Manual",
        "Owner": 0
    }

## 🏆 Key Learnings

    - Handling categorical variables safely in production
    
    - Avoiding feature mismatch between training and inference
    
    - Building modular, reusable ML code
    
    - Transitioning from notebook-based ML to production scripts
    
    - Debugging real-world ML pipeline issues

## 📌 Future Improvements
    - Add unit tests

    - Add logging

    - Deploy as a REST API

    - Experiment with other regression models

## 👹Author

    Aspiring Machine Learning / Data Scientist
                                - Mad_titaN 

⭐ If you like this project, consider giving it a star!