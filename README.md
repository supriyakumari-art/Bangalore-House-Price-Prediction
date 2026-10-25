# Bangalore House Price Prediction 🏡

## 📌 Project Overview

This project uses Machine Learning to predict house prices in Bangalore, India, based on property features such as location, total square footage, number of bedrooms (BHK), and number of bathrooms.

The project includes data preprocessing, exploratory data analysis, feature engineering, outlier detection, and the implementation and comparison of multiple regression algorithms to predict house prices.

## 🎯 Objectives

- Clean and preprocess the Bengaluru house price dataset.
- Handle missing values and inconsistent data.
- Convert property sizes and square footage into usable numerical features.
- Perform feature engineering and calculate price per square foot.
- Detect and remove outliers to improve data quality.
- Train and evaluate different regression models.
- Save the trained model for future predictions.

## 🛠️ Technologies Used

- **Python** — Programming language
- **Pandas** — Data manipulation and analysis
- **NumPy** — Numerical computations
- **Scikit-learn** — Machine learning and model evaluation
- **Pickle** — Saving the trained model
- **Jupyter Notebook / Google Colab** — Development environment

## 📂 Dataset

The project uses `bengaluru_house_prices.csv`, containing information about residential properties in Bangalore.

The features used in the prediction process include:

| Feature | Description |
|---|---|
| `location` | Location of the property |
| `total_sqft` | Total area of the property in square feet |
| `bath` | Number of bathrooms |
| `bhk` | Number of bedrooms |
| `price` | House price (target variable) |

The `price` column is the target variable that the model learns to predict.

## ⚙️ Project Workflow

### 1. Data Exploration
- Inspected the dataset using `head()`, `info()`, and `describe()`.
- Checked dataset dimensions and missing values.
- Examined the distribution of categorical features.

### 2. Data Cleaning
- Removed unnecessary columns such as `area_type`, `availability`, `society`, and `balcony`.
- Handled missing values in `location`, `size`, and `bath`.
- Extracted the number of bedrooms from the `size` column.
- Converted square footage ranges into numerical values.
- Standardized location names by removing extra spaces.
- Grouped infrequent locations into an `other` category.

### 3. Feature Engineering
- Created the `bhk` feature from the property size.
- Calculated `price_per_sqrt` to support outlier detection.
- Removed properties with less than 300 square feet per bedroom.

### 4. Outlier Detection and Removal
- Removed price-per-square-foot outliers using location-based mean and standard deviation.
- Applied BHK-based filtering to identify unusually low-priced properties compared with the previous BHK group.

### 5. Model Building
The dataset was split into training and testing sets using an 80:20 ratio.

Three regression algorithms were implemented:

- **Linear Regression** — Baseline model for predicting house prices.
- **Lasso Regression** — Uses L1 regularization to reduce less useful feature coefficients.
- **Ridge Regression** — Uses L2 regularization to control large coefficients.

A Scikit-learn pipeline was used to combine one-hot encoding of the `location` feature, feature scaling, and model training.

### 6. Model Evaluation
The models were evaluated using the **R² score**, which measures how well the predictions explain variation in the target values.

The scores for Linear Regression, Lasso Regression, and Ridge Regression were printed for comparison.

### 7. Model Saving
The trained Ridge Regression pipeline was saved as:

`RidgeModel.pkl`

The cleaned dataset was exported as:

`Cleaned_data.csv`

## 📁 Project Structure

```text
bangalore-house-price-prediction/
│
├── banglore_house_price_prediction.py
├── bengaluru_house_prices.csv
├── Cleaned_data.csv
├── RidgeModel.pkl
└── README.md
```

*Note: The CSV files and saved model are generated or required project artifacts. Include them in your repository only if you intend to share them.*

## 🚀 Getting Started

### Prerequisites

Install Python and the required libraries.

```bash
pip install pandas numpy scikit-learn jupyter
```

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/YOUR-USERNAME/bangalore-house-price-prediction.git
   ```

2. Navigate to the project directory:

   ```bash
   cd bangalore-house-price-prediction
   ```

3. Place `bengaluru_house_prices.csv` in the project directory.

4. Run the Python script:

   ```bash
   python banglore_house_price_prediction.py
   ```

The script performs preprocessing, trains and evaluates the regression models, exports the cleaned dataset, and saves the Ridge model.

## 📊 Results

The project compares the performance of three regression models using the R² score.

| Model | Evaluation Metric |
|---|---|
| Linear Regression | R² Score |
| Lasso Regression | R² Score |
| Ridge Regression | R² Score |

Actual scores depend on the dataset and execution environment. Run the script to obtain the numerical results.

## 🔮 Future Improvements

- Build an interactive web application using Streamlit or Flask.
- Allow users to enter property details and receive price predictions.
- Evaluate models using MAE, MSE, and RMSE in addition to R².
- Tune hyperparameters using cross-validation.
- Improve data validation and outlier detection.
- Add visualizations for property prices and location-wise trends.

## 👨‍💻 Author

**Your Name**

- GitHub: https://github.com/supriyakumari-art

## 📄 License

This project is intended for educational and learning purposes. Add a suitable open-source license if you plan to distribute it.

---

⭐ If you find this project useful, consider giving the repository a star!
