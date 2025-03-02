# Price Prediction Using Spark ML

## Overview
This project leverages Apache Spark ML to build a regression model for predicting house prices based on the California Housing Prices dataset. The dataset contains various features, such as location, number of rooms, and median income, which influence housing prices. The model is trained and evaluated using different regression algorithms.

## Dataset
- **Source:** [California Housing Prices](https://www.kaggle.com/datasets/camnugent/california-housing-prices)
- **Features:**
  - `longitude`: Geographic coordinate (westward position)
  - `latitude`: Geographic coordinate (northward position)
  - `housingMedianAge`: Median age of houses in a block
  - `totalRooms`: Total number of rooms per block
  - `totalBedrooms`: Total number of bedrooms per block
  - `population`: Population per block
  - `households`: Number of households per block
  - `medianIncome`: Median household income
  - `medianHouseValue`: Median house price (Target Variable)
  - `oceanProximity`: Proximity to the ocean (categorical feature)

## Environment Setup
- This notebook is designed to run in **Google Colab**.
- The necessary dependencies, including PySpark, must be installed:
  ```python
  !pip install pyspark
  ```
- A Spark session is initialized before data processing:
  ```python
  from pyspark.sql import SparkSession
  spark = SparkSession.builder.appName('PricePrediction').getOrCreate()
  ```

## Data Preprocessing
1. **Loading Data:**
   ```python
   dataset = spark.read.csv('/path/to/housing.csv', inferSchema=True, header=True)
   ```
2. **Handling Missing Values:**
   - The `total_bedrooms` column has missing values.
   - We use **Imputer** to replace missing values with the median:
     ```python
     from pyspark.ml.feature import Imputer
     imputer = Imputer(inputCols=["total_bedrooms"], outputCols=["total_bedrooms_imputed"]).setStrategy("median")
     dataset = imputer.fit(dataset).transform(dataset)
     ```
3. **Encoding Categorical Features:**
   - `oceanProximity` is converted into numerical format using **StringIndexer**:
     ```python
     from pyspark.ml.feature import StringIndexer
     indexer = StringIndexer(inputCol="oceanProximity", outputCol="oceanProximityIndex")
     dataset = indexer.fit(dataset).transform(dataset)
     ```
4. **Feature Scaling:**
   - Standardizing numerical features using **StandardScaler**:
     ```python
     from pyspark.ml.feature import StandardScaler
     scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures")
     data = scaler.fit(data).transform(data)
     ```
5. **Feature Vectorization:**
   - Combining all feature columns into a single vector:
     ```python
     from pyspark.ml.feature import VectorAssembler
     assembler = VectorAssembler(inputCols=["longitude", "latitude", "housingMedianAge", "totalRooms", "totalBedrooms_imputed", "population", "households", "medianIncome", "oceanProximityIndex"], outputCol="features")
     data = assembler.transform(dataset).select("features", "medianHouseValue")
     ```

## Model Training
- The dataset is split into **80% training** and **20% testing**:
  ```python
  train, test = data.randomSplit([0.8, 0.2])
  ```
- Various regression models are implemented:
  - **Linear Regression**
  - **Decision Tree Regressor**
  - **Random Forest Regressor**
  - **Gradient Boosted Tree Regressor**

### Example: Training a Linear Regression Model
```python
from pyspark.ml.regression import LinearRegression
lr = LinearRegression(featuresCol="features", labelCol="medianHouseValue")
lr_model = lr.fit(train)
predictions = lr_model.transform(test)
```

## Model Evaluation
- The models are evaluated using **Root Mean Squared Error (RMSE)** and **R² Score**:
  ```python
  from pyspark.ml.evaluation import RegressionEvaluator
  evaluator = RegressionEvaluator(labelCol="medianHouseValue", predictionCol="prediction", metricName="rmse")
  rmse = evaluator.evaluate(predictions)
  print("RMSE:", rmse)
  ```

## Results
- The best performing model can be chosen based on evaluation metrics.
- The insights can be used to predict house prices efficiently using Spark ML.

## Conclusion
This project demonstrates the use of **Apache Spark ML** for regression modeling in large datasets. It highlights essential data preprocessing steps, feature engineering, model training, and evaluation. The approach can be extended to other datasets for predictive analytics.



# Stock Time Series Classification

## Overview
This project focuses on classifying stocks based on their sectoral similarities using time-series analysis. The goal is to determine which sector a given stock belongs to by analyzing its historical price movements. The approach involves data collection, preprocessing, feature extraction, model training, and evaluation.

## Dataset
- **Data Source:**
  - Stock price data is retrieved using APIs such as `yfinance`, `investpy`, and `quandl`.
  - Sector and industry information is obtained via web scraping.
  
- **Time Range:**
  - Monthly stock returns starting from **January 1, 2005**.
  
- **Features:**
  - Historical stock prices
  - Sector and industry classification
  - Derived financial indicators (momentum, volatility, RSI, MACD, etc.)

## Environment Setup
- Required dependencies:
  ```python
  !pip install yfinance investpy tsfresh scikit-learn xgboost catboost matplotlib seaborn
  ```
- The dataset is loaded from yfinance API and processed using Pandas.

## Data Preprocessing
1. **Handling Missing Values:**
   - Missing stock data is filled using forward/backward filling or mean/median imputation.
   - Stocks with excessive missing values are removed.

2. **Transformations:**
   - Log transformation and differencing are applied to stabilize time series data.
   - Categorical variables (sector labels) are converted using one-hot or label encoding.

3. **Scaling:**
   - Features are standardized using `StandardScaler` or normalized with `MinMaxScaler`.

## Feature Extraction and Selection
- **Automated Feature Extraction:**
  - `tsfresh` is used to extract statistical features from time series data.

- **Feature Selection Methods:**
  - Lasso (L1 Regularization)
  - Recursive Feature Elimination (RFE)
  - Mutual Information Selection

## Model Training
- The dataset is split into **80% training** and **20% testing**.
- The following classifiers are trained and compared:
  - **Random Forest Classifier**
  - **Gradient Boosting Classifier**
  - **XGBoost Classifier**
  - **CatBoost Classifier**

### Example: Training an XGBoost Model
```python
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

model = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5)
model.fit(X_train, y_train)
```

## Model Evaluation
- The models are evaluated using classification metrics:
  ```python
  from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
  
  y_pred = model.predict(X_test)
  acc = accuracy_score(y_test, y_pred)
  f1 = f1_score(y_test, y_pred, average='weighted')
  roc_auc = roc_auc_score(y_test, model.predict_proba(X_test), multi_class='ovr')
  print(f'Accuracy: {acc}, F1 Score: {f1}, ROC AUC: {roc_auc}')
  ```

## Sector Similarity Analysis
- Stocks are compared to determine sectoral similarity based on:
  - Clustering techniques (e.g., K-Means, Hierarchical Clustering)
  - Cosine similarity between feature vectors

## Advanced Analysis (Bonus)
- Factor analysis is performed using financial indicators:
  - Momentum, Volatility, RSI, MACD
  - Time-series decomposition techniques

## Results and Visualization
- Sectoral similarity matrices are generated.
- Classification performance metrics are visualized using Seaborn and Matplotlib.
- The final model provides predictions for new stock data.

## Conclusion
This project demonstrates an end-to-end stock classification system using machine learning and time-series analysis. The model helps in identifying sectoral patterns and can be applied to portfolio management and investment strategies.

