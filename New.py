# Step 1: Import Required Libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedShuffleSplit 
from sklearn.metrics import accuracy_score
import joblib  
import sqlite3 #For saving and loading models
# Step 2: Load and Prepare the Data
# Load dataset

df = pd.read_csv('./clean_EGT209_T4_Group3_raw_data_wk3.csv')
df = df.dropna()
# Select relevant features

features = ["temperature", "humidity", "air_quality"]


# Display the first few rows of the dataset
print("Data Sample:\n", df.head())

# Define transformers for preprocessing
num_transformer = StandardScaler()  # Standardize numerical features

# Combine transformers into a single preprocessor
preprocessor = ColumnTransformer([
    ('num', num_transformer, features)
])

# Step 4: Split Data into Training and Testing Sets
# Define target and features
p = df.temperature+ df.humidity


# Split into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(p, df.air_quality , test_size=0.2, random_state=42)
X_2train, X_2test, y_2train, y_2test = train_test_split(df.temperature + df.air_quality, df.humidity , test_size=0.2, random_state=42)
X_temperature_train, X_temperature_test, y_temperature_train, y_temperature_test = train_test_split(df.air_quality+ df.humidity , df.temperature , test_size=0.2, random_state=42)







# Step 5: Build the Machine Learning Pipeline
# Define the pipeline (includes preprocessing + RandomForest classifier)
pipeline = Pipeline([
    ('preprocessor', preprocessor),  # Apply preprocessing steps
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))  # ML model (RandomForest)
])
pipeline2 = Pipeline([
    ('preprocessor', preprocessor),  # Apply preprocessing steps
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))  # ML model (RandomForest)
])
pipeline_2 = Pipeline([
    ('preprocessor', preprocessor),  # Apply preprocessing steps
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))  # ML model (RandomForest)
])

# Step 6: Train the Model
# Train the model using the pipeline
pipeline.fit(X_train, y_train)
pipeline2.fit(X_2train, y_2train)
pipeline_2.fit(X_temperature_train, y_temperature_train)
print("Model training complete!")

# Step 7: Evaluate the Model
# Make predictions on the test data
y_pred = pipeline.predict(X_test)
y_pred_2 = pipeline.predict(X_2test)
y_pred2 = pipeline.predict(X_temperature_test)
# Compute accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Random Forest Model Accuracy: {accuracy:.2f}")

# Step 8: Save and Load the Model
# Save the trained pipeline (preprocessing + model)
joblib.dump(pipeline, 'ml_pipeline.pkl')

# Load the model back
loaded_pipeline = joblib.load('ml_pipeline.pkl')

# Predict using the loaded model
#sample_data = 
prediction = loaded_pipeline.predict(sample_data)

# Output prediction for a sample input