import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib

# Load the dataset
df = pd.read_csv('accidents.csv')  # Ensure this file is in the same directory or provide the correct path

# Step 1: Data Preparation
# Specify dependent and independent variables based on the dataset
dependent_variable = 'Accident_severity'
independent_variables = ['Age_band_of_driver', 'Sex_of_driver', 'Driving_experience',
                         'Weather_conditions', 'Road_surface_type']

# Filter the dataset to include only selected columns
df_filtered = df[independent_variables + [dependent_variable]]

# Convert categorical variables to numerical using one-hot encoding
df_encoded = pd.get_dummies(df_filtered, columns=independent_variables, drop_first=True)

# Define X (independent variables) and y (dependent variable)
X = df_encoded.drop(columns=[dependent_variable])
y = df_encoded[dependent_variable]

# Step 2: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Model Training
model = LinearRegression()
model.fit(X_train, y_train)

# Display model details
print("Model Coefficients:", model.coef_)
print("Model Intercept:", model.intercept_)

# Step 4: Model Evaluation
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# Step 5: Save the model
model_filename = 'road_accident_severity_model.pkl'
joblib.dump(model, model_filename)
print(f"Model saved as '{model_filename}'")

# Example of creating a hypothetical data input with appropriate columns
hypothetical_data_dict = {
    'Age_band_of_driver_18-30': [0],
    'Age_band_of_driver_31-50': [1],
    'Age_band_of_driver_Over 51': [0],
    'Sex_of_driver_Male': [1],
    'Driving_experience_5-10yr': [0],
    'Driving_experience_Above 10yr': [1],
    'Driving_experience_Below 5yr': [0],
    'Weather_conditions_Raining': [1],
    'Weather_conditions_Fog': [0],
    'Weather_conditions_Clear': [0],
    'Road_surface_type_Tar': [1],
    'Road_surface_type_Gravel': [0],
    # Add other necessary columns with dummy values of 0
    # Make sure to include all encoded columns, assigning 0 where necessary
}

# Convert the dictionary to a DataFrame with the same columns as X
hypothetical_data_df = pd.DataFrame(hypothetical_data_dict)

# Fill missing columns with 0s to match the full set of 26 features
for col in X.columns:
    if col not in hypothetical_data_df.columns:
        hypothetical_data_df[col] = 0

# Reorder columns to match X exactly
hypothetical_data_df = hypothetical_data_df[X.columns]

# Predict with the hypothetical data
predicted_severity = model.predict(hypothetical_data_df)
print("Predicted Accident Severity for hypothetical data:", predicted_severity[0])

