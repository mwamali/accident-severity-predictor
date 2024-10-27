import joblib
import pandas as pd

# Load the trained model
model = joblib.load('road_accident_severity_model.pkl')

# Create hypothetical data for prediction (ensure it matches the model’s expected format)
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
    # Ensure you include all other columns needed by the model with dummy values
}

# Convert dictionary to DataFrame and fill missing columns
hypothetical_data_df = pd.DataFrame(hypothetical_data_dict)
for col in model.feature_names_in_:  # model.feature_names_in_ contains the expected column names
    if col not in hypothetical_data_df.columns:
        hypothetical_data_df[col] = 0

# Ensure the columns are in the right order
hypothetical_data_df = hypothetical_data_df[model.feature_names_in_]

# Predict the accident severity
predicted_severity = model.predict(hypothetical_data_df)
print("Predicted Accident Severity:", predicted_severity[0])
