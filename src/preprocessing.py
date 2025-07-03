import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess_data(df):
    df = df.copy()  # Avoid modifying the original Df

# Dropping "Student_ID" Column because its a identifier 
    if 'Student_ID' in df.columns:
        df.drop(columns=['Student_ID'], inplace=True)

# Defining categorical columns for encoding 
    categorical_cols = [
        "Gender",
        "Preferred_Learning_Style",
        "Participation_in_Discussions",
        "Use_of_Educational_Tech",
        "Self_Reported_Stress_Level"
    ]

#LabelEncoder
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le 

#Encoding the target column
    target_encoder = LabelEncoder()
    df["Final_Grade"] = target_encoder.fit_transform(df["Final_Grade"])

# Feature Engineering 
    df["Effort_Score"] = df["Study_Hours_per_Week"] * df["Assignment_Completion_Rate (%)"]
    df["Distraction_Index"] = df["Time_Spent_on_Social_Media (hours/week)"] / (df["Study_Hours_per_Week"] + 1e-5)
    df["Wellbeing_Score"] = df["Sleep_Hours_per_Night"] * df["Attendance_Rate (%)"]

#Returning cleaned DataFrame and encoders
    return df, label_encoders, target_encoder







    