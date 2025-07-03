import pandas as pd
from src.eda import plot_grade_distribution, plot_correlation_heatmap, plot_boxplots, class_balance
from src.preprocessing import preprocess_data
from src.feature_selection import feature_importance_plot
from sklearn.model_selection import train_test_split
from src.models import train_logistic_regression, train_random_forest, evaluate_model
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler

data_path = "/Users/hitarthwadhwani/Desktop/ML projects /data/student_performance_large_dataset.xlsx"
df = pd.read_excel(data_path, sheet_name="student_performance_large_datas")

# STEP 2: Run EDA
# plot_grade_distribution(df)
# plot_correlation_heatmap(df)
# plot_boxplots(df)
# class_balance(df)

clean_df, label_encoders, target_encoder = preprocess_data(df)

# print(clean_df.head())

# STEP 4: Feature Selection
importances = feature_importance_plot(clean_df)
# print("\nTop Features:")
# print(importances.sort_values(ascending=False))

# STEP 5: Feature Selection - Keep only the top features
features_to_keep = [
    "Attendance_Rate (%)",
    "Assignment_Completion_Rate (%)",
    "Study_Hours_per_Week",
    "Time_Spent_on_Social_Media (hours/week)",
    "Online_Courses_Completed",
    "Age",
    "Sleep_Hours_per_Night",
    "Effort_Score",
    "Distraction_Index",
    "Wellbeing_Score"
]

# Define target column
target_column = "Final_Grade"

# Separate features and target
X = clean_df[features_to_keep]
y = clean_df[target_column]

# Train-test split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# print(f"Train shape: {X_train.shape}")
# print(f"Test shape: {X_test.shape}")
# print("Data succesfully splitted")

#Step5.5: Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# STEP 6: Train Models
rf_model = train_random_forest(X_train, y_train)
logreg_model = train_logistic_regression(X_train, y_train)

# Train XGBoost
def train_xgboost(X_train, y_train):
    model = XGBClassifier(
        use_label_encoder=False,
        eval_metric='mlogloss',
        learning_rate=0.1,
        max_depth=5,
        n_estimators=100,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model

xgb_model = train_xgboost(X_train, y_train)

# STEP 7: Evaluate Models (Step 6 and 7 are together from models.py)
evaluate_model(rf_model, X_test, y_test, model_name="Random Forest")
evaluate_model(logreg_model, X_test, y_test, model_name="Logistic Regression")

