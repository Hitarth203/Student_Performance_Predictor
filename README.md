# Student Performance Predictor

A supervised machine learning project to predict students’ final grades using behavioral, academic, and demographic features. The goal is to identify early indicators of student success and explore model performance in a multi-class classification setting.

---

## Project Overview

This project explores whether a student's final academic performance can be predicted using features such as:

- Attendance Rate
- Study Hours
- Sleep Duration
- Online Course Completion
- Time Spent on Social Media
- Stress Levels
- Learning Style and Participation

To maintain integrity, we excluded `Exam Score (%)`, which was highly correlated with the final grade and would have introduced data leakage.

---

## Dataset Summary

- Format: Excel (`student_performance_large_dataset.xlsx`)
- Size: 10,000+ records
- Target variable: `Final_Grade` (multi-class: 0, 1, 2, 3)

---

## Exploratory Analysis Summary

- Exam Score was strongly correlated with Final Grade and therefore excluded from training.
- Other predictors (e.g., sleep, social media usage, participation) showed moderate or weak individual correlations.
- The target distribution was fairly balanced, allowing for multi-class classification.

---

## Pipeline Workflow

1. **Preprocessing**
   - Dropped identifiers like `Student_ID`
   - Encoded categorical features using `LabelEncoder`
   - Added derived features using domain logic

2. **Feature Engineering**
   - `Effort_Score`: Study Hours × Assignment Completion Rate
   - `Distraction_Index`: Time on Social Media / (Study Hours + ε)
   - `Wellbeing_Score`: Sleep × Attendance Rate

3. **Feature Selection**
   - Removed high-leakage features
   - Selected top predictors based on feature importance from a Random Forest model

4. **Model Training**
   - Random Forest Classifier
   - Logistic Regression
   - XGBoost Classifier (included after recommendation by ChatGPT for experimentation)

5. **Evaluation**
   - Accuracy, Precision, Recall, F1-score
   - Confusion matrix visualizations

---

## Model Performance

| Model               | Accuracy |
|--------------------|----------|
| Random Forest       | ~24%     |
| Logistic Regression | ~26%     |
| XGBoost             | ~25%     |

Despite various tuning and experimentation, none of the models significantly outperformed random chance. This is largely attributed to the lack of strong predictive features in the dataset (after removing exam scores for fairness).

---

## Key Takeaways

- Avoiding data leakage is essential for honest evaluation.
- Even with correct methodology, models can perform poorly if input features lack predictive signal.
- The project demonstrates a complete ML pipeline and rigorous analysis.
- ChatGPT was consulted during modeling and suggested the addition of XGBoost for further experimentation.

---

## Folder Structure

Student_Performance_Predictor/
│
├── data/
│ └── student_performance_large_dataset.xlsx
│
├── src/
│ ├── eda.py
│ ├── preprocessing.py
│ ├── feature_selection.py
│ └── models.py
│
├── main.py
├── requirements.txt
└── README.md

---

## Future Improvements

- Reframe the problem as binary classification (e.g., pass/fail)
- Integrate time-series or behavioral data
- Apply unsupervised clustering prior to classification
- Visualize insights through a dashboard (e.g., using Streamlit)

---

## Author

Developed by Hitarth Wadhwani (2025)  
Guided by ChatGPT for debugging, architecture, and modeling ideas
