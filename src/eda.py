import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns 

df = pd.read_excel("/Users/hitarthwadhwani/Desktop/ML projects /data/student_performance_large_dataset.xlsx")

#Previewing the data
# print(df.head())

#Making a Countplot for Distribution of Final Grades
def plot_grade_distribution(df):
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x="Final_Grade", order=sorted(df["Final_Grade"].unique()))
    plt.title("Distribution of Final Grades")
    plt.xlabel("Final Grade")
    plt.ylabel("Number of Students")
    plt.show()

def plot_correlation_heatmap(df):
    numeric_cols = df.select_dtypes(include="number")
    plt.figure(figsize=(10, 8))
    sns.heatmap(numeric_cols.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.show()

#Boxplot of Feature vs Final Grade

def plot_boxplots(df):
    numeric_features = [
        "Study_Hours_per_Week",
        "Exam_Score (%)",
        "Attendance_Rate (%)",
        "Assignment_Completion_Rate (%)",
        "Time_Spent_on_Social_Media (hours/week)",
        "Sleep_Hours_per_Night"
    ]

    for col in numeric_features:
        plt.figure(figsize=(8, 5))
        sns.boxplot(data=df, x="Final_Grade", y=col)
        plt.title(f"{col} vs Final Grade")
        plt.xlabel("Final Grade")
        plt.ylabel(col)
        plt.tight_layout()
        plt.show()

#Checking class balance
def class_balance(df):
    print(df['Final_Grade'].value_counts())
    print(df['Final_Grade'].value_counts(normalize=True) * 100)



