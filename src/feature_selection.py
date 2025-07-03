import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.ensemble import RandomForestClassifier

def feature_importance_plot(df, target_column="Final_Grade"):
    df = df.copy()

    # 1. Separating features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # 2. Fiting Random Forest
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    # 3. Getting feature importances
    importances = pd.Series(model.feature_importances_, index=X.columns)
    importances = importances.sort_values(ascending=True)

    # 4. Plotting the importances
    plt.figure(figsize=(10, 6))
    importances.plot(kind="barh", color="skyblue")
    plt.title("Feature Importance (Random Forest)")
    plt.xlabel("Importance Score")
    plt.tight_layout()
    plt.show()

    return importances