from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

#Random Forest 
def train_random_forest(X_train,y_train):
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train,y_train)
    return rf

#Logistic Regression
def train_logistic_regression(X_train, y_train):
    logreg = LogisticRegression(max_iter=1000,random_state=42)
    logreg.fit(X_train,y_train)
    return logreg

#Scores for both models 
def evaluate_model(model, X_test, y_test, model_name="Model"):
    y_pred = model.predict(X_test)
    print(f"\n📊 {model_name} Evaluation:")
    print("Accuracy:", round(accuracy_score(y_test, y_pred), 2))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
