import pandas as pd
import time
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

def evaluate_classical_models(X_train, y_train, X_test, y_test):
    """Trains and profiles baseline setups on full data matrices."""
    
    # --- Random Forest ---
    rf_start = time.time()
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)
    rf_time = time.time() - rf_start
    
    print("=== Random Forest Base Performance ===")
    print(f"Train + Inference Time: {rf_time:.4f} seconds")
    print(f"Accuracy:  {accuracy_score(y_test, rf_preds)*100:.2f}%")
    print(f"F1-Score:  {f1_score(y_test, rf_preds)*100:.2f}%\n")

    # --- XGBoost ---
    xgb_start = time.time()
    xgb = XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss')
    xgb.fit(X_train, y_train)
    xgb_preds = xgb.predict(X_test)
    xgb_time = time.time() - xgb_start
    
    print("=== XGBoost Base Performance ===")
    print(f"Train + Inference Time: {xgb_time:.4f} seconds")
    print(f"Accuracy:  {accuracy_score(y_test, xgb_preds)*100:.2f}%")
    print(f"F1-Score:  {f1_score(y_test, xgb_preds)*100:.2f}%\n")
    
    print("Detailed Classification Report (XGBoost):")
    print(classification_report(y_test, xgb_preds, target_names=["Normal Operation", "Imminent Failure"]))

if __name__ == "__main__":
    print("Classical training entrypoint initialized. Import functions to parse engineered data frames.")
