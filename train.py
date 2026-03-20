import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score, classification_report, hamming_loss

def train_cuisine_model(file_path="dataset.csv"):
    print("🚀 Starting Model Training Pipeline...")
    
    # 1. Load Dataset
    try:
        df = pd.read_csv(file_path, encoding="utf-8-sig")
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        return

    # 2. Preprocessing Cuisines (Multi-label)
    print("📦 Preprocessing data...")
    df["Cuisines"] = df["Cuisines"].fillna("").apply(
        lambda x: [c.strip().lower() for c in x.split(",") if c.strip() != ""]
    )

    # Encoding Target Variable
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(df["Cuisines"])
    
    # Selecting Features
    X = df[["Average Cost for two", "Price range", "Aggregate rating", "Votes"]]

    # 3. Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4. Model Building (Random Forest with OneVsRest for Multi-label)
    print("🤖 Training RandomForest Classifier (This may take a moment)...")
    rf = RandomForestClassifier(n_estimators=150, max_depth=20, random_state=42, n_jobs=-1)
    model = OneVsRestClassifier(rf)
    model.fit(X_train, y_train)

    # 5. Evaluation
    print("\n✅ Training Complete! Evaluating Performance...")
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    h_loss = hamming_loss(y_test, y_pred)
    
    print(f"--- Metrics ---")
    print(f"Accuracy Score: {acc:.4f}")
    print(f"Hamming Loss (Lower is better): {h_loss:.4f}")
    print("----------------\n")

    # 6. Save Assets
    print("💾 Saving model and encoder to disk...")
    with open('cuisine_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('mlb.pkl', 'wb') as f:
        pickle.dump(mlb, f)
    
    print("✨ Process finished successfully! Files created: 'cuisine_model.pkl', 'mlb.pkl'")

if __name__ == "__main__":
    train_cuisine_model()