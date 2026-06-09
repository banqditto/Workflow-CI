import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import mlflow
import shutil
from mlflow.models import infer_signature

def setup_mlflow():
    REPO_OWNER = "banqditto"
    REPO_NAME = "Eksperimen_SML_Anggi-permana"
    
    # Cek apakah skrip sedang dijalankan oleh Runner GitHub Actions
    if os.environ.get("GITHUB_ACTIONS") == "true":
        print("Mendeteksi GitHub Actions. Menggunakan environment tracking bawaan CLI...")
    else:
        print("Mendeteksi lingkungan Lokal laptop...")
        remote_url = f"https://dagshub.com/{REPO_OWNER}/{REPO_NAME}.mlflow"
        mlflow.set_tracking_uri(remote_url)
        mlflow.set_experiment("Latihan Diabetes")

def train_and_log():
    # 1. Memuat data bersih hasil Preprocessing
    train_df = pd.read_csv("diabetes_preprocessing/train_clean.csv")
    test_df = pd.read_csv("diabetes_preprocessing/test_clean.csv")
    
    X_train = train_df.drop(columns=['Outcome'])
    y_train = train_df['Outcome']
    X_test = test_df.drop(columns=['Outcome'])
    y_test = test_df['Outcome']
    
    # 2. Proses Hyperparameter Tuning via GridSearchCV
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [3, 5]
    }
    
    grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3)
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    
    print("Mencatat metrik dan parameter ke MLflow...")
    
    # Log Hyperparameters
    for param_name, param_val in best_params.items():
        mlflow.log_param(param_name, param_val)
    mlflow.log_param("model_type", "RandomForestClassifier_Tuned")
    
    # Evaluasi model pada data uji
    y_pred = best_model.predict(X_test)
    
    # Log Metrik Evaluasi
    mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
    mlflow.log_metric("precision", precision_score(y_test, y_pred))
    mlflow.log_metric("recall", recall_score(y_test, y_pred))
    mlflow.log_metric("f1_score", f1_score(y_test, y_pred))
    
    # Membuat signature skema data
    signature = infer_signature(X_train, best_model.predict(X_train))
    
    # Simpan model ke dalam folder bernama 'model'
    mlflow.sklearn.log_model(
        sk_model=best_model, 
        artifact_path="model", 
        signature=signature,
        input_example=X_train.head(1)
    )
    
    # --- Membuat & Mengunggah 2 Artefak Tambahan ---
    # Artefak 1: Plot Gambar Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax, cmap='Blues')
    plt.title("Confusion Matrix (Tuned)")
    
    cm_path = "training_confusion_matrix.png"
    plt.savefig(cm_path)
    plt.close()
    mlflow.log_artifact(cm_path)
    
    # Artefak 2: Ringkasan Dataset dalam bentuk file JSON
    dataset_info = {
        "train_samples": int(X_train.shape[0]),
        "test_samples": int(X_test.shape[0]),
        "features_list": list(X_train.columns)
    }
    json_path = "metric_info.json"
    with open(json_path, "w") as f:
        json.dump(dataset_info, f, indent=4)
    mlflow.log_artifact(json_path)
    
    # Bersihkan file lokal setelah sukses terunggah
    if os.path.exists(cm_path): os.remove(cm_path)
    if os.path.exists(json_path): os.remove(json_path)
    
    # Salin model terbaik ke folder statis untuk target inference.py
    if os.path.exists("target_model"):
        shutil.rmtree("target_model")
    mlflow.sklearn.save_model(best_model, "target_model")
    print("Proses otomatisasi retraining selesai dengan sukses!")

def main():
    setup_mlflow()
    
    # Jika berjalan di GitHub Actions, langsung eksekusi tanpa membungkus start_run manual
    if os.environ.get("GITHUB_ACTIONS") == "true":
        train_and_log()
    else:
        # Jika di lokal komputer, gunakan pembungkus start_run biasa
        with mlflow.start_run(run_name="retraining_local"):
            train_and_log()

if __name__ == "__main__":
    main()
