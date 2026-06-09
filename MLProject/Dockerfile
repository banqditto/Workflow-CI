FROM python:3.12-slim

WORKDIR /app

# Install base dependencies langsung di level OS kontainer
RUN pip install --no-cache-dir mlflow==2.19.0 scikit-learn pandas numpy scipy cloudpickle

# Salin folder model yang sudah dibuat oleh skrip Python kamu
COPY target_model /app/target_model

# Ekspos port default MLflow server
EXPOSE 5001

# Perintah untuk menjalankan serving model saat kontainer aktif
CMD ["mlflow", "models", "serve", "-m", "/app/target_model", "-h", "0.0.0.0", "-p", "5001", "--no-conda"]