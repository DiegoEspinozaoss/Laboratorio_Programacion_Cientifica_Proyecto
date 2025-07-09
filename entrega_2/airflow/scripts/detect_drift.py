import pandas as pd
import joblib
from pathlib import Path

def detect_drift():
    print("Verificando drift...")

    data_path = Path("/opt/airflow/data/transacciones.parquet")
    model_stats_path = Path("/opt/airflow/models/train_feature_stats.joblib")

    if not data_path.exists():
        raise FileNotFoundError(f"❌ Archivo de datos no encontrado: {data_path}")

    df_new = pd.read_parquet(data_path)
    df_new = df_new[df_new['items'] > 0]
    df_new['purchase_date'] = pd.to_datetime(df_new['purchase_date'])
    df_new['semana'] = df_new['purchase_date'].dt.to_period("W").dt.start_time

    if not model_stats_path.exists():
        print("No se encontraron stats previos. Se forzará el reentrenamiento.")
        return True

    old_stats = joblib.load(model_stats_path)
    new_stats = df_new[['items']].mean()

    delta = abs(new_stats - old_stats).max()
    print(f"🔎 Delta detectado: {delta}")

    if delta > 0.1:
        print("Drift detectado. Se requiere reentrenamiento.")
        return True
    else:
        print("No hay drift. Se omitirá reentrenamiento.")
        return False
