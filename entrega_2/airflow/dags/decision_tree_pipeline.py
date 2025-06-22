from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import subprocess
from pathlib import Path

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=2),
}

dag = DAG(
    'decision_tree_pipeline',
    default_args=default_args,
    description='Pipeline completo con DecisionTree, MLflow y verificación de drift',
    schedule_interval='@daily',
    start_date=datetime(2024, 1, 1),
    catchup=False,
)

def check_data():
    data_dir = Path("/opt/airflow/data")
    required_files = ["transacciones.parquet", "clientes.parquet", "productos.parquet"]
    for file in required_files:
        file_path = data_dir / file
        if not file_path.exists():
            raise FileNotFoundError(f"❌ Falta el archivo requerido: {file_path}")
        else:
            print(f"✅ Archivo encontrado: {file_path}")

def run_training_script():
    try:
        print("🚀 Ejecutando script de entrenamiento...")
        subprocess.run(["python3", "/opt/airflow/scripts/train_decision_tree.py"], check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError("❌ Error al ejecutar el script de entrenamiento") from e

def detect_drift():
    print("🔍 Drift detection pendiente de implementación...")
    # Aquí se puede agregar validación con Evidently, estadísticas delta, etc.

check_data_task = PythonOperator(
    task_id='check_data',
    python_callable=check_data,
    dag=dag,
)

train_model_task = PythonOperator(
    task_id='train_model',
    python_callable=run_training_script,
    dag=dag,
)

drift_detection_task = PythonOperator(
    task_id='drift_detection',
    python_callable=detect_drift,
    dag=dag,
)

# Flujo de tareas
check_data_task >> train_model_task >> drift_detection_task
