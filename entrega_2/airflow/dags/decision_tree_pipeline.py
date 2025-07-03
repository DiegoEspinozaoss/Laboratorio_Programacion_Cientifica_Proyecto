from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator
from datetime import datetime, timedelta
import subprocess
from pathlib import Path
from scripts.detect_drift import detect_drift
from airflow.utils.trigger_rule import TriggerRule

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

# DEFS

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

def decide_retraining(**kwargs):
    if detect_drift():
        return 'train_model'
    else:
        return 'skip_training'

def run_batch_prediction():
    subprocess.run(["python3", "/opt/airflow/scripts/predict_batch.py"], check=True)

# OPS

check_data_task = PythonOperator(
    task_id='check_data',
    python_callable=check_data,
    dag=dag,
)

branch_task = BranchPythonOperator(
    task_id='branch_drift_check',
    python_callable=decide_retraining,
    provide_context=True,
    dag=dag,
)

train_model_task = PythonOperator(
    task_id='train_model',
    python_callable=run_training_script,
    dag=dag,
)

drift_detection_task = PythonOperator(
    task_id='drift_detection',
    python_callable=lambda: print("📊 Registro de que drift fue evaluado"),
    dag=dag,
)

predict_batch_task = PythonOperator(
    task_id='predict_batch',
    python_callable=run_batch_prediction,
    dag=dag,
    trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS,
)

skip_training = EmptyOperator(task_id='skip_training', dag=dag)
end = EmptyOperator(task_id='end', dag=dag)

# FLOW

check_data_task >> branch_task
branch_task >> train_model_task >> drift_detection_task >> predict_batch_task >> end
branch_task >> skip_training >> predict_batch_task >> end
