# Lab 9 - Airflow

## Run Dockerfile

From airflow folder or app folder:

```bash
docker-compose down --volumes --remove-orphans
docker compose build
docker compose up
```

Airflow runs at: http://0.0.0.0:8080 (🔑 admin:admin)
