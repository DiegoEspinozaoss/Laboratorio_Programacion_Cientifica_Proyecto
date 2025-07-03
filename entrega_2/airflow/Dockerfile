FROM python:3.10-slim

ENV AIRFLOW_HOME=/opt/airflow
ENV AIRFLOW__CORE__SQL_ALCHEMY_CONN=sqlite:////opt/airflow/airflow.db
ENV PYTHONPATH=${AIRFLOW_HOME}

WORKDIR $AIRFLOW_HOME

RUN mkdir -p dags scripts data models plugins logs \
    && touch $AIRFLOW_HOME/airflow.db

RUN apt-get update && apt-get install -y \
    curl \
    libgomp1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ./dags $AIRFLOW_HOME/dags
COPY ./scripts $AIRFLOW_HOME/scripts
COPY ./data $AIRFLOW_HOME/data
COPY ./plugins $AIRFLOW_HOME/plugins

EXPOSE 8080

CMD ["sh", "-c", "\
  airflow db init && \
  airflow users create \
    --username admin \
    --firstname admin \
    --lastname admin \
    --role Admin \
    --email admin@example.com \
    --password admin && \
  airflow webserver -p 8080 & \
  airflow scheduler"]
