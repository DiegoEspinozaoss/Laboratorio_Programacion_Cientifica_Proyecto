# Lab 9 - Airflow - Conclusiones

## Reflexiones
### Tracking y experimentación con MLflow

La incorporación de MLflow permitió realizar un seguimiento ordenado y métodico de los experimentos de entrenamiento. Gracias al tracking de métricas y parámetros, fue posible comparar distintos ajustes del árbol de decisión y evaluar su rendimiento. Esta trazabilidad es fundamental para iterar modelos de forma sistemática y justificar decisiones en producción.

### Despliegue con FastAPI y Gradio

Combinar FastAPI en el backend con Gradio en el frontend facilitó el despliegue rápido de un sistema de recomendación accesible para usuarios finales. El desarrollo modular permitió separar claramente la lógica del modelo, la API y la interfaz gráfica.

### Automatización con Airflow

Airflow permitió orquestar las tareas del flujo de machine learning: verificar la disponibilidad, detectar *data drift* y entrenar el modelo de forma automática, dando robustez al pipeline y permitiendo mínima intervención manual.

### Oportunidades de mejora

Mejorar la detección de data drifting por tema de desbalance, implementar un sistema de evaluación post producción para evaluzar si las predicciones se traducen en compras, enviar notificaciones cuando se detecte drift o una caída en las métricas del modelo.

## Conclusión general
El enfoque MLOps aplicado permitió transaformar un procedimiento estático en un sistema de recomendación trazable y modular. MLflow, FastAPI, Gradio y Airflow pueden, de forma complementaria, asistir a la elaboración de un flujo completo y con una buena arquitectura, levantar un sistema de recomendación/predicción perdurable en el tiempo.