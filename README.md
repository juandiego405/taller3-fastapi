# Taller 3 — Pipeline de ML con FastAPI

API educativa de regresión para predecir `charges` a partir de
age, sex, bmi, children, smoker y region.

## Requisitos

- Python 3.12
- Git
- Dependencias fijadas en requirements.txt

## Preparar el entorno local

Clone este repositorio y entre en su carpeta. Después ejecute:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Entrenar y guardar el pipeline

```bash
python train.py
```

El entrenamiento separa los conjuntos de entrenamiento y prueba.
La imputación y la codificación se ajustan únicamente con entrenamiento.

Se generan:

- artifacts/pipeline.joblib: preprocesamiento y modelo entrenados.
- artifacts/training_metadata.json: hiperparámetros, versiones y métricas.

El pipeline se guarda y vuelve a cargar para comprobar que conserva
las predicciones. Las métricas de evaluación son MAE, RMSE y R².

## Ejecutar la API local

```bash
python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

Documentación interactiva: http://localhost:8000/docs

Comprobación de disponibilidad:

```bash
curl http://localhost:8000/health
```

Predicción, desde otra terminal situada en la carpeta del proyecto:

```bash
curl -i http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  --data-binary @examples/predict.json
```

La respuesta contiene una lista `predictions`.
Los datos inválidos generan una respuesta HTTP 422.

## Despliegue en EC2

Se incluye deploy/taller3-api.service, preparado para:

- Proyecto instalado en /opt/taller3-fastapi.
- Entorno virtual recreado con Python 3.12 en esa carpeta.
- Usuario de servicio `mlops`, con permisos de lectura del proyecto.
- Inicio automático al habilitar el servicio en systemd.
- Reinicio automático ante fallos.
- Uvicorn escuchando en 127.0.0.1:8000.
- Acceso externo mediante Nginx con HTTPS y autenticación.

El usuario, las rutas y el entorno deben existir antes de activar
el servicio. La configuración de Nginx y el certificado se realiza
en la instancia de destino.

## Referencias

Ejemplos y datos utilizados en clase:

- https://github.com/smm-0216/MLOps-FastAPI
- https://github.com/smm-0216/MLOps-MLFlow

Se utiliza la estructura de Pipeline y ColumnTransformer trabajada
en el Taller 1.
