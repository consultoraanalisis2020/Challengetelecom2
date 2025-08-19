# Telecom X — Parte 2: Predicción de Cancelación (Churn)

Pipeline de **Machine Learning** para predecir la **evasión de clientes** en Telecom X. Este proyecto continúa la Parte 1 (EDA/ETL) y asume que cuentas con un **CSV limpio** con las columnas relevantes y la variable objetivo `Churn`.

## 🚀 Objetivos
- Preparación de datos (limpieza, codificación One-Hot, normalización).
- Análisis de correlación e **inspección de variables clave**.
- Entrenamiento de **≥2 modelos** (Logistic Regression / KNN / RandomForest).
- Evaluación con **accuracy, precision, recall, F1, ROC-AUC y matriz de confusión**.
- Interpretabilidad (coeficientes / importancias).
- **Informe** automático con conclusiones y recomendaciones.

## 📁 Estructura
```
TelecomX-Churn-ML/
├─ notebooks/TelecomX_Churn_Modelado.ipynb
├─ src/prep.py
├─ src/models.py
├─ data/.gitkeep                 # coloca aquí tu CSV limpio
├─ figs/.gitkeep                 # se guardan gráficos
├─ reporte/informe_modelado.md   # se genera al ejecutar el notebook
├─ README.md
└─ requirements.txt
```

## 🧰 Requisitos
- Python 3.9+
- pandas, numpy, matplotlib
- scikit-learn
- imbalanced-learn (opcional, para SMOTE)

Instalación rápida:
```bash
pip install -r requirements.txt
```

## ▶️ Uso
1. Coloca tu **CSV limpio** en `data/`, por ejemplo `data/TelecomX_Data_clean.csv`. Debe incluir la columna `Churn` (0/1 o Yes/No).
2. Abre `notebooks/TelecomX_Churn_Modelado.ipynb` y ajusta la variable `CSV_PATH` si es necesario.
3. Ejecuta todas las celdas. Se generarán gráficos en `figs/` y `reporte/informe_modelado.md`.

## 📈 Notas de modelado
- **Regresión Logística / KNN** usan **normalización** (StandardScaler), por ser sensibles a escala.
- **RandomForest** no requiere normalización.
- Si hay desbalance en `Churn`, puedes activar **SMOTE** (opcional) desde una celda del notebook.

## 🔗 Datos de referencia
En la Parte 1, el JSON público está en GitHub (puedes convertirlo a CSV):  
`https://raw.githubusercontent.com/alura-cursos/challenge2-data-science-LATAM/main/TelecomX_Data.json`

> Si no tienes el CSV limpio, ejecuta la Parte 1 para generarlo y deja solo columnas relevantes.
