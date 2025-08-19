# Alura Store Latam — Análisis de Ventas (Proyecto)

Este proyecto ayuda al Sr. Juan a decidir **qué tienda vender** en la cadena Alura Store a partir de datos de 4 tiendas. Calculamos indicadores clave, generamos visualizaciones y emitimos una **recomendación final basada en datos**.

## Objetivos
- Cargar y manipular CSV con **pandas**.
- Crear **gráficos con matplotlib** (≥3 tipos).
- Analizar **ingresos**, **categorías vendidas**, **reseñas**, **productos top/bottom** y **costo de envío**.
- Recomendar **la tienda menos eficiente** para vender.

## Estructura
```
AluraStore-Proyecto/
├─ notebooks/
│  └─ AluraStore_Analisis.ipynb
├─ src/
│  └─ metrics.py
├─ data/
│  └─ .gitkeep
├─ figs/
│  └─ .gitkeep
├─ reporte/
│  └─ informe_final.md
├─ README.md
└─ requirements.txt
```

## Requisitos
Python 3.9+ · pandas · matplotlib · numpy · jupyter/colab

Instalación:
```bash
pip install -r requirements.txt
```

## Datos
El notebook descarga los CSV desde:
`https://raw.githubusercontent.com/alura-es-cursos/challenge1-data-science-latam/refs/heads/main/base-de-datos-challenge1-latam/`

Archivos:
- `tienda_1%20.csv` (nota: espacio antes del .csv)
- `tienda_2.csv`
- `tienda_3.csv`
- `tienda_4.csv`

(Alternativa) Colócalos en `data/` y usa `USE_LOCAL=True`.

## Cómo ejecutar
1. Abre `notebooks/AluraStore_Analisis.ipynb` en Jupyter o Colab.
2. Ejecuta todas las celdas.
3. Se generan:
   - KPIs por tienda.
   - Gráficos en `figs/`.
   - Informe `reporte/informe_final.md` con recomendación.

## Metodología
Score de eficiencia por tienda:
- Ingresos (50%)
- Calificación promedio (30%)
- Costo de envío promedio (20%, menor es mejor)

La **menor eficiencia** = candidata para vender.

## Notas
- Se respeta el código base de carga; se agregan análisis, visualizaciones y reporte.
- `src/metrics.py` normaliza columnas (acentos/sinónimos).
