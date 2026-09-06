# Predicción de precios de viviendas — Ames Iowa Housing

Proyecto de Machine Learning — MLY1101, Evaluación Parcial N°1, Caso B.
Modelo de regresión para predecir `SalePrice` a partir de las características de
viviendas en Ames, Iowa, siguiendo la metodología CRISP-DM.

> **¿Buscas el informe técnico?** Este archivo (`README.md`) explica cómo instalar,
> ejecutar y reproducir el proyecto. El análisis completo (problema de negocio,
> objetivos, KPIs, EDA, metodología) está en [`informe.md`](./informe.md).

---

## Estructura del repositorio

```
├── data/
│   ├── Ames_Iowa_Housing_Dataset.csv              # dataset original (2930 x 82)
│   ├── Ames_Housing_Parte3_legible.csv            # dataset limpio, sin one-hot (2929 x 86)
│   └── Ames_Housing_Parte3_listo_para_modelar.csv # dataset final, codificado (2929 x 220)
├── notebooks/
│   └── notebook_ames_housing.ipynb                # notebook único, ejecutable de principio a fin
├── models/                                        # modelos entrenados (fase de Modelamiento)
├── images/                                        # gráficos exportados del EDA
├── informe.md                                     # informe técnico (6 secciones exigidas)
└── README.md                                      # este archivo
```

> Ajusta los nombres si tu repositorio final usa otra convención — lo importante es
> que las rutas relativas dentro del notebook (`../data/...`) coincidan con esta estructura.

---

## Requisitos

- Python 3.10 o superior
- Jupyter Notebook, JupyterLab, VS Code, o Google Colab

### Librerías utilizadas

```
pandas
numpy
matplotlib
```

> Cuando el equipo llegue a la fase de Modelamiento probablemente necesiten agregar
> `scikit-learn` — se deja fuera por ahora porque el notebook, tal como está, no la usa.

Puedes instalarlas todas con:

```bash
pip install pandas numpy matplotlib
```

O crear un `requirements.txt` con esas mismas líneas y ejecutar:

```bash
pip install -r requirements.txt
```

---

## Cómo ejecutar el proyecto (reproducibilidad)

### Opción A — Localmente (Jupyter / VS Code)

1. Clona o descarga este repositorio, manteniendo la estructura de carpetas de arriba.
2. Verifica que `Ames_Iowa_Housing_Dataset.csv` esté dentro de `data/`.
3. Instala las librerías (ver sección anterior).
4. Abre `notebooks/notebook_ames_housing.ipynb`.
5. Ejecuta todas las celdas en orden, de principio a fin (`Run All` / "Ejecutar todo").
   El notebook está dividido en las 3 partes del proyecto, en este orden:
   - **Parte 1** — carga de datos, diccionario de variables, evaluación ética preliminar
   - **Parte 2** — análisis exploratorio (EDA): estadística descriptiva, nulos, outliers, correlaciones
   - **Parte 3** — preparación de datos: corrección de errores, imputación, encoding, feature engineering
6. Al finalizar, el notebook genera automáticamente los dos archivos de salida en `data/`:
   `Ames_Housing_Parte3_legible.csv` y `Ames_Housing_Parte3_listo_para_modelar.csv`.

### Opción B — Google Colab

1. Sube la carpeta completa a Google Drive (o clona el repositorio de GitHub directamente en Colab).
2. Abre `notebook_ames_housing.ipynb` con Google Colab.
3. Si usas Drive, monta tu unidad al inicio del notebook:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
4. Ajusta las rutas de lectura del CSV (`../data/...`) según dónde quede montada la carpeta.
5. Ejecuta todas las celdas en orden.

---

## Datos

- **Fuente:** Ames Iowa Housing Dataset (público), 2930 registros × 82 variables originalmente.
- **Ubicación esperada:** `data/Ames_Iowa_Housing_Dataset.csv`
- **Variable objetivo:** `SalePrice`
- **Diccionario de variables:** además del diccionario propio del equipo, se usó como referencia
  oficial De Cock, D. (2011), *"Ames, Iowa: Alternative to the Boston Housing Data..."*,
  Journal of Statistics Education, 19(3) — https://jse.amstat.org/v19n3/decock/DataDocumentation.txt

## Resultados de la preparación de datos

Tras ejecutar el notebook completo, el dataset queda con **2929 filas** (se eliminó 1 registro
de venta parcial no representativa, justificado en `informe.md`) y **0 valores nulos**. Se generan
dos versiones:

| Archivo | Filas × Columnas | Uso |
|---|---|---|
| `Ames_Housing_Parte3_legible.csv` | 2929 × 86 | Inspección, documentación, EDA posterior |
| `Ames_Housing_Parte3_listo_para_modelar.csv` | 2929 × 220 | Entrenamiento de modelos (fase de Modelamiento) |

## Metodología

El proyecto sigue **CRISP-DM**. El detalle completo de cada fase, las decisiones tomadas y su
justificación están documentados en [`informe.md`](./informe.md).

## Equipo

Trabajo grupal de 3 integrantes, con defensa individual y preguntas cruzadas. División de
responsabilidades documentada en `division_proyecto_MLY1101.md` (IE1+IE4 / IE3 / IE2).
