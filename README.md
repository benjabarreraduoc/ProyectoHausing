## Predicción de precios de viviendas — Ames Iowa Housing

Informe técnico del proyecto de Machine Learning — Caso B 

## Descripción del problema de negocio
Una inmobiliaria/tasadora necesita estimar el precio de venta de viviendas en Ames, 
Iowa, de forma más objetiva y rápida que la tasación manual tradicional, reduciendo 
errores de valoración que afectan tanto a compradores como vendedores.

## Objetivos del proyecto
- Desarrollar un modelo de Machine Learning que prediga SalePrice a partir de las 
  características de la vivienda.
- Identificar las variables con mayor influencia en el precio.
- Entregar una herramienta que apoye decisiones de tasación.

## Descripción de las fuentes de datos utilizadas

El proyecto utiliza el **Ames Iowa Housing Dataset**, un conjunto de datos público que registra información detallada de ventas de propiedades residenciales en la ciudad de Ames, Iowa, Estados Unidos, entre los años 2006 y 2010.

**Características generales de la fuente:**

| Atributo | Detalle |
| Archivo | `Ames_Iowa_Housing_Dataset.csv` |
| N° de registros | 2930 |
| N° de variables | 82 |
| Variable objetivo | `SalePrice` (precio de venta, en USD) |
| Tipo de variables | Numéricas (continuas y discretas) y categóricas (nominales y ordinales) |
| Nivel de granularidad | Una fila = una propiedad vendida |

**Contenido general de las variables:**
El dataset cubre distintas dimensiones de una propiedad, entre ellas:
- **Identificación:** `PID` (identificador único de la propiedad, sin datos personales del propietario)
- **Ubicación:** `Neighborhood`, zonificación (`MS Zoning`)
- **Características físicas:** superficie habitable, superficie de sótano, garaje, cantidad de habitaciones y baños, año de construcción y remodelación
- **Calidad y condición:** evaluaciones cualitativas de calidad general, calidad de cocina, condición exterior, etc.
- **Variable objetivo:** `SalePrice`

## KPIs que resolverán el problema de negocio
- Error absoluto medio (MAE) del modelo en USD.
- % de predicciones dentro de un margen de error aceptable (ej. ±10% del precio real).
- Reducción del tiempo de tasación respecto al proceso manual.

## Evaluación preliminar de ética, sesgos y privacidad

- **Privacidad:** el dataset no contiene nombres de propietarios ni direcciones 
  exactas, pero el PID podría, en teoría, cruzarse con registros públicos del 
  condado para reidentificar la propiedad. Se recomienda no exponerlo en 
  entregables públicos.
- **Sesgo potencial:** la variable `Neighborhood` puede actuar como proxy de 
  nivel socioeconómico. Un modelo entrenado sobre estos datos podría replicar 
  desigualdades históricas de valoración entre barrios en vez de basarse 
  puramente en características objetivas de la vivienda.
- **Riesgo de uso:** si este modelo se usara para decisiones de crédito 
  hipotecario o tasación automática sin supervisión humana, un sesgo por barrio 
  podría traducirse en discriminación indirecta.

## Preparación y análisis exploratorio de los datos (EDA)

Pendiente de desarrollar
> - Estadística descriptiva de la variable objetivo (`SalePrice`) y variables clave
> - Identificación y tratamiento de valores faltantes
> - Identificación de outliers y anomalías
> - Análisis de correlación entre variables numéricas y `SalePrice`
> - Análisis de variables categóricas relevantes (`Neighborhood`, `Overall Qual`, etc.)
> - Visualizaciones (histogramas, boxplots, heatmap de correlación)
> - Transformaciones aplicadas: imputación, encoding, tratamiento de outliers, ingeniería de variables
> - Conclusiones sobre calidad final del dataset preparado para modelamiento

## Metodología utilizada (CRISP-DM)

El proyecto se desarrolla siguiendo la metodología **CRISP-DM (Cross-Industry Standard Process for Data Mining)**, que estructura el trabajo en las siguientes fases:

1. **Comprensión del negocio (Business Understanding)**
   Definición del problema de negocio, objetivos del proyecto y KPIs que permitirán evaluar si la solución de Machine Learning resuelve la necesidad planteada (predicción del precio de viviendas).

2. **Comprensión de los datos (Data Understanding)**
   Identificación de la fuente de datos, revisión de su estructura (2930 registros, 82 variables), elaboración de un diccionario de variables y evaluación preliminar de calidad, sesgos y privacidad.

3. **Preparación de los datos (Data Preparation)**
   Limpieza de valores faltantes, tratamiento de outliers, codificación de variables categóricas e ingeniería de variables, a partir de los hallazgos del análisis exploratorio.

4. **Modelamiento (Modeling)**
   Modelamiento de datos según corresponda para maximizar la compresión de los precios establecidos (histogramas,boxplot,heatmap,etc)

