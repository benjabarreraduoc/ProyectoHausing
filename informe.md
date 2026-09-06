## Predicción de precios de viviendas — Ames Iowa Housing

Informe técnico del proyecto de Machine Learning — Caso B (MLY1101, Evaluación Parcial N°1)

---

## Descripción del problema de negocio

Una inmobiliaria/tasadora necesita estimar el precio de venta de viviendas en Ames,
Iowa, de forma más objetiva y rápida que la tasación manual tradicional, reduciendo
errores de valoración que afectan tanto a compradores como vendedores.

## Objetivos del proyecto

- Desarrollar un modelo de Machine Learning que prediga `SalePrice` a partir de las
  características de la vivienda.
- Identificar las variables con mayor influencia en el precio.
- Entregar una herramienta que apoye decisiones de tasación.

## Definición de KPIs que resolverán el problema de negocio

- Error absoluto medio (MAE) del modelo en USD.
- % de predicciones dentro de un margen de error aceptable (ej. ±10% del precio real).
- Reducción del tiempo de tasación respecto al proceso manual.

## Descripción de las fuentes de datos utilizadas

El proyecto utiliza el **Ames Iowa Housing Dataset**, un conjunto de datos público que
registra información detallada de ventas de propiedades residenciales en la ciudad de
Ames, Iowa, Estados Unidos, entre los años 2006 y 2010.

**Características generales de la fuente:**

| Atributo | Detalle |
|---|---|
| Archivo | `Ames_Iowa_Housing_Dataset.csv` |
| N° de registros original | 2930 |
| N° de registros tras preparación (Parte 3) | 2929 (ver justificación en sección de Preparación) |
| N° de variables original | 82 |
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

**Diccionario oficial de variables:** además del diccionario básico elaborado por el equipo,
se utilizó como respaldo académico la ficha técnica oficial del dataset — De Cock, D. (2011).
*"Ames, Iowa: Alternative to the Boston Housing Data as an End of Semester Regression Project."*
Journal of Statistics Education, 19(3). (https://jse.amstat.org/v19n3/decock/DataDocumentation.txt) —
que clasifica formalmente las 82 variables en nominales, ordinales, discretas y continuas, y que
se usó para decidir el tipo de encoding aplicado en la Preparación de datos.

**Evaluación preliminar de ética, sesgos y privacidad:**

- **Privacidad:** el dataset no contiene nombres de propietarios ni direcciones exactas, pero
  el `PID` podría, en teoría, cruzarse con registros públicos del condado para reidentificar la
  propiedad. Se recomienda no exponerlo en entregables públicos.
- **Sesgo potencial:** la variable `Neighborhood` puede actuar como proxy de nivel socioeconómico.
  Un modelo entrenado sobre estos datos podría replicar desigualdades históricas de valoración
  entre barrios en vez de basarse puramente en características objetivas de la vivienda. El EDA
  (ver más abajo) confirma diferencias de precio muy marcadas entre barrios, lo que refuerza este
  riesgo señalado desde la Parte 1.
- **Riesgo de uso:** si este modelo se usara para decisiones de crédito hipotecario o tasación
  automática sin supervisión humana, un sesgo por barrio podría traducirse en discriminación indirecta.

---

## Preparación y análisis exploratorio de los datos (EDA)

### 1. Estadística descriptiva de la variable objetivo (`SalePrice`)

| Estadístico | Valor |
|---|---:|
| Media | 180,796.06 |
| Desv. estándar | 79,886.69 |
| Mínimo | 12,789 |
| Percentil 25 | 129,500 |
| Mediana | 160,000 |
| Percentil 75 | 213,500 |
| Máximo | 755,000 |
| Skewness | 1.74 |
| Curtosis | 5.12 |

La media supera a la mediana y el skewness es marcadamente positivo (1.74): existe una cola
larga de precios altos que se extiende hasta USD 755.000, mientras la mayoría de las ventas
se concentra entre USD 100.000 y 220.000. Esta asimetría es la razón por la que se documenta
—aunque no se aplica en esta etapa— la transformación `log1p(SalePrice)` como propuesta para
la fase de Modelamiento.

### 2. Identificación y tratamiento de valores faltantes

El diagnóstico inicial (`df.isnull().sum()` sobre 2930 filas) mostró 27 variables con nulos,
desde un 0.03% hasta un 99.56% de la columna. El diccionario oficial de variables (De Cock, 2011)
permitió confirmar que la gran mayoría de estos nulos **no son datos perdidos por error**, sino
**ausencia física del atributo**:

| Grupo | Variables | Significado del nulo | Tratamiento aplicado |
|---|---|---|---|
| Piscina, callejón, cerco, extras | `Pool QC` (99.56%), `Misc Feature` (96.38%), `Alley` (93.24%), `Fence` (80.48%) | La casa no tiene ese elemento | Categoría `'None'` |
| Revestimiento exterior | `Mas Vnr Type` (60.58%), `Mas Vnr Area` (0.78%) | Sin revestimiento de mampostería | `'None'` / `0` |
| Chimenea | `Fireplace Qu` (48.53%) | Sin chimenea | `'None'` |
| Garaje | `Garage Qual`, `Garage Cond`, `Garage Yr Blt`, `Garage Finish` (5.43% c/u), `Garage Type` (5.36%), `Garage Cars`/`Garage Area` (0.03% c/u) | Sin garaje | `'None'` / `0` |
| Sótano | `Bsmt Exposure` (2.83%), `BsmtFin Type 2` (2.76%), `Bsmt Qual`, `Bsmt Cond`, `BsmtFin Type 1` (2.73% c/u), y variables numéricas asociadas | Sin sótano | `'None'` / `0` |
| `Electrical` | 1 caso (0.03%) | Dato aislado, no ausencia física | Imputado con la moda |

**Caso especial — `Lot Frontage` (16.72% nulos):** a diferencia del resto, no corresponde a
ausencia física sino a dato no registrado. Se imputó con la **mediana agrupada por `Neighborhood`**,
ya que el tamaño de frente de lote está fuertemente condicionado por el barrio. Dos barrios
(`GrnHill`, `Landmrk`) no tenían ningún valor registrado dentro del grupo, por lo que se usó como
respaldo la mediana global del dataset para esos casos puntuales.

**Casos borde detectados durante la implementación** (no evidentes en el análisis estadístico
agregado, solo al revisar registro por registro): dos propiedades (`PID 910201180` y
`PID 903426160`) tienen `Garage Type` informado —es decir, sí tienen garaje— pero el resto de
los datos de garaje (año, capacidad, superficie) faltantes. Aplicarles la regla general de
`'None'`/`0` habría declarado incorrectamente que no tienen garaje. Se imputaron en su lugar
con la mediana/moda de propiedades con el mismo tipo de garaje.

### 3. Identificación y tratamiento de outliers y anomalías

**Anomalía puntual (error de dato):** el registro `PID 916384070` presentaba `Garage Yr Blt = 2207`,
un valor imposible. Se corrigió a `2007`, coincidiendo con `Year Remod/Add` del mismo registro.

**Outlier con validación externa:** el grupo de propiedades con `Gr Liv Area > 4000 ft²` (5 registros)
incluye 3 ventas de tipo `Partial` (venta no representativa del valor de mercado, según la
documentación oficial del dataset — De Cock, 2011, identifica estos casos como candidatos a
revisión antes de modelar). De esos 3, se decidió **eliminar únicamente el `PID 908154235`**
(5642 ft², USD 160.000), por ser el caso más extremo e inconsistente con el resto del grupo.
Los otros dos casos `Partial` (`PID 908154195` y `PID 908154205`, con precios de 183.850 y
184.750) se mantuvieron, al ser coherentes con el resto de la distribución. Este es el motivo
por el cual el dataset final tiene **2929 registros en vez de 2930**.

**Outliers estadísticos (método IQR) — SalePrice y Lot Frontage:**

| Variable | Límite inferior | Límite superior | Outliers detectados | % del dataset |
|---|---:|---:|---:|---:|
| `SalePrice` | 3,500 | 339,500 | 137 | 4.68% |
| `Lot Frontage` | 25 | 113 | 205 | 7.00% |

Se decidió **no eliminar estos outliers en bloque**: la evidencia (barrios de alto valor como
`NoRidge`, `StoneBr`, `NridgHt`, y niveles altos de `Overall Qual`) indica que corresponden
mayoritariamente a viviendas caras reales, no a errores de carga. Eliminarlos habría hecho
perder señal legítima para el modelo.

### 4. Correlación entre variables numéricas y `SalePrice`

| Variable | Correlación |
|---|---:|
| `Overall Qual` | 0.799 |
| `Gr Liv Area` | 0.707 |
| `Garage Cars` | 0.648 |
| `Garage Area` | 0.640 |
| `Total Bsmt SF` | 0.632 |
| `1st Flr SF` | 0.622 |
| `Year Built` | 0.558 |
| `Full Bath` | 0.546 |
| `Year Remod/Add` | 0.533 |

**Multicolinealidad detectada entre predictores** (pares con correlación severa entre sí):
`Garage Cars`–`Garage Area` (r=0.89), `Year Built`–`Garage Yr Blt` (r=0.83),
`Gr Liv Area`–`TotRms AbvGrd` (r=0.81), `Total Bsmt SF`–`1st Flr SF` (r=0.80).

### 5. Análisis de variables categóricas relevantes

**`Neighborhood`:** diferencias de precio muy marcadas entre barrios — desde una media de
~USD 95.756 en `MeadowV` hasta ~USD 330.319 en `NoRidge`. Varias categorías tienen muy pocas
observaciones (`GrnHill`=2, `Landmrk`=1, `Greens`=8, `Blueste`=10), lo que motivó agruparlas
en `'Other'` antes del encoding (ver sección de transformaciones).

**`Overall Qual`:** relación ascendente clara con `SalePrice` (es la variable con mayor
correlación del dataset, r=0.799). En los niveles altos (8, 9, 10) la dispersión del precio
aumenta drásticamente por la interacción con área construida, ubicación y acabados de lujo.

### 6. Transformaciones aplicadas (Parte 3 — Preparación de datos)

Resumen de las decisiones y su justificación:

| Transformación | Detalle | Justificación |
|---|---|---|
| Corrección de error | `Garage Yr Blt: 2207 → 2007` | Valor imposible; coincide con `Year Remod/Add` |
| Eliminación de registro | `PID 908154235` removido | Venta parcial no representativa (De Cock, 2011) |
| Imputación por ausencia física | `'None'`/`0` en bloques `Garage*`, `Bsmt*`, `Pool QC`, `Alley`, `Fence`, `Misc Feature`, `Mas Vnr Type/Area`, `Fireplace Qu` | Confirmado por diccionario oficial de variables |
| Imputación contextual | `Lot Frontage` → mediana por `Neighborhood` | Tamaño de lote condicionado por ubicación |
| Imputación de casos borde | `PID 910201180`, `PID 903426160` → mediana/moda del mismo `Garage Type` | Garaje confirmado presente; `'None'` habría sido incorrecto |
| Feature engineering | `House_Age`, `Remod_Age`, `Garage_Age` (en vez de años absolutos) | Más interpretables para el modelo |
| Feature engineering | `Total_SF = Total Bsmt SF + 1st Flr SF + 2nd Flr SF` | Mitiga multicolinealidad `Total Bsmt SF`–`1st Flr SF` |
| Reducción de redundancia | Se elimina `Garage Area` (se conserva `Garage Cars`) | r=0.89 entre ambas; variable discreta más simple de interpretar |
| Encoding ordinal | 23 variables mapeadas a escala numérica según orden oficial (ej. `Kitchen Qual`: Po=1…Ex=5) | Preserva la información de orden, evita perderla con one-hot |
| Agrupamiento de categorías raras | `Neighborhood`: categorías con <20 casos → `'Other'` | Evita sobreajuste por categorías con muy pocas observaciones |
| Corrección de tipo | `MS SubClass` forzada a categórica antes de encodear | Es nominal aunque esté codificada con números |
| One-hot encoding | Resto de variables nominales | Sin orden natural entre categorías |
| Retiro de identificadores | `PID`, `Order` excluidos del set de modelado | No aportan información predictiva |

Transformación **no aplicada** en esta etapa (queda documentada como propuesta):
`log1p(SalePrice)` para reducir el sesgo positivo (skew 1.74) — se decidió dejarla para la
fase de Modelamiento en vez de aplicarla en la Preparación.

### 7. Conclusiones sobre la calidad final del dataset preparado

El dataset preparado quedó con **2929 registros y 0 valores nulos**. La reducción de 2930 a
2929 filas está justificada y documentada (eliminación de una venta parcial no representativa).
Se generaron dos versiones del dataset:

- Una versión **legible** (2929 × 86), con las transformaciones de limpieza y feature
  engineering aplicadas pero sin one-hot encoding, útil para inspección y documentación.
- Una versión **lista para modelar** (2929 × 220), completamente codificada y sin nulos.

Los riesgos éticos señalados en la Parte 1 —particularmente el sesgo asociado a `Neighborhood`—
se mantienen vigentes: el EDA confirma que el barrio explica una parte importante de la
variabilidad del precio, por lo que el modelo final debería evaluarse también por barrio,
no solo con métricas agregadas, antes de usarse en decisiones reales de tasación o crédito.

---

## Metodología utilizada (CRISP-DM)

El proyecto se desarrolla siguiendo la metodología **CRISP-DM (Cross-Industry Standard Process
for Data Mining)**, que estructura el trabajo en las siguientes fases:

1. **Comprensión del negocio (Business Understanding)** — *Completado, Parte 1*
   Definición del problema de negocio, objetivos del proyecto y KPIs que permitirán evaluar
   si la solución de Machine Learning resuelve la necesidad planteada.

2. **Comprensión de los datos (Data Understanding)** — *Completado, Parte 1 y 2*
   Identificación de la fuente de datos, revisión de su estructura (2930 registros, 82
   variables), diccionario de variables (básico y oficial), evaluación preliminar de calidad,
   sesgos y privacidad, y análisis exploratorio completo (estadística descriptiva, nulos,
   outliers, correlaciones, variables categóricas clave).

3. **Preparación de los datos (Data Preparation)** — *Completado, Parte 3*
   Corrección de errores puntuales, imputación diferenciada de valores faltantes (ausencia
   física vs. dato no registrado), tratamiento documentado de outliers y anomalías, ingeniería
   de variables (edades, superficie total, reducción de multicolinealidad) y encoding
   (ordinal y one-hot) según el diccionario oficial de variables. Resultado: dataset de
   2929 × 220 sin valores nulos, listo para modelar.

4. **Modelamiento (Modeling)** — *Pendiente, fase posterior*
   Entrenamiento y evaluación de modelos de regresión sobre el dataset preparado, usando
   los KPIs definidos (MAE, % de predicciones dentro de ±10%) para medir el desempeño.
