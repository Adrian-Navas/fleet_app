# Fleet Optimizer & Forecaster (España)

Aplicación educativa en **Streamlit** para simular una red de estaciones de alquiler de coches en España, predecir su demanda diaria y optimizar la recolocación de flota para maximizar la cobertura al menor coste logístico.

## Características principales
- **Generación de datos sintéticos**: crea más de 50 estaciones con coordenadas reales aproximadas, capacidad y demanda base diferenciada por segmento (ciudad/aeropuerto) y seasonality local (islas, verano, festivos).【F:data_generator.py†L5-L116】
- **Predicción de demanda**: entrena modelos de **Regresión Lineal** o **XGBoost** usando lags temporales, variables de calendario y banderas de estacionalidad; incluye evaluación con MAE/RMSE/R² y predicciones a 3 meses por estación.【F:app.py†L105-L204】【F:forecasting.py†L5-L36】
- **Análisis visual**: tabs para inspeccionar series históricas, descomposición estacional, comparativa real vs. predicción, mapas interactivos (Folium) y métricas de error agregadas por estación.【F:app.py†L43-L214】【F:app.py†L215-L389】【F:app.py†L435-L532】
- **Optimización de flota**: modelo de **programación lineal** con `PuLP` que calcula movimientos óptimos de vehículos minimizando kilómetros recorridos y penalizando déficit de demanda (táctico diario o estratégico a 3 meses).【F:optimization.py†L13-L142】【F:optimization.py†L144-L221】
- **Mapas operativos**: visualización de calor por utilización prevista y flechas animadas que muestran recolocaciones sugeridas entre estaciones.【F:app.py†L532-L665】【F:app.py†L666-L733】

## Estructura del repositorio
- `app.py`: interfaz Streamlit con pestañas de introducción, datos, predicción, mapa, optimización y comparativa.【F:app.py†L12-L733】
- `data_generator.py`: utilidades para generar estaciones, demanda diaria y estado inicial de flota.【F:data_generator.py†L1-L122】
- `forecasting.py`: clase `ForecastModeler` con preprocesamiento (escalado + one-hot) y modelos LR/XGBoost.【F:forecasting.py†L5-L34】
- `optimization.py`: clase `FleetOptimizer` para calcular distancias, resolver recolocación táctica o estratégica y devolver movimientos/estados finales.【F:optimization.py†L5-L221】
- `utils.py`: helpers menores para UI (popovers, tablas de métricas).【F:utils.py†L1-L10】
- `requirements.txt`: dependencias necesarias para ejecutar la app.【F:requirements.txt†L1-L12】

## Requisitos previos
- Python 3.9+ recomendado.
- Dependencias del sistema para `geopy`, `statsmodels` y `xgboost` (instalables vía `apt` en la mayoría de entornos Linux).

## Instalación y ejecución
1. Clona el repositorio y crea un entorno virtual opcional.
2. Instala dependencias:
   ```bash
   pip install -r requirements.txt
   ```
3. Lanza la aplicación Streamlit:
   ```bash
   streamlit run app.py
   ```
4. En la barra lateral, pulsa **"Generar Datos Mock"**, entrena un modelo en la pestaña **Predicción** y navega por las pestañas **Mapa** y **Optimización** para ver resultados.

## Flujo de trabajo sugerido
1. **Generar datos**: crea la red de estaciones y 2 años de demanda diaria con lags y variables de calendario.【F:data_generator.py†L63-L122】
2. **Explorar demanda**: usa la pestaña **Datos** para ver históricos y descomposición estacional por estación.【F:app.py†L215-L334】
3. **Entrenar y evaluar**: selecciona modelo y parámetros, entrena y revisa métricas/curvas de predicción en **Predicción**.【F:app.py†L339-L532】
4. **Simular movilidad**: visualiza predicciones en el mapa (calor y burbujas) y ejecuta el optimizador para obtener movimientos recomendados y KPIs de cobertura.【F:app.py†L532-L733】

## Modelado y decisiones clave
- **Features**: lags `rentals_lag_1`, `rentals_lag_7`, día de la semana, mes, fines de semana y temporada alta; los lags se actualizan recursivamente al proyectar futuros 90 días por estación.【F:app.py†L141-L204】
- **Generación de demanda**: seasonality reforzada en verano, fines de semana diferenciados por segmento (aeropuerto vs. ciudad), factor de festivos y ruido reducido para señales más predecibles.【F:data_generator.py†L69-L116】
- **Optimización**: minimiza kilómetros recorridos ponderados por coches movidos y penaliza déficit; respeta capacidad por estación y conservación de flota total.【F:optimization.py†L35-L118】【F:optimization.py†L146-L221】

## Notas
- El código está orientado a propósitos didácticos y de prototipado; adapte parámetros y restricciones si se usa con datos reales.
- Para reproducir resultados coherentes se fija la semilla de NumPy al generar demanda.【F:data_generator.py†L63-L64】
