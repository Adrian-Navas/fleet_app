import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import folium
from streamlit_folium import st_folium
from data_generator import generate_stations, generate_daily_demand, generate_fleet_state
from forecasting import ForecastModeler
from optimization import FleetOptimizer
from utils import make_popover

# --- Page Config ---
st.set_page_config(
    page_title="Fleet Optimizer & Forecaster",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Session State ---
if "data_loaded" not in st.session_state:
    st.session_state.data_loaded = False
if "model_trained" not in st.session_state:
    st.session_state.model_trained = False
if "opt_results" not in st.session_state:
    st.session_state.opt_results = None

# --- Sidebar ---
with st.sidebar:
    st.title("⚙️ Configuración")
    capacity_factor = st.number_input(
        "Factor multiplicativo de capacidad",
        min_value=0.1,
        max_value=5.0,
        value=1.0,
        step=0.1,
        help="Escala la capacidad base de todas las estaciones (1 = capacidad original)."
    )

    if st.button("Generar Datos Mock"):
        with st.spinner("Generando red de estaciones y demanda..."):
            stations = generate_stations(capacity_factor=capacity_factor)
            demand = generate_daily_demand(stations)

            st.session_state.stations = stations
            st.session_state.demand = demand
            st.session_state.data_loaded = True
            st.session_state.model_trained = False
            st.success("¡Datos generados!")
            
    st.markdown("---")
    st.info("Herramienta pedagógica para optimización de flota en Renting.")

# --- Main Title ---
st.title("🚗 Predicción de Demanda y Optimización de Flota")
st.markdown("### Forecasting inteligente y reubicación de vehículos para maximizar ingresos.")

# --- Tabs ---
tab_intro, tab_data, tab_forecast, tab_map, tab_opt = st.tabs(
    ["📚 Introducción", "📊 Datos", "📈 Predicción", "🗺️ Mapa", "🚚 Optimización"]
)

# --- Tab 1: Intro ---
# --- Tab 1: Introducción ---
with tab_intro:
    st.markdown("""
    ### 🚗 Optimización Inteligente de Flota: El Motor de la Rentabilidad
    
    En el competitivo sector del **Rent a Car (RAC)**, la gestión eficiente de la flota no es solo una cuestión logística, es el principal **driver de rentabilidad**. 
    Tener el coche adecuado, en el lugar adecuado, en el momento adecuado es el "Santo Grial" de las operaciones.
    
    Esta herramienta demuestra cómo la **Inteligencia Artificial** y la **Investigación Operativa** pueden transformar la toma de decisiones, pasando de la intuición a la precisión matemática.
    """)
    
    st.divider()
    
    col_ctx1, col_ctx2 = st.columns(2)
    
    with col_ctx1:
        st.info("#### 📉 El Problema: Desequilibrio Oferta-Demanda")
        st.markdown("""
        La demanda de alquiler de coches es altamente volátil y estacional:
        *   **Picos Estacionales:** Verano en costas e islas, Navidad en ciudades.
        *   **Patrones Semanales:** Viajes de negocios (L-J) vs. Ocio (V-D).
        *   **Eventos:** Congresos (MWC), ferias, festivos locales.
        
        Esto provoca dos situaciones costosas:
        1.  **Stockouts (Rotura de Stock):** Tienes demanda pero no coches. **Pierdes ventas y clientes.**
        2.  **Overstock (Exceso de Flota):** Tienes coches parados. **Coste de oportunidad y depreciación.**
        """)
        
    with col_ctx2:
        st.success("#### 🚀 La Solución: Predicción + Optimización")
        st.markdown("""
        Esta aplicación aborda el problema en dos fases:
        1.  **Predicción de Demanda (Forecasting):** Usamos algoritmos de Machine Learning (**XGBoost**) para anticipar cuántos coches se necesitarán en cada estación.
        2.  **Optimización de Flota (Solver):** Un algoritmo matemático (**Programación Lineal**) decide qué coches mover de A a B para cubrir la demanda al **mínimo coste**.
        """)

    st.divider()

    st.subheader("🚀 Impacto Real en Negocio")
    st.markdown("Implementar sistemas de optimización basados en IA permite alcanzar mejoras tangibles respecto a la gestión manual:")
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Utilización de Flota", "+5-10%", delta="Mejora", help="Incremento en el tiempo que los vehículos están alquilados gracias a mejor posicionamiento.")
    m2.metric("Costes Logísticos", "-15%", delta="Ahorro", delta_color="normal", help="Reducción de movimientos improductivos y optimización de rutas de camiones/drivers.")
    m3.metric("Ventas Perdidas", "-20%", delta="Reducción", delta_color="normal", help="Menos stockouts en días pico gracias a la anticipación de demanda.")
    m4.metric("RevPAR", "+8%", delta="Incremento", help="Aumento del ingreso por vehículo disponible al capturar demanda de mayor valor.")

    st.markdown("""
    > **📊 Datos del Sector:**
    > Según estudios de *McKinsey* y *BCG* en movilidad, las empresas que digitalizan su gestión de flota reducen su **TCO (Total Cost of Ownership)** en un **10-15%** y aumentan su margen operativo significativamente. 
    >
    > *Ejemplo:* Para una flota de 5,000 vehículos, un **+1% de utilización** equivale a **~50 coches adicionales** generando ingresos sin coste de adquisición (CAPEX).
    """)
    
    with st.expander("📚 Ver Glosario Técnico"):
        st.markdown("""
        *   **Shortfall (Déficit):** Demanda no cubierta por falta de stock.
        *   **Surplus (Excedente):** Stock sobrante que no se va a alquilar.
        *   **Relocation (Movimiento):** Traslado de vehículos entre bases (Trucking o Drivers).
        *   **Lead Time:** Tiempo de antelación con el que se realiza la reserva o la predicción.
        """)

# --- Tab 2: Data ---
with tab_data:
    if not st.session_state.data_loaded:
        st.warning("Genera datos primero.")
    else:
        st.header("Análisis de Demanda por Estación")
        
        # Station selector
        station_names = st.session_state.stations.set_index('station_id')['station_name'].to_dict()
        selected_station_id = st.selectbox(
            "Selecciona una Estación:",
            options=st.session_state.stations['station_id'].tolist(),
            format_func=lambda x: f"{station_names[x]} (ID: {x})"
        )
        
        # Filter data for selected station
        station_data = st.session_state.demand[st.session_state.demand['station_id'] == selected_station_id].copy()
        station_data = station_data.sort_values('date')
        
        # 1. Historical Demand Plot
        st.subheader("📈 Histórico de Demanda")
        fig_hist = px.line(
            station_data, 
            x='date', 
            y='rentals',
            title=f"Demanda Histórica - {station_names[selected_station_id]}",
            labels={'rentals': 'Alquileres', 'date': 'Fecha'}
        )
        fig_hist.update_traces(line_color='#1f77b4', line_width=2)
        st.plotly_chart(fig_hist, use_container_width=True)
        
        # 2. Seasonal Decomposition
        st.subheader("🔍 Descomposición Estacional")
        
        try:
            from statsmodels.tsa.seasonal import seasonal_decompose
            
            # Prepare time series (need regular frequency)
            ts_data = station_data.set_index('date')['rentals']
            ts_data = ts_data.asfreq('D')  # Daily frequency
            
            # Perform decomposition (period=7 for weekly seasonality)
            decomposition = seasonal_decompose(ts_data, model='additive', period=7, extrapolate_trend='freq')
            
            # Create subplots for decomposition components
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            
            fig_decomp = make_subplots(
                rows=4, cols=1,
                subplot_titles=('Observed (Original)', 'Trend (Tendencia)', 'Seasonal (Estacionalidad)', 'Residual (Ruido)'),
                vertical_spacing=0.08
            )
            
            # Observed
            fig_decomp.add_trace(
                go.Scatter(x=decomposition.observed.index, y=decomposition.observed.values, 
                          mode='lines', name='Observed', line=dict(color='blue')),
                row=1, col=1
            )
            
            # Trend
            fig_decomp.add_trace(
                go.Scatter(x=decomposition.trend.index, y=decomposition.trend.values, 
                          mode='lines', name='Trend', line=dict(color='orange')),
                row=2, col=1
            )
            
            # Seasonal
            fig_decomp.add_trace(
                go.Scatter(x=decomposition.seasonal.index, y=decomposition.seasonal.values, 
                          mode='lines', name='Seasonal', line=dict(color='green')),
                row=3, col=1
            )
            
            # Residual
            fig_decomp.add_trace(
                go.Scatter(x=decomposition.resid.index, y=decomposition.resid.values, 
                          mode='lines', name='Residual', line=dict(color='red')),
                row=4, col=1
            )
            
            fig_decomp.update_layout(height=800, showlegend=False, title_text="Componentes de la Serie Temporal")
            fig_decomp.update_xaxes(title_text="Fecha", row=4, col=1)
            
            st.plotly_chart(fig_decomp, use_container_width=True)
            
            with st.expander("ℹ️ ¿Qué significan estos componentes?"):
                st.markdown("""
                - **Observed:** La serie temporal original (demanda real)
                - **Trend:** La tendencia a largo plazo (¿está creciendo o decreciendo?)
                - **Seasonal:** Patrones que se repiten periódicamente (ej. picos los fines de semana)
                - **Residual:** Ruido aleatorio que no se puede explicar con tendencia o estacionalidad
                
                **Fórmula:** `Observed = Trend + Seasonal + Residual` (modelo aditivo)
                """)
                
        except Exception as e:
            st.error(f"No se pudo realizar la descomposición estacional: {e}")
            st.info("Asegúrate de tener suficientes datos históricos (mínimo 2 ciclos estacionales).")
        



# --- Tab 3: Forecast ---
with tab_forecast:
    if not st.session_state.data_loaded:
        st.warning("Genera datos primero.")
    else:
        st.header("Modelo de Predicción")
        
        col1, col2 = st.columns(2)
        with col1:
            model_choice = st.radio("Modelo", ["Linear Regression", "XGBoost"], help="Lineal es simple, XGBoost captura complejidad.")
            
            xgb_params = {}
            if model_choice == "XGBoost":
                n_est = st.slider("Nº Árboles", 50, 500, 200)
                depth = st.slider("Profundidad", 3, 10, 6)
                xgb_params = {"n_estimators": n_est, "max_depth": depth}
                
        with col2:
            if st.button("Entrenar Modelo"):
                with st.spinner("Entrenando..."):
                    # Train/Test Split (Time based)
                    df = st.session_state.demand.copy()
                    cutoff = df["date"].max() - pd.Timedelta(days=60)
                    
                    train = df[df["date"] < cutoff]
                    test = df[df["date"] >= cutoff]
                    
                    features = ["rentals_lag_1", "rentals_lag_7", "day_of_week", "month", "is_weekend", "is_high_season"]
                    target = "rentals"
                    
                    modeler = ForecastModeler()
                    modeler.train(train[features], train[target], model_choice, xgb_params)
                    
                    preds = modeler.predict(test[features])
                    metrics = modeler.evaluate(test[target], preds)
                    
                    # Save results
                    test["prediction"] = preds
                    st.session_state.modeler = modeler
                    st.session_state.test_results = test
                    st.session_state.metrics = metrics
                    
                    # --- Generate Global Future Predictions (3 Months) ---
                    future_preds_all = []
                    last_date_test = test["date"].max()
                    future_dates_global = [last_date_test + pd.Timedelta(days=i) for i in range(1, 91)]  # 90 days = 3 months
                    
                    # Iterate over all stations to generate future predictions
                    for station_id in st.session_state.stations["station_id"].unique():
                        # Get recent data for this station to initialize lags
                        station_test_data = test[test["station_id"] == station_id].sort_values("date")
                        
                        if len(station_test_data) > 0:
                            last_known_rental = station_test_data.iloc[-1]["rentals"]
                            current_lag_1 = last_known_rental
                            
                            station_future_preds = []
                            
                            for date in future_dates_global:
                                # Construct features
                                date_7_days_ago = date - pd.Timedelta(days=7)
                                
                                # Find lag_7
                                found_lag_7 = False
                                # Check in already generated future preds for this station
                                for fp in station_future_preds:
                                    if fp['date'] == date_7_days_ago:
                                        lag_7 = fp['prediction']
                                        found_lag_7 = True
                                        break
                                
                                if not found_lag_7:
                                    # Check in test data
                                    lag_7_row = station_test_data[station_test_data['date'] == date_7_days_ago]
                                    if not lag_7_row.empty:
                                        lag_7 = lag_7_row.iloc[0]['rentals']
                                    else:
                                        lag_7 = current_lag_1 # Fallback
                                
                                month = date.month
                                dow = date.dayofweek
                                is_weekend = 1 if dow >= 5 else 0
                                is_high_season = 1 if month in [6, 7, 8, 9] else 0
                                
                                feat_row = pd.DataFrame([{
                                    "rentals_lag_1": current_lag_1,
                                    "rentals_lag_7": lag_7,
                                    "day_of_week": dow,
                                    "month": month,
                                    "is_weekend": is_weekend,
                                    "is_high_season": is_high_season
                                }])
                                
                                pred = modeler.predict(feat_row)[0]
                                pred = max(0, pred)
                                
                                station_future_preds.append({
                                    "station_id": station_id,
                                    "date": date,
                                    "prediction": pred
                                })
                                current_lag_1 = pred
                            
                            future_preds_all.extend(station_future_preds)
                            
                    st.session_state.future_preds_df = pd.DataFrame(future_preds_all)
                    st.session_state.model_trained = True
                    st.success("Modelo entrenado y predicciones futuras generadas.")
                    
        if st.session_state.model_trained:
            st.markdown("---")
            st.subheader("Visualización de Predicciones")
            
            stat_id = st.selectbox("Selecciona Estación", st.session_state.stations["station_id"].unique())
            stat_data = st.session_state.test_results[st.session_state.test_results["station_id"] == stat_id]
            
            fig = px.line(stat_data, x="date", y=["rentals", "prediction"], title=f"Predicción vs Realidad - Estación {stat_id}")
            st.plotly_chart(fig, use_container_width=True)
            
            c1, c2, c3 = st.columns(3)
            c1.metric("MAE", f"{st.session_state.metrics['MAE']:.2f}")
            c2.metric("RMSE", f"{st.session_state.metrics['RMSE']:.2f}")
            c3.metric("R2", f"{st.session_state.metrics['R2']:.2f}")
            
            # Weekly Forecast Comparison
            st.markdown("---")
            st.subheader("📊 Comparación Semanal: Real vs Predicción")
            
            # Get test results for selected station
            station_test = st.session_state.test_results[
                st.session_state.test_results['station_id'] == stat_id
            ].copy()
            
            if len(station_test) >= 14:  # Need at least 2 weeks
                # Get last 2 weeks of data
                last_date = station_test['date'].max()
                two_weeks_ago = last_date - pd.Timedelta(days=13)
                
                recent_data = station_test[station_test['date'] >= two_weeks_ago].copy()
                recent_data = recent_data.sort_values('date')
                
                # --- Future Prediction (3 Weeks Calculation) ---
                # We calculate 3 weeks (21 days) even if we only show 1 week
                future_dates = [last_date + pd.Timedelta(days=i) for i in range(1, 22)]
                future_preds = []
                
                # We need the last known rental for lag_1
                last_known_rental = recent_data.iloc[-1]['rentals']
                
                # We need rentals from 7 days ago for lag_7
                # We can get them from recent_data (assuming it has enough history)
                
                current_lag_1 = last_known_rental
                
                for i, date in enumerate(future_dates):
                    # Construct features
                    # lag_7 comes from 7 days ago (which is in recent_data or history)
                    date_7_days_ago = date - pd.Timedelta(days=7)
                    
                    # Try to find lag_7 in recent_data or previous future preds
                    # First check if it's in the future predictions we just made
                    found_lag_7 = False
                    for fp in future_preds:
                        if fp['date'] == date_7_days_ago:
                            lag_7 = fp['prediction']
                            found_lag_7 = True
                            break
                    
                    if not found_lag_7:
                        # Check in recent data
                        lag_7_row = recent_data[recent_data['date'] == date_7_days_ago]
                        if not lag_7_row.empty:
                            lag_7 = lag_7_row.iloc[0]['rentals']
                        else:
                            # Fallback if not found (shouldn't happen with enough history)
                            lag_7 = current_lag_1 
                    
                    month = date.month
                    dow = date.dayofweek
                    is_weekend = 1 if dow >= 5 else 0
                    is_high_season = 1 if month in [6, 7, 8, 9] else 0
                    
                    # Create feature row (must match training columns)
                    feat_row = pd.DataFrame([{
                        "rentals_lag_1": current_lag_1,
                        "rentals_lag_7": lag_7,
                        "day_of_week": dow,
                        "month": month,
                        "is_weekend": is_weekend,
                        "is_high_season": is_high_season
                    }])
                    
                    # Predict
                    pred = st.session_state.modeler.predict(feat_row)[0]
                    # Ensure non-negative predictions
                    pred = max(0, pred)
                    
                    future_preds.append({
                        "date": date,
                        "prediction": pred
                    })
                    
                    # Update lag_1 for next iteration (recursive)
                    current_lag_1 = pred
                
                future_df = pd.DataFrame(future_preds)
                
                # --- Unified Prediction DataFrame ---
                # Combine past predictions (from recent_data) and future predictions
                # NOTE: We only show 1 week of future predictions in the plot
                future_df_plot = future_df.iloc[:7].copy()
                
                past_preds = recent_data[['date', 'prediction']].copy()
                all_preds = pd.concat([past_preds, future_df_plot], ignore_index=True)
                all_preds = all_preds.sort_values('date')
                
                # --- Visualization ---
                import plotly.graph_objects as go
                fig_weekly = go.Figure()
                
                # Real demand
                fig_weekly.add_trace(go.Scatter(
                    x=recent_data['date'],
                    y=recent_data['rentals'],
                    mode='lines+markers',
                    name='Demanda Real',
                    line=dict(color='blue', width=2),
                    marker=dict(size=6)
                ))
                
                # Unified Predicted demand (Past + Future)
                fig_weekly.add_trace(go.Scatter(
                    x=all_preds['date'],
                    y=all_preds['prediction'],
                    mode='lines+markers',
                    name='Predicción (Test + Futura)',
                    line=dict(color='red', width=2, dash='dash'),
                    marker=dict(size=6, symbol='x')
                ))
                
                fig_weekly.update_layout(
                    title=f"Últimas 2 Semanas + Próxima Semana - {st.session_state.stations[st.session_state.stations['station_id']==stat_id].iloc[0]['station_name']}",
                    xaxis_title="Fecha",
                    yaxis_title="Alquileres",
                    hovermode='x unified',
                    height=400
                )
                
                st.plotly_chart(fig_weekly, use_container_width=True)
                
                # Metrics
                col1, col2, col3 = st.columns(3)
                mae = np.mean(np.abs(recent_data['rentals'] - recent_data['prediction']))
                mape = np.mean(np.abs((recent_data['rentals'] - recent_data['prediction']) / recent_data['rentals'])) * 100
                
                col1.metric("MAE (Test)", f"{mae:.2f}", help="Error absoluto medio en datos pasados")
                col2.metric("MAPE (Test)", f"{mape:.1f}%", help="Error porcentual medio en datos pasados")
                col3.metric("Horizonte", "1 semana futura")
            else:
                st.info("Selecciona una estación con al menos 14 días de predicciones para ver la comparación semanal.")

# --- Tab 4: Map ---
with tab_map:
    if not st.session_state.model_trained:
        st.warning("Entrena el modelo para ver predicciones en el mapa.")
    else:
        st.header("Mapa de Demanda Predicha")
        
        map_type = st.radio("Tipo de Vista", ["Diaria", "Semanal"], horizontal=True)
        
        if map_type == "Diaria":
            # Default to today, allow up to 3 weeks future
            today = pd.Timestamp.now().normalize()
            max_date = today + pd.Timedelta(days=21)
            
            date_select = st.date_input("Fecha", today, min_value=st.session_state.test_results["date"].min(), max_value=max_date)
            date_select = pd.Timestamp(date_select)
            
            # Fetch data based on date
            if date_select <= st.session_state.test_results["date"].max():
                day_data = st.session_state.test_results[st.session_state.test_results["date"] == date_select]
            else:
                # Use future predictions
                if "future_preds_df" in st.session_state:
                    day_data = st.session_state.future_preds_df[st.session_state.future_preds_df["date"] == date_select]
                else:
                    day_data = pd.DataFrame()
            
            if day_data.empty:
                st.warning("No hay predicciones para esta fecha.")
            else:
                m = folium.Map(location=[40.0, -3.7], zoom_start=6)
                
                for _, row in day_data.iterrows():
                    stat_info = st.session_state.stations[st.session_state.stations["station_id"] == row["station_id"]].iloc[0]
                    
                    # Color by demand intensity relative to capacity (Utilization)
                    util = row["prediction"] / stat_info["capacity"]
                    color = "green" if util < 0.5 else "orange" if util < 0.8 else "red"
                    
                    # Radius by absolute demand (Size)
                    # Scale: 20 demand -> 4px, 200 demand -> 40px
                    radius = max(4, row["prediction"] / 5)
                    
                    folium.CircleMarker(
                        location=[stat_info["lat"], stat_info["lon"]],
                        radius=radius,
                        color=color,
                        fill=True,
                        popup=f"{stat_info['station_name']}<br>Pred: {int(row['prediction'])}<br>Cap: {stat_info['capacity']}<br>Util: {util:.0%}"
                    ).add_to(m)
                
                st_folium(m, width=800, height=500)
                
        else: # Weekly View
            # Aggregate by week (combine test and future results)
            df_test = st.session_state.test_results[['station_id', 'date', 'prediction']].copy()
            df_future = st.session_state.future_preds_df[['station_id', 'date', 'prediction']].copy() if "future_preds_df" in st.session_state else pd.DataFrame()
            
            df_combined = pd.concat([df_test, df_future], ignore_index=True)
            df_combined['week_start'] = df_combined['date'].dt.to_period('W').apply(lambda r: r.start_time)
            
            # Select week
            available_weeks = sorted(df_combined['week_start'].unique())
            # Default to current week
            current_week_start = pd.Timestamp.now().normalize().to_period('W').start_time
            
            # Find closest available week to current week
            default_index = 0
            if current_week_start in available_weeks:
                default_index = available_weeks.index(current_week_start)
            
            week_select = st.selectbox("Semana (Inicio)", available_weeks, index=default_index, format_func=lambda x: x.strftime('%Y-%m-%d'))
            
            # Filter and sum
            week_data = df_combined[df_combined['week_start'] == week_select]
            week_agg = week_data.groupby('station_id')['prediction'].sum().reset_index()
            
            if week_agg.empty:
                st.warning("No hay datos para esta semana.")
            else:
                m = folium.Map(location=[40.0, -3.7], zoom_start=6)
                
                for _, row in week_agg.iterrows():
                    stat_info = st.session_state.stations[st.session_state.stations["station_id"] == row["station_id"]].iloc[0]
                    
                    # Weekly capacity (approx 7 * daily capacity)
                    weekly_capacity = stat_info["capacity"] * 7
                    util = row["prediction"] / weekly_capacity
                    
                    # Color logic (Utilization)
                    color = "green" if util < 0.5 else "orange" if util < 0.8 else "red"
                    
                    # Radius by absolute demand (Size)
                    # Scale: 140 demand -> 4px, 1400 demand -> 40px
                    radius = max(4, row["prediction"] / 35)
                    
                    folium.CircleMarker(
                        location=[stat_info["lat"], stat_info["lon"]],
                        radius=radius, # Scale radius
                        color=color,
                        fill=True,
                        popup=f"{stat_info['station_name']}<br>Pred Semanal: {int(row['prediction'])}<br>Cap Semanal: {weekly_capacity}<br>Util: {util:.0%}"
                    ).add_to(m)
                
                st_folium(m, width=800, height=500)

# --- Tab 5: Optimization ---
with tab_opt:
    if not st.session_state.model_trained:
        st.warning("Necesitas predicciones para optimizar.")
    else:
        st.header("Optimización Estratégica de Flota (3 Meses)")
        
        st.markdown("""
        ### 🎯 Optimización a Largo Plazo
        
        Este optimizador redistribuye la flota basándose en la **demanda agregada de los próximos 3 meses**:
        1. **Calcula demanda total** por estación para los próximos 90 días
        2. **Identifica zonas** con exceso vs déficit de demanda
        3. **Mueve coches** de zonas de baja demanda a zonas de alta demanda
        4. **Minimiza costes** de transporte manteniendo cobertura óptima
        
        > **Ventaja**: Optimización estratégica en lugar de táctica (día a día)
        """)
        
        # Check if we have future predictions
        if "future_preds_df" not in st.session_state or st.session_state.future_preds_df.empty:
            st.error("⚠️ No hay predicciones futuras disponibles. Entrena el modelo primero.")
            st.stop()
        
        # Date Selection (starting point for 3-month window)
        st.subheader("📅 Ventana de Optimización")
        
        # Get available dates from future predictions
        future_dates = sorted(st.session_state.future_preds_df["date"].unique())
        
        # Filter to dates that have at least 90 days of predictions ahead
        max_date = max(future_dates)
        valid_start_dates = [d for d in future_dates if (max_date - d).days >= 89]
        
        if not valid_start_dates:
            st.error("⚠️ No hay suficientes predicciones futuras (se necesitan al menos 90 días).")
            st.stop()
        
        # Default to first available date (usually today or tomorrow)
        start_date = st.selectbox(
            "Fecha de inicio (próximos 3 meses desde esta fecha)",
            valid_start_dates,
            index=0,
            format_func=lambda x: x.strftime('%Y-%m-%d (%A)'),
            help="Selecciona la fecha de inicio. El optimizador usará los 90 días siguientes."
        )
        
        end_date = start_date + pd.Timedelta(days=89)
        
        st.info(f"📊 Optimizando para el período: **{start_date.strftime('%Y-%m-%d')}** → **{end_date.strftime('%Y-%m-%d')}** (90 días)")
        
        # Filter predictions for the 3-month window
        future_window = st.session_state.future_preds_df[
            (st.session_state.future_preds_df["date"] >= start_date) &
            (st.session_state.future_preds_df["date"] <= end_date)
        ].copy()
        
        if len(future_window) == 0:
            st.error("No hay predicciones disponibles para esta ventana.")
            st.stop()
        
        # Generate Fleet State for start date (Mock)
        if "fleet_state" not in st.session_state or st.session_state.get("last_opt_date") != start_date:
            st.session_state.fleet_state = generate_fleet_state(st.session_state.stations, start_date)
            st.session_state.last_opt_date = start_date
            
        current_dict = dict(zip(st.session_state.fleet_state["station_id"], st.session_state.fleet_state["cars_available"]))
        
        # Calculate aggregate demand
        agg_demand = future_window.groupby('station_id')['prediction'].sum()
        
        # Show preview of aggregate demand
        with st.expander("📊 Vista Previa: Demanda Agregada por Estación (Top 10)"):
            preview_df = agg_demand.sort_values(ascending=False).head(10).reset_index()
            preview_df.columns = ["Station ID", "Demanda Total (90 días)"]
            preview_df["Demanda Promedio Diaria"] = (preview_df["Demanda Total (90 días)"] / 90).round(1)
            st.dataframe(preview_df, use_container_width=True)
        
        # Controls
        st.subheader("⚙️ Parámetros de Optimización")
        
        col_param1, col_param2 = st.columns(2)
        
        with col_param1:
            lambda_pen = st.slider(
                "Penalización por Demanda No Cubierta (€/coche)", 
                1.0, 100.0, 10.0,
                help="**λ (lambda)**: Penalización por cada coche de demanda no cubierta en el período de 3 meses. Valores altos priorizan cobertura sobre costes de transporte."
            )
            
        with col_param2:
            total_fleet = sum(current_dict.values())
            total_demand_agg = agg_demand.sum()
            avg_daily_demand = total_demand_agg / 90

            st.metric("Total Flota Disponible", f"{total_fleet} coches")
            st.metric("Demanda Total (3 meses)", f"{int(total_demand_agg)} alquileres")
            st.metric("Demanda Promedio Diaria", f"{int(avg_daily_demand)} alquileres/día")

        def render_opt_results(moves, res_df, total_demand_val):
            st.success(f"✅ Optimización completada. Se proponen **{len(moves)} movimientos estratégicos**.")

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("🚗 Movimientos Propuestos")
                if moves:
                    moves_df = pd.DataFrame(moves)[["from_name", "to_name", "amount", "cost"]]
                    moves_df.columns = ["Origen", "Destino", "Coches", "Distancia (km)"]
                    # Sort by amount descending
                    moves_df = moves_df.sort_values("Coches", ascending=False)
                    st.dataframe(moves_df, use_container_width=True)

                    # Show summary
                    st.markdown("**Resumen de Movimientos:**")
                    st.write(f"- **Total movimientos**: {len(moves)}")
                    st.write(f"- **Mayor movimiento**: {moves_df['Coches'].max()} coches")
                    st.write(f"- **Distancia promedio**: {moves_df['Distancia (km)'].mean():.1f} km")
                else:
                    st.info("✨ No es necesario mover coches. La distribución actual es óptima para los próximos 3 meses.")

            with col2:
                st.subheader("📊 Estado Final por Estación")
                display_df = res_df[["station_name", "initial", "final", "target_aggregate", "shortfall"]].copy()
                display_df.columns = ["Estación", "Inicial", "Final", "Demanda 3M", "Déficit 3M"]
                # Show only stations with changes or shortfall
                display_df["Cambio"] = display_df["Final"] - display_df["Inicial"]
                display_df = display_df[
                    (display_df["Cambio"] != 0) | (display_df["Déficit 3M"] > 0)
                ].sort_values("Déficit 3M", ascending=False)

                if len(display_df) > 0:
                    st.dataframe(display_df, use_container_width=True)
                else:
                    st.info("Todas las estaciones mantienen su flota actual.")

            # Metrics
            st.markdown("---")
            st.subheader("📈 Métricas de Optimización")
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)

            total_cost = sum(m["cost"] for m in moves)
            total_shortfall = res_df["shortfall"].sum()
            total_moved = sum(m["amount"] for m in moves)
            coverage = ((total_demand_val - total_shortfall) / total_demand_val * 100) if total_demand_val > 0 else 100

            col_m1.metric("Distancia Total", f"{total_cost:.0f} km", help="Suma de distancias × coches movidos")
            col_m2.metric("Demanda No Cubierta", f"{int(total_shortfall)} alquileres", help="Demanda total que no podrá ser atendida en 3 meses")
            col_m3.metric("Coches Movidos", f"{int(total_moved)}", help="Total de coches a redistribuir")
            col_m4.metric("Cobertura", f"{coverage:.1f}%", help="% de demanda cubierta en el período")

            # Visualization: Demand vs Capacity
            st.markdown("---")
            st.subheader("📊 Análisis de Cobertura por Estación")

            viz_df = res_df.copy()
            viz_df["capacity_3m"] = viz_df["capacity_aggregate"]
            viz_df["demand_3m"] = viz_df["target_aggregate"]
            viz_df["supply_3m"] = viz_df["final"] * 90

            # Top 10 stations by demand
            top_stations = viz_df.nlargest(10, "demand_3m")

            import plotly.graph_objects as go
            fig_coverage = go.Figure()

            fig_coverage.add_trace(go.Bar(
                name='Demanda (3M)',
                x=top_stations['station_name'],
                y=top_stations['demand_3m'],
                marker_color='lightblue'
            ))

            fig_coverage.add_trace(go.Bar(
                name='Oferta (3M)',
                x=top_stations['station_name'],
                y=top_stations['supply_3m'],
                marker_color='darkblue'
            ))

            fig_coverage.update_layout(
                title="Top 10 Estaciones: Demanda vs Oferta (3 Meses)",
                xaxis_title="Estación",
                yaxis_title="Alquileres Totales",
                barmode='group',
                height=400
            )

            st.plotly_chart(fig_coverage, use_container_width=True)

            # Map Visualization with Movement Arrows
            st.markdown("---")
            st.subheader("🗺️ Mapa de Movimientos de Flota")

            if moves:
                # Create map centered on Spain
                m_moves = folium.Map(location=[40.0, -3.7], zoom_start=6)

                # Add all stations as markers
                for _, station in st.session_state.stations.iterrows():
                    folium.CircleMarker(
                        location=[station["lat"], station["lon"]],
                        radius=5,
                        color="gray",
                        fill=True,
                        fillColor="lightgray",
                        fillOpacity=0.6,
                        popup=station["station_name"]
                    ).add_to(m_moves)

                # Add arrows for movements
                for move in moves:
                    from_station = st.session_state.stations[
                        st.session_state.stations["station_id"] == move["from_id"]
                    ].iloc[0]
                    to_station = st.session_state.stations[
                        st.session_state.stations["station_id"] == move["to_id"]
                    ].iloc[0]

                    from_loc = [from_station["lat"], from_station["lon"]]
                    to_loc = [to_station["lat"], to_station["lon"]]

                    # Arrow thickness based on number of cars
                    # Scale: 1-5 cars -> 2px, 6-10 -> 4px, 11+ -> 6px
                    weight = 2 if move["amount"] <= 5 else (4 if move["amount"] <= 10 else 6)

                    # Color intensity based on amount
                    if move["amount"] <= 5:
                        color = "#FF6B6B"  # Light red
                    elif move["amount"] <= 10:
                        color = "#FF4444"  # Medium red
                    else:
                        color = "#CC0000"  # Dark red

                    # Draw arrow using PolyLine
                    arrow = folium.PolyLine(
                        locations=[from_loc, to_loc],
                        color=color,
                        weight=weight,
                        opacity=0.8,
                        popup=f"<b>{move['from_name']}</b> → <b>{move['to_name']}</b><br>🚗 {move['amount']} coches<br>📏 {move['cost']:.1f} km"
                    )
                    arrow.add_to(m_moves)

                    # Add directional arrow using plugins.AntPath for animation
                    from folium import plugins
                    plugins.AntPath(
                        locations=[from_loc, to_loc],
                        color=color,
                        weight=weight,
                        opacity=0.6,
                        delay=800,
                        dash_array=[10, 20]
                    ).add_to(m_moves)

                    # Add destination marker (larger circle)
                    folium.CircleMarker(
                        location=to_loc,
                        radius=8,
                        color=color,
                        fill=True,
                        fillColor=color,
                        fillOpacity=0.8,
                        popup=f"<b>Destino:</b> {move['to_name']}<br>🚗 Recibe {move['amount']} coches"
                    ).add_to(m_moves)

                st_folium(m_moves, width=800, height=600)

                st.caption("🔴 Las flechas animadas indican movimientos de coches. El grosor representa la cantidad de coches trasladados.")
            else:
                st.info("No hay movimientos que visualizar (la distribución actual es óptima).")

        if st.button("🚀 Ejecutar Optimizador Estratégico", type="primary"):
            optimizer = FleetOptimizer(st.session_state.stations)
            with st.spinner("Resolviendo optimización a 3 meses..."):
                moves, res_df = optimizer.solve_relocation_longterm(current_dict, future_window, lambda_pen)

            st.session_state.opt_results = {
                "moves": moves,
                "res_df": res_df,
                "total_demand_agg": float(total_demand_agg),
                "start_date": start_date,
                "end_date": end_date,
                "lambda_pen": lambda_pen,
            }

if st.session_state.opt_results is not None:
    meta = st.session_state.opt_results
    st.caption(
        f"Mostrando resultados para ventana {meta['start_date'].strftime('%Y-%m-%d')} → {meta['end_date'].strftime('%Y-%m-%d')} (λ={meta['lambda_pen']:.1f})."
    )
    render_opt_results(meta["moves"], meta["res_df"], meta["total_demand_agg"])
