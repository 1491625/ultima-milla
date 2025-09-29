# 🔍 Validación de Factibilidad: Dataset Sintético + ML + Optimización VRP

## 📋 **Objetivo de esta Fase**

Desarrollar y validar de forma autónoma los componentes fundamentales del sistema de ruteo de última milla para evaluar:
- **Realismo del dataset sintético** (evitar métricas ML artificialmente perfectas)
- **Factibilidad de la optimización** (convergencia y interpretabilidad)
- **Integración ML → Optimización** (flujo de datos coherente)
- **Insights pedagógicos** (resultados útiles para estudiantes)

**Criterio de éxito:** Generar evidencia suficiente para redefinir/ajustar el proyecto principal basándose en resultados empíricos.

---

## 🏗️ **Arquitectura del Sistema de Validación**

```
Dataset Sintético (50 clientes, 5 días) 
         ↓
RandomForest Predictor (tiempos de viaje + disponibilidad)
         ↓  
MINLP Optimizer (VRP con ventanas de tiempo)
         ↓
Dashboard de Análisis (métricas + visualizaciones)
```

---

## 📁 **Estructura del Proyecto**

```
vrp_validation/
├── README.md
├── requirements.txt
├── config/
│   └── validation_config.yaml
├── data/
│   ├── synthetic/           # Dataset generado
│   └── results/             # Resultados de validación
├── src/
│   ├── __init__.py
│   ├── data_generator.py    # Generación dataset real-scale
│   ├── ml_pipeline.py       # RandomForest + validación
│   ├── optimization_model.py # Formulación MINLP
│   └── analysis_dashboard.py # Dashboard análisis
├── notebooks/
│   └── validation_results.ipynb
└── tests/
    └── test_integration.py
```

---

## 🎯 **Componente 1: Generador de Dataset Real-Scale**

### **Especificación del Dataset:**
- **Escala:** 50 clientes, 3 vehículos, 5 días operativos
- **Área geográfica:** Grid urbano 15x15 km (ciudad media)
- **Complejidad:** Patrones temporales realistas + correlaciones imperfectas
- **Incertidumbre:** Variabilidad en tiempos de viaje (±20%) y disponibilidad cliente (70-95%)

### **Estructura de Datos:**

#### **customers.csv**
```python
{
    'customer_id': int,
    'lat': float, 'lon': float,        # Coordenadas realistas
    'demand_kg': float,                # 5-50 kg por entrega
    'service_time_min': float,         # 5-20 minutos
    'time_window_start': int,          # 8-18 horas
    'time_window_end': int,            # window_start + 2-6 horas  
    'availability_base_prob': float,   # 0.7-0.95
    'customer_type': str,              # 'residential', 'business', 'priority'
    'access_difficulty': float         # Factor que afecta service_time
}
```

#### **historical_deliveries.csv (para entrenamiento ML)**
```python
{
    'delivery_id': int,
    'customer_id': int,
    'date': datetime,
    'planned_arrival': datetime,
    'actual_arrival': datetime,
    'travel_time_min': float,          # Target para ML
    'customer_available': bool,        # Target para ML
    'weather': str,                    # 'sunny', 'rain', 'cloudy'
    'traffic_level': float,            # 1.0-2.5 multiplicador
    'day_of_week': int,
    'hour_of_day': int,
    'distance_km': float,
    'previous_stop_delay': float       # Retraso acumulado
}
```

#### **vehicles.csv**
```python
{
    'vehicle_id': int,
    'capacity_kg': float,              # 200-500 kg
    'cost_per_km': float,              # 0.8-1.2 EUR/km
    'max_working_hours': int,          # 8-10 horas
    'depot_lat': float, 'depot_lon': float
}
```

### **Generación de Realismo:**
- **Distribuciones asimétricas** (no perfectamente gaussianas)
- **Correlaciones con ruido:** distancia-tiempo con factor aleatorio ±15%
- **Patrones estacionales:** rush hours, días de semana vs weekend
- **Outliers controlados:** 5% de casos "difíciles" (cliente no disponible, tráfico extremo)

---

## 🤖 **Componente 2: Pipeline RandomForest**

### **Objetivo:** Predecir tiempos de viaje y disponibilidad del cliente

### **Features Engineering:**
```python
# Para predicción de tiempos de viaje
travel_features = [
    'distance_km',
    'hour_of_day', 'day_of_week',
    'weather_encoded',
    'traffic_level',
    'road_type_factor',
    'previous_stop_delay',
    'vehicle_type_encoded'
]

# Para predicción de disponibilidad cliente  
availability_features = [
    'customer_availability_base_prob',
    'hour_of_day', 'day_of_week',
    'customer_type_encoded', 
    'weather_encoded',
    'previous_failed_attempts',
    'time_since_last_delivery'
]
```

### **Implementación:**
```python
class VRPPredictor:
    def __init__(self):
        self.travel_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.availability_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=8, 
            random_state=42,
            class_weight='balanced'
        )
    
    def train_and_validate(self, data):
        """
        Entrenamiento con validación temporal estricta:
        - Training: días 1-20
        - Validation: días 21-25  
        - Test: días 26-30
        """
        pass
    
    def predict_with_uncertainty(self, X):
        """
        Retornar predicción + intervalo de confianza
        usando std de árboles en RandomForest
        """
        pass
```

### **Validaciones Específicas:**
- **Cross-validation temporal:** No aleatorio, respeta orden cronológico
- **Calibración probabilística:** Para predicciones de disponibilidad
- **Feature importance:** Análisis de qué variables son más predictivas
- **Residual analysis:** Detectar bias en predicciones

---

## ⚙️ **Componente 3: Formulación MINLP del VRP**

### **Formulación Matemática:**

#### **Conjuntos:**
- `C = {1,2,...,50}`: Clientes
- `V = {0,1,2,...,50}`: Nodos (0=depot)
- `K = {1,2,3}`: Vehículos  
- `T = {480,490,...,1080}`: Slots de tiempo (8:00-18:00 en minutos)

#### **Variables de Decisión:**
```python
# Variables binarias
x[i,j,k] = 1 si vehículo k va de i a j
s[i,k] = 1 si vehículo k sirve cliente i
z[i,t] = 1 si cliente i es visitado en slot t

# Variables continuas  
arrival_time[i,k] = momento de llegada del vehículo k al cliente i
load[i,k] = carga del vehículo k al salir del nodo i
```

#### **Parámetros (Predichos por ML):**
```python
travel_time[i,j] = tiempo predicho por RandomForest
availability_prob[i,t] = probabilidad cliente disponible en slot t
service_time[i] = tiempo de servicio en cliente i
demand[i] = demanda del cliente i
```

#### **Función Objetivo:**
```
minimize:
    Σ(i,j,k) cost_per_km[k] * distance[i,j] * x[i,j,k] +           # Coste transporte
    Σ(i,t) penalty_unavailable * (1-availability_prob[i,t]) * z[i,t] # Penalización riesgo
```

#### **Restricciones Principales:**
```python
# Cada cliente servido exactamente una vez
∀i∈C: Σ(k) s[i,k] = 1

# Conservación de flujo
∀k∈K, j∈V: Σ(i) x[i,j,k] = Σ(i) x[j,i,k]

# Ventanas de tiempo
∀i∈C: time_window_start[i] ≤ arrival_time[i,k] ≤ time_window_end[i]

# Precedencia temporal con tiempos ML
∀i,j∈V, k∈K: arrival_time[i,k] + service_time[i] + travel_time[i,j] 
              ≤ arrival_time[j,k] + M*(1-x[i,j,k])

# Capacidad vehículos
∀k∈K: Σ(i) demand[i] * s[i,k] ≤ capacity[k]

# Jornada laboral
∀k∈K: arrival_time[depot,k] ≤ max_working_hours[k] * 60
```

### **Implementación:**
```python
import pulp
from ortools.linear_solver import pywraplp

class VRPOptimizer:
    def __init__(self, solver='CBC'):
        self.solver_name = solver
        self.model = None
        self.solution = None
        
    def formulate_minlp(self, customers, vehicles, ml_predictions):
        """
        Crear modelo MINLP completo con predicciones ML
        """
        pass
    
    def solve_with_analysis(self, time_limit=600):
        """
        Resolver + análisis de sensibilidad automático
        """
        pass
    
    def extract_solution_insights(self):
        """
        Extraer insights interpretables:
        - Qué restricciones están activas
        - Utilización de vehículos  
        - Patterns en las rutas
        - Sensibilidad a parámetros clave
        """
        pass
```

---

## 📊 **Componente 4: Dashboard de Análisis**

### **Métricas de Validación a Reportar:**

#### **ML Performance:**
```python
ml_metrics = {
    # Predicción tiempos de viaje
    'travel_time_mae': float,           # Target: 10-25% del promedio
    'travel_time_r2': float,            # Target: 0.6-0.8 
    'travel_time_mape': float,
    
    # Predicción disponibilidad
    'availability_auc': float,          # Target: 0.7-0.85
    'availability_precision': float,
    'availability_recall': float,
    'calibration_score': float,         # Reliability diagram
    
    # Estabilidad temporal
    'temporal_stability': float,        # Performance días 21-25 vs 26-30
    'feature_importance_stability': dict
}
```

#### **Optimization Performance:**  
```python
optimization_metrics = {
    # Convergencia
    'solve_time_seconds': float,        # Target: <300s
    'optimality_gap': float,            # Target: <5%
    'solver_status': str,               # 'Optimal', 'Feasible', etc.
    
    # Calidad de solución
    'total_distance_km': float,
    'total_time_hours': float, 
    'vehicle_utilization': dict,        # Por vehículo
    'time_window_violations': int,
    
    # Interpretabilidad
    'active_constraints': dict,         # Cuáles están en el límite
    'shadow_prices': dict,              # Valor marginal restricciones
    'solution_robustness': float        # Sensibilidad a cambios ML
}
```

#### **Integration Health:**
```python
integration_metrics = {
    # Flujo ML → Opt
    'prediction_to_optimization_time': float,
    'parameter_consistency_checks': dict,
    'extreme_predictions_handled': bool,
    
    # Business insights
    'cost_vs_service_tradeoff': dict,
    'bottleneck_identification': list,
    'scalability_indicators': dict
}
```

### **Visualizaciones Automáticas:**
1. **Mapa de rutas** con clientes coloreados por disponibilidad predicha
2. **Distribución de métricas ML** vs benchmarks típicos  
3. **Análisis de sensibilidad** de la optimización
4. **Timeline de convergencia** del solver
5. **Feature importance** de modelos ML
6. **Correlation matrix** entre parámetros y KPIs de negocio

### **Implementación Dashboard:**
```python
import matplotlib.pyplot as plt
import seaborn as sns
import folium
import plotly.express as px

class ValidationDashboard:
    def __init__(self, ml_results, optimization_results, raw_data):
        self.ml_results = ml_results
        self.opt_results = optimization_results  
        self.data = raw_data
        
    def generate_full_report(self):
        """
        Generar reporte completo con:
        1. Executive summary de viabilidad
        2. Análisis detallado por componente
        3. Recomendaciones para proyecto principal
        4. Red flags identificados
        """
        pass
    
    def plot_ml_performance(self):
        """Visualizar métricas ML vs umbrales aceptables"""
        pass
        
    def plot_optimization_analysis(self):
        """Análisis de convergencia y interpretabilidad"""
        pass
        
    def plot_route_visualization(self):
        """Mapa interactivo con rutas optimizadas"""
        pass
```

---

## 🎯 **Criterios de Evaluación de la Validación**

### **🟢 Señales Positivas (Proceder con proyecto principal):**
- ML R² entre 0.6-0.8 (no artificialmente alto)
- Optimización converge en <5 minutos con gap <5%
- Soluciones muestran trade-offs interpretables
- Predicciones ML impactan significativamente en rutas óptimas
- Insights pedagógicos claros emergen del análisis

### **🟡 Señales de Alerta (Ajustar enfoque):**
- ML R² >0.9 (datos demasiado "perfectos")
- Optimización muy sensible a pequeños cambios en predicciones
- Soluciones difíciles de interpretar o contradictorias
- Tiempo de solve >10 minutos consistentemente

### **🔴 Red Flags (Rediseñar completamente):**
- ML colapsa en validación temporal
- Problemas de optimización infactibles
- Resultados técnicamente correctos pero pedagógicamente vacíos
- Integración ML→Opt introduce errores sistemáticos

---

## 📋 **Configuración para Claude Code**

### **requirements.txt**
```
pandas>=1.5.0
numpy>=1.21.0
scikit-learn>=1.1.0
matplotlib>=3.5.0
seaborn>=0.11.0
plotly>=5.10.0
folium>=0.12.0

# Optimization
pulp>=2.6.0
ortools>=9.4.0

# Development
pytest>=7.0.0
jupyter>=1.0.0
pyyaml>=6.0.0
```

### **Orden de Implementación para Claude Code:**
1. **Crear estructura de carpetas** según especificación
2. **Implementar data_generator.py** con dataset de 50 clientes realista
3. **Desarrollar ml_pipeline.py** con RandomForest y validación temporal
4. **Formular optimization_model.py** con MINLP completo
5. **Construir analysis_dashboard.py** con métricas y visualizaciones
6. **Integrar todos los componentes** en pipeline automatizado
7. **Generar notebook final** con análisis completo de resultados

### **Test de Integración:**
```python
def test_full_pipeline():
    """Test end-to-end del sistema de validación"""
    # Generar datos
    dataset = generate_synthetic_data(n_customers=50, n_days=5)
    
    # Entrenar ML
    ml_model = train_ml_pipeline(dataset)
    
    # Resolver optimización  
    solution = solve_vrp_minlp(dataset, ml_model.predictions)
    
    # Validar resultados
    assert solution.status == 'Optimal'
    assert ml_model.r2_score < 0.9  # No artificialmente perfecto
    assert solution.solve_time < 600  # Converge en tiempo razonable
```

---

## 🎯 **Output Esperado de esta Validación**

Al completar esta fase, deberás tener:

1. **Reporte ejecutivo** con recomendación clara: ¿Proceder, ajustar, o rediseñar?
2. **Dataset sintético validado** que produce ML/optimización realistas
3. **Baseline de performance** para componentes ML y optimización  
4. **Identificación de bottlenecks** y aspectos más pedagógicamente valiosos
5. **Roadmap ajustado** para el proyecto principal basado en evidencia empírica

**Este análisis será la base para decidir si el CLAUDE.md del proyecto principal necesita modificaciones en complejidad, enfoque, o estructura pedagógica.**
