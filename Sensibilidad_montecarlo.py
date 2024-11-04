'''
Simulación dinamico de escenarios de cambio de tasa de interes. Este codigo utiliza las tires y Durations de
los activos investment grade del fondo RFG.
El proposito es conocer cuales son las trayectorias del valor del portafolio dado un umbral de tasas manteniendo la
duration constante.
Los parametros iniciales son: 1000 simulaciones para llegar a la normalidad, 1.6 de duration incial (Mod Dur actual del RFG),
tolerancia de 5% en el desvio en el valor de la duration diaria del portafolio.
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Parámetros de simulación
num_simulations = 1000  # Reducido a 100 simulaciones para pruebas
target_duration = 1.6  # Duración objetivo del portafolio tras el rebalanceo
duration_tolerance = 0.10  # Tolerancia max  de desvio 10%

# Parámetro de frecuencia
frequency = 'daily'

# Ajustes de rango de cambio de tasa según la frecuencia seleccionada
if frequency == 'daily':
    min_rate_change = -0.04
    max_rate_change = 0.04
    num_periods = 252
elif frequency == 'monthly':
    min_rate_change = -0.1
    max_rate_change = 0.1
    num_periods = 12

# Datos de la cartera: Nombre, TIR (en %), Duración Modificada
data = pd.read_excel('data/base_tasas.xlsx')
portfolio_df = pd.DataFrame(data)

# Función de optimización de weight que cumple con la duración objetivo
'''
- objective: Minimizar el cuadrado de la desviación entre la dur. de la simulación y el objetivo
- initial_weights: Pesos iniciales para comenzar la optimización, arranca igual ponderado
- bounds: Límites de 0 a 1 para cada peso. 
- constraints Los pesos sumen 1
'''
def optimize_weights(durations, target_duration):
    n_assets = len(durations)

    # Función objetivo para minimizar la desviación de la duración objetivo

    def objective(weights):
        effective_duration = np.dot(weights, durations)
        return (effective_duration - target_duration) ** 2

    constraints = {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}  # Restricción de que los pesos sumen 1

    bounds = [(0, 1) for _ in range(n_assets)]  #Límite en los pesos (entre 0 y 1)

    initial_weights = np.full(n_assets, 1 / n_assets)  # Peso inicial igual para todos los activos

    result = minimize(objective, initial_weights, bounds=bounds, constraints=constraints)
    return result.x if result.success else initial_weights

# Función para calcular el impacto del cambio en tasa sobre el precio del portafolio con rebalanceo y validación de duración
def simulate_portfolio_sensitivity(portfolio_df, num_simulations, num_periods, min_rate_change, max_rate_change,
                                   target_duration, duration_tolerance):

    portfolio_df['TIR'] = portfolio_df['TIR'] / 100

    portfolio_values = np.zeros((num_simulations, num_periods + 1)) # Matriz de dimension: Cant. Simulaciones x Cant. Periodos
    initial_portfolio_value = 100  # Valor inicial del portafolio
    portfolio_values[:, 0] = initial_portfolio_value

    for i in range(num_simulations):
        for t in range(1, num_periods + 1):

            rate_change = np.random.uniform(min_rate_change, max_rate_change)

            current_value = portfolio_values[i, t - 1]  # valor del portafolio en el período anterior

            durations = portfolio_df['Modified Duration'].values
            weights = optimize_weights(durations, target_duration)

            # Calcular la duración efectiva del portafolio después de la optimización
            effective_duration = np.dot(weights, durations)

            # Verificar que la duración esté dentro del rango de tolerancia
            lower_bound = target_duration * (1 - duration_tolerance)
            upper_bound = target_duration * (1 + duration_tolerance)
            if effective_duration < lower_bound or effective_duration > upper_bound:
                print(f"La duración efectiva ({effective_duration:.2f}) se desvía del objetivo por fuera del tolareado "
                      f"en el período {t} de la simulación {i}. Duración objetivo: {target_duration}")

            # Calcular el impacto de los cambios de tasas sobre el portafolio en t-1
            rebalance_impact = sum(
                -weights[j] * durations[j] * rate_change * current_value for j in range(len(weights)))

            # Actualizar el valor del portafolio con el impacto ajustado
            portfolio_values[i, t] = portfolio_values[i, t - 1] + rebalance_impact

    return portfolio_values

# Ejecutar la simulación
portfolio_values = simulate_portfolio_sensitivity(portfolio_df, num_simulations, num_periods, min_rate_change,
                                                   max_rate_change, target_duration, duration_tolerance)

# Resultados
# VaR y CVaR en el último periodo simulado
final_portfolio_values = portfolio_values[:, -1]
var_95 = np.percentile(final_portfolio_values, 5)
cvar_95 = final_portfolio_values[final_portfolio_values <= var_95].mean()

# 2. Mostrar estadísticas
print(f"Valor en Riesgo (VaR) al 95%: {var_95:.2f}")
print(f"Valor Condicional en Riesgo (CVaR) al 95%: {cvar_95:.2f}")
print(f"Valor promedio del portafolio al final de la simulación: {final_portfolio_values.mean():.2f}")

# 3. Visualización de resultados
plt.figure(figsize=(10, 6))
plt.hist(final_portfolio_values, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
plt.axvline(var_95, color='red', linestyle='--', label=f'VaR al 95%: {var_95:.2f}')
plt.axvline(cvar_95, color='blue', linestyle='--', label=f'CVaR al 95%: {cvar_95:.2f}')
plt.title('Distribución de Valores Finales del Portafolio')
plt.xlabel('Valor del Portafolio')
plt.ylabel('Frecuencia')
plt.legend()
plt.show()

# Gráfico de líneas para simulaciones individuales
plt.figure(figsize=(14, 8))

num_lines = min(num_simulations, 1000)
colors = plt.cm.viridis(np.linspace(0, 1, num_lines))  # Colores degradados

# Graficar las simulaciones con líneas degradadas
for i in range(num_lines):
    plt.plot(portfolio_values[i], color=colors[i], linewidth=0.6, alpha=0.7)

# Añadir percentiles (5% y 95%)
percentile_5 = np.percentile(portfolio_values, 5, axis=0)
percentile_95 = np.percentile(portfolio_values, 95, axis=0)
plt.plot(percentile_5, color='red', linestyle='--', linewidth=1.5, label='Percentil 5%')
plt.plot(percentile_95, color='blue', linestyle='--', linewidth=1.5, label='Percentil 95%')

# Configuración adicional para el gráfico
plt.title('Simulaciones Monte Carlo', fontsize=16)
plt.xlabel('Período', fontsize=14)
plt.ylabel('Valor del Portafolio', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()
