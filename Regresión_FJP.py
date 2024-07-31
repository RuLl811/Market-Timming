import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option( 'display.max_columns', None )

# Importo la base de precios
df = pd.read_excel('NEM.xlsx', sheet_name='Acciones')
print(df.columns)

# Regresión

# Treynor y Mazuy
# rp = a + b * (Rm) + c*(Rm) ^ 2 + e
# Henkriksson y Merton
# rp = a + b * (Rm) + c*(Rm) * d_+_merval + e
# Probar nivel de significancia de los valores
# Prueba de Hipotesis de las volatilidades y betas

# Exploración y Limpieza de Datos
print(df.describe())

# Visualización de Datos
#sns.pairplot(df[['Rp', 'Rm', 'Rm2']])
#plt.show()

# Variables independientes y dependiente
X = df[['Rm', 'Rm2']]
y = df['Rp']

# Agregar una constante para el término independiente
X = sm.add_constant(X)

# Crear y entrenar el modelo
model = sm.OLS(y, X).fit()

# Resultados del modelo
print(model.summary())

# Visualización de Residuos

residuals = model.resid
'''
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
sns.histplot(residuals, kde=True, ax=ax[0])
ax[0].set_title('Distribución de Residuos')
sm.qqplot(residuals, line='45', ax=ax[1])
ax[1].set_title('Q-Q Plot de Residuos') # No siguen una dist normal
plt.show()
'''
# Prueba de Heterocedasticidad
import statsmodels.stats.api as sms
from statsmodels.compat import lzip
name = ['Lagrange multiplier statistic', 'p-value', 'f-value', 'f p-value'] # o hay suficiente evidencia para rechazar la hipótesis nula de homocedasticidad al nivel de significancia del 5%
test = sms.het_breuschpagan(residuals, model.model.exog)
print(lzip(name, test))

# Prueba de Normalidad de Jarque-Bera
from statsmodels.stats.stattools import jarque_bera
jb_test = jarque_bera(residuals)
print(f'Prueba de Jarque-Bera: {jb_test}')

# Pruebas de Supuestos del Modelo
# Homocedasticidad
'''
plt.scatter(model.fittedvalues, residuals)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Valores Ajustados')
plt.ylabel('Residuos')
plt.title('Valores Ajustados vs Residuos') # hay cierta dispersión mayor en valores ajustados más altos, lo que puede indicar heterocedasticidad.
plt.show()
'''
# Prueba de Hipótesis de Coeficientes
print('Prueba de Hipótesis de los Coeficientes')
print(model.t_test('Rm = 0'))
print(model.t_test('Rm2 = 0'))

# Evaluación del Modelo
mse = np.mean((y - model.fittedvalues) ** 2)
print(f'Error Cuadrático Medio (MSE): {mse}')

# Métricas Adicionales
r2 = model.rsquared
adj_r2 = model.rsquared_adj
print(f'R^2: {r2}')
print(f'R^2 Ajustado: {adj_r2}')

