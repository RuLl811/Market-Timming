import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.stats.api as sms
from statsmodels.compat import lzip
from statsmodels.stats.outliers_influence import variance_inflation_factor
pd.set_option( 'display.max_columns', None )

# Importo la base de precios
df = pd.read_excel('NEM.xlsx', sheet_name='Acciones')
print(df.columns)

# Regresión
# Henkriksson y Merton
# rp = a + b * (Rm) + c*(Rm) * d_+_merval + e
# Probar nivel de significancia de los valores
# Prueba de Hipotesis de las volatilidades y betas

# Treynor y Mazuy
# Variables independientes y dependiente
X = df[['Rm', 'd_+_merval']]
y = df['Rp']

# Agregar una constante para el término independiente
X = sm.add_constant(X)

# Crear y entrenar el modelo
model = sm.OLS(y, X).fit()

# Resultados del modelo
print(model.summary())

residuals = model.resid

# Prueba de Heterocedasticidad
name = ['Lagrange multiplier statistic', 'p-value', 'f-value', 'f p-value'] # No hay suficiente evidencia para rechazar la hipótesis nula de homocedasticidad al nivel de significancia del 5%
test = sms.het_breuschpagan(residuals, model.model.exog)
bp_test = lzip(name, test)
print( '\nPrueba de Heterocedasticidad (Breusch-Pagan):' )
print( bp_test )
if test[1] < 0.05:
    print( "Hay suficiente evidencia para rechazar la hipótesis nula de homocedasticidad." )
else:
    print( "No hay suficiente evidencia para rechazar la hipótesis nula de homocedasticidad." )

# Prueba de Normalidad de Jarque-Bera
from statsmodels.stats.stattools import jarque_bera
jb_test = jarque_bera(model.resid)
print(f'\nPrueba de Jarque-Bera: {jb_test}')
if jb_test[1] < 0.05:
    print("Hay suficiente evidencia para rechazar la hipótesis nula de normalidad en los residuos.")
else:
    print("No hay suficiente evidencia para rechazar la hipótesis nula de normalidad en los residuos.")

# Prueba de Hipótesis de Coeficientes
print('\nPrueba de Hipótesis de los Coeficientes')

rm = model.t_test( 'Rm = 0' )
D = model.t_test( 'd_+_merval = 0' )

print( rm )
print( D )
if rm.pvalue < 0.05:
    print( "El coeficiente del retorno del mercado es significativo al 5%." )
else:
    print( "El coeficiente del retorno del mercado no es significativo al 5%." )
if D.pvalue < 0.05:
    print( "El coeficiente del término interactivo es significativo al 5%." )
else:
    print( "El coeficiente del término interactivo no es significativo al 5%." )

# Prueba de Multicolinealidad
print('\nPrueba de Multicolinealidad (VIF):')
vif = pd.DataFrame()
vif['Variable'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
print(vif)
if vif['VIF'].max() > 10:
    print("Existe evidencia de multicolinealidad significativa.\n")
else:
    print("No existe evidencia de multicolinealidad significativa.\n")

# Prueba de Autocorrelación de Durbin-Watson
dw_test = sm.stats.durbin_watson(model.resid)
print(f'Prueba de Autocorrelación de Durbin-Watson: {dw_test}')
if dw_test < 1.5 or dw_test > 2.5:
    print("Existe evidencia de autocorrelación en los residuos.\n")
else:
    print("No existe evidencia de autocorrelación en los residuos.\n")

# Graficos
# Rolling Beta
window_size = 12
def calculate_rolling_betas(X, y, window_size):
    rolling_betas = {'Beta2': []}
    for start in range(len(y) - window_size + 1):
        end = start + window_size
        X_window = X.iloc[start:end]
        y_window = y.iloc[start:end]
        model = sm.OLS(y_window, X_window).fit()
        rolling_betas['Beta2'].append(model.params['d_+_merval'])
    return pd.DataFrame(rolling_betas, index=y.index[window_size-1:])

# Calcular los rolling betas
rolling_betas = calculate_rolling_betas(X, y, window_size)

# Graficar los rolling betas
plt.figure(figsize=(12, 6))
plt.plot(rolling_betas['Beta2'], label='Rolling Beta 2 (dummy)')
plt.xlabel('Fecha')
plt.ylabel('Beta')
plt.title('Rolling Beta2 - Coef Mkt Timming')
plt.legend()
plt.show()
# Visualización de Datos
'''
#sns.pairplot(df[['Rp', 'Rm', 'Rm2']])
plt.show()
'''
# QQ plot y distribución de los residuos.
'''
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
sns.histplot(residuals, kde=True, ax=ax[0])
ax[0].set_title('Distribución de Residuos')
sm.qqplot(residuals, line='45', ax=ax[1])
ax[1].set_title('Q-Q Plot de Residuos') # No siguen una dist normal
plt.show()
'''
# Homocedasticidad
'''
plt.scatter(model.fittedvalues, residuals)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Valores Ajustados')
plt.ylabel('Residuos')
plt.title('Valores Ajustados vs Residuos') # hay cierta dispersión mayor en valores ajustados más altos, lo que puede indicar heterocedasticidad.
plt.show()
'''

