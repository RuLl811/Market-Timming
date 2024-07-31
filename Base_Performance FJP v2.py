import pandas as pd
import numpy as np
#pd.set_option( 'display.max_columns', None )

# Importo la base de precios
df = pd.read_excel('Base.xlsx')

# Lectura de Benchmark
benchmark = pd.read_excel( "Benchmark informe.xls", skiprows=5, header=None)

# Elimina las columnas no necesarias
columnas_a_omitir = (
        benchmark.select_dtypes( include='datetime64' ).columns.tolist()[1:] +
        benchmark.columns[benchmark.isna().all()].tolist() +
        [benchmark.columns[0]])

columnas_a_mantener = [columna for columna in benchmark.columns if columna not in columnas_a_omitir]
benchmark = benchmark[columnas_a_mantener]

# Renombrar las columnas del DataFrame
nombres_columnas = [
    'fecha', 'MERVAL Index', 'bna', 'dti', 'ccl', 'IBOV $', 'INDU Index', 'SX5P Index', 'BADLARPP Index', 'BADLAR',
    'ARDRT30P Index', 'EUR Curncy', 'BRL Curncy', 'GT2 Govt', 'ACERCER Index', 'GOLDS Comdty', 'IBOV Index',
    'IBOV USD', 'USIBOV Equity', 'IAMCCOPP Index', 'IAMCCODD Index', 'ALRECNC AR Equity', 'BWMING Index', 'BWAGRI Index',
    'BWOILP Index', 'Benchmark RRNN', 'Benchmark RRNN ARS', 'IAMCGRAD Index', 'Benchmark MIX',
    'MAR Index', 'MAR Index USD', 'IAMCLAPP Index', 'IAMCCOPP Index2', 'IAMCLADD Index', 'IAMCCODD Index2',
    'IAMCCODD Index ARS', 'IGSB US Equity', 'ARS BNAG Curncy', '.FXIMPL Index', 'EURUSD Curncy', 'IAMCCODP Index',
    'IAMCLADP Index', 'IAMCCOPD Index', 'IAMCLAPD Index', 'LQD US Equity', 'IEF US Equity', 'PFF US Equity',
    '.BWOILP en AR$', 'Benchmark RRNN NEW', 'Benchmark Mercosur NEW', '1784AHP Equity', 'ARDRARPP Index',
    'IAMCGRAP Index', 'ILF Equity', 'ILF Equity ARS', 'CER']

benchmark.columns = nombres_columnas
benchmark = benchmark[['fecha', 'MERVAL Index', 'BADLARPP Index', 'IAMCCOPP Index2', 'IAMCCODD Index2', 'bna']]
base = pd.merge(df, benchmark, on='fecha')

# Duplico la base para trabajar sobre NEM
nem = base


def calcular_badlar_ret_acum(df):
    df['badlar_ret_acum'] = 100.0
    for i in range(1, len(df)):
        days_diff = (df.fecha.iloc[i] - df.fecha.iloc[i-1]).days
        df.loc[df.index[i], 'badlar_ret_acum'] = df['badlar_ret_acum'].iloc[i-1] * \
                                                 (1 + df['BADLARPP Index'].iloc[i] * 30 / 36500) ** (days_diff / 30)
    return df

nem = calcular_badlar_ret_acum(nem)

# Calculo de retornos brutos
def calcular_retornos_alpha(df, columnas_alpha):
    for columna in columnas_alpha:
        df[f'ret_{columna}'] = np.log( df[columna] / df[columna].shift(1) ) + df[f'fee_{columna.split( "_" )[1]}']
    return df

columnas_alpha = ['alpha_acciones', 'alpha_mega', 'alpha_rcp', 'alpha_cobertura', 'alpha_rp']
columnas_benchmark = ['benchmark_rv', 'benchmark_rcp', 'benchmark_cobertura', 'benchmark_rp',
                      'merval', 'IAMC_COPP', 'IAMC_CODD', 'bna'] # , 'alpha_rcd', 'benchmark_rcd'
nem = calcular_retornos_alpha(nem, columnas_alpha)

# Calculo de retornos de índices
indices = {
    'ret_merval': 'MERVAL Index',
    'ret_badlar': 'badlar_ret_acum',
    'ret_IAMC_COPP': 'IAMCCOPP Index2',
    'ret_IAMC_CODD': 'IAMCCODD Index2',
    'ret_bna': 'bna'
}

for ret, idx in indices.items():
    nem[ret] = np.log(nem[idx] / nem[idx].shift(1) )

# Lectura del principal
principal = pd.read_csv( 'Principal_concatenado.csv', sep=',', low_memory=False )

# Lista de clases para los benchmarks
clases_bench = {
    'rv': [35, 41, 142, 216, 304, 343, 381, 384, 662, 683, 772, 814, 821, 963, 1038, 1193, 1243, 1400, 1843],
    'rcp': [670, 636, 831, 684, 1242, 1000, 1818, 682, 765, 565, 305, 2452, 1158, 1186, 624, 1683, 101, 951, 1445],
    #'rcd': [3363, 1625, 1561, 1734, 1593, 1590, 1759, 1427],
    'cobertura': [1085, 995, 3342, 1040, 1007, 1249, 738, 1104, 1094, 763, 800, 845, 1166],
    'rp': [1114, 714, 1010, 728, 735, 934, 717, 786, 1019, 994, 895, 731, 967, 1047, 73, 711, 3252, 719, 810, 282]
}

# Asegurarse de que la columna 'fecha' es de tipo datetime
principal['fecha'] = pd.to_datetime( principal['fecha'] )
base['fecha'] = pd.to_datetime( base['fecha'] )

# Obtener las fechas del archivo base
fechas_base = base['fecha'].unique()

# Función para filtrar y calcular rendimientos y AUM mensuales usando las fechas del archivo base
def calcular_rendimientos_aum_mensuales(df, clases, base_dates):
    resultados = {}
    for key, clase in clases.items():
        bench = df[df['clase_id'].isin( clase )]

        # Filtrar los datos para los últimos días de cada mes usando las fechas del archivo base
        bench = bench[bench['fecha'].isin( base_dates )]

        # Calcular rendimientos mensuales
        rendimientos = bench.pivot_table( index='fecha', columns='clase_id', values='compute_0013' ).pct_change(
            fill_method=None )
        rendimientos.replace( [np.inf, -np.inf], np.nan, inplace=True )
        rendimientos.fillna( 0, inplace=True )
        rendimientos = rendimientos.iloc[1:]

        # Calcular AUM mensual
        aum = bench.pivot_table( index='fecha', columns='clase_id', values='patrimonio', aggfunc='sum', fill_value=0 )
        aum.fillna( 0, inplace=True )
        aum = aum.iloc[1:]

        resultados[key] = (rendimientos, aum)
    return resultados

resultados = calcular_rendimientos_aum_mensuales( principal, clases_bench, fechas_base )

# Calculo de rendimiento ponderado del benchmark
def calcular_rto_benchmark(resultados):
    rto_benchmarks = {}
    for key, (rendimientos, aum) in resultados.items():
        # Asegurar que los DataFrames tienen el mismo índice
        common_index = rendimientos.index.intersection( aum.index )
        rendimientos = rendimientos.loc[common_index]
        aum = aum.loc[common_index]

        # Calcular el rendimiento ponderado por AUM
        rto_benchmark = (rendimientos * aum).sum( axis=1 ) / aum.sum( axis=1 )
        rto_benchmark.name = f'ret_benchmark_{key}'
        rto_benchmark = rto_benchmark.reset_index()
        rto_benchmark.rename( columns={'index': 'fecha'}, inplace=True )
        rto_benchmark['fecha'] = pd.to_datetime( rto_benchmark['fecha'] )
        rto_benchmarks[key] = rto_benchmark
    return rto_benchmarks

rto_benchmarks = calcular_rto_benchmark( resultados )

# Merge de benchmarks
for key, rto_benchmark in rto_benchmarks.items():
    nem = pd.merge(nem, rto_benchmark, on='fecha')

# Calculo de risk premium
# Calculo de risk premium
def calcular_risk_premium(df, columnas_alpha, columnas_benchmark):
    for columna in columnas_alpha:
        df[f'rp_{columna}'] = df[f'ret_{columna}'] - df['ret_badlar']
    for columna in columnas_benchmark:
        df[f'rp_{columna}'] = df[f'ret_{columna}'] - df['ret_badlar']
    return df

nem = calcular_risk_premium(nem, columnas_alpha, columnas_benchmark)


# Calculo de dummies y producto con risk premium
def calcular_dummies(df, columnas_benchmark):
    for key in columnas_benchmark:
        if f'rp_{key}' in df.columns:  # Verificar si la columna existe
            df[f'd_+_{key}'] = np.where(df[f'rp_{key}'] > 0, 1, 0)
            df[f'd_-_{key}'] = np.where(df[f'rp_{key}'] < 0, 1, 0)
            df[f'{key}_+'] = df[f'd_+_{key}'] * df[f'rp_{key}']
            df[f'{key}_-'] = df[f'd_-_{key}'] * df[f'rp_{key}']
            df[f'{key}_+'] = df[f'{key}_+'].replace(-0, 0)
    return df

nem = calcular_dummies(nem, columnas_benchmark)
nem['d_+_ret_badlar'] = np.where(nem['ret_badlar'] > 0, 1, 0)
nem['d_-_ret_badlar'] = np.where(nem['ret_badlar'] < 0, 1, 0)
nem['ret_badlar_+'] = nem['d_+_ret_badlar'] * nem['ret_badlar']
nem['ret_badlar_-'] = nem['d_-_ret_badlar'] * nem['ret_badlar']
nem['rp_badlar'] = nem['ret_badlar']

# Filtrar y Preparar DataFrames para Diferentes Estrategias
def preparar_df_estrategia(df, columnas, estrategia, nombre_columna_estrategia, bench, fondo):
    df_estrategia = df[columnas].dropna()

    df_estrategia[f'estrat_{estrategia}_badlar'] = (
        (df_estrategia[nombre_columna_estrategia] / df_estrategia['badlar_ret_acum']) -
        (df_estrategia[nombre_columna_estrategia] / df_estrategia['badlar_ret_acum']).rolling(window=10).mean()
    )
    df_estrategia = df_estrategia.dropna().reset_index(drop=True)

    # Estrategia 0.8/1.2
    df_estrategia['Beta 0,8/1,2'] = np.nan
    for i in range(1, len(df_estrategia)):
        if df_estrategia.loc[i - 1, f'estrat_{estrategia}_badlar'] <= 0:
            df_estrategia.loc[i, 'Beta 0,8/1,2'] = 0.8 * df_estrategia.loc[i, f'ret_{estrategia}']
        else:
            df_estrategia.loc[i, 'Beta 0,8/1,2'] = 1.2 * df_estrategia.loc[i, f'ret_{estrategia}']
        df_estrategia.loc[i, 'Beta 0,8/1,2'] -= df_estrategia.loc[i, 'ret_badlar']

    # Estrategia 0/1
    df_estrategia['Beta 0/1'] = np.nan
    for i in range(1, len(df_estrategia)):
        if df_estrategia.loc[i - 1, f'estrat_{estrategia}_badlar'] <= 0:
            df_estrategia.loc[i, 'Beta 0/1'] = df_estrategia.loc[i, 'ret_badlar']
        else:
            df_estrategia.loc[i, 'Beta 0/1'] = df_estrategia.loc[i, f'ret_{estrategia}']
        df_estrategia.loc[i, 'Beta 0/1'] -= df_estrategia.loc[i, 'ret_badlar']

        # Verificar si las columnas existen antes de renombrar
    rename_dict = {
        f'rp_alpha_{fondo}': 'Rp',
        f'rp_{estrategia}': 'Rm',
        f'{estrategia}_+': 'RmD_+',
        f'{estrategia}_-': 'RmD_-',
        'Beta 0,8/1,2': 'Est',
        'Beta 0/1': 'Est_2',
        f'rp_benchmark_{bench}': 'RmB',
        f'benchmark_{bench}_+': 'RmBD_+',
        f'benchmark_{bench}_-': 'RmBD_-'
    }

    rename_dict = {k: v for k, v in rename_dict.items() if k in df_estrategia.columns}
    df_estrategia.rename(columns=rename_dict, inplace=True)

    # Verificar si las columnas existen antes de eliminar
    drop_columns = [
        nombre_columna_estrategia, f'ret_{estrategia}', 'ret_badlar', 'badlar_ret_acum',
        f'estrat_{estrategia}_badlar', f'ret_benchmark_{bench}'
    ]

    drop_columns = [col for col in drop_columns if col in df_estrategia.columns]
    df_estrategia = df_estrategia.drop(columns=drop_columns).dropna()

    if 'Rm' in df_estrategia.columns:
        df_estrategia['Rm2'] = df_estrategia['Rm'] ** 2
    if 'RmB' in df_estrategia.columns:
        df_estrategia['RmB2'] = df_estrategia['RmB'] ** 2

    return df_estrategia


# Columnas específicas para cada estrategia
columnas_acciones = ['fecha', 'MERVAL Index', 'ret_merval', 'ret_badlar', 'ret_benchmark_rv', 'badlar_ret_acum',
                     'rp_alpha_acciones', 'rp_merval', 'rp_benchmark_rv', 'd_+_merval', 'd_-_merval',
                     'd_+_benchmark_rv', 'd_-_benchmark_rv', 'merval_+', 'merval_-', 'benchmark_rv_+', 'benchmark_rv_-']

columnas_mega = ['fecha', 'MERVAL Index', 'ret_merval', 'ret_badlar', 'ret_benchmark_rv', 'badlar_ret_acum',
                 'rp_alpha_mega', 'rp_merval', 'rp_benchmark_rv', 'd_+_merval', 'd_-_merval', 'd_+_benchmark_rv',
                 'd_-_benchmark_rv', 'merval_+', 'merval_-', 'benchmark_rv_+', 'benchmark_rv_-']

columnas_rcp = ['fecha', 'IAMCCOPP Index2', 'ret_IAMC_COPP', 'ret_badlar', 'ret_benchmark_rcp', 'badlar_ret_acum',
                'rp_alpha_rcp', 'rp_IAMC_COPP', 'rp_benchmark_rcp', 'd_+_benchmark_rcp',
                'd_-_benchmark_rcp', 'd_+_IAMC_COPP', 'd_-_IAMC_COPP', 'IAMC_COPP_+', 'IAMC_COPP_-',
                'benchmark_rcp_+', 'benchmark_rcp_-']
'''
columnas_rcd = ['fecha', 'IAMCCODD Index2', 'ret_IAMC_CODD', 'ret_badlar', 'ret_benchmark_rcd',
                'badlar_ret_acum', 'rp_alpha_rcd', 'rp_IAMC_CODD', 'rp_benchmark_rcd', 'd_+_benchmark_rcd',
                'd_-_benchmark_rcd', 'd_+_IAMC_CODD', 'd_-_IAMC_CODD', 'IAMC_CODD_+', 'IAMC_CODD_-',
                'benchmark_rcd_+', 'benchmark_rcd_-']
'''
columnas_cobertura = ['fecha', 'bna', 'ret_bna', 'ret_badlar', 'ret_benchmark_cobertura', 'badlar_ret_acum',
                      'rp_alpha_cobertura', 'rp_bna', 'rp_benchmark_cobertura', 'd_+_benchmark_cobertura',
                      'd_-_benchmark_cobertura', 'd_+_bna', 'd_-_bna', 'bna_+', 'bna_-',
                      'benchmark_cobertura_+', 'benchmark_cobertura_-']

columnas_rp = ['fecha', 'ret_badlar', 'ret_benchmark_rp', 'badlar_ret_acum', 'rp_alpha_rp',
               'rp_benchmark_rp', 'd_+_benchmark_rp', 'd_-_benchmark_rp', 'd_+_ret_badlar', 'd_-_ret_badlar',
               'ret_badlar_+', 'ret_badlar_-', 'benchmark_rp_+', 'benchmark_rp_-']


# Crear DataFrames para cada estrategia
nem_acciones = preparar_df_estrategia(nem, columnas_acciones, 'merval', 'MERVAL Index', 'rv', 'acciones')
nem_mega = preparar_df_estrategia(nem, columnas_mega, 'merval', 'MERVAL Index', 'rv', 'mega')
nem_rcp = preparar_df_estrategia(nem, columnas_rcp, 'IAMC_COPP', 'IAMCCOPP Index2', 'rcp', 'rcp')
#nem_rcd = preparar_df_estrategia(nem, columnas_rcd, 'IAMC_CODD', 'IAMCCODD Index2', 'rcd', 'rcd')
nem_cobertura = preparar_df_estrategia(nem, columnas_cobertura, 'bna', 'bna', 'cobertura', 'cobertura')
nem_rp = preparar_df_estrategia(nem, columnas_rp, 'badlar', 'ret_badlar', 'rp', 'rp')


# Metricas de riesgos y retornos

def calcular_metricas(df, nombre):
    # Asegurarse de que la columna de fecha es de tipo datetime
    df['fecha'] = pd.to_datetime(df['fecha'])
    df.set_index('fecha', inplace=True)

    # Definir las columnas a calcular
    columns = ['Rp', 'Rm', 'RmB']

    # Crear un DataFrame vacío para almacenar los resultados anuales
    years = df.index.year.unique()
    metricas = pd.DataFrame(index=years, columns=[f'{nombre}_Retorno', f'{nombre}_Desvio', f'{nombre}_Sharpe',
                                                 f'{nombre}_RM_Retorno', f'{nombre}_RM_Desvio', f'{nombre}_RM_Sharpe',
                                                 f'{nombre}_Competencia_Retorno', f'{nombre}_Competencia_Desvio',
                                                 f'{nombre}_Competencia_Sharpe'])

    # Calcular los retornos anuales
    annual_returns = df[columns].resample('YE').apply(lambda x: (x + 1).prod() - 1)

    # Calcular las desviaciones anuales
    annual_std = df[columns].resample('YE').std() * np.sqrt(12)

    # Calcular el Sharpe ratio (usando una tasa libre de riesgo de 0 para simplificar)
    annual_sharpe = annual_returns / annual_std

    # Ajustar los índices para que solo contengan los años
    annual_returns.index = annual_returns.index.year
    annual_std.index = annual_std.index.year
    annual_sharpe.index = annual_sharpe.index.year

    # Asignar los resultados a las columnas correspondientes en el DataFrame de métricas
    metricas[f'{nombre}_Retorno'] = annual_returns['Rp'] * 100
    metricas[f'{nombre}_Desvio'] = annual_std['Rp'] * 100
    metricas[f'{nombre}_Sharpe'] = annual_sharpe['Rp']

    metricas[f'{nombre}_RM_Retorno'] = annual_returns['Rm'] * 100
    metricas[f'{nombre}_RM_Desvio'] = annual_std['Rm'] * 100
    metricas[f'{nombre}_RM_Sharpe'] = annual_sharpe['Rm']

    metricas[f'{nombre}_Competencia_Retorno'] = annual_returns['RmB'] * 100
    metricas[f'{nombre}_Competencia_Desvio'] = annual_std['RmB'] * 100
    metricas[f'{nombre}_Competencia_Sharpe'] = annual_sharpe['RmB']

    return metricas

# Calcular las métricas para cada DataFrame
metricas_acciones = calcular_metricas(nem_acciones, 'Acciones')
metricas_mega = calcular_metricas(nem_mega, 'Mega')
metricas_rcp = calcular_metricas(nem_rcp, 'RCP')
#metricas_rcd = calcular_metricas(nem_rcd, 'RCD')
metricas_cobertura = calcular_metricas(nem_cobertura, 'Cobertura')

# Exportar los DataFrames
writer = pd.ExcelWriter( 'NEM.xlsx', engine='xlsxwriter' )
nem.to_excel(writer, sheet_name='Base_Nem', index=False)
nem_acciones.to_excel(writer, sheet_name='Acciones', index=True)
nem_mega.to_excel(writer, sheet_name='Mega', index=True)
nem_rcp.to_excel(writer, sheet_name='RCP', index=True)
#nem_rcd.to_excel(writer, sheet_name='RCD', index=True)
nem_cobertura.to_excel(writer, sheet_name='Cobertura', index=True)
nem_rp.to_excel(writer, sheet_name='RP', index=False)

# Exportación de metricas de riesgo
metricas_acciones.to_excel(writer, sheet_name='Metricas_riesgo', index=True, startcol=0, startrow=0)
metricas_mega.to_excel(writer, sheet_name='Metricas_riesgo', index=True, startcol=0, startrow=15)
metricas_rcp.to_excel(writer, sheet_name='Metricas_riesgo', index=True, startcol=0, startrow=30)
#metricas_rcd.to_excel(writer, sheet_name='Metricas_riesgo', index=True, startcol=0, startrow=30)
metricas_cobertura.to_excel(writer, sheet_name='Metricas_riesgo', index=True, startcol=0, startrow=45)
writer.close()
