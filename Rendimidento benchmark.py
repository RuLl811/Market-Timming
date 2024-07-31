import pandas as pd
import numpy as np

pd.set_option( 'display.max_columns', None )

# Lectura del principal y del archivo base
principal = pd.read_csv( 'Principal_concatenado.csv', sep=',', low_memory=False )
base = pd.read_excel( 'Base.xlsx' )

# Lista de clases para los benchmarks
clases_bench = {
    'rv': [35, 41, 142, 216, 304, 343, 381, 384, 662, 683, 772, 814, 821, 963, 1038, 1193, 1243, 1400, 1843],
}

# Asegurarse de que la columna 'fecha' es de tipo datetime
principal['fecha'] = pd.to_datetime( principal['fecha'] )
base['fecha'] = pd.to_datetime( base['fecha'] )

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


# Obtener las fechas del archivo base
fechas_base = base['fecha'].unique()

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

# Exportar resultados a un archivo Excel en una sola hoja
with pd.ExcelWriter( 'bench_rv.xlsx', engine='openpyxl' ) as writer:
    start_row = 0
    for key, rto_benchmark in rto_benchmarks.items():
        rto_benchmark.to_excel( writer, sheet_name='Benchmark', startrow=start_row, index=False )
        start_row += len( rto_benchmark ) + 2  # Dejar una fila vacía entre cada conjunto de métricas

print( "Exportación a Excel completada." )
