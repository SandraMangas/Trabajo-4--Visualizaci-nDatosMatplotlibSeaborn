# Trabajo 4: VisualizaciónDatosMatplotlibSeaborn
Trabajo 4: VisualizaciónDatosMatplotlibSeaborn
# -------------------------------
# ANÁLISIS DEL DATASET SUPERSTORE 2012
# -------------------------------

# 1.- Importar librerías
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configurar estilo general de Seaborn
sns.set_style('whitegrid')

# -------------------------------
# 2.- Cargar dataset
# -------------------------------
data_path = 'superstore_dataset2012.csv'  # ajustar si se encuentra en otra ruta
df = pd.read_csv(data_path)

# Exploración inicial
print(df.head())
print(df.info())
print(df.describe())
print(df.isnull().sum())

# -------------------------------
# 3.- Preprocesamiento
# -------------------------------
# Convertir columnas de fecha si existen
if 'Order Date' in df.columns:
    df['Order Date'] = pd.to_datetime(df['Order Date'])

# -------------------------------
# 4.- Visualización univariante con Matplotlib
# -------------------------------
plt.figure(figsize=(8,6))
plt.hist(df['Sales'], bins=30, color='skyblue', edgecolor='black')
plt.title('Distribución de Ventas', fontsize=14)
plt.xlabel('Ventas')
plt.ylabel('Frecuencia')
plt.show()
# Conclusión: La mayoría de ventas se concentran en valores bajos, con pocas ventas muy altas.

# -------------------------------
# 5.- Visualización univariante con Seaborn
# -------------------------------
plt.figure(figsize=(8,6))
sns.boxplot(x='Category', y='Sales', data=df, palette='pastel')
plt.title('Distribución de Ventas por Categoría', fontsize=14)
plt.xlabel('Categoría')
plt.ylabel('Ventas')
plt.show()
# Conclusión: Se pueden observar posibles outliers en la categoría Technology.

# -------------------------------
# 6.- Gráfico bivariante con Matplotlib
# -------------------------------
plt.figure(figsize=(8,6))
plt.scatter(df['Sales'], df['Profit'], alpha=0.6, color='purple')
plt.title('Ventas vs Beneficio', fontsize=14)
plt.xlabel('Ventas')
plt.ylabel('Beneficio')
plt.show()
# Conclusión: Existe una correlación positiva, aunque algunos pedidos generan pérdidas.

# -------------------------------
# 7.- Gráfico bivariante con Seaborn
# -------------------------------
plt.figure(figsize=(8,6))
sns.regplot(x='Sales', y='Profit', data=df, scatter_kws={'alpha':0.6}, line_kws={'color':'red'})
plt.title('Regresión Ventas vs Beneficio', fontsize=14)
plt.xlabel('Ventas')
plt.ylabel('Beneficio')
plt.show()
# Conclusión: La línea de regresión muestra la tendencia general de ganancias crecientes con ventas.

# -------------------------------
# 8.- Visualización multivariante con Seaborn
# -------------------------------
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Mapa de correlación de variables numéricas', fontsize=14)
plt.show()
# Conclusión: Ventas y Beneficio muestran correlación positiva alta; Ship Cost tiene poca correlación.

# Alternativa: pairplot
sns.pairplot(df[['Sales','Profit','Quantity','Discount']], kind='scatter', diag_kind='kde')
plt.show()

# -------------------------------
# 9.- Subplots con múltiples visualizaciones
# -------------------------------
fig, axs = plt.subplots(2, 2, figsize=(15,12))

# Histograma de Ventas
axs[0,0].hist(df['Sales'], bins=30, color='skyblue', edgecolor='black')
axs[0,0].set_title('Distribución de Ventas')

# Boxplot de Beneficio por Categoría
sns.boxplot(x='Category', y='Profit', data=df, palette='pastel', ax=axs[0,1])
axs[0,1].set_title('Beneficio por Categoría')

# Scatter Ventas vs Beneficio
axs[1,0].scatter(df['Sales'], df['Profit'], alpha=0.6, color='green')
axs[1,0].set_title('Ventas vs Beneficio')
axs[1,0].set_xlabel('Ventas')
axs[1,0].set_ylabel('Beneficio')

# Heatmap de correlación
sns.heatmap(df[['Sales','Profit','Quantity','Discount']].corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=axs[1,1])
axs[1,1].set_title('Correlación variables numéricas')

plt.tight_layout()
plt.suptitle('Análisis Superstore 2012 - Múltiples Visualizaciones', fontsize=16, y=1.02)
plt.show()

# -------------------------------
# 10.- Guardar una figura como imagen
# -------------------------------
fig.savefig('analisis_superstore_multivis.png', dpi=300)

# -------------------------------
# 11.- Conclusiones generales (comentarios)
# -------------------------------
# - La mayoría de ventas se concentran en valores bajos.
# - Algunas categorías presentan outliers de ventas y beneficios.
# - Ventas y Beneficio muestran correlación positiva.
# - Ship Cost y Discount tienen baja correlación con beneficio.
# - Los subplots permiten comparar distribuciones y relaciones de forma clara.
