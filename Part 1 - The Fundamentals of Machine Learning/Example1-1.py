import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.linear_model

# Load data / Carregar dados

oecd_bli = pd.read_csv("oecd_bli_2015.csv", thousands=',')
gdp_per_carpita = pd.read_csv("gdp_per_capita.csv", thousands=',' ,delimiter='\t', encoding='latin1', na_values="n/a")

# Prepare data / preparar dados

country_stats = prepare_country_stats(oecd_bli, gdp_per_carpita)
x = np.c_[country_stats["GPD per capita"]]
y = np.c_[country_stats["Life satisfaction"]]

# Visualize the data / Visualizando os dados

country_stats.plot(kind='scatter', x='GPD per capita', y='Life satisfaction')
plt.show()

# Select linear model
model = sklearn.linear_model.LinearRegression()

# Train the model
model.fit(x, y)

# Make a prediction for Cyprus

X_new = [[22587]]
print(model.predict(X_new))