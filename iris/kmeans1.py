import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
import random
import math
import numpy as np


virginica = pd.read_csv('virginica.txt', names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class'])
versicolor = pd.read_csv('versicolor.txt', names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class'])
setosa = pd.read_csv('setosa.txt', names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class'])

plt.suptitle('Sepal Length x Sepal Width', fontsize=18)
plt.xlabel('Sepal Length (cm)', fontsize=16)
plt.ylabel('Sepal Width (cm)', fontsize=16)
plt.scatter(virginica['sepal_length'], virginica['sepal_width'], alpha=0.5, c='#006400') #dark green
plt.scatter(versicolor['sepal_length'], versicolor['sepal_width'], alpha=0.5, c='#FF8C00') #dark orange
plt.scatter(setosa['sepal_length'], setosa['sepal_width'], alpha=0.5, c='#FF69B4') #hot pink
plt.show()

plt.suptitle('Petal Length x Petal Width', fontsize=18)
plt.xlabel('Petal Length (cm)', fontsize=16)
plt.ylabel('Petal Width (cm)', fontsize=16)
plt.scatter(virginica['petal_length'], virginica['petal_width'], alpha=0.5, c='#006400') #dark green
plt.scatter(versicolor['petal_length'], versicolor['petal_width'], alpha=0.5, c='#FF8C00') #dark orange
plt.scatter(setosa['petal_length'], setosa['petal_width'], alpha=0.5, c='#FF69B4') #hot pink
plt.show()

plt.suptitle('Petal Length x Sepal Width', fontsize=18)
plt.xlabel('Petal Length (cm)', fontsize=16)
plt.ylabel('Sepal Width (cm)', fontsize=16)
plt.scatter(virginica['petal_length'], virginica['sepal_width'], alpha=0.5, c='#006400') #dark green
plt.scatter(versicolor['petal_length'], versicolor['sepal_width'], alpha=0.5, c='#FF8C00') #dark orange
plt.scatter(setosa['petal_length'], setosa['sepal_width'], alpha=0.5, c='#FF69B4') #hot pink
plt.show()

plt.suptitle('Petal Length x Sepal Length', fontsize=18)
plt.xlabel('Petal Length (cm)', fontsize=16)
plt.ylabel('Sepal Length (cm)', fontsize=16)
plt.scatter(virginica['petal_length'], virginica['sepal_length'], alpha=0.5, c='#006400') #dark green
plt.scatter(versicolor['petal_length'], versicolor['sepal_length'], alpha=0.5, c='#FF8C00') #dark orange
plt.scatter(setosa['petal_length'], setosa['sepal_length'], alpha=0.5, c='#FF69B4') #hot pink
plt.show()

plt.suptitle('Sepal Length x Petal Width', fontsize=18)
plt.xlabel('Sepal Length (cm)', fontsize=16)
plt.ylabel('Petal Width (cm)', fontsize=16)
plt.scatter(virginica['sepal_length'], virginica['petal_width'], alpha=0.5, c='#006400') #dark green
plt.scatter(versicolor['sepal_length'], versicolor['petal_width'], alpha=0.5, c='#FF8C00') #dark orange
plt.scatter(setosa['sepal_length'], setosa['petal_width'], alpha=0.5, c='#FF69B4') #hot pink
plt.show()

plt.suptitle('Sepal Width x Petal Width', fontsize=18)
plt.xlabel('Sepal Length (cm)', fontsize=16)
plt.ylabel('Petal Width (cm)', fontsize=16)
plt.scatter(virginica['sepal_width'], virginica['petal_width'], alpha=0.5, c='#006400') #dark green
plt.scatter(versicolor['sepal_width'], versicolor['petal_width'], alpha=0.5, c='#FF8C00') #dark orange
plt.scatter(setosa['sepal_width'], setosa['petal_width'], alpha=0.5, c='#FF69B4') #hot pink
plt.show()

