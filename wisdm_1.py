# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 21:43:25 2023

@author: celia
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
#Cargo los datos
columns = ['user','activity','timestamp', 'x-axis', 'y-axis', 'z-axis']
df_har = pd.read_csv('WISDM_ar_v1.1/WISDM_ar_v1.1_raw.txt', header = None, names = columns)

#Quito los valores nulos
df_har=df_har.dropna()

#Si hago df_har.info veo que z es una variable object, la transformo a float
df_har['z-axis'] = df_har['z-axis'].str.replace(';', '')
df_har['z-axis'] = df_har['z-axis'].apply(lambda x:float(x))

#Quito los datos cuya timestamp es 0
df = df_har[df_har['timestamp'] != 0]

#Ordeno los datos en funcion del numero de usuario y el timestamp
df = df.sort_values(by = ['user', 'timestamp'], ignore_index=True)

#Para ver el numero de muestras por cada actividad
#uso df.activity.value_counts()

#Vemos como cada user por individual ha contribuido a cada actividad
sns.countplot(x = 'user',  hue = 'activity', data = df)
plt.title('Activities by Users')
plt.show()
#A pesar de que el peso de cada actividad en funcion del usuario
#es diferente, esto no afectara a l resultado final ya que el numero de muestras
#empleadas en este estudio es suficientemente grande y suponemos que todos los users son iguales??

#Vemos como los valores de x y z varian con el tiempo.
#Lo hacemos para el usuario 36 y para cada actividad

#Consideramos un subset de 400 muestras. Esto es equivalente a 
#20 segundos de actividad ya que la freccuencia de recogida de datos es de 
#20 hz (20 datos en 1 segundo)
activities = ['Walking', 'Jogging', 'Upstairs', 'Downstairs', 'Sitting', 'Standing']
for i in activities:
    data36 = df[(df['user'] == 36) & (df['activity'] == i)][:400]
    sns.lineplot(y = 'x-axis', x = 'timestamp', data = data36)
    sns.lineplot(y = 'y-axis', x = 'timestamp', data = data36)
    sns.lineplot(y = 'z-axis', x = 'timestamp', data = data36)
    plt.legend(['x-axis', 'y-axis', 'z-axis'])
    plt.ylabel(i)
    plt.title(i, fontsize = 15)
    plt.show()
    
#Podemos observar que las señales dinamicas tienen un comportamiento periodico
#mientras que las actividades estaticas como sentarse o tumbarse son más o menos constantes

#ENTRENAMIENTO DEL MODELO CON LSTM
#Los modelos LSTM esperan secuencias de un tamaño fijo como datos de entrenamiento.
#Cada secuencia generada contiene 50 muestras que son 2.5 segundos de actividad.

random_seed = 42   
n_time_steps = 50 
n_features = 3 
step = 10 
n_classes = 6 
n_epochs = 50       
batch_size = 1024   
learning_rate = 0.0025
l2_loss = 0.0015
segments = []
labels = []
#En vez de decir de 0 hasta el final hace de 0 hasta el final -el tiempo de steps
#y con un paso de step ???
for i in range(0,  df.shape[0]- n_time_steps, step):  

    xs = df['x-axis'].values[i: i + 50]
    ys = df['y-axis'].values[i: i + 50]
    zs = df['z-axis'].values[i: i + 50]
    label = stats.mode(df['activity'][i: i + 50])[0][0]
    segments.append([xs, ys, zs])
    labels.append(label)

#reshape the segments which is (list of arrays) to a list
reshaped_segments = np.asarray(segments, dtype= np.float32).reshape(-1, n_time_steps, n_features)
labels = np.asarray(pd.get_dummies(labels), dtype = np.float32)

#Tras cambiar el tamaño conseguimos 108531 secuencias de 200 filas??? 
#cada una de ellas contiene datos x y z
#La clase que se ha asignado a una secuencia (ventana) determinada es 
#la actividad que ocurre con mas frecuencia en esa ventana.

#Dividimos los datos entre train y test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(reshaped_segments, labels, test_size = 0.2, random_state = random_seed)

#CONTRUIMOS LA ARQUITECTURA DEL MODELO
#from keras.models import Sequential
#from keras.layers import LSTM, Dense, Flatten, Dropout
from keras import Sequential, LSTM, Dense, Flatten, Dropout

model = Sequential()
# RNN layer. Numero de neuronas 128, forma de entrada 
model.add(LSTM(units = 128, input_shape = (X_train.shape[1], X_train.shape[2])))
# Dropout layer
model.add(Dropout(0.5)) 
# Dense layer with ReLu
model.add(Dense(units = 64, activation='relu'))
# Softmax layer
model.add(Dense(y_train.shape[1], activation = 'softmax'))
# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])