# Réseau de Neurones Récurrents / Recurrent Neural Network




# Part 1 - Préparation des données


# Import des librairies
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Import du jeux d'entrainement
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[:, 1:2].values   #value transforme en tableau


# Changement d'échelle / Feature Scaling : normalisation
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)   #apprend à mettre à l'échelle


# Création de la structure des données avec 60 timesteps(=les 60jours de bourses précédent) et 1 sortie
X_train = []   #60 timesteps
y_train = []   #le jour évalué
for i in range(60, 1258):
    X_train.append(training_set_scaled[i-60:i, 0])   #append ajoute
    y_train.append(training_set_scaled[i, 0])
X_train = np.array(X_train)   #transforme la liste en tableau  
y_train = np.array(y_train)   
# ou X_train, y_train = np.array(X_train), np.array(y_train)   


# Ajout d'une dimension / Reshaping 
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))




# Part 2 - Construction du RNN


# Import de la librairies Keras and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout


# Initialisation du RNN
regressor = Sequential()


# Ajout de la première couche LSTM avec Dropout évitant le surentrainement 
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))


# Ajout d'une deuxième couche LSTM avec Dropout évitant le surentrainement
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))


# Ajout d'une troisième couche LSTM avec Dropout évitant le surentrainement
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))


# Ajout d'une quatrième couche LSTM avec Dropout évitant le surentrainement
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))


# Ajout de la couche de sortie
regressor.add(Dense(units = 1))


# Compilation du RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')


# Fitting the RNN to the Training set
regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)




# Part 3 - Faire la prediction et la visualisation du résultat


# Import des données de janvier 2017
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:, 1:2].values


# Prédiction pour janvier 2017
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)   #concaténe les lignes(axis=0 sinon colonnes axis=1)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1,1)   #changement du format
inputs = sc.transform(inputs)   #met à l'échelle sans réapprentissage
X_test = []
for i in range(60, 80):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)   ##transforme la liste en tablea
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)   #inverse la mise à l'échelle


# Visualisation des resultats
plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()




# Part 4 - Evaluation des performance du modèle


# Racine de l'Erreur Quadratique Moyenne / Root Mean Squared Error : RMSE
import math
from sklearn.metrics import mean_squared_error
rmse = math.sqrt(mean_squared_error(real_stock_price, predicted_stock_price))
