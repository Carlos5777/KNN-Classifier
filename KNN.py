"""KNN.ipynb

KNN Movie Review Classifier

This script implements a K-Nearest Neighbors model to classify
movie reviews based on sentiment (positive/negative).
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

df = pd.read_csv('reseñas.csv')
print(df.head())

x = df ['Reseña']
y = df ['Sentimiento']

vectorizer = CountVectorizer()
x_vec = vectorizer.fit_transform(x)

print(x_vec)

X_train, X_test, y_train, y_test = train_test_split(x_vec, y, test_size=0.2, random_state=42)

model = KNeighborsClassifier()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print('\n metricas del modelo')
print('Accuracy:', accuracy_score(y_test, y_pred))
print('\nMatriz de confunsion', confusion_matrix(y_test, y_pred))
print('\nReporte de clasificacion', classification_report(y_test, y_pred))

nuevas_reseñas = [
    'Que peli tan mala aburrida y sin sentido. Fui a verla porque tenia buenas reseñas pero no me gustó nada',
    'Peli de culto. Empieza muy lento y da un poco de miedo el tono que puede pillar. Pero desde el cambio temporal en adelante la peli es muy frenética y divertidísima.',
    'Una completa pérdida de tiempo, no la recomiendo',
    'La película fue increíble, me hizo llorar'
]

X_new = vectorizer.transform(nuevas_reseñas)
y_new = model.predict(X_new)

for texto, etiqueta in zip(nuevas_reseñas, y_new):
    print(f'Mensaje: {texto} Etiqueta: {etiqueta}')
