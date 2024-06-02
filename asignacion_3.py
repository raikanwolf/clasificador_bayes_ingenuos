import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar el conjunto de datos Iris
iris = datasets.load_iris()
X = iris.data #matriz (150, 4), Filas=flores Columnas=medidas
y = iris.target # guarda las clases clases correspondientes a cada instancia

# Convertir a un DataFrame de Pandas para una mejor visualización
df = pd.DataFrame(data=X, columns=iris.feature_names)
df['target'] = y

#mostrando las 5 primeras flores
df.head()

# Dividir el conjunto de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Crear el modelo de Naive Bayes
nb_model = GaussianNB()

#print(X_train) #use esto para comprobar que use el 60% para entrenar

# Entrenar el modelo
nb_model.fit(X_train, y_train)

# Hacer predicciones con un metodo de naive bayes de sklearn
y_pred = nb_model.predict(X_test)

# Evaluar el modelo
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Mostrar el reporte de clasificación
print('Classification Report:')
print(classification_report(y_test, y_pred))

# Mostrar la matriz de confusión (basicamente si confundio las flores)
print('Confusion Matrix:')
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='g')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()