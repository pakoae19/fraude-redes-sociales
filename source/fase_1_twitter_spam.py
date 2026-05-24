from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

import pandas as pd

#twitter = pd.read_csv("/Users/pko/Desktop/twitter_spam.csv")
twitter = pd.read_csv("twitter_spam.csv")

#print(twitter.columns) Imprime columnas:
#Index(['class', 'tweets', 'Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], dtype='object')

#Usa solo columnas 'tweets' y 'class'
twitter = twitter[['tweets', 'class']].dropna()

#Renombra columnas
twitter.columns = ['texto', 'etiqueta']

#Vectoriza texto
vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(twitter['texto'])
y = twitter['etiqueta']

#Función para entrenar y evaluar modelos
def evaluar_modelos(X, y, division, tamannio_prueba):
    print(f"\n====== División: {division} ({int((1-tamannio_prueba)*100)}% entrenamiento / {int(tamannio_prueba*100)}% prueba) ======")

    #Separa datos
    X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = train_test_split(X, y, test_size = tamannio_prueba, random_state = 42)

    #Clasificadores
    arbol = DecisionTreeClassifier(random_state = 42)
    bayes = MultinomialNB()
    bpn = MLPClassifier(hidden_layer_sizes=(10,), max_iter=2000, random_state=42)

    #Entrenamiento
    arbol.fit(X_entrenamiento, y_entrenamiento)
    bayes.fit(X_entrenamiento, y_entrenamiento)
    bpn.fit(X_entrenamiento, y_entrenamiento)

    #Predicciones
    pred_arbol = arbol.predict(X_prueba)
    pred_bayes = bayes.predict(X_prueba)
    pred_bpn = bpn.predict(X_prueba)

    #Evaluación
    acc_arbol = accuracy_score(y_prueba, pred_arbol)
    acc_bayes = accuracy_score(y_prueba, pred_bayes)
    acc_bpn = accuracy_score(y_prueba, pred_bpn)
    cm_arbol = confusion_matrix(y_prueba, pred_arbol)
    cm_bayes = confusion_matrix(y_prueba, pred_bayes)
    cm_bpn = confusion_matrix(y_prueba, pred_bpn)

    print(f" Árbol de decisión - Precisión: {acc_arbol:.2f}")
    print(f" Naive Bayes        - Precisión: {acc_bayes:.2f}")
    print(f" Red Neuronal (BPN) - Precisión: {acc_bpn:.2f}")
    
    print("\nMatriz de Confusión - Árbol de decisión:")
    print(cm_arbol)
    print("\nMatriz de Confusión - Naive Bayes:")
    print(cm_bayes)
    print("\nMatriz de Confusión - Backpropagation:")
    print(cm_bpn)

#Evalua modelo con divisiones (50/50 y 80/20)
evaluar_modelos(X, y, "50/50", tamannio_prueba = 0.5)
evaluar_modelos(X, y, "80/20", tamannio_prueba = 0.2)
