import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)



# ------------------------ SELECCIÓN DEL DATASET ------------------------


# Cambiar esta variable según el dataset que se quiera probar:
# "twitter" para Twitter Spam Dataset
# "isot" para ISOT Fake News Dataset

# dataset = "twitter"
dataset = "isot"


# ------------------------ CARGA DE DATASETS ------------------------

def cargar_twitter_spam():
    twitter = pd.read_csv("twitter_spam.csv")

    twitter = twitter[["tweets", "class"]].dropna()
    twitter.columns = ["texto", "etiqueta"]

    return twitter


def cargar_isot():

    train = pd.read_csv("ISOT Fake News Dataset/train.csv", sep=";")
    test = pd.read_csv("ISOT Fake News Dataset/test.csv", sep=";")

    # Junta train y test en un solo dataset
    isot = pd.concat([train, test], ignore_index=True)

    # Cambia "label" por "etiqueta"
    isot = isot.rename(columns={"label": "etiqueta"})

    # Usa el texto de la noticia
    isot["texto"] = isot["text"].astype(str)

    isot = isot[["texto", "etiqueta"]].dropna()

    return isot


if dataset == "twitter":
    datos = cargar_twitter_spam()
    nombre_dataset = "Twitter Spam Dataset"

elif dataset == "isot":
    datos = cargar_isot()
    nombre_dataset = "ISOT Fake News Dataset"

else:
    raise ValueError("Dataset no válido.")


print(f"\nDataset utilizado: {nombre_dataset}")
print(f"Cantidad de registros: {len(datos)}")
print(datos["etiqueta"].value_counts())


# ------------------------ VECTORIZADO DE TEXTO CON TF-IDF ------------------------

vectorizer = TfidfVectorizer(
    max_features=5000,
    stop_words="english"
)

X = vectorizer.fit_transform(datos["texto"])
y = datos["etiqueta"]


# ------------------------ DEFINICIÓN DE MODELOS ------------------------

modelos = {
    # Modelos implementados en la fase 1
    "Árbol de decisión": DecisionTreeClassifier(random_state=42),

    "Naive Bayes": MultinomialNB(),

    "Backpropagation": MLPClassifier(
        hidden_layer_sizes=(10,),
        max_iter=2000,
        random_state=42
    ),

    # Modelos implementados en la fase 2
    "k-NN": KNeighborsClassifier(n_neighbors=5),

    "SVM": LinearSVC(random_state=42),

    "DNN": MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),
        max_iter=1000,
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
}


# ------------------------ EVALUACIÓN DE MODELOS ------------------------

def evaluar_modelos(X, y, division, tamannio_prueba):
    print("\n" + "=" * 70)
    print(f"División: {division}")
    print(f"Entrenamiento: {int((1 - tamannio_prueba) * 100)}%")
    print(f"Prueba: {int(tamannio_prueba * 100)}%")
    print("=" * 70)

    X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = train_test_split(
        X,
        y,
        test_size=tamannio_prueba,
        random_state=42,
        stratify=y
    )

    resultados = []

    for nombre_modelo, modelo in modelos.items():
        print(f"\nEntrenando modelo: {nombre_modelo}")

        modelo.fit(X_entrenamiento, y_entrenamiento)

        predicciones = modelo.predict(X_prueba)

        accuracy = accuracy_score(y_prueba, predicciones)
        precision = precision_score(y_prueba, predicciones)
        recall = recall_score(y_prueba, predicciones)
        f1 = f1_score(y_prueba, predicciones)

        matriz = confusion_matrix(y_prueba, predicciones)

        # Validación cruzada con 5 folds usando F1-score
        cv_scores = cross_val_score(
            modelo,
            X,
            y,
            cv=5,
            scoring="f1"
        )

        cv_promedio = cv_scores.mean()

        resultados.append({
            "Dataset": nombre_dataset,
            "División": division,
            "Modelo": nombre_modelo,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1-score": f1,
            "Cross Validation F1": cv_promedio
        })

        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-score:  {f1:.4f}")
        print(f"CV F1:     {cv_promedio:.4f}")

        print("Matriz de confusión:")
        print(matriz)

    return resultados


# ------------------------ EJECUCIÓN DE PRUEBAS ------------------------

resultados_50_50 = evaluar_modelos(
    X,
    y,
    "50/50",
    tamannio_prueba=0.5
)

resultados_80_20 = evaluar_modelos(
    X,
    y,
    "80/20",
    tamannio_prueba=0.2
)


# ------------------------ TABLA FINAL DE RESULTADOS ------------------------

resultados_totales = resultados_50_50 + resultados_80_20

tabla_resultados = pd.DataFrame(resultados_totales)

print("\n" + "=" * 70)
print("TABLA FINAL DE RESULTADOS")
print("=" * 70)

pd.set_option("display.max_columns", None)
print(tabla_resultados)


# Guardar resultados en CSV
tabla_resultados.to_csv(
    f"resultados_{dataset}.csv",
    index=False
)

print(f"\nResultados guardados en: resultados_{dataset}.csv")
