# JHAMPIER TAPIA SUCAPUCA
import pandas as pd
import numpy as np
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import io


# Read the text file, using semicolon as separator - JHAMPIER
df_train = pd.read_csv('../data/train.txt', 
                 sep=';', 
                 header=None, 
                 names=['text', 'label'])
df_test = pd.read_csv('../data/test.txt', 
                 sep=';', 
                 header=None, 
                 names=['text', 'label'])



# --- 3. Vectorización de Texto (TF-IDF) ---
# importantes que no aparecen en muchos documentos.
tfidf_vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
# ngram_range=(1, 2) incluye palabras solas (unigramas) y pares de palabras (bigramas)
tfidf_vectorizer                 

X_train = df_train['text']
y_train = df_train['label']
X_test = df_test['text']
y_test = df_test['label']

# Ajustar (aprender vocabulario) y transformar los datos de entrenamiento
X_train_vectorized = tfidf_vectorizer.fit_transform(X_train)

# Transformar los datos de prueba usando el vocabulario aprendido
X_test_vectorized = tfidf_vectorizer.transform(X_test)

print("Datos vectorizados correctamente.")
print("-" * 35)

# --- 4. Entrenamiento del Modelo ---
# La Regresión Logística (Logistic Regression) es un clasificador simple y robusto 
# que funciona bien como baseline para tareas de clasificación de texto.
print("--- Iniciando Entrenamiento del Modelo: Logistic Regression ---")
model = LogisticRegression(max_iter=1000, random_state=42, solver='liblinear')
model.fit(X_train_vectorized, y_train)
print("Entrenamiento Finalizado. 🎉")
print("-" * 35)

# --- 5. Evaluación del Modelo ---
y_pred = model.predict(X_test_vectorized)

# Cálculo de la Precisión
accuracy = accuracy_score(y_test, y_pred)

cloudpickle.dump(model, open(model_path, "wb"))