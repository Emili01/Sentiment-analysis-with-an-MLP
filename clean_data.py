import pandas as pd
import re 
import numpy as np
from sklearn.utils import shuffle
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')




df = pd.read_csv('Twitter_Data.csv')
print(f"Filas originales: {df.shape[0]}")
print(df.head())


def clean_text(text):
    # Convertir a string por si hay valores no-texto
    text = str(text)

    # 1. Eliminar URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # 2. Eliminar menciones (@usuario) y hashtags
    text = re.sub(r'\@\w+|\#\w+', '', text)
    
    # 3. Eliminar caracteres especiales (conservando signos de puntuación básicos)
    text = re.sub(r'[^\w\s.,!?¿¡]', ' ', text)
    
    # 4. Eliminar números 
    text = re.sub(r'\d+', '', text)
    
    # 5. Eliminar espacios múltiples y saltos de línea
    text = ' '.join(text.split())
    
    # 6. Convertir a minúsculas
    text = text.lower()
    
    return text.strip()



def advanced_text_processing(text):
    # Tokenización
    tokens = word_tokenize(text)
    
    # Eliminar stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word.lower() not in stop_words]
    
    # Lematización
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    # Unir tokens de nuevo a texto
    return ' '.join(tokens)

df['clean_text'] = df['clean_text'].apply(clean_text)

df = df[df['clean_text'].str.strip().astype(bool)]

print(f"Filas después de limpieza: {df.shape[0]}")

# Ver distribución inicial de categorías
print("\nDistribución original de categorías:")
print(df['category'].value_counts())

# Aplicar a todo el dataset
df['processed_text'] = df['clean_text'].apply(advanced_text_processing)




from sklearn.model_selection import train_test_split

# Dividir los datos 
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)


print(f"\nTamaños finales:")
print(f"Entrenamiento: {len(train_df)}")
print(f"Prueba: {len(test_df)}")


import gensim.downloader as api
from gensim.models import KeyedVectors

# Cargar embeddings (ej: GloVe Twitter)
#glove_twitter = api.load("glove-twitter-200")

import time

max_retries = 3
retry_delay = 10  # segundos entre intentos

for attempt in range(max_retries):
    try:
        print(f"Intento {attempt + 1} de descargar embeddings...")
        glove_twitter = api.load("glove-twitter-200")
        print("¡Descarga completada!")
        break
    except Exception as e:
        print(f"Error en intento {attempt + 1}: {str(e)}")
        if attempt < max_retries - 1:
            print(f"Reintentando en {retry_delay} segundos...")
            time.sleep(retry_delay)
        else:
            print("Falló después de varios intentos. Probando soluciones alternativas...")


def text_to_avg_embedding(text, model):
    words = text.split()
    embeddings = [model[word] for word in words if word in model]
    if len(embeddings) > 0:
        return np.mean(embeddings, axis=0)
    else:
        return np.zeros(model.vector_size)

# Aplicar a todos los textos
X_train = np.array([text_to_avg_embedding(text, glove_twitter) for text in train_df['processed_text']])
X_test = np.array([text_to_avg_embedding(text, glove_twitter) for text in test_df['processed_text']])

print(f"\nDimensiones de los embeddings:")
print(f"Entrenamiento: {X_train.shape}")
print(f"Prueba: {X_test.shape}")

# Extraer las etiquetas
y_train = train_df['category'].values
y_test = test_df['category'].values

glove_twitter.save('glove_twitter_200.model')

np.save('X_train_embeddings.npy', X_train)
np.save('X_test_embeddings.npy', X_test)
np.save('y_train.npy', y_train)
np.save('y_test.npy', y_test)