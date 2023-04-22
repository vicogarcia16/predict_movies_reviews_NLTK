import nltk
from nltk.corpus import movie_reviews
from nltk.classify import NaiveBayesClassifier
from nltk.tokenize import word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer

# Descargar los recursos necesarios de NLTK
# Puedes usar la consola Python
# nltk.download('punkt')
# nltk.download('movie_reviews')
# nltk.download('vader_lexicon')

# Crear una lista de tuplas (oraciones, polaridad)
documents = []
for category in movie_reviews.categories():
    for fileid in movie_reviews.fileids(category):
        words = movie_reviews.words(fileid)
        documents.append((words, category))

# Definir una función para extraer características de las oraciones
def extract_features(words):
    return dict([(word, True) for word in words])

# Crear un conjunto de características para cada oración
featuresets = [(extract_features(words), category) for (words, category) in documents]

# Dividir los datos en un conjunto de entrenamiento y un conjunto de prueba
train_set, test_set = featuresets[100:], featuresets[:100]

# Entrenar el clasificador
classifier = NaiveBayesClassifier.train(train_set)

# Definir una función para predecir la polaridad de una oración
def predict_sentiment(sentence):
    # Tokenizar la oración
    words = word_tokenize(sentence)
    # Extraer las características de la oración
    features = extract_features(words)
    # Predecir la polaridad de la oración utilizando el clasificador
    return classifier.classify(features)

# Probar el clasificador con algunas oraciones de prueba
positive_sentence = "This movie is absolutly awesome, I loved it!"
negative_sentence = "This movie is terrible, I hated it!"
print(f'Oración 1: {positive_sentence} *Predicción: {predict_sentiment(positive_sentence)}')  
print(f'Oración 2: {negative_sentence} *Predicción: {predict_sentiment(negative_sentence)}')

# Utilizar el analizador de sentimientos de NLTK para analizar la polaridad de una oración
sia = SentimentIntensityAnalyzer()
sentence = "This movie is really boring and I didn't like it at all."
sentiment_scores = sia.polarity_scores(sentence)
print(f'Oracion 3: {sentence} \n*Analizador de sentimientos: {sentiment_scores}') 
