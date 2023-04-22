### Predicción de reseñas en inglés de peliculas

#### Resumen de funcionalidad :computer:
1. Se utiliza el NaiveBayesClassifier de NLTK para entrenar un modelo de clasificación de polaridad utilizando el corpus de movie_reviews. 
2. Luego, se define una función predict_sentiment que toma una oración como entrada y utiliza el modelo entrenado para predecir la polaridad de la oración (pos o neg).
3. Por último, se utiliza el analizador de sentimientos de NLTK (SentimentIntensityAnalyzer) para analizar la polaridad de una oración. Este analizador utiliza un enfoque diferente al del clasificador NaiveBayesClassifier y proporciona una puntuación de polaridad en lugar de simplemente una etiqueta 'pos' o 'neg'.

#### Ejecución de código ⚙️

1. Realizar un entorno virtual. En este caso se hace uso de pipenv.
    - <code>pipenv shell</code>
2. Instalar las librerias necesarias.
    - <code>pipenv install -r requirements.txt</code>
3. Instalar los recursos necesarios de NLTK. Utiliza una consola de python.
    - <code>pipenv run python</code>
        - <code>import nlkt</code>
        - <code>nltk.download('punkt')</code>
        - <code>nltk.download('movie_reviews')</code>
        - <code>nltk.download('vader_lexicon')</code>
4. Ejecutar el archivo python.
    - <code>pipenv run movie_reviews_classifier.py</code>

#### Construido con 🛠️

* [Python](https://www.python.org/) - Lenguaje de programación
* [NLTK](https://www.nltk.org/) - Libreria NLTK

#### Autor ✒️

* **Víctor García** [vicogarcia16](https://github.com/vicogarcia16) 
