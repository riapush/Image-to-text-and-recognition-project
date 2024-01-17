import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import RegexpTokenizer
from sklearn.model_selection import train_test_split


stop_words = stopwords.words()

nb=pd.read_csv('IMDB Dataset.csv')
nb['sentiment'] = [0 if each == "negative" else 1 for each in nb['sentiment']]
tokenized_review=nb['review'].apply(lambda x: x.split())


token = RegexpTokenizer(r'[a-zA-Z0-9]+')
cv = CountVectorizer(stop_words='english', ngram_range=(1,1),tokenizer=token.tokenize)
text_counts = cv.fit_transform(nb['review'])

X=text_counts
y=nb['sentiment']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20,random_state=30)


NB = MultinomialNB()
NB.fit(X_train, y_train)

predicted = NB.predict(X_test)
accuracy_score = metrics.accuracy_score(predicted, y_test)

# for test sample from IMDB dataset
print('MultinomialNB model accuracy is',str('{:04.2f}'.format(accuracy_score*100))+'%')

data = pd.read_csv('processed_images_modified.csv', encoding='utf-8')
data['tokenized_review'] = data['review'].apply(lambda x: x.split())
X_test = cv.transform(data['review'])
predicted = NB.predict(X_test)
data['predicted_sentiment'] = predicted
data['sentiment'] = [0 if each == "negative" else 1 for each in data['sentiment']]
print(data[['sentiment', 'predicted_sentiment']])

accuracy_score = metrics.accuracy_score(predicted, data['sentiment'])

# for test sample from image_to_text texts
print('MultinomialNB model accuracy is',str('{:04.2f}'.format(accuracy_score*100))+'%')


