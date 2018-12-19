# importinng pandas for dataset manipulationa
import pandas as pd

# reading the data
df = pd.read_excel(
    'D:/python/UpWORK/Imam/Imam.xlsx', head=None)


from langdetect import detect

# applying language detection into new coulmn name Language_detection
df['Language_detection'] = df['Text'].apply(detect)
df.head()

# code from github comment don't work
#from textblob import TextBlob

for index, row in df.iterrows():
    zh_blob = df.iloc[index]['Text']
    translation = TextBlob(zh_blob)
    en_blob = translation.translate(from_lang='ar', to='en')
    df.at[index, str('Text translated')] = str(en_blob)
df.head()
# End github comment
#########


# Imports the Google Cloud client library
from googletrans import Translator
translator = Translator()
# translting Text coulmn into a new column Text_to_English
df['Text_to_English'] = df['Text'].apply(translator.translate, dest='en')
df.head()
# cleaning the tranlslated text by removing the first 33 character
df['Text_to_English'] = df['Text_to_English'].map(lambda x: str(x)[33:])
df.head()
# exploring the data
df['Category'].value_counts()


# ML code for the text data
import re
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.ensemble import RandomForestClassifier


stemmer = SnowballStemmer('english')
words = stopwords.words("english")


stemmer
words
df['cleaned'] = df['Text_to_English'].apply(lambda x: " ".join(
    [stemmer.stem(i) for i in re.sub("[^a-zA-Z]", " ", x).split() if i not in words]).lower())

df.head()

X_train, X_test, y_train, y_test = train_test_split(
    df['cleaned'], df['Category'], test_size=0.2, random_state=42)

pipeline = Pipeline([('vect', TfidfVectorizer(ngram_range=(1, 6), stop_words="english", sublinear_tf=False)),
                     ('chi',  SelectKBest(chi2, k='all')),
                     ('clf', LinearSVC(C=100000000000, max_iter=1000000, dual=False))])


model = pipeline.fit(X_train, y_train)

vectorizer = model.named_steps['vect']
chi = model.named_steps['chi']
clf = model.named_steps['clf']

print("accuracy score: " + str(model.score(X_train, y_train)))
print("accuracy score: " + str(model.score(X_test, y_test)))

# Testing the modeel
print(model.predict(["the ahly team plays football in the stadium"]))
print(model.predict(["The Muslims prays five times a day and faith "]))
print(model.predict(["the parliament decided that the president should be elected "]))
print(model.predict(["the bank decided to invest in stock markets by many loans"]))
print(model.predict(["as god said in the bible and koran the man should obay god and be faithful"]))
print(model.predict(["stock market is broke and have no money in my bank account"]))
print(model.predict(["the winner of the game and the chambion is "]))
print(model.predict(["the god created the universe"]))
print(model.predict(["the prime minster is at the head of the government"]))
print(model.predict(["profet mohamed is the messenger of islam"]))


# example  of count vectorizer cleaning
from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer()
vect.fit(df['cleaned'])
vect.get_feature_names()
simple_train_dtm = vect.transform(df['cleaned'])
simple_train_dtm
simple_train_dtm.toarray()
sampledf = pd.DataFrame(simple_train_dtm.toarray(), columns=vect.get_feature_names())
sampledf.shape
sampledf.head()
