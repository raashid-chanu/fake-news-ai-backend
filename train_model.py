import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle

fake = pd.read_csv("Fake.csv")
real = pd.read_csv("True.csv")

fake["label"] = 0
real["label"] = 1

data = pd.concat([fake, real])
data = data.sample(frac=1)

X = data["text"]
y = data["label"]

vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)
X_vec = vectorizer.fit_transform(X)

model = LogisticRegression()
model.fit(X_vec, y)

pickle.dump(model, open("model.pkl","wb"))
pickle.dump(vectorizer, open("vectorizer.pkl","wb"))

print("Large AI Model Trained Successfully")
