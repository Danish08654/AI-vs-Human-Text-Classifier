import os
import joblib
import nltk
import string
import random
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import MaxAbsScaler
from imblearn.over_sampling import SMOTE
from nltk.tokenize import RegexpTokenizer
from scipy.sparse import hstack

def load_data():
    ai_data, human_data = [], []
    for fname in os.listdir('data/ai'):
        fpath = os.path.join('data/ai', fname)
        if os.path.isfile(fpath) and fpath.endswith('.txt'):
            with open(fpath, 'r', encoding='utf-8') as f:
                ai_data.append(f.read())
    for fname in os.listdir('data/human'):
        fpath = os.path.join('data/human', fname)
        if os.path.isfile(fpath) and fpath.endswith('.txt'):
            with open(fpath, 'r', encoding='utf-8') as f:
                human_data.append(f.read())

    min_len = min(len(ai_data), len(human_data))
    ai_data = random.sample(ai_data, min_len)
    human_data = random.sample(human_data, min_len)

    data = ai_data + human_data
    labels = [1] * min_len + [0] * min_len
    return data, labels

def preprocess(text):
    text = text.lower()
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(text)
    return ' '.join(tokens)

def prepare_dataset(data, labels):
    data = [preprocess(t) for t in data]

    char_vect = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 6), max_features=6000)
    word_vect = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), max_features=6000)

    X_char = char_vect.fit_transform(data)
    X_word = word_vect.fit_transform(data)
    X = hstack([X_char, X_word])

    scaler = MaxAbsScaler()
    X_scaled = scaler.fit_transform(X)
    y = np.array(labels)

    return train_test_split(X_scaled, y, test_size=0.3, stratify=y, random_state=42), (char_vect, word_vect, scaler)

def apply_smote(X_train, y_train):
    smote = SMOTE(random_state=42)
    return smote.fit_resample(X_train, y_train)

def train_model(X_train, y_train):
    base_model = LogisticRegressionCV(
        Cs=10,
        cv=5,
        max_iter=2000,
        solver='liblinear',
        penalty='l2',
        scoring='accuracy',
        n_jobs=-1
    )
    model = CalibratedClassifierCV(base_model, cv=5)
    model.fit(X_train, y_train)
    return model

def evaluate(model, X_test, y_test):
    preds = model.predict(X_test)
    print("\nTest Accuracy:", accuracy_score(y_test, preds))
    print(classification_report(y_test, preds))
    print("Confusion Matrix:\n", confusion_matrix(y_test, preds))

def save(model, vectorizers):
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/plagiarism_detector.pkl')
    joblib.dump(vectorizers, 'models/vectorizer.pkl')

def main():
    data, labels = load_data()
    (X_train, X_test, y_train, y_test), vectorizers = prepare_dataset(data, labels)
    X_train, y_train = apply_smote(X_train, y_train)
    model = train_model(X_train, y_train)
    evaluate(model, X_test, y_test)
    save(model, vectorizers)
    print("\Enhanced Model trained and saved!")

if __name__ == '__main__':
    main()
