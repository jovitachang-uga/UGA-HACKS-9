import ctypes
import csv
import pickle
import random
import string
import sys
from nltk import word_tokenize
from nltk.corpus import stopwords
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# file names
PHISHING_DATA = "Phishing_Email.csv"
SPAM_DATA = "mail_data.csv"
CLF_FILE_NAME = "classifier.pkl"
VEC_FILE_NAME = "vectorizer.pkl"

csv.field_size_limit(int(ctypes.c_ulong(-1).value // 2))
stop_words = set(stopwords.words('english'))

# removes punctuation and converts all words into lowercase
def parse_content(content):
    return content.translate(str.maketrans("","", string.punctuation)).lower()

# formats data from two training datasets
# returns a list in the format of (category, content)
def normalize_data():
    dataset = []
    # Adds phishing data
    with open(PHISHING_DATA, 'r', encoding = "utf8") as file:
        reader = csv.reader(file)
        for row in reader:
            dataset.append((row[2], row[1]))

    # Adds spam data
    with open(SPAM_DATA, 'r', encoding = "utf8") as file:
        reader = csv.reader(file)
        for row in reader:
            if row[0] == "ham":
                category = "Safe Email"
            else:
                category = "Phishing Email"
            dataset.append((category, row[1]))

    return dataset

# splits data into two testing and two training lists
# returns four data lists to train the classifier
def split_data(dataset):
    random.shuffle(dataset)
    x_train = []
    y_train = []

    x_test = []
    y_test = []

    # 80% of our data should be training data
    pivot = int(0.8 * len(dataset))

    for i in range(0, pivot):
        x_train.append(dataset[i][1])
        y_train.append(dataset[i][0])

    for i in range(pivot, len(dataset)):
        x_test.append(dataset[i][1])
        y_test.append(dataset[i][0])

    return x_train, x_test, y_train, y_test

# checks how well the classifier performs
def evaluate_classifer(data_type, classifier, vectorizer, x_test, y_test):
    x_test_tfidf = vectorizer.transform(x_test)
    y_pred = classifier.predict(x_test_tfidf)

    precision = metrics.precision_score(y_test, y_pred, pos_label = 'Safe Email')
    recall = metrics.recall_score(y_test, y_pred, pos_label = 'Safe Email')
    f1 = metrics.f1_score(y_test, y_pred, pos_label = 'Safe Email')

    print(f"{data_type}     {precision}     {recall}        {f1}")

# trains classifier on dataset and saves vectorizer and classifer to categorize future data
def train_classifier(dataset):
    x_train, x_test, y_train, y_test = split_data(dataset)
    vectorizer = CountVectorizer(stop_words = 'english', ngram_range = (1,3), min_df = 3, analyzer = 'word')
    
    dtm = vectorizer.fit_transform(x_train)
    
    classifier = MultinomialNB().fit(dtm, y_train)
    evaluate_classifer("Training Data", classifier, vectorizer, x_train, y_train)
    evaluate_classifer("Testing Data", classifier, vectorizer, x_test, y_test)

    pickle.dump(classifier, open(CLF_FILE_NAME, 'wb'))
    pickle.dump(vectorizer, open(VEC_FILE_NAME, 'wb'))

if __name__ == '__main__':
    dataset = normalize_data() # represented as [(label, "text")]    
    train_classifier(dataset)
    
    print("Done training and saving classifier")