import string

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import auc, roc_curve, confusion_matrix, roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


class PorterStemmer:
    pass


class EmailClassifier:
    def __init__(self, data_path, max_features=3000, test_size=0.3, random_state=2):
        self.data = pd.read_csv(data_path, nrows=3000)
        self.max_features = max_features
        self.test_size = test_size
        self.random_state = random_state
        self.ps = PorterStemmer()
        self.tf_idf = TfidfVectorizer(max_features=self.max_features)
        self.model = SVC(kernel='linear')

    def explore_data(self):
        print(self.data, "\n")
        print(self.data.info(), "\n")
        print("Duplicates:", self.data.duplicated().sum(), "\n")
        print("Label Distribution:", self.data['label'].value_counts(), "\n")

    def transform_text(self, text, stopwords=None, nltk=None):
        text = nltk.word_tokenize(text.lower())
        text = [word for word in text if word.isalnum()]
        text = [self.ps.stem(word) for word in text if
                word not in stopwords.words('english') and word not in string.punctuation]
        return " ".join(text)

    def preprocess_data(self):
        self.data['keywords'] = self.data['text'].apply(self.transform_text)
        X = self.tf_idf.fit_transform(self.data['keywords']).toarray()
        y = self.data['label'].values
        print(X, "\n")
        print("\n", sorted(list(X[1, :]), reverse=True)[:30])
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=self.test_size,
                                                                                random_state=self.random_state)

    def train_model(self):
        self.model.fit(self.X_train, self.y_train)

    def evaluate_model(self):
        y_pred = self.model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        roc_auc = roc_auc_score(self.y_test, y_pred)
        confusion_mat = confusion_matrix(self.y_test, y_pred)
        print("Accuracy Score:", accuracy)
        print("ROC AUC Score:", roc_auc)
        print("Confusion Matrix:\n", confusion_mat)

    def plot_roc_curve(self):
        y_pred_proba = self.model.decision_function(self.X_test)
        fpr, tpr, thresholds = roc_curve(self.y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.show()


def svm():
    classifier = EmailClassifier("Downloads/spam_email.csv")
    classifier.train_model()
    classifier.evaluate_model()
    classifier.plot_roc_curve()