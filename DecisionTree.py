import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def DecisionTree():
    data = pd.read_csv("C:\\Users\\user\\Desktop\\AML\\data.csv")
    X = data.drop(["diagnosis", "id"], axis=1).values
    Y = data.loc[:, "diagnosis"].values

    label_encoder = LabelEncoder()
    Y = label_encoder.fit_transform(Y)

    X_train, X_test, y_train, y_test = train_test_split(X, Y, stratify=Y, random_state=0)

    dt = DecisionTreeClassifier()
    dt.fit(X_train, y_train)

    y_train_pred = dt.predict(X_train)
    y_test_pred = dt.predict(X_test)

    tree_train = accuracy_score(y_train, y_train_pred)
    tree_test = accuracy_score(y_test, y_test_pred)
    print(f"Decision tree train/test accuracies:{tree_train:.3f}/{tree_test:.3f}")

    dt = DecisionTreeClassifier()
    parameters = {"max_depth": [1, 2, 3, 4, 5],
                  "min_samples_leaf": [1, 3, 6, 10, 20]}
    clf = GridSearchCV(dt, parameters, n_jobs=1)
    clf.fit(X_train, y_train)
    print(clf.best_params_)

    y_train_pred = clf.predict(X_train)
    y_test_pred = clf.predict(X_test)
    tree_train = accuracy_score(y_train, y_train_pred)
    tree_test = accuracy_score(y_test, y_test_pred)
    print(f"Decision tree train/test accuracies:{tree_train:.3f}/{tree_test:.3f}")

    cm = confusion_matrix(y_test, y_test_pred)

    # Visualize confusion matrix as a heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Predicted Negative', 'Predicted Positive'],
                yticklabels=['Actual Negative', 'Actual Positive'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()