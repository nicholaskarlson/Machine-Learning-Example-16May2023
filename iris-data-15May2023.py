from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def load_data():
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    return X, y, iris.target_names

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=1) 
    clf = RandomForestClassifier(random_state=1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
    return y_pred, y_test, clf

def output_results(y_pred, y_test, target_names):
    print("\nPredicted Class - Actual Class")
    for pred, actual in zip(y_pred, y_test):
        print(f"{target_names[pred]} - {target_names[actual]}")

if __name__ == "__main__":
    X, y, target_names = load_data()
    y_pred, y_test, clf = train_model(X, y)
    output_results(y_pred, y_test, target_names)

