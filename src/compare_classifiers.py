# compare_classifiers.py
# Compares Decision Tree and Gaussian Naive Bayes on Iris and Wine datasets.
# Uses two train/test splits: 50/50 and 80/20.

from sklearn.datasets import load_iris, load_wine
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix

def evaluate_models(X, y, dataset_name, test_size, random_state=42):
    """
    Train and evaluate Decision Tree and Gaussian Naive Bayes on a given dataset.

    Parameters:
        X : feature matrix
        y : target vector
        dataset_name : str, name of the dataset (e.g., "Iris")
        test_size : float, proportion of data to use for testing (e.g., 0.5)
        random_state : int, seed for reproducibility
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    train_pct = int((1 - test_size) * 100)
    test_pct = int(test_size * 100)

    print(f"Dataset: {dataset_name}")
    print(f"Train/Test split = {train_pct}/{test_pct}")


    # Decision Tree
    dt = DecisionTreeClassifier(random_state=random_state)
    dt.fit(X_train, y_train)
    y_pred_dt = dt.predict(X_test)

    acc_dt = accuracy_score(y_test, y_pred_dt)
    err_dt = 1 - acc_dt
    cm_dt = confusion_matrix(y_test, y_pred_dt)

    print("\n Decision Tree ")
    print(f"Accuracy:  {acc_dt:.6f}")
    print(f"Error:     {err_dt:.6f}")
    print("Confusion matrix:")
    print(cm_dt)

    # Gaussian Naive Bayes
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    y_pred_gnb = gnb.predict(X_test)

    acc_gnb = accuracy_score(y_test, y_pred_gnb)
    err_gnb = 1 - acc_gnb
    cm_gnb = confusion_matrix(y_test, y_pred_gnb)

    print("\n Gaussian Naive Bayes ")
    print(f"Accuracy:  {acc_gnb:.6f}")
    print(f"Error:     {err_gnb:.6f}")
    print("Confusion matrix:")
    print(cm_gnb)

def main():
    # Load datasets
    iris = load_iris()
    wine = load_wine()

    # Evaluate on Iris with 50/50 and 80/20 splits
    evaluate_models(iris.data, iris.target, "IRIS", test_size=0.5)
    evaluate_models(iris.data, iris.target, "IRIS", test_size=0.2)

    # Evaluate on Wine with 50/50 and 80/20 splits
    evaluate_models(wine.data, wine.target, "WINE", test_size=0.5)
    evaluate_models(wine.data, wine.target, "WINE", test_size=0.2)

if __name__ == "__main__":
    main()
