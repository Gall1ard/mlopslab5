import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.pipeline import Pipeline
from sklearn import tree
from sklearn.metrics import root_mean_squared_error, f1_score, accuracy_score, precision_score, recall_score
import numpy as np
import mlflow
import pickle
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from mlflow.models import infer_signature
from sklearn.model_selection import GridSearchCV


def train_model():
    df_train = pd.read_csv("./data/train_iris.csv")
    df_test = pd.read_csv("./data/test_iris.csv")

    X_train, y_train = df_train.drop(columns = ['species']).values, df_train['species'].values
    X_val, y_val  = df_test.drop(columns = ['species']).values, df_test['species'].values
    
    print(X_val)
    
    params = {
        'max_depth': [3, 5, 10, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'criterion': ['gini', 'entropy']  # or 'log_loss' for some versions
    }
    mlflow.set_experiment("tree model irises")
    with mlflow.start_run():
        # Change to Classifier
        clf_tree = tree.DecisionTreeClassifier(random_state=42)

        # GridSearchCV works the same way
        clf = GridSearchCV(clf_tree, params, cv=5, scoring='accuracy')  # or 'f1_macro', 'roc_auc'
        clf.fit(X_train, y_train)  # y_train should be categorical labels (0, 1, 2...) not continuous

        best = clf.best_estimator_

        # Predictions (class labels)
        y_pred = best.predict(X_val)

        # Classification metrics (no inverse_transform needed for standard classification)
        acc = accuracy_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred, average='weighted')  # adjust average as needed
        recall = recall_score(y_val, y_pred, average='weighted')
        f1 = f1_score(y_val, y_pred, average='weighted')

        # Log tree-specific params
        mlflow.log_param("max_depth", best.max_depth)
        mlflow.log_param("min_samples_split", best.min_samples_split)
        mlflow.log_param("min_samples_leaf", best.min_samples_leaf)
        mlflow.log_param("criterion", best.criterion)

        # Log classification metrics (instead of RMSE/MAE/R2)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        # Signature (predictions are now class labels or probabilities)
        predictions = best.predict(X_train)
        signature = infer_signature(X_train, predictions)
        mlflow.sklearn.log_model(best, "model", signature=signature)

        with open("./models/iris.joblib", "wb") as file:
            pickle.dump(best, file)