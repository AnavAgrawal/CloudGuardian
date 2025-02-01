import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import precision_recall_fscore_support, roc_curve, auc, confusion_matrix
import os
import pickle

os.makedirs('plots', exist_ok=True)
os.makedirs('models', exist_ok=True)


def prepare_dataset(df):
    """Prepare the BETH dataset features"""
    df["processId"] = df["processId"].map(lambda x: 0 if x in [0, 1, 2] else 1)
    df["parentProcessId"] = df["parentProcessId"].map(
        lambda x: 0 if x in [0, 1, 2] else 1)
    df["userId"] = df["userId"].map(lambda x: 0 if x < 1000 else 1)
    df["mountNamespace"] = df["mountNamespace"].map(
        lambda x: 0 if x == 4026531840 else 1)
    df["returnValue"] = df["returnValue"].map(
        lambda x: 0 if x == 0 else (1 if x > 0 else 2))

    features = df[["processId", "parentProcessId", "userId", "mountNamespace",
                  "eventId", "argsNum", "returnValue"]]
    labels = df['sus']
    return features, labels


def train_models(X_train, y_train):
    """Train multiple models and return a dictionary of trained models"""
    models = {
        'Isolation Forest': IsolationForest(contamination=0.1, random_state=42, n_jobs=-1),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        # 'LOF': LocalOutlierFactor(contamination=0.1, n_jobs=-1)
    }

    for name, model in models.items():
        print(f"Training {name}...")
        if name == 'Random Forest':
            model.fit(X_train, y_train)
        else:
            model.fit(X_train)

        print('done')
    return models


def evaluate_model(model, X_test, y_test, model_name):
    """Evaluate a single model and return metrics"""
    if isinstance(model, RandomForestClassifier):
        y_pred = model.predict(X_test)
        y_score = model.predict_proba(X_test)[:, 1]
    elif isinstance(model, LocalOutlierFactor):
        y_pred = model.predict(X_test)
        # Convert to binary (1 for anomaly)
        y_pred = np.where(y_pred == 1, 0, 1)
        y_score = -model.negative_outlier_factor_
    else:
        y_pred = model.predict(X_test)
        # Convert to binary (1 for anomaly)
        y_pred = np.where(y_pred == 1, 0, 1)
        y_score = -model.score_samples(X_test)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average='weighted')
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'predictions': y_pred,
        'scores': y_score
    }


def plot_roc_curves(models_dict, X_test, y_test):
    """Plot ROC curves for all models"""
    plt.figure(figsize=(10, 6))

    for name, model in models_dict.items():
        if isinstance(model, RandomForestClassifier):
            y_score = model.predict_proba(X_test)[:, 1]
        elif isinstance(model, LocalOutlierFactor):
            y_score = -model.negative_outlier_factor_
        else:
            y_score = -model.score_samples(X_test)

        fpr, tpr, _ = roc_curve(y_test, y_score)
        roc_auc = auc(fpr, tpr)

        plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves Comparison')
    plt.legend()
    plt.savefig('plots/roc_curves_comparison.png')
    plt.close()


def plot_confusion_matrices(models_dict, X_test, y_test):
    """Plot confusion matrices for all models"""
    fig, axes = plt.subplots(1, len(models_dict), figsize=(15, 4))

    for idx, (name, model) in enumerate(models_dict.items()):
        if isinstance(model, RandomForestClassifier):
            y_pred = model.predict(X_test)
        elif isinstance(model, LocalOutlierFactor):
            y_pred = model.predict(X_test)
            y_pred = np.where(y_pred == 1, 0, 1)
        else:
            y_pred = np.where(model.predict(X_test) == 1, 0, 1)

        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', ax=axes[idx])
        axes[idx].set_title(f'{name}\nConfusion Matrix')
        axes[idx].set_xlabel('Predicted')
        axes[idx].set_ylabel('True')

    plt.tight_layout()
    plt.savefig('plots/confusion_matrices.png')
    plt.close()


def plot_feature_importance(X_train, rf_model):
    """Plot feature importance for Random Forest"""
    importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=importance, x='importance', y='feature')
    plt.title('Feature Importance (Random Forest)')
    plt.tight_layout()
    plt.savefig('plots/feature_importance.png')
    plt.close()


if __name__ == "__main__":

    print("Loading datasets...")
    train_df = pd.read_csv("data/labelled_training_data.csv")
    test_df = pd.read_csv("data/labelled_testing_data.csv")
    val_df = pd.read_csv("data/labelled_validation_data.csv")

    # Prepare datasets
    print("Preparing datasets...")
    X_train, y_train = prepare_dataset(train_df)
    X_test, y_test = prepare_dataset(test_df)
    X_val, y_val = prepare_dataset(val_df)

    print("Training models...")
    models = train_models(X_train, y_train)

    # Evaluate and plot
    print("Evaluating models and creating plots...")
    results = {}
    for name, model in models.items():
        results[name] = evaluate_model(model, X_test, y_test, name)
        print(f"\n{name} Results:")
        print(f"Precision: {results[name]['precision']:.3f}")
        print(f"Recall: {results[name]['recall']:.3f}")
        print(f"F1-Score: {results[name]['f1']:.3f}")

    # Generate plots
    plot_roc_curves(models, X_test, y_test)
    plot_confusion_matrices(models, X_test, y_test)
    plot_feature_importance(X_train, models['Random Forest'])

    best_model_name = max(results.items(), key=lambda x: x[1]['f1'])[0]
    best_model = models[best_model_name]

    with open('models/best_model.pkl', 'wb') as f:
        pickle.dump(best_model, f)

    print(f"\nBest model ({best_model_name}) saved to models/best_model.pkl")
    print("All plots saved in plots/ directory")
