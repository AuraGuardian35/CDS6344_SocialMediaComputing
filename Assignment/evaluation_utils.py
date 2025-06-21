# Evaluation utilities for sentiment analysis models

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model(y_true, y_pred, model_name):
    """Standard evaluation function for both persons to use"""
    accuracy = accuracy_score(y_true, y_pred)
    f1_weighted = f1_score(y_true, y_pred, average='weighted')
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    
    results = {
        'model_name': model_name,
        'accuracy': accuracy,
        'f1_weighted': f1_weighted,
        'precision': precision,
        'recall': recall,
        'classification_report': classification_report(y_true, y_pred)
    }
    
    print(f"\n{model_name} Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1-Score (weighted): {f1_weighted:.4f}")
    print(f"Precision (weighted): {precision:.4f}")
    print(f"Recall (weighted): {recall:.4f}")
    
    return results

def plot_confusion_matrix(y_true, y_pred, labels=['Negative', 'Neutral', 'Positive'], title='Confusion Matrix'):
    """Standard confusion matrix plotting function"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()
    return cm
