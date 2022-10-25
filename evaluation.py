from cProfile import label
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import label_binarize


def get_performance(predictions, y_test, labels=[1, 0]):

    accuracy = metrics.accuracy_score(y_true=y_test, y_pred=predictions)  
    precision = metrics.precision_score(y_true=y_test, y_pred=predictions)
    recall = metrics.recall_score(y_true=y_test, y_pred=predictions)
    f1_score = metrics.f1_score(y_true=y_test, y_pred=predictions)
    
    report = metrics.classification_report(y_true=y_test, y_pred=predictions, labels=labels)
    
    cm = metrics.confusion_matrix(y_true=y_test, y_pred=predictions,labels=labels)  
    cm_as_dataframe = pd.DataFrame(data=cm)
    
    print('Model Performance metrics:')
    print('-'*30)
    print('Accuracy:', accuracy)
    print('Precision:', precision)
    print('Recall:', recall)
    print('F1 Score:', f1_score)
    print('\nModel Classification report:')
    print('-'*30)
    print(report)
    print('\nPrediction Confusion Matrix:')
    print('-'*30)
    print(cm_as_dataframe)
    
    return accuracy, precision, recall, f1_score


def plot_roc(model, y_test, features):
    
    pred_proba = model.predict_proba(features)[:, 1]
    fpr, tpr, Thresholds = roc_curve(y_test, pred_proba)
    roc_auc = roc_auc_score(y_test,pred_proba)

    plt.figure(figsize=(10, 5))
    plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc})', linewidth=2.5)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()

    return roc_auc