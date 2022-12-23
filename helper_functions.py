## function to evaluate: accuracy, precision, recall, f1_score

from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def calculate_accuracy(y_true, y_pred):
    """
    Calculate model accuracy, precision, recall and f1_score for a binary classification model
    """
    # calculate model accuracy
    model_accuracy = accuracy_score(y_true, y_pred) * 100
    # calculate model precision, recall and f1_score using 'weighted average'
    model_precision, model_recall, model_f1_score, _ = precision_recall_fscore_support(y_true, y_pred, average = 'weighted')
    ## model results
    model_results = {"Accuracy Score": model_accuracy,
                     "Precision Score" : model_precision,
                     "Recall Score": model_recall,
                     "F1 Score": model_f1_score}

    ## returing results
    return model_results        
  
  
