from math import isnan
import numpy as np
import pandas as pd


def get_solution_vec(solutions, tags=None, positive_tags=None, negative_tags=None):
    """
    get solutions as string and return solutions as indices
    :param solutions: List[string]
    :param tags: List[string]
    :param positive_tags: List[string]
    :param negative_tags: List[string]
    :return: np.array
    """
    solutions = pd.Series(solutions)

    if tags:
        for i, tag in enumerate(tags):
            solutions = solutions.replace(tag, i)

    elif positive_tags and negative_tags:
        for pos_tag in positive_tags:
            solutions = solutions.replace(pos_tag, 1)
        for neg_tag in negative_tags:
            solutions = solutions.replace(neg_tag, 0)

    else:
        raise ValueError('ether tags or positive_tags and negative_tags need to be passed to the method')

    return solutions.as_matrix()


# rebuilt function:
#  - calculate confusion matrix
#  - calculate metrics for all classes with a single function call
#  - returns metrics as array - one element per class (metrics['metric'].sum() / len(metrics['metric']) => avg)
#  - category no longer needed
def calculate_metrics(clf, X, Y, avg_metrics):
    prediction = clf.predict(X)
    num_classes = prediction.shape[1]
    if len(prediction.shape) > 1:
        prediction = np.argmax(prediction, axis=1)
    if len(Y.shape) > 1:
        Y = np.argmax(Y, axis=1)

    confusion_matrix = np.zeros([num_classes, num_classes])

    for actual_class in range(num_classes):
        indices = np.where(Y == actual_class)
        for p in prediction[indices]:
            confusion_matrix[actual_class, p] += 1

    tp = np.zeros(num_classes)

    for c in range(confusion_matrix.shape[0]):
        tp[c] = confusion_matrix[c, c]

    fp = confusion_matrix.sum(axis=0) - tp
    fn = confusion_matrix.sum(axis=1) - tp
    tn = np.array([confusion_matrix.sum()] * num_classes) - tp - fp - fn

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    f1 = 2 * (precision * recall) / (precision + recall)
    accuracy = (tp + tn) / len(prediction)

    if avg_metrics:
        metrics_data = {
            'accuracy': accuracy.sum() / len(accuracy),
            'f1': f1.sum() / len(f1),
            'precision': precision.sum() / len(precision),
            'recall': recall.sum() / len(recall),
        }

        for key, value in metrics_data.items():
            if isnan(value):
                metrics_data[key] = 0

    else:
        metrics_data = {
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'confusion_matrix': confusion_matrix
        }

    return metrics_data


def encode_one_hot(vector):
    try:
        vec = np.zeros([len(vector), np.max(vector) + 1], dtype=np.float)
        vec[np.arange(0, len(vector)), vector] = 1
    except IndexError:
        raise RuntimeError
    return vec
