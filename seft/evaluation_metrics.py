"""Module containing convinience wrapped evaluation metrics."""
import numpy as np

from sklearn.metrics import average_precision_score, roc_auc_score,\
    accuracy_score, mean_absolute_error, mean_squared_error, balanced_accuracy_score


def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / (y_true + 0.1))) * 100


def to_one_hot(a, num_classes):
    return np.squeeze(np.eye(num_classes)[a.reshape(-1)])


def mc_metric_wrapper(metric, **kwargs):
    """Wrap metric for multi class classification.

    If classifiction task is binary, select minority label as positive.
    Otherwise compute weighted average over classes.
    """
    def wrapped(y_true, y_score):
        if y_true.ndim == 1 and y_score.ndim == 2:
            # Multi class classification task where gt is given as int class
            # indicator. First need to convert to one hot label.
            n_classes = y_score.shape[-1]
            y_true = to_one_hot(y_true, n_classes)
        return metric(y_true, y_score, **kwargs)
    return wrapped


def accuracy(y_true, y_score):
    """Compute accuracy using one-hot representaitons."""
    if isinstance(y_true, list) and isinstance(y_score, list):
        # Online scenario
        if y_true[0].ndim == 2 and y_score[0].ndim == 2:
            # Flatten to single (very long prediction)
            y_true = np.concatenate(y_true, axis=0)
            y_score = np.concatenate(y_score, axis=0)
    if y_score.ndim == 3 and y_score.shape[-1] == 1:
        y_score = np.ravel(y_score)
        y_true = np.ravel(y_true).astype(int)
        y_score = np.around(y_score).astype(int)
    if y_true.ndim == 2 and y_true.shape[-1] != 1:
        y_true = np.argmax(y_true, axis=-1)
    if y_true.ndim == 2 and y_true.shape[-1] == 1:
        y_true = np.round(y_true).astype(int)
    if y_score.ndim == 2 and y_score.shape[-1] != 1:
        y_score = np.argmax(y_score, axis=-1)
    if y_score.ndim == 2 and y_score.shape[-1] == 1:
        y_score = np.round(y_score).astype(int)
    return accuracy_score(y_true, y_score)

def balanced_accuracy(y_true, y_score):
    """Compute accuracy using one-hot representaitons."""
    if isinstance(y_true, list) and isinstance(y_score, list):
        # Online scenario
        if y_true[0].ndim == 2 and y_score[0].ndim == 2:
            # Flatten to single (very long prediction)
            y_true = np.concatenate(y_true, axis=0)
            y_score = np.concatenate(y_score, axis=0)
    if y_score.ndim == 3 and y_score.shape[-1] == 1:
        y_score = np.ravel(y_score)
        y_true = np.ravel(y_true).astype(int)
        y_score = np.around(y_score).astype(int)
    if y_true.ndim == 2 and y_true.shape[-1] != 1:
        y_true = np.argmax(y_true, axis=-1)
    if y_true.ndim == 2 and y_true.shape[-1] == 1:
        y_true = np.round(y_true).astype(int)
    if y_score.ndim == 2 and y_score.shape[-1] != 1:
        y_score = np.argmax(y_score, axis=-1)
    if y_score.ndim == 2 and y_score.shape[-1] == 1:
        y_score = np.round(y_score).astype(int)
    return balanced_accuracy_score(y_true, y_score)


def mgp_wrapper(fn):
    def wrapped(y_true, y_score):
        if isinstance(y_true, list) and isinstance(y_score, list):
            # Online scenario
            if y_true[0].ndim == 2:
                # Flatten to single (very long prediction)
                y_true = np.concatenate(y_true, axis=0)
                y_score = np.concatenate(y_score, axis=0)

        assert y_true.size == y_score.size
        return fn(np.ravel(y_true), np.ravel(y_score))
    return wrapped


def compute_prediction_utility(labels, predictions, dt_early=-12,
                               dt_optimal=-6, dt_late=3.0, max_u_tp=1,
                               min_u_fn=-2, u_fp=-0.05, u_tn=0,
                               check_errors=True):
    """Compute utility score of physionet 2019 challenge."""
    # Check inputs for errors.
    if check_errors:
        if len(predictions) != len(labels):
            raise Exception('Numbers of predictions and labels must be the same.')

        for label in labels:
            if not label in (0, 1):
                raise Exception('Labels must satisfy label == 0 or label == 1.')

        for prediction in predictions:
            if not prediction in (0, 1):
                raise Exception('Predictions must satisfy prediction == 0 or prediction == 1.')

        if dt_early >= dt_optimal:
            raise Exception('The earliest beneficial time for predictions must be before the optimal time.')

        if dt_optimal >= dt_late:
            raise Exception('The optimal time for predictions must be before the latest beneficial time.')

    # Does the patient eventually have sepsis?
    if np.any(labels):
        is_septic = True
        t_sepsis = np.argmax(labels) - dt_optimal
    else:
        is_septic = False
        t_sepsis = float('inf')

    n = len(labels)

    # Define slopes and intercept points for utility functions of the form
    # u = m * t + b.
    m_1 = float(max_u_tp) / float(dt_optimal - dt_early)
    b_1 = -m_1 * dt_early
    m_2 = float(-max_u_tp) / float(dt_late - dt_optimal)
    b_2 = -m_2 * dt_late
    m_3 = float(min_u_fn) / float(dt_late - dt_optimal)
    b_3 = -m_3 * dt_optimal

    # Compare predicted and true conditions.
    u = np.zeros(n)
    for t in range(n):
        if t <= t_sepsis + dt_late:
            # TP
            if is_septic and predictions[t]:
                if t <= t_sepsis + dt_optimal:
                    u[t] = max(m_1 * (t - t_sepsis) + b_1, u_fp)
                elif t <= t_sepsis + dt_late:
                    u[t] = m_2 * (t - t_sepsis) + b_2
            # FP
            elif not is_septic and predictions[t]:
                u[t] = u_fp
            # FN
            elif is_septic and not predictions[t]:
                if t <= t_sepsis + dt_optimal:
                    u[t] = 0
                elif t <= t_sepsis + dt_late:
                    u[t] = m_3 * (t - t_sepsis) + b_3
            # TN
            elif not is_septic and not predictions[t]:
                u[t] = u_tn

    # Find total utility for patient.
    return np.sum(u)


def physionet2019_utility(y_true, y_score):
    """Compute physionet 2019 Sepsis eary detection utility.

    Code based on:
    

    Args:
        y_true:
        y_score:

    Returns:
    """
    dt_early = -12
    dt_optimal = -6
    dt_late = 3.0

    utilities = []
    best_utilities = []
    inaction_utilities = []

    for labels, observed_predictions in zip(y_true, y_score):
        observed_predictions = np.round(observed_predictions)
        num_rows = len(labels)
        best_predictions = np.zeros(num_rows)
        inaction_predictions = np.zeros(num_rows)

        if np.any(labels):
            t_sepsis = np.argmax(labels) - dt_optimal
            pred_begin = int(max(0, t_sepsis + dt_early))
            pred_end = int(min(t_sepsis + dt_late + 1, num_rows))
            best_predictions[pred_begin:pred_end] = 1

        utilities.append(
            compute_prediction_utility(labels, observed_predictions))
        best_utilities.append(
            compute_prediction_utility(labels, best_predictions))
        inaction_utilities.append(
            compute_prediction_utility(labels, inaction_predictions))

    unnormalized_observed_utility = sum(utilities)
    unnormalized_best_utility = sum(best_utilities)
    unnormalized_inaction_utility = sum(inaction_utilities)
    normalized_observed_utility = (
        (unnormalized_observed_utility - unnormalized_inaction_utility)
        / (unnormalized_best_utility - unnormalized_inaction_utility)
    )
    return normalized_observed_utility


auroc = mgp_wrapper(roc_auc_score)
auprc = mgp_wrapper(average_precision_score)
auprc_micro = mc_metric_wrapper(average_precision_score, average='micro')
auprc_macro = mc_metric_wrapper(average_precision_score, average='macro')
auprc_weighted = mc_metric_wrapper(average_precision_score, average='weighted')

auroc_micro = mc_metric_wrapper(roc_auc_score, average='micro')
auroc_macro = mc_metric_wrapper(roc_auc_score, average='macro')
auroc_weighted = mc_metric_wrapper(roc_auc_score, average='weighted')
