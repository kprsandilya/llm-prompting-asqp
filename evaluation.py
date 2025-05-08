from sklearn.metrics import precision_recall_fscore_support
import numpy as np

def compute_scores_single(pred_pt, gold_pt, evaluation_type):
    """
    Function to compute F1 scores with pred and gold quads
    The input needs to be already processed
    """
    idx_sentiment_element = {"single_at": 0, "single_ac": 1, "single_pol": 2, "single_ot": 3}
    
    pred_pt = [list(set([tuple[idx_sentiment_element[evaluation_type]] for tuple in pred])) for pred in pred_pt]
    pred_pt = [[t for t in pred if t != "NULL"] for pred in pred_pt]
    
    gold_pt = [list(set([tuple[idx_sentiment_element[evaluation_type]] for tuple in gold])) for gold in gold_pt]
    gold_pt = [[t for t in gold if t != "NULL"] for gold in gold_pt]
    
    
    n_tp, n_fp, n_fn = 0, 0, 0
    n_gold, n_pred = 0, 0

    for i in range(len(pred_pt)):
        n_gold += len(gold_pt[i])
        n_pred += len(pred_pt[i])    

        # Compute True Positives and False Positives
        for t in pred_pt[i]:
            if t in gold_pt[i]:
                n_tp += 1
            else:
                n_fp += 1

        # Compute False Negatives
        for t in gold_pt[i]:
            if t not in pred_pt[i]:
                n_fn += 1

    precision = float(n_tp) / float(n_pred) if n_pred != 0 else 0
    recall = float(n_tp) / float(n_gold) if n_gold != 0 else 0
    f1 = 2 * precision * recall / \
        (precision + recall) if precision != 0 or recall != 0 else 0

    scores = {
        'precision': precision * 100,
        'recall': recall * 100,
        'f1': f1 * 100,
        'TP': n_tp,
        'FP': n_fp,
        'FN': n_fn
    }

    return scores

def count_regenerations(data):
    return { "invalid_precitions_label" : sum([len(sample["invalid_precitions_label"]) for sample in data]), 
              "mean_invalid_precitions_label" : np.mean([len(sample["invalid_precitions_label"]) for sample in data]),
              "median_invalid_precitions_label" : np.median([len(sample["invalid_precitions_label"]) for sample in data]) }

def compute_f1_scores_quad(pred_pt, gold_pt):
    """
    Function to compute F1 scores with pred and gold quads
    The input needs to be already processed
    """
    n_tp, n_fp, n_fn = 0, 0, 0
    n_gold, n_pred = 0, 0

    for i in range(len(pred_pt)):
        n_gold += len(gold_pt[i])
        n_pred += len(pred_pt[i])

        # Compute True Positives and False Positives
        for t in pred_pt[i]:
            if t in gold_pt[i]:
                n_tp += 1
            else:
                n_fp += 1

        # Compute False Negatives
        for t in gold_pt[i]:
            if t not in pred_pt[i]:
                n_fn += 1

    precision = float(n_tp) / float(n_pred) if n_pred != 0 else 0
    recall = float(n_tp) / float(n_gold) if n_gold != 0 else 0
    f1 = 2 * precision * recall / \
        (precision + recall) if precision != 0 or recall != 0 else 0

    scores = {
        'precision': precision * 100,
        'recall': recall * 100,
        'f1': f1 * 100,
        'TP': n_tp,
        'FP': n_fp,
        'FN': n_fn
    }

    return scores


def compute_scores_acd(pred_pt, gold_pt, unique_aspect_categories):
    categories_pred = [[tuple[1] for tuple in pred] for pred in pred_pt]
    categories_gold = [[tuple[1] for tuple in pred] for pred in gold_pt]
    one_hot_pred = [[1 if category in categories_pred[i]
                     else 0 for category in unique_aspect_categories] for i in range(len(categories_pred))]
    one_hot_gold_pt = [[1 if category in categories_gold[i]
                        else 0 for category in unique_aspect_categories] for i in range(len(categories_gold))]

    # Berechnung von Mikro- und Makro-F1-Score
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        one_hot_gold_pt, one_hot_pred, average='macro')
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
        one_hot_gold_pt, one_hot_pred, average='micro')

    scores = {
        'precision_macro': precision_macro * 100,
        'recall_macro': recall_macro * 100,
        'f1_macro': f1_macro * 100,
        'precision': precision_micro * 100,
        'recall': recall_micro * 100,
        'f1': f1_micro * 100,
    }
    return scores


def compute_scores_acsa(pred_pt, gold_pt, unique_aspect_categories):
    categories_sentiments_pred = [
        [(tuple[1], tuple[2]) for tuple in pred] for pred in pred_pt]
    categories_sentiments_gold = [
        [(tuple[1], tuple[2]) for tuple in pred] for pred in gold_pt]

    unique_category_sentiments = list(set(
        [cs for sublist in categories_sentiments_gold + categories_sentiments_pred for cs in sublist]))

    one_hot_pred = [[1 if cs in categories_sentiments_pred[i] else 0 for cs in unique_category_sentiments]
                    for i in range(len(categories_sentiments_pred))]
    one_hot_gold_pt = [[1 if cs in categories_sentiments_gold[i] else 0 for cs in unique_category_sentiments]
                       for i in range(len(categories_sentiments_gold))]

    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        one_hot_gold_pt, one_hot_pred, average='macro')
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
        one_hot_gold_pt, one_hot_pred, average='micro')

    scores = {
        'precision_macro': precision_macro * 100,
        'recall_macro': recall_macro * 100,
        'f1_macro': f1_macro * 100,
        'precision': precision_micro * 100,
        'recall': recall_micro * 100,
        'f1': f1_micro * 100,
    }
    return scores


def evaluate(pred_pt, gold_pt, unique_aspect_categories):
    scores = {}

    scores.update({"acd": compute_scores_acd(
        pred_pt, gold_pt, unique_aspect_categories)})
    scores.update({"acsa": compute_scores_acsa(
        pred_pt, gold_pt, unique_aspect_categories)})
    scores.update(compute_f1_scores_quad(
        pred_pt, gold_pt))

    return scores
