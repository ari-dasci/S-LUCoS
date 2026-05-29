from sklearn.metrics import roc_auc_score
from collections import Counter

def auc_score(y_test, logits):
    if logits.shape[1] == 2:
        # Standard binary AUC
        return float(roc_auc_score(y_test, logits[:, 1]))
    else:
        # One-vs-Rest AUC for multiclass
        return float(roc_auc_score(y_test, logits, multi_class="ovr"))

def imbalance_ratio(y):
    # ratio of the majority class over the minority class
    counts = Counter(y) # examples per class
    return max(counts.values()) / min(counts.values())
