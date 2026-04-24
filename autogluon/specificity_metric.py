from autogluon.core.metrics import make_scorer
import sklearn.metrics


def specificity_score(y_true, y_pred):
    return sklearn.metrics.recall_score(y_true, y_pred, pos_label=0)


ag_specificity_scorer = make_scorer(
    name="specificity",
    score_func=specificity_score,
    optimum=1,
    greater_is_better=True,
)
