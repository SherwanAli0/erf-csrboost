"""The three models compared by the app:

- plain_model      : standard RandomForest + class weights (the one-line baseline)
- csrboost_exact   : CSRBoost (Yadav et al., IEEE Access 2025) as replicated for this project
- erf_csrboost     : ERF-CSRBoost, an experimental, unpublished extension of CSRBoost
"""
from .plain_model import make_plain
from .csrboost_exact import csrboost_fit_predict
from .erf_csrboost import ERFModel, train_erf, N_BAGS, N_TREES
