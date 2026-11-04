"""Small display/evaluation helpers. Training remains in qtabgan/model.py."""
from contextlib import contextmanager
import random
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score, brier_score_loss, confusion_matrix


@contextmanager
def evaluation_rng(seed):
    """Keep CPU analytic evaluations from changing subsequent training randomness."""
    states=(random.getstate(),np.random.get_state(),torch.get_rng_state())
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    try:yield
    finally:
        random.setstate(states[0]);np.random.set_state(states[1]);torch.set_rng_state(states[2])


def classifier(x,y,seed):
    return RandomForestClassifier(n_estimators=100,random_state=seed,n_jobs=1).fit(x,y)


def score(clf,x,y):
    predicted=clf.predict(x)
    positive=clf.predict_proba(x)[:,list(clf.classes_).index(1)]
    return {'accuracy':float(accuracy_score(y,predicted)),
            'macro_f1':float(f1_score(y,predicted,average='macro',zero_division=0)),
            'auroc':float(roc_auc_score(y,positive)),
            'average_precision':float(average_precision_score(y,positive)),
            'brier_mse':float(brier_score_loss(y,positive))},predicted,positive


def diversity(real,fake,labels,numerical,categorical):
    """Compare distributions within each class; rows have no one-to-one pairing."""
    rows=[]
    for column in numerical:
        a=real[column].to_numpy(dtype=float);b=fake[column].to_numpy(dtype=float)
        ratios=[]
        for label in np.unique(labels):
            mask=labels==label
            if a[mask].std()>0:ratios.append(float(b[mask].std()/a[mask].std()))
        rows.append({'feature':column,'real_mean':float(a.mean()),'fake_mean':float(b.mean()),
                     'real_std':float(a.std()),'fake_std':float(b.std()),
                     'minimum_within_class_std_ratio':min(ratios) if ratios else None,
                     'outside_train_range_fraction':float(((b<a.min())|(b>a.max())).mean())})
    categories=pd.DataFrame([{'feature':c,'real_categories':real[c].nunique(),
        'fake_categories':fake[c].nunique(),'coverage':len(set(real[c])&set(fake[c]))/real[c].nunique()}
        for c in categorical])
    return pd.DataFrame(rows),categories
