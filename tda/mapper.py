from sklearn_tda import MapperComplex
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


def identify_membership(pick):
    """
    Given a set of input parameters, re-run Mapper and create a dataframe that
    identifies which participants belong to which features. Note that this 
    function retains significant features only.
    """
    # Re-run Mapper
    mapper = MapperComplex(**pick['params']).fit(pick['X'])
    # Get nodes for each feature
    nodes = {k: v['indices'] for k, v in mapper.node_info_.items()}
    # Get node memberships for each participant
    pid = {}
    for i in range(np.shape(pick['X'])[0]):
        pid[i] = {}
        pid[i]['nodes'] = []
        for k, v in nodes.items():
            if i in v:
                pid[i]['nodes'].append(k)
    # Get feature memberships
    feature_types = ['downbranch', 'upbranch', 'connected_component', 'loop']
    for i in feature_types:
        id = 1
        for f in pick['sbnd'][i]:
            if not all([len(i) == 0 for i in pick['sbnd'][i]]):
                # Only compute feature types that exist
                feature = i[0].upper() + str(id)
                id += 1
                for pk, pv in pid.items():
                    if len(set(f).intersection(pv['nodes'])) > 0:
                        pid[pk][feature] = True
                    else:
                        pid[pk][feature] = False
    membership = pd.DataFrame.from_dict(pid, orient='index')
    return(membership)


def remove_small_features(membership, prop=0.1):
    for col in list(membership)[1:]:
        y_prop = np.mean(membership[col])
        if (y_prop > (1-prop)) or (y_prop < prop):
            membership = membership.drop(col, axis=1)
    return(membership)


def count_sig(sbnd):
    tot = 0
    for k, v in sbnd.items():
        tot += len(v)
    return(tot)


def get_imps(X, clf):
    imp = pd.DataFrame(clf.feature_importances_,
                       list(X),
                       columns=['imp'])
    return(imp.sort_values(by='imp',
                           ascending=False)[:10].reset_index())


def get_scores(X, y, clf):
    scores = cross_val_score(clf,
                             X.values,
                             y,
                             cv=3,
                             scoring='roc_auc')
    return({'max': np.max(scores),
            'mean': np.mean(scores)})


def predict_feature_membership(X, membership, XGBoost=True):
    """
    Predicts feature membership with RF or XGBoost based on baseline clinical
    variables, using 3-fold CV. Returns max/mean AUC
    scores AND the top 10 most influential features.
    """
    outcomes = membership.drop(['nodes'], axis=1)
    results = {}
    for c, y in outcomes.iteritems():
        i = {}
        if XGBoost:
            clf = XGBClassifier(random_state=42).fit(X.values, y)
            i['auc'] = get_scores(X, y, clf)
            i['imp'] = get_imps(X, clf)
        else:
            clf = RandomForestClassifier(n_estimators=100,
                                         max_depth=2,
                                         random_state=42).fit(X.values, y)
            i['auc'] = get_scores(X, y, clf)
            i['imp'] = get_imps(X, clf)
        results[c] = i
    return(results)


def create_table_of_predictions_results(list_of_solutions):
    # Get best-fitting feature for each graph
    for i in list_of_solutions:
        if i['prediction'] == "No features large enough":
            i['summary'] = [np.nan, np.nan]
        else:
            # Pick the best-fitting feature, if graph has multiple significant
            # topological features
            highest = 0
            for k, v in i['prediction'].items():
                if v['auc']['mean'] > highest:
                    highest = v['auc']['mean']
                    feat = k
            i['summary'] = [feat, highest]
    # Produce a table of best-fitting features, across all graphs
    results = {k: v for k, v in enumerate(list_of_solutions)}
    perf = pd.DataFrame([(k,
                          v['fil_lab'],
                          v['summary'][0],
                          v['summary'][1]) for k, v in results.items()],
                        columns=['model', 'filter', 'feature', 'mean_auc'])
    perf = perf.sort_values(by=['mean_auc'], ascending=False).dropna()
    return(perf)
