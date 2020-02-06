from sklearn_tda import MapperComplex
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier


def identify_membership(pick):
    """
    Given a set of input parameters, re-run Mapper and create a dataframe that
    identifies which participants belong to which features.
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


def remove_small_features(membership):
    for col in list(membership)[1:]:
        y_prop = np.mean(membership[col])
        if (y_prop > 0.9) or (y_prop < 0.1):
            membership = membership.drop(col, axis=1)
    return(membership)


def count_sig(sbnd):
    tot = 0
    for k, v in sbnd.items():
        tot += len(v)
    return(tot)


def predict_feature_membership(X, membership):
    """
    Returns max/mean prediction accuracy, predicting membership to each
    features based on baseline clinical variables. 3-fold CV, RF.
    """
    outcomes = membership.drop(['nodes'], axis=1)
    scores = []
    for c, y in outcomes.iteritems():
        clf = RandomForestClassifier(n_estimators=100,
                                     max_depth=2,
                                     random_state=42).fit(X, y)
        scores.append(np.mean(cross_val_score(clf, X, y, cv=3)))
    return({'max': np.max(scores),
            'mean': np.mean(scores)})


def create_table_of_predictions_results(list_of_solutions):
    results = {k: v for k, v in enumerate(list_of_solutions)}
    perf = pd.DataFrame([(k, v['max'], v['mean']) for k, v in results.items()],
                        columns=['model', 'max', 'mean'])
    perf = perf.sort_values(by=['max'], ascending=False).dropna()
    return(perf)


