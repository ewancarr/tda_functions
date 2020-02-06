import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import pygraphviz as pgv
import statmapper as stm

def matches(item, comparison):
    comparison = ["{:10.10f}".format(i) for i in comparison]
    return("{:10.10f}".format(item) in comparison)


def extract_xy(d):
    x = np.array([i[1][0] for i in d])
    y = np.array([i[1][1] for i in d])
    return([x, y])


def compare_features(M, computed, ax=None):
    ax = ax or plt.gca()
    x0, y0 = extract_xy(M.compute_persistence_diagrams()[0])
    x1, y1 = extract_xy(M.compute_persistence_diagrams()[0])

    x_orig = np.hstack([x0, x1])
    y_orig = np.hstack([y0, y1])

    x_comp, y_comp = extract_xy(computed)

    color = []
    for i in range(len(x_orig)):
        if matches(x_orig[i], x_comp) and matches(y_orig[i], y_comp):
            color.append('red')
        else:
            color.append('grey')

    # Make the plot
    dag = np.arange(np.min(x_orig), np.max(x_orig))
    _ = ax.scatter(x_orig, y_orig, c=color)
    _ = ax.plot(dag, dag)
    _ = ax.set_xlabel('Birth')
    _ = ax.set_ylabel('Death')
    return(_)


def convert_gain(gain_aysadi):
    return(1-(1/gain_aysadi))


# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃                                                                           ┃
# ┃          Functions to count/summarise features in Mapper graphs           ┃
# ┃                                                                           ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛


def dg(M, X=None):
    """Describe a Mapper graph"""
    # Get total number of rows in input data
    if X is not None:
        print(np.shape(X)[0], 'rows in input data')
    # Get total number of nodes; unique members
    members = np.array([])
    for k, v in M.node_info_.items():
        members = np.hstack([members, v['indices']])
    print(len(M.node_info_), 'nodes')
    print(len(np.unique(members)), 'unique members')
    print('┌────────┬─────────────┐')
    print('│ ID     │ N members   │')
    print('├────────┼─────────────┤')
    for k, v in M.node_info_.items():
        print('│ ', k, ' ' * (4 - len(str(k))),
              '│', v['size'], ' ' * (10 - len(str(v['size']))), '│')
    print('└────────┴─────────────┘')


def count_features(M):
    feature_types = ['downbranch', 'upbranch', 'connected_component', 'loop']
    n_feat, n_sig = 0, 0
    for topo in feature_types:
        n_feat = n_feat + len(M['dgm'][topo])
        n_sig = n_sig + len(M['sdgm'][topo])
    res = [n_feat, n_sig]
    for i in ['dgm', 'sdgm']:
        for j in feature_types:
            res.append(len(M[i][j]))
    return(res)


def representative_features(M, confidence, bootstrap, inp):
    features = ['downbranch', 'upbranch',
                'connected_component', 'loop']
    for topo in features:
        # Compute and save representative features
        if 'dgm' not in M.keys():
            M['dgm'], M['bnd'] = {}, {}
        dgm, bnd = stm.compute_topological_features(M=M['map'],
                                                    func=M['fil'][:, 0],
                                                    func_type="data",
                                                    topo_type=topo,
                                                    threshold=confidence)
        M['dgm'][topo] = dgm
        M['bnd'][topo] = bnd
        # Run bootstrap for representative features
        if 'sdgm' not in M.keys():
            M['sdgm'], M['sbnd'] = {}, {}
        sdgm, sbnd = stm.evaluate_significance(dgm=M['dgm'][topo],
                                               bnd=M['bnd'][topo],
                                               X=M['X'],
                                               M=M['map'],
                                               func=M['fil'],
                                               params=M['params'],
                                               topo_type=topo,
                                               threshold=confidence,
                                               N=bootstrap,
                                               input=inp)
        M['sdgm'][topo] = sdgm
        M['sbnd'][topo] = sbnd
    return(M)
