import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import pygraphviz as pgv
import statmapper as stm


# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃                                                                           ┃
# ┃                          Summmarise Mapper graph                          ┃
# ┃                                                                           ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛


def graph_summary(M):
    """
    Return dictionary with key graph characteristics
    Expects a dictionary
    """
    feature_types = ['downbranch', 'upbranch', 'connected_component', 'loop']
    # Get number of significant features
    n_feat, n_sig = 0, 0
    for topo in feature_types:
        n_feat = n_feat + len(M['dgm'][topo])
        n_sig = n_sig + len(M['sdgm'][topo])
    ret = {'n_feat': n_feat,
           'n_sig': n_sig}
    for i in ['dgm', 'sdgm']:
        for j in feature_types:
            ret[i + '_' + j] = len(M[i][j])
    # Get number of people inside each significant feature
    feature_sizes = []
    for s in M['sbnd']:
        for topo in M['sbnd'][s]:
            feature_sizes.append(len(topo))
    if feature_sizes:
        ret['median_size'] = np.median(feature_sizes)
        ret['max_size'] = np.max(feature_sizes)
        ret['morethan2'] = len([x for x in feature_sizes if x > 2])
        ret['morethan5'] = len([x for x in feature_sizes if x > 5])
    # Get parameter values
    ret['reso'] = M['params']['resolutions'][0]
    ret['gain'] = M['params']['gains'][0]
    ret['eps'] = M['params']['clustering'].get_params()['eps']
    ret['filter'] = M['fil_lab'],
    ret['X'] = M['X_lab'],
    return({k: [v] for k, v in ret.items()})


"""
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃                                                                             ┃
┃                                  graphviz                                   ┃
┃                                                                             ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
"""


def get_colors(v):
    """Convert input array to list of colors suitable for plotting"""
    if isinstance(v[0], np.int64):
        palette = plt.get_cmap('Pastel2')
        labels = np.repeat('', len(v))
    else:
        palette = plt.get_cmap('YlGnBu')
        labels = ['{:1.1f}'.format(x) for x in v]
    colors = ['{:1.4f}'.format(x[0]) +
              ' ' + '{:1.4f}'.format(x[1]) +
              ' ' + '{:1.4f}'.format(x[2]) for x in palette(v)]
    return(labels, colors)


def get_mean(graph, var):
    """
    Extract means for each node in a Mapper graph

    graph: Mapper graph produced by MapperComplex
    var:   A variable (e.g. pd.Series) where row order matches ordering in
           point cloud used in Mapper
    """
    res = []
    for i in graph.node_info_.items():
        res.append(np.mean(var[i[1]['indices']]))
    return(res)


def to_graphviz(graph, marker=None, size=True, scale=10):
    """
    Convert Mapper graph to graphviz

    graph:  Mapper object returned by MapperComplex
    marker: An optional array giving numeric values with which to color nodes
    size:   Size nodes based on membership
    scale:  Scale node sizes, or set fixed size (depends on value of 'size')
    """
    # Get node colors/labels
    if marker is None:
        for d in graph.node_info_.items():
            d[1]['nodecolor'] = 'white'
            d[1]['label'] = ''
    else:
        labels, colors = get_colors(marker)
        for d, c, l in zip(graph.node_info_.items(),
                           colors,
                           labels):
            d[1]['nodecolor'] = c
            d[1]['label'] = l

    if not size:
        for d in graph.node_info_.items():
            d[1]['size'] = 10

    # Construct the graph
    G = pgv.AGraph()
    G.node_attr['style'] = 'filled'
    G.node_attr['fixedsize'] = 'true'
    # Add nodes
    for k, v in graph.node_info_.items():
        G.add_node(k,
                   label='id:' + str(k) + '\n' + str(v['label']),
                   width=v['size']/scale,
                   height=v['size']/scale,
                   fontsize=10,
                   fontname='Lato',
                   fillcolor=v['nodecolor'],
                   fontcolor='black')
    # Add edges
    for v in graph.mapper_.get_skeleton(2):
        if len(v[0]) == 2:
            G.add_edge(v[0][0], v[0][1])
    return(G)


def graph_features(map, sbnd):
    """
    Draw significant features on a Mapper graph

    map:  output from MapperComplex
    sbnd: list containing the features to plot, produced by the
          representative_features function.
    """

    feature_types = ['downbranch', 'upbranch', 'connected_component', 'loop']

    # Get type and instance for each significant feature
    nodeinfo = {}
    for type, color in zip(feature_types, [1, 2, 3, 4]):
        for i, feat in enumerate(sbnd[type]):
            for k in feat:
                nodeinfo[k] = (color, i, str(type[0].upper()) + str(i))

    # Construct the graph
    G = pgv.AGraph()
    G.graph_attr['bgcolor'] = 'gray75'
    G.node_attr['style'] = 'filled'
    G.node_attr['shape'] = 'circle'
    G.node_attr['fixedsize'] = 'true'
    G.node_attr['width'] = '0.4'
    G.node_attr['fontname'] = 'Arial'
    # Add nodes
    for k, v in map.node_info_.items():
        if k in nodeinfo.keys():
            G.add_node(k,
                       fillcolor=nodeinfo[k][0],
                       label=nodeinfo[k][2],
                       colorscheme='set34')
        else:
            G.add_node(k, label='')
    for v in map.mapper_.get_skeleton(2):
        if len(v[0]) == 2:
            start, end = v[0][0], v[0][1]
            edgecolor = '#EAEAEA'
            w = 1
            if start in nodeinfo.keys() and end in nodeinfo.keys():
                w = 3
                edgecolor = str(nodeinfo[start][1]) + ';0.5:' + str(nodeinfo[end][1])
            G.add_edge(start,
                       end,
                       colorscheme='paired12',
                       penwidth=w,
                       color=edgecolor)
    return(G)


# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃                                                                           ┃
# ┃                                  networkx                                 ┃
# ┃                                                                           ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛


def default_plot(graph, ax):
    ax = ax or plt.gca()
    return(nx.draw(graph,
                   with_labels=True,
                   pos=nx.spring_layout(graph, iterations=50),
                   node_color='black',
                   edge_color='gray',
                   font_color='white',
                   node_size=300))


def to_networkx(map):
    nodes = map.node_info_.keys()
    edges = []
    for v in map.mapper_.get_skeleton(2):
        if len(v[0]) == 2:
            edges.append((v[0][0], v[0][1]))
    g = nx.Graph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    return(g)
