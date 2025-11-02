import networkx as nx
import json

def n_edges(net, S, T):
    edges_reweight = list(nx.edge_boundary(net, S, T, data='weight', default=1))
    return sum(weight for u, v, weight in edges_reweight if (u in S and v in T))


def get_edge_weight(graph, node1, node2):
    edge_data = graph.get_edge_data(node1, node2)
    if edge_data is None:
        return 0
    else:
        return edge_data['weight']

def add_SE(nodes, st,en):
    if nodes & set(st.keys()):
        nodes.add('start')
    if nodes & set(en.keys()):
        nodes.add('end')
    return nodes


def aggregate_dictionaries(deviating, missing):
    aggregated_dict = {}

    for d in deviating:
        for key, value in d.items():
            if key in aggregated_dict:
                aggregated_dict[key]['deviating'] += value
            else:
                aggregated_dict[key] = {'deviating':value,'missing':0}

    for d in missing:
        for key, value in d.items():
            if key in aggregated_dict:
                aggregated_dict[key]['missing'] += value
            else:
                aggregated_dict[key] = {'deviating':0,'missing':value}
    return aggregated_dict


def generate_nx_graph_from_log(log,nt):
    G = nx.DiGraph()
    for trace in log:
        tr_art = ('start',) + trace + ('end',)
        for i in range(len(tr_art) - 1):
            if G.has_edge(tr_art[i], tr_art[i + 1]):
                G[tr_art[i]][tr_art[i + 1]]['weight'] += log[trace]
            else:
                G.add_edge(tr_art[i], tr_art[i + 1], weight=log[trace])

    st = {}
    en = {}
    # Filter outgoing edges from start_node based on weight
    outgoing_edges = list(G.edges('start', data=True))
    for u, v, data in outgoing_edges:
        if data['weight'] < nt:
            G.remove_edge(u, v)
            if not nx.has_path(G, 'start', v):
               G.add_edge(u, v, weight=data['weight'])
               st[v] = data['weight']
        else:
            st[v] = data['weight']

    # Filter incoming edges to end_node based on frequency
    incoming_edges = list(G.in_edges('end', data=True))
    for u, v, data in incoming_edges:
        if data['weight'] < nt:
            G.remove_edge(u, v)
            if not nx.has_path(G, u, 'end'):
               G.add_edge(u, v, weight=data['weight'])
               en[u] = data['weight']
        else:
            en[u] = data['weight']
    return G

def generate_nx_indirect_graph_from_log(log):
    G = nx.DiGraph()
    for trace in log:
        for i in range(0,len(trace)):
            visited = set()
            for j in range(i+1,len(trace)):
                if trace[j] not in visited:
                    visited.add(trace[j])
                    if G.has_edge(trace[i], trace[j]):
                        G[trace[i]][trace[j]]['weight'] += log[trace]
                    else:
                        G.add_edge(trace[i], trace[j], weight=log[trace])
    return G

def read_append_write_json(file_path, new_item):
    # Read the current content of the JSON file
    with open(file_path, 'r') as file:
        data = json.load(file)

    # Append the new item to the list
    data.append(new_item)

    # Write the updated list back to the JSON file
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)

