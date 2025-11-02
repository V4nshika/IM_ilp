import pulp
from IM_ilp.Helper_Functions import preprocess_graph, extract_activities
import numpy as np

def nx_to_mat_and_weights_xor(G):
    nodes = list(G.nodes())
    node_index = {node: i for i, node in enumerate(nodes)}
    n = len(nodes)
    A = np.zeros((n, n), dtype=int)
    W = np.zeros((n, n), dtype=float)
    for u, v, data in G.edges(data=True):
        i = node_index[u]
        j = node_index[v]
        A[i, j] = 1
        W[i, j] = data.get('weight', 1.0)
    return A, W, nodes, node_index


def xor_cut_ilp(G, sup = 1.0):

    start_node = 'start'
    end_node = 'end'

    # Preprocess graph
    reduced_graph, _, _ = preprocess_graph(G, start_node, end_node)
    A, W, node_names, node_index = nx_to_mat_and_weights_xor(reduced_graph)
    n = A.shape[0]

    start_idx = node_names.index(start_node)
    end_idx = node_names.index(end_node)
    non_terminal = [i for i in range(n) if i != start_idx and i != end_idx]

    # ILP Model
    prob = pulp.LpProblem("XOR_MinCut_Flow_Corrected", pulp.LpMinimize)

    # Variables
    x = pulp.LpVariable.dicts("x", non_terminal, cat=pulp.LpBinary)
    y = pulp.LpVariable.dicts("y", [(i,j) for i in range(n) for j in range(n) if A[i,j]], cat=pulp.LpBinary)

    # Flow variables from start
    f_s = pulp.LpVariable.dicts("f_s", [(i,j) for i in range(n) for j in range(n) if A[i,j]], lowBound=0, cat='Integer')
    # Flow variables to end
    f_e = pulp.LpVariable.dicts("f_e", [(i,j) for i in range(n) for j in range(n) if A[i,j]], lowBound=0, cat='Integer')

    ### Constraints ###

    # Don't cut start -> * or * -> end edges
    for i in range(n):
        for j in range(n):
            if A[i,j]:
                if i == start_idx or j == end_idx:
                    prob += y[(i,j)] == 0
                elif i in non_terminal and j in non_terminal:
                    prob += y[(i,j)] >= x[i] - x[j]
                    prob += y[(i,j)] >= x[j] - x[i]

    # non-trivial partition
    prob += pulp.lpSum(x[i] for i in non_terminal) >= 1
    prob += pulp.lpSum(1 - x[i] for i in non_terminal) >= 1

    # reachability FROM start
    prob += pulp.lpSum(f_s[(start_idx, j)] for j in range(n) if A[start_idx, j]) == n - 1

    for i in range(n):
        if i == start_idx:
            continue
        inflow = pulp.lpSum(f_s[(j, i)] for j in range(n) if A[j, i])
        outflow = pulp.lpSum(f_s[(i, j)] for j in range(n) if A[i, j])
        prob += (inflow - outflow) == 1

    for (i, j) in f_s:
        prob += f_s[(i, j)] <= (n - 1) * (1 - y[(i, j)])

    # reachability TO end
    prob += pulp.lpSum(f_e[(i, end_idx)] for i in range(n) if A[i, end_idx]) == n - 1

    for i in range(n):
        if i == end_idx:
            continue
        inflow = pulp.lpSum(f_e[(j, i)] for j in range(n) if A[j, i])
        outflow = pulp.lpSum(f_e[(i, j)] for j in range(n) if A[i, j])
        prob += (outflow - inflow) == 1

    for (i, j) in f_e:
        prob += f_e[(i, j)] <= (n - 1) * (1 - y[(i, j)])

    # objective: minimize total cut weight
    prob += pulp.lpSum(W[i,j] * y[(i,j)] for (i,j) in y.keys())

    # Solve
    status = prob.solve(pulp.GUROBI_CMD(msg=False))

    # Extract results
    Sigma_1 = [node_names[i] for i in non_terminal if pulp.value(x[i]) == 0]
    Sigma_2 = [node_names[i] for i in non_terminal if pulp.value(x[i]) == 1]
    total_cost = pulp.value(prob.objective)

    if not Sigma_1 and not Sigma_2:
        total_cost = None

    return (
        extract_activities(Sigma_1),
        extract_activities(Sigma_2),
        total_cost,
        node_names
    )


def xor_cut_tau(G, sup=1.0):
    total_cases = sum(data["weight"] for src, tgt, data in G.edges(data=True) if src == "start")
    start_to_end_weight = G.get_edge_data('start', 'end', {'weight': 0})['weight']
    #total_cost = total_cases * sup - start_to_end_weight
    total_cost = max(0, total_cases * sup - start_to_end_weight)  # Add max(0, ...) here
    nodes = list(G.nodes())

    Sigma_1 = extract_activities(nodes)  # All regular activities
    Sigma_2 = set()  # Represents tau/empty

    return Sigma_1, Sigma_2, total_cost, nodes
