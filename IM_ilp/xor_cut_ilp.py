import gurobipy as gp
from gurobipy import GRB
from IM_ilp.Helper_Functions import preprocess_graph, extract_activities
import numpy as np

# This helper function has no PuLP code and remains unchanged.
def nx_to_mat_and_weights_xor(G):
    nodes = list(G.nodes())
    node_index = {node: i for i, node in enumerate(nodes)}
    n = len(nodes)
    A = np.zeros((n, n), dtype=int)
    W = np.zeros((n, n), dtype=float)
    for u, v, data in G.edges(data=True):
        # check for robustness 
        if u not in node_index or v not in node_index:
            continue
        i = node_index[u]
        j = node_index[v]
        A[i, j] = 1
        W[i, j] = data.get('weight', 1.0)
    return A, W, nodes, node_index

# gurobi
def xor_cut_ilp(G, sup=1.0):

    start_node = 'start'
    end_node = 'end'

    # Preprocess graph 
    reduced_graph, _, _ = preprocess_graph(G, start_node, end_node)
    A, W, node_names, node_index = nx_to_mat_and_weights_xor(reduced_graph)
    n = A.shape[0]

    # --- Robustness checks  ---
    if n == 0 or not node_names:
        return set(), set(), None, []
        
    start_idx = node_names.index(start_node)
    end_idx = node_names.index(end_node)
    non_terminal = [i for i in range(n) if i != start_idx and i != end_idx]
    
    if not non_terminal:
        # Graph only contains start/end nodes
        return set(), set(), None, node_names
    # --- End robustness checks ---

    # --- Gurobi Model ---
    model = gp.Model("XOR_MinCut_Flow_Corrected")
    model.setParam('OutputFlag', 0) # Suppress Gurobi output

    # 1. variable indices
    edge_indices = [(i, j) for i in range(n) for j in range(n) if A[i, j]]

    # 2.  variables (matching PuLP V1)
    # x[i] = 1 if node i is in Sigma_2, 0 if in Sigma_1
    x = model.addVars(non_terminal, vtype=GRB.BINARY, name="x")
    
    # y[i,j] = 1 if edge (i,j) is cut
    y = model.addVars(edge_indices, vtype=GRB.BINARY, name="y")
    
    # Flow variables from start 
    f_s = model.addVars(edge_indices, lb=0, vtype=GRB.INTEGER, name="f_s")
    # Flow variables to end 
    f_e = model.addVars(edge_indices, lb=0, vtype=GRB.INTEGER, name="f_e")

    ### 3. constraints (matching PuLP V1 logic) ###

    # Don't cut start -> * or * -> end edges
    for i, j in edge_indices:
        if i == start_idx or j == end_idx:
            model.addConstr(y[i, j] == 0)
        elif i in non_terminal and j in non_terminal:
            model.addConstr(y[i, j] >= x[i] - x[j])
            model.addConstr(y[i, j] >= x[j] - x[i])

    # non-trivial partition
    model.addConstr(gp.quicksum(x[i] for i in non_terminal) >= 1)
    model.addConstr(gp.quicksum(1 - x[i] for i in non_terminal) >= 1)

    M = n - 1

    # --- reachability FROM start (f_s) ---
    model.addConstr(f_s.sum(start_idx, '*') == M, name="f_s_start_flow")

    for i in range(n):
        if i == start_idx:
            continue
        # (inflow - outflow) == 1
        model.addConstr(f_s.sum('*', i) - f_s.sum(i, '*') == 1, name=f"f_s_node_{i}")

    for (i, j) in edge_indices:
        # f_s[(i, j)] <= (n - 1) * (1 - y[(i, j)])
        model.addConstr(f_s[i, j] <= M * (1 - y[i, j]), name=f"f_s_cap_{i}_{j}")

    # --- reachability TO end (f_e) ---
    model.addConstr(f_e.sum('*', end_idx) == M, name="f_e_end_flow")

    for i in range(n):
        if i == end_idx:
            continue
        #(outflow - inflow) == 1
        model.addConstr(f_e.sum(i, '*') - f_e.sum('*', i) == 1, name=f"f_e_node_{i}")

    for (i, j) in edge_indices:
        # f_e[(i, j)] <= (n - 1) * (1 - y[(i, j)])
        model.addConstr(f_e[i, j] <= M * (1 - y[i, j]), name=f"f_e_cap_{i}_{j}")

    # 4. objective
    objective = gp.quicksum(W[i, j] * y[i, j] for i, j in edge_indices)
    model.setObjective(objective, GRB.MINIMIZE)

    # 5. solve
    model.optimize()

    # 6. results
    Sigma_1 = []
    Sigma_2 = []
    total_cost = None

    if model.SolCount > 0:
        total_cost = model.ObjVal

        Sigma_1 = [node_names[i] for i in non_terminal if x[i].X < 0.5]
        Sigma_2 = [node_names[i] for i in non_terminal if x[i].X > 0.5]
        
        # Check for valid partition 
        if not Sigma_1 or not Sigma_2:
            print("Empty partition detected (should be prevented by constraints)")
            total_cost = None # Treat as invalid
        
        # Warn if not proven optimal
        if model.status != GRB.OPTIMAL:
            print(f"Warning: Solution is feasible but not proven optimal. Status: {model.status}")

    #else:
        # This block runs only if NO solution was found
        #print(f"ILP failed to find any feasible solution. Status: {model.status}")

    return (
        extract_activities(Sigma_1),
        extract_activities(Sigma_2),
        total_cost,
        node_names
    )       


def xor_cut_tau(G, sup=1.0):
    total_cases = sum(data["weight"] for src, tgt, data in G.edges(data=True) if src == "start")
    start_to_end_weight = G.get_edge_data('start', 'end', {'weight': 0})['weight']
    total_cost = max(0, total_cases * sup - start_to_end_weight)
    nodes = list(G.nodes())

    Sigma_1 = extract_activities(nodes)  # All regular activities
    Sigma_2 = set()  # Represents tau/empty

    return Sigma_1, Sigma_2, total_cost, nodes