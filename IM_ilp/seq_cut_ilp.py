import gurobipy as gp
from gurobipy import GRB
from IM_ilp.Helper_Functions import preprocess_graph, cost_eventual, extract_activities
import numpy as np

def nx_to_mat_and_weights_seq(G, log, sup=1.0):
        nodes = list(G.nodes())
        node_index = {node: i for i, node in enumerate(nodes)}
        n = len(nodes)
        A = np.zeros((n, n), dtype=float)

        W_backward = {}  # For deviating edges (actual weights)
        for u, v, data in G.edges(data=True):
                i = node_index[u]
                j = node_index[v]
                A[i, j] = 1
                W_backward[(i, j)] = data.get('weight', 0)  # Use index tuple (i, j) as key
              
        # Always recalculate W_forward based on the current local log/graph
        W_forward = cost_eventual(G, log, sup)  # For missing edges (expected - actual)

        return A, W_forward, W_backward, nodes, node_index

def seq_cut_ilp_linearized(G, log, sup=1.0):
    start_node = 'start'
    end_node = 'end'

    # Preprocess graph
    reduced_graph, _, _ = preprocess_graph(G, start_node, end_node)

    # Calculate from scratch for the current recursion level
    A, W_forward, W_backward, node_names, node_index = nx_to_mat_and_weights_seq(reduced_graph, log, sup)
    n = A.shape[0]

    if n == 0 or not node_names:
        return set(), set(), None, []
      
    start_idx = node_names.index(start_node)
    end_idx = node_names.index(end_node)
    non_terminal = [i for i in range(n) if i != start_idx and i != end_idx]
  
    if not non_terminal:
        # Graph only contains start/end nodes
        return set(), set(), None, node_names

    # ILP Model - No NonConvex parameter needed
    model = gp.Model("Sequence_Cut_MinCost_Gurobi_Linearized")
    model.setParam('OutputFlag', 0)
    model.setParam('Threads', 1)

    # --- Variables ---
    # x[i] = 1 if node i is in Sigma_2 (partition 2), 0 if in Sigma_1
    x = model.addVars(non_terminal, vtype=GRB.BINARY, name="x")
  
    # Flow variables for Sigma_1 (partition 1)
    f1_indices = [(i, j) for i in range(n) for j in range(n) if A[i, j]]
    f1 = model.addVars(f1_indices, vtype=GRB.CONTINUOUS, name="f1")
  
    # Flow variables for Sigma_2 (partition 2)
    f2_indices = [(i, j) for i in range(n) for j in range(n) if A[i, j]]
    f2 = model.addVars(f2_indices, vtype=GRB.CONTINUOUS, name="f2")
    
    # Linearization variables for forward cost: y_ij = (1-x[i]) * x[j]
    forward_pairs = []
    for i in non_terminal:
        i_name = node_names[i]
        for j in non_terminal:
            if i == j: continue
            j_name = node_names[j]
            if W_forward.get((i_name, j_name), 0) > 0:
                forward_pairs.append((i, j))
    
    y = model.addVars(forward_pairs, vtype=GRB.CONTINUOUS, name="y", lb=0, ub=1)
    
    # Linearization variables for backward cost: z_ij = x[i] * (1-x[j])
    backward_pairs = []
    for i in non_terminal:
        for j in non_terminal:
            if i == j: continue
            if A[i, j]:  # Only if edge actually exists
                if W_backward.get((i, j), 0) > 0:
                    backward_pairs.append((i, j))
    
    z = model.addVars(backward_pairs, vtype=GRB.CONTINUOUS, name="z", lb=0, ub=1)

    # --- Linearization Constraints ---
    # y_ij = (1-x[i]) * x[j]
    for i, j in forward_pairs:
        model.addConstr(y[i, j] <= 1 - x[i], name=f"y_leq_1mx_{i}_{j}")
        model.addConstr(y[i, j] <= x[j], name=f"y_leq_x_{i}_{j}")
        model.addConstr(y[i, j] >= (1 - x[i]) + x[j] - 1, name=f"y_geq_{i}_{j}")
    
    # z_ij = x[i] * (1-x[j])
    for i, j in backward_pairs:
        model.addConstr(z[i, j] <= x[i], name=f"z_leq_x_{i}_{j}")
        model.addConstr(z[i, j] <= 1 - x[j], name=f"z_leq_1mx_{i}_{j}")
        model.addConstr(z[i, j] >= x[i] + (1 - x[j]) - 1, name=f"z_geq_{i}_{j}")

    # --- Linear Objective ---
    obj_terms = []
    
    # Forward cost terms
    for i, j in forward_pairs:
        i_name = node_names[i]
        j_name = node_names[j]
        cost = W_forward.get((i_name, j_name), 0)
        if cost > 0:
            obj_terms.append(cost * y[i, j])
    
    # Backward cost terms
    for i, j in backward_pairs:
        cost = W_backward.get((i, j), 0)
        if cost > 0:
            obj_terms.append(cost * z[i, j])
    
    model.setObjective(gp.quicksum(obj_terms), GRB.MINIMIZE)

    # --- Constraints (same as before) ---
    # Non-trivial partition
    model.addConstr(gp.quicksum(1 - x[i] for i in non_terminal) >= 1, name="non_trivial_1")
    model.addConstr(gp.quicksum(x[i] for i in non_terminal) >= 1, name="non_trivial_2")

    # Sigma_1 flow constraints
    model.addConstr(f1.sum(start_idx, '*') == gp.quicksum(1 - x[u] for u in non_terminal), name="flow1_start")
    
    for u in non_terminal:
        inflow = f1.sum('*', u)
        outflow = f1.sum(u, '*')
        model.addConstr((inflow - outflow) == (1 - x[u]), name=f"flow1_node_{u}")

    for i, j in f1_indices:
        if i != start_idx:
            x_i_term = x[i] if i in non_terminal else 0
            model.addConstr(f1[i, j] <= (n * n) * (1 - x_i_term), name=f"flow1_cap_i_{i}_{j}")
        if j != end_idx:
            x_j_term = x[j] if j in non_terminal else 0
            model.addConstr(f1[i, j] <= (n * n) * (1 - x_j_term), name=f"flow1_cap_j_{i}_{j}")

    # Sigma_2 flow constraints
    model.addConstr(f2.sum('*', end_idx) == gp.quicksum(x[u] for u in non_terminal), name="flow2_end")
    
    for u in non_terminal:
        inflow = f2.sum('*', u)
        outflow = f2.sum(u, '*')
        model.addConstr((outflow - inflow) == x[u], name=f"flow2_node_{u}")
  
    for i, j in f2_indices:
        if i != start_idx:
            x_i_term = x[i] if i in non_terminal else 1
            model.addConstr(f2[i, j] <= (n * n) * x_i_term, name=f"flow2_cap_i_{i}_{j}")
        if j != end_idx:
            x_j_term = x[j] if j in non_terminal else 1
            model.addConstr(f2[i, j] <= (n * n) * x_j_term, name=f"flow2_cap_j_{i}_{j}")

    # Solve
    model.optimize()

    # Results
    if model.Status == GRB.OPTIMAL:
        Sigma_1 = [node_names[i] for i in non_terminal if x[i].X < 0.5]
        Sigma_2 = [node_names[i] for i in non_terminal if x[i].X > 0.5]
        total_cost = model.ObjVal

        if not Sigma_1 or not Sigma_2:
            total_cost = None
    else:
        print(f"Gurobi solver for seq_cut failed with status: {model.Status}")
        Sigma_1, Sigma_2, total_cost = set(), set(), None

    return (
        extract_activities(Sigma_1),
        extract_activities(Sigma_2),
        total_cost,
        node_names
    )
