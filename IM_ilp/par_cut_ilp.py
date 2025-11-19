import gurobipy as gp
from gurobipy import GRB
from IM_ilp.Helper_Functions import preprocess_graph, nx_to_mat_and_weights, extract_activities, cost_
import numpy as np

def par_cut_ilp(G, sup=1.0):
    start_node = 'start'
    end_node = 'end'

    # 1. Preprocess graph
    reduced_graph, _, _ = preprocess_graph(G, start_node, end_node)
    A, _, node_names, node_index = nx_to_mat_and_weights(reduced_graph, sup)
    n = A.shape[0]

    # 2. Get cost matrix
    W = cost_(G, sup)  # This has costs for ALL possible pairs
    
    # 3. Handle graph/node indices
    if n == 0 or not node_names:
        return set(), set(), None, []

    start_idx = node_names.index(start_node)
    end_idx = node_names.index(end_node)
    non_terminal = [i for i in range(n) if i not in (start_idx, end_idx)]
    
    if not non_terminal:
        # Graph only contains start/end nodes
        return set(), set(), None, node_names

    # === Gurobipy Model  ===
    model = gp.Model("PAR_MinCut_Flow_Corrected")
    model.setParam('OutputFlag', 0) # Suppress console output

    # --- Variables  ---
    
    # Partition variables: x[i]
    x = model.addVars(non_terminal, vtype=GRB.BINARY, name="x")

    # Cost/Cut variables: y[i,j] for ALL non-terminal pairs
    y_indices = [(i, j) for i in non_terminal for j in non_terminal]
    y = model.addVars(y_indices, vtype=GRB.BINARY, name="y")

    # --- 2-FLOWS  ---
    
    # Flow variables (reachability from start) - only for existing edges
    edge_indices = [(i, j) for i in range(n) for j in range(n) if A[i, j]]
    f_s = model.addVars(edge_indices, vtype=GRB.INTEGER, lb=0, name="f_s")
    
    # Flow variables (reachability to end) - only for existing edges  
    f_e = model.addVars(edge_indices, vtype=GRB.INTEGER, lb=0, name="f_e")

    # === Constraints  ===

    # 1. Define y cost variables
    for i in non_terminal:
        for j in non_terminal:
            if i != j:
                model.addConstr(y[i, j] >= x[i] - x[j])
                model.addConstr(y[i, j] >= x[j] - x[i])
            else:
                model.addConstr(y[i, j] == 0)

    # 2. Non-trivial partition constraints
    model.addConstr(gp.quicksum(x[i] for i in non_terminal) >= 1, name="non_trivial_1")
    model.addConstr(gp.quicksum(1 - x[i] for i in non_terminal) >= 1, name="non_trivial_2")

    M = n - 1

    # 3. Flow reachability from start (f_s)
    model.addConstr(f_s.sum(start_idx, '*') == M, name="f_s_start_flow")

    for i in range(n):
        if i == start_idx:
            continue
        # (inflow - outflow) == 1
        model.addConstr(f_s.sum('*', i) - f_s.sum(i, '*') == 1, name=f"f_s_node_{i}")

    # Block flow through cut edges
    for (i, j) in edge_indices:
        if i in non_terminal and j in non_terminal:
            # f_s[(i, j)] <= (n - 1) * (1 - y[(i, j)])
            model.addConstr(f_s[i, j] <= M * (1 - y[i, j]), name=f"f_s_cap_{i}_{j}")

    # 4. Flow reachability to end (f_e)
    # Receive (n-1) units of flow at end
    model.addConstr(f_e.sum('*', end_idx) == M, name="f_e_end_flow")

    for i in range(n):
        if i == end_idx:
            continue
        # (outflow - inflow) == 1
        model.addConstr(f_e.sum(i, '*') - f_e.sum('*', i) == 1, name=f"f_e_node_{i}")

    # Block flow through cut edges
    for (i, j) in edge_indices:
        if i in non_terminal and j in non_terminal:
            #f_e[(i, j)] <= (n - 1) * (1 - y[(i, j)])
            model.addConstr(f_e[i, j] <= M * (1 - y[i, j]), name=f"f_e_cap_{i}_{j}")

    # --- Objective ---
    # Minimize total crossing cost for ALL possible pairs
    objective_expr = gp.quicksum(W.get((node_names[i], node_names[j]), 0) * y[i, j] 
                                for i, j in y_indices)
    model.setObjective(objective_expr, GRB.MINIMIZE)

    # --- Solve ---
    model.optimize()
    Sigma_1 = []
    Sigma_2 = []
    total_cost = None

    #  *any* solution was found, not just the optimal one
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