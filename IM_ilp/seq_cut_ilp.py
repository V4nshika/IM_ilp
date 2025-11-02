import pulp
from IM_ilp.Helper_Functions import preprocess_graph, cost_eventual, extract_activities
import numpy as np

def nx_to_mat_and_weights_seq(G, log, sup=1.0):
    nodes = list(G.nodes())
    print("nodes? ")
    node_index = {node: i for i, node in enumerate(nodes)}
    n = len(nodes)
    A = np.zeros((n, n), dtype=float)

    W_backward = {}  # For deviating edges (actual weights)
    print("maybe this one is the problem")
    for u, v, data in G.edges(data=True):
        i = node_index[u]
        j = node_index[v]
        A[i, j] = 1
        W_backward[i, j] = data.get('weight', 0)  # Actual weights

    W_forward = cost_eventual(G, log, sup)  # For missing edges (expected - actual)

    return A, W_forward, W_backward, nodes, node_index


def seq_cut_ilp_old(G,log,  sup=1.0):
    start_node = 'start'
    end_node = 'end'

    # Preprocess graph
    reduced_graph, _, _ = preprocess_graph(G, start_node, end_node)
    print("preprocesses")
    A, W_forward, W_backward, node_names, node_index = nx_to_mat_and_weights_seq(reduced_graph, log, sup)
    print("nx done")
    n = A.shape[0]

    start_idx = node_names.index(start_node)
    end_idx = node_names.index(end_node)
    non_terminal = [i for i in range(n) if i != start_idx and i != end_idx]
    print("non terminal done")

    # ILP Model
    prob = pulp.LpProblem("Sequence_Cut_MinCost", pulp.LpMinimize)

    # Variables
    x = pulp.LpVariable.dicts("x", non_terminal, cat=pulp.LpBinary)
    f1 = pulp.LpVariable.dicts("f1", [(i, j) for i in range(n) for j in range(n) if A[i, j]], lowBound=0,
                               cat='Continuous')
    f2 = pulp.LpVariable.dicts("f2", [(i, j) for i in range(n) for j in range(n) if A[i, j]], lowBound=0,
                               cat='Continuous')

    cost_terms = []

    # --- Objective: ONLY edges between Σ₁ and Σ₂ ---
    for i in non_terminal:
        i_name = node_names[i]
        for j in non_terminal:
            if A[i, j]:
                j_name = node_names[j]

                cost_val_fwd = W_forward.get((i_name, j_name), 0)
                cost_val_bwd = W_backward.get((i_name, j_name), 0)

                # Auxiliary binary variables
                z_forward = pulp.LpVariable(f"z_fwd_{i}_{j}", cat=pulp.LpBinary)
                z_backward = pulp.LpVariable(f"z_bwd_{i}_{j}", cat=pulp.LpBinary)

                # Forward: i in Σ1, j in Σ2
                prob += z_forward >= (1 - x[i]) + x[j] - 1
                prob += z_forward <= (1 - x[i])
                prob += z_forward <= x[j]
                cost_terms.append(cost_val_fwd * z_forward)

                # Backward: i in Σ2, j in Σ1
                prob += z_backward >= x[i] + (1 - x[j]) - 1
                prob += z_backward <= x[i]
                prob += z_backward <= (1 - x[j])
                cost_terms.append(cost_val_bwd * z_backward)
            

    # Objective
    prob += pulp.lpSum(cost_terms)

    # Constraints
    # Non-trivial partition
    prob += pulp.lpSum(1 - x[i] for i in non_terminal) >= 1
    prob += pulp.lpSum(x[i] for i in non_terminal) >= 1

    # Flow constraints (Sigma_1)
    prob += pulp.lpSum(f1[(start_idx, j)] for j in range(n) if A[start_idx, j]) == pulp.lpSum(
        1 - x[u] for u in non_terminal)
    for u in non_terminal:
        inflow = pulp.lpSum(f1[(i, u)] for i in range(n) if A[i, u])
        outflow = pulp.lpSum(f1[(u, j)] for j in range(n) if A[u, j])
        prob += (inflow - outflow) == (1 - x[u])
    for i, j in f1:
        if i != start_idx:
            prob += f1[(i, j)] <= (n * n) * (1 - x.get(i, 0))
        if j != end_idx:
            prob += f1[(i, j)] <= (n * n) * (1 - x.get(j, 0))

    # Flow constraints (Sigma_2)
    prob += pulp.lpSum(f2[(i, end_idx)] for i in range(n) if A[i, end_idx]) == pulp.lpSum(x[u] for u in non_terminal)
    for u in non_terminal:
        inflow = pulp.LpAffineExpression([(f2[(i, u)], 1) for i in range(n) if A[i, u]])
        outflow = pulp.LpAffineExpression([(f2[(u, j)], 1) for j in range(n) if A[u, j]])
        prob += (outflow - inflow) == x[u]
    for i, j in f2:
        if i != start_idx:
            prob += f2[(i, j)] <= (n * n) * x.get(i, 1)
        if j != end_idx:
            prob += f2[(i, j)] <= (n * n) * x.get(j, 1)

    # Solve
    status = prob.solve(pulp.GUROBI_CMD(msg=False))
    # print(f"Solver status: {pulp.LpStatus[status]}")

    # Manual cost calculation
    total_cost = 0

    # Non-terminal edges
    for i in non_terminal:
        for j in non_terminal:
            if A[i, j]:
                i_val = pulp.value(x[i])
                j_val = pulp.value(x[j])
                i_name = node_names[i]
                j_name = node_names[j]

                if i_val < 0.5 and j_val > 0.5:  # Forward
                    cost_val = W_forward.get((i_name, j_name), 0)
                    total_cost += cost_val
                    # print(f"Forward: {i_name} -> {j_name}: cost = {cost_val}")
                elif i_val > 0.5 and j_val < 0.5:  # Backward
                    cost_val = W_backward.get((i_name, j_name), 0)
                    total_cost += cost_val
                    # print(f"Backward: {i_name} -> {j_name}: cost = {cost_val}")

    # print(f"Total calculated cost: {total_cost}")

    # Extract results
    Sigma_1 = [node_names[i] for i in non_terminal if pulp.value(x[i]) == 0]
    Sigma_2 = [node_names[i] for i in non_terminal if pulp.value(x[i]) == 1]

    if not Sigma_1 and not Sigma_2:
        total_cost = None

    return (
        extract_activities(Sigma_1),
        extract_activities(Sigma_2),
        total_cost,
        node_names
    )

def seq_cut_ilp(G, log, sup=1.0):
    start_node = 'start'
    end_node = 'end'

    # Preprocess graph
    reduced_graph, _, _ = preprocess_graph(G, start_node, end_node)
    A, W_forward, W_backward, node_names, node_index = nx_to_mat_and_weights_seq(reduced_graph, log, sup)
    n = A.shape[0]

    start_idx = node_names.index(start_node)
    end_idx = node_names.index(end_node)
    non_terminal = [i for i in range(n) if i != start_idx and i != end_idx]

    # ILP Model
    prob = pulp.LpProblem("Sequence_Cut_MinCost", pulp.LpMinimize)

    # Variables
    x = pulp.LpVariable.dicts("x", non_terminal, cat=pulp.LpBinary)
    f1 = pulp.LpVariable.dicts("f1", [(i, j) for i in range(n) for j in range(n) if A[i, j]], lowBound=0,
                               cat='Continuous')
    f2 = pulp.LpVariable.dicts("f2", [(i, j) for i in range(n) for j in range(n) if A[i, j]], lowBound=0,
                               cat='Continuous')

    cost_terms = []

    # --- FORWARD COST: All possible edges from Σ₁ to Σ₂ ---
    for i in non_terminal:
        i_name = node_names[i]
        for j in non_terminal:
            if i == j:
                continue
                
            j_name = node_names[j]
            
            # Get forward cost (expected - actual for eventual-follows)
            cost_val_fwd = W_forward.get((i_name, j_name), 0)
            
            # Only add cost if i in Σ₁ AND j in Σ₂
            z_forward = pulp.LpVariable(f"z_fwd_{i}_{j}", cat=pulp.LpBinary)
            
            # z_forward = 1 iff (1-x[i]) = 1 AND x[j] = 1
            prob += z_forward >= (1 - x[i]) + x[j] - 1
            prob += z_forward <= (1 - x[i])
            prob += z_forward <= x[j]
            
            cost_terms.append(cost_val_fwd * z_forward)

    # --- BACKWARD COST: Only actual edges from Σ₂ to Σ₁ ---
    for i in non_terminal:
        i_name = node_names[i]
        for j in non_terminal:
            if A[i, j]:  # Only if edge actually exists
                j_name = node_names[j]
                
                # Get backward cost (actual weight of deviating edge)
                cost_val_bwd = W_backward.get((i, j), 0)
                
                # Only add cost if i in Σ₂ AND j in Σ₁
                z_backward = pulp.LpVariable(f"z_bwd_{i}_{j}", cat=pulp.LpBinary)
                
                # z_backward = 1 iff x[i] = 1 AND (1-x[j]) = 1
                prob += z_backward >= x[i] + (1 - x[j]) - 1
                prob += z_backward <= x[i]
                prob += z_backward <= (1 - x[j])
                
                cost_terms.append(cost_val_bwd * z_backward)

    # Objective
    prob += pulp.lpSum(cost_terms)

    # Constraints (same as before)
    # Non-trivial partition
    prob += pulp.lpSum(1 - x[i] for i in non_terminal) >= 1
    prob += pulp.lpSum(x[i] for i in non_terminal) >= 1

    # Flow constraints (Sigma_1)
    prob += pulp.lpSum(f1[(start_idx, j)] for j in range(n) if A[start_idx, j]) == pulp.lpSum(
        1 - x[u] for u in non_terminal)
    for u in non_terminal:
        inflow = pulp.lpSum(f1[(i, u)] for i in range(n) if A[i, u])
        outflow = pulp.lpSum(f1[(u, j)] for j in range(n) if A[u, j])
        prob += (inflow - outflow) == (1 - x[u])
    for i, j in f1:
        if i != start_idx:
            prob += f1[(i, j)] <= (n * n) * (1 - x.get(i, 0))
        if j != end_idx:
            prob += f1[(i, j)] <= (n * n) * (1 - x.get(j, 0))

    # Flow constraints (Sigma_2)
    prob += pulp.lpSum(f2[(i, end_idx)] for i in range(n) if A[i, end_idx]) == pulp.lpSum(x[u] for u in non_terminal)
    for u in non_terminal:
        inflow = pulp.LpAffineExpression([(f2[(i, u)], 1) for i in range(n) if A[i, u]])
        outflow = pulp.LpAffineExpression([(f2[(u, j)], 1) for j in range(n) if A[u, j]])
        prob += (outflow - inflow) == x[u]
    for i, j in f2:
        if i != start_idx:
            prob += f2[(i, j)] <= (n * n) * x.get(i, 1)
        if j != end_idx:
            prob += f2[(i, j)] <= (n * n) * x.get(j, 1)

    # Solve
    status = prob.solve(pulp.GUROBI_CMD(msg=False))

    # Extract results and calculate actual cost
    Sigma_1 = [node_names[i] for i in non_terminal if pulp.value(x[i]) == 0]
    Sigma_2 = [node_names[i] for i in non_terminal if pulp.value(x[i]) == 1]

    # Calculate actual cost manually
    total_cost = 0

    # Forward cost: all possible edges from Σ₁ to Σ₂
    for i in non_terminal:
        if pulp.value(x[i]) == 0:  # i in Σ₁
            i_name = node_names[i]
            for j in non_terminal:
                if pulp.value(x[j]) == 1:  # j in Σ₂
                    j_name = node_names[j]
                    cost_val = W_forward.get((i_name, j_name), 0)
                    total_cost += cost_val

    # Backward cost: only actual edges from Σ₂ to Σ₁
    for i in non_terminal:
        if pulp.value(x[i]) == 1:  # i in Σ₂
            for j in non_terminal:
                if pulp.value(x[j]) == 0 and A[i, j]:  # j in Σ₁ AND edge exists
                    cost_val = W_backward.get((i, j), 0)
                    total_cost += cost_val

    if not Sigma_1 and not Sigma_2:
        total_cost = None

    return (
        extract_activities(Sigma_1),
        extract_activities(Sigma_2),
        total_cost,
        node_names
    )