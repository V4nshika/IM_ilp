import pulp
from IM_ilp.Helper_Functions import preprocess_graph, nx_to_mat_and_weights, extract_activities, cost_

def par_cut_ilp(G, sup=1.0):
    start_node = 'start'
    end_node = 'end'

    # Preprocess graph
    reduced_graph, _, _ = preprocess_graph(G, start_node, end_node)
    #print("processed")
    A, W, node_names, node_index = nx_to_mat_and_weights(reduced_graph, sup)
    #print("nx done")
    n = A.shape[0]
    #print(node_names, node_index)

    W = cost_(G, sup)  # This has costs for ALL possible pairs
    #print("cost done")
    #print(W)
    start_idx = node_names.index(start_node)
    end_idx = node_names.index(end_node)
    non_terminal = [i for i in range(n) if i not in (start_idx, end_idx)]
    #print(f"non_terminal done: {len(non_terminal)} nodes")

    # ILP Model
    prob = pulp.LpProblem("PAR_MinCut_Flow_Corrected", pulp.LpMinimize)

    # Partition variables
    x = pulp.LpVariable.dicts("x", non_terminal, cat=pulp.LpBinary)

    # Edge cut indicators for ALL possible pairs between non-terminal nodes
    y = pulp.LpVariable.dicts("y", [(i, j) for i in non_terminal for j in non_terminal], cat=pulp.LpBinary)

    # Flow variables (reachability from start) - only for existing edges
    f_s = pulp.LpVariable.dicts("f_s", [(i, j) for i in range(n) for j in range(n) if A[i, j]], lowBound=0, cat='Integer')
    
    # Flow variables (reachability to end) - only for existing edges  
    f_e = pulp.LpVariable.dicts("f_e", [(i, j) for i in range(n) for j in range(n) if A[i, j]], lowBound=0, cat='Integer')

    #### Constraints ####

    # Define y variables for ALL pairs (i,j) where i != j and i,j in non_terminal
    # y[(i,j)] = 1 if x[i] != x[j] (nodes in different partitions)
    for i in non_terminal:
        for j in non_terminal:
            if i != j:
                prob += y[(i, j)] >= x[i] - x[j]
                prob += y[(i, j)] >= x[j] - x[i]
            else:
                prob += y[(i, j)] == 0 
                # No upper bound needed since we're minimizing

    # Non-trivial partition constraint
    prob += pulp.lpSum(x[i] for i in non_terminal) >= 1
    prob += pulp.lpSum(1 - x[i] for i in non_terminal) >= 1

    ##### Flow reachability from start #####
    # Send (n-1) units of flow from start
    prob += pulp.lpSum(f_s[(start_idx, j)] for j in range(n) if A[start_idx, j]) == n - 1

    for i in range(n):
        if i == start_idx:
            continue
        inflow = pulp.lpSum(f_s[(j, i)] for j in range(n) if A[j, i])
        outflow = pulp.lpSum(f_s[(i, j)] for j in range(n) if A[i, j])
        prob += (inflow - outflow) == 1

    # Block flow through cut edges (only for existing edges)
    for (i, j) in f_s:
        if i in non_terminal and j in non_terminal:
            prob += f_s[(i, j)] <= (n - 1) * (1 - y[(i, j)])

    ##### Flow reachability to end #####
    # Receive (n-1) units of flow at end
    prob += pulp.lpSum(f_e[(i, end_idx)] for i in range(n) if A[i, end_idx]) == n - 1

    for i in range(n):
        if i == end_idx:
            continue
        inflow = pulp.lpSum(f_e[(j, i)] for j in range(n) if A[j, i])
        outflow = pulp.lpSum(f_e[(i, j)] for j in range(n) if A[i, j])
        prob += (outflow - inflow) == 1

    # Block flow through cut edges (only for existing edges)
    for (i, j) in f_e:
        if i in non_terminal and j in non_terminal:
            prob += f_e[(i, j)] <= (n - 1) * (1 - y[(i, j)])

    ##### Objective: Minimize total crossing weight for ALL possible pairs #####
    #objective_terms = []
    #for i in non_terminal:
        #for j in non_terminal:
            #if i != j:
                #i_name = node_names[i]
                #j_name = node_names[j]
                #cost_val = W.get((i_name, j_name), 0)  # Get cost from W dict
                #objective_terms.append(cost_val * y[(i, j)])
    
    #prob += pulp.lpSum(objective_terms)
    prob += pulp.lpSum(W[node_names[i], node_names[j]] * y[(i, j)] for (i, j) in y)

    #print(f"Objective has {len(objective_terms)} terms")

    # Solve
    prob.solve(pulp.GUROBI_CMD(msg=False))  # Enable messages for debugging

    # Check solution status
    if prob.status != pulp.LpStatusOptimal:
        print(f"ILP failed with status: {pulp.LpStatus[prob.status]}")
        return (set(), set(), None, node_names)

    # Extract results
    Sigma_1 = [node_names[i] for i in non_terminal if pulp.value(x[i]) == 0]
    Sigma_2 = [node_names[i] for i in non_terminal if pulp.value(x[i]) == 1]
    total_cost = pulp.value(prob.objective)
    
    print(f"Partition: Sigma_1={Sigma_1}, Sigma_2={Sigma_2}, cost={total_cost}")
    
    if not Sigma_1 or not Sigma_2:
        print("Empty partition detected")
        total_cost = None

    return (
        extract_activities(Sigma_1),
        extract_activities(Sigma_2),
        total_cost,
        node_names
    )

def par_cut_ilp_old(G, sup=1.0):
    start_node = 'start'
    end_node = 'end'

    # Preprocess graph
    reduced_graph, _, _ = preprocess_graph(G, start_node, end_node)
    A, W, node_names, node_index = nx_to_mat_and_weights(reduced_graph, sup)
    n = A.shape[0]

    W = cost_(G, sup)

    start_idx = node_names.index(start_node)
    end_idx = node_names.index(end_node)
    non_terminal = [i for i in range(n) if i not in (start_idx, end_idx)]

    # ILP Model
    prob = pulp.LpProblem("PAR_MinCut_Flow_Corrected", pulp.LpMinimize)

    # Partition variables
    x = pulp.LpVariable.dicts("x", non_terminal, cat=pulp.LpBinary)

    # Edge cut indicators
    y = pulp.LpVariable.dicts("y", [(i, j) for i in range(n) for j in range(n) if A[i, j]], cat=pulp.LpBinary)

    # Flow variables (reachability from start)
    f_s = pulp.LpVariable.dicts("f_s", [(i, j) for i in range(n) for j in range(n) if A[i, j]], lowBound=0,
                                cat='Integer')
    # Flow variables (reachability to end)
    f_e = pulp.LpVariable.dicts("f_e", [(i, j) for i in range(n) for j in range(n) if A[i, j]], lowBound=0,
                                cat='Integer')

    #### Constraints ####

    # Prevent cutting start → * and * → end
    for i in range(n):
        for j in range(n):
            if A[i, j]:
                if i == start_idx or j == end_idx:
                    prob += y[(i, j)] == 0
                elif i in non_terminal and j in non_terminal:
                    prob += y[(i, j)] >= x[i] - x[j]
                    prob += y[(i, j)] >= x[j] - x[i]

    # Non-trivial partition
    prob += pulp.lpSum(x[i] for i in non_terminal) >= 1
    prob += pulp.lpSum(1 - x[i] for i in non_terminal) >= 1

    ##### Flow reachability from start #####
    # Send (n-1) units of flow from start
    prob += pulp.lpSum(f_s[(start_idx, j)] for j in range(n) if A[start_idx, j]) == n - 1

    for i in range(n):
        if i == start_idx:
            continue
        inflow = pulp.lpSum(f_s[(j, i)] for j in range(n) if A[j, i])
        outflow = pulp.lpSum(f_s[(i, j)] for j in range(n) if A[i, j])
        prob += (inflow - outflow) == 1

    # Block flow through cut edges
    for (i, j) in f_s:
        prob += f_s[(i, j)] <= (n - 1) * (1 - y[(i, j)])

    ##### Flow reachability to end #####
    # Receive (n-1) units of flow at end
    prob += pulp.lpSum(f_e[(i, end_idx)] for i in range(n) if A[i, end_idx]) == n - 1

    for i in range(n):
        if i == end_idx:
            continue
        inflow = pulp.lpSum(f_e[(j, i)] for j in range(n) if A[j, i])
        outflow = pulp.lpSum(f_e[(i, j)] for j in range(n) if A[i, j])
        prob += (outflow - inflow) == 1

    # Block flow through cut edges
    for (i, j) in f_e:
        prob += f_e[(i, j)] <= (n - 1) * (1 - y[(i, j)])

    ##### Objective: Minimize total crossing weight #####
    prob += pulp.lpSum(W[node_names[i], node_names[j]] * y[(i, j)] for (i, j) in y)

    # Solve
    prob.solve(pulp.GUROBI_CMD(msg=False))

    # Extract results
    Sigma_1 = [node_names[i] for i in non_terminal if pulp.value(x[i]) == 0]
    Sigma_2 = [node_names[i] for i in non_terminal if pulp.value(x[i]) == 1]
    total_cost = pulp.value(prob.objective)
    print(prob.objective)
    print(pulp.value(prob.objective))
    if not Sigma_1 and not Sigma_2:
        total_cost = None

    return (
        extract_activities(Sigma_1),
        extract_activities(Sigma_2),
        total_cost,
        node_names
    )

