import networkx as nx
import gurobipy as gp
from gurobipy import GRB
from IM_ilp.Helper_Functions import preprocess_graph, extract_activities, log_to_graph
import numpy as np
from local_pm4py.cut_quality.cost_functions.cost_functions import cost_loop


def nx_to_mat_and_weights_loop(G):

    nodes = list(G.nodes())
    node_index = {node: i for i, node in enumerate(nodes)}
    n = len(nodes)
    A = np.zeros((n, n), dtype=int)  # Adjacency matrix (1 if edge exists)
    W = np.zeros((n, n), dtype=int)  # Weight matrix (frequency of edge)
    for u, v, data in G.edges(data=True):
        if u not in node_index or v not in node_index:
            continue
        i = node_index[u]
        j = node_index[v]
        A[i, j] = 1
        W[i, j] = data.get('weight', 1)
    return A, W, nodes, node_index


def add_mccormick_piecewise(model, z, x, y, x_min, x_max, y_min, y_max, n_pieces=4, base_name="prod"):
   
    
    # Standard McCormick envelopes (a basic approximation)
    model.addConstr(z >= x_min * y + x * y_min - x_min * y_min)
    model.addConstr(z >= x_max * y + x * y_max - x_max * y_max)
    model.addConstr(z <= x_min * y + x * y_max - x_min * y_max)
    model.addConstr(z <= x_max * y + x * y_min - x_max * y_min)

    # --- Piecewise Approximation (a tighter approximation) ---
    
    # delta = 1 if the 'i-th' piece of the approximation is active
    # We use 'base_name' here instead of z.VarName to avoid the Gurobi bug
    delta = model.addVars(range(n_pieces), vtype=GRB.BINARY, name=f"d_{base_name}")
    model.addConstr(gp.quicksum(delta[i] for i in range(n_pieces)) == 1) # Only one piece can be active

    # Create helper variables for x, y, and z for each piece
    x_p = model.addVars(range(n_pieces), vtype=GRB.CONTINUOUS, lb=0, name=f"x_p_{base_name}")
    y_p = model.addVars(range(n_pieces), vtype=GRB.CONTINUOUS, lb=0, name=f"y_p_{base_name}")
    z_p = model.addVars(range(n_pieces), vtype=GRB.CONTINUOUS, lb=0, name=f"z_p_{base_name}")

    # x_breaks defines the boundaries of each linear piece
    x_breaks = [x_min + i * (x_max - x_min) / n_pieces for i in range(n_pieces + 1)]
    
    # Reconstruct the original x, y, z as the sum of their pieces
    model.addConstr(x == gp.quicksum(x_p[i] for i in range(n_pieces)))
    model.addConstr(y == gp.quicksum(y_p[i] for i in range(n_pieces)))
    model.addConstr(z == gp.quicksum(z_p[i] for i in range(n_pieces)))

    for i in range(n_pieces):
        # Force x_p[i] and y_p[i] to be 0 unless piece 'i' is active (delta[i] == 1)
        model.addConstr(x_p[i] >= delta[i] * x_breaks[i])
        model.addConstr(x_p[i] <= delta[i] * x_breaks[i+1])
        
        model.addConstr(y_p[i] >= delta[i] * y_min)
        model.addConstr(y_p[i] <= delta[i] * y_max)

        # Apply McCormick constraints *only* to the single active piece
        # This is much tighter than the standard envelopes at the top
        model.addConstr(z_p[i] >= y_min * x_p[i] + x_breaks[i] * y_p[i] - delta[i] * y_min * x_breaks[i])
        model.addConstr(z_p[i] >= y_max * x_p[i] + x_breaks[i+1] * y_p[i] - delta[i] * y_max * x_breaks[i+1])
        model.addConstr(z_p[i] <= y_max * x_p[i] + x_breaks[i] * y_p[i] - delta[i] * y_max * x_breaks[i])
        model.addConstr(z_p[i] <= y_min * x_p[i] + x_breaks[i+1] * y_p[i] - delta[i] * y_min * x_breaks[i+1])


def _compute_partition_parameters(model, x, nodes, node_index, A, W, sup=1.0):
   
    n = A.shape[0]
    start_idx = nodes.index('start')
    end_idx = nodes.index('end')
    non_terminal = [i for i in range(n) if i not in (start_idx, end_idx)]

    def edge_count(i, j):
        # Correctly index the NumPy weight matrix W with integer indices i, j
        return int(W[i, j])

    # Find nodes in Sigma_1 that connect to start/end
    sigma_start_nodes = [a for a in non_terminal if A[start_idx, a]] # Nodes after 'start'
    sigma_end_nodes = [a for a in non_terminal if A[a, end_idx]]     # Nodes before 'end'
    candidate_bs = non_terminal # Sigma_2 is all non_terminal nodes

    # === c1, c2, c3 (Linear Deviating Costs - Def 12, c1, c2, c3) ===
    
    # c1: Cost of start -> Sigma_2 or Sigma_2 -> end
    c1_term1 = gp.quicksum(edge_count(start_idx, j) * x[j] for j in non_terminal if A[start_idx, j])
    c1_term2 = gp.quicksum(edge_count(i, end_idx) * x[i] for i in non_terminal if A[i, end_idx])
    c1 = c1_term1 + c1_term2

    # c2: Cost of Sigma_1 (not end) -> Sigma_2
    w2 = {}
    for i in non_terminal:
        if i in sigma_end_nodes: continue
        for j in non_terminal:
            if A[i, j]:
                w2[(i, j)] = model.addVar(vtype=GRB.BINARY, name=f"w2_{i}_{j}")
                # w2 = 1 iff i in Sigma_1 (x[i]==0) AND j in Sigma_2 (x[j]==1)
                model.addConstr(w2[(i, j)] <= 1 - x[i])
                model.addConstr(w2[(i, j)] <= x[j])
                model.addConstr(w2[(i, j)] >= (1 - x[i]) + x[j] - 1)
    c2 = gp.quicksum(edge_count(i, j) * w2[(i, j)] for (i, j) in w2)

    # c3: Cost of Sigma_2 -> Sigma_1 (not start)
    w3 = {}
    for i in non_terminal:
        for j in non_terminal:
            if j in sigma_start_nodes: continue
            if A[i, j]:
                w3[(i, j)] = model.addVar(vtype=GRB.BINARY, name=f"w3_{i}_{j}")
                # w3 = 1 iff i in Sigma_2 (x[i]==1) AND j in Sigma_1 (x[j]==0)
                model.addConstr(w3[(i, j)] <= x[i])
                model.addConstr(w3[(i, j)] <= 1 - x[j])
                model.addConstr(w3[(i, j)] >= x[i] + (1 - x[j]) - 1)
    c3 = gp.quicksum(edge_count(i, j) * w3[(i, j)] for (i, j) in w3)

    # === M (Max flow variable - Def 12, M) ===
    # This M is the *total frequency* of edges from Sigma_2 to Sigma_1_start
    # or from Sigma_1_end to Sigma_2.
    
    # z_start[a] = 1 if node 'a' is in Sigma_1 (x[a]==0)
    z_start = {}
    for a in sigma_start_nodes:
        z_start[a] = model.addVar(vtype=GRB.BINARY, name=f"z_start_{a}")
        model.addConstr(z_start[a] == 1 - x[a])

    # z_end[a] = 1 if node 'a' is in Sigma_1 (x[a]==0)
    z_end = {}
    for a in sigma_end_nodes:
        z_end[a] = model.addVar(vtype=GRB.BINARY, name=f"z_end_{a}")
        model.addConstr(z_end[a] == 1 - x[a])

    # Total flow incoming to Sigma_1_start (from anywhere)
    incoming_sigma1start_terms = [
        edge_count(i, a) * z_start[a]
        for i in range(n) for a in sigma_start_nodes if A[i, a]
    ]
    incoming_sigma1start = gp.quicksum(incoming_sigma1start_terms) if incoming_sigma1start_terms else 0

    # Total flow outgoing from Sigma_1_end (to anywhere)
    outgoing_sigma1end_terms = [
        edge_count(a, j) * z_end[a]
        for a in sigma_end_nodes for j in range(n) if A[a, j]
    ]
    outgoing_sigma1end = gp.quicksum(outgoing_sigma1end_terms) if outgoing_sigma1end_terms else 0

    total_edges_upper = sum(edge_count(i, j) for i in range(n) for j in range(n) if A[i, j])
    M = model.addVar(lb=0, ub=total_edges_upper, vtype=GRB.CONTINUOUS, name="M")
    model.addConstr(M >= incoming_sigma1start)
    model.addConstr(M >= outgoing_sigma1end)
    M_max = total_edges_upper

    # === c4 (Missing Cost - Def 12, c4) ===
    # This is the non-linear part: cost = max(0, sup * M * ... - count(b,a))
    
    # z_b_2out[b] = 1 if node 'b' is in Sigma_2 (x[b]==1)
    z_b_2out = {}
    for b in non_terminal:
        z_b_2out[b] = model.addVar(vtype=GRB.BINARY, name=f"z2out_{b}")
        model.addConstr(z_b_2out[b] == x[b])

    # w_ab[(a,b)] = 1 iff a in Sigma_1_start AND b in Sigma_2_out
    w_ab = {}
    for a in sigma_start_nodes:
        for b in candidate_bs:
            if A[b, a]: # Only if edge (b,a) exists
                w_ab[(a, b)] = model.addVar(vtype=GRB.BINARY, name=f"w_ab_{a}_{b}")
                model.addConstr(w_ab[(a, b)] <= z_start[a])
                model.addConstr(w_ab[(a, b)] <= z_b_2out[b])
                model.addConstr(w_ab[(a, b)] >= z_start[a] + z_b_2out[b] - 1)

    # S1: Total frequency of start -> Sigma_1_start
    S1_max = sum(edge_count(start_idx, a) for a in sigma_start_nodes)
    S1 = model.addVar(lb=0, ub=S1_max, vtype=GRB.CONTINUOUS, name="S1")
    model.addConstr(S1 == (gp.quicksum(edge_count(start_idx, a) * z_start[a] for a in sigma_start_nodes) if sigma_start_nodes else 0))

    # t_b[b]: Total frequency of (b -> a) where b is in Sigma_2 and a is in Sigma_1_start
    t_b = {}
    for b in candidate_bs:
        sum_q = sum(edge_count(b, a) for a in sigma_start_nodes if A[b, a])
        if sum_q > 0:
            t_b[b] = model.addVar(lb=0, ub=sum_q, vtype=GRB.CONTINUOUS, name=f"t_b_{b}")
            terms = [edge_count(b, a) * w_ab[(a, b)] for a in sigma_start_nodes if (a, b) in w_ab]
            model.addConstr(t_b[b] == (gp.quicksum(terms) if terms else 0))

    # S2: Sum of all t_b[b] (total freq of Sigma_2 -> Sigma_1_start)
    S2_max = sum(edge_count(b, a) for b in candidate_bs for a in sigma_start_nodes if A[b, a])
    S2 = model.addVar(lb=0, ub=S2_max, vtype=GRB.CONTINUOUS, name="S2")
    model.addConstr(S2 == (gp.quicksum(t_b[b] for b in t_b) if t_b else 0))

    # --- Linearization of (p_a * z_start[a]) / S1 ---
    # u[a] will be the variable for this term
    u, y_u = {}, {}
    for a in sigma_start_nodes:
        p_a = edge_count(start_idx, a)
        u[a] = model.addVar(lb=0, ub=1, vtype=GRB.CONTINUOUS, name=f"u_{a}")
        y_u[a] = model.addVar(lb=0, ub=S1_max, vtype=GRB.CONTINUOUS, name=f"y_u_{a}")
        model.addConstr(y_u[a] == p_a * z_start[a])
        # These constraints linearize y_u[a] = u[a] * S1
        model.addConstr(y_u[a] >= 0)
        model.addConstr(y_u[a] <= S1)
        model.addConstr(y_u[a] <= S1_max * u[a])
        model.addConstr(y_u[a] >= S1 - S1_max * (1 - u[a]))
        model.addConstr(u[a] <= z_start[a])

    # --- Linearization of t_b[b] / S2 ---
    # v[b] will be the variable for this term
    v, y_v = {}, {}
    for b in t_b:
        v[b] = model.addVar(lb=0, ub=1, vtype=GRB.CONTINUOUS, name=f"v_{b}")
        y_v[b] = model.addVar(lb=0, ub=S2_max, vtype=GRB.CONTINUOUS, name=f"y_v_{b}")
        model.addConstr(y_v[b] == t_b[b])
        # These constraints linearize y_v[b] = v[b] * S2
        model.addConstr(y_v[b] >= 0)
        model.addConstr(y_v[b] <= S2)
        model.addConstr(y_v[b] <= S2_max * v[b])
        model.addConstr(y_v[b] >= S2 - S2_max * (1 - v[b]))
        model.addConstr(v[b] <= z_b_2out[b])

    # r[(a,b)] = u[a] * v[b]
    r = {}
    for (a, b) in w_ab:
        if a in u and b in v:
            r[(a, b)] = model.addVar(lb=0, ub=1, vtype=GRB.CONTINUOUS, name=f"r_{a}_{b}")
            # Standard linearization for product of two binary-related variables
            model.addConstr(r[(a, b)] >= u[a] + v[b] - 1)
            model.addConstr(r[(a, b)] <= u[a])
            model.addConstr(r[(a, b)] <= v[b])
            model.addConstr(r[(a, b)] >= 0)

    # m[(a,b)] = M * r[(a,b)]
    m = {}
    for (a, b) in r:
        # *** BUG FIX IS HERE ***
        # Create a simple string name first
        m_name = f"m_{a}_{b}"
        m[(a, b)] = model.addVar(lb=0, ub=M_max, vtype=GRB.CONTINUOUS, name=m_name)
        # Pass the simple string 'm_name' as the base_name
        add_mccormick_piecewise(model, m[(a, b)], M, r[(a, b)], 0, M_max, 0, 1, n_pieces=4, base_name=m_name)

    # t4 is the final cost component: max(0, sup * m[a,b] - e_ba)
    t4 = {}
    for (a, b) in m:
        e_ba = edge_count(b, a) # Get weight of edge (b,a)
        t4[(a, b)] = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"t4_{a}_{b}")
        model.addConstr(t4[(a, b)] >= sup * m[(a, b)] - e_ba)

    c4 = gp.quicksum(t4[(a, b)] for (a, b) in t4) if t4 else 0

    # === c5 (Symmetric to c4 - Def 12, c5) ===
    # This is the same logic as c4, but for the other direction:
    # (Sigma_1_end -> Sigma_2)
    
    z_b_2in = {}
    for b in non_terminal:
        z_b_2in[b] = model.addVar(vtype=GRB.BINARY, name=f"z2in_{b}")
        model.addConstr(z_b_2in[b] == x[b])

    wprime = {}
    for a in sigma_end_nodes:
        for b in candidate_bs:
            if A[a, b]:
                wprime[(a, b)] = model.addVar(vtype=GRB.BINARY, name=f"wprime_{a}_{b}")
                model.addConstr(wprime[(a, b)] <= z_end[a])
                model.addConstr(wprime[(a, b)] <= z_b_2in[b])
                model.addConstr(wprime[(a, b)] >= z_end[a] + z_b_2in[b] - 1)

    # S3: Total frequency of Sigma_1_end -> end
    S3_max = sum(edge_count(a, end_idx) for a in sigma_end_nodes)
    S3 = model.addVar(lb=0, ub=S3_max, vtype=GRB.CONTINUOUS, name="S3")
    model.addConstr(S3 == (gp.quicksum(edge_count(a, end_idx) * z_end[a] for a in sigma_end_nodes) if sigma_end_nodes else 0))

    # tprime_b[b]: Total frequency of (a -> b) where a in Sigma_1_end and b in Sigma_2
    tprime_b = {}
    for b in candidate_bs:
        sum_qp = sum(edge_count(a, b) for a in sigma_end_nodes if A[a, b])
        if sum_qp > 0:
            tprime_b[b] = model.addVar(lb=0, ub=sum_qp, vtype=GRB.CONTINUOUS, name=f"tprime_b_{b}")
            terms = [edge_count(a, b) * wprime[(a, b)] for a in sigma_end_nodes if (a, b) in wprime]
            model.addConstr(tprime_b[b] == (gp.quicksum(terms) if terms else 0))

    # S4: Sum of all tprime_b[b] (total freq of Sigma_1_end -> Sigma_2)
    S4_max = sum(edge_count(a, b) for a in sigma_end_nodes for b in candidate_bs if A[a, b])
    S4 = model.addVar(lb=0, ub=S4_max, vtype=GRB.CONTINUOUS, name="S4")
    model.addConstr(S4 == (gp.quicksum(tprime_b[b] for b in tprime_b) if tprime_b else 0))

    # --- Linearization of (ppa * z_end[a]) / S3 ---
    u_p, y_up = {}, {}
    for a in sigma_end_nodes:
        ppa = edge_count(a, end_idx)
        u_p[a] = model.addVar(lb=0, ub=1, vtype=GRB.CONTINUOUS, name=f"uprime_{a}")
        y_up[a] = model.addVar(lb=0, ub=S3_max, vtype=GRB.CONTINUOUS, name=f"y_up_{a}")
        model.addConstr(y_up[a] == ppa * z_end[a])
        model.addConstr(y_up[a] >= 0)
        model.addConstr(y_up[a] <= S3)
        model.addConstr(y_up[a] <= S3_max * u_p[a])
        model.addConstr(y_up[a] >= S3 - S3_max * (1 - u_p[a]))
        model.addConstr(u_p[a] <= z_end[a])

    # --- Linearization of tprime_b[b] / S4 ---
    v_p, y_vp = {}, {}
    for b in tprime_b:
        v_p[b] = model.addVar(lb=0, ub=1, vtype=GRB.CONTINUOUS, name=f"vprime_{b}")
        y_vp[b] = model.addVar(lb=0, ub=S4_max, vtype=GRB.CONTINUOUS, name=f"y_vp_{b}")
        model.addConstr(y_vp[b] == tprime_b[b])
        model.addConstr(y_vp[b] >= 0)
        model.addConstr(y_vp[b] <= S4)
        model.addConstr(y_vp[b] <= S4_max * v_p[b])
        model.addConstr(y_vp[b] >= S4 - S4_max * (1 - v_p[b]))
        model.addConstr(v_p[b] <= z_b_2in[b])

    # r_p[(a,b)] = u_p[a] * v_p[b]
    r_p = {}
    for (a, b) in wprime:
        if a in u_p and b in v_p:
            r_p[(a, b)] = model.addVar(lb=0, ub=1, vtype=GRB.CONTINUOUS, name=f"rprime_{a}_{b}")
            model.addConstr(r_p[(a, b)] >= u_p[a] + v_p[b] - 1)
            model.addConstr(r_p[(a, b)] <= u_p[a])
            model.addConstr(r_p[(a, b)] <= v_p[b])
            model.addConstr(r_p[(a, b)] >= 0)

    # m_p[(a,b)] = M * r_p[(a,b)]
    m_p = {}
    for (a, b) in r_p:
        # *** BUG FIX IS HERE ***
        # Create a simple string name first
        mp_name = f"mprime_{a}_{b}"
        m_p[(a, b)] = model.addVar(lb=0, ub=M_max, vtype=GRB.CONTINUOUS, name=mp_name)
        # Pass the simple string 'mp_name' as the base_name
        add_mccormick_piecewise(model, m_p[(a, b)], M, r_p[(a, b)], 0, M_max, 0, 1, n_pieces=4, base_name=mp_name)

    # t5 is the final cost component: max(0, sup * m_p[a,b] - e_ab)
    t5 = {}
    for (a, b) in m_p:
        e_ab = edge_count(a, b) # Get weight of edge (a,b)
        t5[(a, b)] = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"t5_{a}_{b}")
        model.addConstr(t5[(a, b)] >= sup * m_p[(a, b)] - e_ab)

    c5 = gp.quicksum(t5[(a, b)] for (a, b) in t5) if t5 else 0

    # Return all 5 cost components
    return c1, c2, c3, c4, c5


# --- CONVERTED: This is the main function, now using gurobipy ---
def loop_cut_ilp(G, sup=1.0, debug=False):
    

    start_node = 'start'
    end_node = 'end'

    # --- 1. Preprocessing ---
    try:
        reduced_graph, _, _ = preprocess_graph(G, start_node, end_node)
    except ValueError as e:
        print(f"Error in preprocessing: {e}")
        return []

    if reduced_graph.number_of_nodes() == 0:
        print("Warning: Reduced graph is empty after preprocessing.")
        return []

    if nx.is_directed_acyclic_graph(reduced_graph):
        return set(), set(), None, list(reduced_graph.nodes())

    A, W, nodes, node_index = nx_to_mat_and_weights_loop(reduced_graph)
    n = A.shape[0]

    if start_node not in nodes:
        print(f"Warning: Start node '{start_node}' not in reduced graph nodes: {nodes}")
        return []
    if end_node not in nodes:
        print(f"Warning: End node '{end_node}' not in reduced graph nodes: {nodes}")
        return []
    
    start_idx = nodes.index(start_node)
    end_idx = nodes.index(end_node)
    non_terminal = [i for i in range(n) if i not in (start_idx, end_idx)]

    if not non_terminal:
        print("Warning: No non-terminal nodes found.")
        return []

    # --- 2. Model Setup ---
    model = gp.Model("LOOP_Cut_ILP_Tighter")
    
    # x[i] = 1 if node i is in Sigma_2 (the loop body), 0 if in Sigma_1
    x = model.addVars(non_terminal, vtype=GRB.BINARY, name="x")

    # --- 3. Core Constraints ---
    
    # Non-trivial partition: Both Sigma_1 and Sigma_2 must have at least one node
    model.addConstr(gp.quicksum(x[i] for i in non_terminal) >= 1, name="non_trivial_S2")
    model.addConstr(gp.quicksum(1 - x[i] for i in non_terminal) >= 1, name="non_trivial_S1")

    # === 4. Compute all cost components ===
    # This function builds the entire complex ILP model
    c1, c2, c3, c4, c5 = _compute_partition_parameters(
        model, x, nodes, node_index, A, W, sup
    )

    # === 5. Final Objective ===
    # The objective is just the sum of all 5 cost components
    model.setObjective(c1 + c2 + c3 + c4 + c5, GRB.MINIMIZE)

    # === 6. Solve ===
    if not debug:
        model.setParam('OutputFlag', 0) # Silence Gurobi logs
    else:
        model.write("loop_cut_problem.lp") # Write LP file for debugging
        
    model.optimize()
    
    # === 7. Extract Results ===
    if model.Status == GRB.OPTIMAL:
        Sigma_1 = [nodes[i] for i in non_terminal if round(x[i].X) == 0]
        Sigma_2 = [nodes[i] for i in non_terminal if round(x[i].X) == 1]
        total_cost_ilp = model.ObjVal # Get objective value from Gurobi

        if Sigma_1 or Sigma_2:
            # --- Recalculate cost using the paper's true non-linear function ---
            # This is done because the ILP finds an *approximation* of the cost.
            # We return the *true* cost of the partition the ILP found.
            start_A, end_A, input_B, output_B, start_activities, end_activities = compute_partition_parameters(
            reduced_graph, Sigma_1, Sigma_2, start_node, end_node, node_index, A
            )
    
            Sigma_1_set = set(Sigma_1)
            Sigma_2_set = set(Sigma_2)
            start_A_set = set(start_A)
            end_A_set = set(end_A)
            input_B_set = set(input_B)
            output_B_set = set(output_B)
            start_activities_set = set(start_activities)
            end_activities_set = set(end_activities)
    
            cost_dict = cost_loop(
                reduced_graph, 
                Sigma_1_set, 
                Sigma_2_set, 
                sup, 
                start_A_set, 
                end_A_set, 
                input_B_set, 
                output_B_set, 
                start_activities_set, 
                end_activities_set
               )
            total_cost_paper = calculate_total_cost(cost_dict)
            
            if debug and abs(total_cost_paper - total_cost_ilp) > 1e-5: # Compare floats
                print(f"Cost approximated. ILP Cost: {total_cost_ilp}, Paper Cost: {total_cost_paper}")
        
            return (
                extract_activities(Sigma_1),
                extract_activities(Sigma_2),
                total_cost_paper, # Return the more accurate "paper" cost
                nodes
            )
        else:
             print("Warning: Solver found optimal solution but partitions are empty.")
             return [], [], None, nodes
    else:
        print(f"Warning: Solver failed to find optimal solution. Status: {model.Status}")
        return [], [], None, nodes


def compute_partition_parameters(G, Sigma_1, Sigma_2, start_node, end_node, node_index, A):
    """
    Helper function to find the sets of nodes (start_A, end_A, etc.)
    needed by the final 'cost_loop' function, based on the partition
    (Sigma_1, Sigma_2) found by the solver.
    """
    start_idx = node_index[start_node]
    end_idx = node_index[end_node]
    
    Sigma_1_indices = [node_index[node] for node in Sigma_1]
    Sigma_2_indices = [node_index[node] for node in Sigma_2]
    
    # start_A: nodes in Sigma_1 that have incoming edge from start
    start_A = [node for node in Sigma_1 if A[start_idx, node_index[node]] > 0]
    
    # end_A: nodes in Sigma_1 that have outgoing edge to end  
    end_A = [node for node in Sigma_1 if A[node_index[node], end_idx] > 0]
    
    # input_B: nodes in Sigma_2 that have incoming edges from *outside* Sigma_2
    input_B = []
    for node in Sigma_2:
        node_idx = node_index[node]
        for pred_idx in range(len(A)):
            if A[pred_idx, node_idx] > 0:
                pred_node = [n for n, idx in node_index.items() if idx == pred_idx][0]
                if pred_node not in Sigma_2 or pred_node == start_node:
                    input_B.append(node)
                    break
    
    # output_B: nodes in Sigma_2 that have outgoing edges to *outside* Sigma_2
    output_B = []
    for node in Sigma_2:
        node_idx = node_index[node]
        for succ_idx in range(len(A)):
            if A[node_idx, succ_idx] > 0:
                succ_node = [n for n, idx in node_index.items() if idx == succ_idx][0]
                if succ_node not in Sigma_2 or succ_node == end_node:
                    output_B.append(node)
                    break
    
    # start_activities: *all* nodes (regardless of partition) after start
    start_activities = [node for node in G.nodes() if A[start_idx, node_index.get(node, -1)] > 0]
    
    # end_activities: *all* nodes (regardless of partition) before end
    end_activities = [node for node in G.nodes() if A[node_index.get(node, -1), end_idx] > 0]
    
    return start_A, end_A, input_B, output_B, start_activities, end_activities
     
    


def calculate_total_cost(cost_dict):
    """
    Calculates total cost from the dictionary returned by cost_loop.
    """
    total_cost = 0.0
    
    for edge, costs in cost_dict.items():
        missing = costs.get('missing', 0)
        deviating = costs.get('deviating', 0)
        total_cost += missing + deviating
    
    return total_cost

def loop_cut_tau(G, sup=1.0):
    """
    Calculates the 'tau' cut (empty loop body), which serves as a
    baseline cost.
    """
    start_activities = [tgt for src, tgt in G.out_edges('start') if tgt != 'end']
    end_activities = [src for src, tgt in G.in_edges('end') if src != 'start']

    # M: total weight from any end activity to any start activity
    M = 0
    for a in end_activities:
        for b in start_activities:
            if G.has_edge(a, b):
                M += G[a][b].get('weight', 0)

    total_start_weight = sum(G['start'][b].get('weight', 0) for b in start_activities)
    total_end_weight = sum(G[a]['end'].get('weight', 0) for a in end_activities)

    # Calculate cost using the paper's formula for a tau cut
    total_cost = 0
    for a in end_activities:
        for b in start_activities:
            weight_a_to_b = G.get_edge_data(a, b, {'weight': 0})['weight']
            weight_start_to_b = G['start'][b].get('weight', 0)
            weight_a_to_end = G[a]['end'].get('weight', 0)

            if total_start_weight > 0 and total_end_weight > 0:
                expected = M * sup * (weight_start_to_b / total_start_weight) * (weight_a_to_end / total_end_weight)
                cost_component = max(0, expected - weight_a_to_b)
                total_cost += cost_component

    nodes = list(G.nodes())
    Sigma_1 = extract_activities(nodes)  # All regular activities
    Sigma_2 = set()  # Represents tau/empty
    #total_cost = float('inf')
    #print(start_activities, end_activities)
    return Sigma_1, Sigma_1, total_cost, nodes
