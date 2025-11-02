import pulp
from IM_ilp.Helper_Functions import preprocess_graph, extract_activities, log_to_graph
import numpy as np
from local_pm4py.cut_quality.cost_functions.cost_functions import cost_loop

def nx_to_mat_and_weights_loop(G):
    """ Converts NetworkX graph to adjacency/weight matrices and node maps. """
    nodes = list(G.nodes())
    node_index = {node: i for i, node in enumerate(nodes)}
    n = len(nodes)
    A = np.zeros((n, n), dtype=int)
    W = np.zeros((n, n), dtype=int)
    for u, v, data in G.edges(data=True):
        if u not in node_index or v not in node_index:
            continue
        i = node_index[u]
        j = node_index[v]
        A[i, j] = 1
        W[i, j] = data.get('weight', 1)
    return A, W, nodes, node_index

def add_mccormick_piecewise(prob, z, x, y, x_min, x_max, y_min, y_max, n_pieces=4):
    """
    Adds a tighter piecewise-linear McCormick approximation for z = x * y.
    This replaces the standard 4 inequalities with a more complex but tighter set.
    
    z: pulp.LpVariable for the product (z = x*y)
    x, y: pulp.LpVariable for the two variables
    x_min, x_max: Bounds for x
    y_min, y_max: Bounds for y
    n_pieces: Number of linear pieces to use (more is tighter but slower)
    """
    
    # Standard McCormick envelopes (still useful)
    prob += z >= x_min * y + x * y_min - x_min * y_min
    prob += z >= x_max * y + x * y_max - x_max * y_max
    prob += z <= x_min * y + x * y_max - x_min * y_max
    prob += z <= x_max * y + x * y_min - x_max * y_min

    # --- Piecewise Approximation ---
    # Create binary variables to select a "piece"
    delta = pulp.LpVariable.dicts(f"d_{z.name}", range(n_pieces), cat=pulp.LpBinary)
    prob += pulp.lpSum(delta[i] for i in range(n_pieces)) == 1

    # Create continuous variables for each piece
    x_p = pulp.LpVariable.dicts(f"x_p_{z.name}", range(n_pieces), lowBound=0)
    y_p = pulp.LpVariable.dicts(f"y_p_{z.name}", range(n_pieces), lowBound=0)
    z_p = pulp.LpVariable.dicts(f"z_p_{z.name}", range(n_pieces), lowBound=0)

    # Partition the x-domain
    x_breaks = [x_min + i * (x_max - x_min) / n_pieces for i in range(n_pieces + 1)]
    
    prob += x == pulp.lpSum(x_p[i] for i in range(n_pieces))
    prob += y == pulp.lpSum(y_p[i] for i in range(n_pieces))
    prob += z == pulp.lpSum(z_p[i] for i in range(n_pieces))

    for i in range(n_pieces):
        # x_p[i] is only active in its partition
        prob += x_p[i] >= delta[i] * x_breaks[i]
        prob += x_p[i] <= delta[i] * x_breaks[i+1]
        
        # y_p[i] is only active if this partition is selected
        prob += y_p[i] >= delta[i] * y_min
        prob += y_p[i] <= delta[i] * y_max

        # Link z_p to its corresponding x_p and y_p
        # This applies the standard McCormick bounds *only* to the active partition
        # This is much tighter than applying it to the whole domain
        prob += z_p[i] >= y_min * x_p[i] + x_breaks[i] * y_p[i] - delta[i] * y_min * x_breaks[i]
        prob += z_p[i] >= y_max * x_p[i] + x_breaks[i+1] * y_p[i] - delta[i] * y_max * x_breaks[i+1]
        prob += z_p[i] <= y_max * x_p[i] + x_breaks[i] * y_p[i] - delta[i] * y_max * x_breaks[i]
        prob += z_p[i] <= y_min * x_p[i] + x_breaks[i+1] * y_p[i] - delta[i] * y_min * x_breaks[i+1]


def _compute_partition_parameters(prob, x, nodes, node_index, A, W, sup=1.0):
    """
    Internal helper function to compute all cost parameters (c1-c5, M, S, etc.)
    and add their constraints to the PuLP problem.
    """
    n = A.shape[0]
    start_idx = nodes.index('start')
    end_idx = nodes.index('end')
    non_terminal = [i for i in range(n) if i not in (start_idx, end_idx)]

    def edge_count(i, j):
        try:
            return int(W[nodes[i], nodes[j]])
        except Exception:
            return int(A[i, j])

    sigma_start_nodes = [a for a in non_terminal if A[start_idx, a]]
    sigma_end_nodes = [a for a in non_terminal if A[a, end_idx]]
    candidate_bs = non_terminal

    # === c1, c2, c3 (Linear Deviating Costs) ===
    c1_term1 = pulp.lpSum(edge_count(start_idx, j) * x[j] for j in non_terminal if A[start_idx, j])
    c1_term2 = pulp.lpSum(edge_count(i, end_idx) * x[i] for i in non_terminal if A[i, end_idx])
    c1 = c1_term1 + c1_term2

    w2 = {}
    for i in non_terminal:
        if i in sigma_end_nodes: continue
        for j in non_terminal:
            if A[i, j]:
                w2[(i, j)] = pulp.LpVariable(f"w2_{i}_{j}", cat=pulp.LpBinary)
                prob += w2[(i, j)] <= 1 - x[i]
                prob += w2[(i, j)] <= x[j]
                prob += w2[(i, j)] >= (1 - x[i]) + x[j] - 1
    c2 = pulp.lpSum(edge_count(i, j) * w2[(i, j)] for (i, j) in w2)

    w3 = {}
    for i in non_terminal:
        for j in non_terminal:
            if j in sigma_start_nodes: continue
            if A[i, j]:
                w3[(i, j)] = pulp.LpVariable(f"w3_{i}_{j}", cat=pulp.LpBinary)
                prob += w3[(i, j)] <= x[i]
                prob += w3[(i, j)] <= 1 - x[j]
                prob += w3[(i, j)] >= x[i] + (1 - x[j]) - 1
    c3 = pulp.lpSum(edge_count(i, j) * w3[(i, j)] for (i, j) in w3)

    # === M (Max flow variable) ===
    z_start = {}
    for a in sigma_start_nodes:
        z_start[a] = pulp.LpVariable(f"z_start_{a}", cat=pulp.LpBinary)
        prob += z_start[a] == 1 - x[a]

    z_end = {}
    for a in sigma_end_nodes:
        z_end[a] = pulp.LpVariable(f"z_end_{a}", cat=pulp.LpBinary)
        prob += z_end[a] == 1 - x[a]

    incoming_sigma1start_terms = [
        edge_count(i, a) * z_start[a]
        for i in range(n) for a in sigma_start_nodes if A[i, a]
    ]
    incoming_sigma1start = pulp.lpSum(incoming_sigma1start_terms) if incoming_sigma1start_terms else 0

    outgoing_sigma1end_terms = [
        edge_count(a, j) * z_end[a]
        for a in sigma_end_nodes for j in range(n) if A[a, j]
    ]
    outgoing_sigma1end = pulp.lpSum(outgoing_sigma1end_terms) if outgoing_sigma1end_terms else 0

    total_edges_upper = sum(edge_count(i, j) for i in range(n) for j in range(n) if A[i, j])
    M = pulp.LpVariable("M", lowBound=0, upBound=total_edges_upper)
    prob += M >= incoming_sigma1start
    prob += M >= outgoing_sigma1end
    M_max = total_edges_upper

    # === c4 (Missing Cost) ===
    z_b_2out = {}
    for b in non_terminal:
        z_b_2out[b] = pulp.LpVariable(f"z2out_{b}", cat=pulp.LpBinary)
        prob += z_b_2out[b] == x[b]

    w_ab = {}
    for a in sigma_start_nodes:
        for b in candidate_bs:
            if A[b, a]:
                w_ab[(a, b)] = pulp.LpVariable(f"w_ab_{a}_{b}", cat=pulp.LpBinary)
                prob += w_ab[(a, b)] <= z_start[a]
                prob += w_ab[(a, b)] <= z_b_2out[b]
                prob += w_ab[(a, b)] >= z_start[a] + z_b_2out[b] - 1

    S1_max = sum(edge_count(start_idx, a) for a in sigma_start_nodes)
    S1 = pulp.LpVariable("S1", lowBound=0, upBound=S1_max)
    prob += S1 == (pulp.lpSum(edge_count(start_idx, a) * z_start[a] for a in sigma_start_nodes) if sigma_start_nodes else 0)

    t_b = {}
    for b in candidate_bs:
        sum_q = sum(edge_count(b, a) for a in sigma_start_nodes if A[b, a])
        if sum_q > 0:
            t_b[b] = pulp.LpVariable(f"t_b_{b}", lowBound=0, upBound=sum_q)
            terms = [edge_count(b, a) * w_ab[(a, b)] for a in sigma_start_nodes if (a, b) in w_ab]
            prob += t_b[b] == (pulp.lpSum(terms) if terms else 0)

    S2_max = sum(edge_count(b, a) for b in candidate_bs for a in sigma_start_nodes if A[b, a])
    S2 = pulp.LpVariable("S2", lowBound=0, upBound=S2_max)
    prob += S2 == (pulp.lpSum(t_b[b] for b in t_b) if t_b else 0)

    u, y_u = {}, {}
    for a in sigma_start_nodes:
        p_a = edge_count(start_idx, a)
        u[a] = pulp.LpVariable(f"u_{a}", lowBound=0, upBound=1)
        y_u[a] = pulp.LpVariable(f"y_u_{a}", lowBound=0, upBound=S1_max)
        prob += y_u[a] == p_a * z_start[a]
        prob += y_u[a] >= 0
        prob += y_u[a] <= S1
        prob += y_u[a] <= S1_max * u[a]
        prob += y_u[a] >= S1 - S1_max * (1 - u[a])
        prob += u[a] <= z_start[a]

    v, y_v = {}, {}
    for b in t_b:
        v[b] = pulp.LpVariable(f"v_{b}", lowBound=0, upBound=1)
        y_v[b] = pulp.LpVariable(f"y_v_{b}", lowBound=0, upBound=S2_max)
        prob += y_v[b] == t_b[b]
        prob += y_v[b] >= 0
        prob += y_v[b] <= S2
        prob += y_v[b] <= S2_max * v[b]
        prob += y_v[b] >= S2 - S2_max * (1 - v[b])
        prob += v[b] <= z_b_2out[b]

    r = {}
    for (a, b) in w_ab:
        if a in u and b in v:
            r[(a, b)] = pulp.LpVariable(f"r_{a}_{b}", lowBound=0, upBound=1)
            prob += r[(a, b)] >= u[a] + v[b] - 1
            prob += r[(a, b)] <= u[a]
            prob += r[(a, b)] <= v[b]
            prob += r[(a, b)] >= 0

    m = {}
    for (a, b) in r:
        m[(a, b)] = pulp.LpVariable(f"m_{a}_{b}", lowBound=0, upBound=M_max)
        # --- START: Tighter Approximation ---
        # Replaced the 4 standard McCormick constraints
        add_mccormick_piecewise(prob, m[(a, b)], M, r[(a, b)], 0, M_max, 0, 1, n_pieces=4)
        # --- END: Tighter Approximation ---

    t4 = {}
    for (a, b) in m:
        e_ba = edge_count(b, a)
        t4[(a, b)] = pulp.LpVariable(f"t4_{a}_{b}", lowBound=0)
        prob += t4[(a, b)] >= sup * m[(a, b)] - e_ba

    c4 = pulp.lpSum(t4[(a, b)] for (a, b) in t4) if t4 else 0

    # === c5 (Symmetric to c4) ===
    z_b_2in = {}
    for b in non_terminal:
        z_b_2in[b] = pulp.LpVariable(f"z2in_{b}", cat=pulp.LpBinary)
        prob += z_b_2in[b] == x[b]

    wprime = {}
    for a in sigma_end_nodes:
        for b in candidate_bs:
            if A[a, b]:
                wprime[(a, b)] = pulp.LpVariable(f"wprime_{a}_{b}", cat=pulp.LpBinary)
                prob += wprime[(a, b)] <= z_end[a]
                prob += wprime[(a, b)] <= z_b_2in[b]
                prob += wprime[(a, b)] >= z_end[a] + z_b_2in[b] - 1

    S3_max = sum(edge_count(a, end_idx) for a in sigma_end_nodes)
    S3 = pulp.LpVariable("S3", lowBound=0, upBound=S3_max)
    prob += S3 == (pulp.lpSum(edge_count(a, end_idx) * z_end[a] for a in sigma_end_nodes) if sigma_end_nodes else 0)

    tprime_b = {}
    for b in candidate_bs:
        sum_qp = sum(edge_count(a, b) for a in sigma_end_nodes if A[a, b])
        if sum_qp > 0:
            tprime_b[b] = pulp.LpVariable(f"tprime_b_{b}", lowBound=0, upBound=sum_qp)
            terms = [edge_count(a, b) * wprime[(a, b)] for a in sigma_end_nodes if (a, b) in wprime]
            prob += tprime_b[b] == (pulp.lpSum(terms) if terms else 0)

    S4_max = sum(edge_count(a, b) for a in sigma_end_nodes for b in candidate_bs if A[a, b])
    S4 = pulp.LpVariable("S4", lowBound=0, upBound=S4_max)
    prob += S4 == (pulp.lpSum(tprime_b[b] for b in tprime_b) if tprime_b else 0)

    u_p, y_up = {}, {}
    for a in sigma_end_nodes:
        ppa = edge_count(a, end_idx)
        u_p[a] = pulp.LpVariable(f"uprime_{a}", lowBound=0, upBound=1)
        y_up[a] = pulp.LpVariable(f"y_up_{a}", lowBound=0, upBound=S3_max)
        prob += y_up[a] == ppa * z_end[a]
        prob += y_up[a] >= 0
        prob += y_up[a] <= S3
        prob += y_up[a] <= S3_max * u_p[a]
        prob += y_up[a] >= S3 - S3_max * (1 - u_p[a])
        prob += u_p[a] <= z_end[a]

    v_p, y_vp = {}, {}
    for b in tprime_b:
        v_p[b] = pulp.LpVariable(f"vprime_{b}", lowBound=0, upBound=1)
        y_vp[b] = pulp.LpVariable(f"y_vp_{b}", lowBound=0, upBound=S4_max)
        prob += y_vp[b] == tprime_b[b]
        prob += y_vp[b] >= 0
        prob += y_vp[b] <= S4
        prob += y_vp[b] <= S4_max * v_p[b]
        prob += y_vp[b] >= S4 - S4_max * (1 - v_p[b])
        prob += v_p[b] <= z_b_2in[b]

    r_p = {}
    for (a, b) in wprime:
        if a in u_p and b in v_p:
            r_p[(a, b)] = pulp.LpVariable(f"rprime_{a}_{b}", lowBound=0, upBound=1)
            prob += r_p[(a, b)] >= u_p[a] + v_p[b] - 1
            prob += r_p[(a, b)] <= u_p[a]
            prob += r_p[(a, b)] <= v_p[b]
            prob += r_p[(a, b)] >= 0

    m_p = {}
    for (a, b) in r_p:
        m_p[(a, b)] = pulp.LpVariable(f"mprime_{a}_{b}", lowBound=0, upBound=M_max)
        # --- START: Tighter Approximation ---
        add_mccormick_piecewise(prob, m_p[(a, b)], M, r_p[(a, b)], 0, M_max, 0, 1, n_pieces=4)
        # --- END: Tighter Approximation ---

    t5 = {}
    for (a, b) in m_p:
        e_ab = edge_count(a, b)
        t5[(a, b)] = pulp.LpVariable(f"t5_{a}_{b}", lowBound=0)
        prob += t5[(a, b)] >= sup * m_p[(a, b)] - e_ab

    c5 = pulp.lpSum(t5[(a, b)] for (a, b) in t5) if t5 else 0

    return c1, c2, c3, c4, c5


def loop_cut_ilp(G, sup=1.0, debug=False):
    """
    MIQCP formulation for the loop cut cost.
    This version uses a tighter piecewise-linear approximation for the
    M * r and M * r' terms, getting it "closer" to the true
    non-linear cost function from the paper.
    """

    start_node = 'start'
    end_node = 'end'

    try:
        reduced_graph, _, _ = preprocess_graph(G, start_node, end_node)
    except ValueError as e:
        print(f"Error in preprocessing: {e}")
        return []

    if reduced_graph.number_of_nodes() == 0:
        print("Warning: Reduced graph is empty after preprocessing.")
        return []

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

    def edge_count(i, j):
        try:
            return int(W[nodes[i], nodes[j]])
        except Exception:
            return int(A[i, j])

    prob = pulp.LpProblem("LOOP_Cut_ILP_Tighter", pulp.LpMinimize)
    x = pulp.LpVariable.dicts("x", non_terminal, cat=pulp.LpBinary)

    # Non-trivial partition
    prob += pulp.lpSum(x[i] for i in non_terminal) >= 1
    prob += pulp.lpSum(1 - x[i] for i in non_terminal) >= 1

    # === Compute all cost components ===
    c1, c2, c3, c4, c5 = _compute_partition_parameters(
        prob, x, nodes, node_index, A, W, sup
    )

    # === Final Objective ===
    prob += c1 + c2 + c3 + c4 + c5

    # Solve
    solver_options = {'msg': False}
    if debug:
        solver_options = {'msg': True}
        prob.writeLP("loop_cut_problem.lp")
        
    prob.solve(pulp.GUROBI_CMD(**solver_options))
    
    if prob.status == pulp.LpStatusOptimal:
        Sigma_1 = [nodes[i] for i in non_terminal if pulp.value(x[i]) == 0]
        Sigma_2 = [nodes[i] for i in non_terminal if pulp.value(x[i]) == 1]
        total_cost = pulp.value(prob.objective)

        if Sigma_1 or Sigma_2:  # Only compute if we have a valid partition
            start_A, end_A, input_B, output_B, start_activities, end_activities = compute_partition_parameters(
            reduced_graph, Sigma_1, Sigma_2, start_node, end_node, node_index, A
            )
    
    # Convert lists to sets as expected by cost_loop
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
        if total_cost_paper != total_cost:
            print("Cost approximated")
        
        return (
            extract_activities(Sigma_1),
            extract_activities(Sigma_2),
            total_cost,
            nodes
        )
    else:
        print(f"Warning: Solver failed to find optimal solution. Status: {pulp.LpStatus[prob.status]}")
        return [], [], None, nodes


def compute_partition_parameters(G, Sigma_1, Sigma_2, start_node, end_node, node_index, A):
    """Compute the required parameters for cost_loop function"""
    start_idx = node_index[start_node]
    end_idx = node_index[end_node]
    
    # Convert to node indices for easier processing
    Sigma_1_indices = [node_index[node] for node in Sigma_1]
    Sigma_2_indices = [node_index[node] for node in Sigma_2]
    
    # start_A: nodes in Sigma_1 that have incoming edge from start
    start_A = [node for node in Sigma_1 if A[start_idx, node_index[node]] > 0]
    
    # end_A: nodes in Sigma_1 that have outgoing edge to end  
    end_A = [node for node in Sigma_1 if A[node_index[node], end_idx] > 0]
    
    # input_B: nodes in Sigma_2 that have incoming edges from outside Sigma_2
    input_B = []
    for node in Sigma_2:
        node_idx = node_index[node]
        # Check if any predecessor is not in Sigma_2 (could be in Sigma_1, start, or end)
        for pred_idx in range(len(A)):
            if A[pred_idx, node_idx] > 0:
                pred_node = [n for n, idx in node_index.items() if idx == pred_idx][0]
                if pred_node not in Sigma_2 or pred_node == start_node:
                    input_B.append(node)
                    break
    
    # output_B: nodes in Sigma_2 that have outgoing edges to outside Sigma_2
    output_B = []
    for node in Sigma_2:
        node_idx = node_index[node]
        # Check if any successor is not in Sigma_2 (could be in Sigma_1, start, or end)
        for succ_idx in range(len(A)):
            if A[node_idx, succ_idx] > 0:
                succ_node = [n for n, idx in node_index.items() if idx == succ_idx][0]
                if succ_node not in Sigma_2 or succ_node == end_node:
                    output_B.append(node)
                    break
    
    # start_activities: all nodes that have incoming edges from start
    start_activities = [node for node in G.nodes() if A[start_idx, node_index.get(node, -1)] > 0]
    
    # end_activities: all nodes that have outgoing edges to end
    end_activities = [node for node in G.nodes() if A[node_index.get(node, -1), end_idx] > 0]
    
    return start_A, end_A, input_B, output_B, start_activities, end_activities
     
    


def calculate_total_cost(cost_dict):
    """
    Calculate total cost from cost dictionary by summing missing and deviating values.
    
    Args:
        cost_dict: Dictionary with edge tuples as keys and {'deviating': x, 'missing': y} as values
    
    Returns:
        float: Total cost (sum of all missing and deviating values)
    """
    total_cost = 0.0
    
    for edge, costs in cost_dict.items():
        missing = costs.get('missing', 0)
        deviating = costs.get('deviating', 0)
        total_cost += missing + deviating
    
    return total_cost


def loop_cut_tau(G, sup=1.0):
    # Get all start activities (nodes connected from start)
    start_activities = [tgt for src, tgt in G.out_edges('start') if tgt != 'end']
    # Get all end activities (nodes connected to end)
    end_activities = [src for src, tgt in G.in_edges('end') if src != 'start']

    # Calculate M: total weight from any end activity to any start activity
    M = 0
    for a in end_activities:
        for b in start_activities:
            if G.has_edge(a, b):
                M += G[a][b].get('weight', 0)

    # Calculate total weights
    total_start_weight = sum(G['start'][b].get('weight', 0) for b in start_activities)
    total_end_weight = sum(G[a]['end'].get('weight', 0) for a in end_activities)

    # Calculate cost using your formula
    total_cost = 0
    for a in end_activities:
        for b in start_activities:
            weight_a_to_b = G.get_edge_data(a, b, {'weight': 0})['weight']
            weight_start_to_b = G['start'][b].get('weight', 0)
            weight_a_to_end = G[a]['end'].get('weight', 0)

            # Avoid division by zero
            if total_start_weight > 0 and total_end_weight > 0:
                expected = M * sup * (weight_start_to_b / total_start_weight) * (weight_a_to_end / total_end_weight)
                cost_component = max(0, expected - weight_a_to_b)
                total_cost += cost_component

    nodes = list(G.nodes())
    Sigma_1 = extract_activities(nodes)  # All regular activities
    Sigma_2 = set()  # Represents tau/empty

    return Sigma_1, Sigma_2, total_cost, nodes

