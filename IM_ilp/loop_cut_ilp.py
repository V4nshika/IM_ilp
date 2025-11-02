import pulp
from IM_ilp.Helper_Functions import preprocess_graph, extract_activities, log_to_graph
import numpy as np
from local_pm4py.cut_quality.cost_functions.cost_functions import cost_loop

def nx_to_mat_and_weights_loop(G):
    """
    Converts a NetworkX graph (G) into NumPy adjacency (A) and weight (W)
    matrices. It also provides mappings between node names and their
    integer indices in the matrices.
    """
    nodes = list(G.nodes())
    node_index = {node: i for i, node in enumerate(nodes)}
    n = len(nodes)
    
    # Initialize adjacency (A) and weight (W) matrices
    A = np.zeros((n, n), dtype=int)
    W = np.zeros((n, n), dtype=int)
    
    # Populate matrices based on graph edges
    for u, v, data in G.edges(data=True):
        if u not in node_index or v not in node_index:
            continue  # Skip if node somehow isn't in the list
        i = node_index[u]
        j = node_index[v]
        A[i, j] = 1  # Mark edge existence
        W[i, j] = data.get('weight', 1)  # Store edge weight
    return A, W, nodes, node_index

def add_mccormick_piecewise(prob, z, x, y, x_min, x_max, y_min, y_max, n_pieces=4):
    """
    Adds a tighter piecewise-linear McCormick approximation for z = x * y.
    This replaces the standard 4 inequalities with a more complex but tighter set
    by breaking the domain of 'x' into 'n_pieces'.

    This is used to linearize a product of two variables (z = x * y),
    which is a non-linear constraint.
    
    z: pulp.LpVariable for the product (z = x*y)
    x, y: pulp.LpVariable for the two variables
    x_min, x_max: Bounds for x
    y_min, y_max: Bounds for y
    n_pieces: Number of linear pieces to use (more is tighter but slower)
    """
    
    # Standard McCormick envelopes (still useful as loose bounds)
    prob += z >= x_min * y + x * y_min - x_min * y_min
    prob += z >= x_max * y + x * y_max - x_max * y_max
    prob += z <= x_min * y + x * y_max - x_min * y_max
    prob += z <= x_max * y + x * y_min - x_max * y_min

    # --- Piecewise Approximation ---
    # Create binary variables to select exactly one "piece"
    delta = pulp.LpVariable.dicts(f"d_{z.name}", range(n_pieces), cat=pulp.LpBinary)
    prob += pulp.lpSum(delta[i] for i in range(n_pieces)) == 1

    # Create continuous variables for each piece
    # x_p[i] will hold the value of x *if* it's in piece i
    # y_p[i] will hold the value of y *if* piece i is selected
    # z_p[i] will hold the value of z *if* piece i is selected
    x_p = pulp.LpVariable.dicts(f"x_p_{z.name}", range(n_pieces), lowBound=0)
    y_p = pulp.LpVariable.dicts(f"y_p_{z.name}", range(n_pieces), lowBound=0)
    z_p = pulp.LpVariable.dicts(f"z_p_{z.name}", range(n_pieces), lowBound=0)

    # Partition the x-domain into 'n_pieces' equal segments
    x_breaks = [x_min + i * (x_max - x_min) / n_pieces for i in range(n_pieces + 1)]
    
    # Reconstruct the original variables from their piecewise counterparts
    prob += x == pulp.lpSum(x_p[i] for i in range(n_pieces))
    prob += y == pulp.lpSum(y_p[i] for i in range(n_pieces))
    prob += z == pulp.lpSum(z_p[i] for i in range(n_pieces))

    for i in range(n_pieces):
        # x_p[i] is only active (non-zero) in its partition
        # If delta[i] is 0, x_p[i] is forced to 0.
        # If delta[i] is 1, x_p[i] is bounded by the piece's x-breaks.
        prob += x_p[i] >= delta[i] * x_breaks[i]
        prob += x_p[i] <= delta[i] * x_breaks[i+1]
        
        # y_p[i] is only active if this partition is selected
        # If delta[i] is 0, y_p[i] is forced to 0.
        # If delta[i] is 1, y_p[i] is bounded by y's full domain.
        prob += y_p[i] >= delta[i] * y_min
        prob += y_p[i] <= delta[i] * y_max

        # Link z_p to its corresponding x_p and y_p
        # This applies the standard McCormick bounds *only* to the active partition
        # (using x_breaks[i] and x_breaks[i+1] as the tighter bounds for x).
        # This is much tighter than applying it to the whole domain.
        prob += z_p[i] >= y_min * x_p[i] + x_breaks[i] * y_p[i] - delta[i] * y_min * x_breaks[i]
        prob += z_p[i] >= y_max * x_p[i] + x_breaks[i+1] * y_p[i] - delta[i] * y_max * x_breaks[i+1]
        prob += z_p[i] <= y_max * x_p[i] + x_breaks[i] * y_p[i] - delta[i] * y_max * x_breaks[i]
        prob += z_p[i] <= y_min * x_p[i] + x_breaks[i+1] * y_p[i] - delta[i] * y_min * x_breaks[i+1]


def _compute_partition_parameters(prob, x, nodes, node_index, A, W, sup=1.0):
    """
    Internal helper function to compute all cost parameters (c1-c5, M, S, etc.)
    and add their constraints to the PuLP problem.
    
    This function defines the core logic of the loop cut cost calculation.
    
    x: PuLP variable dict (binary), x[i]=1 if node i is in Sigma_2, 0 if in Sigma_1.
    """
    n = A.shape[0]
    start_idx = nodes.index('start')
    end_idx = nodes.index('end')
    # non_terminal nodes are the actual activities (not 'start' or 'end')
    non_terminal = [i for i in range(n) if i not in (start_idx, end_idx)]

    def edge_count(i, j):
        """Helper to get edge weight safely."""
        try:
            # Note: This is accessing W by node names, not indices.
            # This seems incorrect based on the function signature.
            # Corrected version (assuming W uses indices):
            # return int(W[i, j])
            # Sticking to original code's logic:
            return int(W[nodes[i], nodes[j]])
        except Exception:
            # Fallback, but the above access pattern is unusual.
            # If W is the matrix from nx_to_mat_and_weights_loop,
            # it should be accessed as W[i, j].
            # Let's assume the helper is intended to be:
            return int(A[i, j])
    
    # Re-defining edge_count based on how W is built in nx_to_mat_and_weights_loop
    # The matrix W is indexed by integers, not node names.
    def edge_count(i, j):
        """Helper to get edge weight safely (Corrected)."""
        if i >= 0 and i < n and j >= 0 and j < n:
            return int(W[i, j])
        return 0

    # Identify key node sets based on graph structure
    sigma_start_nodes = [a for a in non_terminal if A[start_idx, a]] # Nodes after 'start'
    sigma_end_nodes = [a for a in non_terminal if A[a, end_idx]]     # Nodes before 'end'
    candidate_bs = non_terminal # All activities are candidates for loops

    # === c1, c2, c3 (Linear Deviating Costs) ===
    
    # c1: Cost of edges from 'start' to Sigma_2 (x[j]=1) or from Sigma_1 (x[i]=0) to 'end'.
    # This is slightly different from paper; it's 'start' to Sigma_2 or Sigma_2 to 'end'.
    # Let's re-read the code:
    # c1_term1: 'start' -> j (in Sigma_2, since x[j]=1)
    # c1_term2: i (in Sigma_2, since x[i]=1) -> 'end'
    # Ah, no, x[i] is the variable. So:
    # c1_term1: sum(weight(start, j) * x[j]) for j in non_terminal
    # c1_term2: sum(weight(i, end) * x[i]) for i in non_terminal
    # This is the cost for edges between 'start'/'end' and Sigma_2.
    c1_term1 = pulp.lpSum(edge_count(start_idx, j) * x[j] for j in non_terminal if A[start_idx, j])
    c1_term2 = pulp.lpSum(edge_count(i, end_idx) * x[i] for i in non_terminal if A[i, end_idx])
    c1 = c1_term1 + c1_term2

    # c2: Cost of edges from Sigma_1 to Sigma_2
    # w2[(i,j)] = 1 IF (i in Sigma_1) AND (j in Sigma_2)
    # This is (1 - x[i]) * x[j]
    w2 = {}
    for i in non_terminal:
        if i in sigma_end_nodes: continue # Optimization: ignore nodes before 'end'
        for j in non_terminal:
            if A[i, j]:
                w2[(i, j)] = pulp.LpVariable(f"w2_{i}_{j}", cat=pulp.LpBinary)
                # Linearization of w2 = (1-x[i]) * x[j]
                prob += w2[(i, j)] <= 1 - x[i]
                prob += w2[(i, j)] <= x[j]
                prob += w2[(i, j)] >= (1 - x[i]) + x[j] - 1
    c2 = pulp.lpSum(edge_count(i, j) * w2[(i, j)] for (i, j) in w2)

    # c3: Cost of edges from Sigma_2 to Sigma_1
    # w3[(i,j)] = 1 IF (i in Sigma_2) AND (j in Sigma_1)
    # This is x[i] * (1 - x[j])
    w3 = {}
    for i in non_terminal:
        for j in non_terminal:
            if j in sigma_start_nodes: continue # Optimization: ignore nodes after 'start'
            if A[i, j]:
                w3[(i, j)] = pulp.LpVariable(f"w3_{i}_{j}", cat=pulp.LpBinary)
                # Linearization of w3 = x[i] * (1-x[j])
                prob += w3[(i, j)] <= x[i]
                prob += w3[(i, j)] <= 1 - x[j]
                prob += w3[(i, j)] >= x[i] + (1 - x[j]) - 1
    c3 = pulp.lpSum(edge_count(i, j) * w3[(i, j)] for (i, j) in w3)

    # === M (Max flow variable) ===
    # This represents the total "loop flow" from Sigma_1 end nodes to Sigma_1 start nodes.
    # It's approximated by the max of flow *into* Sigma_1 start nodes
    # or *out of* Sigma_1 end nodes.

    # z_start[a] = 1 IF a in Sigma_1 (i.e., x[a]=0)
    z_start = {}
    for a in sigma_start_nodes:
        z_start[a] = pulp.LpVariable(f"z_start_{a}", cat=pulp.LpBinary)
        prob += z_start[a] == 1 - x[a]

    # z_end[a] = 1 IF a in Sigma_1 (i.e., x[a]=0)
    z_end = {}
    for a in sigma_end_nodes:
        z_end[a] = pulp.LpVariable(f"z_end_{a}", cat=pulp.LpBinary)
        prob += z_end[a] == 1 - x[a]

    # Total weight of edges from *anywhere* into Sigma_1 start nodes
    incoming_sigma1start_terms = [
        edge_count(i, a) * z_start[a] # Cost is counted only if a is in Sigma_1
        for i in range(n) for a in sigma_start_nodes if A[i, a]
    ]
    incoming_sigma1start = pulp.lpSum(incoming_sigma1start_terms) if incoming_sigma1start_terms else 0

    # Total weight of edges from Sigma_1 end nodes to *anywhere*
    outgoing_sigma1end_terms = [
        edge_count(a, j) * z_end[a] # Cost is counted only if a is in Sigma_1
        for a in sigma_end_nodes for j in range(n) if A[a, j]
    ]
    outgoing_sigma1end = pulp.lpSum(outgoing_sigma1end_terms) if outgoing_sigma1end_terms else 0

    total_edges_upper = sum(edge_count(i, j) for i in range(n) for j in range(n) if A[i, j])
    M = pulp.LpVariable("M", lowBound=0, upBound=total_edges_upper)
    # M is the maximum of these two flows
    prob += M >= incoming_sigma1start
    prob += M >= outgoing_sigma1end
    M_max = total_edges_upper # Upper bound for McCormick approximation

    # === c4 (Missing Cost - Part 1) ===
    # Cost for *missing* loop-back edges (from Sigma_2 to Sigma_1 start nodes)
    
    # z_b_2out[b] = 1 IF b in Sigma_2
    z_b_2out = {}
    for b in non_terminal:
        z_b_2out[b] = pulp.LpVariable(f"z2out_{b}", cat=pulp.LpBinary)
        prob += z_b_2out[b] == x[b]

    # w_ab[(a,b)] = 1 IF (a in Sigma_1 start nodes) AND (b in Sigma_2)
    # w_ab = z_start[a] * z_b_2out[b]
    w_ab = {}
    for a in sigma_start_nodes:
        for b in candidate_bs:
            if A[b, a]: # Only for existing edges (b -> a)
                w_ab[(a, b)] = pulp.LpVariable(f"w_ab_{a}_{b}", cat=pulp.LpBinary)
                # Linearization of w_ab = z_start[a] * z_b_2out[b]
                prob += w_ab[(a, b)] <= z_start[a]
                prob += w_ab[(a, b)] <= z_b_2out[b]
                prob += w_ab[(a, b)] >= z_start[a] + z_b_2out[b] - 1

    # S1: Total weight from 'start' to Sigma_1 start nodes
    S1_max = sum(edge_count(start_idx, a) for a in sigma_start_nodes)
    S1 = pulp.LpVariable("S1", lowBound=0, upBound=S1_max)
    prob += S1 == (pulp.lpSum(edge_count(start_idx, a) * z_start[a] for a in sigma_start_nodes) if sigma_start_nodes else 0)

    # t_b[b]: Total weight from node b (in Sigma_2) to Sigma_1 start nodes
    t_b = {}
    for b in candidate_bs:
        sum_q = sum(edge_count(b, a) for a in sigma_start_nodes if A[b, a])
        if sum_q > 0:
            t_b[b] = pulp.LpVariable(f"t_b_{b}", lowBound=0, upBound=sum_q)
            # Sum of weights (b -> a) *if* a in Sigma_1 AND b in Sigma_2
            terms = [edge_count(b, a) * w_ab[(a, b)] for a in sigma_start_nodes if (a, b) in w_ab]
            prob += t_b[b] == (pulp.lpSum(terms) if terms else 0)

    # S2: Total weight from Sigma_2 to Sigma_1 start nodes
    S2_max = sum(edge_count(b, a) for b in candidate_bs for a in sigma_start_nodes if A[b, a])
    S2 = pulp.LpVariable("S2", lowBound=0, upBound=S2_max)
    prob += S2 == (pulp.lpSum(t_b[b] for b in t_b) if t_b else 0)

    # The following (u, y_u, v, y_v, r) is a complex linearization for:
    # cost = M * (p_a / S1) * (t_b[b] / S2)
    # This is to avoid division by S1 and S2, which are variables.
    # It seems to be calculating proportions.

    # u[a] is a helper variable, active if z_start[a] is active.
    u, y_u = {}, {}
    for a in sigma_start_nodes:
        p_a = edge_count(start_idx, a) # Weight 'start' -> a
        u[a] = pulp.LpVariable(f"u_{a}", lowBound=0, upBound=1)
        y_u[a] = pulp.LpVariable(f"y_u_{a}", lowBound=0, upBound=S1_max)
        # y_u[a] = p_a * z_start[a] (weight 'start'->a if a in Sigma_1)
        prob += y_u[a] == p_a * z_start[a]
        # This is a linearization of u[a] = y_u[a] / S1 (proportion)
        prob += y_u[a] >= 0
        prob += y_u[a] <= S1
        prob += y_u[a] <= S1_max * u[a]
        prob += y_u[a] >= S1 - S1_max * (1 - u[a])
        prob += u[a] <= z_start[a]

    # v[b] is a helper variable, active if z_b_2out[b] is active.
    v, y_v = {}, {}
    for b in t_b:
        v[b] = pulp.LpVariable(f"v_{b}", lowBound=0, upBound=1)
        y_v[b] = pulp.LpVariable(f"y_v_{b}", lowBound=0, upBound=S2_max)
        # y_v[b] = t_b[b] (total weight from b (in S2) to S1 start nodes)
        prob += y_v[b] == t_b[b]
        # This is a linearization of v[b] = y_v[b] / S2 (proportion)
        prob += y_v[b] >= 0
        prob += y_v[b] <= S2
        prob += y_v[b] <= S2_max * v[b]
        prob += y_v[b] >= S2 - S2_max * (1 - v[b])
        prob += v[b] <= z_b_2out[b]

    # r[(a,b)] = u[a] * v[b]
    # Represents (p_a / S1) * (t_b[b] / S2)
    r = {}
    for (a, b) in w_ab:
        if a in u and b in v:
            r[(a, b)] = pulp.LpVariable(f"r_{a}_{b}", lowBound=0, upBound=1)
            # Linearization of r = u * v (product of two continuous vars)
            # This is a standard McCormick relaxation, not the piecewise one.
            # This could be a source of approximation error.
            prob += r[(a, b)] >= u[a] + v[b] - 1
            prob += r[(a, b)] <= u[a]
            prob += r[(a, b)] <= v[b]
            prob += r[(a, b)] >= 0

    # m[(a,b)] = M * r[(a,b)]
    # This is the "expected" flow for this (a, b) pair
    m = {}
    for (a, b) in r:
        m[(a, b)] = pulp.LpVariable(f"m_{a}_{b}", lowBound=0, upBound=M_max)
        # --- START: Tighter Approximation ---
        # Using the piecewise approximation for the product of M * r
        add_mccormick_piecewise(prob, m[(a, b)], M, r[(a, b)], 0, M_max, 0, 1, n_pieces=4)
        # --- END: Tighter Approximation ---

    # t4 is the final cost term
    # t4 = max(0, sup * expected_flow - actual_flow)
    t4 = {}
    for (a, b) in m:
        e_ba = edge_count(b, a) # Actual flow (b -> a)
        t4[(a, b)] = pulp.LpVariable(f"t4_{a}_{b}", lowBound=0)
        # t4 >= sup * m - e_ba
        prob += t4[(a, b)] >= sup * m[(a, b)] - e_ba

    # c4 is the sum of all missing costs
    c4 = pulp.lpSum(t4[(a, b)] for (a, b) in t4) if t4 else 0

    # === c5 (Symmetric to c4) ===
    # Cost for *missing* loop-forward edges (from Sigma_1 end nodes to Sigma_2)
    
    # z_b_2in[b] = 1 IF b in Sigma_2
    z_b_2in = {}
    for b in non_terminal:
        z_b_2in[b] = pulp.LpVariable(f"z2in_{b}", cat=pulp.LpBinary)
        prob += z_b_2in[b] == x[b]

    # wprime[(a,b)] = 1 IF (a in Sigma_1 end nodes) AND (b in Sigma_2)
    # wprime = z_end[a] * z_b_2in[b]
    wprime = {}
    for a in sigma_end_nodes:
        for b in candidate_bs:
            if A[a, b]: # Only for existing edges (a -> b)
                wprime[(a, b)] = pulp.LpVariable(f"wprime_{a}_{b}", cat=pulp.LpBinary)
                # Linearization
                prob += wprime[(a, b)] <= z_end[a]
                prob += wprime[(a, b)] <= z_b_2in[b]
                prob += wprime[(a, b)] >= z_end[a] + z_b_2in[b] - 1

    # S3: Total weight from Sigma_1 end nodes to 'end'
    S3_max = sum(edge_count(a, end_idx) for a in sigma_end_nodes)
    S3 = pulp.LpVariable("S3", lowBound=0, upBound=S3_max)
    prob += S3 == (pulp.lpSum(edge_count(a, end_idx) * z_end[a] for a in sigma_end_nodes) if sigma_end_nodes else 0)

    # tprime_b[b]: Total weight from Sigma_1 end nodes to node b (in Sigma_2)
    tprime_b = {}
    for b in candidate_bs:
        sum_qp = sum(edge_count(a, b) for a in sigma_end_nodes if A[a, b])
        if sum_qp > 0:
            tprime_b[b] = pulp.LpVariable(f"tprime_b_{b}", lowBound=0, upBound=sum_qp)
            # Sum of weights (a -> b) *if* a in Sigma_1 AND b in Sigma_2
            terms = [edge_count(a, b) * wprime[(a, b)] for a in sigma_end_nodes if (a, b) in wprime]
            prob += tprime_b[b] == (pulp.lpSum(terms) if terms else 0)

    # S4: Total weight from Sigma_1 end nodes to Sigma_2
    S4_max = sum(edge_count(a, b) for a in sigma_end_nodes for b in candidate_bs if A[a, b])
    S4 = pulp.LpVariable("S4", lowBound=0, upBound=S4_max)
    prob += S4 == (pulp.lpSum(tprime_b[b] for b in tprime_b) if tprime_b else 0)

    # Symmetric linearization for proportions
    # u_p[a] represents (weight(a -> end) / S3)
    u_p, y_up = {}, {}
    for a in sigma_end_nodes:
        ppa = edge_count(a, end_idx)
        u_p[a] = pulp.LpVariable(f"uprime_{a}", lowBound=0, upBound=1)
        y_up[a] = pulp.LpVariable(f"y_up_{a}", lowBound=0, upBound=S3_max)
        prob += y_up[a] == ppa * z_end[a] # y_up = weight(a -> end) if a in S1
        # Linearization of u_p[a] = y_up[a] / S3
        prob += y_up[a] >= 0
        prob += y_up[a] <= S3
        prob += y_up[a] <= S3_max * u_p[a]
        prob += y_up[a] >= S3 - S3_max * (1 - u_p[a])
        prob += u_p[a] <= z_end[a]

    # v_p[b] represents (tprime_b[b] / S4)
    v_p, y_vp = {}, {}
    for b in tprime_b:
        v_p[b] = pulp.LpVariable(f"vprime_{b}", lowBound=0, upBound=1)
        y_vp[b] = pulp.LpVariable(f"y_vp_{b}", lowBound=0, upBound=S4_max)
        prob += y_vp[b] == tprime_b[b] # y_vp = weight(S1 end nodes -> b)
        # Linearization of v_p[b] = y_vp[b] / S4
        prob += y_vp[b] >= 0
        prob += y_vp[b] <= S4
        prob += y_vp[b] <= S4_max * v_p[b]
        prob += y_vp[b] >= S4 - S4_max * (1 - v_p[b])
        prob += v_p[b] <= z_b_2in[b]

    # r_p[(a,b)] = u_p[a] * v_p[b]
    r_p = {}
    for (a, b) in wprime:
        if a in u_p and b in v_p:
            r_p[(a, b)] = pulp.LpVariable(f"rprime_{a}_{b}", lowBound=0, upBound=1)
            # Standard McCormick relaxation for r_p = u_p * v_p
            prob += r_p[(a, b)] >= u_p[a] + v_p[b] - 1
            prob += r_p[(a, b)] <= u_p[a]
            prob += r_p[(a, b)] <= v_p[b]
            prob += r_p[(a, b)] >= 0

    # m_p[(a,b)] = M * r_p[(a,b)]
    # This is the "expected" flow for this (a, b) pair
    m_p = {}
    for (a, b) in r_p:
        m_p[(a, b)] = pulp.LpVariable(f"mprime_{a}_{b}", lowBound=0, upBound=M_max)
        # --- START: Tighter Approximation ---
        add_mccormick_piecewise(prob, m_p[(a, b)], M, r_p[(a, b)], 0, M_max, 0, 1, n_pieces=4)
        # --- END: Tighter Approximation ---

    # t5 = max(0, sup * expected_flow - actual_flow)
    t5 = {}
    for (a, b) in m_p:
        e_ab = edge_count(a, b) # Actual flow (a -> b)
        t5[(a, b)] = pulp.LpVariable(f"t5_{a}_{b}", lowBound=0)
        prob += t5[(a, b)] >= sup * m_p[(a, b)] - e_ab

    # c5 is the sum of all missing costs
    c5 = pulp.lpSum(t5[(a, b)] for (a, b) in t5) if t5 else 0

    return c1, c2, c3, c4, c5


def loop_cut_ilp(G, sup=1.0, debug=False):

    start_node = 'start'
    end_node = 'end'

    try:
        # Simplify graph by removing nodes not on a path from start to end
        reduced_graph, _, _ = preprocess_graph(G, start_node, end_node)
    except ValueError as e:
        print(f"Error in preprocessing: {e}")
        return [], [], None, [] # Return 4-tuple on failure

    if reduced_graph.number_of_nodes() == 0:
        print("Warning: Reduced graph is empty after preprocessing.")
        return [], [], None, [] # Return 4-tuple on failure

    # Convert graph to matrix representation for easier ILP formulation
    A, W, nodes, node_index = nx_to_mat_and_weights_loop(reduced_graph)
    n = A.shape[0]

    if start_node not in nodes:
        print(f"Warning: Start node '{start_node}' not in reduced graph nodes: {nodes}")
        return [], [], None, [] # Return 4-tuple on failure
    if end_node not in nodes:
        print(f"Warning: End node '{end_node}' not in reduced graph nodes: {nodes}")
        return [], [], None, [] # Return 4-tuple on failure
    
    start_idx = nodes.index(start_node)
    end_idx = nodes.index(end_node)
    # Get all activity nodes (excluding start/end)
    non_terminal = [i for i in range(n) if i not in (start_idx, end_idx)]

    if not non_terminal:
        print("Warning: No non-terminal nodes found.")
        return [], [], None, [] # Return 4-tuple on failure

    # Local helper, assuming W is indexed by integer indices
    def edge_count(i, j):
        try:
            return int(W[i, j])
        except Exception:
            return int(A[i, j])

    # --- Define the ILP Problem ---
    prob = pulp.LpProblem("LOOP_Cut_ILP_Tighter", pulp.LpMinimize)
    
    # x[i] = 1 if node i is in Sigma_2, 0 if node i is in Sigma_1
    x = pulp.LpVariable.dicts("x", non_terminal, cat=pulp.LpBinary)

    # --- Constraints ---
    # Non-trivial partition constraints:
    # Sigma_1 must not be empty (at least one node has x[i]=0)
    prob += pulp.lpSum(1 - x[i] for i in non_terminal) >= 1
    # Sigma_2 must not be empty (at least one node has x[i]=1)
    prob += pulp.lpSum(x[i] for i in non_terminal) >= 1

    # === Compute all cost components ===
    # This function adds all variables and constraints for c1-c5 to 'prob'
    c1, c2, c3, c4, c5 = _compute_partition_parameters(
        prob, x, nodes, node_index, A, W, sup
    )

    # === Final Objective Function ===
    # The total cost is the sum of all deviating (c1,c2,c3) and missing (c4,c5) costs
    prob += c1 + c2 + c3 + c4 + c5

    # --- Solve the problem ---
    solver_options = {'msg': False} # Suppress solver output by default
    if debug:
        solver_options = {'msg': True} # Show solver output
        prob.writeLP("loop_cut_problem.lp") # Save LP file for inspection
        
    prob.solve(pulp.GUROBI_CMD(**solver_options))
    
    # --- Process the solution ---
    if prob.status == pulp.LpStatusOptimal:
        # Extract the two partitions based on the values of x[i]
        Sigma_1 = [nodes[i] for i in non_terminal if pulp.value(x[i]) == 0]
        Sigma_2 = [nodes[i] for i in non_terminal if pulp.value(x[i]) == 1]
        
        # Get the objective value (approximated cost) from the ILP
        total_cost = pulp.value(prob.objective)

        if Sigma_1 or Sigma_2:  # Only compute if we have a valid partition
            # Re-calculate the cost using the "paper's" non-linear formula
            # This is to get the *true* cost of the partition, as the ILP
            # objective is just a (piecewise) linear approximation.
            
            # First, compute the sets required by the cost_loop function
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
    
            # Calculate the true non-linear cost
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
        # Sum up the costs from the dictionary
        total_cost_paper = calculate_total_cost(cost_dict)
        
        # Check if the ILP's approximated cost matches the true cost
        if total_cost_paper != total_cost:
            print(f"Cost approximated: ILP cost={total_cost}, True cost={total_cost_paper}")
        
        return (
            extract_activities(Sigma_1), # Return activity names, not nodes
            extract_activities(Sigma_2), # Return activity names, not nodes
            total_cost_paper,            # Return the true cost
            nodes
        )
    else:
        # Solver failed
        print(f"Warning: Solver failed to find optimal solution. Status: {pulp.LpStatus[prob.status]}")
        return [], [], None, nodes


def compute_partition_parameters(G, Sigma_1, Sigma_2, start_node, end_node, node_index, A):
    """
    Compute the required parameter sets for the `cost_loop` function
    based on the discovered partitions Sigma_1 and Sigma_2.
    """
    start_idx = node_index[start_node]
    end_idx = node_index[end_node]
    
    # Convert node names back to integer indices for matrix lookup
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
        # Check if any predecessor is not in Sigma_2
        for pred_idx in range(len(A)):
            if A[pred_idx, node_idx] > 0: # If there is an edge (pred -> node)
                pred_node = [n for n, idx in node_index.items() if idx == pred_idx][0]
                # If predecessor is in Sigma_1 or is the 'start' node
                if pred_node not in Sigma_2 or pred_node == start_node:
                    input_B.append(node)
                    break # Found one, no need to check other predecessors
    
    # output_B: nodes in Sigma_2 that have outgoing edges to outside Sigma_2
    output_B = []
    for node in Sigma_2:
        node_idx = node_index[node]
        # Check if any successor is not in Sigma_2
        for succ_idx in range(len(A)):
            if A[node_idx, succ_idx] > 0: # If there is an edge (node -> succ)
                succ_node = [n for n, idx in node_index.items() if idx == succ_idx][0]
                # If successor is in Sigma_1 or is the 'end' node
                if succ_node not in Sigma_2 or succ_node == end_node:
                    output_B.append(node)
                    break # Found one, no need to check other successors
    
    # start_activities: all non-terminal nodes that have incoming edges from start
    start_activities = [node for node in G.nodes() 
                        if node not in (start_node, end_node) and A[start_idx, node_index.get(node, -1)] > 0]
    
    # end_activities: all non-terminal nodes that have outgoing edges to end
    end_activities = [node for node in G.nodes() 
                      if node not in (start_node, end_node) and A[node_index.get(node, -1), end_idx] > 0]
    
    return start_A, end_A, input_B, output_B, start_activities, end_activities
    
    

def calculate_total_cost(cost_dict):
    """
    Calculate total cost from the cost dictionary returned by `cost_loop`.
    This dictionary maps edges to their 'missing' and 'deviating' costs.
    
    Args:
        cost_dict: Dictionary with edge tuples as keys and 
                   {'deviating': x, 'missing': y} as values
    
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
    """
    Calculates the cost of a "tau loop" (loop-to-self) cut.
    This is a special case of the loop cut where Sigma_1 contains all
    activities and Sigma_2 is empty (representing 'tau').
    
    The cost is based on the idea that all "loop-back" edges
    (from end_activities to start_activities) are "missing".
    
    Args:
        G (nx.DiGraph): The Directly-Follows Graph.
        sup (float): The support threshold.
        
    Returns:
        tuple: (Sigma_1, Sigma_2, total_cost, nodes)
               Sigma_1 (set): All activities.
               Sigma_2 (set): Empty set.
               total_cost (float): The calculated cost of this "tau loop".
               nodes (list): All nodes in the graph.
    """
    # Get all start activities (nodes connected from start)
    start_activities = [tgt for src, tgt in G.out_edges('start') if tgt != 'end' and tgt in G]
    # Get all end activities (nodes connected to end)
    end_activities = [src for src, tgt in G.in_edges('end') if src != 'start' and src in G]

    # Calculate M: total weight of all loop-back edges
    # (from any end activity to any start activity)
    M = 0
    for a in end_activities:
        for b in start_activities:
            if G.has_edge(a, b):
                M += G[a][b].get('weight', 0)

    # Calculate total weights from start / to end
    total_start_weight = sum(G['start'][b].get('weight', 0) for b in start_activities if G.has_edge('start', b))
    total_end_weight = sum(G[a]['end'].get('weight', 0) for a in end_activities if G.has_edge(a, 'end'))

    # Calculate cost using the non-linear formula from the paper
    total_cost = 0
    for a in end_activities:
        for b in start_activities:
            # Actual weight of the loop-back edge (a -> b)
            weight_a_to_b = G.get_edge_data(a, b, {'weight': 0})['weight']
            
            # Weight from start to 'b'
            weight_start_to_b = G.get_edge_data('start', b, {'weight': 0})['weight']
            # Weight from 'a' to end
            weight_a_to_end = G.get_edge_data(a, 'end', {'weight': 0})['weight']

            # Avoid division by zero if there are no start/end weights
            if total_start_weight > 0 and total_end_weight > 0:
                # Expected flow = M * (proportion of flow into b) * (proportion of flow out of a)
                expected = M * sup * (weight_start_to_b / total_start_weight) * (weight_a_to_end / total_end_weight)
                
                # Cost is max(0, expected - actual)
                cost_component = max(0, expected - weight_a_to_b)
                total_cost += cost_component

    nodes = list(G.nodes())
    # In a tau-loop, Sigma_1 is all activities, Sigma_2 is empty
    Sigma_1 = extract_activities(nodes)  # All regular activities
    Sigma_2 = set()  # Represents tau/empty

    return Sigma_1, Sigma_2, total_cost, nodes
