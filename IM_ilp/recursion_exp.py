import IM_ilp.split_try as split
# from prolysis.discovery.split_functions import split
from collections import Counter
import pm4py
from IM_ilp.Helper_Functions import log_to_graph, _convert_log_to_counter, generate_nx_indirect_graph
from IM_ilp.seq_cut_ilp import seq_cut_ilp_linearized as seq_cut_ilp
from IM_ilp.xor_cut_ilp import xor_cut_ilp, xor_cut_tau
from IM_ilp.par_cut_ilp import par_cut_ilp
from IM_ilp.loop_cut_ilp import loop_cut_ilp, loop_cut_tau


class ProcessTreeNode:

    def __init__(self, operator=None, children=None, activity=None):
        self.operator = operator
        self.children = children if children is not None else []
        self.activity = activity  # Only for leaf nodes

    def to_dict(self):
        """Convert to nested dictionary"""
        if self.activity:
            return {"activity": self.activity}
        else:
            return {
                "operator": self.operator,
                "children": [child.to_dict() for child in self.children]
            }

    def __repr__(self):
        if self.activity:
            return f"'{self.activity}'"
        else:
            children_repr = ", ".join(repr(child) for child in self.children)
            return f"({self.operator}, {children_repr})"


def flatten_process_tree(node):
    """
    Flattens a ProcessTreeNode based on the reduction rules from the
    Inductive Miner paper (Leemans et al., 2013, Property 10).

    Rules applied:
    1. Singleton: Operator(Child) -> Child
    2. Associativity: OP(A, OP(B, C)) -> OP(A, B, C) for SEQ, XOR, PAR
    3. Loop Normal Form:
       - Loop(Loop(A, B), C) -> Loop(A, B, C)
       - Loop(A, XOR(B, C)) -> Loop(A, B, C)
    """
    # 1. Base Case: Leaf node
    if node.activity is not None:
        return node

    # 2. Recursive Step: Flatten children first (Bottom-Up)
    # We rebuild the children list to handle replacements
    flattened_children = []
    for child in node.children:
        flattened_children.append(flatten_process_tree(child))
    node.children = flattened_children

    # 3. Apply Reduction Rules

    # --- Rule 1: Singleton Reduction ---
    # If a node has an operator but only 1 child, replace node with child.
    if len(node.children) == 1:
        return node.children[0]

    # --- Rule 2: Associativity (SEQ, XOR, PAR) ---
    # If parent and child have the same operator (and it's not a loop), flatten.
    if node.operator in ['seq', 'exc', 'par']:
        new_children = []
        for child in node.children:
            if child.activity is None and child.operator == node.operator:
                # Extend with grandchildren (flattening)
                new_children.extend(child.children)
            else:
                new_children.append(child)
        node.children = new_children

    # --- Rule 3: Loop Reduction ---
    # Specific rules for Loop operator defined in the IM paper
    if node.operator == 'loop':
        # The first child is the 'body' (do), the rest are 'redos'
        body = node.children[0]
        redos = node.children[1:]

        # 3a. Nested Loop in Body: Loop(Loop(A, B), C) -> Loop(A, B, C)
        # If the body is itself a loop, merge its redo parts into the parent
        if body.activity is None and body.operator == 'loop':
            grand_body = body.children[0]
            grand_redos = body.children[1:]

            # New body is the grandchild body
            # New redos are grandchild redos + current redos
            node.children = [grand_body] + grand_redos + redos

            # Update local vars for next check
            body = node.children[0]
            redos = node.children[1:]

        # 3b. XOR in Redo: Loop(A, XOR(B, C)) -> Loop(A, B, C)
        # If a redo part is an XOR, flatten it into the list of redo parts
        new_redos = []
        for redo in redos:
            if redo.activity is None and redo.operator == 'exc':
                new_redos.extend(redo.children)
            else:
                new_redos.append(redo)

        node.children = [body] + new_redos

    return node

def _get_all_activities_from_counter(log_counter):
    """Extract all unique activities from a Counter log"""
    all_activities = set()
    for trace in log_counter:
        all_activities.update(trace)
    return all_activities


def extract_activities_from_nodes(nodes):
    """Extract activity names from node list, excluding 'start' and 'end'"""
    return {node for node in nodes if node not in ['start', 'end']}


def get_start_end_activities(G):
    """Get start and end activities from graph"""
    start_activities = {tgt for src, tgt in G.out_edges('start') if tgt != 'end'}
    end_activities = {src for src, tgt in G.in_edges('end') if src != 'start'}
    return start_activities, end_activities


def n_edges(G, source_set, target_set):
    """
    Sum of weights of edges going from any node in source_set to any node in target_set.
    """
    weight = 0
    for u in source_set:
        if u in G:
            for v in target_set:
                if v in G and G.has_edge(u, v):
                    weight += G[u][v].get('weight', 0)
    return weight


def check_loop_tau(G_indirect, start_activities, end_activities):
    """
    Checks if Loop Tau is structurally allowed.
    STRICT MODE:
    1. Must have back-edges from End -> Start.
    2. ALL start activities must have a self-loop (x,x).
    """
    # 1. Check for back-edge flow (End -> Start)
    weight_end_to_start = n_edges(G_indirect, end_activities, start_activities)

    if weight_end_to_start <= 0:
        return False

    # 2. Check strict self-loop condition (ALL must exist)
    for act in start_activities:
        if not G_indirect.has_edge(act, act):
            # If any start activity is missing a self-loop, reject the cut.
            return False

    return True


def calculate_exc_tau_cost_v2(G, sup):
    """
    mis = max(0, sup * sum_out_degree(start) - weight(start->end))
    """
    # Calculate sum_out_degree_single(start)
    # In V1 this is the total weight leaving the start node
    total_start_out_weight = sum(data['weight'] for _, _, data in G.out_edges('start', data=True))

    # Calculate get_edge_weight(start, end)
    start_to_end_weight = 0
    if G.has_edge('start', 'end'):
        start_to_end_weight = G['start']['end'].get('weight', 0)

    # Formula: mis = max(0, sup * start_sum - start_end_weight)
    mis = max(0, sup * total_start_out_weight - start_to_end_weight)

    return mis


def calculate_loop_tau_cost_v2(G, sup, start_acts, end_acts):
    # M_P: Total weight of edges going from End Activities -> Start Activities
    M_P = n_edges(G, end_acts, start_acts)

    # start_sum: Total weight from Global Start -> Start Activities
    start_sum = n_edges(G, {'start'}, start_acts)

    # end_sum: Total weight from End Activities -> Global End
    end_sum = n_edges(G, end_acts, {'end'})

    mis_cost = 0.0

    # Avoid division by zero
    if start_sum == 0 or end_sum == 0:
        return float('inf')

        # Iterate over all pairs (a, b) where a in Start_Acts and b in End_Acts
    for a in start_acts:
        for b in end_acts:
            # Weight from Start -> a
            w_start_a = 0
            if G.has_edge('start', a):
                w_start_a = G['start'][a].get('weight', 0)

            # Weight from b -> End
            w_b_end = 0
            if G.has_edge(b, 'end'):
                w_b_end = G[b]['end'].get('weight', 0)

            # Actual weight of back-edge b -> a
            w_b_a = 0
            if G.has_edge(b, a):
                w_b_a = G[b][a].get('weight', 0)

            # Expected flow calculation from V1 logic
            if w_b_a > 0:
                # If edge exists, V1 logic sums deviations
                expected_flow = M_P * sup * (w_start_a / start_sum) * (w_b_end / end_sum)
                mis_cost += max(0, expected_flow - w_b_a)
            else:
                # If edge missing, pure penalty
                expected_flow = M_P * sup * (w_start_a / start_sum) * (w_b_end / end_sum)
                mis_cost += expected_flow

    return mis_cost


def recursion_full(log, depth=0, max_depth=20, sup=1.0, debug=False, tau_cost_threshold=0):
    """
    Recursively discovers a process tree from an event log.
    """

    if debug:
        print(f"\n=== Recursion Depth {depth} ===")

    if not isinstance(log, Counter):
        log_counter = _convert_log_to_counter(log)
    else:
        log_counter = log

    if debug:
        print(f"Log has {sum(log_counter.values())} total trace occurrences")
        print(f"Log has {len(log_counter)} unique traces")

    if not log_counter:
        if debug:
            print("Empty log detected, returning tau")
        return ProcessTreeNode(activity=None)

    all_activities = _get_all_activities_from_counter(log_counter)
    if debug:
        print(f"Unique activities: {all_activities}")

    # BASE CASE: If event log has only empty traces
    if len(all_activities) == 0:
        if debug:
            print("Base case: Only empty traces detected, returning tau")
        return ProcessTreeNode(activity=None)

    # Create graph from log
    try:
        G = log_to_graph(log_counter)
    except Exception as e:
        if debug:
            print(f"Graph creation failed: {e}")
        raise e

    # Get start and end activities (Required for V1 logic checks)
    start_activities_set, end_activities_set = get_start_end_activities(G)

    # exc_tau: Checks if 'start' -> 'end' edge exists
    can_have_xor_tau = (
            G.has_edge('start', 'end') #and G['start']['end'].get('weight', 0) > 0
    )

    # loop_tau: Checks for back-edges and strict self-loops on start nodes
    G_indirect = generate_nx_indirect_graph(log_counter)
    can_have_loop_tau = check_loop_tau(G_indirect, start_activities_set, end_activities_set)

    # SPECIAL CASE: Single activity
    if len(all_activities) == 1:
        activity = next(iter(all_activities))
        if debug:
            print(log_counter)
            print('xor:', can_have_xor_tau, 'loop:', can_have_loop_tau)
        if debug:
            print(f"Single activity '{activity}' detected, checking for tau cuts...")
            print('xor_tau_allowed?', can_have_xor_tau, 'loop_tau_allowed', can_have_loop_tau)
        if can_have_xor_tau and can_have_loop_tau:
            try:
                # Calculate cost using V1 formula
                start_to_act_end = n_edges(G, {'start'}, {activity, 'end'})
                start_to_end = n_edges(G, {'start'}, {'end'})

                total_costx = calculate_exc_tau_cost_v2(G, sup)

                in_to_act = n_edges(G, {'start', activity}, {activity})
                act_self_loop = n_edges(G, {activity}, {activity})

                total_costl = max(0, sup * (in_to_act / 2) - act_self_loop)

                if debug:
                    print(f"both tau cost for single activity: {total_costx, total_costl}")

                if total_costx <= 0 and total_costl <= 0:
                    if debug:
                        print(f"Using exc_tau (optional activity) for single activity '{activity}'")

                    # τ ⊕ A (either do nothing OR do A)
                    child1 = ProcessTreeNode(activity=None)  # tau
                    child2 = ProcessTreeNode(activity=activity)  # activity
                    child3 = ProcessTreeNode(operator='loop', children=[child2, child1])
                    return ProcessTreeNode(operator='exc', children=[child3, child1])

            except Exception as e:
                if debug:
                    print(f"exc_tau calculation failed: {e}")
        # Check for exc_tau (optional activity): τ ⊕ A
        if can_have_xor_tau:
            try:
                # Calculate cost using V1 formula
                start_to_act_end = n_edges(G, {'start'}, {activity, 'end'})
                start_to_end = n_edges(G, {'start'}, {'end'})

                total_cost = calculate_exc_tau_cost_v2(G, sup)

                if debug:
                    print(f"exc_tau cost for single activity: {total_cost}")

                if total_cost <= 0:
                    if debug:
                        print(f"Using exc_tau (optional activity) for single activity '{activity}'")

                    # τ ⊕ A (either do nothing OR do A)
                    child1 = ProcessTreeNode(activity=None)  # tau
                    child2 = ProcessTreeNode(activity=activity)  # activity
                    return ProcessTreeNode(operator='exc', children=[child1, child2])

            except Exception as e:
                if debug:
                    print(f"exc_tau calculation failed: {e}")

        # Check for loop_tau (repeating activity): LOOP(A, τ)
        if can_have_loop_tau:
            try:
                # print('calculating loop_tau cost')
                in_to_act = n_edges(G, {'start', activity}, {activity})
                act_self_loop = n_edges(G, {activity}, {activity})

                total_cost = max(0, sup * (in_to_act / 2) - act_self_loop)

                # print('no worries')
                if debug:
                    print(f"loop_tau cost for single activity: {total_cost}")

                if total_cost <= tau_cost_threshold:
                    if debug:
                        print(f"Using loop_tau (repeating activity) for single activity '{activity}'")

                    # LOOP(A, τ) - do A at least once, then optionally repeat
                    child1 = ProcessTreeNode(activity=activity)  # do part (at least once)
                    child2 = ProcessTreeNode(activity=None)  # redo part (τ, optional)
                    return ProcessTreeNode(operator='loop', children=[child1, child2])

            except Exception as e:
                if debug:
                    print(f"loop_tau calculation failed: {e}")

        # If no tau cuts apply, return leaf node
        if debug:
            print(f"No tau cuts apply, returning leaf node for '{activity}'")
        return ProcessTreeNode(activity=activity)

    # MULTIPLE ACTIVITIES: Proceed with regular cut detection
    if debug:
        print("Multiple activities detected, proceeding with cut detection")

    cut_results = {

        'exc_tau': {'Sigma_1': None, 'Sigma_2': None, 'cost': float('inf'), 'success': False},
        'seq': {'Sigma_1': None, 'Sigma_2': None, 'cost': float('inf'), 'success': False},
        'exc': {'Sigma_1': None, 'Sigma_2': None, 'cost': float('inf'), 'success': False},
        'par': {'Sigma_1': None, 'Sigma_2': None, 'cost': float('inf'), 'success': False},
        'loop': {'Sigma_1': None, 'Sigma_2': None, 'cost': float('inf'), 'success': False},
        'loop_tau': {'Sigma_1': None, 'Sigma_2': None, 'cost': float('inf'), 'success': False},
    }


    if can_have_xor_tau:
        try:
            # print('calculating total_cost xor_tau:')
            total_cost = calculate_exc_tau_cost_v2(G, sup)
            # print('total_cost xor_tau:', total_cost)
            # V1 logic: If structure matches, the split is "All Acts" vs "Empty"
            Sigma_1 = all_activities
            Sigma_2 = None

            if debug:
                print(f"exc_tau result: cost={total_cost}, Sigma_1={Sigma_1}, Sigma_2={Sigma_2}")

            cut_results['exc_tau'] = {
                'Sigma_1': Sigma_1,  # All activities
                'Sigma_2': Sigma_2,  # Empty set (τ)
                'cost': total_cost,
                'success': True
            }
        except Exception as e:
            if debug:
                print(f"exc_tau calculation failed: {e}")

    if can_have_loop_tau:
        try:
            # print('calculating total_cost loop_tau:')
            total_cost = calculate_loop_tau_cost_v2(G, sup, start_activities_set, end_activities_set)
            # print('total_cost loop_tau:', total_cost)
            # V1 logic: If structure matches, the split is "All Acts" vs "Empty"
            Sigma_1 = all_activities
            Sigma_2 = None

            if debug:
                print(f"loop_tau result: cost={total_cost}, Sigma_1={Sigma_1}, Sigma_2={Sigma_2}")

            cut_results['loop_tau'] = {
                'Sigma_1': Sigma_1,  # All activities (do part - at least once)
                'Sigma_2': Sigma_2,  # Empty set (τ) - redo part
                'cost': total_cost,
                'success': True
            }
        except Exception as e:
            if debug:
                print(f"loop_tau calculation failed: {e}")

    # Use ILP functions for non-tau cuts (Original V2 Logic)
    cut_functions = [
        ('exc', xor_cut_ilp),
        ('seq', seq_cut_ilp),
        ('par', par_cut_ilp),
        ('loop', loop_cut_ilp),
    ]

    found_perfect_cut = False

    for op_name, cut_func in cut_functions:
        if found_perfect_cut:
            continue

        try:
            if debug:
                print(f"Trying {op_name} cut...")

            if op_name == 'seq':
                Sigma_1, Sigma_2, total_cost, nodes = cut_func(G, log_counter, sup)
            elif op_name == 'loop':
                Sigma_1, Sigma_2, total_cost, nodes = cut_func(G, sup)
            else:
                Sigma_1, Sigma_2, total_cost, nodes = cut_func(G, sup)

            if debug:
                print(f" {op_name} cut result: Sigma_1={Sigma_1}, Sigma_2={Sigma_2}, cost={total_cost}")

            # For non-tau cuts, both Sigma_1 and Sigma_2 must be non-empty
            valid_binary_cut = (Sigma_1 and Sigma_2)

            if (total_cost is not None and total_cost < float('inf')
                    and valid_binary_cut):

                cut_results[op_name] = {
                    'Sigma_1': Sigma_1, 'Sigma_2': Sigma_2,
                    'cost': total_cost, 'success': True
                }

                if total_cost < 1e-7:
                    found_perfect_cut = True
            else:
                if debug:
                    print(f" ✗ {op_name} cut invalid")

        except Exception as e:
            if debug:
                print(f" ✗ {op_name} cut failed: {e}")

    # 1. Gather all successful cuts
    candidates = []
    for op, res in cut_results.items():
        if res['success']:
            candidates.append((op, res['cost']))

    # 2. Sort by cost (Ascending)
    candidates.sort(key=lambda x: x[1])

    if not candidates:
        # No valid cut found - this happens when the model is a flower model
        if debug:
            print(f"No valid cut found, returning flower model for activities: {all_activities}")

        # Create an exclusive choice over all activities
        children = []
        for activity in all_activities:
            children.append(ProcessTreeNode(activity=activity))

        return ProcessTreeNode(operator='exc', children=children)

    # 3. Iterate through candidates until one successfully splits the log
    for current_op, cost in candidates:
        if debug:
            print(f"Splitting log with {current_op} (cost={cost})...")

        try:
            best_Sigma_1 = cut_results[current_op]['Sigma_1']
            best_Sigma_2 = cut_results[current_op]['Sigma_2']

            if current_op in ['loop_tau', 'exc_tau']:
                # For tau cuts, use start and end activities for splitting
                start_activities_set, end_activities_set = get_start_end_activities(G)

                start_activities_list = list(start_activities_set)
                end_activities_list = list(end_activities_set)
                if current_op == 'loop_tau':
                    log1_counter, log2_counter = split.split(current_op, [start_activities_list, end_activities_list],
                                                         log_counter)
                else:
                    log1_counter, log2_counter = split.split(current_op, [all_activities, []],
                                                         log_counter)
            else:
                log1_counter, log2_counter = split.split(current_op, [best_Sigma_1, best_Sigma_2], log_counter)

            # --- FAIL SAFE CHECK ---
            # Check for infinite recursion
            if len(log1_counter) == 0 and len(log2_counter) == 0:
                if debug:
                    print(f"✗ Split failed for {current_op}: Both logs empty.")
                continue

            if len(log1_counter) == len(log_counter) and set(log1_counter.keys()) == set(log_counter.keys()):
                if debug:
                    print(f"✗ Split failed for {current_op}: Infinite recursion (log1 unchanged).")
                continue

            if len(log2_counter) == len(log_counter) and set(log2_counter.keys()) == set(log_counter.keys()):
                if debug:
                    print(f"✗ Split failed for {current_op}: Infinite recursion (log2 unchanged).")
                continue

            # For binary cuts, both logs must be non-empty
            if current_op not in ['exc_tau', 'loop_tau']:
                if not log2_counter:
                    if debug:
                        print(f"✗ Split failed for {current_op}: Binary split resulted in empty second log.")
                    continue
                elif not log1_counter:
                    if debug:
                        print(f"✗ Split failed for {current_op}: Binary split resulted in empty first log.")
                    continue

            # --- RECURSE ---
            # Handle tau cuts
            if current_op in ['exc_tau', 'loop_tau']:

                # IMPORTANT: For tau cuts, one log should be empty (τ) and the other should contain activities

                if not log1_counter and log2_counter:
                    # log1 is empty (τ), log2 has activities
                    if current_op == 'exc_tau':
                        # τ ⊕ acts: left child is τ, right child is acts
                        child1 = ProcessTreeNode(activity=None)
                        child2 = recursion_full(log2_counter, depth + 1, max_depth, sup, debug=debug)
                    else:  # loop_tau
                        # LOOP(acts, τ): left child is acts, right child is τ
                        child1 = recursion_full(log2_counter, depth + 1, max_depth, sup, debug=debug)
                        child2 = ProcessTreeNode(activity=None)
                elif log1_counter and not log2_counter:
                    # log1 has activities, log2 is empty (τ)
                    if current_op == 'exc_tau':
                        # τ ⊕ acts: left child is τ, right child is acts
                        child1 = ProcessTreeNode(activity=None)
                        child2 = recursion_full(log1_counter, depth + 1, max_depth, sup, debug=debug)
                    else:  # loop_tau
                        # LOOP(acts, τ): left child is acts, right child is τ
                        child1 = recursion_full(log1_counter, depth + 1, max_depth, sup, debug=debug)
                        child2 = ProcessTreeNode(activity=None)
                elif log1_counter and log2_counter:
                    # Both have activities (unusual for tau cuts)
                    child1 = recursion_full(log1_counter, depth + 1, max_depth, sup, debug=debug)
                    child2 = recursion_full(log2_counter, depth + 1, max_depth, sup, debug=debug)
                else:
                    # Both empty
                    child1 = ProcessTreeNode(activity=None)
                    child2 = ProcessTreeNode(activity=None)

                # Remove '_tau' suffix for operator
                standard_op = current_op.replace('_tau', '')
                return ProcessTreeNode(operator=standard_op, children=[child1, child2])
            else:
                # For non-tau cuts, both logs should have activities
                child1 = recursion_full(log1_counter, depth + 1, max_depth, sup, debug=debug)
                child2 = recursion_full(log2_counter, depth + 1, max_depth, sup, debug=debug)
                return ProcessTreeNode(operator=current_op, children=[child1, child2])

        except Exception as e:
            if debug:
                print(f"Split failed for operator {current_op}: {e}")
            continue  # Try next best candidate

    if debug:
        print(f"All {len(candidates)} valid cuts failed to partition the log. Returning flower model.")


    return print("something horrible happened")


def to_pm4py_tree(node, parent=None):
    """Converts the internal ProcessTreeNode to a PM4Py ProcessTree object"""
    tree = ProcessTree()
    tree.parent = parent

    if node.activity is not None:
        tree.label = node.activity
    else:
        op_map = {
            'seq': Operator.SEQUENCE,
            'exc': Operator.XOR,
            'par': Operator.PARALLEL,
            'loop': Operator.LOOP
        }
        tree.operator = op_map.get(node.operator, None)
        for child in node.children:
            child_tree = to_pm4py_tree(child, parent=tree)
            tree.children.append(child_tree)
    return tree


def apply(log_path, sup=1.0, print_time_taken=False):
    log = xes_importer.apply(str(log_path))
    log_counter = _convert_log_to_counter(log)
    s = time.time()
    process_tree_exp_ilp_counter = recursion_full_improved(filtered_log_counter, sup=sup)
    tree_data_pm4py = to_pm4py_tree(process_tree_exp_ilp_counter)
    net, im, fm = pm4py.objects.conversion.process_tree.converter.apply(tree_data_pm4py)
    e = time.time()
    if print_time_taken:
        print(e-s)
    return net, im, fm


