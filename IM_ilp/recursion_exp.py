import IM_ilp.split as split
from collections import Counter
from IM_ilp.Helper_Functions import log_to_graph, _convert_log_to_counter, cost_eventual
from IM_ilp.seq_cut_ilp import seq_cut_ilp
from IM_ilp.xor_cut_ilp import xor_cut_ilp, xor_cut_tau
from IM_ilp.par_cut_ilp import par_cut_ilp
from IM_ilp.loop_cut_ilp import loop_cut_ilp, loop_cut_tau
from pm4py.objects.process_tree.obj import ProcessTree
from pm4py.objects.process_tree.utils import generic
from pm4py.objects.process_tree.obj import Operator


class ProcessTreeNode:
    """Class to represent a process tree node"""
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


def _create_exclusive_choice_over_traces_simple(log_counter):
    """Simple fallback that prevents duplicate activities by using exclusive choice"""
    children = []
    for trace in log_counter:
        if len(trace) == 0:
            children.append(ProcessTreeNode(activity=None))
        else:
            unique_activities = set(trace)
            if len(unique_activities) == 1:
                children.append(ProcessTreeNode(activity=next(iter(unique_activities))))
            else:
                # Fallback: just take the first activity to represent the trace
                children.append(ProcessTreeNode(activity=next(iter(unique_activities))))

    # Remove duplicates in children
    unique_children = {}
    for child in children:
        key = child.activity
        if key not in unique_children:
            unique_children[key] = child

    children_list = list(unique_children.values())

    if len(children_list) == 1:
        return children_list[0]
    else:
        return ProcessTreeNode(operator='exc', children=children_list)

# Add w_forward_cache=None to the function signature
def recursion_full(log, w_forward_cache=None, depth=0, max_depth=20, sup=1.0, debug=False, parent_was_tau=False):
    """
    Recursively discovers a process tree from an event log.

    :param log: Event log (Counter, list of tuples, or PM4Py EventLog)
    :param w_forward_cache: A pre-calculated eventual-follows cost map (for 'seq' cut)
    :param depth: Current recursion depth
    :param max_depth: Maximum recursion depth
    :param sup: Support threshold
    :param debug: Print debug information
    :param parent_was_tau: Flag to indicate if the parent call used a tau cut
    :return: A ProcessTreeNode
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
        if log_counter:
            sample_traces = list(log_counter.keys())[:3]
            print(f"Sample traces: {sample_traces}")

    if not log_counter:
        if debug:
            print("Empty log detected, returning tau")
        return ProcessTreeNode(activity=None)

    all_activities = _get_all_activities_from_counter(log_counter)
    if debug:
        print(f"Unique activities: {all_activities}")

    # IMbi BASE CASE 1: If event log has only empty traces (|Σ| = 0)
    if len(all_activities) == 0:
        if debug:
            print("Base case: Only empty traces detected, returning tau")
        return ProcessTreeNode(activity=None)

    # IMbi BASE CASE 2: If all traces have only one activity (|Σ| ≤ 1)
    if len(all_activities) == 1:
        activity = next(iter(all_activities))
        if debug:
            print(f"Base case: Single activity '{activity}' detected")

        total_traces = sum(log_counter.values())
        empty_traces = log_counter.get((), 0)

        if debug:
            print(f"Total traces: {total_traces}, Empty traces: {empty_traces}")

        c_plus = max(0, total_traces * sup - empty_traces)

        if debug:
            print(f"τ-xor check: c+ = max(0, {total_traces} * {sup} - {empty_traces}) = {c_plus}")

        if c_plus <= 0 and empty_traces > 0:
            if debug:
                print("τ-xor condition satisfied, creating τ-xor structure")

            non_empty_log = Counter()
            for trace, count in log_counter.items():
                if len(trace) > 0:
                    non_empty_log[trace] = count

            if non_empty_log:
                if debug:
                    print(f"Recursing on non-empty log with {sum(non_empty_log.values())} traces")
                
                # Pass parent_was_tau=True because this is a tau-based structure
                # Pass the cache down
                child = recursion_full(non_empty_log, w_forward_cache, depth + 1, max_depth, sup, debug=debug, parent_was_tau=True)
                return ProcessTreeNode(operator='exc', children=[
                    ProcessTreeNode(activity=None),
                    child
                ])
            else:
                return ProcessTreeNode(activity=None)

        total_events = 0
        self_loop_count = 0

        for trace in log_counter:
            trace_count = log_counter[trace]
            total_events += len(trace) * trace_count

            for i in range(len(trace) - 1):
                if trace[i] == trace[i + 1] == activity:
                    self_loop_count += trace_count

        events_to_traces_ratio = total_events / total_traces if total_traces > 0 else 0

        if debug:
            print(f"τ-loop check: events/traces ratio = {events_to_traces_ratio:.2f}")
            print(f"Self-loop count: {self_loop_count}")

        p_plus = max(0, sup * (total_events / 2) - self_loop_count)

        if debug:
            print(f"p+ = max(0, {sup} * ({total_events} / 2) - {self_loop_count}) = {p_plus}")

        if p_plus <= 0 and self_loop_count > 0 and events_to_traces_ratio > 1.1:
            if debug:
                print("τ-loop condition satisfied, creating loop structure")
            return ProcessTreeNode(operator='loop', children=[
                ProcessTreeNode(activity=activity),
                ProcessTreeNode(activity=None)
            ])

        if debug:
            print("Returning single activity leaf node")
        return ProcessTreeNode(activity=activity)

    if debug:
        print("Multiple activities detected, proceeding with cut detection")

    try:
        G = log_to_graph(log_counter)
        if debug:
            print(f"Graph created with {len(G.nodes())} nodes and {len(G.edges())} edges")
    except Exception as e:
        if debug:
            print(f"Graph creation failed: {e}")
        raise e

    # --- Caching logic for seq_cut (W_forward_cache) ---
    # If a cache is passed, use it.
    # If not (e.g., top-level call), calculate it once for this subgraph.
    # This cache will then be passed down to all children.
    if w_forward_cache is None:
        if debug:
            print("Calculating W_forward_cache (eventual-follows) for this subgraph...")
        # Note: This passes 'log', which is the original log object,
        # not 'log_counter'. cost_eventual is built to handle this.
        w_forward_cache = cost_eventual(G, log_counter, sup) 
        if debug:
            print("... W_forward_cache calculation complete.")
    # --- End Caching Logic ---

    cut_results = {
        'exc_tau': {'Sigma_1': None, 'Sigma_2': None, 'cost': float('inf'), 'n': 0, 'success': False},
        'loop_tau': {'Sigma_1': None, 'Sigma_2': None, 'cost': float('inf'), 'n': 0, 'success': False},
        'exc': {'Sigma_1': None, 'Sigma_2': None, 'cost': float('inf'), 'n': 0, 'success': False},
        'seq': {'Sigma_1': None, 'Sigma_2': None, 'cost': float('inf'), 'n': 0, 'success': False},
        'par': {'Sigma_1': None, 'Sigma_2': None, 'cost': float('inf'), 'n': 0, 'success': False},
        'loop': {'Sigma_1': None, 'Sigma_2': None, 'cost': float('inf'), 'n': 0, 'success': False}
    }

    # Define all possible cut functions
    all_cut_functions = [
        ('seq', seq_cut_ilp),
        ('exc', xor_cut_ilp),
        ('par', par_cut_ilp),
        ('loop', loop_cut_ilp),
        ('exc_tau', xor_cut_tau),
        ('loop_tau', loop_cut_tau)
    ]

    # Filter out tau cuts if the parent was a tau cut
    if parent_was_tau:
        if debug:
            print("🔒 Parent was a tau cut, filtering to non-tau cuts only.")
        cut_functions = [(op, func) for op, func in all_cut_functions if '_tau' not in op]
    else:
        cut_functions = all_cut_functions

    # Track best and second-best cuts
    best_op = None
    best_cost = float('inf')
    second_best_op = None
    second_best_cost = float('inf')
    
    found_perfect_cut = False

    for op_name, cut_func in cut_functions:
        if found_perfect_cut:
            if debug:
                print(f"  Skipping {op_name} cut - already found perfect solution")
            continue

        try:
            if debug:
                print(f"Trying {op_name} cut...")

            if op_name == 'seq':
                # Pass the w_forward_cache to seq_cut_ilp
                Sigma_1, Sigma_2, total_cost, nodes = cut_func(G, log, sup, W_forward_cache=w_forward_cache)
            elif op_name == 'loop':
                # Pass the w_forward_cache to seq_cut_ilp
                Sigma_1, Sigma_2, total_cost, nodes = cut_func(G, sup, debug)
            else:
                Sigma_1, Sigma_2, total_cost, nodes = cut_func(G, sup)

            if debug:
                print(f"  {op_name} cut result: Sigma_1={Sigma_1}, Sigma_2={Sigma_2}, cost={total_cost}")

            is_tau_cut = op_name in ['exc_tau', 'loop_tau']
            valid_binary_cut = (not is_tau_cut) and (Sigma_1 and Sigma_2)
            
            valid_tau_cut = False
            if is_tau_cut and Sigma_1:
                valid_tau_cut = True
                if Sigma_2 is None:
                    Sigma_2 = set()

            if (total_cost is not None and total_cost < float('inf')
                    and (valid_binary_cut or valid_tau_cut)):

                if Sigma_2 is None:
                    Sigma_2 = set()

                cut_results[op_name] = {
                    'Sigma_1': Sigma_1, 'Sigma_2': Sigma_2,
                    'cost': total_cost, 'n': nodes, 'success': True
                }

                # Logic to find best and second-best
                if total_cost < best_cost:
                    # Demote old best to second-best
                    second_best_cost = best_cost
                    second_best_op = best_op
                    # Promote new to best
                    best_cost = total_cost
                    best_op = op_name
                    
                    if debug:
                        print(f"  ✓ {op_name} cut successful with cost {total_cost} (new best)")
                        
                elif total_cost < second_best_cost:
                    # Set new second-best
                    second_best_cost = total_cost
                    second_best_op = op_name
                    if debug:
                        print(f"  ✓ {op_name} cut successful with cost {total_cost} (new second-best)")
                else:
                    if debug:
                         print(f"  ✓ {op_name} cut successful with cost {total_cost}")

                if total_cost < 1e-7:
                    if debug:
                        print(f"  🎯 PERFECT {op_name.upper()} CUT FOUND! Stopping cut evaluation.")
                    found_perfect_cut = True

            else:
                if debug:
                    print(f"  ✗ {op_name} cut invalid - empty sets or infinite cost")

        except Exception as e:
            if debug:
                print(f"  ✗ {op_name} cut failed: {e}")

    if debug:
        print(f"Best cut: {best_op}, Second best: {second_best_op}")


    current_op = best_op
    
    # We will try the best_op. If it's a tau cut and fails,
    # this loop will set current_op = second_best_op and try again once.
    while current_op is not None:
        try:
            # 1. Get Sigmas for the current_op
            if current_op not in cut_results or not cut_results[current_op]['success']:
                 raise KeyError(f"No valid cut data found for {current_op}.")
                 
            best_Sigma_1 = cut_results[current_op]['Sigma_1']
            best_Sigma_2 = cut_results[current_op]['Sigma_2']

            if debug:
                print(f"Splitting log with {current_op} cut...")
                print(f"  Log: {len(log_counter)} unique traces, {sum(log_counter.values())} total occurrences")
                print(f"  Sigma_1: {best_Sigma_1}")
                print(f"  Sigma_2: {best_Sigma_2}")
            
            # 2. Perform the split
            if current_op in ['loop_tau', 'exc_tau']:
                start_activities = [tgt for src, tgt in G.out_edges('start') if tgt != 'end']
                end_activities = [src for src, tgt in G.in_edges('end') if src != 'start']
                if debug:
                    print(f"  (Tau cut) Start activities: {start_activities}")
                    print(f"  (Tau cut) End activities: {end_activities}")
                log1_counter, log2_counter = split.split(current_op, [start_activities, end_activities], log_counter)
            else:
                log1_counter, log2_counter = split.split(current_op, [best_Sigma_1, best_Sigma_2], log_counter)
            # After the split operation, add this:
            if debug:
                print("DEBUG - Checking split results:")
                print(f"Original activities: {all_activities}")
                print(f"Sigma_1: {best_Sigma_1}")
                print(f"Sigma_2: {best_Sigma_2}")
                print(f"Log1 activities: {_get_all_activities_from_counter(log1_counter)}")
                print(f"Log2 activities: {_get_all_activities_from_counter(log2_counter)}")
                print(f"Log1 sample traces: {list(log1_counter.keys())[:3]}")
                print(f"Log2 sample traces: {list(log2_counter.keys())[:3]}")
            # 3. Check for a failed tau split (no change in log)
            is_tau_cut = current_op in ['loop_tau', 'exc_tau']
            # Split fails if log2 is empty (no split) or log1 is unchanged (failing split)
            split_failed = is_tau_cut and (not log2_counter or log1_counter == log_counter)

            if split_failed:
                if debug:
                    print(f"✗ Split failed for {current_op} (no actual change in log).")
                
                if second_best_op is not None:
                    if debug:
                        print(f"  ... Switching to second best cut: {second_best_op}")
                    # Promote second_op to current_op and try again
                    current_op = second_best_op
                    second_best_op = None # Don't try to switch again
                    continue # Restart the loop with the new op
                else:
                    if debug:
                        print(f"  ... No second best cut available. Halting.")
                    # No second option, we must fail.
                    raise Exception(f"Tau split failed for {current_op} and no second best cut was found.")

            # 4. If we are here, the split was successful
            if debug:
                print(f"✓ Split successful with {current_op}")
                print(f"  Log1: {len(log1_counter)} unique traces, {sum(log1_counter.values())} total occurrences")
                print(f"  Log2: {len(log2_counter)} unique traces, {sum(log2_counter.values())} total occurrences")
            
            # Determine if the current op is a tau cut
            current_op_is_tau = current_op in ['exc_tau', 'loop_tau']
            if debug and current_op_is_tau:
                print(f"🔒 Propagating parent_was_tau=True to children, as {current_op} is a tau cut.")

            # Pass the cache and the new flag to the recursive calls
            child1 = recursion_full(log1_counter, w_forward_cache, depth + 1, max_depth, sup, debug=debug, parent_was_tau=current_op_is_tau)
            child2 = recursion_full(log2_counter, w_forward_cache, depth + 1, max_depth, sup, debug=debug, parent_was_tau=current_op_is_tau)

            # Map tau cuts to standard operators
            if current_op in ['exc_tau', 'loop_tau']:
                standard_op = current_op.replace('_tau', '')
                if debug:
                    print(f"Converting tau cut {current_op} to standard operator {standard_op}")
                return ProcessTreeNode(operator=standard_op, children=[child1, child2])
            else:
                return ProcessTreeNode(operator=current_op, children=[child1, child2])

        except Exception as e:
            if debug:
                print(f"Split failed for operator {current_op}: {e}")
                import traceback
                traceback.print_exc()
            raise e # Re-raise the exception

    # If we exit the loop (e.g., best_op was None to begin with), raise the original error
    raise Exception(f"No valid cut found for subgraph with activities: {all_activities}. Halting.")
    

def _get_all_activities_from_counter(log_counter):
    """Extract all unique activities from a Counter log"""
    all_activities = set()
    for trace in log_counter:
        all_activities.update(trace)
    return all_activities


def _get_all_activities(log):
    """Helper to extract all unique activities from log (handles multiple formats)"""
    if isinstance(log, Counter):
        return _get_all_activities_from_counter(log)
    elif hasattr(log, '__class__') and 'EventLog' in str(log.__class__):
        all_activities = set()
        for trace in log:
            all_activities.update(event['concept:name'] for event in trace)
        return all_activities
    else:
        all_activities = set()
        for trace in log:
            if trace:
                if isinstance(trace[0], dict):
                    all_activities.update(ev['concept:name'] for ev in trace)
                else:
                    all_activities.update(trace)
        return all_activities


def _extract_activity_name(activity):
    """Extract just the activity name from an event object"""
    if isinstance(activity, dict) and 'concept:name' in activity:
        return activity['concept:name']
    elif isinstance(activity, str):
        return activity
    else:
        return str(activity)


def _create_exclusive_choice_node(activities):
    """Helper function to create an exclusive choice node from a set of activities"""
    activities_list = list(activities)
    if len(activities_list) == 1:
        activity_name = _extract_activity_name(activities_list[0])
        return ProcessTreeNode(activity=activity_name)
    else:
        children = [ProcessTreeNode(activity=_extract_activity_name(act)) for act in activities_list]
        return ProcessTreeNode(operator='exc', children=children)


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