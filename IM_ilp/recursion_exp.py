import time
import pandas as pd
from collections import Counter
from IM_ilp.Helper_Functions import log_to_graph, _convert_log_to_counter
import local_pm4py.split_functions.split as split
from IM_ilp.seq_cut_ilp import seq_cut_ilp
from IM_ilp.xor_cut_ilp import xor_cut_ilp, xor_cut_tau
from IM_ilp.par_cut_ilp import par_cut_ilp
from IM_ilp.loop_cut_ilp_test import loop_cut_ilp, loop_cut_tau


class ProcessTreeNode:
    """Class to represent a process tree node"""
    def __init__(self, operator=None, children=None, activity=None):
        self.operator = operator  # 'seq', 'exc', 'par', 'loop', or None for leaf
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
            children.append(ProcessTreeNode(activity="tau"))
        else:
            # For any non-empty trace, create a single activity node
            # This prevents sequences with duplicate activities
            unique_activities = set(trace)
            if len(unique_activities) == 1:
                # All activities are the same - use single activity
                children.append(ProcessTreeNode(activity=next(iter(unique_activities))))
            else:
                # Mixed activities - use the first unique activity as representative
                # This is a simplification to avoid duplicates
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


def recursion_full(log, timing_data=None, depth=0, max_depth=20, sup=1.0):
    if timing_data is None:
        timing_data = []

    if depth > max_depth:
        print(f"Max depth reached at depth {depth}, returning fallback")
        all_activities = _get_all_activities(log)
        return _create_exclusive_choice_node(all_activities)

    print(f"\n=== Recursion Depth {depth} ===")

    # Convert log to standard Counter format
    log_counter = _convert_log_to_counter(log)

    print(f"Log has {sum(log_counter.values())} total trace occurrences")
    print(f"Log has {len(log_counter)} unique traces")

    # Print sample traces
    if log_counter:
        sample_traces = list(log_counter.keys())[:3]
        print(f"Sample traces: {sample_traces}")

    # Handle empty log
    if not log_counter:
        print("Empty log detected, returning tau")
        return ProcessTreeNode(activity="tau")

    # Get unique activities
    all_activities = _get_all_activities_from_counter(log_counter)
    print(f"Unique activities: {all_activities}")

    # IMbi BASE CASE 1: If event log has only empty traces (|Σ| = 0)
    if len(all_activities) == 0:
        print("Base case: Only empty traces detected, returning tau")
        return ProcessTreeNode(activity="tau")

    # IMbi BASE CASE 2: If all traces have only one activity (|Σ| ≤ 1)
    if len(all_activities) == 1:
        activity = next(iter(all_activities))
        print(f"Base case: Single activity '{activity}' detected")
        
        # Convert log to graph for the dedicated functions
        try:
            G = log_to_graph(log_counter)
            print(f"Graph created with {len(G.nodes())} nodes and {len(G.edges())} edges")
            
            # Use xor_cut_tau to check for τ-xor structure
            print("Checking for τ-xor structure using xor_cut_tau...")
            Sigma_1_tau, Sigma_2_tau, cost_tau, nodes_tau = xor_cut_tau(G, sup)
            
            if cost_tau == 0 and Sigma_1_tau is not None and Sigma_2_tau is not None:
                print(f"τ-xor condition satisfied with cost {cost_tau}")
                print(f"  Sigma_1: {Sigma_1_tau}, Sigma_2: {Sigma_2_tau}")
                
                # Split the log using the τ-xor cut
                log1_counter, log2_counter = split.split('exc', [Sigma_1_tau, Sigma_2_tau], log_counter)
                
                print(f"τ-xor split results:")
                print(f"  Log1: {len(log1_counter)} unique traces, {sum(log1_counter.values())} total occurrences")
                print(f"  Log2: {len(log2_counter)} unique traces, {sum(log2_counter.values())} total occurrences")
                
                # Recursively process both parts
                child1 = recursion_full(log1_counter, timing_data, depth + 1, max_depth, sup)
                child2 = recursion_full(log2_counter, timing_data, depth + 1, max_depth, sup)
                
                return ProcessTreeNode(operator='exc', children=[child1, child2])
            
            # Use loop_cut_ilp to check for loop structure
            print("Checking for loop structure using loop_cut_ilp...")
            Sigma_1_loop, Sigma_2_loop, cost_loop, nodes_loop = loop_cut_ilp(G, sup)
            
            if cost_loop == 0 and Sigma_1_loop is not None and Sigma_2_loop is not None:
                print(f"Loop condition satisfied with cost {cost_loop}")
                print(f"  Sigma_1: {Sigma_1_loop}, Sigma_2: {Sigma_2_loop}")
                
                # Split the log using the loop cut
                log1_counter, log2_counter = split.split('loop', [Sigma_1_loop, Sigma_2_loop], log_counter)
                
                print(f"Loop split results:")
                print(f"  Log1: {len(log1_counter)} unique traces, {sum(log1_counter.values())} total occurrences")
                print(f"  Log2: {len(log2_counter)} unique traces, {sum(log2_counter.values())} total occurrences")
                
                # Recursively process both parts
                child1 = recursion_full(log1_counter, timing_data, depth + 1, max_depth, sup)
                child2 = recursion_full(log2_counter, timing_data, depth + 1, max_depth, sup)
                
                return ProcessTreeNode(operator='loop', children=[child1, child2])
                
        except Exception as e:
            print(f"Error in base case cut detection: {e}")
            import traceback
            traceback.print_exc()
        
        # If neither special structure is found, return the single activity
        print("No special structure found, returning single activity leaf node")
        return ProcessTreeNode(activity=activity)

    # Normal case: multiple activities, proceed with cut detection
    print("Multiple activities detected, proceeding with cut detection")

    # Convert log to graph
    try:
        G = log_to_graph(log_counter)
        print(f"Graph created with {len(G.nodes())} nodes and {len(G.edges())} edges")
    except Exception as e:
        print(f"Graph creation failed: {e}")
        return _create_exclusive_choice_node(all_activities)

    # Initialize cut results
    cut_results = {
        'exc': {'Sigma_1': None, 'Sigma_2': None, 'cost': float('inf'), 'n': 0, 'time': 0, 'success': False},
        'seq': {'Sigma_1': None, 'Sigma_2': None, 'cost': float('inf'), 'n': 0, 'time': 0, 'success': False},
        'par': {'Sigma_1': None, 'Sigma_2': None, 'cost': float('inf'), 'n': 0, 'time': 0, 'success': False},
        'loop': {'Sigma_1': None, 'Sigma_2': None, 'cost': float('inf'), 'n': 0, 'time': 0, 'success': False}
    }

    # Try all cuts in order of preference with early termination
    cut_functions = [
        ('exc', xor_cut_ilp),
        ('seq', seq_cut_ilp), 
        ('par', par_cut_ilp),
        ('loop', loop_cut_ilp)
    ]

    best_op = None
    best_cost = float('inf')
    found_perfect_cut = False
    
    for op_name, cut_func in cut_functions:
        # Early termination: if we already found a perfect cut (cost 0), stop evaluating
        if found_perfect_cut:
            print(f"  Skipping {op_name} cut - already found perfect solution")
            continue
            
        try:
            print(f"Trying {op_name} cut...")
            start = time.time()
            
            if op_name == 'exc':
                Sigma_1, Sigma_2, total_cost, nodes = cut_func(G, sup)
            elif op_name == 'seq':
                Sigma_1, Sigma_2, total_cost, nodes = cut_func(G, log, sup)
            elif op_name == 'par':
                Sigma_1, Sigma_2, total_cost, nodes = cut_func(G, sup)
            elif op_name == 'loop':
                Sigma_1, Sigma_2, total_cost, nodes = cut_func(G, sup)
                
            end = time.time()
            
            print(f"  {op_name} cut result: Sigma_1={Sigma_1}, Sigma_2={Sigma_2}, cost={total_cost}")
            
            # Check if we have a valid cut
            if (Sigma_1 is not None and Sigma_2 is not None and 
                len(Sigma_1) > 0 and len(Sigma_2) > 0 and
                total_cost is not None and total_cost < float('inf')):
                
                cut_results[op_name] = {
                    'Sigma_1': Sigma_1, 'Sigma_2': Sigma_2, 
                    'cost': total_cost, 'n': nodes, 'time': end - start, 'success': True
                }
                
                # Update best cut found so far
                if total_cost < best_cost:
                    best_op = op_name
                    best_cost = total_cost
                    print(f"  ✓ {op_name} cut successful with cost {total_cost} (new best)")
                else:
                    print(f"  ✓ {op_name} cut successful with cost {total_cost}")
                
                # CRITICAL: Check for perfect solution (cost 0)
                if total_cost == 0:
                    print(f"  🎯 PERFECT {op_name.upper()} CUT FOUND! Stopping cut evaluation for this subgraph.")
                    found_perfect_cut = True
                    # Don't break - we still want to record timing for this cut
                    
            else:
                print(f"  ✗ {op_name} cut invalid - empty sets or infinite cost")
                
        except Exception as e:
            print(f"  ✗ {op_name} cut failed: {e}")

    print(f"Best cut: {best_op} with cost {best_cost}")

    # Record timing data
    timing_entry = {
        'depth': depth,
        'activities': list(all_activities),
        'num_traces': sum(log_counter.values()),
        'num_trace_variants': len(log_counter),
        'best_operator': best_op,
        'best_cost': best_cost,
        'found_perfect_cut': found_perfect_cut
    }
    
    for op in cut_results.keys():
        timing_entry[f'{op}_time'] = cut_results[op]['time']
        timing_entry[f'{op}_cost'] = cut_results[op]['cost'] if cut_results[op]['success'] else None
        timing_entry[f'{op}_n'] = cut_results[op]['n']
        timing_entry[f'{op}_success'] = cut_results[op]['success']
    
    timing_data.append(timing_entry)

    # If no valid cut found, use fallback
    if best_op is None or best_cost == float('inf'):
        print(f"No valid cut found, creating exclusive choice with {len(all_activities)} activities")
        return _create_exclusive_choice_node(all_activities)

    # Split the log using the best cut
    best_Sigma_1 = cut_results[best_op]['Sigma_1']
    best_Sigma_2 = cut_results[best_op]['Sigma_2']
    
    print(f"Splitting log with {best_op} cut...")
    print(f"  Sigma_1: {best_Sigma_1}")
    print(f"  Sigma_2: {best_Sigma_2}")
    
    try:
        log1_counter, log2_counter = split.split(best_op, [best_Sigma_1, best_Sigma_2], log_counter)
        
        print(f"Split results:")
        print(f"  Log1: {len(log1_counter)} unique traces, {sum(log1_counter.values())} total occurrences")
        print(f"  Log2: {len(log2_counter)} unique traces, {sum(log2_counter.values())} total occurrences")
        
        # Check if splits are valid
        if not log1_counter or not log2_counter:
            print("One of the splits is empty, using exclusive choice fallback")
            return _create_exclusive_choice_node(all_activities)
        
        # Recursively process the split logs
        child1 = recursion_full(log1_counter, timing_data, depth + 1, max_depth, sup)
        child2 = recursion_full(log2_counter, timing_data, depth + 1, max_depth, sup)
        
        return ProcessTreeNode(operator=best_op, children=[child1, child2])
        
    except Exception as e:
        print(f"Split failed for operator {best_op}: {e}")
        import traceback
        traceback.print_exc()
        return _create_exclusive_choice_node(all_activities)

def recursion_full_old(log, timing_data=None, depth=0, max_depth=20, sup=1.0):

    if timing_data is None:
        timing_data = []

    if depth > max_depth:
        print(f"Max depth reached at depth {depth}, returning fallback")
        all_activities = _get_all_activities(log)
        return _create_exclusive_choice_node(all_activities)

    print(f"\n=== Recursion Depth {depth} ===")

    # Convert log to standard Counter format
    log_counter = _convert_log_to_counter(log)

    print(f"Log has {sum(log_counter.values())} total trace occurrences")
    print(f"Log has {len(log_counter)} unique traces")

    # Print sample traces
    if log_counter:
        sample_traces = list(log_counter.keys())[:3]
        print(f"Sample traces: {sample_traces}")

    # Handle empty log
    if not log_counter:
        print("Empty log detected, returning tau")
        return ProcessTreeNode(activity="tau")

    # Get unique activities
    all_activities = _get_all_activities_from_counter(log_counter)
    print(f"Unique activities: {all_activities}")

    # IMbi BASE CASE 1: If event log has only empty traces (|Σ| = 0)
    if len(all_activities) == 0:
        print("Base case: Only empty traces detected, returning tau")
        return ProcessTreeNode(activity="tau")

    # IMbi BASE CASE 2: If all traces have only one activity (|Σ| ≤ 1)
    if len(all_activities) == 1:
        activity = next(iter(all_activities))
        print(f"Base case: Single activity '{activity}' detected")
        
        # Check for τ-xor case (empty traces proportion)
        total_traces = sum(log_counter.values())
        empty_traces = log_counter.get((), 0)  # Count of empty traces
        
        print(f"Total traces: {total_traces}, Empty traces: {empty_traces}")
        
        # Calculate c+ cost for empty traces proportion
        # c+ = max(0, total_traces * sup - empty_traces)
        c_plus = max(0, total_traces * sup - empty_traces)
        
        print(f"τ-xor check: c+ = max(0, {total_traces} * {sup} - {empty_traces}) = {c_plus}")
        
        # For single log, we only have c+ (no negative log, so c- = 0)
        # If c+ <= 0, use τ-xor structure
        if c_plus <= 0 and empty_traces > 0:
            print("τ-xor condition satisfied, creating τ-xor structure")
            
            # Remove empty traces for recursive call
            non_empty_log = Counter()
            for trace, count in log_counter.items():
                if len(trace) > 0:  # Keep only non-empty traces
                    non_empty_log[trace] = count
            
            if non_empty_log:
                print(f"Recursing on non-empty log with {sum(non_empty_log.values())} traces")
                child = recursion_full(non_empty_log, timing_data, depth + 1, max_depth, sup)
                return ProcessTreeNode(operator='exc', children=[
                    ProcessTreeNode(activity="tau"),
                    child
                ])
            else:
                # This shouldn't happen due to our condition, but for safety
                return ProcessTreeNode(activity="tau")
        
        # Check for τ-loop case (self-loop behavior)
        # Count total events and check if events > traces (indicating loops)
        total_events = 0
        self_loop_count = 0
        
        for trace in log_counter:
            trace_count = log_counter[trace]
            total_events += len(trace) * trace_count
            
            # Count self-loops (consecutive same activities)
            for i in range(len(trace) - 1):
                if trace[i] == trace[i + 1] == activity:
                    self_loop_count += trace_count
        
        events_to_traces_ratio = total_events / total_traces if total_traces > 0 else 0
        
        print(f"τ-loop check: events/traces ratio = {events_to_traces_ratio:.2f}")
        print(f"Self-loop count: {self_loop_count}")
        
        # p+ = max(0, sup * (total_events / 2) - self_loop_count)
        p_plus = max(0, sup * (total_events / 2) - self_loop_count)
        
        print(f"p+ = max(0, {sup} * ({total_events} / 2) - {self_loop_count}) = {p_plus}")
        
        # If p+ <= 0 and we have evidence of looping, use τ-loop structure
        if p_plus <= 0 and self_loop_count > 0 and events_to_traces_ratio > 1.1:
            print("τ-loop condition satisfied, creating loop structure")
            return ProcessTreeNode(operator='loop', children=[
                ProcessTreeNode(activity=activity),
                ProcessTreeNode(activity="tau")
            ])
        
        # Otherwise, return the single activity
        print("Returning single activity leaf node")
        return ProcessTreeNode(activity=activity)

    # Normal case: multiple activities, proceed with cut detection
    print("Multiple activities detected, proceeding with cut detection")

    # Convert log to graph
    try:
        G = log_to_graph(log_counter)
        print(f"Graph created with {len(G.nodes())} nodes and {len(G.edges())} edges")
    except Exception as e:
        print(f"Graph creation failed: {e}")
        return _create_exclusive_choice_node(all_activities)

    # Initialize cut results
    cut_results = {
        'exc': {'Sigma_1': None, 'Sigma_2': None, 'cost': float('inf'), 'n': 0, 'time': 0, 'success': False},
        'seq': {'Sigma_1': None, 'Sigma_2': None, 'cost': float('inf'), 'n': 0, 'time': 0, 'success': False},
        'par': {'Sigma_1': None, 'Sigma_2': None, 'cost': float('inf'), 'n': 0, 'time': 0, 'success': False},
        'loop': {'Sigma_1': None, 'Sigma_2': None, 'cost': float('inf'), 'n': 0, 'time': 0, 'success': False}
    }

    # Try all cuts in order of preference with early termination
    cut_functions = [
        ('exc', xor_cut_ilp),
        ('seq', seq_cut_ilp), 
        ('par', par_cut_ilp),
        ('loop', loop_cut_ilp)
    ]

    best_op = None
    best_cost = float('inf')
    found_perfect_cut = False
    
    for op_name, cut_func in cut_functions:
        # Early termination: if we already found a perfect cut (cost 0), stop evaluating
        if found_perfect_cut:
            print(f"  Skipping {op_name} cut - already found perfect solution")
            continue
            
        try:
            print(f"Trying {op_name} cut...")
            start = time.time()
            
            if op_name == 'exc':
                Sigma_1, Sigma_2, total_cost, nodes = cut_func(G, sup)
            elif op_name == 'seq':
                Sigma_1, Sigma_2, total_cost, nodes = cut_func(G, log, sup)
            elif op_name == 'par':
                Sigma_1, Sigma_2, total_cost, nodes = cut_func(G, sup)
            elif op_name == 'loop':
                Sigma_1, Sigma_2, total_cost, nodes = cut_func(G, sup)
                
            end = time.time()
            
            print(f"  {op_name} cut result: Sigma_1={Sigma_1}, Sigma_2={Sigma_2}, cost={total_cost}")
            
            # Check if we have a valid cut
            if (Sigma_1 is not None and Sigma_2 is not None and 
                len(Sigma_1) > 0 and len(Sigma_2) > 0 and
                total_cost is not None and total_cost < float('inf')):
                
                cut_results[op_name] = {
                    'Sigma_1': Sigma_1, 'Sigma_2': Sigma_2, 
                    'cost': total_cost, 'n': nodes, 'time': end - start, 'success': True
                }
                
                # Update best cut found so far
                if total_cost < best_cost:
                    best_op = op_name
                    best_cost = total_cost
                    print(f"  ✓ {op_name} cut successful with cost {total_cost} (new best)")
                else:
                    print(f"  ✓ {op_name} cut successful with cost {total_cost}")
                
                # CRITICAL: Check for perfect solution (cost 0)
                if total_cost == 0:
                    print(f"  🎯 PERFECT {op_name.upper()} CUT FOUND! Stopping cut evaluation for this subgraph.")
                    found_perfect_cut = True
                    # Don't break - we still want to record timing for this cut
                    
            else:
                print(f"  ✗ {op_name} cut invalid - empty sets or infinite cost")
                
        except Exception as e:
            print(f"  ✗ {op_name} cut failed: {e}")

    print(f"Best cut: {best_op} with cost {best_cost}")

    # Record timing data
    timing_entry = {
        'depth': depth,
        'activities': list(all_activities),
        'num_traces': sum(log_counter.values()),
        'num_trace_variants': len(log_counter),
        'best_operator': best_op,
        'best_cost': best_cost,
        'found_perfect_cut': found_perfect_cut  # Track if we terminated early
    }
    
    for op in cut_results.keys():
        timing_entry[f'{op}_time'] = cut_results[op]['time']
        timing_entry[f'{op}_cost'] = cut_results[op]['cost'] if cut_results[op]['success'] else None
        timing_entry[f'{op}_n'] = cut_results[op]['n']
        timing_entry[f'{op}_success'] = cut_results[op]['success']
    
    timing_data.append(timing_entry)

    # If no valid cut found, use fallback
    if best_op is None or best_cost == float('inf'):
        print(f"No valid cut found, creating exclusive choice with {len(all_activities)} activities")
        return _create_exclusive_choice_node(all_activities)

    # Split the log using the best cut
    best_Sigma_1 = cut_results[best_op]['Sigma_1']
    best_Sigma_2 = cut_results[best_op]['Sigma_2']
    
    print(f"Splitting log with {best_op} cut...")
    print(f"  Sigma_1: {best_Sigma_1}")
    print(f"  Sigma_2: {best_Sigma_2}")
    
    try:
        log1_counter, log2_counter = split.split(best_op, [best_Sigma_1, best_Sigma_2], log_counter)
        
        print(f"Split results:")
        print(f"  Log1: {len(log1_counter)} unique traces, {sum(log1_counter.values())} total occurrences")
        print(f"  Log2: {len(log2_counter)} unique traces, {sum(log2_counter.values())} total occurrences")
        
        # Check if splits are valid
        if not log1_counter or not log2_counter:
            print("One of the splits is empty, using exclusive choice fallback")
            return _create_exclusive_choice_node(all_activities)
        
        # Recursively process the split logs
        child1 = recursion_full(log1_counter, timing_data, depth + 1, max_depth, sup)
        child2 = recursion_full(log2_counter, timing_data, depth + 1, max_depth, sup)
        
        return ProcessTreeNode(operator=best_op, children=[child1, child2])
        
    except Exception as e:
        print(f"Split failed for operator {best_op}: {e}")
        import traceback
        traceback.print_exc()
        return _create_exclusive_choice_node(all_activities)



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

def create_timing_dataframe(timing_data):
    """Convert timing data to pandas DataFrame"""
    return pd.DataFrame(timing_data)

from pm4py.objects.process_tree.obj import ProcessTree
from pm4py.objects.process_tree.utils import generic
from pm4py.objects.process_tree.obj import Operator

def to_pm4py_tree(node, parent=None):
    tree = ProcessTree()
    tree.parent = parent
    
    if node.activity:
        # Leaf node
        tree.label = node.activity
    else:
        # Operator node
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