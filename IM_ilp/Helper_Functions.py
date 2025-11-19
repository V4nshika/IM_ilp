import networkx as nx
import pulp
import numpy as np
import os
from collections import Counter
from pm4py.discovery import discover_dfg
from local_pm4py.cut_quality.cost_functions import cost_functions
from pm4py.objects.log.importer.xes import importer as xes_importer
from local_pm4py.functions.functions import get_edge_weight
from collections import defaultdict


def log_to_graph_old(log):
    edge_counts = defaultdict(int)
    has_traces = False

    # Handle PM4Py EventLog object
    if hasattr(log, '__class__') and 'EventLog' in str(log.__class__):
        for trace in log:
            # Extract activity names from PM4Py events
            activities = [event['concept:name'] for event in trace]

            if not activities:
                continue

            has_traces = True
            # Add edges
            edge_counts[('start', activities[0])] += 1
            edge_counts[(activities[-1], 'end')] += 1

            # Add intermediate transitions
            for i in range(len(activities) - 1):
                edge_counts[(activities[i], activities[i + 1])] += 1

    # Handle Counter input (where keys are tuples of activities)
    elif isinstance(log, Counter):
        for trace, count in log.items():
            if not trace:  # Skip empty traces
                continue

            has_traces = True
            # Convert tuple to list of activities
            activities = list(trace)

            # Add edges with frequency count
            edge_counts[('start', activities[0])] += count
            edge_counts[(activities[-1], 'end')] += count

            # Add intermediate transitions
            for i in range(len(activities) - 1):
                edge_counts[(activities[i], activities[i + 1])] += count

    # Handle list of traces input
    else:
        for trace in log:
            # Skip empty traces
            if not trace:
                continue

            # Extract activity names if events are dictionaries
            if isinstance(trace[0], dict):
                activities = [event["concept:name"] for event in trace]
            else:
                # Assume trace is already a list of activity names
                activities = list(trace)

            if not activities:
                continue

            has_traces = True
            # Add edges (weight = 1 for each occurrence)
            edge_counts[('start', activities[0])] += 1
            edge_counts[(activities[-1], 'end')] += 1

            # Add intermediate transitions
            for i in range(len(activities) - 1):
                edge_counts[(activities[i], activities[i + 1])] += 1

    # Build the graph - ensure all nodes are strings
    G = nx.DiGraph()

    # Always add start and end nodes to the graph, even if empty
    G.add_node('start')
    G.add_node('end')

    # Add edges from edge_counts
    for (src, tgt), weight in edge_counts.items():
        # Convert all nodes to strings to avoid PM4Py object issues
        G.add_edge(str(src), str(tgt), weight=weight)

    # If we have no traces but want to preserve the empty graph structure,
    # we need to handle this case in your ILP functions
    if not has_traces:
        print("Warning: No valid traces found in log")

    return G


def log_to_graph(log):
    edge_counts = defaultdict(int)
    has_traces = False

    empty_trace_count = 0

    # Handle PM4Py EventLog object
    if hasattr(log, '__class__') and 'EventLog' in str(log.__class__):
        for trace in log:
            # Extract activity names from PM4Py events
            activities = [event['concept:name'] for event in trace]

            if not activities:
                continue

            has_traces = True
            # Add edges
            edge_counts[('start', activities[0])] += 1
            edge_counts[(activities[-1], 'end')] += 1

            # Add intermediate transitions
            for i in range(len(activities) - 1):
                edge_counts[(activities[i], activities[i + 1])] += 1

    # Handle Counter input (where keys are tuples of activities)
    elif isinstance(log, Counter):
        for trace, count in log.items():
            if not trace:  # Skip empty traces
                continue

            has_traces = True
            # Convert tuple to list of activities
            activities = list(trace)

            # Add edges with frequency count
            edge_counts[('start', activities[0])] += count
            edge_counts[(activities[-1], 'end')] += count

            # Add intermediate transitions
            for i in range(len(activities) - 1):
                edge_counts[(activities[i], activities[i + 1])] += count

    # Handle list of traces input
    else:
        for trace in log:
            # Skip empty traces
            if not trace:
                empty_trace_count += 1
                continue

            # Extract activity names if events are dictionaries
            if isinstance(trace[0], dict):
                activities = [event["concept:name"] for event in trace]
            else:
                # Assume trace is already a list of activity names
                activities = list(trace)

            if not activities:
                continue

            has_traces = True
            # Add edges (weight = 1 for each occurrence)
            edge_counts[('start', activities[0])] += 1
            edge_counts[(activities[-1], 'end')] += 1

            # Add intermediate transitions
            for i in range(len(activities) - 1):
                edge_counts[(activities[i], activities[i + 1])] += 1

    if empty_trace_count > 0:
        edge_counts[('start', 'end')] += empty_trace_count

    # Build the graph - ensure all nodes are strings
    G = nx.DiGraph()

    # Always add start and end nodes to the graph, even if empty
    G.add_node('start')
    G.add_node('end')

    # Add edges from edge_counts
    for (src, tgt), weight in edge_counts.items():
        # Convert all nodes to strings to avoid PM4Py object issues
        G.add_edge(str(src), str(tgt), weight=weight)

    # If we have no traces but want to preserve the empty graph structure,
    # we need to handle this case in your ILP functions
    if not has_traces:
        print("Warning: No valid traces found in log")

    return G

def preprocess_graph_old(G, start, end):
    # Ensure all nodes are reachable from start and can reach end
    reachable_from_start = nx.descendants(G, start) | {start}
    reachable_to_end = nx.ancestors(G, end) | {end}
    valid_nodes = reachable_from_start & reachable_to_end
    return G.subgraph(valid_nodes), reachable_from_start, reachable_to_end


def preprocess_graph(G, start, end):
    # Check if start and end nodes exist
    if start not in G or end not in G:
        # Return an empty graph with start and end nodes
        empty_G = nx.DiGraph()
        empty_G.add_node(start)
        empty_G.add_node(end)
        return empty_G, start, end

    # Ensure all nodes are reachable from start and can reach end
    try:
        reachable_from_start = nx.descendants(G, start) | {start}
        reachable_to_end = nx.ancestors(G, end) | {end}
        valid_nodes = reachable_from_start & reachable_to_end

        # Create a subgraph with only valid nodes
        reduced_graph = G.subgraph(valid_nodes).copy()

        # Remove nodes that have no edges (isolated) except start and end
        isolated_nodes = [node for node in reduced_graph.nodes()
                          if reduced_graph.degree(node) == 0 and node not in (start, end)]
        reduced_graph.remove_nodes_from(isolated_nodes)

        # Ensure start and end are still in the graph
        if start not in reduced_graph:
            reduced_graph.add_node(start)
        if end not in reduced_graph:
            reduced_graph.add_node(end)

        return reduced_graph, start, end
    except nx.NetworkXError as e:
        # If there's an error, return a minimal graph with start and end
        print(f"Graph preprocessing error: {e}")
        minimal_G = nx.DiGraph()
        minimal_G.add_node(start)
        minimal_G.add_node(end)
        return minimal_G, start, end


def nx_to_mat_and_weights(G, sup=1.0):
    # Always include start and end nodes
    if 'start' not in G:
        G.add_node('start')
    if 'end' not in G:
        G.add_node('end')

    nodes = list(G.nodes())
    node_index = {node: i for i, node in enumerate(nodes)}
    n = len(nodes)

    # Create adjacency matrix
    A = np.zeros((n, n), dtype=int)
    for u, v, data in G.edges(data=True):
        i = node_index[u]
        j = node_index[v]
        A[i, j] = 1

    # Get weight matrix
    W_result = cost_(G, sup)

    # Create weight matrix
    W = np.zeros((n, n), dtype=int)

    if isinstance(W_result, dict):
        # Handle dictionary return type
        for key, weight in W_result.items():
            if isinstance(key, tuple) and len(key) == 2:
                # Dictionary key is (source, target)
                u, v = key
                if u in node_index and v in node_index:
                    i = node_index[u]
                    j = node_index[v]
                    W[i, j] = weight
            else:
                # Dictionary key might be index or something else
                # Fall back to using edge weights from graph
                for u, v, data in G.edges(data=True):
                    i = node_index[u]
                    j = node_index[v]
                    W[i, j] = data.get('capacity', 1)
                break
    else:
        # Use edge weights from graph
        for u, v, data in G.edges(data=True):
            i = node_index[u]
            j = node_index[v]
            W[i, j] = data.get('capacity', 1)

    return A, W, nodes, node_index


def extract_activities(node_list):
    # Helper to filter out start/end if needed
    return {node for node in node_list if node not in {'start', 'end'}}


def cost_(G, sup=1):
    nodes = list(G.nodes())
    out_deg = {n: G.out_degree(n, weight='weight') for n in nodes}
    total_out = sum(out_deg.values())

    edge_weights = nx.get_edge_attributes(G, "weight")

    cost_dict = {}

    for n in nodes:
        for m in nodes:
            if n == m:
                cost_dict[(n, m)] = 0
            expected = 0
            if total_out > 0:
                expected = (out_deg[n] * sup * out_deg[m]) / total_out
            actual = edge_weights.get((n, m), 0)
            cost = max(0, expected - actual)
            cost_dict[(n, m)] = cost

    return cost_dict

def cost_eventual(G_direct, log, sup=1.0):
    """
    Calculate eventual-follows costs for ALL possible edges according to Definition 9.
    This is edge-wise and partition-independent.
    """
    #print("graph start")
    # Build eventual-follows graph from log
    #print("indirect graph start")
    G_eventual = generate_nx_indirect_graph_from_log(log)
    #print("graph done")
    # Calculate node frequencies from direct graph
    node_freq = {}
    for node in G_direct.nodes():
        if node not in ['start', 'end']:
            node_freq[node] = sum(data.get('weight', 1) 
                                for _, _, data in G_direct.out_edges(node, data=True))
    #print("node freq done")
    # Calculate total frequency of all activities
    total_freq = sum(node_freq.values())
    
    if total_freq == 0:
        return {}
    
    # Calculate costs for ALL activity pairs - this is edge-wise and partition-independent
    cost_dict = {}
    activities = [node for node in G_direct.nodes() if node not in ['start', 'end']]
    #print("acts done")
    for a in activities:
        for b in activities:
            if a == b:
                continue
                
            # Get actual eventual-follows frequency from INDIRECT graph
            actual = G_eventual.get_edge_data(a, b, {'weight': 0})['weight']
            
            # Calculate expected frequency using Definition 9 formula
            # This depends ONLY on node frequencies, not on partition
            expected = (node_freq[a] * sup * node_freq[b]) / total_freq
            
            # Edge-wise cost: max(0, expected - actual)
            cost = max(0, expected - actual)
            
            cost_dict[(a, b)] = cost
    
    return cost_dict



def _convert_log_to_counter(log):
    """Convert any log format to Counter format"""
    if isinstance(log, Counter):
        #print("__")
        return log
    
    freq_dict = Counter()
    #print(log)
    
    # Handle PM4Py EventLog object
    if hasattr(log, '__class__') and 'EventLog' in str(log.__class__):
        for trace in log:
            activities = tuple(event['concept:name'] for event in trace)
            freq_dict[activities] += 1
    
    # Handle list of traces
    elif isinstance(log, list):
        for trace in log:
            if trace and isinstance(trace[0], dict):
                activities = tuple(event['concept:name'] for event in trace)
            else:
                activities = tuple(trace)
            freq_dict[activities] += 1
    
    else:
        raise ValueError(f"Unsupported log type: {type(log)}")
    
    return freq_dict

# In Helper_Functions.py

def generate_nx_indirect_graph_from_log(log):
    """
    Build eventual-follows graph from log.
    THIS IS THE OPTIMIZED VERSION.
    """
    # 1. Use a defaultdict (your "master dict" for this task)
    edge_weights = defaultdict(int)

    # 2. Convert log to Counter (if not already)
    # This also handles all the different log types
    log_counter = _convert_log_to_counter(log)
    
    # 3. Populate the dict. This is the fast "in-place editing."
    for trace, frequency in log_counter.items():
        _add_trace_to_eventual_graph(edge_weights, trace, frequency)
    
    # 4. Build the graph ONCE from the aggregated weights
    G = nx.DiGraph()
    # This is much, much faster than adding edges one by one
    G.add_edges_from((src, tgt, {'weight': weight}) 
                     for (src, tgt), weight in edge_weights.items())
    
    return G

def _add_trace_to_eventual_graph(edge_weights, trace, frequency): # Note: G is now edge_weights
    """
    Helper to add a single trace to the eventual graph.
    THIS IS THE OPTIMIZED VERSION.
    """
    if not trace:
        return

    # No need to convert, _convert_log_to_counter already gives tuples
    activities = list(trace) 
        
    for i in range(len(activities)):
        visited = set()
        src = activities[i]
        for j in range(i + 1, len(activities)):
            tgt = activities[j]
            # Add check to avoid self-loops if not needed
            if tgt not in visited and src != tgt: 
                visited.add(tgt)
                
                # This is the "in-place edit"
                # It's one of the fastest operations in Python.
                # No G.has_edge() or G[src][tgt] lookups.
                edge_weights[(src, tgt)] += frequency