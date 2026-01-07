import networkx as nx
import numpy as np
from collections import Counter, defaultdict

def log_to_graph(log):
                edge_counts = defaultdict(int)
                has_traces = False
                empty_trace_count = 0

                # Handle PM4Py EventLog object
                if hasattr(log, '__class__') and 'EventLog' in str(log.__class__):
                                for trace in log:
                                                activities = [event['concept:name'] for event in trace]
                                                if not activities:
                                                                empty_trace_count += 1
                                                                continue
                                                has_traces = True
                                                edge_counts[('start', activities[0])] += 1
                                                edge_counts[(activities[-1], 'end')] += 1
                                                for i in range(len(activities) - 1):
                                                                edge_counts[(activities[i], activities[i + 1])] += 1

                # Handle Counter input (This is the one recursion_exp uses)
                elif isinstance(log, Counter):
                                for trace, count in log.items():
                                                if not trace: 
                                                                empty_trace_count += count 
                                                                continue
                                                has_traces = True
                                                activities = list(trace)
                                                edge_counts[('start', activities[0])] += count
                                                edge_counts[(activities[-1], 'end')] += count
                                                for i in range(len(activities) - 1):
                                                                edge_counts[(activities[i], activities[i + 1])] += count

                # Handle list of traces input
                else:
                                for trace in log:
                                                if not trace:
                                                                empty_trace_count += 1
                                                                continue
                                                if isinstance(trace[0], dict):
                                                                activities = [event["concept:name"] for event in trace]
                                                else:
                                                                activities = list(trace)
                                                if not activities:
                                                                continue
                                                has_traces = True
                                                edge_counts[('start', activities[0])] += 1
                                                edge_counts[(activities[-1], 'end')] += 1
                                                for i in range(len(activities) - 1):
                                                                edge_counts[(activities[i], activities[i + 1])] += 1

                # --- GRAPH CONSTRUCTION (Must be unindented) ---
                if empty_trace_count > 0:
                                edge_counts[('start', 'end')] += empty_trace_count

                G = nx.DiGraph()
                G.add_node('start')
                G.add_node('end')

                for (src, tgt), weight in edge_counts.items():
                                G.add_edge(str(src), str(tgt), weight=weight)
              
                return G



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


def cost_(G, sup=1):
                nodes = list(G.nodes())
                non_terminal = [n for n in nodes if n not in ['start', 'end']]
          
                out_deg = {n: G.out_degree(n, weight='weight') for n in nodes}
          
                total_out = sum(out_deg[n] for n in non_terminal)

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


def preprocess_graph(G, start, end):
                if start not in G or end not in G:
                                empty_G = nx.DiGraph()
                                empty_G.add_node(start)
                                empty_G.add_node(end)
                                return empty_G, start, end

                try:
                                reachable_from_start = nx.descendants(G, start) | {start}
                                reachable_to_end = nx.ancestors(G, end) | {end}
                                valid_nodes = reachable_from_start & reachable_to_end

                                reduced_graph = G.subgraph(valid_nodes).copy()

                                isolated_nodes = [node for node in reduced_graph.nodes()
                                                                                                        if reduced_graph.degree(node) == 0 and node not in (start, end)]
                                reduced_graph.remove_nodes_from(isolated_nodes)

                                if start not in reduced_graph:
                                                reduced_graph.add_node(start)
                                if end not in reduced_graph:
                                                reduced_graph.add_node(end)

                                return reduced_graph, start, end
                except nx.NetworkXError as e:
                                print(f"Graph preprocessing error: {e}")
                                minimal_G = nx.DiGraph()
                                minimal_G.add_node(start)
                                minimal_G.add_node(end)
                                return minimal_G, start, end


def extract_activities(node_list):
                return {node for node in node_list if node not in {'start', 'end'}}



def _convert_log_to_counter(log):
    """
    Convert any log format to Counter format using optimized standard library approaches.
    """
    if isinstance(log, Counter):
        return log

    # If it's a Pandas DataFrame (fastest non-pandas method)
    if hasattr(log, 'itertuples'):
        # Assuming standard PM4Py structure: Case ID is index or column
        # We need to group by case and tuple-ize events.
        # However, since Pandas was slow, we assume the input might simply be 
        # a list of traces (which is what recursion_full often passes internally).
        pass 

    # FAST PATH: List of Lists (Standard internal format for recursion)
    if isinstance(log, list):
        # If the log is empty
        if not log:
            return Counter()
            
        # Optimization: Check the type of the first element only once
        first_element = log[0]
        
        # Scenario 1: List of dicts (e.g., standard PM4Py trace)
        if isinstance(first_element, list) and len(first_element) > 0 and isinstance(first_element[0], dict):
            # Optimizing the extraction using map implies less interpreter overhead
            return Counter(
                tuple(event['concept:name'] for event in trace) 
                for trace in log
            )
        
        # Scenario 2: List of lists/tuples of strings (already simplified traces)
        # This is what 'split' usually returns
        return Counter(tuple(trace) for trace in log)

    # Legacy PM4Py EventLog Object support
    if hasattr(log, '__class__') and 'EventLog' in str(log.__class__):
        return Counter(
            tuple(event['concept:name'] for event in trace) 
            for trace in log
        )

    raise ValueError(f"Unsupported log type: {type(log)}")


def generate_nx_indirect_graph(log):
                """
                Build eventual-follows graph from log.
                Optimized version using in-place dict updates.
                """
                edge_weights = defaultdict(int)
                log_counter = _convert_log_to_counter(log)
          
                for trace, frequency in log_counter.items():
                                _add_trace_to_eventual_graph(edge_weights, trace, frequency)
          
                G = nx.DiGraph()
                G.add_edges_from((src, tgt, {'weight': weight}) 
                                                                                 for (src, tgt), weight in edge_weights.items())
                return G


def _add_trace_to_eventual_graph(edge_weights, trace, frequency):
                """Helper to add a single trace to the eventual graph."""
                if not trace:
                                return

                activities = list(trace) 
                for i in range(len(activities)):
                                visited = set()
                                src = activities[i]
                                for j in range(i + 1, len(activities)):
                                                tgt = activities[j]
                                                if tgt not in visited: 
                                                                visited.add(tgt)
                                                                edge_weights[(src, tgt)] += frequency


def cost_eventual(G_direct, log, sup=1.0):
                """
                Calculate eventual-follows costs for ALL possible edges.
                1. Builds the Indirect Graph from scratch for the current log (Actual).
                2. Calculates Node Frequencies from the current Direct Graph (Expected).
                3. Returns the cost dictionary.
                """
                # 1. Actual: Build eventual-follows graph from current log
                G_eventual = generate_nx_indirect_graph(log)
          
                # 2. Expected: Calculate node frequencies from direct graph
                node_freq = {}
                for node in G_direct.nodes():
                                if node not in ['start', 'end']:
                                                node_freq[node] = sum(data.get('weight', 1) 
                                                                                                                                for _, _, data in G_direct.out_edges(node, data=True))
          
                total_freq = sum(node_freq.values())
          
                if total_freq == 0:
                                return {}
          
                cost_dict = {}
                activities = [node for node in G_direct.nodes() if node not in ['start', 'end']]
          
                for a in activities:
                                for b in activities:
                                                # Get actual eventual-follows frequency from INDIRECT graph
                                                actual = G_eventual.get_edge_data(a, b, {'weight': 0})['weight']
                                          
                                                # Calculate expected frequency
                                                expected = (node_freq[a] * sup * node_freq[b]) / total_freq
                                          
                                                # Edge-wise cost: max(0, expected - actual)
                                                cost = max(0, expected - actual)
                                          
                                                if cost > 0:
                                                                cost_dict[(a, b)] = cost
                                                          
                return cost_dict


def n_edges(G, source_set, target_set):
        """
        Sum of weights of edges going from any node in source_set to any node in target_set.
        Mirrors prolysis.util.functions.n_edges logic.
        """
        weight = 0
        for u in source_set:
                if u in G:
                        for v in target_set:
                                if v in G and G.has_edge(u, v):
                                        weight += G[u][v].get('weight', 0)
        return weight
