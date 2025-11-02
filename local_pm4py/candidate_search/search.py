from local_pm4py.functions.functions import n_edges
from local_pm4py.functions.functions import add_SE
from local_pm4py.candidate_search.is_allowed import is_allowed
import networkx as nx


def adj(node_set, net,n_thr):
    adj_set = set()
    for node in node_set:
        if len(set(net.neighbors(node)))>0:
            max_out = max([n_edges(net, {node}, {x}) for x in net.neighbors(node)])
            filt_num = max_out * n_thr
            adj_set = adj_set.union(set([x for x in net.neighbors(node) if n_edges(net, {node}, {x})>=filt_num]))
        else:
            adj_set = adj_set.union(set(net.neighbors(node)))
    return adj_set


def find_possible_partitions(net, rules, st_net, en_net,fp):
    activity_list = set(net.nodes) - {'start', 'end'}
    queue = [(set(), {'start'})]
    visited = set()
    valid = []
    reserve = []
    # if n_edges(net, {'start'}, {'end'}) > 0:
    #     na, block, exclude_dic = is_allowed(activity_list, set(), rules, st_net, en_net)
    #     if 'exc_tau' not in na:
    #         valid.append((activity_list, set(), {'exc_tau'}))
    #     else:
    #         reserve.append(((activity_list, set(), {'exc_tau'}), 'exc_tau', sum([y[2] for y in exclude_dic['exc_tau']])))

    # if (n_edges(net, set(en_net.keys()), set(st_net.keys())) > 0) and set(st_net.keys())==set([x for x in st_net.keys() if (x,x) in fp.edges() and fp[x][x]['weight'] > 0]):
    #     na, block, exclude_dic = is_allowed(activity_list, set(), rules, st_net, en_net)
    #     if 'loop_tau' not in na:
    #         valid.append((activity_list, set(), {'loop_tau'}))
    #     else:
    #         reserve.append(((activity_list, set(), {'loop_tau'}), 'loop_tau', sum([y[2] for y in exclude_dic['loop_tau']])))

    while len(queue) != 0:
        current = queue.pop()
        current_set, current_adj = current

        for x in current_adj:
            new_state = current_set | {x}
            new_state = add_SE(new_state, st_net, en_net)

            if frozenset(new_state) not in visited:
                new_adj = (current_adj | adj({x}, net,0)) - new_state
                visited.add(frozenset(new_state))
                possible_cuts = set()

                B = activity_list - new_state
                na, block, exclude_dic = is_allowed(new_state, B, rules, st_net, en_net)

                if (len(B) == 0) or (len(B) == len(activity_list)):
                    queue.append((new_state, new_adj))
                    continue

                B = add_SE(B,st_net,en_net)
                netA = net.subgraph(new_state)
                netB = net.subgraph(B)

                # 'start' ~> netB
                if 'start' in netB:
                    not2startB = set(netB.nodes) - set(nx.descendants(netB, 'start')) - {'start', 'end'}
                else:
                    not2startB = set(netB.nodes) - {'start', 'end'}
                # 'end' ~> netA
                if 'end' in netA:
                    not2endA = set(netA.nodes) - set(nx.ancestors(netA, 'end')) - {'start', 'end'}
                else:
                    not2endA = set(netA.nodes) - {'start', 'end'}
                # 'end' ~> netB
                if 'end' in netB:
                    not2endB = set(netB.nodes) - set(nx.ancestors(netB, 'end')) - {'start', 'end'}
                else:
                    not2endB = set(netB.nodes) - {'start', 'end'}

                if not not2endB:
                    possible_cuts.add("seq")
                    if len(not2endA) == 0:
                        if n_edges(net, B-{'start','end'}, new_state & set(st_net.keys())) != 0 and n_edges(net, new_state & set(en_net.keys()), B-{'start','end'}) != 0:
                                possible_cuts.add("loop")
                        if len(not2startB) == 0 and (B not in visited):
                            possible_cuts.update(["exc", "par"])
                elif not not2endA:
                    if n_edges(net, B, new_state & set(st_net.keys())) != 0 and n_edges(net,new_state & set(en_net.keys()),B) != 0:
                            possible_cuts.add("loop")
                for x in possible_cuts:
                    if x in na:
                        reserve.append(((new_state, B, possible_cuts),x, sum([y[2] for y in exclude_dic[x]])))
                possible_cuts = possible_cuts - na
                if len(possible_cuts) > 0:
                    valid.append((new_state, B, possible_cuts))

                if not block:
                    queue.append((new_state, new_adj))

    return valid,reserve



