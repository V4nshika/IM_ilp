from local_pm4py.functions.functions import n_edges,get_edge_weight, aggregate_dictionaries


def cost_seq(net, A, B, sup, flow):
    c1 = {}
    c2 = {}
    sum_out_degrees_A = sum(net.out_degree(p, weight='weight') for p in A)
    sum_out_degrees_B = sum(net.out_degree(p, weight='weight') for p in B)
    sum_out_degrees_total = sum_out_degrees_A + sum_out_degrees_B

    for x in A:
        for y in B:
            c1[f'({y}, {x})'] = c1.get(f'({y}, {x})', 0) + get_edge_weight(net, y, x)
            out_degree_x = net.out_degree(x, weight='weight')
            out_degree_y = net.out_degree(y, weight='weight')
            c2[f'({x}, {y})'] = c2.get(f'({x}, {y})', 0) + max(0, out_degree_x * sup * (out_degree_y / sum_out_degrees_total) - get_edge_weight(flow, x, y))
    return aggregate_dictionaries([c1],[c2])


def cost_exc(net, A, B):
    c1 = {}
    for x in A:
        for y in B:
            c1[f'({x}, {y})'] = c1.get(f'({x}, {y})', 0) + get_edge_weight(net, x, y)
            c1[f'({y}, {x})'] = c1.get(f'({y}, {x})', 0) + get_edge_weight(net, y, x)
    return aggregate_dictionaries([c1],[{}])



def cost_exc_tau(net, sup):
    c = {}
    if 'start' in net.nodes():
        c['xor_tau'] = max(0, sup * net.out_degree('start', weight='weight') - get_edge_weight(net,'start','end'))
    return aggregate_dictionaries([c],[{}])


def cost_par(net, A, B, sup):
    c1 = {}
    c2 = {}

    sum_out_degrees_A = sum(net.out_degree(p, weight='weight') for p in A)
    sum_out_degrees_B = sum(net.out_degree(p, weight='weight') for p in B)
    sum_out_degrees_total = sum_out_degrees_A + sum_out_degrees_B

    for a in A:
        out_degree_a = net.out_degree(a, weight='weight')
        for b in B:
            out_degree_b = net.out_degree(b, weight='weight')

            c1[f'({a}, {b})'] = c1.get(f'({a}, {b})', 0) + max(0,(out_degree_a * sup * out_degree_b) / sum_out_degrees_total - get_edge_weight(net, a, b))
            c2[f'({b}, {a})'] = c2.get(f'({b}, {a})', 0) + max(0,(out_degree_b * sup * out_degree_a) / sum_out_degrees_total - get_edge_weight(net, b, a))

    return aggregate_dictionaries([{}],[c1,c2])


def cost_loop(net, A, B, sup, start_A, end_A, input_B, output_B, start_activities, end_activities):
    M_P = max(n_edges(net,output_B,start_A), n_edges(net,end_A,input_B))
    c1 = {}
    c2 = {}
    c3 = {}
    c4 = {}
    c5 = {}

    for x in B:
        c1[f'(start,{x})'] = c1.get(f'(start,{x})', 0) + get_edge_weight(net, 'start', x)
        c1[f'({x}, end)'] = c1.get(f'({x}, end)', 0) + get_edge_weight(net, x, 'end')
        for y in A - end_A:
            c2[f'({y},{x})'] = c2.get(f'({y},{x})', 0) + get_edge_weight(net, y, x)
        for y in A - start_A:
            c3[f'({x},{y})'] = c3.get(f'({x},{y})', 0) + get_edge_weight(net, x, y)


    if n_edges(net, output_B, start_A):
        # c4 = 0
        for a in start_A:
            for b in output_B:
                c4[f'({b},{a})'] = c4.get(f'({b},{a})', 0) + max(0, M_P * sup * (n_edges(net,{'start'},{a})/n_edges(net, {'start'}, start_A)) * (n_edges(net, {b}, start_A)/ n_edges(net, output_B, start_A))- get_edge_weight(net, b, a))
    else:
        c4['no_edge'] = c4.get('no_edge', 0) + M_P * sup

    if n_edges(net, end_A, input_B):
        # c5 = 0
        for a in end_A:
            for b in input_B:
               c5[f'({a},{b})'] = c5.get(f'({a},{b})', 0) + max(0, M_P * sup * (n_edges(net,{a}, {'end'})/n_edges(net, end_A, {'end'})) * (n_edges(net, end_A, {b})/ n_edges(net, end_A, input_B))- get_edge_weight(net, a, b))
    else:
        c5['no_edge'] = c5.get('no_edge', 0) + M_P * sup

    return aggregate_dictionaries([c1,c2,c3],[c4,c5])


def cost_loop_tau(net, sup, start_activities, end_activities):
    c = {}
    M_P = n_edges(net,set(end_activities.keys()),set(start_activities.keys()))
    start_sum = sum(start_activities.values())
    end_sum = sum(end_activities.values())

    for x in start_activities:
        for y in end_activities:
            if (y, x) in net.edges:
                c[f'({y},{x})'] = c.get(f'({y},{x})',0) + max(0, M_P * sup * (start_activities[x] / start_sum) * (end_activities[y] / end_sum) - get_edge_weight(net, y, x))
            else:
                c[f'({y},{x})'] = c.get(f'({y},{x})',0) +M_P * sup * (start_activities[x] / start_sum) * (end_activities[y] / end_sum)

    return aggregate_dictionaries([{}],[c])

def overal_cost(costP,costM,ratio, size_par):
    ov_c= costP - ratio * size_par * costM
    return ov_c