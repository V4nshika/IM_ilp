

EXISTENCE = "existence"
EXACTLY_ONE = "exactly_one"
INIT = "init"
END = "end"
RESPONDED_EXISTENCE = "responded_existence"
RESPONSE = "response"
PRECEDENCE = "precedence"
COEXISTENCE = "coexistence"
NONCOEXISTENCE = "noncoexistence"
NONSUCCESSION = "nonsuccession"
ATMOST_ONE = "atmost1"

def is_allowed(S1,S2,rules,st_net,en_net):
    exclude = []
    exclude_dic = {'seq':set(), 'exc':set(), 'par':set(), 'loop':set(), 'loop_tau':set()}
    black_list = set()
    block = False


    for r,info in rules[ATMOST_ONE].items():
        if r in S1:
            exclude.append('loop')
            exclude_dic['loop'].add((ATMOST_ONE,r,info['support']))
            exclude.append('loop_tau')
            exclude_dic['loop_tau'].add((ATMOST_ONE,r,info['support']))
            black_list.add(r)
        elif r in S2:
            exclude.append('loop')
            exclude_dic['loop'].add((ATMOST_ONE,r,info['support']))
            exclude.append('loop_tau')
            exclude_dic['loop_tau'].add((ATMOST_ONE,r,info['support']))
            black_list.add(r)

    for r,info in rules[EXISTENCE].items():
        if r in S1:
            exclude.append('exc')
            exclude_dic['exc'].add((EXISTENCE,r,info['support']))
            black_list.add(r)
        elif r in S2:
            exclude.append('exc')
            exclude_dic['exc'].add((EXISTENCE,r,info['support']))
            exclude.append('loop')
            exclude_dic['loop'].add((EXISTENCE,r,info['support']))
            black_list.add(r)



    for r,info in rules[NONSUCCESSION].items():
        if r[0] in S1 and r[1] in S2:
            exclude.append('seq')
            exclude_dic['seq'].add((NONSUCCESSION,r,info['support']))
            exclude.append('loop')
            exclude_dic['loop'].add((NONSUCCESSION,r,info['support']))
            exclude.append('loop_tau')
            exclude_dic['loop_tau'].add((NONSUCCESSION,r,info['support']))
            exclude.append('par')
            exclude_dic['par'].add((NONSUCCESSION,r,info['support']))
            black_list.add(r)
            # block = True
        elif r[0] in S2 and r[1] in S1:
            exclude.append('par')
            exclude_dic['par'].add((NONSUCCESSION,r,info['support']))
            exclude.append('loop')
            exclude_dic['loop'].add((NONSUCCESSION,r,info['support']))
            exclude.append('loop_tau')
            exclude_dic['loop_tau'].add((NONSUCCESSION,r,info['support']))
            black_list.add(r)
        elif r[0] in S1 and r[1] in S1:
            exclude.append('loop')
            exclude_dic['loop'].add((NONSUCCESSION,r,info['support']))
            exclude.append('loop_tau')
            exclude_dic['loop_tau'].add((NONSUCCESSION,r,info['support']))
            black_list.add(r)
        elif r[0] in S2 and r[1] in S2:
            exclude.append('loop')
            exclude_dic['loop'].add((NONSUCCESSION,r,info['support']))
            exclude.append('loop_tau')
            exclude_dic['loop_tau'].add((NONSUCCESSION,r,info['support']))
            black_list.add(r)



    for r,info in rules[RESPONDED_EXISTENCE].items():
        if r[0] in S1 and r[1] in S2:
            exclude.append('exc')
            exclude_dic['exc'].add((RESPONDED_EXISTENCE,r,info['support']))
            exclude.append('loop')
            exclude_dic['loop'].add((RESPONDED_EXISTENCE,r,info['support']))
        elif r[0] in S2 and r[1] in S1:
            exclude.append('exc')
            exclude_dic['exc'].add((RESPONDED_EXISTENCE,r,info['support']))


    for r,info in rules[NONCOEXISTENCE].items():
        if r[0] in S1 and r[1] in S2:
            exclude.append('par')
            exclude_dic['par'].add((NONCOEXISTENCE,r,info['support']))
            exclude.append('seq')
            exclude_dic['seq'].add((NONCOEXISTENCE,r,info['support']))
            exclude.append('loop')
            exclude_dic['loop'].add((NONCOEXISTENCE,r,info['support']))
            exclude.append('loop_tau')
            exclude_dic['loop_tau'].add((NONCOEXISTENCE,r,info['support']))
        elif r[0] in S2 and r[1] in S1:
            exclude.append('par')
            exclude_dic['par'].add((NONCOEXISTENCE,r,info['support']))
            exclude.append('seq')
            exclude_dic['seq'].add((NONCOEXISTENCE,r,info['support']))
            exclude.append('loop')
            exclude_dic['loop'].add((NONCOEXISTENCE,r,info['support']))
            exclude.append('loop_tau')
            exclude_dic['loop_tau'].add((NONCOEXISTENCE,r,info['support']))
        elif r[0] in S1 and r[1] in S1:
            exclude.append('loop')
            exclude_dic['loop'].add((NONCOEXISTENCE,r,info['support']))
            exclude.append('loop_tau')
            exclude_dic['loop_tau'].add((NONCOEXISTENCE,r,info['support']))
        elif r[0] in S2 and r[1] in S2:
            exclude.append('loop')
            exclude_dic['loop'].add((NONCOEXISTENCE,r,info['support']))
            exclude.append('loop_tau')
            exclude_dic['loop_tau'].add((NONCOEXISTENCE,r,info['support']))


    for r,info in rules[COEXISTENCE].items():
        if r[0] in S1 and r[1] in S2:
            exclude.append('exc')
            exclude_dic['exc'].add((COEXISTENCE,r,info['support']))
            exclude.append('loop')
            exclude_dic['loop'].add((COEXISTENCE,r,info['support']))
        elif r[0] in S2 and r[1] in S1:
            exclude.append('exc')
            exclude_dic['exc'].add((COEXISTENCE,r,info['support']))
            exclude.append('loop')
            exclude_dic['loop'].add((COEXISTENCE,r,info['support']))


    for r,info in rules[RESPONSE].items():
        if (r[1], r[0]) not in rules[RESPONSE]:
            if r[0] in S1 and r[1] in S2:
                exclude.append('exc')
                exclude_dic['exc'].add((RESPONSE,r,info['support']))
                exclude.append('loop')
                exclude_dic['loop'].add((RESPONSE,r,info['support']))
                exclude.append('par')
                exclude_dic['par'].add((RESPONSE,r,info['support']))
            if r[0] in S2 and r[1] in S1:
                exclude.append('exc')
                exclude_dic['exc'].add((RESPONSE,r,info['support']))
                exclude.append('seq')
                exclude_dic['seq'].add((RESPONSE,r,info['support']))
                exclude.append('par')
                exclude_dic['par'].add((RESPONSE,r,info['support']))
                block = True

        else:
            if r[0] in S1 and r[1] in S2:
                exclude.append('exc')
                exclude_dic['exc'].add((RESPONSE,r,info['support']))
                exclude.append('loop')
                exclude_dic['loop'].add((RESPONSE,r,info['support']))
                exclude.append('par')
                exclude_dic['par'].add((RESPONSE,r,info['support']))
            if r[0] in S2 and r[1] in S1:
                exclude.append('exc')
                exclude_dic['exc'].add((RESPONSE,r,info['support']))
                exclude.append('seq')
                exclude_dic['seq'].add((RESPONSE,r,info['support']))
                exclude.append('par')
                exclude_dic['par'].add((RESPONSE,r,info['support']))

    for r,info in rules[PRECEDENCE].items():
        if (r[1], r[0]) not in rules[PRECEDENCE]:
            if r[0] in S1 and r[1] in S2:
                exclude.append('par')
                exclude_dic['par'].add((PRECEDENCE,r,info['support']))
                exclude.append('exc')
                exclude_dic['exc'].add((PRECEDENCE,r,info['support']))

            if r[0] in S2 and r[1] in S1:
                exclude.append('exc')
                exclude_dic['exc'].add((PRECEDENCE,r,info['support']))
                exclude.append('loop')
                exclude_dic['loop'].add((PRECEDENCE,r,info['support']))
                exclude.append('seq')
                exclude_dic['seq'].add((PRECEDENCE,r,info['support']))
                exclude.append('par')
                exclude_dic['par'].add((PRECEDENCE,r,info['support']))

                block = True
        else:
            if r[0] in S1 and r[1] in S2:
                exclude.append('par')
                exclude_dic['par'].add((PRECEDENCE,r,info['support']))
                exclude.append('exc')
                exclude_dic['exc'].add((PRECEDENCE,r,info['support']))

            if r[0] in S2 and r[1] in S1:
                exclude.append('exc')
                exclude_dic['exc'].add((PRECEDENCE,r,info['support']))
                exclude.append('loop')
                exclude_dic['loop'].add((PRECEDENCE,r,info['support']))
                exclude.append('seq')
                exclude_dic['seq'].add((PRECEDENCE,r,info['support']))
                exclude.append('par')
                exclude_dic['par'].add((PRECEDENCE,r,info['support']))

    return set(exclude), block,exclude_dic

