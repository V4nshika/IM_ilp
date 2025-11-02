from collections import Counter

def split(cut_type, cut, l):
    if not isinstance(l, dict):
        freq_dict = Counter()
        for trace in l:
            # Extract activity names if events are dictionaries
            if trace and isinstance(trace[0], dict):
                activities = tuple(event['concept:name'] for event in trace)
            else:
                activities = tuple(trace)  # Assume trace is already activities
            freq_dict[activities] += 1
        l = freq_dict

    LA = Counter()
    LB = Counter()


    if cut_type == 'seq':
        for tr in l:
            if tr:
                cost = [sum(ev in cut[1] for ev in tr[:i]) + sum(ev in cut[0] for ev in tr[i:]) for i in range(len(tr) + 1)]
                split_point = cost.index(min(cost))
                trace_A = tuple(x for x in tr[0:split_point] if x in cut[0])
                trace_B = tuple(x for x in tr[split_point:] if x in cut[1])
                LA[trace_A] += l[tr]
                LB[trace_B] += l[tr]
            # if not tr:
            #     LA[()] += l[tr]
            #     LB[()] += l[tr]


    elif cut_type == 'exc':
        for tr in l:
            if tr:
                A_count = sum(1 for ev in tr if ev in cut[0])
                B_count = len(tr) - A_count
                target = cut[0] if A_count >= B_count else cut[1]
                T = tuple(ev for ev in tr if ev in target)
                (LA if A_count >= B_count else LB)[T] += l[tr]
            # if not tr:
            #     LA[()] += l[tr]
            #     LB[()] += l[tr]

    elif cut_type == 'exc2' or cut_type == 'exc_tau':
        for tr in l:
            if not tr:
                LB[()] = l[tr]
                continue
            A_count = sum(1 for ev in tr if ev in cut[0])
            B_count = len(tr) - A_count
            target = cut[0] if A_count >= B_count else cut[1]
            T = tuple(ev for ev in tr if ev in target)
            (LA if A_count >= B_count else LB)[T] += l[tr]



    elif cut_type == 'par':
        for tr in l:
            if tr:
                T1 = tuple([ev for ev in tr if ev in cut[0]])
                T2 = tuple([ev for ev in tr if ev in cut[1]])
                LA[T1] += l[tr]
                LB[T2] += l[tr]
            # if not tr:
            #     LA[()] += l[tr]
            #     LB[()] += l[tr]

    elif cut_type == 'loop':
        for tr in l:
            if tr:
                flagA = tr[0] in cut[0]
                # if not flagA:
                #     LA[()] += l[tr]
                T = ()
                for ind, ev in enumerate(tr):
                    T = T + (ev,)
                    if ind < len(tr) - 1:
                        if flagA and tr[ind + 1] in cut[1]:
                            LA[T] += l[tr]
                            T = ()
                            flagA = False
                        elif not flagA and tr[ind + 1] in cut[0]:
                            LB[T] += l[tr]
                            T = ()
                            flagA = True
                    else:
                        if flagA:
                            LA[T] += l[tr]
                        else:
                            # LA[()] += l[tr]
                            LB[T] += l[tr]
            # if not tr:
            #     LA[()] += l[tr]


    elif cut_type == 'loop1':
        for tr in l:
            if tr:
                if len(tr) == 1:
                    LA[tr] += l[tr]
                else:
                    LA[(tr[0]),] += l[tr]
                    for _ in tr[1:]:
                        LB[()] += l[tr]
                        LA[(tr[0]),] += l[tr]
            # if not tr:
            #     LA[()] += l[tr]



    elif cut_type == 'loop_tau':
        # from collections import Counter
        st_acts = cut[0]
        en_acts = cut[1]
        for tr in l:
            if tr:
                T = ()
                for i, ev in enumerate(tr):
                    T = T + (ev,)
                    if i < len(tr) - 1 and tr[i] in en_acts and tr[i + 1] in st_acts:
                        # if max(Counter(T+(tr[i+1],)).values())>1:
                        # if tr[i] in en_acts and tr[i + 1] in st_acts:
                        LA[T] += l[tr]
                        T = ()
                        LB[()] += l[tr]

                if T:
                    LA[T] += l[tr]
            # if not tr:
            #     LA[()] += l[tr]

    return LA,LB  # new_logs is a list that contains logs



