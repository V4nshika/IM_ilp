from collections import Counter

def split(cut_type, cut, l):
    # pre-convert cuts to sets for O(1) lookups
    S0 = set(cut[0]) if cut and len(cut) > 0 else set()
    S1 = set(cut[1]) if cut and len(cut) > 1 else set()

    # Handle list-to-counter conversion
    if not isinstance(l, dict):
        freq_dict = Counter()
        for trace in l:
            if trace and isinstance(trace[0], dict):
                activities = tuple(event['concept:name'] for event in trace)
            else:
                activities = tuple(trace)
            freq_dict[activities] += 1
        l = freq_dict

    LA = Counter()
    LB = Counter()

    if cut_type == 'seq':
        
        for tr, freq in l.items(): 
            if tr:
                
                # Initial state: Split at 0. All events are on the Right.
                # Cost = count of events in trace that belong to Left (S0).
                current_cost = sum(1 for ev in tr if ev in S0)
                min_cost = current_cost
                split_point = 0
                
                # Slide split point from left to right
                for i, ev in enumerate(tr):
                    # Moving ev from Right to Left side of the split
                    if ev in S0:
                        current_cost -= 1 # It was an error on Right, now correct on Left
                    elif ev in S1:
                        current_cost += 1 # It was correct on Right, now error on Left
                    
                    if current_cost < min_cost:
                        min_cost = current_cost
                        split_point = i + 1
                
                trace_A = tuple(x for x in tr[0:split_point] if x in S0)
                trace_B = tuple(x for x in tr[split_point:] if x in S1)
                LA[trace_A] += freq
                LB[trace_B] += freq

    elif cut_type == 'exc':
        for tr, freq in l.items():
            if tr:
                # Use sets S0/S1 for faster lookup
                A_count = sum(1 for ev in tr if ev in S0)
                B_count = len(tr) - A_count
                target = S0 if A_count >= B_count else S1
                # Generator expression directly to tuple (no intermediate list)
                T = tuple(ev for ev in tr if ev in target)
                (LA if A_count >= B_count else LB)[T] += freq

    elif cut_type == 'exc2' or cut_type == 'exc_tau':
        for tr, freq in l.items():
            if not tr:
                LB[()] = freq
                continue
            A_count = sum(1 for ev in tr if ev in S0)
            B_count = len(tr) - A_count
            target = S0 if A_count >= B_count else S1
            T = tuple(ev for ev in tr if ev in target)
            (LA if A_count >= B_count else LB)[T] += freq

    elif cut_type == 'par':
        for tr, freq in l.items():
            if tr:
                # Removed square brackets [] inside tuple() to avoid creating temp list
                T1 = tuple(ev for ev in tr if ev in S0)
                T2 = tuple(ev for ev in tr if ev in S1)
                LA[T1] += freq
                LB[T2] += freq

    elif cut_type == 'loop':
        for tr, freq in l.items():
            if tr:
                flagA = tr[0] in S0
                
                current_trace = [] 
                
                for i, ev in enumerate(tr):
                    current_trace.append(ev)
                    
                    # Check next event (boundary check included)
                    if i < len(tr) - 1:
                        if flagA and tr[i + 1] in S1:
                            LA[tuple(current_trace)] += freq
                            current_trace = []
                            flagA = False
                        elif not flagA and tr[i + 1] in S0:
                            LB[tuple(current_trace)] += freq
                            current_trace = []
                            flagA = True
                    else:
                        # End of trace
                        if flagA:
                            LA[tuple(current_trace)] += freq
                        else:
                            LB[tuple(current_trace)] += freq

    elif cut_type == 'loop1':
        for tr, freq in l.items():
            if tr:
                if len(tr) == 1:
                    LA[tr] += freq
                else:
                    # Pre-calculate tuple to avoid repeated tuple creation
                    first_element = (tr[0],)
                    LA[first_element] += freq
                    # Optimized loop simply by adding freq * count
                    # instead of looping N times
                    count_remaining = len(tr) - 1
                    if count_remaining > 0:
                        LB[()] += (freq * count_remaining)
                        LA[first_element] += (freq * count_remaining)

    elif cut_type == 'loop_tau':
        st_acts = S0
        en_acts = S1
        for tr, freq in l.items():
            if tr:
                
                current_trace = []
                for i, ev in enumerate(tr):
                    current_trace.append(ev)
                    if i < len(tr) - 1 and tr[i] in en_acts and tr[i + 1] in st_acts:
                        LA[tuple(current_trace)] += freq
                        current_trace = []
                        LB[()] += freq

                if current_trace:
                    LA[tuple(current_trace)] += freq

    return LA, LB