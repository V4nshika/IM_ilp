from local_pm4py.functions.functions import n_edges

def check_base_case(netP, netM, sup_thr, ratio,self):
    activitiesP = set(netP.nodes) - {'start', 'end'}
    activitiesM = set(netM.nodes) - {'start', 'end'}

    if len(activitiesP)>1:
        base_check = False
        cut = "not_base"
    elif len(activitiesP)==0:
        base_check = True
        self.detected_cut = 'empty_log'
        cut = ('none', 'empty_log', 'none', 'none')
    else:
        base_check = True
        act = activitiesP.pop()
        cP = max(0, sup_thr * n_edges(netP,{'start'},{act, 'end'})-n_edges(netP,{'start'}, {'end'}))
        cM = max(0,sup_thr * n_edges(netM,{'start'},{act, 'end'})-n_edges(netM,{'start'}, {'end'}))
        cost_single_exc =  cP - ratio * self.size_adj * cM
        if (cost_single_exc <= 0) and n_edges(netP,{'start'}, {'end'}) > 0:
            cut = ((({act}, set()), 'exc2', cP, cM,cost_single_exc,{str(act): {'missing': cP, 'deviating': 0}},{str(act): {'missing': cM, 'deviating': 0}}))
        else:
            clP = max(0,sup_thr* ((n_edges(netP,{'start',act},{act}))/2)-n_edges(netP,{act},{act}))

            if len(activitiesM) > 0:
                clM = max(0, sup_thr * ((n_edges(netM, {'start', act}, {act})) / 2) - n_edges(netM, {act}, {act}))
            else:
                clM = max(0, sup_thr * ((n_edges(netM, {'start', act}, {act})) / 2))
            cost_single_loop = clP - ratio * self.size_adj * clM
            if (cost_single_loop <= 0) and n_edges(netP,{act},{act})>0:
                cut = (({act}, set()), 'loop1', 'none', 'none')
            else:
                # single activity
                self.detected_cut = 'single_activity'
                cut =((
                ({act}, set()), 'single_activity', n_edges(netP,{'start'}, {'end'}), n_edges(netM,{'start'}, {'end'}),
                n_edges(netP,{'start'}, {'end'}) - ratio * self.size_adj * n_edges(netM,{'start'}, {'end'}),
                {str(act): {'deviating': n_edges(netP,{'start'}, {'end'}), 'missing': 0}},
                {str(act): {'deviating': n_edges(netM,{'start'}, {'end'}), 'missing': 0}}))
    return base_check, cut

