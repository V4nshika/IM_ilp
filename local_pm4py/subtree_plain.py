import local_pm4py.split_functions.split as split
from local_pm4py.candidate_search.search import find_possible_partitions
from local_pm4py.base_case.check import check_base_case
from local_pm4py.cut_quality.cost_functions import cost_functions
from local_pm4py.functions.functions import generate_nx_graph_from_log, generate_nx_indirect_graph_from_log, read_append_write_json
from local_pm4py.cut_quality.cost_functions.cost_functions import overal_cost

class SubtreePlain(object):
    def __init__(self, logp,logm, rec_depth, noise_threshold= None,
                   parameters=None, sup= None, ratio = None, size_par = None, rules = None):

        nt = 0.0 * logp.total()
        self.sorted_cuts = None
        self.rec_depth = rec_depth
        self.noise_threshold = noise_threshold
        self.log = logp
        self.netP = generate_nx_graph_from_log(self.log,nt)
        if self.log.total()==0:
            self.activitiesP = set()
            self.start_activitiesP = {}
            self.end_activitiesP = {}
        else:
            self.activitiesP = set(self.netP.nodes()) - {'start', 'end'}
            self.start_activities = {x:self.netP.get_edge_data('start', x)['weight'] for x in self.netP.successors('start') if x!='end'}
            self.end_activities = {x: self.netP.get_edge_data(x, 'end')['weight'] for x in self.netP.predecessors('end') if x!='start'}
        self.logM = logm
        self.netM = generate_nx_graph_from_log(self.logM,nt)
        if self.logM.total()==0:
            self.activitiesM = set()
            self.start_activitiesM = {}
            self.end_activitiesM = {}
            self.size_adj = 1
        else:
            self.activitiesM = set(self.netM.nodes()) - {'start', 'end'}
            self.start_activitiesM = {x: self.netM.get_edge_data('start', x)['weight'] for x in
                                      self.netM.successors('start') if x != 'end'}
            self.end_activitiesM = {x: self.netM.get_edge_data(x, 'end')['weight'] for x in
                                    self.netM.predecessors('end') if x != 'start'}
            self.size_adj = self.log.total()/self.logM.total()
        self.original_log = logp
        self.parameters = parameters
        self.detected_cut = None
        self.children = []
        self.detect_cut(parameters=parameters, sup= sup, ratio = ratio, size_par = size_par, rules = rules)


    def detect_cut(self, parameters=None, sup= None, ratio = None, size_par = None, rules = None):

        if parameters is None:
            parameters = {}

        # check base cases:
        isbase, cut = check_base_case(self.netP, self.netM, sup, ratio, self)

        if isbase==False:
            fP = generate_nx_indirect_graph_from_log(self.log)
            fM = generate_nx_indirect_graph_from_log(self.logM)

            possible_partitions, reserve = find_possible_partitions(self.netP, rules, self.start_activities, self.end_activities,fP)
            if len(possible_partitions)==0 or (len(possible_partitions)==1 and possible_partitions[0][2]=={'loop_tau'}):
                bbb = min(reserve, key=lambda x: x[2])
                possible_partitions = [(bbb[0][0],bbb[0][1],bbb[1]) for item in reserve if item[2]==bbb[2]]
                print('no cut available')

            cut = []

            ratio_backup = ratio

            for pp in possible_partitions:
                A = pp[0] - {'start', 'end'}
                B = pp[1] - {'start', 'end'}

                start_A_P = set(self.start_activities.keys()) & A
                end_A_P = set(self.end_activities.keys()) & A
                input_B_P = set([x[1] for x in self.netP.edges if ((x[0] not in B) and (x[1] in B))])
                output_B_P = set([x[0] for x in self.netP.edges if ((x[0] in B) and (x[1] not in B))])

                start_A_M = set(self.start_activitiesM.keys()) & A
                end_A_M = set(self.end_activitiesM.keys()) & A
                input_B_M = set([x[1] for x in self.netM.edges if ((x[0] not in B) and (x[1] in B))])
                output_B_M = set([x[0] for x in self.netM.edges if ((x[0] in B) and (x[1] not in B))])

                type = pp[2]
                if self.logM.total()==0:
                    ratio = 0
                else:
                    ratio = ratio_backup

                #####################################################################
                # seq check
                if "seq" in type:
                    cost_seq_P_dict = cost_functions.cost_seq(self.netP, A, B, sup, fP)
                    cost_seq_P = sum(x['missing']+x['deviating'] for x in cost_seq_P_dict.values())
                    cost_seq_M_dict = cost_functions.cost_seq(self.netM, A.intersection(self.activitiesM), B.intersection(self.activitiesM), sup, fM)
                    cost_seq_M = sum(x['missing']+x['deviating'] for x in cost_seq_M_dict.values())
                    cut.append(((A, B), 'seq', cost_seq_P, cost_seq_M, overal_cost(cost_seq_P,cost_seq_M,ratio,self.size_adj),cost_seq_P_dict,cost_seq_M_dict))
                #####################################################################
                # xor check
                if "exc" in type:
                    cost_exc_P_dict = cost_functions.cost_exc(self.netP, A, B)
                    cost_exc_P = sum(x['missing']+x['deviating'] for x in cost_exc_P_dict.values())
                    cost_exc_M_dict = cost_functions.cost_exc(self.netM, A.intersection(self.activitiesM), B.intersection(self.activitiesM))
                    cost_exc_M = sum(x['missing']+x['deviating'] for x in cost_exc_M_dict.values())
                    cut.append(((A, B), 'exc', cost_exc_P, cost_exc_M, overal_cost(cost_exc_P,cost_exc_M,ratio,self.size_adj),cost_exc_P_dict,cost_exc_M_dict))
                #####################################################################
                # xor-tau check
                if "exc_tau" in type:
                    cost_exc_tau_P_dict = cost_functions.cost_exc_tau(self.netP, sup)
                    cost_exc_tau_P = sum(x['missing']+x['deviating'] for x in cost_exc_tau_P_dict.values())
                    cost_exc_tau_M_dict = cost_functions.cost_exc_tau(self.netM, sup)
                    cost_exc_tau_M = sum(x['missing']+x['deviating'] for x in cost_exc_tau_M_dict.values())
                    cut.append(((A.union(B), set()), 'exc_tau',cost_exc_tau_P , cost_exc_tau_M, overal_cost(cost_exc_tau_P,cost_exc_tau_M,ratio,self.size_adj),cost_exc_tau_P_dict,cost_exc_tau_M_dict))
                #####################################################################
                # parallel check
                if "par" in type:
                    cost_par_P_dict = cost_functions.cost_par(self.netP, A, B,sup)
                    cost_par_P = sum(x['missing']+x['deviating'] for x in cost_par_P_dict.values())
                    cost_par_M_dict = cost_functions.cost_par(self.netM, A.intersection(self.activitiesM), B.intersection(self.activitiesM), sup)
                    cost_par_M = sum(x['missing']+x['deviating'] for x in cost_par_M_dict.values())
                    cut.append(((A, B), 'par', cost_par_P, cost_par_M, overal_cost(cost_par_P,cost_par_M,ratio,self.size_adj),cost_par_P_dict,cost_par_M_dict))
                #####################################################################
                # loop check
                if "loop" in type:
                    cost_loop_P_dict = cost_functions.cost_loop(self.netP, A, B, sup, start_A_P, end_A_P, input_B_P,output_B_P,self.start_activities,self.end_activities)
                    cost_loop_P = sum(x['missing']+x['deviating'] for x in cost_loop_P_dict.values())
                    cost_loop_M_dict = cost_functions.cost_loop(self.netM, A, B, sup, start_A_M, end_A_M, input_B_M,output_B_M,self.start_activitiesM,self.end_activitiesM)
                    cost_loop_M = sum(x['missing']+x['deviating'] for x in cost_loop_M_dict.values())
                    cut.append(((A, B), 'loop', cost_loop_P, cost_loop_M, overal_cost(cost_loop_P,cost_loop_M,ratio,self.size_adj),cost_loop_P_dict,cost_loop_M_dict))

                if "loop_tau" in type:
                    cost_loop_P_dict = cost_functions.cost_loop_tau(self.netP, sup, self.start_activities,self.end_activities)
                    cost_loop_P = sum(x['missing']+x['deviating'] for x in cost_loop_P_dict.values())
                    cost_loop_M_dict = cost_functions.cost_loop_tau(self.netM, sup, self.start_activitiesM,self.end_activitiesM)
                    cost_loop_M = sum(x['missing']+x['deviating'] for x in cost_loop_M_dict.values())
                    cut.append(((set(self.start_activities.keys()), set(self.end_activities.keys())), 'loop_tau', cost_loop_P,cost_loop_M, overal_cost(cost_loop_P,cost_loop_M,ratio,self.size_adj), cost_loop_P_dict,cost_loop_M_dict))

            if not cut:
                print("no good cut exists")
            else:
                sorted_cuts = sorted(cut, key=lambda x: (
                x[4], x[2], ['exc_tau', 'exc', 'seq', 'par', 'loop', 'loop_tau'].index(x[1]),
                abs(len(x[0][0]) - len(x[0][1]))))
                self.sorted_cuts = sorted_cuts  # Store sorted_cuts in the class
                cut = sorted_cuts[0]
        # print(cut[:-2])

        map_cut_op = {'par': 'parallel', 'seq': 'sequential', 'exc': 'xor', 'exc_tau':'xor', 'exc2': 'xor',
                      'loop': 'loopCut', 'loop1': 'loopCut', 'loop_tau': 'loopCut'}

        if cut[1] in {'par','seq','exc','exc_tau','exc2','loop','loop_tau', 'single_activity'}:
            # updates = {'cut_type':cut[1],'set1':list(cut[0][0]), 'set2':list(cut[0][1]), 'cost+:':cut[2], 'cost-':ratio*self.size_adj*cut[3], 'contribusions+':{x:cut[5][x] for x in cut[5] if cut[5][x]>0} , 'contributions-':{x:(ratio*self.size_adj*cut[6][x]) for x in cut[6] if cut[6][x]>0},'overal_cost':cut[4]}
            updates = {'cut_type': cut[1], 'set1': list(cut[0][0]), 'set2': list(cut[0][1]), 'cost+:': cut[2],
                       'cost-': ratio * self.size_adj * cut[3],
                       'contribusions+': {x: {'deviating': cut[5][x]['deviating'], 'missing': cut[5][x]['missing']} for
                                          x in cut[5] if (cut[5][x]['missing'] > 0 or cut[5][x]['deviating'])},
                       'contributions-': {x: {'deviating': ratio * self.size_adj * cut[6][x]['deviating'],
                                              'missing': ratio * self.size_adj * cut[6][x]['missing']} for x in cut[6]
                                          if (cut[6][x]['missing'] > 0 or cut[6][x]['deviating'])},
                       'overal_cost': cut[4]}
            file_path = f'E:\\PADS\Projects\\IMbi_paper_revision\\experiments\\models\\data_s{int(10*sup)}_r{int(10*ratio)}.json'
            # read_append_write_json(file_path, updates)


        if cut[1] in map_cut_op.keys():
            self.detected_cut = map_cut_op[cut[1]]
            LAP, LBP = split.split(cut[1], [cut[0][0], cut[0][1]], self.log)
            LAM, LBM = split.split(cut[1], [cut[0][0], cut[0][1]], self.logM)
            new_logs = [[LAP, LAM], [LBP, LBM]]
            for l in new_logs:
                self.children.append(
                    SubtreePlain(l[0], l[1],
                                 self.rec_depth + 1,
                                 noise_threshold=self.noise_threshold,
                                 parameters=parameters, sup=sup, ratio=ratio, size_par=size_par, rules=rules))
        elif cut[1]!='single_activity' and cut[1]!='empty_log':
            print('It should not happen, if you see this error there could be a bug in your code')
