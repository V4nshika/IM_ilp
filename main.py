from IM_ilp import recursion_exp as recursion
from IM_ilp.post_recursion import save_multiple_petri_nets_to_pdf
import time

log_path = 'sample_logs/BPI_2012_A_O.xes' 

net_03, im_03, fm_03 = recursion.apply(log, sup=0.3, print_time_taken=True)

net_06, im_06, fm_06 = recursion.apply(log, sup=0.6, print_time_taken=True)

petri_nets = [
    (net_exp_03, im_exp_03, fm_exp_03),
    (net_exp_06, im_exp_06, fm_exp_06)
]

save_multiple_petri_nets_to_pdf(petri_nets, "IM_ilp/models_2017.pdf")
