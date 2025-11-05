from pm4py.objects.log.importer.xes import importer as xes_importer
import pm4py
from IM_ilp import recursion_exp as recursion 
from IM_ilp.recursion_exp import to_pm4py_tree
from IM_ilp.post_recursion import save_multiple_petri_nets_to_pdf
import time

log_path = 'BPI_Challenge_2017.xes' 
log = xes_importer.apply(log_path)

net_03, im_03, fm_03, time_taken_03 = recursion.apply(log, sup=0.3)

net_06, im_06, fm_06, time_taken_06 = recursion.apply(log, sup=0.6)

petri_nets = [
    (net_exp_03, im_exp_03, fm_exp_03),
    (net_exp_06, im_exp_06, fm_exp_06)
]

save_multiple_petri_nets_to_pdf(petri_nets, "IM_ilp/models_2017.pdf")
