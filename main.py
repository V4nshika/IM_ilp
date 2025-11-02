from pm4py.objects.log.importer.xes import importer as xes_importer
import pm4py
from IM_ilp.recursion_exp import recursion_full as recursion_full, to_pm4py_tree
from IM_ilp.post_recursion import save_multiple_petri_nets_to_pdf
import time

log_path = 'BPI_Challenge/BPI_Challenge_2017.xes'
log = xes_importer.apply(log_path)

process_tree_03 = recursion_full(log, sup =0.3)

process_tree_06 = recursion_full(log, sup =0.6)

tree_data_pm4py_03 = to_pm4py_tree(process_tree_03)

tree_data_pm4py_06 = to_pm4py_tree(process_tree_06)

net_03, im_03, fm_03 = pm4py.objects.conversion.process_tree.converter.apply(tree_data_pm4py_03)

net_06, im_06, fm_06 = pm4py.objects.conversion.process_tree.converter.apply(tree_data_pm4py_06)

petri_nets = [
    (net_exp_03, im_exp_03, fm_exp_03),
    (net_exp_06, im_exp_06, fm_exp_06)
]

save_multiple_petri_nets_to_pdf(petri_nets, "IM_ilp/models_2017.pdf")
