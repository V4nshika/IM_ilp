from pm4py.objects.log.importer.xes import importer as xes_importer
import pm4py
from IM_ilp.recursion_exp import recursion_full as recursion_full, to_pm4py_tree
import time

log_path = 'BPI_Challenge/BPI_Challenge_2017.xes'
log = xes_importer.apply(log_path)

process_tree = recursion_full(log, sup =0.2)

tree_data_pm4py = to_pm4py_tree(process_tree)

net, im, fm = pm4py.objects.conversion.process_tree.converter.apply(tree_data_pm4py)

