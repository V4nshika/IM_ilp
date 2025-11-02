"""
This file contains the core elements to reproduce the paper
    "Mining Behavioral Patterns for Conformance Diagnostics"

Due to licensing and intellectual property, many parts are left out, including the User Interface.
Also, less sophisticated implementations are used, e.g. to compute the Petri net's reachability graph
or to remove silent transitions (which we do by determinizing the NFA instead of using the Murata rules).
This entails a higher runtime compared to the reported on the paper, especially for the "scalable" templates.

We are not releasing the code to perform Petri net reductions,
    which means that the check for Gamma-invariant constraints will take longer.

Everything is packed in a single file to make it easier to run. At some points the implementation feels
verbose or unsophisticated. This is intentional as we wanted to make the code straightforward to read.
"""
import dataclasses
import json
import glob
import time

from collections import deque, defaultdict, Counter
from copy import deepcopy
from functools import reduce
from itertools import combinations, permutations, chain, product

from pathlib import Path
from typing import Dict, FrozenSet, Generator, List, Optional, Set, Tuple, NamedTuple, Iterable

import automata
import networkx
from automata.fa.dfa import DFA
from automata.fa.nfa import NFA


# Set this to True just for running the experiments, else keep it False while developing/testing
MAX_PERFORMANCE = True
if MAX_PERFORMANCE:
    automata.base.automaton.global_config.should_validate_automata = False
    automata.base.automaton.global_config.allow_mutable_automata = True

OMEGA_LABEL = 'O'


@dataclasses.dataclass
class Template:
    name: str
    parameter_count: int
    activation_templates: Dict[str, Set[str]]
    target_activities: Set[str]
    placeholder_equivalences: List[Set[str]]
    dfa: DFA
    vacuity_condition: DFA
    type_priority: int
    subsumption_hierarchy: int

    def __post_init__(self):
        # We must serialize the placeholders to
        self._invariant_labels = get_invariant_labels(self.dfa)
        self._invariant_labels_vacuity_condition = get_invariant_labels(self.vacuity_condition)
        self._placeholders = [symbol for symbol in self.dfa.input_symbols if symbol != OMEGA_LABEL]
        self._placeholders.sort()
        self._placeholder_offsets = {
            placeholder: i
            for i, equivalence_class in enumerate(self.placeholder_equivalences)
            for placeholder in equivalence_class if placeholder != OMEGA_LABEL
        }
        self._relevant_labels = [symbol for symbol in self.dfa.input_symbols if symbol not in self._invariant_labels]

        # We require the vacuity condition to contain only relevant labels
        assert not get_invariant_labels(self.vacuity_condition)

        # Fixup for the Absense constraint
        if not self._relevant_labels:
            self._relevant_labels = list(self.dfa.input_symbols)

    def __call__(self, *args, **kwargs):
        parameter_map = {k: v for k, v in zip(self._placeholders, args)}
        assert len(parameter_map) == self.parameter_count, \
            f"Rule requires {self.parameter_count} DISTINCT parameters"

        if self._is_repeated_constraint(*args):
            return None

        activations = inject_activation_parameters(self.activation_templates, parameter_map)
        target_activities = set(parameter_map[label] for label in self.target_activities)
        argument_str = ', '.join(args)

        constraint_traits = ConstraintTraits(
            name=f"{self.name}({argument_str})",
            type_priority=self.type_priority,
            subsumption_hierarchy=self.subsumption_hierarchy,
            activations=activations,
            tgt_activities=target_activities,
            is_omega_invariant=OMEGA_LABEL in self._invariant_labels,
            max_label_distance=kwargs["max_label_distance"],
        )

        constraint_dfa = inject_dfa_parameters(self.dfa, parameter_map)
        constraint_vacuity_condition = inject_dfa_parameters(self.vacuity_condition, parameter_map)
        relevant_labels = frozenset(parameter_map[label] for label in self._relevant_labels)

        return Constraint(
            constraint_traits,
            constraint_dfa,
            constraint_vacuity_condition,
            relevant_labels,
        )

    def _is_repeated_constraint(self, *args):
        # placeholder_equivalences groups placeholders that are "equivalent",
        #   i.e. swapping their assigned parameters returns a DFA of the same language
        # If that is the case, then we only need to instantiate these parameters once
        # This code establishes that the parameters must be assigned in increasing lexographic order
        # If the assignment violates it, then it's a repetition
        last_assigned_arg = {}
        for placeholder, arg in zip(self._placeholders, args):
            equivalence_group = self._placeholder_offsets[placeholder]
            if equivalence_group in last_assigned_arg and arg < last_assigned_arg[equivalence_group]:
                # If we are trying to assign a label to an equivalence group that decreases, we have repetition
                return True
            else:
                last_assigned_arg[equivalence_group] = arg

        return False


def get_invariant_labels(dfa: DFA) -> Set[str]:
    """Returns the maximal set of labels Gamma s.t. the dfa is Gamma-invariant"""
    assert not dfa.allow_partial, "Invariant label computation not possible on partial DFAs"

    self_loops = (
        {label for label, tgt_state in transitions.items() if tgt_state == src_state}
        for src_state, transitions in dfa.transitions.items()
    )
    return set.intersection(*self_loops)


def inject_dfa_parameters(dfa: DFA, label_map: Dict[str, str]):
    """ Replace the DFA's input symbols by new ones provided by label_map """
    # Map the omega labels to itself
    label_map[OMEGA_LABEL] = OMEGA_LABEL

    new_func = {}
    for src_state, trans_func in dfa.transitions.items():
        mapped_func = {label_map[label]: tgt_state
                       for label, tgt_state in trans_func.items()}
        new_func[src_state] = mapped_func

    # The alphabet must contain only the labels that got effectively replaced
    alphabet = set(label for trans_func in new_func.values() for label in trans_func.keys())

    return DFA(
        states=deepcopy(dfa.states),
        input_symbols=alphabet,
        transitions=new_func,
        initial_state=deepcopy(dfa.initial_state),
        final_states=deepcopy(dfa.final_states),
    )


def inject_activation_parameters(
        activation_templates: Dict[str, Set[str]], label_map: Dict[str, str]
) -> List[str]:
    return [activation.format(*(label_map[label] for label in labels))
            for activation, labels in activation_templates.items()]


@dataclasses.dataclass
class ConstraintTraits:
    name: str
    # These are used to order the constraints later
    type_priority: int
    subsumption_hierarchy: int
    activations: List[str]
    tgt_activities: Set[str]
    is_omega_invariant: bool
    max_label_distance: int


@dataclasses.dataclass
class Constraint:
    traits: ConstraintTraits
    dfa: DFA
    vacuity_condition: DFA
    # We could compute this for each newly created constraint,
    #   but that would increase the total runtime, so we take it from the template
    relevant_labels: FrozenSet[str]


@dataclasses.dataclass
class Repository:
    name: str
    max_k: int
    templates: List[Template]


REPOSITORIES_FOLDER = "repositories"
DATASETS_FOLDER = "datasets"

NFA_INVISIBLE_LABEL = ''


def main():
    repositories = load_repositories(REPOSITORIES_FOLDER)
    datasets = load_datasets(DATASETS_FOLDER)
    repositories.sort(key=lambda x: x.name)
    datasets.sort(key=lambda x: x.name)

    for dataset in datasets:
        for repository in repositories:
            run_experiment(dataset, repository)


def run_experiment(dataset, repository):
    start = time.time()
    print(dataset.name, repository.name)
    valid_constraints = _get_dataset(dataset, repository)
    end_valid = time.time()
    ordered_constraints = constraint_ordering_hard_coded(valid_constraints)
    ordered_dfas = [ordered_constraint.dfa for ordered_constraint in ordered_constraints]
    max_m, max_k = 2, repository.max_k
    minimal_dfas = minimal_dfas_approximated(ordered_dfas, max_m, max_k)
    minimal_constraints = [
        constraint for constraint, is_minimal in zip(ordered_constraints, minimal_dfas) if is_minimal
    ]
    print(len(minimal_constraints))
    violating_constraints = verify_log(dataset.aligned_log, minimal_constraints)
    print("VARIANTS VIOLATING AT LEAST ONE", len([var for var in violating_constraints if var]))
    print(end_valid - start)
    print(time.time() - end_valid)

    return minimal_constraints, violating_constraints


# ======================== LOAD DATA ========================


def load_repositories(repositories_folder: str) -> List[Repository]:
    repositories_paths = [Path(file_path) for file_path in glob.glob(f"{repositories_folder}/*.json")]
    print(repositories_paths)
    return [load_repository(repository_path) for repository_path in repositories_paths]


def load_repository(repository_path: Path) -> Repository:
    # It is a bit cumbersome to perform this conversion, but we do not want to use pickle or any exotic solution
    #   The problem is that JSON does not support sets and integer keys in a dictionary
    with open(repository_path, 'r') as repository_file:
        repository_data = json.load(repository_file)

    return Repository(
        name=repository_data["repository"]["name"],
        max_k=repository_data["repository"]["max_k"],
        templates=[load_template(template_data) for template_data in repository_data["templates"]]
    )


def load_template(template_data: Dict) -> Template:
    return Template(
        name=template_data["name"],
        parameter_count=template_data["parameter_count"],
        activation_templates={
            activation: {*tgts} for activation, tgts in template_data["activation_templates"].items()
        },
        target_activities={*template_data["target_activities"]},
        placeholder_equivalences=[
            set(equivalence_class) for equivalence_class in template_data["placeholder_equivalences"]
        ],
        dfa=load_dfa(template_data["dfa"]),
        vacuity_condition=load_dfa(template_data["vacuity_condition"]),
        type_priority=template_data["type_priority"],
        subsumption_hierarchy=template_data["subsumption_hierarchy"],

    )


def load_dfa(dfa_data: Dict) -> DFA:
    return DFA(
        states={int(state) for state in dfa_data["states"]},
        input_symbols={*dfa_data["input_symbols"]},
        transitions={int(src_state): {label: int(tgt_state) for label, tgt_state in lookup.items()}
                     for src_state, lookup in dfa_data["transitions"].items()},
        initial_state=int(dfa_data["initial_state"]),
        final_states={int(final_state) for final_state in dfa_data["final_states"]},
        allow_partial=dfa_data["allow_partial"],
    )


# Since it is unclear whether using PM4Py would force me to distribute this code as GPL,
#   we use our own implementation of Petri nets
class PetriNet:
    __slots__ = ('places', 'transitions')

    class TransitionData(NamedTuple):
        id: str
        label: Optional[str]
        pre_set: Set[str]
        post_set: Set[str]

    # We use sets to represent markings, hence we assume the net to be *safe*
    places: Set[str]
    # transition_id -> (label, in_places, out_places)
    transitions: Dict[str, TransitionData]

    def __init__(self, places: Set[str], transitions: Dict[str, TransitionData]):
        self.places = places
        self.transitions = transitions

        assert places.isdisjoint(transitions.keys()), "A transition and a place have the same id"
        assert all(transition.pre_set.issubset(places) for transition in transitions.values()), \
            "Graph is not bipartite or refers to non-existing nodes"
        assert all(transition.post_set.issubset(places) for transition in transitions.values()), \
            "Graph is not bipartite or refers to non-existing nodes"

    def transition_to_label(self, transition_id: str) -> Optional[str]:
        return self.transitions[transition_id].label

    def next_markings(self, marking: FrozenSet[str]) -> Generator[Tuple[str, FrozenSet[str]], None, None]:
        for transition in self.transitions.values():
            if transition.pre_set.issubset(marking):
                yield transition.id, marking.difference(transition.pre_set).union(transition.post_set)


@dataclasses.dataclass
class WFNet:
    net: PetriNet
    im: FrozenSet[str]
    fm: FrozenSet[str]


@dataclasses.dataclass
class AlignmentMove:
    move_on_log: Optional[str]
    # Transition ID (or None if log move)
    move_on_model: Optional[str]
    # Transition Label (or None if invisible move).
    # TODO We could get this information from the Petri net
    model_label: Optional[str]

    @property
    def is_log_move(self) -> bool:
        return self.move_on_model is None

    @property
    def is_model_move(self) -> bool:
        return self.move_on_log is None

    @property
    def is_sync_move(self) -> bool:
        return self.move_on_log is not None and self.move_on_model is not None

    @property
    def is_invisible_move(self) -> bool:
        return self.move_on_model is not None and self.model_label is None


Alignment = List[AlignmentMove]


@dataclasses.dataclass
class AlignedLog:
    variants: List[Tuple[Alignment, int]]


@dataclasses.dataclass
class Dataset:
    name: str
    description: str
    wf_net: WFNet
    aligned_log: AlignedLog


# TODO delete me
def _get_dataset(dataset: Dataset, repository: Repository):
    instantiated_constraints = instantiate_constraints(repository.templates, dataset.wf_net)
    print(len(instantiated_constraints))
    valid_constraints = check_valid_constraints(instantiated_constraints, dataset.wf_net)
    print(len(valid_constraints))
    return valid_constraints


def load_datasets(datasets_folder: str) -> List[Dataset]:
    datasets_paths = [Path(file_path) for file_path in glob.glob(f"{datasets_folder}/*.json")]
    print(datasets_paths)
    return [load_dataset(dataset_path) for dataset_path in datasets_paths]


def load_dataset(dataset_path: Path) -> Dataset:
    with open(dataset_path, 'r') as dataset_file:
        dataset_data = json.load(dataset_file)

    return Dataset(
        name=dataset_data["name"],
        description=dataset_data["description"],
        wf_net=load_wf_net(dataset_data["wf_net"]),
        aligned_log=load_aligned_log(dataset_data["aligned_log"]),
    )


def load_wf_net(wf_net_data: Dict) -> WFNet:
    return WFNet(
        net=load_petri_net(wf_net_data["net"]),
        im=frozenset(wf_net_data["im"]),
        fm=frozenset(wf_net_data["fm"]),
    )


def load_petri_net(petri_net_data: Dict) -> PetriNet:
    return PetriNet(
        places={*petri_net_data["places"]},
        transitions={
            transition_id: PetriNet.TransitionData(transition_id, label, {*in_places}, {*out_places})
            for transition_id, (label, in_places, out_places) in petri_net_data["transitions"].items()
        },
    )


def load_aligned_log(aligned_log_data: Dict) -> AlignedLog:
    return AlignedLog(
        variants=[(
            [AlignmentMove(move_on_log=move_on_log, move_on_model=move_on_model, model_label=model_label)
             for move_on_log, move_on_model, model_label in aligned_variant],
            freq
        ) for aligned_variant, freq in aligned_log_data]
    )


# ======================== REASONING ========================


def get_behavioral_automaton(wf_net: WFNet, **kwargs) -> DFA:
    return DFA.from_nfa(get_reachability_graph(wf_net, **kwargs))


def get_reachability_graph(wf_net: WFNet, max_number_of_nodes: int = 1_000_000) -> NFA:
    net, im, fm = wf_net.net, wf_net.im, wf_net.fm
    markings_to_explore = deque()
    seen_markings = set()
    explored_markings = set()
    # {src_marking => {enabled_transition => tgt_marking}}
    nfa = {}
    nfa_start_node = im
    nfa_final_nodes = set()

    markings_to_explore.append(im)
    seen_markings.add(im)

    while markings_to_explore:
        to_explore = markings_to_explore.popleft()
        if to_explore in explored_markings:
            continue

        explored_markings.add(to_explore)
        if to_explore not in nfa:
            nfa[to_explore] = {}

        for transition_id, tgt_marking in net.next_markings(to_explore):
            nfa[to_explore][transition_id] = tgt_marking

            if tgt_marking == fm:
                nfa_final_nodes.add(tgt_marking)

            if tgt_marking not in seen_markings:
                markings_to_explore.append(tgt_marking)
                seen_markings.add(tgt_marking)

            if len(seen_markings) > max_number_of_nodes:
                raise RuntimeError(f"Net is too complex. Exceeded the maximum number "
                                   f"of nodes [{max_number_of_nodes}]. "
                                   f"Try to simplify it or increase the limit")

    # Sanity check
    assert explored_markings == seen_markings

    # Build the NFA's transition function, map to states to int, map None label to ''
    #   We could have done it all in the previous loop but like this we decouple the logic
    state_to_id = {marking: i for i, marking in enumerate(seen_markings)}
    transitions = {state_id: {} for state_id in state_to_id.values()}
    for src_state, state_lookup in nfa.items():
        src_state_id = state_to_id[src_state]

        for transition, tgt_state in state_lookup.items():
            net_transition_label = net.transition_to_label(transition)
            transition_str = NFA_INVISIBLE_LABEL if net_transition_label is None else net_transition_label
            tgt_state_id = state_to_id[tgt_state]

            if transition_str not in transitions[src_state_id]:
                transitions[src_state_id][transition_str] = set()

            transitions[src_state_id][transition_str].add(tgt_state_id)

    start_state = state_to_id[nfa_start_node]
    end_states = {state_to_id[end_state] for end_state in nfa_final_nodes}

    return NFA(
        states={state_id for state_id in state_to_id.values()},
        input_symbols={
            label for state_lookup in transitions.values() for label in state_lookup.keys()
            if label != NFA_INVISIBLE_LABEL
        },
        transitions=transitions,
        initial_state=start_state,
        final_states=end_states,
    )


class ProjectedModels:
    """
    The original implementation uses Petri net reductions to reduce the state space
        We do not release the code for that and instead proceed by:
            - Computing the behavioral automaton
            - Mapping activities to the empty transition
            - Determinizing it

    We acknowledge that this approach fails for models where it is unfeasible to compute the behavioral automaton
    It is also more computationally expensive compared to reducing first and then computing the behavioral automaton
    Still, since the experiments use models of moderate size, this has only a limited impact in the results
    """
    def __init__(self, behavioral_automaton: DFA):
        self._dfa = behavioral_automaton
        self._cache: Dict[FrozenSet[str], DFA] = {}

    def project(self, labels_to_keep: FrozenSet[str]) -> DFA:
        if labels_to_keep not in self._cache:
            nfa = project_dfa(self._dfa, labels_to_keep)
            self._cache[labels_to_keep] = DFA.from_nfa(nfa)

        return self._cache[labels_to_keep]


def project_dfa(dfa: DFA, labels_to_keep: FrozenSet[str]) -> NFA:
    new_transitions = {state: {} for state in dfa.states}
    label_map = {label: label if label in labels_to_keep else NFA_INVISIBLE_LABEL
                 for label in dfa.input_symbols}

    for src_state, state_lookup in dfa.transitions.items():
        src_state_new_lookup = new_transitions[src_state]

        for label, tgt_state in state_lookup.items():
            new_label = label_map[label]
            if new_label not in src_state_new_lookup:
                src_state_new_lookup[new_label] = set()
            src_state_new_lookup[new_label].add(tgt_state)

    return NFA(
        states=dfa.states,
        input_symbols=labels_to_keep,
        transitions=new_transitions,
        initial_state=dfa.initial_state,
        final_states=dfa.final_states,
    )


# ======================== DISCOVERY ========================


def get_model_alphabet(model: WFNet) -> Set[str]:
    return {t.label for t in model.net.transitions.values() if t.label is not None}


# def is_valid_constraint(model_dfa: DFA, constraint: Constraint):
#     constraint_dfa = replace_label_with(constraint.dfa, OMEGA_LABEL, [])
#     return model_dfa <= constraint_dfa


def get_max_label_dist(activity_set: frozenset, distances: Dict[Tuple[str, str], int]):
    if len(activity_set) < 2:
        return -1
    return max(distances[(src, tgt)] for src, tgt in permutations(activity_set, 2))


def instantiate_constraints(template_list: List[Template], model: WFNet) -> List[Constraint]:
    # We limit how far parameters might be (this counts as an instantiation strategy)
    model_distances = activity_distances(model)
    alphabet = get_model_alphabet(model)

    # We group templates by their number of parameters
    templates = defaultdict(list)
    for template in template_list:
        templates[template.parameter_count].append(template)

    ret = []
    for k, k_templates in templates.items():
        for combination in combinations(alphabet, k):
            activity_set = frozenset(combination)
            max_label_distance = get_max_label_dist(activity_set, model_distances)

            if max_label_distance > 6:
                continue

            for template in k_templates:
                for permutation in permutations(combination, k):
                    constraint = template(*permutation, max_label_distance=max_label_distance)

                    if constraint is None:
                        # This permutation is uninteresting for this template
                        # (i.e. another permutation produces a constraint with the same language)
                        continue
                    ret.append(constraint)

    return ret


def activity_distances(model: WFNet):
    """ Returns the graph distance between activities in the model """
    net = model.net
    net_graph = get_net_graph(model)
    pairwise_shortest_path = networkx.all_pairs_shortest_path_length(net_graph)
    pairwise_distances = [
        (src_node, tgt_node, distance // 2) for src_node, src_shortest_paths in pairwise_shortest_path
        for tgt_node, distance in src_shortest_paths.items()
    ]

    alphabet = get_model_alphabet(model)
    ret = {pair: -1 for pair in product(alphabet, repeat=2)}
    for src_node, tgt_node, distance in pairwise_distances:
        if src_node in net.places or tgt_node in net.places:
            continue

        src_label = net.transitions[src_node].label
        tgt_label = net.transitions[tgt_node].label
        if src_label is None or tgt_label is None:
            continue

        if ret[(src_label, tgt_label)] == -1:
            ret[(src_label, tgt_label)] = distance

        ret[(src_label, tgt_label)] = min(ret[(src_label, tgt_label)], distance)

    return ret


def get_net_graph(model: WFNet) -> networkx.DiGraph:
    net = model.net
    nodes = chain(net.places, net.transitions)
    place_transition_edges = chain(
        ((pre_place, transition.id) for transition in net.transitions.values() for pre_place in transition.pre_set)
    )
    transition_places_edges = chain(
        ((transition.id, post_place) for transition in net.transitions.values() for post_place in transition.post_set)
    )

    nodes = [*nodes]
    edges = [*chain(place_transition_edges, transition_places_edges)]

    graph = networkx.DiGraph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(edges)

    return graph


def check_valid_constraints(constraints: List[Constraint], model: WFNet) -> List[Constraint]:
    model_dfa = get_behavioral_automaton(model)
    projected_models = ProjectedModels(model_dfa)

    ret = []
    for constraint in constraints:
        # We check first for the constraint activation, since we don't want to discover trivial constraints
        projected_model_vacuity = model_dfa
        if OMEGA_LABEL not in constraint.vacuity_condition.input_symbols:
            projected_model_vacuity = projected_models.project(frozenset(constraint.vacuity_condition.input_symbols))

        if large_dfa1_disjoint(projected_model_vacuity, constraint.vacuity_condition):
            continue

        # Then, we check whether a constraint is active
        projected_model = model_dfa
        if OMEGA_LABEL not in constraint.relevant_labels:
            projected_model = projected_models.project(constraint.relevant_labels)

        if large_dfa1_contains(projected_model, constraint.dfa):
            ret.append(constraint)

    return ret


def large_dfa1_disjoint(dfa1: DFA, dfa2: DFA):
    omega_replace_set = list(dfa1.input_symbols.difference(dfa2.input_symbols))
    dfa2_expanded = replace_label_with(dfa2, OMEGA_LABEL, omega_replace_set) if omega_replace_set else dfa2

    return dfa1.isdisjoint(dfa2_expanded)


def large_dfa1_contains(dfa1: DFA, dfa2: DFA):
    omega_replace_set = dfa1.input_symbols.difference(dfa2.input_symbols)
    if OMEGA_LABEL in dfa1.input_symbols:
        return dfa1 <= expand_omega_alphabet(dfa2, dfa1.input_symbols)
    else:
        return dfa1 <= replace_label_with(dfa2, OMEGA_LABEL, list(omega_replace_set))


def expand_omega_alphabet(dfa: DFA, new_alphabet: Set[str]) -> DFA:
    # We must split the omega labels to consider the new symbols
    new_letters = new_alphabet.difference(dfa.input_symbols)

    if len(new_letters) == 0:
        return dfa

    def _map_state_trans_func(trans_func: Dict[str, str]):
        if OMEGA_LABEL not in trans_func:
            return trans_func

        omega_target = trans_func[OMEGA_LABEL]
        other = {new_letter: omega_target for new_letter in new_letters}
        ret = {**other, **trans_func}

        return ret

    new_transition_func = {
        state: _map_state_trans_func(trans_func)
        for state, trans_func in dfa.transitions.items()
    }

    return DFA(
        states=dfa.states,
        input_symbols=new_alphabet,
        transitions=new_transition_func,
        initial_state=dfa.initial_state,
        final_states=dfa.final_states,
        allow_partial=True,
    )


def replace_label_with(dfa: DFA, src_label: str, replace_labels: Iterable[str]):
    """Remove all src_labels by replacing with replace_labels"""
    assert not dfa.allow_partial, "Automata must be complete"

    def _map_state_trans_func(trans_func: Dict[str, str]):
        if src_label not in trans_func:
            return trans_func

        omega_target = trans_func[src_label]
        other = {label: omega_target for label in replace_labels}
        ret = {**other, **trans_func}
        ret.pop(src_label)

        return ret

    new_transition_func = {
        state: _map_state_trans_func(trans_func)
        for state, trans_func in dfa.transitions.items()
    }

    # TODO new_alphabet = frozenset(x for x in chain(dfa.input_symbols, replace_labels) if x != src_label)
    new_alphabet = frozenset(label for lookup in new_transition_func.values() for label in lookup.keys())

    return DFA(
        states=dfa.states,
        input_symbols=new_alphabet,
        transitions=new_transition_func,
        initial_state=dfa.initial_state,
        final_states=dfa.final_states,
    )


# ======================== MINIMIZATION ========================

def constraint_ordering_hard_coded(constraints: List[Constraint]) -> List[Constraint]:
    """
    Uses the MINERful ordering, which is a hard-coded ordering based on a series of static attributes of the tempaltes.
    This is not necessarily a strict total order. There might be equal or incomparable constraints.
    We break ties with the constraint's name
    """
    targets_for_activation = compute_targets_for_activation(constraints)
    ordered_constraints = [
        OrderedConstraint(constraint, get_activation_linkage(constraint, targets_for_activation))
        for constraint in constraints
    ]
    ordered_constraints.sort()

    return [ordered_constraint.constraint for ordered_constraint in ordered_constraints]


# To check if constraint c1 is stronger than c2 (c1 < c2) we test (in order):
#   - c1 has higher activation count
#   - c1 has lower priority
#   - c1 not omega invariant
#   - c1 has more activities
#   - c1's language is more restrictive
@dataclasses.dataclass
class OrderedConstraint:
    constraint: Constraint
    activation_linkage: int

    def __lt__(self, other):
        c1 = self.constraint
        c2: Constraint = other.constraint

        if c1.traits.type_priority != c2.traits.type_priority:
            return c1.traits.type_priority < c2.traits.type_priority
        if c1.traits.is_omega_invariant != c2.traits.is_omega_invariant:
            return not c1.traits.is_omega_invariant
        if c1.traits.max_label_distance != c2.traits.max_label_distance:
            return c1.traits.max_label_distance < c2.traits.max_label_distance
        if len(c1.dfa.input_symbols) != len(c2.dfa.input_symbols):
            return len(c1.dfa.input_symbols) < len(c2.dfa.input_symbols)
        if self.activation_linkage != other.activation_linkage:
            # TODO there seems to be a bug. One move is not covered anymore if we reverse this sign
            return self.activation_linkage > other.activation_linkage

        dfa_1_is_in_2, dfa_2_is_in_1 = dfa_compare(c1.dfa, c2.dfa)
        if dfa_1_is_in_2 != dfa_2_is_in_1:
            # Not equal or incomparable
            return dfa_1_is_in_2

        return c1.traits.name < c2.traits.name


def dfa_compare(dfa1: DFA, dfa2: DFA) -> Tuple[bool, bool]:
    # Returns whether 1 is in 2 and 2 is in 1
    intersect_alphabet = dfa1.input_symbols.intersection(dfa2.input_symbols)

    if len(intersect_alphabet) == 0 or (len(intersect_alphabet) == 1 and OMEGA_LABEL in intersect_alphabet):
        # If they have no label, or only the omega label in common, then we do not compare
        return False, False

    common_alphabet = dfa1.input_symbols.union(dfa2.input_symbols)

    dfa1_expanded = expand_omega_alphabet(dfa1, common_alphabet)
    dfa2_expanded = expand_omega_alphabet(dfa2, common_alphabet)

    dfa_1_minus_2 = dfa1_expanded - dfa2_expanded
    dfa_1_is_in_2 = dfa_1_minus_2.isempty()

    dfa_2_minus_1 = dfa2_expanded - dfa1_expanded
    dfa_2_is_in_1 = dfa_2_minus_1.isempty()

    return dfa_1_is_in_2, dfa_2_is_in_1


def compute_targets_for_activation(constraints: List[Constraint]) -> Dict[str, Set[str]]:
    """
    For each activation, computes the target activities that share that same activation

    This is an implementation of what MINERful does, which is buggy,
        but for consistency we keep it like that
    """
    activation_tgts: Dict[str, Set[str]] = defaultdict(set)

    for constraint in constraints:
        for activation in constraint.traits.activations:
            activation_tgts[activation].update(constraint.traits.tgt_activities)

    return activation_tgts


def get_activation_linkage(constraint: Constraint, targets_for_activation: Dict[str, Set[str]]):
    """ The activation linkage is just the size of the target activities considering the constraint's activation """
    return len({target for activation in constraint.traits.activations for target in targets_for_activation[activation]})


# ======================== MINIMIZATION ========================


def minimal_dfas_approximated(dfas: List[DFA], max_lhs_size: int, num_params: int) -> List[bool]:
    """
    Minimizes a list of DFAs using an approximated approach that checks for productions
        c_a, ..., c_m -> c_n with c_a < c_b < ... < c_m < c_n

    Returns a list of bools the same size of its input such that ret[i] is True if
        the i-th DFA is minimal, and False otherwise.

    @:param max_m: int the maximum number of parameters in the production rules on the lhs
    @:param max_k: int the maximum number of parameters for the template library
        Ideally this is inferred from the template library, but for now we require it
    """
    result = [True] * len(dfas)
    # We coumput the invariant labels here because it is more difficult to compute them on partial DFA
    dfa_ids_and_invariant_labels = [
        (dfa_id, dfa.to_partial(minify=False), get_invariant_labels(dfa)) for dfa_id, dfa in enumerate(dfas)
    ]

    # The DFAs are numbered. For each DFA we compute the non-invariant labels
    # We group DFAs by shared non-invariant labels. Start with large intersections and move to smaller
    min_intersection_size = 1
    for lhs_size in range(1, max_lhs_size + 1):
        for intersection_size in range(num_params, min_intersection_size - 1, -1):
            if intersection_size < lhs_size:
                # Effectively removing 1 1, 1 2, 2 1, 2 2
                continue

            partitions = defaultdict(lambda: [])

            for dfa_id, dfa, invariant_labels in dfa_ids_and_invariant_labels:
                # A bit inefficient to keep unpacking and packing it
                dfa_and_id = (dfa_id, dfa)
                non_invariant_labels = dfa.input_symbols - invariant_labels - {OMEGA_LABEL}
                sorted_labels = tuple(sorted(non_invariant_labels))
                if len(sorted_labels) < intersection_size:
                    partitions[sorted_labels].append(dfa_and_id)
                for labels in combinations(sorted_labels, intersection_size):
                    partitions[labels].append(dfa_and_id)

            for i, (labels, partition) in enumerate(partitions.items()):
                minimal_ids = [dfa_id for dfa_id, _ in partition if result[dfa_id]]
                minimal_dfas = [dfa for dfa_id, dfa in partition if result[dfa_id]]
                minimal_partition_dfas = _minimize_partition(minimal_dfas, lhs_size)

                for dfa_id, minimal_dfa, is_minimal in zip(minimal_ids, minimal_dfas, minimal_partition_dfas):
                    if not is_minimal:
                        result[dfa_id] = False

    # Final pass
    minimal_ids = [dfa_id for dfa_id, _, _ in dfa_ids_and_invariant_labels if result[dfa_id]]
    minimal_dfas = [dfa for dfa_id, dfa, _ in dfa_ids_and_invariant_labels if result[dfa_id]]
    minimal_partition_dfas = _minimize_partition(minimal_dfas, 1)

    for dfa_id, minimal_dfa, is_minimal in zip(minimal_ids, minimal_dfas, minimal_partition_dfas):
        if not is_minimal:
            result[dfa_id] = False

    return result


def _minimize_partition(dfas: List[DFA], k: int) -> List[bool]:
    common_alphabet = {symbol for dfa in dfas for symbol in dfa.input_symbols}
    # The bug referred in the reviewer letter was here. 
    # We were calling the "replace_label_with" function, instead of this one
    # This was leading to the omega label being discarded when intersection_size = #template_parameter
    dfas = [expand_omega_alphabet(dfa, common_alphabet) for dfa in dfas]
    return _minimize_approximated(dfas, k)


def _minimize_approximated(dfas: List[DFA], k: int) -> List[bool]:
    result = [True] * len(dfas)
    if len(dfas) == 0:
        return result

    # For lower Ks we improve by grouping DFAs together and combining groups on the LHS
    per_block_threshold = {1: 1_000, 2: 100, 3: 40}.get(k, 1)

    grouped_dfas = []
    universal_dfa = DFA.universal_language(input_symbols=dfas[0].input_symbols)

    prev_dfa = universal_dfa
    group_start_index = 0
    for index, dfa in enumerate(dfas):
        cur_dfa = prev_dfa & dfa

        # Both are minimized, so we know that they are only equal if they have the same # of states
        if len(prev_dfa.states) == len(cur_dfa.states) and prev_dfa == cur_dfa:
            result[index] = False
        else:
            if len(cur_dfa.states) > per_block_threshold:
                group_end_index = index + 1
                grouped_dfas.append((cur_dfa.minify(), group_start_index, group_end_index))
                group_start_index = group_end_index
                prev_dfa = universal_dfa
            else:
                prev_dfa = cur_dfa
    grouped_dfas.append((prev_dfa, group_start_index, len(dfas)))

    for lhs in combinations(grouped_dfas, k):
        # If one of the LHS is not minimal, we ignore the combination
        if not all(any(result[start:end]) for _, start, end in lhs):
            continue

        all_dfas = [dfa for dfa, _, _ in lhs]
        intersect_dfa = reduce(DFA.__and__, all_dfas[1:], all_dfas[0])

        start_idx = lhs[-1][-1]
        for tgt_idx, tgt_dfa in enumerate(dfas[start_idx:], start_idx):
            if not result[tgt_idx]:
                continue
            if intersect_dfa <= tgt_dfa:
                result[tgt_idx] = False

    return result


# ======================== VERIFY LOG AND VERBALIZE ========================

def verify_log(aligned_log: AlignedLog, constraints: List[Constraint]) -> List[List[Constraint]]:
    """ Returns the violated constraints for each variant """
    common_alphabet = set.union(*({*constraint.dfa.input_symbols} for constraint in constraints))
    expanded_dfas = [expand_omega_alphabet(constraint.dfa, common_alphabet) for constraint in constraints]

    print(f"Log contains {len(aligned_log.variants)} variants")
    ret = []
    for aligned_variant, _ in aligned_log.variants:
        variant = [move.move_on_log for move in aligned_variant if move.move_on_log is not None]

        violating_for_variant = []
        for constraint, expanded_dfa in zip(constraints, expanded_dfas):
            if not expanded_dfa.accepts_input(variant):
                violating_for_variant.append(constraint)

        ret.append(violating_for_variant)

    _explain_moves(aligned_log, constraints)

    return ret


def _explain_moves(aligned_log: AlignedLog, constraints: List[Constraint]):
    common_alphabet = set.union(*({*constraint.dfa.input_symbols} for constraint in constraints))
    expanded_dfas = [expand_omega_alphabet(constraint.dfa, common_alphabet) for constraint in constraints]

    violating_traces = sum(freq for alignment, freq in aligned_log.variants
                           if not all(move.is_sync_move or move.is_invisible_move is None for move in alignment))

    # Detecting which constraints are violated per variant
    violated_constraints = []
    for alignment, _ in aligned_log.variants:
        variant = [move.move_on_log for move in alignment if move.move_on_log]
        violating_for_variant = []
        for constraint, expanded_dfa in zip(constraints, expanded_dfas):
            if not expanded_dfa.accepts_input(variant):
                violating_for_variant.append(constraint)
        violated_constraints.append(violating_for_variant)

    # Count how many variants violate at least one constraint
    alignments_violating_at_least_one_constraint = sum(1 for variant_violation in violated_constraints if len(variant_violation) != 0)

    # Get the constraints explaining each alignment move. We must check all minimal constraints
    # because some moves only happen as a consequence of other move that was inserted to fix an unrelated constraint
    # i.e. violating constraint C1 causes move M1,
    #   as a consequence, the previously satisfied constraint C2 becomes violated, which causes move M2
    alignment_explanations = []
    for alignment, _ in aligned_log.variants:
        alignment_explanation: List[Optional[List[Constraint]]] = [None] * len(alignment)

        for constraint in constraints:
            dfa = constraint.dfa
            for i, move_explanation in enumerate(_get_explained_moves(alignment, dfa)):
                if move_explanation is None:
                    continue
                if alignment_explanation[i] is None:
                    alignment_explanation[i] = []
                if move_explanation:
                    alignment_explanation[i].append(constraint)
        alignment_explanations.append(alignment_explanation)

    # We flatten it to a sequence of None or List[ExplainingConstraints]
    move_explanations = [*chain.from_iterable(alignment_explanations)]

    # Mark for each move whether it is sync/invisible (None), or whether it was explained by at least one constraint
    move_is_explained: List[Optional[bool]] = [
        None if move_explanation is None else len(move_explanation) != 0
        for move_explanation in move_explanations
    ]
    counted_explanations = Counter(move_is_explained)

    # Compute log-level statistics
    explained_log_or_model = counted_explanations[True]
    not_explained_log_or_model = counted_explanations[False]
    log_or_model_counts = explained_log_or_model + not_explained_log_or_model

    # Again, flatten
    constraints_total = sum(len(move_explanation) for move_explanation in move_explanations if move_explanation)

    print(f"violating_traces: {violating_traces}")
    print(f"violating_at_least_one_constraint: {alignments_violating_at_least_one_constraint}")
    print(f"number_model_or_log_moves: {log_or_model_counts}")
    print(f"number_model_or_log_moves_explained: {explained_log_or_model}")
    print(f"violating_constraints_per_violating_traces: {constraints_total / max(alignments_violating_at_least_one_constraint, 1)}")
    # TODO there is some bug in this function that is leading to slightly different results here
    print(f"violating_constraints_per_move: {constraints_total / max(explained_log_or_model, 1)}")


def _get_explained_moves(alignment: Alignment, dfa: DFA) -> List[Optional[bool]]:
    """ Returns for which move whether the DFA can explain it (True) or not (False) """
    ret = []
    for i, move in enumerate(alignment):
        if move.is_sync_move or move.is_invisible_move:
            ret.append(None)
            continue

        projected_alignment = _project_and_change(alignment, i)
        ret.append(not _trace_fits(projected_alignment, dfa))

    assert len(ret) == len(alignment)
    return ret


def _project_and_change(alignment: Alignment, change_position):
    for j, move_ in enumerate(alignment):
        if j == change_position:
            if move_.is_log_move:
                yield move_.move_on_log
            elif move_.is_model_move:
                continue
            else:
                raise RuntimeError("Expected log or model move in i-th position")
        else:
            model_label = move_.model_label
            if model_label != '>>' and model_label is not None:
                yield model_label


def _trace_fits(trace: Iterable[str], dfa: DFA):
    if OMEGA_LABEL in dfa.input_symbols:
        trace = (x if x in dfa.input_symbols else OMEGA_LABEL for x in trace)
    return dfa.accepts_input(trace)


def main_qualitative():
    dataset = load_dataset(Path('datasets/road_fines_enriched.json'))
    repository = load_repository(Path('repositories/ELHAM_SCALABLE_RULES_K2.json'))

    # We are looking into the most violating constraints
    #   and doing some data mining to figure out which constraints co-occur
    minimal_constraints, verify_log_result = qualitative_pre_processing(dataset, repository)

    # We expand from variants to case
    new_verify_log_result = []
    for (_, freq), verify_variant_result in zip(dataset.aligned_log.variants, verify_log_result):
        new_verify_log_result.extend([verify_variant_result]*freq)
    verify_log_result = new_verify_log_result

    constraint_grouping = Counter(chain.from_iterable(verify_log_result))
    constraint_grouping = list(constraint_grouping.items())
    constraint_grouping.sort(key=lambda x: x[1], reverse=True)

    for constraint_id, freq in constraint_grouping:
        print(f'{freq} \t {constraint_id} {repr(minimal_constraints[constraint_id])}')

    print(f"NUMBER OF TRACES {len(verify_log_result)}")
    print(f"NUMBER VIOLATING TRACES {sum(1 for verify_trace_result in verify_log_result if verify_trace_result)}")

    # Compute the relative frequency of each constraint with respect to the other co-occurring and the respective lift
    constraint_results = _check_co_occurring_constraints(minimal_constraints, verify_log_result)
    constraint_results = [*constraint_results.items()]
    constraint_results.sort(key=lambda x: x[1]['frequency'], reverse=True)
    print(*constraint_results, sep='\n')

    print(f"NUM CONSTRAINTS {len(constraint_results)}")
    print(f"NUM VIOLATED CONSTRAINTS {sum(1 for _, constraint_result in constraint_results if constraint_result['frequency'] > 0)}")


def qualitative_pre_processing(dataset: Dataset, repository: Repository) \
        -> Tuple[Dict[int, Constraint], List[List[int]]]:
    minimal_constraints, verify_log_results = run_experiment(dataset, repository)

    minimal_constraints = {i: constraint for i, constraint in enumerate(minimal_constraints)}
    constraint_to_id = {repr(constraint): constraint_id for constraint_id, constraint in minimal_constraints.items()}

    verify_log_results = [[constraint_to_id[repr(constraint)] for constraint in verify_variant_result]
                          for verify_variant_result in verify_log_results]

    # Return the parameter is bad code
    return minimal_constraints, verify_log_results


def _check_co_occurring_constraints(minimal_constraints: Dict[int, Constraint],
                                    verify_log_result: List[List[int]]):
    result = {constraint_id: {'frequency': 0, 'co_occurring': []} for constraint_id in minimal_constraints.keys()}

    violated_per_trace = []
    for verify_trace_result in verify_log_result:
        trace_violations = {}
        for constraint_id in verify_trace_result:
            result[constraint_id]['frequency'] += 1
            trace_violations[constraint_id] = 1

        violated_per_trace.append(Counter(trace_violations))

    def _get_lift(_violating_traces_count, _count_source_constraint, _count_target_constraint, _count_both):
        return _violating_traces_count * _count_both / (_count_source_constraint * _count_target_constraint)

    # For each rule, count how often each other rule co-occur
    for constraint_id in minimal_constraints.keys():
        violated_traces = [trace_violations for trace_violations in violated_per_trace
                           if constraint_id in trace_violations]
        result[constraint_id]['co_occurring'] = [*sum(violated_traces, Counter()).items()]

    # Compute lift, order and trim
    violating_traces_count = sum(1 for verify_trace_result in verify_log_result if verify_trace_result)
    for src_constraint_id, src_constraint_result in result.items():
        src_constraint_freq = src_constraint_result['frequency']

        co_occurring = [
            (tgt_constraint_id, freq, _get_lift(violating_traces_count, src_constraint_freq, result[tgt_constraint_id]['frequency'], freq))
            for tgt_constraint_id, freq in src_constraint_result['co_occurring']
        ]
        # Filter for positive lift
        co_occurring = [(tgt_constraint_id, freq, lift) for tgt_constraint_id, freq, lift in co_occurring if lift > 1]

        # Order them by most frequently co-occurring,
        #   filter by threshold that they must co-occur together in at least 10% of cases
        frequently_co_occurring = sorted(co_occurring, key=lambda x: x[1], reverse=True)
        frequency_threshold = src_constraint_freq // 10
        co_occurring = [(rule_id, freq, round(lift, 2)) for rule_id, freq, lift in frequently_co_occurring
                        if freq > frequency_threshold]

        # Keep the top 10, skip 0 since this is the constraint itself
        co_occurring = co_occurring[1:10]

        src_constraint_result['co_occurring'] = co_occurring[1:10]

    return result


if __name__ == '__main__':
    main()
    main_qualitative()
