from __future__ import annotations

import queue
from typing import TYPE_CHECKING

from .variables import VariableBase

if TYPE_CHECKING:
    import graphviz


def try_import_graphviz():
    try:
        import graphviz

        return graphviz
    except ImportError:
        return None


def draw_variable(graph: graphviz.Digraph, var: VariableBase):
    # Draw Variable
    graph.attr('node', shape='oval')
    graph.node(var.id, str(var))

    # Draw Tracker
    tracker = var.tracker
    tracker_name = tracker.__class__.__name__
    graph.attr('node', shape='rect')
    graph.node(tracker.id, tracker_name)
    # graph.node(str(tracker_id))

    # Draw edge (Tracker -> Variable)
    graph.edge(tracker.id, var.id)

    # Draw edge (Tracker inputs -> Tracker)
    for input in tracker.inputs:
        graph.edge(input.id, tracker.id)


def view_tracker(root_variables: list[VariableBase]):
    graphviz = try_import_graphviz()
    if graphviz is None:
        print("Cannot import graphviz, please install it first.")
        return

    graph = graphviz.Digraph("graph", filename="out", format="png")
    visited = set()
    var_queue = queue.Queue()
    for var in root_variables:
        var_queue.put(var)

    while not var_queue.empty():
        var = var_queue.get()
        if var.id in visited:
            continue
        visited.add(var.id)
        draw_variable(graph, var)
        for input in var.tracker.inputs:
            if input not in var_queue.queue:
                var_queue.put(input)

    graph.render(view=False)
