from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from .models import Scenario, Step


@dataclass(slots=True)
class Graph:
    nodes: dict[int, Step]
    parents: dict[int, set[int]] = field(default_factory=dict)
    children: dict[int, set[int]] = field(default_factory=dict)
    edge_types: dict[tuple[int, int], set[str]] = field(default_factory=dict)

    def add_edge(self, source: int, target: int, edge_type: str) -> None:
        if source == target:
            return
        self.parents.setdefault(target, set()).add(source)
        self.children.setdefault(source, set()).add(target)
        self.parents.setdefault(source, set())
        self.children.setdefault(target, set())
        self.edge_types.setdefault((source, target), set()).add(edge_type)


def build_graph(scenario: Scenario) -> Graph:
    graph = Graph(nodes={step.step_id: step for step in scenario.steps})
    last_step_by_agent: dict[str, int] = {}
    latest_producer: dict[str, int] = {}

    for step in scenario.steps:
        graph.parents.setdefault(step.step_id, set())
        graph.children.setdefault(step.step_id, set())

        if step.agent in last_step_by_agent:
            graph.add_edge(last_step_by_agent[step.agent], step.step_id, "sequential")

        for variable in step.consumes:
            producer = latest_producer.get(variable)
            if producer is None:
                continue
            graph.add_edge(producer, step.step_id, "data_dependency")
            if graph.nodes[producer].agent != step.agent:
                graph.add_edge(producer, step.step_id, "communication")

        for variable in step.produces:
            latest_producer[variable] = step.step_id
        last_step_by_agent[step.agent] = step.step_id

    return graph


def backward_trace(graph: Graph, error_node_id: int, max_depth: int = 10) -> set[int]:
    visited = {error_node_id}
    frontier = {error_node_id}
    for _ in range(max_depth):
        next_frontier: set[int] = set()
        for node_id in frontier:
            for parent in graph.parents.get(node_id, set()):
                if parent in visited:
                    continue
                visited.add(parent)
                next_frontier.add(parent)
        if not next_frontier:
            break
        frontier = next_frontier
    return visited


def reverse_distances_to_error(graph: Graph, error_node_id: int) -> dict[int, int]:
    distances = {error_node_id: 0}
    queue: deque[int] = deque([error_node_id])
    while queue:
        node_id = queue.popleft()
        for parent in graph.parents.get(node_id, set()):
            if parent in distances:
                continue
            distances[parent] = distances[node_id] + 1
            queue.append(parent)
    return distances


def descendants(graph: Graph, start: int) -> set[int]:
    seen: set[int] = set()
    queue: deque[int] = deque([start])
    while queue:
        node_id = queue.popleft()
        for child in graph.children.get(node_id, set()):
            if child in seen:
                continue
            seen.add(child)
            queue.append(child)
    seen.discard(start)
    return seen


def ancestors(graph: Graph, start: int) -> set[int]:
    seen: set[int] = set()
    queue: deque[int] = deque([start])
    while queue:
        node_id = queue.popleft()
        for parent in graph.parents.get(node_id, set()):
            if parent in seen:
                continue
            seen.add(parent)
            queue.append(parent)
    seen.discard(start)
    return seen


def incoming_cross_agent(graph: Graph, node_id: int) -> bool:
    step = graph.nodes[node_id]
    return any(graph.nodes[parent].agent != step.agent for parent in graph.parents.get(node_id, set()))


def outgoing_cross_agent(graph: Graph, node_id: int) -> bool:
    step = graph.nodes[node_id]
    return any(graph.nodes[child].agent != step.agent for child in graph.children.get(node_id, set()))


def edge_type_count(graph: Graph, node_id: int, edge_type: str) -> int:
    count = 0
    for (source, target), edge_types in graph.edge_types.items():
        if edge_type in edge_types and (source == node_id or target == node_id):
            count += 1
    return count


def approximate_betweenness(graph: Graph) -> dict[int, float]:
    node_ids = sorted(graph.nodes)
    result = {node_id: 0.0 for node_id in node_ids}
    if len(node_ids) < 3:
        return result

    ancestor_cache = {node_id: ancestors(graph, node_id) for node_id in node_ids}
    descendant_cache = {node_id: descendants(graph, node_id) for node_id in node_ids}
    normalizer = max(1, (len(node_ids) - 1) * (len(node_ids) - 2))
    for node_id in node_ids:
        result[node_id] = (len(ancestor_cache[node_id]) * len(descendant_cache[node_id])) / normalizer
    return result
