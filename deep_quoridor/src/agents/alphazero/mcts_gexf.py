from collections import deque

import networkx as nx

from .mcts import Node


def generate_gexf_from_nodes(root_node: Node, max_depth: int, file_name: str):
    """
    Generates a GEXF file from a graph of Node objects.

    Each node's attributes (visit_count, value_sum, wins, losses, ucb_c, prior,
    action_taken, and game state description) are included in the GEXF output.

    Args:
        root_node: The starting Node object from which to traverse the graph.
        file_name: The name of the GEXF file to create (e.g., "game_tree.gexf").
    """
    # Create a directed graph using networkx
    nx_graph = nx.DiGraph()

    # Use a queue for Breadth-First Search (BFS) to traverse the graph
    queue = deque([(root_node, 0)])
    # Keep track of visited nodes to prevent infinite loops in case of cycles
    # (though MCTS trees are typically acyclic, this is good practice)
    visited_nodes = set()

    # Map Node objects to unique string IDs for NetworkX
    node_to_id_map = {}
    next_node_id_counter = 0

    def get_node_id(node: Node) -> str:
        """Helper to get a unique string ID for a Node object."""
        nonlocal next_node_id_counter
        if node not in node_to_id_map:
            node_to_id_map[node] = f"node_{next_node_id_counter}"
            next_node_id_counter += 1
        return node_to_id_map[node]

    while queue:
        current_node, depth = queue.popleft()
        current_node_id = get_node_id(current_node)

        if current_node_id in visited_nodes:
            continue

        visited_nodes.add(current_node_id)

        # Prepare node attributes for GEXF
        node_attributes = {
            "label": f"Node {current_node_id.split('_')[-1]}",  # Default label
            "visit_count": current_node.visit_count,
            "value_sum": current_node.value_sum,
            "wins": current_node.wins,
            "losses": current_node.losses,
            "ucb_c": current_node.ucb_c,
            "prior": current_node.prior,
        }

        # Add action_taken details if available
        if current_node.action_taken:
            node_attributes["action_taken"] = str(current_node.action_taken)
            node_attributes["label"] = f"Action: {current_node.action_taken}"

        # Add game state description if available
        if current_node.game:
            node_attributes["game_str"] = str(current_node.game)

        # Add the node to the networkx graph with all its attributes
        nx_graph.add_node(current_node_id, **node_attributes)

        if depth < max_depth:
            # Add edges to children and add children to the queue
            for child in current_node.children:
                child_id = get_node_id(child)
                nx_graph.add_edge(current_node_id, child_id)
                if child_id not in visited_nodes:
                    queue.append((child, depth + 1))

    # Write the networkx graph to a GEXF file
    nx.write_gexf(nx_graph, file_name)
    print(f"Successfully generated GEXF file: {file_name}")
