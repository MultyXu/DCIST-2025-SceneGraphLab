import pathlib

import gdown
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import spark_dsg as dsg

URLS = {
    "uhumans2_office_ade20k_full_dsg_with_mesh": "1jwcjrE4-6PvOgEgJipETkQaLgC43biFT",
    "mit_courtyard_ade20k_full_dsg_with_mesh": "1wysF6XgSUOmOnQ6k-zLCnfEXlI9lcwM9",
}


def load_examples(force=False):
    examples = {}
    output = pathlib.Path(__file__).absolute().parent / "scene_graphs"
    output.mkdir(parents=True, exist_ok=True)
    for name, file_id in URLS.items():
        graph_path = output / f"{name}.json"
        if force:
            graph_path.unlink(missing_ok=True)

        if not graph_path.exists():
            with graph_path.open("wb") as fout:
                gdown.download(id=file_id, output=fout)

        examples[graph_path.stem] = dsg.DynamicSceneGraph.load(graph_path)

    return examples


def draw_layer(ax, G, layer_name, node_alpha=1.0, node_color="lightblue"):
    layer = G.get_layer(layer_name)
    mapping = {x.id.value: idx for idx, x in enumerate(layer.nodes)}
    pos = np.array([x.attributes.position for x in layer.nodes])

    sources = []
    targets = []
    for edge in layer.edges:
        sources.append(pos[mapping[edge.source]])
        targets.append(pos[mapping[edge.target]])

    sources = np.array(sources)
    targets = np.array(targets)
    edge_x = np.vstack((sources[:, 0].T, targets[:, 0].T))
    edge_y = np.vstack((sources[:, 1].T, targets[:, 1].T))
    ax.plot(edge_x, edge_y, c="k", alpha=0.5, lw=0.3)
    ax.scatter(pos[:, 0], pos[:, 1], alpha=node_alpha, c=node_color)


def draw_path(ax, G, nodes):
    x = []
    y = []
    for idx in range(1, len(nodes)):
        p_prev = G.get_node(nodes[idx - 1]).attributes.position
        p_curr = G.get_node(nodes[idx]).attributes.position
        x += [p_prev[0], p_curr[0]]
        y += [p_prev[1], p_curr[1]]

    ax.plot(x, y, c="r")


def show_planning_result(G, path):
    fig = plt.figure(figsize=(8, 5))
    with sns.axes_style("whitegrid"):
        ax = fig.add_subplot()
        ax.axis("equal")

    draw_layer(ax, G, dsg.DsgLayers.PLACES)

    draw_path(ax, G, path)
    fig.tight_layout()


def show_region_planning_result(G, path, region):
    fig = plt.figure(figsize=(8, 5))
    with sns.axes_style("whitegrid"):
        ax = fig.add_subplot()
        ax.axis("equal")

    draw_layer(ax, G, dsg.DsgLayers.PLACES, node_alpha=0.5, node_color="gray")

    pos = np.array(
        [G.get_node(x).attributes.position for x in get_room_children(G, region)]
    )
    ax.scatter(pos[:, 0], pos[:, 1])

    draw_path(ax, G, path)
    fig.tight_layout()


def get_room_children(G, node_id):
    """Get place IDs that are children of a room node."""
    node = G.get_node(node_id)
    return [x for x in node.children() if G.get_layer(dsg.DsgLayers.PLACES).has_node(x)]


def get_room_connection(G, room1, room2):
    """Get a pair of place IDs that share and edge between the two rooms."""
    r1_node = G.get_node(room1)
    for child in r1_node.children():
        child_node = G.get_node(child)
        for sibling in child_node.siblings():
            sibling_node = G.get_node(sibling)
            if not sibling_node.has_parent():
                continue

            if sibling_node.get_parent() == room2:
                return child_node.id.value, sibling_node.id.value

    return None
