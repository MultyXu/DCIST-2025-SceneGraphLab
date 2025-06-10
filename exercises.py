# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# ## Introduction to 3D Scene Graphs
#
# This notebook is designed to teach you how to use
# [spark_dsg](https://github.com/MIT-SPARK/Spark-DSG/tree/develop).
# Note that we are using the `develop` branch of `spark_dsg`.
#
# This notebook is divided into 4 sections:
#   1. [Introduction to Spark-DSG](#Introduction-to-Spark-Dsg): Examples and documentation around the Spark-DSG API
#   2. [The Scene Graph Datastructure](#The-Scene-Graph-Datastructure): Exercises and examples for working with nodes, edges and attributes
#   3. [Planning and Graph Search on 3D Scene Graphs](#Planning-and-Graph-Search-on-3D-Scene-Graphs): More detailed exercises on how to using the graph datastructure
#   4. [Using External Libraries](#Using-External-Libraries): How to export and use 3D scene graphs with `networkx` and `pytorch_geometric`
#
# This notebook will download several example scene graphs from [here](https://drive.google.com/drive/folders/1ONZ0Sx_tgNtS1gmGAiyRuhp1B6iw4-9E?usp=sharing).
# Most examples will reference the `uhumans2_office_ade20k_full_dsg_with_mesh.json` file.
#
# If you are having trouble getting the automatic download to work, we can help manually download them for you (they should go in the `src/dcist_sgl/scene_graphs` directory relative to this file).

# %% [markdown]
# ### Introduction to Spark-DSG
#
# This section of the notebook introduces parts of the Spark-DSG API you will need to be familiar with to do the rest of the notebook.

# %%
import spark_dsg as dsg

# %% [markdown]
# #### Reading and Writing Scene Graphs
#
# We often distribute 3D scene graphs in their serialized JSON format.
# This is compatiable with [nx.node_link_graph](https://networkx.org/documentation/stable/reference/readwrite/generated/networkx.readwrite.json_graph.node_link_graph.html#networkx.readwrite.json_graph.node_link_graph) (which is based on the `d3.js` graph representation).
# We also have a binary serialization format (roughly based on `msgpack`) that we use for inter-process communication (e.g., ROS).
#
# You can load a scene graph from a saved file by
# ```python
# G = dsg.load("path/to/dsg.json")
# ```
# and save the scene graph in the same manner:
# ```python
# G.save("output.json", include_mesh=True)
# ```
#
# The following cell sets up a few dependencies for the rest of the notebook and downloads and loads all the example scene graphs. Internally, the library we've set up for this lab (`dcist_sgl`) is calling `dsg.load("scene_graphs/[FILENAME].json")` for every downloaded file.

# %%
import heapq
import pathlib
import pprint
import random

import dcist_sgl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# %matplotlib ipympl

graphs = dcist_sgl.load_examples()

# This sets the graph to use for the following examples and exercises.
# Feel free to browse the available examples and pick a different one!
G = graphs["uhumans2_office_ade20k_full_dsg_with_mesh"].clone()

# %% [markdown]
# #### Layers and Partitions
#
# Any scene graph in `spark_dsg` contains some number of layers; these layers each have a numerical layer ID. These layer IDs enforcing a hierarchy between the layers. For example, our typical "schema" has objects assigned to layer 2 and places assigned to layer 3. Any edge between a place node and an object node considers the place node the "parent" in the relationship and the object node the "child".
#
# Each numerical layer ID can be associated with multiple partitions (each with their own numerical partition ID). Each layer has a default "primary" partition (ID 0) and arbitrary "secondary" partitions.
# You can access any layer with numerical IDs via
# ```python
# G.get_layer(layer_id, partition_id)
# ```
#
# We also support lookup of layers by human-readable names, e.g.,
# ```python
# G.get_layer(dsg.DsgLayers.OBJECTS)
# ```
# where `dsg.DsgLayers.OBJECTS` maps to the string "OBJECTS".
#
# See the cell below for more examples!

# %%
# basic information about the different layers
print(f"Number of objects: {G.get_layer(dsg.DsgLayers.OBJECTS).num_nodes()}")
print(f"Number of places: {G.get_layer(dsg.DsgLayers.PLACES).num_nodes()}")
print(f"Number of rooms: {G.get_layer(dsg.DsgLayers.ROOMS).num_nodes()}")

# more about "a" as a partition ID later, but "a" is mapped internally to the corresponding ASCII value 97
print(f"Number of robot poses: {G.get_layer(2, 'a').num_nodes()}")

# get the numerical IDs associated with a layer name
print("Places layer and partition:", G.get_layer_key(dsg.DsgLayers.PLACES))
print("2D Places layer and partition:", G.get_layer_key(dsg.DsgLayers.MESH_PLACES))

# %% [markdown]
# #### Nodes and Edges
#
# Every node in the scene graph has an unique integer ID that has a corresponding human-readable "symbol" that consists of a prefix character (e.g., 'O', 'p', 'R') and an index (e.g., 3, 15). You can get the integer ID from a "node symbol" via
# ```python
# dsg.NodeSymbol('P', 4).value  # returns 5764607523034234884
# ```
#
# Many scene graph and scene graph layer methods will convert node symbol arguments, i.e., `G.has_node(5764607523034234884)` and `G.has_node(dsg.NodeSymbol('P', 4))` will function the same.
#
# Every node in the graph has a set of attributes; these differ depending on the type of node. However, every node at least has a position and timestamp associated with it.
#
# Edges are keyed as source and target node IDs and are undirected. Edges also carry attributes, though these are less commonly used.
#
# See the cell below for some examples with nodes and edges!

# %%
first_object = None
for node in G.get_layer(dsg.DsgLayers.OBJECTS).nodes:
    first_object = node
    break

print(first_object)

first_place_edge = None
for edge in G.get_layer(dsg.DsgLayers.PLACES).edges:
    first_place_edge = edge

print(first_place_edge)

# you can get nodes and edges directly from the scene graph
source = G.get_node(first_place_edge.source)
print(f"Source {source.id.str()} siblings: {[dsg.NodeSymbol(x).str() for x in source.siblings()]}")
print(f"Target information:\n{G.get_node(first_place_edge.target).attributes}")

# %% [markdown]
# #### Getting information about `spark_dsg`
#
# You may find `help` useful in determining avaiable methods and properties of different objects in `spark_dsg`. We've already ran `help` for a couple of the more helpful objects that you may use!

# %%
help(dsg.SceneGraphNode)

# %%
help(dsg.SemanticNodeAttributes)  # many of the node attributes inherit from this class

# %% [markdown]
# ### The Scene Graph Datastructure
#
# This section of the notebook is designed to familiarize you with the 3D scene graph
# datastructure as it is represented in `spark_dsg`.

# %% [markdown]
# #### Exercise 1.1
# Find the node IDs of all the object nodes with bounding boxes of volume larger than 1 m^3. You may find the following helpful:


# %%
help(dsg.BoundingBox.volume)


# %%
def get_big_objects(G: dsg.DynamicSceneGraph, volume: float = 1.0):
    big_nodes: List[dsg.NodeSymbol] = []

    # =======================
    # TODO: Fill in code here
    # =======================

    return big_nodes


big_objects = get_big_objects(G, volume=1.5)
total_nodes = G.get_layer(dsg.DsgLayers.OBJECTS).num_nodes()
print(f"Objects: {len(big_objects)} / {total_nodes} ({len(big_objects) / total_nodes:.3%})")
pprint.pprint([x.str(False) for x in big_objects])


# %% [markdown]
# <details>
#     <summary>Solution (click to reveal!)</summary>
#
# The solution for this exercise is to just iterate through the nodes in the object layer and use the bounding box attribute:
# ```python
# big_nodes = [
#     x.id
#     for x in G.get_layer(dsg.DsgLayers.OBJECTS).nodes
#     if x.attributes.bounding_box.volume() > volume
# ]
# ```

# %% [markdown]
# #### Exercise 1.2
# Compute a histogram of object labels keyed by human-readable category name


# %%
def get_object_label_histogram(G: dsg.DynamicSceneGraph):
    key = G.get_layer_key(dsg.DsgLayers.OBJECTS)
    labelspace = G.get_labelspace(key.layer, key.partition)

    histogram = {}
    for name in labelspace.names_to_labels:
        histogram[name] = 0

    # =======================
    # TODO: Fill in code here
    # =======================

    return histogram


label_counts = get_object_label_histogram(G)
pprint.pprint({k: v for k, v in label_counts.items() if v > 0})


# %% [markdown]
# <details>
#     <summary>Solution (click to reveal!)</summary>
#
# The solution for this exercise (i) iterates through the object layer nodes and (ii) uses the semantic (integer) label of the node to look up the corresponding category name:
# ```python
# for node in G.get_layer(dsg.DsgLayers.OBJECTS).nodes:
#     histogram[labelspace.labels_to_names[node.attributes.semantic_label]] += 1
# ```
# <br>
#
# This example only applies to closed-set objects.
# A later exercise shows how language features can be accessed and used through the `semantic_feature` attribute,
# which would be required (along with the associated category embeddings) if you tried to implement something similar for
# some sort of zero-shot set of categories.
#
# </details>

# %% [markdown]
# #### Exercise 1.3
#
# Plot the 2D x-y projection of the robot trajectory. Note that information about the trajectory is contained in a partition of the AGENTS layer, where the partition ID is the ASCII value of the identifier for that particular robot (typically one of `['a', 'b', 'c', 'd', 'e']`). All of the provided scene graphs are built with a single robot that has the identifier 'a'. The layer name AGENTS maps to the "primary" partition of the layer that contains all of the agent trajectories.


# %%
def plot_agent_trajectory(G):
    fig = plt.figure(figsize=(8, 5))
    with sns.axes_style("whitegrid"):
        ax = fig.add_subplot()
        ax.axis("equal")
    
    # =======================
    # TODO: Fill in code here
    # =======================

    fig.tight_layout()
    

plot_agent_trajectory(G)

# %% [markdown]
# <details>
#     <summary>Solution (click to reveal!)</summary>
#
# ```python
# key = G.get_layer_key(dsg.DsgLayers.AGENTS)
#
# points = []
# for node in G.get_layer(key.layer, "a").nodes:
#     points.append(node.attributes.position)
#
# points = np.array(points)
# ax.plot(points[:, 0], points[:, 1])
# ```
# <br>
#
# </details>

# %% [markdown]
# #### Exercise 1.4
#
# Create a histogram of the object categories in each room


# %%
def get_object_counts_per_room(G):
    # This should be a mapping between the node ID of each room and a dictionary
    # containing the category name and number of object instances for that category
    room_object_counts = {}
    
    # =======================
    # TODO: Fill in code here
    # =======================
    
    return room_object_counts


room_object_counts = get_object_counts_per_room(G)
for room_id, counts in room_object_counts.items():
    print(f"Room {room_id}:")
    pprint.pprint({k: v for k, v in counts.items() if v > 0}, indent=2)


# %% [markdown]
# <details>
#     <summary>Solution (click to reveal!)</summary>
#
# Any solution to this problem requires going through the places layer to connect the objects to the rooms.
# ```python
# key = G.get_layer_key(dsg.DsgLayers.OBJECTS)
# labelspace = G.get_labelspace(key.layer, key.partition)
#
# for room in G.get_layer(dsg.DsgLayers.ROOMS).nodes:
#     curr_histogram = {n: 0 for n in labelspace.names_to_labels}
#     for place_id in room.children():
#         place = G.get_node(place_id)
#         for child_id in place.children():
#             if dsg.NodeSymbol(child_id).category != "O":
#                 continue
#
#             child = G.get_node(child_id)
#             curr_histogram[labelspace.labels_to_names[child.attributes.semantic_label]] += 1
#     
#     room_object_counts[room.id] = curr_histogram
# ```
# <br>
#
# </details>

# %% [markdown]
# ### Planning and Graph Search on 3D Scene Graphs
#
# This section of the notebook is designed to familiarize you with how basic and
# hierarchical search procedures work with `spark_dsg`.
# Note that there is active research on hierarchical planning in 3D scene graphs
# (that we are not attempting to cover).

# %% [markdown]
# #### Exercise 2.1
#
# Given two place nodes, produce the shortest path between them in the places layer. We provide most of the implementation for A*; the only parts you have to fill in is (i) the computation of the path length to each node being considered for expansion and (ii) the computation of the heuristic between the node being considered for expansion and the goal.

# %%
def get_path_recursive(parents, prev_path, source):
    """Translates parent map to an actual list of node IDs in a path."""
    to_expand = prev_path[0]
    if to_expand == source:
        return prev_path

    return get_path_recursive(parents, [parents[to_expand]] + prev_path, source)


def plan_path(G: dsg.DynamicSceneGraph, source: int, target: int, layer_name: str = dsg.DsgLayers.PLACES, is_valid=None):
    """Plan a path between the source and target node."""
    layer = G.get_layer(layer_name)
    if not layer.has_node(source) or not layer.has_node(target):
        print(f"Graph does not contain {dsg.NodeSymbol(source).str()} or {dsg.NodeSymbol(target).str()}")
        return None

    visited = set()
    open_set = [(0.0, source)]
    cost_to_go = {source: 0.0}
    parents = {}
    while len(open_set) > 0:
        _, node_id = heapq.heappop(open_set)
        if node_id == target:
            break
    
        if node_id in visited:
            continue
            
        visited.add(node_id)
        curr_node = G.get_node(node_id)
        curr_dist = cost_to_go[node_id]
        
        for sibling_id in curr_node.siblings():
            if is_valid is not None and not is_valid(sibling_id):
                continue
            
            sibling = G.get_node(sibling_id)
            
            g = 0.0  # cost to go (i.e., total distance to sibling node)
            h = 0.0  # heuristic estimate (euclidean distance to target)
            
            # =======================
            # TODO: Fill in code here
            # =======================
            
            f = g + h
            if sibling_id in cost_to_go and f >= cost_to_go[sibling_id]:
                continue
                
            cost_to_go[sibling_id] = g
            parents[sibling_id] = node_id
            heapq.heappush(open_set, (f, sibling_id))

    if target not in parents:
        print(f"No viable path to target {dsg.NodeSymbol(target).str()}")
        return None

    return get_path_recursive(parents, [target], source)


place_ids = [x.id.value for x in G.get_layer(dsg.DsgLayers.PLACES).nodes]
random.seed(12345678)
random.shuffle(place_ids)

path = plan_path(G, place_ids[0], place_ids[1])
dcist_sgl.show_planning_result(G, path)


# %% [markdown]
# <details>
#     <summary>Solution (click to reveal!)</summary>
#
# Note that it is slightly more efficient to pre-compute the edge distances and/or cache some of the positions earlier in the algorithm, but this is the most self-contained way to compute the relevant quantitites:
#
# ```python
# p_target = G.get_node(target).attributes.position
# p_curr = curr_node.attributes.position
# p_sibling = sibling.attributes.position
# g = np.linalg.norm(p_curr - p_sibling) + curr_dist
# h = np.linalg.norm(p_target - p_sibling)        
# ```
# <br>
#
# </details>

# %% [markdown]
# #### Exercise 2.2
#
# Given a region and two place nodes that are contained in the region, return the shortest path between the two place nodes that remains inside the region. `plan_path` from the previous exercise takes an optional filter when expanding nodes that you can use to accomplish this.

# %%
def plan_path_in_region(G: dsg.DynamicSceneGraph, region: int, source: int, target: int):
    """Plan a path between the source and target node inside a region."""

    # =======================
    # TODO: Fill in code here
    # =======================
    
    def node_in_region(node_id):
        """Check if node exists in region."""
        in_region = True
        # =======================
        # TODO: Fill in code here
        # =======================
        return in_region
            
    
    return plan_path(G, source, target, is_valid=node_in_region)

region_ids = [x.id.value for x in G.get_layer(dsg.DsgLayers.ROOMS).nodes] 
random.shuffle(region_ids)

region = region_ids[0]
children = dcist_sgl.get_room_children(G, region)
random.shuffle(children)

path = plan_path_in_region(G, region, children[0], children[1])
dcist_sgl.show_region_planning_result(G, path, region)


# %% [markdown]
# <details>
#     <summary>Solution (click to reveal!)</summary>
#
# To set up the filter, we populate all the children of the region:
# ```python
# parent = G.get_node(region)
# children = set([x for x in parent.children() if G.get_layer(dsg.DsgLayers.PLACES).has_node(x)])
# ```
# <br>
#
# The filter itself is just
# ```python
# in_region = node_id in children
# ```
# <br>
# and can be condensed to `return node_id in children` if desired. 
#
# </details>

# %% [markdown]
# #### Exercise 2.3
#
# Find the shortest path between two points by:
#   - Looking up the regions that contain the start and goal place
#   - Planning the shortest path through the regions between the start and goal region
#   - Concatenating the shortest path between each pair of regions through the places layer

# %%
def plan_path_through_regions(G: dsg.DynamicSceneGraph, start: int, end: int):
    """Plan a path between two places using the region layer."""
    start_node = G.get_node(start)
    end_node = G.get_node(end)
    if not start_node.has_parent() or not end_node.has_parent():
        return []

    room_sequence = []

    # =========================================================
    # TODO: Fill in code here to get a path through the regions
    # =========================================================

    if len(room_sequence) == 1:
        return plan_path_in_region(G, room_sequence[0], start, end)
    
    path = []
    prev_place = start
    for i in range(1, len(room_sequence)):
        prev_room_place, next_room_place = dcist_sgl.get_room_connection(G, room_sequence[i - 1], room_sequence[i])
        path += plan_path_in_region(G, room_sequence[i - 1], prev_place, prev_room_place)
        path.append(next_room_place)
        prev_place = next_room_place

    return path


path = plan_path_through_regions(G, place_ids[0], place_ids[1])
dcist_sgl.show_planning_result(G, path)

# %% [markdown]
# <details>
#     <summary>Solution (click to reveal!)</summary>
#
# ```python
# room_sequence = plan_path(G, start_node.get_parent(), end_node.get_parent(), layer_name=dsg.DsgLayers.ROOMS)
# ```
# <br>
#
# </details>

# %% [markdown]
# ### Using External Libraries

# %% [markdown]
# #### Example 3.1
#
# This example converts the places layer to networkx and uses an approximate TSP solver from networkx to plan a cycle between a list of place nodes

# %%
from spark_dsg.networkx import layer_to_networkx
import networkx as nx

layer_nx = layer_to_networkx(G.get_layer(dsg.DsgLayers.PLACES))
for source, target in layer_nx.edges:
    pos_s = layer_nx.nodes[source]["position"]
    pos_t = layer_nx.nodes[target]["position"]
    dist = np.linalg.norm(pos_s - pos_t)
    layer_nx.edges[source, target]["dist"] = dist

# weight for TSP solver is actually cost
path = nx.approximation.traveling_salesman_problem(layer_nx, nodes=place_ids[:10], weight="dist")
dcist_sgl.show_planning_result(G, path)

# %% [markdown]
# #### Example 3.2
#
# This example uses [pytorch-geometric](https://pytorch-geometric.readthedocs.io/en/latest/) to locally average the positions of the nodes in the place layer.

# %%
import torch
import torch_geometric.nn as pyg_nn


class Classifier(torch.nn.Module):

    def __init__(self, x_value):
        super().__init__()
        self.x_value = x_value
        self.conv = pyg_nn.SimpleConv(aggr="mean", combine_root="self_loop")

    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        return (x[:, 0] > self.x_value).to(torch.int64)

        
def node_feature(G, node):
    return node.attributes.position


G_torch = G.get_layer(dsg.DsgLayers.PLACES).to_torch(node_feature)
model = Classifier(5.0)
y = model(G_torch.x, G_torch.edge_index.to(torch.int64))

fig = plt.figure(figsize=(8, 5))
with sns.axes_style("whitegrid"):
    ax = fig.add_subplot()
    ax.axis("equal")

colors = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
dcist_sgl.draw_layer(ax, G, dsg.DsgLayers.PLACES)
ax.scatter(G_torch.pos[:, 0], G_torch.pos[:, 1], color=colors[y])
fig.tight_layout()

# %%
