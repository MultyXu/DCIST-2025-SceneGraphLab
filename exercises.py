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
# Please make sure you have at least one scene graph downloaded from [here](https://drive.google.com/drive/folders/1ONZ0Sx_tgNtS1gmGAiyRuhp1B6iw4-9E?usp=sharing).
# Most examples will reference the `uhumans2_office_ade20k_full_dsg.json` file.
# We recommend using the `[NAME]_dsg.json` variants over the `[NAME]_dsg_with_mesh.json`
# variants in most cases.

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
# The following cell loads all downloaded example graphs.

# %%
# TODO(nathan) write library and export

# %%
import pathlib
import pprint
from typing import Dict, List

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# %matplotlib ipympl

# %%
# This loads and deserializes the scene graph from the provided file
dsg_path = pathlib.Path("~/dsg_with_mesh.json").expanduser().resolve()
G = dsg.DynamicSceneGraph.load(dsg_path)


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
def get_object_label_histogram(G) -> Dict[str, int]:
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
# <details open>
#     <summary>Solution (click to reveal!)</summary>
#
# The solution 
# ```python
# points = []
# key = G.get_layer_key(dsg.DsgLayers.AGENTS)
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
# ### Planning and Graph Search on 3D Scene Graphs
#
# This section of the notebook is designed to familiarize you with how basic and
# hierarchical search procedures work with `spark_dsg`.
# Note that there is active research on hierarchical planning in 3D scene graphs
# (that we are not attempting to cover).

# %%

# %% [markdown]
# ### Using External Libraries

# %%

# %%
