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
# This notebook is divided into 3 parts
#   1. [The Scene Graph Datastructure](#The-Scene-Graph-Datastructure): Exercises and examples for working with nodes, edges and attributes
#   2. [Planning and Graph Search on 3D Scene Graphs](#Planning-and-Graph-Search-on-3D-Scene-Graphs): More detailed exercises on how to using the graph datastructure
#   3. [Using External Libraries](#Using-External-Libraries): How to export and use 3D scene graphs with `networkx` and `pytorch_geometric`
#
# Please make sure you have at least one scene graph downloaded from [here](TBD).
# Most examples will reference the `uhumans2_office_dsg.json` file.
# We recommend using the `[NAME]_dsg.json` variants over the `[NAME]_dsg_with_mesh.json`
# variants in most cases.

# %%
import pathlib
import pprint
from typing import Dict, List

import ipywidgets as widgets
import spark_dsg as dsg

# %%
wpath = widgets.Text(
    placeholder="Enter scene graph path", description="Filepath:", disabled=False
)
wpath

# %%
# This loads and deserializes the scene graph from the provided file
dsg_path = pathlib.Path(wpath.value).expanduser().resolve()
G = dsg.DynamicSceneGraph.load(dsg_path)


# %% [markdown]
# ### The Scene Graph Datastructure
#
# This section of the notebook is designed to familiarize you with the 3D scene graph
# datastructure as it is represented in `spark_dsg`.

# %% [markdown]
# **Exercise 1.1**: Find the node IDs of all the object nodes with bounding boxes of
# volume larger than 1 m^3


# %%
def get_big_objects(G: dsg.DynamicSceneGraph, volume: float = 1.0):
    big_nodes: List[dsg.NodeSymbol] = []

    # =======================
    # TODO: Fill in code here
    # =======================

    return big_nodes


big_objects = get_big_objects(G, volume_threshold=1.5)
total_nodes = G.get_layer(dsg.DsgLayers.OBJECTS).num_nodes()
print(
    f"Big objects: {len(big_objects)} / {total_nodes} ({len(big_objects) / total_nodes:.3%})"
)
pprint.pprint([x.str(False) for x in big_objects])


# %% [markdown]
# **Exercise 1.2**: Compute a histogram of object labels keyed by human-readable
# category name


# %%
def get_object_label_histogram(G) -> Dict[str, int]:
    key = G.get_layer_id(dsg.DsgLayers.OBJECTS)
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
# **Exercise 1.3**: Plot the robot trajectory


# %%
def get_object_label_histogram(G) -> Dict[str, int]:
    key = G.get_layer_id(dsg.DsgLayers.OBJECTS)
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
