# DCIST PI Meeting 2025 Labs </br> Metric Semantic SLAM and Scene Graphs

Welcome to our lab!
This repository contains activities that focus on 3D scene graphs and how to use [spark_dsg](https://github.com/MIT-SPARK/Spark-DSG) to work with them.
If you are looking for the handout and activities for working with neural SDF representations, they can be found [here](TBD).

### Requirements

This lab has a minimal set of requirements:

1. A laptop with at least python 3.8 installed and familiarity with python

2. Examples of 3D scene graphs, which can be found [here](TBD)

We recommend using Ubuntu 24.04 if possible.

> :warning: **Warning** </br>
> We have done our best to support platforms other than Linux, but have no experience developing on macOS or Windows.
> We will not be able to diagnose build and installation issues for `spark_dsg`.
> However, it is possible to work with the example scene graphs using `networkx` in a limited capacity without `spark_dsg`.

### Objectives

Objectives for this lab include:

1. Familiarizing you with our version of the 3D scene graph data structure

2. Introducing you to [spark_dsg](https://github.com/MIT-SPARK/Spark-DSG) and the available API to work with scene graphs:
  - Layers, nodes, edges, and attributes
  - Graph structure and working with hierarchy
  - Working with external libraries

3. Getting feedback on `spark_dsg`

### Getting Started

<details open>

<summary><b>Getting the Lab</b></summary>

First, clone this lab
```shell
git clone https://github.com/MIT-SPARK/DCIST-2025-SceneGraphLab ~/scene_graph_lab
```

</details>

We assume you have a virtual python environment set up for this lab. If not, you can expand out the instructions below.

<details closed>

<summary><b>Creating a Python Virtual Environment on Linux</b></summary>

```shell
# Requirements that you may need:
# sudo apt install python3-venv python3-pip
python3 -m venv ~/dcist_lab_env
```

</details>

<details open>

<summary><b>Setting up Your Environment</b></summary>

Source your environment and install the requirements:
```shell
# Use the appropriate invocation for your environment type
source ~/dcist_lab_env/bin/activate
pip install -r ~/scene_graph_lab/requirements.txt
```

</details>

Then, navigate to the repository and start the notebook:
```shell
cd ~/scene_graph_lab
jupyter lab
```

### Other Activities

See [here](https://github.com/MIT-SPARK/Hydra-ROS/blob/feature/ros2_docker/doc/ros2_setup.md#docker) for some information on getting Hydra set up with `docker` in ROS2.

### Feedback

Please consider filling out the short survey (available in lab) and/or chatting with us during the lab session!
