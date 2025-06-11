# DCIST PI Meeting 2025 Labs </br> Metric Semantic SLAM and Scene Graphs

Welcome to our lab!
This repository contains activities that focus on 3D scene graphs and how to use [spark_dsg](https://github.com/MIT-SPARK/Spark-DSG) to work with them.
If you are looking for the handout and activities for working with neural SDF representations, they can be found [here](https://github.com/hwcao17/MISO-DCIST-Lab).

### Requirements

This lab requires a laptop with at least python 3.8 installed and familiarity with python. However, we recommend using Ubuntu 24.04 if possible.

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
git clone https://github.com/MIT-SPARK/DCIST-2025-SceneGraphLab
```

Open the root directory of the repository in the terminal.

</details>

We assume you have a virtual python environment set up for this lab. If not, you can follow the instructions below.

<details open>

<summary><b>Creating a Python Virtual Environment on Linux</b></summary>

```shell
# You may need to install the following requirements if you don't have them
# For ubuntu, this looks like:
# sudo apt install python3-venv python3-pip
python3 -m venv dcist_lab_env
```

</details>

<details open>

<summary><b>Setting up Your Environment</b></summary>

Source your environment and install the requirements
```shell
# Use the appropriate invocation for your environment type
source dcist_lab_env/bin/activate

pip install -e .
# optionally install torch for one of the examples:
pip3 install torch --index-url https://download.pytorch.org/whl/cpu
```

</details>

<details open>

<summary><b>Opening up the Lab</b></summary>

Start the notebook
```shell
jupyter notebook
```

</details>

If you haven't worked with jupyter notebooks and `jupytext` before, you can open the notebook by right-clicking on `exercises.py` as in the screenshot below:
![image](https://github.com/user-attachments/assets/285e151c-16e3-4b94-9e4f-952bf45bfc58)

### Other Activities

See [here](https://github.com/MIT-SPARK/Hydra-ROS/blob/feature/ros2_docker/doc/ros2_setup.md#docker) for some information on getting Hydra set up with `docker` in ROS2.

### Feedback

Please consider chatting with us during the lab session if you have any feedback about `spark_dsg`!
