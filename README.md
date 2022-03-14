# AlphaStar.

[AlphaStar](https://github.com/deepmind/alphastar) is a package from
[DeepMind](http://deepmind.com) that provides the tools to train an agent
to master StarCraft II offered by [Blizzard Entertainment](http://blizzard.com).

As part of our opensourcing efforts to drive more research interest around
StarCraft II, we provide the following key offerings with this package:

1. General purpose architectures to train StarCraftII agents in `architectures/`
that can be used with different learning algorithms in online and offline
settings.

2. Data readers, offline training and evaluation scripts for fully offline
reinforcement learning with Behavior Cloning as a representative example under
`unplugged/` directory.


## Installation

We have tested AlphaStar only in **Python3.9** and **Linux**. Currently, we do
not support other operating systems and recommend users to stick to Linux.

We also recommend using a Python virtual environment to manage dependencies.
This should help to avoid version conflicts and just generally make the
installation process easier.

```shell
python3 -m venv alphastar
source alphastar/bin/activate
pip install --upgrade pip setuptools wheel
```

1.  Installing from GitHub: if you're interested in running the bleeding edge
    versions, you can do so by cloning our GitHub repository and then executing
    the following command from the main directory (where setup.py is located):

    `pip install -e .` (For an editable version) and `pip install .` (For a
    non-editable version)

    Note that this will also install all the dependencies of AlphaStar.

## Quickstart

For quickstart instructions on how to run training and evaluation scripts in
*fully offline* settings, please refer to the README in `alphastar/unplugged/`.
In this repository, we have not provided any online RL training code but for the
architectures only.

## About

Disclaimer: This is not an official Google product.

If you use the agents, architectures and offline RL benchmarks published in
this repository, please cite our
[Alphastar Unplugged](https://openreview.net/pdf?id=Np8Pumfoty) paper.
