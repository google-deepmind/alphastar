# AlphaStar Unplugged - Offline Reinforcement Learning for Starcraft.

We have published a [paper](https://openreview.net/pdf?id=Np8Pumfoty) which
outlines why Starcraft is one of the most challenging offline RL benchmarks to
date along with the performance benchmarks for a spectrum of offline RL
approaches that we have explored on this data. The current set of algorithms
released as part of this repository are

1.  Behavior Cloning (**BC**). One can implement a fine-tuned version
    (**FT-BC**) by warmstarting from a trained checkpoint.

## Quickstart

Our training setup is heavily config driven. Our main config can be found in
`alphastar/configs/alphastar_supervised.py`. Run all these commands from the
root directory where the package is downloaded into.

To run training for a few steps with some config arguments updated, run:

```shell
python alphastar/unplugged/scripts/train.py \
  --config=alphastar/unplugged/configs/alphastar_supervised.py:alphastar.dummy \
  --config.train.max_number_of_frames=16 \
  --config.train.learner_kwargs.batch_size=4 \
  --config.train.datasource.kwargs.shuffle_buffer_size=16 \
  --config.train.optimizer_kwargs.lr_frames_before_decay=4 \
  --config.train.learner_kwargs.unroll_len=3 \
  --config.train.datasource.name=DummyDataSource
```

Do note that the previous command is training with a dummy architecture on a
dummy datasource. To train with real data, follow the instructions for data
generation in `alphastar/unplugged/data/README.md`, and then update the
paths appropriately in `alphastar/unplugged/data/paths.py` and finally run,


```shell
python alphastar/unplugged/scripts/train.py \
  --config=alphastar/unplugged/configs/alphastar_supervised.py:alphastar.dummy \
  --config.train.max_number_of_frames=16 \
  --config.train.learner_kwargs.batch_size=4 \
  --config.train.datasource.kwargs.shuffle_buffer_size=16 \
  --config.train.optimizer_kwargs.lr_frames_before_decay=4 \
  --config.train.learner_kwargs.unroll_len=3 \
  --config.train.datasource.name=OfflineTFRecordDataSource
```

To run default full scale training after the real dataset is generated and the 
paths are updated, run

```shell
python alphastar/unplugged/scripts/train.py \
  --config=alphastar/unplugged/configs/alphastar_supervised.py:alphastar.full
```

To evaluate a random agent in the environment for one full episode, run:

```shell
python alphastar/unplugged/scripts/evaluate.py \
  --config=alphastar/unplugged/configs/alphastar_supervised.py:alphastar.dummy \
  --config.eval.log_to_csv=False \
  --config.eval.evaluator_type=random_params
```

More instructions on how to use these scripts for full-fledged training and
evaluation can be found in the docstrings of the scripts. Information about
different architecture names can be found in README under `architectures/`
directory.

## About

Disclaimer: This is not an official Google product.

If you use the agents, architectures and benchmarks published in this
repository, please cite our
[Alphastar Unplugged](https://openreview.net/pdf?id=Np8Pumfoty) paper.
