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

### Train with dummy data (for debugging)

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

To run the same script using Bazel:

```shell
bazel run --cxxopt='-std=c++17' alphastar/unplugged/scripts:train -- \
  --config=alphastar/unplugged/configs/alphastar_supervised.py:alphastar.dummy \
  --config.train.max_number_of_frames=16 \
  --config.train.learner_kwargs.batch_size=4 \
  --config.train.datasource.kwargs.shuffle_buffer_size=16 \
  --config.train.optimizer_kwargs.lr_frames_before_decay=4 \
  --config.train.learner_kwargs.unroll_len=3 \
  --config.train.datasource.name=DummyDataSource
```

Note the extra `--` between `alphastar/unplugged/scripts:train` and the rest of
the flags.

Do note that these commands are training with a dummy architecture on a dummy
datasource.

### Train with real data

To train with real data

1.  Follow the instructions for data generation in
    [alphastar/unplugged/data/README.md](https://github.com/deepmind/alphastar/blob/master/alphastar/unplugged/data/README.md),

2.  The next step is to create a paths python file with two constants

    -   BASE_PATH and RELATIVE_PATHS. BASE_PATH is the root directory for the
        converted datasets that were generated in Step 1. RELATIVE_PATHS is a
        dictionary mapping of keys and values as follows :
        {(replay_versions, data_split, player_min_mmr) : <Glob pattern relative to BASE_PATH for files>}

    We have provided a
    [template file](https://github.com/deepmind/alphastar/blob/master/alphastar/unplugged/data/paths.py.template)
    for setting the data paths appropriately. Please copy this template file to
    some directory of choice `cp alphastar/unplugged/data/paths.py.template
    /tmp/paths.py` Modify the paths based on step 1 and use the file as \
    `config.train.datasource.kwargs.dataset_paths_fname` while launching
    training.

    While training, the particular data that you want to train on can be set by
    setting the `replay_versions`, `data_split`, `player_min_mmr` and
    `dataset_paths_fname` via the config using `config.train.datasource.kwargs`
    or invoking the same on command line.

3.  After these two steps, run (to confirm the entire training apparatus with
    training data from SC2 version 4.9.2, assuming that the paths file from step
    2 is `/tmp/paths.py`)

    ```shell
    python alphastar/unplugged/scripts/train.py \
      --config=alphastar/unplugged/configs/alphastar_supervised.py:alphastar.dummy \
      --config.train.max_number_of_frames=16 \
      --config.train.learner_kwargs.batch_size=4 \
      --config.train.datasource.kwargs.shuffle_buffer_size=16 \
      --config.train.optimizer_kwargs.lr_frames_before_decay=4 \
      --config.train.learner_kwargs.unroll_len=3 \
      --config.train.datasource.name=OfflineTFRecordDataSource \
      --config.train.datasource.kwargs.dataset_paths_fname='/tmp/paths.py' \
      --config.train.datasource.kwargs.replay_versions='("4.9.2",)'
    ```

4.  To run default full scale training after the real dataset is generated and
    the paths are updated, run the following command. Do note that the default
    setting is to run with all replay versions. If you want to run on specific
    replay versions only, please set
    `config.train.datasource.kwargs.replay_versions` as shown below.

    ```shell
    python alphastar/unplugged/scripts/train.py \
      --config=alphastar/unplugged/configs/alphastar_supervised.py:alphastar.full \
      --config.train.datasource.kwargs.dataset_paths_fname='/tmp/paths.py' \
      --config.train.datasource.kwargs.replay_versions='("4.9.2",)'
    ```

### Evaluate a random agent

To evaluate a random agent in the environment for one full episode, run:

```shell
python alphastar/unplugged/scripts/evaluate.py \
  --config=alphastar/unplugged/configs/alphastar_supervised.py:alphastar.dummy \
  --config.eval.log_to_csv=False \
  --config.eval.evaluator_type=random_params
```

More instructions on how to use these scripts for full-fledged training and
evaluation can be found in the docstrings of the scripts. Information about
different architecture names can be found
[here](https://github.com/deepmind/alphastar/blob/master/alphastar/architectures/README.md).

## About

Disclaimer: This is not an official Google product.

If you use the agents, architectures and benchmarks published in this
repository, please cite our
[Alphastar Unplugged](https://openreview.net/pdf?id=Np8Pumfoty) paper.
