## Dataset generation

The .SC2Replay replay packs available via the
[Blizzard Game Data APIs](https://github.com/Blizzard/s2client-proto/tree/master/samples/replay-api)
are the golden source for human data. These replay files can be decoded directly using
[s2protocol](https://github.com/Blizzard/s2protocol), else consumed as a stream
of [proto](https://developers.google.com/protocol-buffers) messages by loading
them using [PySC2](https://github.com/deepmind/pysc2), for instance.

Unplugged training, however, expects observation data in the format produced by
the
[PySC2 converter](https://github.com/deepmind/pysc2/pysc2/env/converter/converter.py).
[generate_dataset.py](generate_dataset.py), in
conjunction with [generate_partitions.py](generate_partitions.py), can be
used to create that. First, however, the raw replays must be downloaded. For
convenience we provide a shell script for this:

```shell
alphastar/unplugged/data/get_replays.sh
```

We support three kinds of datasets -- `train, test, debug`. `train` downloads
all the replay packs that were used in training our AlphaStar baseline and
`test` offers a small subset of replay packs that can be used for validation
and testing purposes. Purely for testing our data generation pipeline,
we also provide a `debug` replayset which has an extremely minimal number of
replay packs. This is supported only with the 4.9.2 version of StarCraftII.
The debug dataset is not expected to provide any meaningful performance when
an AlphaStar agent is trained with it. The following command-line instructions
are provided for the `test` replays and can be replaced with `train` or `debug`
when working with those datasets.

Once the replays are downloaded they can be transformed from .SC2Replay files
to converted .tfrecord files. As there is a large amount of data a simple means
of parallelizing the computation is provided. First, run the following from the
root of the `alphastar` git repository:


```shell
python3 alphastar/unplugged/data/generate_partitions.py  \
  --sc2_replay_path=${REPLAYS_DIR}/4.9.2/test  \
  --converted_path=${REPLAYS_DIR}/converted/4.9.2/test  \
  --num_partitions=16  \
  --partition_path=${REPLAYS_DIR}/partitions/4.9.2/test
```

Then instantiate the following once for each partition, making sure to update
`--partition_file` appropriately:

```shell
python3 alphastar/unplugged/data/generate_dataset.py  \
  --sc2_replay_path=${REPLAYS_DIR}/4.9.2/test  \
  --converted_path=${REPLAYS_DIR}/converted/4.9.2/test  \
  --partition_file=${REPLAYS_DIR}/partitions/4.9.2/test/partition_0  \
  --converter_settings=alphastar/unplugged/configs/alphastar_supervised_converter_settings.pbtxt  \
  --logtostderr
```

The above should be repeated for each dataset (train, test) and version (4.8.2,
4.8.3, 4.8.4, 4.8.6, 4.9.0, 4.9.1, 4.9.2) that you intend to use. Once this is
done, the ./converted directory tree will contain .tfrecord files that can be
used for training and evaluation.

It is also necessary to download the  Linux package that corresponds to the
replay version being downloaded. To do so, please follow the instructions
[here](https://github.com/Blizzard/s2client-proto#linux-packages).
We also strongly recommend that the easiest approach is to fully process
replays for each version in turn as maintaining and selecting from
multiple versions is error-prone.

