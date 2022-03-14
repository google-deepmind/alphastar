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
used to create that. First, however, the raw replays must be downloaded. To do
so, first download the replayset files for
[train](https://storage.cloud.google.com/dm-starcraft-offline-datasets/offline-train.csv)
and
[test](https://storage.cloud.google.com/dm-starcraft-offline-datasets/offline-test.csv).
Then, for as many of the versions used by these datasets as you want to
support (from 4.8.2, 4.8.3, 4.8.4, 4.8.6, 4.9.0, 4.9.1, 4.9.2), run the
following command:

```
python3 download_replays.py \
  --key=<key>  \
  --secret=<secret>  \
  --version=<version>  \
  --replays_dir=./replays/test  \
  --download_dir=./downloads/test  \
  --filter_version=delete --replayset_csv=./offline-test.csv
```

Repeat the same process to download the train replayset (
replacing 'test' with 'train' in the command above. See the
[replays API documentation](https://github.com/Blizzard/s2client-proto/tree/master/samples/replay-api)
for information about obtaining the key and secret).

Once the replays are downloaded they can be transformed from .SC2Replay files
to converted .tfrecord files. As there is a large amount of data a simple means
of parallelizing the computation is provided. First, run the following:

```
python3 generate_partitions.py  \
  --sc2_replay_path=./replays/test/4.9.2  \
  --converted_path=./converted/test/4.9.2  \
  --num_partitions=16  \
  --partition_path=./partitions/test/4.9.2
```

Then instantiate the following once for each partition, making sure to update
--partition_file appropriately:

```
python3 generate_dataset.py  \
  --sc2_replay_path=./replays/test/4.9.2  \
  --converted_path=./replays/converted/test/4.9.2  \
  --partition_file=./partitions/test/4.9.2/partition_0  \
  --converter_settings=path/to/alphastar_supervised_converter_settings.pbtxt  \
  --logtostderr
```

The above should be repeated for each dataset (train, test) and version (4.8.2,
4.8.3, 4.8.4, 4.8.6, 4.9.0, 4.9.1, 4.9.2) that you intend to use. Once this is
done, the ./converted directory tree will contain .tfrecord files that can be
used for training and evaluation.

