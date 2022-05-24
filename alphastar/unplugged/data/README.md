## Dataset generation


We are not able to publish pre-built datasets for Unplugged training, however we
do provide documentation and a basic set of tools required to build such a
dataset.

Generating a dataset involves 3 main steps:

1. Download replay packs from Blizzard
2. Partition the replay packs into chunks that can be processed in parallel
3. Convert each partition into the final `.tfrecord` format used for training
   and evaluation

Please note that step 3 is **computationally intensive** - it would likely take
many months to convert a full-sized dataset on a single machine. We therefore
recommend parallelizing the conversion step across multiple machines. We do not
currently provide an implementation of such a distributed pipeline
(contributions on this front would be welcome). However we do provide a simple
script that can be used to generate small datasets, and may serve as a starting
point for developing distributed pipelines.

Also note that the size of the final dataset produced in step 3 may be of the
order of **hundreds of TB**, depending on the specific set of replay packs
included.

### Downloading the replay packs

The `.SC2Replay` replay packs available via the [Blizzard Game Data APIs]
are the golden source for human data. These replay files can either be decoded
directly using [`s2protocol`], or else consumed as a stream of [protocol buffer]
messages (for instance by loading them using [PySC2]).

For convenience we provide a shell script for downloading the replay packs:

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


To give a sense of scale, the total sizes of the train and test replay packs for
4.9.2 are approximately 15 GB and 200 MB respectively. The total size of all
replay packs across all supported versions of the game is approximately 1 TB.

### Partitioning the replay data

As noted above, converting the replay packs to the final `.tfrecord` format is
computationally intensive, and generates a large volume of output data. We
therefore provide a script that can be used to split the replay packs into
roughly equally-sized partitions that can then be processed in parallel.
 First, run the following from the root of the `alphastar` git repository:

```shell
python3 alphastar/unplugged/data/generate_partitions.py  \
  --sc2_replay_path=${REPLAYS_DIR}/4.9.2/test  \
  --converted_path=${REPLAYS_DIR}/converted/4.9.2/test  \
  --num_partitions=16  \
  --partition_path=${REPLAYS_DIR}/partitions/4.9.2/test
```

This script, and the following one assume that they are running within a Python
environment where the `alphastar` package has already been installed.

Note that while 16 partitions may provide a tolerable runtime for the test set
and a single SC2 version, as shown in the above example, many more partitions
will be required if the aim is to convert the full dataset.

### Converting the data to `.tfrecord` format

Replay packs consist of compressed streams of game events, whereas Unplugged
training expects data in the `.tfrecord` format produced by the
[PySC2 converter], which consists of tensor observations for every frame
on which the player acted during the game. Because of this, the total size of
the `.tfrecord`s is on the order of **100s of times larger** than that of the
original `.SC2Replay` files. Performing this conversion involves re-rendering
each replay using the StarCraft II game binary, which is computationally
expensive.

We provide a script that converts partitioned replay files serially to
`.tfrecord` format. An example invocation for a single test partition and
Starcraft version would be something like:

```shell
python3 alphastar/unplugged/data/generate_dataset.py  \
  --sc2_replay_path=${REPLAYS_DIR}/4.9.2/test  \
  --converted_path=${REPLAYS_DIR}/converted/4.9.2/test  \
  --partition_file=${REPLAYS_DIR}/partitions/4.9.2/test/partition_0  \
  --converter_settings=alphastar/unplugged/configs/alphastar_supervised_converter_settings.pbtxt  \
  --logtostderr
```

The above should be repeated for each dataset (train, test) and version (4.8.2,
4.8.3, 4.8.4, 4.8.6, 4.9.0, 4.9.1, 4.9.2) that you intend to use. Also,
the conversion settings may be adjusted by providing a modified copy of
`alphastar_supervised_converter_settings.pbtxt`. See
https://github.com/deepmind/pysc2/pysc2/env/converter/proto/converter.proto
for the proto definition.

This step needs the StarCraft II game binary. To download, please follow the
instructions [here](https://github.com/Blizzard/s2client-proto#linux-packages).
Please note that this script **requires that the version
of the Starcraft II binary used by the PySC2 package matches the version of the
replays that are being processed**. Since the PySC2 package can only use a
single Starcraft II binary at a time, this means **you will need to set up a
separate Python environment for each replay version that you want to process**.
Instructions on where PySC2 searches for the StarCraftII binary can be found
[here](https://github.com/deepmind/pysc2#linux)

We strongly recommend that the easiest approach is to fully process
replays for each version in turn as maintaining and selecting from
multiple versions is error-prone. 

At the end of these steps,the ./converted directory tree will contain .tfrecord
files that can be used for AlphaStar training and evaluation.

