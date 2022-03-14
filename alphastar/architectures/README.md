# Architectures used in training AlphaStar.

Currently, we present three different architectures to train RL agents on
StarCraft II as follows:

  1. `alphastar.full` - This is our gold standard architecture with full sized
  scalar, visual and units embeddings. The results we have reported in our work
  [Alphastar Unplugged](https://openreview.net/pdf?id=Np8Pumfoty)
  are always using this architecture unless stated otherwise.

  2. `alphastar.lite` - This has all the modules that alphastar.full contains, but
  is simplified by reducing the sizes and number of layers in the modules.
  This architecture is still good enough to get non-trivial win rates against
  'easy' bot and similar agents.

  3. `alphastar.dummy` - This is a barebones architecture setup which has very
  few modules, but contains the same interfaces as the other two architectures.
  This should be used purely for debugging purposes and training/evaluating with
  this should yield only trivial results.

Both alphastar.full and alphastar.lite architectures are further customizable
beyond the defaults that are provided, while launching train/eval using
command line args. For example, you can change `units_stream_size` using
`config.architecture.kwargs.config.units_stream_size`
