# Must precede the first transformer_engine import in the process, and every SkyRL
# module that imports TE or Megatron lives under this package. See the module docstring.
from skyrl.backends.skyrl_train.patches.te.pin_nvrtc import pin_te_nvrtc_to_pip_cuda

pin_te_nvrtc_to_pip_cuda()

# Lets AutoConfig load Muse Spark Open checkpoints on the pinned transformers.
from skyrl.backends.skyrl_train.patches.transformers.muse_spark_config import (  # noqa: E402
    register_muse_spark_config,
)

register_muse_spark_config()
