# Build configurations for TensorRT.

def if_tensorrt(if_true, if_false=[]):
  """Tests whether TensorRT was enabled during the configure process."""
  return %{if_tensorrt}

def if_tensorrt_v6(if_true, if_false=[]):
  return %{if_tensorrt_v6}

def if_tensorrt_v8(if_true, if_false=[]):
  return %{if_tensorrt_v8}
