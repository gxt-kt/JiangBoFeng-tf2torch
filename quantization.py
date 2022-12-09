import tensorflow as tf


@tf.custom_gradient
def QSign(x):
  output = tf.sign(x)

  def Grad(dy):
    return dy

  return output, Grad


@tf.custom_gradient
def QRound(x):
  output = tf.round(x)

  def Grad(dy):
    return dy

  return output, Grad


@tf.custom_gradient
def QFloor(x):
  output = tf.math.floor(x)

  def Grad(dy):
    return dy

  return output, Grad


def Round2Fixed(x, integer=16, k=32):
  assert integer >= 1, integer
  fraction = k - integer
  bound = tf.math.pow(2.0, integer - 1)
  n = tf.math.pow(2.0, fraction)
  min_val = -bound

  ### It SEEMS that `max_val` should be `bound-1`
  max_val = tf.dtypes.cast(bound- 1./tf.math.pow(2.,fraction), tf.float32)
  # max_val = bound
  # max_val = bound - 1

  # ??? Is this round function correct
  # It SEEMS that tf.floor will get same result as the FPGA output.
  # x_round = QFloor(x * n) / n
  x_round = QFloor(x * n) / n

  clipped_value = tf.clip_by_value(x_round, min_val, max_val)
  return clipped_value


def RoundPower2Exp(x, k=4):
  bound = tf.math.pow(2.0, k - 1)
  min_val = tf.math.pow(2.0, -bound + 1.0)
  s = tf.sign(x)

  # Temporary. `*x` and `/x` in order to avoid overflow.
  # x = tf.clip_by_value(tf.math.abs(x), min_val, 1.0)
  x = tf.clip_by_value(tf.math.abs(x * 64), min_val, 1.0)
  x = tf.clip_by_value(tf.math.abs(x / 64), min_val, 1.0)

  # Temporary. `*8` during inference and don't change during convert.
  # In fact, it should be `/8` during convert and don't change during inference.
  # p = QRound(tf.math.log(x) / tf.math.log(2.))
  p = QRound(tf.math.log(x * 8) / tf.math.log(2.))

  return s, p


def RoundPower2(x, k=4):
  s, p = RoundPower2Exp(x, k)

  return s * tf.math.pow(2.0, p)


def QuantizeFn(w_int, a_int, w_bit, a_bit, quantize):

  def QuantizeWeight(w):
    if w_bit == 1:   # BNN
      output = QSign(w)
    elif w_bit == 32:
      output = w
    else:   # QNN
      if quantize == 'shift':
        output = RoundPower2(w, w_bit)
      elif quantize == 'mul':
        output = Round2Fixed(w, w_int, w_bit)
      else:
        print('[ERROR][quantization.py] Wrong quantization type!!!')

    return output

  def QuantizeActivation(x):
    if a_bit == 1:   # BNN
      output = QSign(x)
    elif a_bit == 32:
      output = x
    else:   # QNN
      output = Round2Fixed(x, a_int, a_bit)

    return output

  return QuantizeWeight, QuantizeActivation


def tangent(x, x_quantilize, alpha):
  filters = \
      x_quantilize - alpha * tf.math.tanh(x - x_quantilize)
  return filters
