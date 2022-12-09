from tensorflow import Variable
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Activation, Flatten
from tensorflow.keras.layers import MaxPooling2D, BatchNormalization
from tensorflow.keras.activations import relu as tfReLu
from tensorflow.keras import regularizers

from nn_utils import QConv2D

class YoloTiny(Model):
  def __init__(
      self,
      quantize,
      quantize_w_int,
      quantize_w,
      quantize_x_int,
      quantize_x):

    super().__init__(name = '')
    self.quantize = quantize
    self.quantize_w_int = quantize_w_int
    self.quantize_w = quantize_w
    self.quantize_x_int = quantize_x_int
    self.quantize_x = quantize_x
    self.alpha = Variable(0., trainable = False)

    self.conv1 = QConv2D(
        64, 3, 2,
        quantize = self.quantize,
        quantize_w_int = self.quantize_w_int,
        quantize_w = self.quantize_w,
        quantize_x_int = self.quantize_x_int,
        quantize_x = self.quantize_x,
        padding = [[0,0], [1,0], [1,0],[0,0]],#add one padding both on the top and left
        weight_decay = 0,
        use_bias = True,
        # use_bias = False,
        alpha = self.alpha)

    self.conv3 = QConv2D(
        64, 3, 2,
        quantize = self.quantize,
        quantize_w_int = self.quantize_w_int,
        quantize_w = self.quantize_w,
        quantize_x_int = self.quantize_x_int,
        quantize_x = self.quantize_x,
        padding = [[0,0], [1,0], [1,0],[0,0]],
        weight_decay = 0,
        use_bias = True,
        # use_bias = False,
        alpha = self.alpha)

    self.conv5 = QConv2D(
        64, 3, 2,
        quantize = self.quantize,
        quantize_w_int = self.quantize_w_int,
        quantize_w = self.quantize_w,
        quantize_x_int = self.quantize_x_int,
        quantize_x = self.quantize_x,
        padding = [[0,0], [1,0], [1,0],[0,0]],
        weight_decay = 0,
        use_bias = True,
        # use_bias = False,
        alpha = self.alpha)

    self.conv7 = QConv2D(
        128, 3, 2,
        quantize = self.quantize,
        quantize_w_int = self.quantize_w_int,
        quantize_w = self.quantize_w,
        quantize_x_int = self.quantize_x_int,
        quantize_x = self.quantize_x,
        padding = [[0,0], [1,0], [1,0],[0,0]],
        weight_decay = 0,
        use_bias = True,
        # use_bias = False,
        alpha = self.alpha)

    self.conv9 = QConv2D(
        256, 3, 2,
        quantize = self.quantize,
        quantize_w_int = self.quantize_w_int,
        quantize_w = self.quantize_w,
        quantize_x_int = self.quantize_x_int,
        quantize_x = self.quantize_x,
        padding = [[0,0], [1,0], [1,0],[0,0]],
        weight_decay = 0,
        use_bias = True,
        # use_bias = False,
        alpha = self.alpha)

    self.conv11 = QConv2D(
        512, 3, 1,
        quantize = self.quantize,
        quantize_w_int = self.quantize_w_int,
        quantize_w = self.quantize_w,
        quantize_x_int = self.quantize_x_int,
        quantize_x = self.quantize_x,
        weight_decay = 0,
        use_bias = True,
        # use_bias = False,
        alpha = self.alpha)

    self.conv13 = QConv2D(
        1024, 3, 1,
        quantize = self.quantize,
        quantize_w_int = self.quantize_w_int,
        quantize_w = self.quantize_w,
        quantize_x_int = self.quantize_x_int,
        quantize_x = self.quantize_x,
        weight_decay = 0,
        use_bias = True,
        # use_bias = False,
        alpha = self.alpha)

    self.conv14 = QConv2D(
        256, 1, 1,
        quantize = self.quantize,
        quantize_w_int = self.quantize_w_int,
        quantize_w = self.quantize_w,
        quantize_x_int = self.quantize_x_int,
        quantize_x = self.quantize_x,
        weight_decay = 0,
        use_bias = True,
        # use_bias = False,
        alpha = self.alpha)

    self.conv15 = QConv2D(
        512, 3, 1,
        quantize = self.quantize,
        quantize_w_int = self.quantize_w_int,
        quantize_w = self.quantize_w,
        quantize_x_int = self.quantize_x_int,
        quantize_x = self.quantize_x,
        weight_decay = 0,
        use_bias = True,
        # use_bias = False,
        alpha = self.alpha)

    self.conv16 = QConv2D(
        256, 1, 1,
        quantize = self.quantize,
        quantize_w_int = self.quantize_w_int,
        quantize_w = self.quantize_w,
        quantize_x_int = self.quantize_x_int,
        quantize_x = self.quantize_x,
        weight_decay = 0,
        use_bias = True,
        # use_bias = False,
        alpha = self.alpha)


  def call(self, input_tensor):
    x = input_tensor

    x_1  = self.conv1(x)
    x1 = tfReLu(x_1, alpha=0.015625)

    x  = self.conv3(x1)
    x2 = tfReLu(x, alpha=0.015625)

    x  = self.conv5(x2)
    x3 = tfReLu(x, alpha=0.015625)

    x  = self.conv7(x3)
    x4 = tfReLu(x, alpha=0.015625)

    x  = self.conv9(x4)
    x5 = tfReLu(x, alpha=0.015625)

    x  = self.conv11(x5)
    x6 = tfReLu(x, alpha=0.015625)

    x  = self.conv13(x6)
    x7 = tfReLu(x, alpha=0.015625)

    x  = self.conv14(x7)
    x8 = tfReLu(x, alpha=0.015625)

    x  = self.conv15(x8)
    x9 = tfReLu(x, alpha=0.015625)

    x10 = self.conv16(x9)

    return x10

def GenerateModel(
    quantize, quantize_w_int, quantize_w, quantize_x_int, quantize_x):
  return YoloTiny(
      quantize, quantize_w_int, quantize_w, quantize_x_int, quantize_x)
