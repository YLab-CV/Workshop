import torch
from torch import nn

# from d2l import torch as d2l

# %% 1. Convolution Arithmetic
# 1.1  No Zero Padding, Unit Strides
# 1.1.1 手写 Normal Convolution: padding = 0, stride = 1


X_conv = torch.tensor(
    [[1.0, 2.0, 1.0],
     [2.0, 1.0, 3.0],
     [3.0, 1.0, 0.0]]
)
K = torch.tensor(
    [[1.0, 2.0],
     [0.0, 3.0]]
)


def conv(X, K):
    h, w = K.shape
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - h + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i:i + h, j:j + w] * K).sum()
    return Y


Y_conv = conv(X_conv, K)
print('--------手写 Normal Convolution: padding = 0, stride = 1--------\n', Y_conv)

# %% Reshape
X_conv = torch.reshape(X_conv, (1, 1, 3, 3))
K = torch.reshape(K, (1, 1, 2, 2))
# %% 1.1.2  torch  Normal Convolution: padding = 0, stride = 1

#   ---------------------PyTorch Conv2d---------------------
#   in_channels (int): Number of channels in the input image
#   out_channels (int): Number of channels produced by the convolution
#   kernel_size (int or tuple): Size of the convolving kernel
#   stride (int or tuple, optional): Stride of the convolution. Default: 1
#   padding (int, tuple or str, optional): Padding added to all four sides of the input. Default: 0
#   padding_mode (string, optional): ``'zeros'``, ``'reflect'``, ``'replicate'`` or ``'circular'``. Default: ``'zeros'``
#   dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
#   groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
#   bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``

torch_conv_p0_s1 = torch.nn.Conv2d(1, 1, kernel_size=2, stride=1, padding=0, dilation=1, bias=False)
torch_conv_p0_s1.weight.data = K
Y_torch_conv_p0_s1 = torch_conv_p0_s1(X_conv)
print('--------1.1.2  torch  Normal Convolution: padding = 0, stride = 1--------\n', Y_torch_conv_p0_s1)
# %% 1.2.1 Zero Padding, Unit Strides: padding = 1, stride = 1

torch_conv_p1_s1 = torch.nn.Conv2d(1, 1, kernel_size=2, stride=1, padding=1, dilation=1, bias=False)
torch_conv_p1_s1.weight.data = K
Y_torch_conv_p1_s1 = torch_conv_p1_s1(X_conv)
print('--------1.2.1 Zero Padding, Unit Strides: padding = 1, stride = 1--------\n', Y_torch_conv_p1_s1)
# %% 1.2.2  Same Padding: stride = 1
K_same = torch.tensor(
    [[0.0, 1.0, 1.0],
     [2.0, 1.0, 0.0],
     [1.0, 2.0, 0.0]]
)
K_same = torch.reshape(K_same, (1, 1, 3, 3))
torch_conv_p1_s1 = torch.nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, dilation=1, bias=False)
torch_conv_p1_s1.weight.data = K_same
Y_torch_conv_same = torch_conv_p1_s1(X_conv)
print('--------1.2.2  Same Padding: stride = 1--------\n', Y_torch_conv_same)

# %% 1.3.1.1 No Zero Padding, Non-unit Strides Stride=2, Padding=0 Eg1
X_conv_s2_p0 = torch.tensor(
    [[1.0, 2.0, 3.0, 4.0],
     [0.0, 0.0, 1.0, 1.0],
     [2.0, 4.0, 3.0, 2.0],
     [0.0, 1.0, 1.0, 3.0]]
)
K_s2_p0 = torch.tensor(
    [[3.0, 0.0],
     [2.0, 1.0]]
)

X_conv_s2_p0 = torch.reshape(X_conv_s2_p0, (1, 1, 4, 4))
K_s2_p0 = torch.reshape(K_s2_p0, (1, 1, 2, 2))
torch_conv_p1_s1 = torch.nn.Conv2d(1, 1, kernel_size=2, stride=2, padding=0, dilation=1, bias=False)
torch_conv_p1_s1.weight.data = K_s2_p0
Y_torch_conv_same = torch_conv_p1_s1(X_conv_s2_p0)
print('--------1.3.1.1 No Zero Padding, Non-unit Strides Stride=2, Padding=0 Eg1--------\n', Y_torch_conv_same)
# %% 1.3.1.2 No Zero Padding, Non-unit Strides Stride=2, Padding=0  Eg2
X_conv_s2_p0 = torch.tensor(
    [[1.0, 2.0, 3.0, 4.0, 0.0],
     [0.0, 0.0, 1.0, 1.0, 1.0],
     [2.0, 4.0, 3.0, 2.0, 1.0],
     [0.0, 1.0, 1.0, 3.0, 0.0],
     [0.0, 0.0, 1.0, 0.0, 1.0]]
)
K_s2_p0 = torch.tensor(
    [[3.0, 0.0],
     [2.0, 1.0]]
)

X_conv_s2_p0 = torch.reshape(X_conv_s2_p0, (1, 1, 5, 5))
K_s2_p0 = torch.reshape(K_s2_p0, (1, 1, 2, 2))
torch_conv_p1_s1 = torch.nn.Conv2d(1, 1, kernel_size=2, stride=2, padding=0, dilation=1, bias=False)
torch_conv_p1_s1.weight.data = K_s2_p0
Y_torch_conv_same = torch_conv_p1_s1(X_conv_s2_p0)
print('--------1.3.1.2 No Zero Padding, Non-unit Strides Stride=2, Padding=0  Eg2--------\n', Y_torch_conv_same)
# %% 1.4.1 Zero Padding, Non-unit Strides Stride=2, Padding=2


K_same_s2_p2 = torch.tensor(
    [[0.0, 1.0, 1.0],
     [2.0, 1.0, 0.0],
     [1.0, 2.0, 0.0]]
)
K_same_s2_p2 = torch.reshape(K_same_s2_p2, (1, 1, 3, 3))
torch_conv_p1_s1 = torch.nn.Conv2d(1, 1, kernel_size=2, stride=2, padding=2, dilation=1, bias=False)
torch_conv_p1_s1.weight.data = K_same_s2_p2
Y_torch_conv_same = torch_conv_p1_s1(X_conv)
print('--------1.4.1 Zero Padding, Non-unit Strides Stride=2, Padding=2--------\n', Y_torch_conv_same)

# %% 2.2.1.1 No Zero Padding, Unit Strides, Transposed: Stride=1, Padding=0 手写转置卷积
X_trans = torch.tensor(
    [[8.0, 13.0],
     [7.0, 7.0]]
)
K = torch.tensor(
    [[1.0, 2.0],
     [0.0, 3.0]]
)


def trans_conv(X, K):
    h, w = K.shape
    Y = torch.zeros((X.shape[0] + h - 1, X.shape[1] + w - 1))
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Y[i:i + h, j:j + w] += X[i, j] * K
    return Y


Y_trans = trans_conv(X_trans, K)
print('--------2.2.1.1 No Zero Padding, Unit Strides, Transposed: Stride=1, Padding=0 手写转置卷积 --------\n', Y_trans)

X_trans = torch.tensor(
    [[6.0, 70],
     [5.0, 7.0]]
)
print(X_trans.shape)
print(K.shape)
X_trans = torch.reshape(X_trans, (1, 1, 2, 2))
K = torch.reshape(K, (1, 1, 2, 2))
print(X_trans.shape)
print(K.shape)

#   ---------------------PyTorch ConvTranspose2d---------------------
# torch.nn.ConvTranspose2d(
# in_channels,
# out_channels,
# kernel_size,
# stride=1,
# padding=0,
# output_padding=0,
# groups=1,
# bias=True,
# dilation=1,
# padding_mode='zeros',
# device=None, dtype=None)


# %% 2.2.1.2 No Zero Padding, Unit Strides, Transposed: Stride=1, Padding=0
torch_trans_conv = nn.ConvTranspose2d(1, 1, kernel_size=2, stride=1, padding=0, output_padding=0, dilation=1,
                                      bias=False)
torch_trans_conv.weight.data = K
Y_torch_trans_s1_p0 = torch_trans_conv(X_trans)
print('--------2.2.1.2 No Zero Padding, Unit Strides, Transposed: Stride=1, Padding=0--------\n', Y_torch_trans_s1_p0)

# %% 2.3.1 Zero Padding, Unit Strides, Transposed: Stride=1, Padding=1
torch_trans_conv_s1_p1 = nn.ConvTranspose2d(1, 1, kernel_size=2, stride=1, padding=1, output_padding=0, dilation=1,
                                            bias=False)
torch_trans_conv_s1_p1.weight.data = K
Y_torch_trans_s1_p1 = torch_trans_conv_s1_p1(X_trans)
print('--------2.3.1 Zero Padding, Unit Strides, Transposed: Stride=1, Padding=1--------\n', Y_torch_trans_s1_p1)
# %% 2.3.2 Zero Padding, Unit Strides, Transposed: Stride=1, Padding=2
X_trans_s1_p2 = torch.tensor(
    [[1.0, 0.0, 2.0, 1.0, 1.0],
     [2.0, 1.0, 0.0, 1.0, 3.0],
     [3.0, 0.0, 0.0, 0.0, 1.0],
     [1.0, 0.0, 3.0, 0.0, 2.0],
     [2.0, 2.0, 1.0, 1.0, 1.0]]
)
K_s1_p2 = torch.tensor(
    [[1.0, 2.0, 3.0, 4.0],
     [5.0, 4.0, 3.0, 2.0],
     [0.0, 1.0, 2.0, 3.0],
     [1.0, 1.0, 3.0, 0.0]]
)
X_trans_s1_p2 = torch.reshape(X_trans_s1_p2, (1, 1, 5, 5))
K_s1_p2 = torch.reshape(K_s1_p2, (1, 1, 4, 4))
torch_trans_conv_s1_p2 = nn.ConvTranspose2d(1, 1, kernel_size=4, stride=1, padding=2, output_padding=0, dilation=1,
                                            bias=False)
torch_trans_conv_s1_p2.weight.data = K_s1_p2
Y_torch_trans_s1_p2 = torch_trans_conv_s1_p2(X_trans_s1_p2)
print('--------2.3.2 Zero Padding, Unit Strides, Transposed: Stride=1, Padding=2--------\n', Y_torch_trans_s1_p2)
# %% 2.3.3 Zero Padding, Unit Strides, Transposed: Same Padding
X_trans_Same = torch.tensor(
    [[1.0, 2.0, 3.0],
     [0.0, 4.0, 1.0],
     [2.0, 1.0, 0.0]]
)
K_Same = torch.tensor(
    [[0.0, 1.0, 1.0],
     [2.0, 1.0, 0.0],
     [1.0, 2.0, 0.0]]
)
X_trans_Same = torch.reshape(X_trans_Same, (1, 1, 3, 3))
K_Same = torch.reshape(K_Same, (1, 1, 3, 3))
torch_trans_Same = nn.ConvTranspose2d(1, 1, kernel_size=3, stride=1, padding=0, output_padding=0, dilation=1,
                                      bias=False)
torch_trans_Same.weight.data = K_Same
Y_torch_trans_Same = torch_trans_Same(X_trans_Same)
print('--------2.3.3 Zero Padding, Unit Strides, Transposed: Same Padding--------\n', Y_torch_trans_Same)
# %% 2.4.1.1 No Zero Padding, Non-unit Strides, Transposed: Stride=2, Padding=0, E.g. 1
torch_trans_conv_s2_p0 = nn.ConvTranspose2d(1, 1, kernel_size=2, stride=2, padding=0, output_padding=0, dilation=1,
                                            bias=False)
torch_trans_conv_s2_p0.weight.data = K
Y_torch_trans_s2_p0 = torch_trans_conv_s2_p0(X_trans)
print('--------2.4.1.1 No Zero Padding, Non-unit Strides, Transposed: Stride=2, Padding=0--------\n',
      Y_torch_trans_s2_p0)
# %% 2.4.1.2 No Zero Padding, Non-unit Strides, Transposed: Stride=2, Padding=0, E.g. 2

X_trans_s2_p0 = torch.tensor(
    [[1.0, 2.0, 3.0],
     [0.0, 4.0, 1.0],
     [2.0, 1.0, 0.0]]
)
K_s2_p0 = torch.tensor(
    [[1.0, 2.0],
     [0.0, 3.0]]
)
X_trans_s2_p0 = torch.reshape(X_trans_s2_p0, (1, 1, 3, 3))
K_s2_p0 = torch.reshape(K_s2_p0, (1, 1, 2, 2))
torch_trans_conv_s2_p0 = nn.ConvTranspose2d(1, 1, kernel_size=2, stride=2, padding=0, output_padding=0, dilation=1,
                                            bias=False)
torch_trans_conv_s2_p0.weight.data = K_s2_p0
Y_torch_trans_s2_p0 = torch_trans_conv_s2_p0(X_trans_s2_p0)
print('--------2.4.1.2 No Zero Padding, Non-unit Strides, Transposed: Stride=2, Padding=0, E.g. 2--------\n',
      Y_torch_trans_s2_p0)
# %% 2.5.1 Zero Padding, Non-unit Strides, Transposed: Stride=2, Padding=1
torch_trans_conv_s2_p1 = nn.ConvTranspose2d(1, 1, kernel_size=2, stride=2, padding=1, output_padding=0, dilation=1,
                                            bias=False)
torch_trans_conv_s2_p1.weight.data = K
Y_torch_trans_s2_p1 = torch_trans_conv_s2_p1(X_trans)
print('--------2.5.1 Zero Padding, Non-unit Strides, Transposed: Stride=2, Padding=1--------\n', Y_torch_trans_s2_p1)
# %% 2.6.1 Output Padding: output_padding = 1, stride = 2, padding = 0
torch_trans_conv_s2_op1 = nn.ConvTranspose2d(1, 1, kernel_size=2, stride=2, padding=0, output_padding=1, dilation=1,
                                             bias=False)
torch_trans_conv_s2_op1.weight.data = K
Y_torch_trans_s2_op1 = torch_trans_conv_s2_op1(X_trans)
print('-------- padding = 0, stride = 2, output_padding=1--------\n', Y_torch_trans_s2_op1)
