import math
import torch.nn as nn
from torch.nn.modules.utils import _pair
import numpy as np
import torch

class Net(nn.Module):
    def __init__(self, classes=10, kernel = 3, channels=3, scaling =1.0):
        super(Net, self).__init__()
        self.kernel = kernel
        self.channels = channels
        self.conv = nn.Conv2d(3,self.channels,self.kernel,1) 
        self.conv.weight.data = self.conv.weight.data * scaling
        self.conv.bias.data = self.conv.bias.data * scaling
        self.pad = math.ceil((32 - (32 - self.kernel + 1))/2)
        self.linear = nn.Linear(self.channels*(32 - self.kernel + 1)*(32 - self.kernel + 1),10)
        self.linear.weight.data = self.linear.weight.data/scaling
        self.flat = nn.Flatten()

    def forward(self, x):
        x = self.conv(x)
        x = self.flat(x)
        x = self.linear(x)
        return x
    
class NetR(nn.Module):
    def __init__(self, classes=10, kernel = 3, channels=3, scaling=1.0):
        super(NetR, self).__init__()
        self.kernel = kernel
        self.channels = channels
        self.conv = nn.Conv2d(3,self.channels,self.kernel,1) 
        self.conv.weight.data = self.conv.weight.data * scaling
        self.conv.bias.data = self.conv.bias.data * scaling
        self.pad = math.ceil((32 - (32 - self.kernel + 1))/2)
        self.linear = nn.Linear(self.channels*(32 - self.kernel + 1)*(32 - self.kernel + 1),10)
        self.linear.weight.data = self.linear.weight.data/scaling
        self.flat = nn.Flatten()

    def forward(self, x):
        x = self.conv(x)
        x = torch.nn.functional.relu(x)
        x = self.flat(x)
        x = self.linear(x)
        return x

class DilNetR(nn.Module):
    def __init__(self, classes=10, kernel = 3, channels=3, scaling=1.0):
        super(DilNetR, self).__init__()
        self.kernel = kernel
        self.channels = channels
        self.conv = nn.Conv2d(3,self.channels,self.kernel,1, dilation=1) 
        self.conv.weight.data = self.conv.weight.data * scaling
        self.conv.bias.data = self.conv.bias.data * scaling
        self.linear = nn.Linear(self.channels*(32 - self.kernel + 1)*(32 - self.kernel + 1),10)
        self.linear.weight.data = self.linear.weight.data/scaling
        self.flat = nn.Flatten()

    def forward(self, x):
        x = self.conv(x)
        x = torch.nn.functional.relu(x)
        x = self.flat(x)
        x = self.linear(x)
        return x

class LocallyConnected2d(nn.Module):
    def __init__(self, in_channels, out_channels, output_size, kernel_size, stride, padding=1, bias=False, conv_init = False):
        super(LocallyConnected2d, self).__init__()
        output_size = _pair(output_size)
        self.conv_init = conv_init
        useless = torch.nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=1, bias=False)
        if conv_init == True:
            self.weight = torch.stack([useless.weight.data for i in range((output_size[0])*(output_size[1]))], axis=2)
            self.weight = self.weight.reshape(1, out_channels, in_channels, output_size[0], output_size[1], kernel_size**2)
            self.weight = nn.Parameter(self.weight)
        else:
            self.weight = torch.randn(1, out_channels, in_channels, output_size[0], output_size[1], kernel_size**2)
            torch.nn.init.normal(self.weight,mean= useless.weight.mean(), std = useless.weight.std())
            self.weight = nn.Parameter(self.weight)
            
        if bias:
            self.bias = nn.Parameter(
                torch.randn(1, out_channels, output_size[0])
            )
            self.bias = torch.stack([self.bias for i in range(output_size[1])],axis=3)
        else:
            self.register_parameter('bias', None)
        self.padding = padding
​
​
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
​
    def forward(self, x):
        x = torch.nn.functional.pad(x, (self.padding, self.padding, self.padding, self.padding), mode='constant', value=0)
        _, c, h, w = x.size()
        kh, kw = self.kernel_size
        dh, dw = self.stride
        x = x.unfold(2, kh, dh).unfold(3, kw, dw)
        x = x.contiguous().view(*x.size()[:-2], -1)
        # Sum in in_channel and kernel_size dims
        out = (x.unsqueeze(1) * self.weight).sum([2, -1])
        if self.bias is not None:
            out += self.bias
        return out
    
class Net3(nn.Module):
    def __init__(self, classes=10, kernel = 3, channels=3, scaling=1.0):
        super(Net3, self).__init__()
        self.kernel = kernel
        self.channels = channels
        
        self.conv1 = nn.Conv2d(3,self.channels,self.kernel,1)
        self.conv1.weight.data = self.conv1.weight.data * scaling
        self.conv1.bias.data = self.conv1.bias.data * scaling
        self.pad1 = math.ceil((32 - (32 - self.kernel + 1))/2)
        self.output1 = int((32 - self.kernel + 1))
        
        self.conv2 = nn.Conv2d(self.channels, self.channels, self.kernel,1) 
        self.conv2.weight.data = self.conv2.weight.data * scaling
        self.conv2.bias.data = self.conv2.bias.data * scaling
        self.pad2 = math.ceil((self.output1 - (self.output1 - self.kernel + 1))/2)
        self.output2 = int((self.output1 - self.kernel + 1))
        
        self.conv3 = nn.Conv2d(self.channels, self.channels, self.kernel,1) 
        self.conv3.weight.data = self.conv3.weight.data * scaling
        self.conv3.bias.data = self.conv3.bias.data * scaling
        self.pad3 = math.ceil((self.output2 - (self.output2 - self.kernel + 1))/2)
        self.output3 = int((self.output2 - self.kernel + 1))
        
        self.linear = nn.Linear(self.channels*(self.output2 - self.kernel + 1)*(self.output2 - self.kernel + 1),10)
        self.linear.weight.data/=scaling**3
        self.flat = nn.Flatten()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flat(x)
        x = self.linear(x)
        return x

    

class Net5(nn.Module):
    def __init__(self, classes=10, kernel = 3, channels=3):
        super(Net3, self).__init__()
        self.kernel = kernel
        self.channels = channels
        
        self.conv1 = nn.Conv2d(3,self.channels,self.kernel,1) 
        self.pad1 = math.ceil((32 - (32 - self.kernel + 1))/2)
        self.output1 = int((32 - self.kernel + 1 + self.pad1*2))
        
        self.conv2 = nn.Conv2d(self.channels, self.channels, self.kernel,1) 
        self.pad2 = math.ceil((self.output1 - (self.output1 - self.kernel + 1))/2)
        self.output2 = int((self.output1 - self.kernel + 1 + self.pad2*2))
        
        self.conv3 = nn.Conv2d(self.channels, self.channels, self.kernel,1) 
        self.pad3 = math.ceil((self.output2 - (self.output2 - self.kernel + 1))/2)
        self.output3 = int((self.output2 - self.kernel + 1 + self.pad3*2))
        
        self.conv4 = nn.Conv2d(self.channels, self.channels, self.kernel,1) 
        self.pad4 = math.ceil((self.output3 - (self.output1 - self.kernel + 1))/2)
        self.output4 = int((self.output1 - self.kernel + 1 + self.pad4*2))
        
        self.conv5 = nn.Conv2d(self.channels, self.channels, self.kernel,1) 
        self.pad5 = math.ceil((self.output4 - (self.output2 - self.kernel + 1))/2)
        self.output5 = int((self.output2 - self.kernel + 1 + self.pad5*2))
        
        self.linear = nn.Linear(self.channels*(self.output4 - self.kernel + 1 + self.pad5*2)*(self.output4 - self.kernel + self.pad5*2 + 1),10)
        self.flat = nn.Flatten()

    def forward(self, x):
        x = torch.nn.functional.pad(x,(self.pad1,self.pad1,self.pad1,self.pad1),'circular')
        x = self.conv1(x)
        x = torch.nn.functional.pad(x,(self.pad2,self.pad2,self.pad2,self.pad2),'circular')
        x = self.conv2(x)
        x = torch.nn.functional.pad(x,(self.pad3,self.pad3,self.pad3,self.pad3),'circular')
        x = self.conv3(x)
        x = torch.nn.functional.pad(x,(self.pad4,self.pad4,self.pad4,self.pad4),'circular')
        x = self.conv4(x)
        x = torch.nn.functional.pad(x,(self.pad5,self.pad5,self.pad5,self.pad5),'circular')
        x = self.conv5(x)
        x = self.flat(x)
        x = self.linear(x)
        return x

class Net5R(nn.Module):
    def __init__(self, classes=10, kernel = 3, channels=3):
        super(Net3, self).__init__()
        self.kernel = kernel
        self.channels = channels
        
        self.conv1 = nn.Conv2d(3,self.channels,self.kernel,1) 
        self.pad1 = math.ceil((32 - (32 - self.kernel + 1))/2)
        self.output1 = int((32 - self.kernel + 1 + self.pad1*2))
        
        self.conv2 = nn.Conv2d(self.channels, self.channels, self.kernel,1) 
        self.pad2 = math.ceil((self.output1 - (self.output1 - self.kernel + 1))/2)
        self.output2 = int((self.output1 - self.kernel + 1 + self.pad2*2))
        
        self.conv3 = nn.Conv2d(self.channels, self.channels, self.kernel,1) 
        self.pad3 = math.ceil((self.output2 - (self.output2 - self.kernel + 1))/2)
        self.output3 = int((self.output2 - self.kernel + 1 + self.pad3*2))
        
        self.conv4 = nn.Conv2d(self.channels, self.channels, self.kernel,1) 
        self.pad4 = math.ceil((self.output3 - (self.output1 - self.kernel + 1))/2)
        self.output4 = int((self.output1 - self.kernel + 1 + self.pad4*2))
        
        self.conv5 = nn.Conv2d(self.channels, self.channels, self.kernel,1) 
        self.pad5 = math.ceil((self.output4 - (self.output2 - self.kernel + 1))/2)
        self.output5 = int((self.output2 - self.kernel + 1 + self.pad5*2))
        
        self.linear = nn.Linear(self.channels*(self.output4 - self.kernel + 1 + self.pad5*2)*(self.output4 - self.kernel + self.pad5*2 + 1),10)
        self.flat = nn.Flatten()

    def forward(self, x):
        x = torch.nn.functional.pad(x,(self.pad1,self.pad1,self.pad1,self.pad1),'circular')
        x = self.conv1(x)
        x = torch.nn.functional.relu(x)
        x = torch.nn.functional.pad(x,(self.pad2,self.pad2,self.pad2,self.pad2),'circular')
        x = self.conv2(x)
        x = torch.nn.functional.relu(x)
        x = torch.nn.functional.pad(x,(self.pad3,self.pad3,self.pad3,self.pad3),'circular')
        x = self.conv3(x)
        x = torch.nn.functional.relu(x)
        x = torch.nn.functional.pad(x,(self.pad4,self.pad4,self.pad4,self.pad4),'circular')
        x = self.conv4(x)
        x = torch.nn.functional.relu(x)
        x = torch.nn.functional.pad(x,(self.pad5,self.pad5,self.pad5,self.pad5),'circular')
        x = self.conv5(x)
        x = torch.nn.functional.relu(x)
        x = self.flat(x)
        x = self.linear(x)
        return x

def random_pos_channel(pos_in, connect_type):
    r"""Randomizes input positions for one channel.

    Args
    ----
    pos_in: (K, S_out), array_like
        The input position index describing a normal convolution connectivity.
        `K` is the number of kernel parameters for a fixed input-output channel
        pair, i.e. `kernel_size**2`. `S_out` is the number of output positions,
        i.e. `H_out*W_out`. `pos_in[k, pos_out]` is the input position index
        of the parameter `k` connecting to `pos_out`.
    connect_type: str
        The sparse connection type, can be ``'normal'``, ``'shuffle'`` or
        ``'scatter'``.

    Returns
    -------
    pos_in: (K, S_out), array_like
        The randomized input position index. No change will be made for
        ``'normal'``. Locality is preserved for ``'shuffle'`` but not for
        ``'scatter'``.

    """
    if connect_type=='shuffle':
        pos_in = np.stack([np.random.permutation(pos_in[:, i]) for i in range(pos_in.shape[1])], axis=1)
    elif connect_type=='scatter':
        pos_in = np.stack([np.random.permutation(pos_in[i]) for i in range(pos_in.shape[0])], axis=0)
    return pos_in

def random_pos_layer(pos_in, connect_type, in_channels, out_channels,
                     in_consistent=True, out_consistent=True):
    r"""Randomizes input positions for one layer.

    Args
    ----
    pos_in: (K, S_out), array_like
        The input position index describing a normal convolution connectivity.
    connect_type: str
        The sparse connection type, can be ``'normal'``, ``'shuffle'`` or
        ``'scatter'``.
    in_channels, out_channels: int
        The number of input and output channels.
    in_consistent, out_consistent: bool
        Whether the spatial pattern is consistent across input or output
        channels.

    Returns
    -------
    pos_in: (in_channels, out_channels, K, S_out), array_like
        The randomized input position index. `pos_in[c_i, c_o, k, pos_out]` is
        the input position index of the parameter `k` connectiing to output
        output position `pos_out` from channel `c_i` to channel `c_o`.

    """
    K, S_out = pos_in.shape
    pos_in = np.array([[
        random_pos_channel(pos_in, connect_type) for _ in range(in_channels if in_consistent else 1)
        ] for _ in range(out_channels if out_consistent else 1)
        ])
    pos_in = np.broadcast_to(pos_in, (out_channels, in_channels, K, S_out)).copy()
    return pos_in

def sparse_support(H_in, W_in, in_channels, out_channels, kernel_size,
                   stride=1, padding=1, padding_mode='zeros',
                   connect_type='normal', in_consistent=True, out_consistent=True):
    r"""Constructs support for a sparse 2D convolution.

    The algorithm starts from a normal convolution connectivity and randomizes
    the input end of each connection. All valid connections are gathered by
    their input indices and output indices in the flattened tensor.

    Args
    ----
    H_in, W_in: int
        The height and width of inputs.
    in_channels, out_channels: int
        The number of input and output channels.
    kernel_size, stride, padding: int
        The kernel size, stride and padding of a baseline convolution.
    padding_mode: str
        Thw padding mode, can be ``'zeros'`` or ``'circular'``.
    connect_type: str
        The sparse connection type, can be ``'normal'``, ``'shuffle'`` or
        ``'scatter'``.
    in_consistent, out_consistent: bool
        Whether the spatial pattern is consistent across input or output
        channels.

    Returns
    -------
    H_out, W_out: int
        The height and width of outputs.
    w_idxs: (3, *), array_like
        Coordinate list of weight parameter index. Each column of `w_idxs` is
        `(pos_in, pos_out, k)`.
    w_size: tuple
        The size of sparse weight matrix, `(n_in, n_out)`.

    """
    H_out = (H_in+2*padding-(kernel_size-1)-1)//stride+1
    W_out = (W_in+2*padding-(kernel_size-1)-1)//stride+1
    n_in, n_out = in_channels*H_in*W_in, out_channels*H_out*W_out

    c_out, c_in, param_idx, pos_out = np.meshgrid(
        np.arange(out_channels), np.arange(in_channels),
        np.arange(kernel_size**2), np.arange(H_out*W_out), indexing='ij'
        )
    param_idx += (c_out*in_channels+c_in)*(kernel_size**2)

    y_out, x_out = np.unravel_index(np.arange(H_out*W_out), (H_out, W_out))
    dy, dx = np.unravel_index(np.arange(kernel_size**2), (kernel_size, kernel_size))
    y_in = y_out[None]*stride+dy[:, None]
    x_in = x_out[None]*stride+dx[:, None]
    pos_in = np.ravel_multi_index((y_in, x_in), (H_in+padding*2, W_in+padding*2))
    pos_in = random_pos_layer(
        pos_in, connect_type, in_channels, out_channels,
        in_consistent, out_consistent
        )
    y_in, x_in = np.unravel_index(pos_in, (H_in+2*padding, W_in+2*padding))
    y_in, x_in = y_in-padding, x_in-padding
    if padding_mode=='circular':
        y_in = y_in%H_in
        x_in = x_in%W_in

    y_out, x_out = np.unravel_index(pos_out, (H_out, W_out))

    valid_mask = np.all([y_in>=0, y_in<H_in, x_in>=0, x_in<W_in], axis=0)
    idx_in = np.ravel_multi_index(
        (c_in[valid_mask], y_in[valid_mask], x_in[valid_mask]),
        (in_channels, H_in, W_in)
        )
    idx_out = np.ravel_multi_index(
        (c_out[valid_mask], y_out[valid_mask], x_out[valid_mask]),
        (out_channels, H_out, W_out)
        )
    w_idxs = np.stack([idx_out, idx_in, param_idx[valid_mask]])
    w_size = (n_out, n_in)
    return H_out, W_out, w_idxs, w_size

    
class SparseConv2d(nn.Module):

    def __init__(
            self, H_in, W_in, in_channels, out_channels, kernel_size,
            stride=1, padding=0, bias=False, padding_mode='zeros',
            connect_type='normal', in_consistent=True, out_consistent=True):
        super(SparseConv2d, self).__init__()
        self.H_in, self.W_in = H_in, W_in
        self.in_channels, self.out_channels = in_channels, out_channels
        self.kernel_size, self.stride, self.padding = kernel_size, stride, padding
        self.padding_mode = padding_mode
        self.connect_type = connect_type
        self.in_consistent, self.out_consistent = in_consistent, out_consistent

        _conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.weight = nn.Parameter(
            _conv.weight.data.reshape(-1), requires_grad=True,
            )
        if bias:
            self.bias = nn.Parameter(
                torch.zeros(out_channels), requires_grad=True,
                )
        else:
            self.bias = None

        self.H_out, self.W_out, w_idxs, self.w_size = sparse_support(
                H_in, W_in, in_channels, out_channels, kernel_size,
                stride, padding, padding_mode,
                connect_type, in_consistent, out_consistent
                )
        self.pos_idxs = nn.Parameter(
            torch.LongTensor(w_idxs[:2]), requires_grad=False
            )
        self.param_idxs = nn.Parameter(
            torch.LongTensor(w_idxs[2]), requires_grad=False,
            )

    def extra_repr(self):
        return '\n'.join([
            '{}, {}, kernel_size={}, stride={}, padding={},'.format(
                self.in_channels, self.out_channels,
                self.kernel_size, self.stride, self.padding),
            '{}, H_in={}, W_in={}, H_out={}, W_out={}'.format(
                self.connect_type, self.H_in, self.W_in, self.H_out, self.W_out),
            ])

    def forward(self, inputs):
        _, C, H, W = inputs.shape
        assert C==self.in_channels and H==self.H_in and W==self.W_in, 'incompatible shape'
        s_weight = torch.sparse.FloatTensor(
            self.pos_idxs, self.weight[self.param_idxs],
            torch.Size(self.w_size)
            )
        inputs = inputs.view(-1, self.in_channels*self.H_in*self.W_in)
        outputs = torch.sparse.mm(s_weight, inputs.t()).t()
        outputs = outputs.view(-1, self.out_channels, self.H_out, self.W_out)
        if self.bias is not None:
            outputs = outputs+self.bias[:, None, None]
        return outputs

    
class Net3R(nn.Module):
    def __init__(self, classes=10, kernel = 3, channels=3, scaling=1.0):
        super(Net3R, self).__init__()
        self.kernel = kernel
        self.channels = channels
        
        self.conv1 = nn.Conv2d(3,self.channels,self.kernel,1)
        self.conv1.weight.data = self.conv1.weight.data * scaling
        self.conv1.bias.data = self.conv1.bias.data * scaling
        self.pad1 = math.ceil((32 - (32 - self.kernel + 1))/2)
        self.output1 = int((32 - self.kernel + 1 ))
        
        self.conv2 = nn.Conv2d(self.channels, self.channels, self.kernel,1) 
        self.conv2.weight.data = self.conv2.weight.data * scaling
        self.conv2.bias.data = self.conv2.bias.data * scaling
        self.pad2 = math.ceil((self.output1 - (self.output1 - self.kernel + 1))/2)
        self.output2 = int((self.output1 - self.kernel + 1 ))
        
        self.conv3 = nn.Conv2d(self.channels, self.channels, self.kernel,1) 
        self.conv3.weight.data = self.conv3.weight.data * scaling
        self.conv3.bias.data = self.conv3.bias.data * scaling
        self.pad3 = math.ceil((self.output2 - (self.output2 - self.kernel + 1))/2)
        self.output3 = int((self.output2 - self.kernel + 1 ))
        
        self.linear = nn.Linear(self.channels*(self.output2 - self.kernel + 1 )*(self.output2 - self.kernel + 1),10)
        self.linear.weight.data/=scaling**3
        self.flat = nn.Flatten()
    def forward(self, x):
        x = self.conv1(x)
        x = torch.nn.functional.relu(x)
        x = self.conv2(x)
        x = torch.nn.functional.relu(x)
        x = self.conv3(x)
        x = torch.nn.functional.relu(x)
        x = self.flat(x)
        x = self.linear(x)
        return x


def return_model(name='fclinearl1'):
    if name == 'fclinearl3':
        model = nn.Sequential(nn.Flatten(), nn.Linear(3*32*32,3*32*32),nn.Linear(3*32*32,3*32*32), nn.Linear(3*32*32,3*32*32), nn.Linear(3*32*32,10))
    elif name == 'fclinearl1':
        model = nn.Sequential(nn.Flatten(),nn.Linear(3*32*32,3*32*32), nn.Linear(3*32*32,10))
    elif name == 'fclinearl1relu':
        model = nn.Sequential(nn.Flatten(),nn.Linear(3*32*32,3*32*32), nn.ReLU(), nn.Linear(3*32*32,10))
    elif name == 'fclinearl3relu':
        model = nn.Sequential(nn.Flatten(), nn.Linear(3*32*32,3*32*32),nn.ReLU(),nn.Linear(3*32*32,3*32*32),nn.ReLU(), nn.Linear(3*32*32,3*32*32),nn.ReLU(), nn.Linear(3*32*32,10))
    elif name == 'convlinearl1k3c3':
        model = Net(kernel=3,channels=3)
    elif name == 'convlinearl3k3c3':
        model = Net3(kernel=3,channels=3)
    elif name == 'convlinearl1k11c3':
        model = Net(kernel=11,channels=3)
    elif name == 'convlinearl3k11c3':
        model = Net3(kernel=11, channels=3)
    elif name == 'convlinearl1k32c3':
        model = Net(kernel=32, channels=3)
    elif name == 'convlinearl3k32c3':
        model = Net3(kernel=32, channels=3)
    elif name == 'convlinearl1k3c8':
        model = Net(kernel=3,channels=8)
    elif name == 'convlinearl1k3c32':
        model = Net(kernel=3, channels=32)
    elif name == 'convlinearl3k3c8':
        model = Net3(kernel=3, channels=8)
    elif name == 'convlinearl3k3c32':
        model = Net3(kernel=3, channels=32)
    elif name == 'convlinearl1k3c3relu':
        model = NetR(kernel=3,channels=3)
    elif name == 'convlinearl3k3c3relu':
        model = Net3R(kernel=3,channels=3)
    elif name == 'convlinearl1k11c3relu':
        model = NetR(kernel=11,channels=3)
    elif name == 'convlinearl3k11c3relu':
        model = Net3R(kernel=11, channels=3)
    elif name == 'convlinearl1k32c3relu':
        model = NetR(kernel=32, channels=3)
    elif name == 'convlinearl3k32c3relu':
        model = Net3R(kernel=32, channels=3)
    elif name == 'convlinearl1k3c8relu':
        model = NetR(kernel=3,channels=8)
    elif name == 'convlinearl1k3c32relu':
        model = NetR(kernel=3, channels=32)
    elif name == 'convlinearl1k3c32reludil':
        model = DilNetR(kernel=3, channels=32)
    elif name == 'sparselinearl1k3c32relu':
        model = nn.Sequential(SparseConv2d(32,32,3,32,3), nn.Flatten(), nn.Linear(32*30*30,10))
    elif name == 'convlinearl1k32c32':
        model = Net(kernel=32, channels=32)
    elif name == 'convlinearl1k32c32relu':
        model = NetR(kernel=32, channels=32)
    elif name == 'convlinearl3k3c8relu':
        model = Net3R(kernel=3, channels=8)
    elif name == 'convlinearl3k3c32relu':
        model = Net3R(kernel=3, channels=32)
    elif name == 'linear':
        model = nn.Sequential(nn.Flatten(), nn.Linear(3*32*32,10))
    elif name == 'lclinearl1k3c32':
        model = nn.Sequential(LocallyConnected2dN(3, 32, 32, 3, 1), nn.Flatten(), nn.Linear(32*32*32,10))
    elif name == 'lclinearl1k3c3201':
        model = nn.Sequential(LocallyConnected2dN(3, 32, 32, 3, 1), nn.Flatten(), nn.Linear(32*32*32,10))
        model[0].weight.data[:]/=10.0
        #model[0].bias.data[:]/=10.0
        model[2].weight.data[:]*=10.0
    elif name == 'lclinearl1k3c3210':
        model = nn.Sequential(LocallyConnected2dN(3, 32, 32, 3, 1), nn.Flatten(), nn.Linear(32*32*32,10))
        model[0].weight.data[:]*=10.0
        #model[0].bias.data[:]*=10.0
        model[2].weight.data[:]/=10.0
    elif name == 'lclinearl1k3c32relu01':
        model = nn.Sequential(LocallyConnected2dN(3, 32, 32, 3, 1), nn.ReLU(), nn.Flatten(), nn.Linear(32*32*32,10))
        model[0].weight.data[:]/=10.0
        model[3].weight.data[:]*=10.0
    elif name == 'lclinearl1k3c32relu10':
        model = nn.Sequential(LocallyConnected2dN(3, 32, 32, 3, 1), nn.ReLU(), nn.Flatten(), nn.Linear(32*32*32,10))
        model[0].weight.data[:]*=10.0
        model[3].weight.data[:]/=10.0
    elif name == 'convlinearl1k3c32relu01':
        model = NetR(kernel=3, channels=32, scaling=0.1)
    elif name == 'convlinearl3k3c32relu01':
        model = Net3R(kernel=3, channels=32, scaling=0.1)
    elif name == 'convlinearl3k3c3201':
        model = Net3(kernel=3, channels=32, scaling=0.1)
    elif name == 'convlinearl3k3c32relu10':
        model = Net3R(kernel=3, channels=32, scaling=10.0)
    elif name == 'convlinearl3k3c3210':
        model = Net3(kernel=3, channels=32, scaling=10.0)
    elif name == 'convlinearl1k3c3201':
        model = Net(kernel=3, channels=32, scaling=0.1)
    elif name == 'convlinearl1k3c32relu10':
        model = NetR(kernel=3, channels=32, scaling=10.0)
    elif name == 'convlinearl1k3c3210':
        model = Net(kernel=3, channels=32, scaling=10.0)
    elif name == 'lclinearl1k3c32relu':
        model = nn.Sequential(LocallyConnected2dN(3, 32, 32, 3, 1), nn.ReLU(), nn.Flatten(), nn.Linear(32*32*32,10))
    elif name == 'lclinearl3k3c32':
        model = nn.Sequential(LocallyConnected2dN(3, 32, 32, 3, 1),LocallyConnected2dN(32, 32, 32, 3, 1),LocallyConnected2dN(32, 32, 32, 3, 1), nn.Flatten(), nn.Linear(32*32*32,10))
    elif name == 'lclinearl3k3c32relu':
        model = nn.Sequential(LocallyConnected2dN(3, 32, 32, 3, 1), nn.ReLU(),LocallyConnected2dN(32, 32, 32, 3, 1), nn.ReLU(),LocallyConnected2dN(32, 32, 32, 3, 1), nn.ReLU(), nn.Flatten(), nn.Linear(32*32*32,10))
    elif name == 'convlinearl3k3c32relu':
        model = Net3R(kernel=3, channels=32)
    elif name == 'lclinearl1k3c32reluconvinit':
        model = nn.Sequential(LocallyConnected2d(3, 32, 32, 3, 1, conv_init=True), nn.ReLU(), nn.Flatten(), nn.Linear(32*32*32,10))
    elif name == 'lclinearl3k3c32reluconvinit':
        model = nn.Sequential(LocallyConnected2d(3, 32, 32, 3, 1, conv_init=True), nn.ReLU(),LocallyConnected2d(32, 32, 32, 3, 1, conv_init=True), nn.ReLU(),LocallyConnected2d(32, 32, 32, 3, 1, conv_init=True), nn.ReLU(), nn.Flatten(), nn.Linear(32*32*32,10))
    
        
    return model