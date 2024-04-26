import torch
import torch.nn as nn
import numpy as np
import math
import sampyl as smp
from ctypes import *
import os
from scipy import integrate, LowLevelCallable


########################
# Initialization methods

def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # For PINNet, Raissi et al. 2019
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    # grab from upstream pytorch branch and paste here for now
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def init_weights_trunc_normal(m):
    # For PINNet, Raissi et al. 2019
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    if type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            fan_in = m.weight.size(1)
            fan_out = m.weight.size(0)
            std = math.sqrt(2.0 / float(fan_in + fan_out))
            mean = 0.
            # initialize with the same behavior as tf.truncated_normal
            # "The generated values follow a normal distribution with specified mean and
            # standard deviation, except that values whose magnitude is more than 2
            # standard deviations from the mean are dropped and re-picked."
            _no_grad_trunc_normal_(m.weight, mean, std, -2 * std, 2 * std)


def init_weights_normal(m):
    if type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            nn.init.kaiming_normal_(m.weight, a=0.0, nonlinearity='relu', mode='fan_in')


def init_weights_relu(m):
    if type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            stdv = 1. / math.sqrt(m.weight.size(1))
            m.weight.data.uniform_(-stdv, stdv)
            if m.bias is not None:
                m.bias.data.uniform_(-stdv, stdv)

def init_weights_selu(m):
    if type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            nn.init.normal_(m.weight, std=1 / math.sqrt(num_input))


def init_weights_elu(m):
    if type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            nn.init.normal_(m.weight, std=math.sqrt(1.5505188080679277) / math.sqrt(num_input))


def init_weights_xavier(m):
    if type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            nn.init.xavier_normal_(m.weight)


def sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            # See supplement Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(-np.sqrt(6 / num_input) / 30, np.sqrt(6 / num_input) / 30)


def first_layer_sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(-1 / num_input, 1 / num_input)


########################
# Input embedding
class IdentityMapping(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.in_dim = in_dim

    @property
    def flops(self):
        return 0

    @property
    def out_dim(self):
        return self.in_dim

    def forward(self, X):
        return X

class PositionalEncoding(nn.Module):
    '''Module to add positional encoding as in NeRF [Mildenhall et al. 2020].'''
    def __init__(self, in_features, num_frequencies=-1, sidelength=None, use_nyquist=True):
        super().__init__()

        self.in_features = in_features

        self.num_frequencies = num_frequencies
        if self.num_frequencies < 0:
            if self.in_features == 3:
                self.num_frequencies = 10
            elif self.in_features == 2:
                assert sidelength is not None
                if isinstance(sidelength, int):
                    sidelength = (sidelength, sidelength)
                self.num_frequencies = 4
                if use_nyquist:
                    self.num_frequencies = self.get_num_frequencies_nyquist(min(sidelength[0], sidelength[1]))
            elif self.in_features == 1:
                assert sidelength is not None
                self.num_frequencies = 4
                if use_nyquist:
                    self.num_frequencies = self.get_num_frequencies_nyquist(sidelength)

    @property
    def out_dim(self):
        return self.in_features + 2 * self.in_features * self.num_frequencies

    @property
    def flops(self):
        return self.in_features + (2 * self.in_features * self.num_frequencies) * 2

    def get_num_frequencies_nyquist(self, samples):
        nyquist_rate = 1 / (2 * (2 * 1 / samples))
        return int(math.floor(math.log(nyquist_rate, 2)))

    def forward(self, coords):
        coords = coords.view(coords.shape[0], -1, self.in_features)

        coords_pos_enc = coords
        for i in range(self.num_frequencies):
            for j in range(self.in_features):
                c = coords[..., j]

                sin = torch.unsqueeze(torch.sin((2 ** i) * np.pi * c), -1)
                cos = torch.unsqueeze(torch.cos((2 ** i) * np.pi * c), -1)

                coords_pos_enc = torch.cat((coords_pos_enc, sin, cos), axis=-1)

        return coords_pos_enc.reshape(coords.shape[0], -1, self.out_dim)

class RBFLayer(nn.Module):
    '''Transforms incoming data using a given radial basis function.
        - Input: (1, N, in_features) where N is an arbitrary batch size
        - Output: (1, N, out_features) where N is an arbitrary batch size'''

    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.centres = nn.Parameter(torch.Tensor(out_features, in_features))
        self.sigmas = nn.Parameter(torch.Tensor(out_features))
        self.reset_parameters()

        self.freq = nn.Parameter(np.pi * torch.ones((1, self.out_features)))

    @property
    def out_dim(self):
        return self.out_features

    @property
    def flops(self):
        raise NotImplementedError

    def reset_parameters(self):
        nn.init.uniform_(self.centres, -1, 1)
        nn.init.constant_(self.sigmas, 10)

    def forward(self, input):
        input = input[0, ...]
        size = (input.size(0), self.out_features, self.in_features)
        x = input.unsqueeze(1).expand(size)
        c = self.centres.unsqueeze(0).expand(size)
        distances = (x - c).pow(2).sum(-1) * self.sigmas.unsqueeze(0)
        return self.gaussian(distances).unsqueeze(0)

    def gaussian(self, alpha):
        phi = torch.exp(-1 * alpha.pow(2))
        return phi

class FourierFeatMapping(nn.Module):
    def __init__(self, in_dim, map_scale=16, map_size=1024, tunable=False):
        super().__init__()

        B = torch.normal(0., map_scale, size=(map_size//2, in_dim))

        if tunable:
            self.B = nn.Parameter(B, requires_grad=True)
        else:
            self.register_buffer('B', B)

    @property
    def out_dim(self):
        return 2 * self.B.shape[0]

    @property
    def flops(self):
        return self.B.shape[0] * self.B.shape[1]

    def forward(self, x):
        x_proj = torch.matmul(x, self.B.T)
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

def exp_sample(a, N):
        '''
        Random features decomposed from Laplacian kernel:
                K = exp(-abs(x)*a) * exp(-abs(y)*a).

        The Inverse Fourier transform of each component in k is
                P(w) = C/(a^2+w^2).

        To draw features, we sample from P(w) using MCMC sampling. 
        '''
        # samples = np.zeros((N*2, 2))
        # i = 0

        # while i < N:
        #         x,y = np.random.uniform(-1000, 1000, (2, N))
        #         p = np.random.uniform(0, 4./(a*a), N)
        #         u = 4*a*a/((a**2+x**2) * (a**2+y**2))

        #         mask = p < u
        #         if mask.sum() > 0:
        #                 samples[i:i+mask.sum()] = np.hstack([
        #                         x[mask].reshape((-1,1)), 
        #                         y[mask].reshape((-1,1))])
        #                 i += mask.sum()
        # return samples[:N]
        start = {'x': 0., 'y': 0.}
        logp = lambda x, y: -smp.np.log((a**2 + x**2) * (a**2 + y**2))
        return MCMCS(logp, start, N)

def exp2_sample(a, N):
        '''
        Random features decomposed from Laplacian kernel:
                K = exp(-a*sqrt(x^2+y^2)).

        This is a special case of Matern class kernel function when nu=1/2.
        '''
        return matern_sample(a, 0.5, N)

def matern_sample(a, nu, N):
        '''
        Random features decomposed from Matern class kernel. 
        The Inverse Fourier transform is
                P(w) = a^{2*nu}/(2*nu*a^2 + ||w||^2)**(n/2+nu).

        To draw features, we sample from P(w) using MCMC sampling. 
        '''
        # samples = np.zeros((N*2, 2))
        # i = 0

        # while i < N:
        #         x,y = np.random.uniform(-1000, 1000, (2, N))
        #         p = np.random.uniform(0, 1./((2*nu)**(nu+1) * a**2), N)
        #         u = a**(2*nu)/(2*nu*a**2 + x**2+y**2)**(nu+1)

        #         mask = p < u
        #         if mask.sum() > 0:
        #                 samples[i:i+mask.sum()] = np.hstack([
        #                         x[mask].reshape((-1,1)), 
        #                         y[mask].reshape((-1,1))])
        #                 i += mask.sum()
        # return samples[:N]
        start = {'x': 0., 'y': 0.}
        logp = lambda x, y: smp.np.log(a**(2*nu)/(2*nu*a**2 + x**2+y**2)**(nu+1))
        return MCMCS(logp, start, N)
def MCMCS(logp, start, N):
        '''
        DO NOT use MHS because it causes performance drop. Slice sampling
        keeps regression performance and is faster.
        '''
        slc = smp.Slice(logp, start)
        chain = slc.sample(N*4+2000, burn=2000, thin=4)
        return chain.copy().view('<f8').reshape((-1, 2))

def gamma_exp2_sample(a, gamma, N):
        '''
        The kernel is defined as 
                K = exp(-(a*sqrt(x^2+y^2))^gamma).

        Becomes EXP2 kernel when gamma = 1.
        '''
        lib = CDLL(os.path.abspath('./integrand_gamma_exp/integrand_gamma_exp.so'))
        lib.integrand_gamma_exp.restype = c_double
        lib.integrand_gamma_exp.argtypes = (c_int, POINTER(c_double), c_void_p)
        return numerical_fourier(lib.integrand_gamma_exp, N, gamma, a)
def rq_sample(a, order, N):
        '''
        The kernel is defined as 
                K = (1+a^2*(x^2+y^2)/(2*order))^(-order).
        '''
        lib = CDLL(os.path.abspath('./integrand_rq/integrand_rq.so'))
        lib.integrand_rq.restype = c_double
        lib.integrand_rq.argtypes = (c_int, POINTER(c_double), c_void_p)

        return numerical_fourier(lib.integrand_rq, N, order, a)
def numerical_fourier(integrand, N, *args):
        ## numerical Fourier transform of the kernel
        frq_r = np.linspace(0, 1000, 100)
        freq = np.zeros_like(frq_r)
        for i, fr in enumerate(frq_r):
                c = np.array([fr]+list(args))
                user_data = cast(c.ctypes.data_as(POINTER(c_double)), c_void_p)
                func = LowLevelCallable(integrand, user_data)
                freq[i] = integrate.dblquad(func, -np.inf, np.inf, lambda x: -np.inf, lambda x: np.inf)[0]

        ## perform rejection sampling
        samples = np.zeros((N*2, 2))
        i = 0
        while i < N:
                x, y = np.random.uniform(-1000, 1000, (2, N))
                p = np.random.uniform(0, freq[0], N)
                u = np.interp((x**2+y**2)**0.5, frq_r, freq, right=0)

                mask = p < u
                if mask.sum() > 0:
                        samples[i:i+mask.sum()] = np.hstack([
                                x[mask].reshape((-1,1)), 
                                y[mask].reshape((-1,1))])
                        i += mask.sum()
        return samples[:N]

def poly_sample(order, N):
        '''
        The kernel is defined as 
                K = max(0, 1-sqrt(x^2+y^2))^(order).
        '''
        lib = CDLL(os.path.abspath('./integrand_poly/integrand_poly.so'))
        lib.integrand_poly.restype = c_double
        lib.integrand_poly.argtypes = (c_int, POINTER(c_double), c_void_p)

        return numerical_fourier(lib.integrand_poly, N, order)

class RandomFourierMapping(nn.Module):
    '''
    Generate Random Fourier Features (RFF) corresponding to the kernel:
        k(x, y) = k_a(x_a, y_a)*k_b(x_b, y_b)
    where 
        k_a(x_a, y_a) = exp(-\norm(x_a-y_a)/gamma_1),
        k_b(x_b, y_b) = <x_b, y_b>^gamma_2.
    '''
    def __init__(self, in_dim, kernel='exp', map_size=1024, tunable=False, **kwargs):
        super().__init__()

        if kernel == 'exp1':
            length_scale = kwargs.get('length_scale', 64)
            W = exp_sample(length_scale, map_size)
        elif kernel == 'exp2':
            length_scale = kwargs.get('length_scale', 64)
            W = exp2_sample(length_scale, map_size)
        elif kernel == 'matern':
            length_scale = kwargs.get('length_scale', 64)
            matern_order = kwargs.get('matern_order', 0.5)
            W = matern_sample(length_scale, matern_order, map_size)
        elif kernel == 'gamma_exp':
            length_scale = kwargs.get('length_scale', 64)
            gamma_order = kwargs.get('gamma_order', 1)
            W = gamma_exp2_sample(length_scale, gamma_order, map_size)
        elif kernel == 'rq':
            length_scale = kwargs.get('length_scale', 64)
            rq_order = kwargs.get('rq_order', 4)
            W = rq_sample(length_scale, rq_order, map_size)
        elif kernel == 'poly':
            poly_order = kwargs.get('poly_order', 4)
            W = poly_sample(poly_order, map_size)
        else:
            raise NotImplementedError()
        b = np.random.uniform(0, np.pi * 2, map_size)

        if tunable:
            self.W = nn.Parameter(W, requires_grad=True)
            self.b = nn.Parameter(b, requires_grad=True)
        else:
            self.register_buffer('W', torch.tensor(np.array(W)).float())
            self.register_buffer('b', torch.tensor(b).float())

    @property
    def out_dim(self):
        return self.W.shape[0]

    def forward(self, x):
        Z = torch.cos(x @ self.W.T + self.b)
        return Z

### Taken from official SIREN repo
class Sine(nn.Module):
    def __init(self):
        super().__init__()

    def forward(self, input):
        # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
        return torch.sin(30 * input)

class FixedDropout(nn.Module):
    def __init__(self, p=0.5, batch_size = 1):
        super(FixedDropout, self).__init__()
        self.p = p  # Dropout probability
        self.steps_count = 0  # Current step count
        self.batch_size = batch_size
        self.register_buffer('mask', None)  # Dropout mask

    def forward(self, x, fix=False):
        if not fix:
            # Generate a new dropout mask
            b,c,l = x.shape
            self.mask = (torch.rand(self.batch_size,c,l) > self.p).float().to(x.device)
        out = x * self.mask
        return out # Apply the dropout mask

class MySequential(nn.Sequential):
    def forward(self, input, *args, **kwargs):
        for module in self:
            if isinstance(module, FixedDropout):
                input = module(input, *args, **kwargs)
            else:
                input = module(input)
        return input
      
class FCBlock(nn.Module):
    '''A fully connected neural network that also allows swapping out the weights when used with a hypernetwork.
    Can be used just as a normal neural network though, as well.
    '''

    def __init__(self, args, in_features, out_features, num_hidden_layers, hidden_features,
                 outermost_linear=False, nonlinearity='relu', weight_init=None, batch_size = 1):
        super().__init__()
        self.args = args
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = hidden_features
        self.num_hidden_layers = num_hidden_layers

        self.first_layer_init = None

        # Dictionary that maps nonlinearity name to the respective function, initialization, and, if applicable,
        # special first-layer initialization scheme
        nls_and_inits = {
            'sine':(Sine(), sine_init, first_layer_sine_init),
            # 'relu':(nn.ReLU(inplace=True), init_weights_normal, None),
            'relu':(nn.ReLU(inplace=True), init_weights_relu, None),
            'sigmoid':(nn.Sigmoid(), init_weights_xavier, None),
            'tanh':(nn.Tanh(), init_weights_xavier, None),
            'selu':(nn.SELU(inplace=True), init_weights_selu, None),
            'softplus':(nn.Softplus(), init_weights_normal, None),
            'elu':(nn.ELU(inplace=True), init_weights_elu, None)
        }

        nl, nl_weight_init, first_layer_init = nls_and_inits[nonlinearity]

        if weight_init is not None:  # Overwrite weight init if passed
            self.weight_init = weight_init
        else:
            self.weight_init = nl_weight_init

        self.net = []
        self.net.append(MySequential(
            nn.Linear(in_features, hidden_features), nl
        ))

        for i in range(num_hidden_layers):
            if self.args.dropout and i>1:
                layer = nn.Linear(hidden_features, hidden_features), FixedDropout(self.args.dropout_rate, batch_size=batch_size),nl
            else:
                layer = nn.Linear(hidden_features, hidden_features),nl
            self.net.append(MySequential(*layer))

        if outermost_linear:
            self.net.append(MySequential(nn.Linear(hidden_features, out_features)))
        else:
            self.net.append(MySequential(
                nn.Linear(hidden_features, out_features), nl
            ))

        self.net = MySequential(*self.net)
        if self.weight_init is not None:
            self.net.apply(self.weight_init)

        if first_layer_init is not None: # Apply special initialization to first layer, if applicable.
            self.net[0].apply(first_layer_init)

    @property
    def flops(self):
        # (in_dim + 1) * out_dim: plus one for bias
        return (self.in_features+1) * self.hidden_features + \
            self.num_hidden_layers * (self.hidden_features+1) * self.hidden_features + \
            (self.hidden_features+1) * self.out_features

    def forward(self, coords, fix=False):
        output = self.net(coords, fix=fix)
        return output

##########################
# SIREN
class Siren(nn.Module):
    '''A canonical representation network.'''

    def __init__(self, args, out_features=1, in_features=2, **kwargs):
        super().__init__()
        self.pos_embed = args.pos_emb

        if self.pos_embed == 'Id':
            self.map = IdentityMapping(in_features)
        elif self.pos_embed == 'rbf':
            self.map = RBFLayer(in_features=in_features,out_features=args.rbf_centers)
        elif self.pos_embed == 'pe':
            self.map = PositionalEncoding(in_features=in_features,
                num_frequencies=args.num_freqs,
                sidelength=kwargs.get('sidelength', None),
                use_nyquist=args.use_nyquist
            )
        elif self.pos_embed == 'ffm':
            self.map = FourierFeatMapping(in_features,
                map_scale=args.ffm_map_scale,
                map_size=args.ffm_map_size,
            )
        elif self.pos_embed == 'gffm':
            self.map = RandomFourierMapping(in_features,
                kernel=args.ffm_kernel,
                map_size=args.ffm_map_size,
                tunable=False,
                **kwargs
            )
        else:
            raise ValueError(f'Unknown type of positional embedding: {self.pos_embed}')
        in_features = self.map.out_dim

        self.net = FCBlock(args, in_features=in_features, out_features=out_features, num_hidden_layers=args.num_layers,
            hidden_features=args.hidden_dim, outermost_linear=True, nonlinearity=args.act_type, batch_size=args.batch_size)
        print(self)

    @property
    def flops(self):
        return self.net.flops

    def forward(self, coords, fix=False):
        # various input processing methods for different applications
        coords = self.map(coords)
        output = self.net(coords, fix=fix)
        return output