# CoNFiLD-LSKP modification notice (2026):
# Added spectral-normalized latent-conditioning options and imports.
# Derived from CoNFiLD; see third_party_licenses/README.md for attribution.
import torch
from torch import nn
import numpy as np

from ConditionalNeuralField.cnf.components import BatchLinear, FeatureMapping, NLS_AND_INITS, FourierLayer, GaborLayer, Sine, Swish
from ConditionalNeuralField.cnf.initialization import *

# auto encoder: full projection 
# siren based 
# auto decoder: full projection 
# in construction 
class SIRENAutoencoder_fp(nn.Module):
    '''
    siren network with author decoding 
    '''
    def __init__(self, hyper_in_features,  hyper_latent_features, hyper_num_hidden_layers, hyper_hidden_features,
                 nf_in_features, out_features , nf_num_hidden_layers, nf_hidden_features, hyper_nonlinearity = 'sine',nf_nonlinearity= 'sine',
                 omega_0_e = DEFAULT_W0, omega_0 = DEFAULT_W0 ,
                 premap_mode = None, **kwargs):
        super().__init__()
        use_sn = kwargs.pop('use_sn', False)
        sn_coeff = kwargs.pop('sn_coeff', 1.0)
        sn_power_iter = kwargs.pop('sn_power_iter', 1)
        self.premap_mode = premap_mode
        if not self.premap_mode ==None: 
            self.premap_layer = FeatureMapping(nf_in_features,mode = premap_mode, **kwargs)
        nf_in_features = self.premap_layer.dim # update the nf in features 

        self.nf_in_features = nf_in_features
        self.out_features = out_features
        self.nf_num_hidden_layers = nf_num_hidden_layers
        self.nf_hidden_features = nf_hidden_features

        self.hyper_latent_features = hyper_latent_features
        self.hyper_out_features = self.calculate_num_parameters(nf_in_features, out_features , nf_num_hidden_layers, nf_hidden_features)
        self.omega_0_e = omega_0_e 
        self.omega_0  = omega_0 
        self.first_layer_init = None
    
                        
        self.hyper_nl, self.hyper_nl_weight_init, self.hyper_first_layer_init = NLS_AND_INITS[hyper_nonlinearity]
        self.nf_nl, _,_ = NLS_AND_INITS[nf_nonlinearity]
        # set omega:
        self.nf_nl.w0 = omega_0

        self.hyper_net = nn.ModuleList([BatchLinear(hyper_in_features,hyper_hidden_features, use_sn=use_sn, sn_coeff=sn_coeff, sn_power_iter=sn_power_iter)] + 
                                  [BatchLinear(hyper_hidden_features,hyper_hidden_features, use_sn=use_sn, sn_coeff=sn_coeff, sn_power_iter=sn_power_iter) for i in range(hyper_num_hidden_layers)] + 
                                  [BatchLinear(hyper_hidden_features,hyper_latent_features, use_sn=use_sn, sn_coeff=sn_coeff, sn_power_iter=sn_power_iter)])

        if self.hyper_nl_weight_init is not None: 
            self.hyper_net.apply(self.hyper_nl_weight_init)

        if self.hyper_first_layer_init is not None: # Apply special initialization to first layer, if applicable.
            self.hyper_net[0].apply(self.hyper_first_layer_init)
        # print('init hyper dims: ', hyper_latent_features,self.hyper_out_features)
        self.hyper_net_last = BatchLinear(hyper_latent_features,self.hyper_out_features, use_sn=use_sn, sn_coeff=sn_coeff, sn_power_iter=sn_power_iter)
        # print('init hyper weight shape: ', self.hyper_net_last.weight.data.shape)
        self.init_hyper_layer()

    def forward(self, coords, priors):
        # coords: <t,h,w,c_coord>
        # latents: <t,h,w,c_priors>
        x = priors
        for i in range(len(self.hyper_net) -1):
            x = self.hyper_net[i](x)
            x = self.hyper_nl(x)
        latent = self.hyper_net[-1](x)

        print(latent.shape, self.hyper_net_last.weight.data.shape)
        params = self.hyper_net_last(latent)
        print('param shape:', params.shape)
        
        # fill x into nf net, full projection 
        cursors = [0]
        cursors.append(cursors[-1] + self.nf_in_features * self.nf_hidden_features)
        w1 = params[..., cursors[-2]:cursors[-1]].reshape(params.shape[:-1]+(self.nf_in_features,self.nf_hidden_features))
        hidden_ws = []
        for i in range(self.nf_num_hidden_layers):
            cursors.append(cursors[-1] + self.nf_hidden_features*self.nf_hidden_features)
            hidden_ws.append(params[..., cursors[-2]: cursors[-1]].reshape(params.shape[:-1]+(self.nf_hidden_features,self.nf_hidden_features)))
        
        cursors.append(cursors[-1] + self.nf_hidden_features*self.out_features)
        w2 = params[..., cursors[-2]:cursors[-1]].reshape(params.shape[:-1]+(self.nf_hidden_features,self.out_features))
        
        # fill bias
        cursors.append(cursors[-1] + self.nf_hidden_features)
        b1 = params[..., cursors[-2]: cursors[-1]]

        hidden_bs = []
        for i in range(self.nf_num_hidden_layers):
            cursors.append(cursors[-1] + self.nf_hidden_features)
            hidden_bs.append(params[..., cursors[-2]: cursors[-1]])
        
        b2 = params[..., cursors[-1]:]
        
        # propagate through the nf net, implementing SIREN
        if not self.premap_mode ==None: 
            out = self.premap_layer(coords)
        else: 
            out = coords

        # pass it through the nf network 
        out = self.nf_nl(torch.einsum('thwi,thwij->thwj', out, w1) + b1)

        for i in range(self.nf_num_hidden_layers):
            # out = torch.sin(self.omega_0 * torch.einsum('thwi,thwij->thwj', out, hidden_ws[i]) + hidden_bs[i])
            out = self.nf_nl(torch.einsum('thwi,thwij->thwj', out, hidden_ws[i]) + hidden_bs[i])
        
        print('ensum shapes: ' , out.shape, w2.shape, b2.shape)
        out = torch.einsum('thwi,thwij->thwj', out, w2) + b2
        return out , latent, params
    
    
    def init_hyper_layer(self):
        # init weights
        w_init = nn.init.uniform_(torch.empty(self.hyper_out_features, self.hyper_latent_features),a=-np.sqrt(6.0 / self.hyper_latent_features) * 1e-2, b=np.sqrt(6.0 / self.hyper_latent_features) * 1e-2)
        # init bias
        tmp = torch.ones((self.hyper_out_features))
        tmp[:self.nf_in_features * self.nf_hidden_features] = tmp[:self.nf_in_features * self.nf_hidden_features] * 1.0 / self.nf_in_features  # 1st layer weights
        temp_num_weights =self.nf_in_features * self.nf_hidden_features+ self.nf_num_hidden_layers*self.nf_hidden_features*self.nf_hidden_features + self.nf_hidden_features*self.out_features
        tmp[self.nf_in_features * self.nf_hidden_features:temp_num_weights] = tmp[self.nf_in_features * self.nf_hidden_features:temp_num_weights] * \
                                                            np.sqrt(6.0 /self.nf_hidden_features) / self.omega_0_e  # other layer weights
        tmp[temp_num_weights:] = 1.0 / self.nf_hidden_features  # all biases
        b_init = torch.distributions.uniform.Uniform(low= -tmp,high= tmp).sample().squeeze(-1)   # uniformly distributed in the range

        with torch.no_grad():
            self.hyper_net_last.weight.data = w_init 
            self.hyper_net_last.bias.data = b_init 

    @staticmethod
    def calculate_num_parameters(nf_in_features, out_features , nf_num_hidden_layers, nf_hidden_features):
        return (nf_in_features+1) * nf_hidden_features+ nf_num_hidden_layers*(nf_hidden_features+1)*nf_hidden_features + (nf_hidden_features+1)*out_features



class SIRENAutoencoder_film(nn.Module):
    '''
    siren network with author decoding 
    '''
    def __init__(self, hyper_in_features,  hyper_latent_features, hyper_num_hidden_layers, hyper_hidden_features,
                 nf_in_features, out_features , nf_num_hidden_layers, nf_hidden_features, hyper_nonlinearity = 'sine',nf_nonlinearity= 'sine',
                 omega_0_e = DEFAULT_W0, omega_0 = DEFAULT_W0,
                 premap_mode = None, **kwargs):
        super().__init__()
        use_sn = kwargs.pop('use_sn', False)
        sn_coeff = kwargs.pop('sn_coeff', 1.0)
        sn_power_iter = kwargs.pop('sn_power_iter', 1)
        self.premap_mode = premap_mode
        if not self.premap_mode ==None: 
            self.premap_layer = FeatureMapping(nf_in_features,mode = premap_mode, **kwargs)
            nf_in_features = self.premap_layer.dim # update the nf in features 

        self.nf_in_features = nf_in_features
        self.out_features = out_features
        self.nf_num_hidden_layers = nf_num_hidden_layers
        self.nf_hidden_features = nf_hidden_features

        self.hyper_latent_features = hyper_latent_features
        self.omega_0_e = omega_0_e 
        self.omega_0  = omega_0 
        self.first_layer_init = None
                        
        self.hyper_nl, self.hyper_nl_weight_init, self.hyper_first_layer_init = NLS_AND_INITS[hyper_nonlinearity]
        self.nf_nl, self.nf_nl_weight_init, self.nf_first_layer_init = NLS_AND_INITS[nf_nonlinearity]
        # set omega:
        self.nf_nl.w0 = omega_0

        self.hyper_net = nn.ModuleList([BatchLinear(hyper_in_features,hyper_hidden_features, use_sn=use_sn, sn_coeff=sn_coeff, sn_power_iter=sn_power_iter)] + 
                                  [BatchLinear(hyper_hidden_features,hyper_hidden_features, use_sn=use_sn, sn_coeff=sn_coeff, sn_power_iter=sn_power_iter) for i in range(hyper_num_hidden_layers)] + 
                                  [BatchLinear(hyper_hidden_features,hyper_latent_features, use_sn=use_sn, sn_coeff=sn_coeff, sn_power_iter=sn_power_iter)])

        self.proj_net = nn.ModuleList([BatchLinear(hyper_latent_features,nf_hidden_features,bias = False, use_sn=use_sn, sn_coeff=sn_coeff, sn_power_iter=sn_power_iter) for i in range(nf_num_hidden_layers+1)])
        
        self.nf_net = nn.ModuleList([BatchLinear(nf_in_features,nf_hidden_features)] + 
                                  [BatchLinear(nf_hidden_features,nf_hidden_features) for i in range(nf_num_hidden_layers)] + 
                                  [BatchLinear(nf_hidden_features,out_features)]) 
    
        # initialize the hyper network 
        if self.hyper_nl_weight_init is not None: 
            self.hyper_net.apply(self.hyper_nl_weight_init)
            self.proj_net.apply(self.hyper_nl_weight_init)

        if self.hyper_first_layer_init is not None: # Apply special initialization to first layer, if applicable.
            self.hyper_net[0].apply(self.hyper_first_layer_init)
            self.proj_net[0].apply(self.hyper_first_layer_init)

        # initialize the hyper network 
        if self.nf_nl_weight_init is not None: 
            self.nf_net.apply(self.nf_nl_weight_init)

        if self.nf_first_layer_init is not None: # Apply special initialization to first layer, if applicable.
            self.nf_net[0].apply(self.nf_first_layer_init)


    def forward(self, coords, priors):
        # coords: <t,h,w,c_coord>
        # latents: <t,h,w,c_priors>
        p = priors
        for i in range(len(self.hyper_net) -1):
            p = self.hyper_net[i](p)
            p = self.hyper_nl(p)
        latents = self.hyper_net[-1](p)


        # propagate through the nf net, implementing SIREN
        if not self.premap_mode ==None:  
            x = self.premap_layer(coords)
        else: 
            x = coords

        # pass it through the nf network
        for i in range(len(self.nf_net) -1):
            # print('net debug:', self.nf_net[i](x).shape, self.proj_net[i](latents).shape)
            x = self.nf_net[i](x) + self.proj_net[i](latents)
            x = self.nf_nl(x)
        out = self.nf_net[-1](x)
        return out,  latents


class SIRENAutoencoder_film_extra_in(SIRENAutoencoder_film):
    def forward(self, coord, priors):
        coord = torch.concat(
            [torch.ones_like(coord[0][...,:1])*coord[1], coord[0]], dim = -1
        )
        return super().forward(coord, priors)



# auto encoder: bias shift 
# auto encoder: weight and shift bias 
# auto encoder: filter

# siren auto decoder: FILM
class SIRENAutodecoder_film_single(nn.Module):
    '''
    siren network with author decoding 
    '''
    def __init__(self, in_coord_features, out_features, num_hidden_layers, hidden_features,
                 outermost_linear=False, nonlinearity='sine', weight_init=None,bias_init=None, #forward_mode = 'without_latent',
                 premap_mode = None, **kwargs):
        super().__init__()
        self.premap_mode = premap_mode
        if not self.premap_mode ==None: 
            self.premap_layer = FeatureMapping(in_coord_features,mode = premap_mode, **kwargs)
            in_coord_features = self.premap_layer.dim # update the nf in features     
        # self.forward_mode = forward_mode # with_latent, without_latent
        self.first_layer_init = None
        
                        
        self.nl, nl_weight_init, first_layer_init = NLS_AND_INITS[nonlinearity]

        if weight_init is not None:  # Overwrite weight init if passed
            self.weight_init = weight_init
        else:
            self.weight_init = nl_weight_init # those are default init funcs 

        self.net1 = nn.ModuleList([BatchLinear(in_coord_features,hidden_features)] + 
                                  [BatchLinear(hidden_features,hidden_features) for i in range(num_hidden_layers)] + 
                                  [BatchLinear(hidden_features,out_features)])

        if self.weight_init is not None:
            self.net1.apply(self.weight_init)

        if first_layer_init is not None: # Apply special initialization to first layer, if applicable.
            self.net1[0].apply(first_layer_init)


    def forward(self, coords, all_latents=None):
        # coords: <t,h,w,c_coord>
        # latents: <t,h,w,c_latent>
        
        # premap 
        if not self.premap_mode ==None: 
            x = self.premap_layer(coords)
        else: 
            x = coords

        if all_latents == None:
            # pass it through  the nf network 
            for i in range(len(self.net1) -1):
                x = self.net1[i](x)
                x = self.nl(x)
            x = self.net1[-1](x)
            return x 
        else: 
            # pass it through  the nf network 
            for i in range(len(self.net1) -1):
                x = self.net1[i](x) + all_latents[i]
                x = self.nl(x)
            x = self.net1[-1](x)
            return x 

    def forward_with_latent(self, coords, latents):
        # coords: <t,h,w,c_coord>
        # latents: <t,h,w,c_latent>
        
        # premap 
        if not self.premap_mode ==None: 
            x = self.premap_layer(coords)
        else: 
            x = coords

        # pass it through  the nf network 
        for i in range(len(self.net1) -1):
            x = self.net1[i](x) + self.net2[i](latents)
            x = self.nl(x)
        x = self.net1[-1](x)
        return x 

# siren auto decoder: FILM
class SIREN_rez_Autodecoder_film(nn.Module):
    '''
    siren network with author decoding 
    '''
    def __init__(self, in_coord_features, in_latent_features, out_features, num_hidden_blocks, hidden_features,num_hidden_layers_rez = 2,
                 outermost_linear=False, nonlinearity='sine', weight_init=None,bias_init=None,
                 premap_mode = None, **kwargs):
        super().__init__()
        use_sn = kwargs.pop('use_sn', False)
        sn_coeff = kwargs.pop('sn_coeff', 1.0)
        sn_power_iter = kwargs.pop('sn_power_iter', 1)

        self.num_hidden_blocks = num_hidden_blocks
        # self.hidden_features = hidden_features
        self.num_hidden_layers_rez = num_hidden_layers_rez

        self.premap_mode = premap_mode
        if not self.premap_mode ==None: 
            self.premap_layer = FeatureMapping(in_coord_features,mode = premap_mode, **kwargs)
            in_coord_features = self.premap_layer.dim # update the nf in features     

        self.first_layer_init = None

                        
        self.nl, nl_weight_init, first_layer_init = NLS_AND_INITS[nonlinearity]

        if weight_init is not None:  # Overwrite weight init if passed
            self.weight_init = weight_init
        else:
            self.weight_init = nl_weight_init # those are default init funcs 

        # create the neural field
        # append the first layer
        self.net1 = nn.ModuleList([BatchLinear(in_coord_features,hidden_features)] + 
                                  [BatchLinear(hidden_features,hidden_features) for i in range(num_hidden_blocks*num_hidden_layers_rez)] + 
                                  [BatchLinear(hidden_features,out_features)])
        
        self.net2 = nn.ModuleList([BatchLinear(in_latent_features,hidden_features,bias = False, use_sn=use_sn, sn_coeff=sn_coeff, sn_power_iter=sn_power_iter) for i in range(num_hidden_blocks*num_hidden_layers_rez+1)])

        if self.weight_init is not None:
            self.net1.apply(self.weight_init)
            self.net2.apply(self.weight_init)

        if first_layer_init is not None: # Apply special initialization to first layer, if applicable.
            self.net1[0].apply(first_layer_init)
            self.net2[0].apply(first_layer_init)
        if bias_init is not None:
            self.net2.apply(bias_init)

    def forward(self, coords, latents):
        # coords: <t,h,w,c_coord>
        # latents: <t,h,w,c_latent>
        
        # premap 
        if not self.premap_mode ==None: 
            x = self.premap_layer(coords)
        else: 
            x = coords

        # pass through first layer
        x = self.net1[0](x) + self.net2[0](latents)
        x = self.nl(x)

        # pass it through the nf network 
        for i in range(self.num_hidden_blocks):
            x0 = x 
            for j in range(self.num_hidden_layers_rez):
                ij = 1+i*self.num_hidden_layers_rez+j
                x = self.net1[ij](x) + self.net2[ij](latents)
                x = self.nl(x)
            x = 0.5*x0 + 0.5*x
        
        # pass it through the last layer
        x = self.net1[-1](x)
        return x 

# siren auto decoder: FILM
class SIRENAutodecoder_tw_film(nn.Module):
    '''
    siren network with author decoding 
    '''
    def __init__(self, in_coord_features, in_latent_features, out_features, num_hidden_layers, hidden_features,
                 outermost_linear=False, nonlinearity='sine_tw', weight_init=None,bias_init=None,w0_init = DEFAULT_W0, 
                 premap_mode = None, **kwargs):
        super().__init__()
        use_sn = kwargs.pop('use_sn', False)
        sn_coeff = kwargs.pop('sn_coeff', 1.0)
        sn_power_iter = kwargs.pop('sn_power_iter', 1)
        self.premap_mode = premap_mode
        if not self.premap_mode ==None: 
            self.premap_layer = FeatureMapping(in_coord_features,mode = premap_mode, **kwargs)
            in_coord_features = self.premap_layer.dim # update the nf in features     

        self.first_layer_init = None
                        
        self.nl, nl_weight_init, first_layer_init = NLS_AND_INITS[nonlinearity]

        if weight_init is not None:  # Overwrite weight init if passed
            self.weight_init = weight_init
        else:
            self.weight_init = nl_weight_init # those are default init funcs 

        self.net1 = nn.ModuleList(
            [BatchLinear(in_coord_features,hidden_features)] + 
            [BatchLinear(hidden_features,hidden_features) for _ in range(num_hidden_layers)] + 
            [BatchLinear(hidden_features,out_features)])
        
        self.net2 = nn.ModuleList(
            [BatchLinear(in_latent_features,hidden_features,bias = False, use_sn=use_sn, sn_coeff=sn_coeff, sn_power_iter=sn_power_iter) for _ in range(num_hidden_layers+1)]
            )

        if self.weight_init is not None:
            # self.net1.apply(self.weight_init)
            # self.net2.apply(self.weight_init)
            self.weight_init(self.net1, w0 = w0_init)
            self.weight_init(self.net2, w0 = w0_init)


        if first_layer_init is not None: # Apply special initialization to first layer, if applicable.
            self.net1[0].apply(first_layer_init)
            self.net2[0].apply(first_layer_init)

        if bias_init is not None:
            self.net2.apply(bias_init)

    def forward(self, coords, latents):
        # coords: <t,h,w,c_coord>
        # latents: <t,h,w,c_latent>
        
        # premap 
        if not self.premap_mode ==None: 
            x = self.premap_layer(coords)
        else: 
            x = coords

        # pass it through  the nf network 
        for i in range(len(self.net1) -1):
            x = self.net1[i](x) + self.net2[i](latents)
            x = self.nl(x)
        x = self.net1[-1](x)
        return x 


# siren auto decoder: FILM
class SIRENAutodecoder_film(nn.Module):
    '''
    siren network with author decoding 
    '''
    def __init__(self, in_coord_features, in_latent_features, out_features, num_hidden_layers, hidden_features,
                 outermost_linear=False, nonlinearity='sine', weight_init=None,bias_init=None,
                 premap_mode = None, **kwargs):
        super().__init__()
        use_sn = kwargs.pop('use_sn', False)
        sn_coeff = kwargs.pop('sn_coeff', 1.0)
        sn_power_iter = kwargs.pop('sn_power_iter', 1)
        self.premap_mode = premap_mode
        if not self.premap_mode ==None: 
            self.premap_layer = FeatureMapping(in_coord_features,mode = premap_mode, **kwargs)
            in_coord_features = self.premap_layer.dim # update the nf in features     

        self.first_layer_init = None
                        
        self.nl, nl_weight_init, first_layer_init = NLS_AND_INITS[nonlinearity]

        if weight_init is not None:  # Overwrite weight init if passed
            self.weight_init = weight_init
        else:
            self.weight_init = nl_weight_init # those are default init funcs 

        self.net1 = nn.ModuleList([BatchLinear(in_coord_features,hidden_features)] + 
                                  [BatchLinear(hidden_features,hidden_features) for i in range(num_hidden_layers)] + 
                                  [BatchLinear(hidden_features,out_features)])
        self.net2 = nn.ModuleList([BatchLinear(in_latent_features,hidden_features,bias = False, use_sn=use_sn, sn_coeff=sn_coeff, sn_power_iter=sn_power_iter) for i in range(num_hidden_layers+1)])

        if self.weight_init is not None:
            self.net1.apply(self.weight_init)
            self.net2.apply(self.weight_init)

        if first_layer_init is not None: # Apply special initialization to first layer, if applicable.
            self.net1[0].apply(first_layer_init)
            self.net2[0].apply(first_layer_init)
        if bias_init is not None:
            self.net2.apply(bias_init)

    def forward(self, coords, latents):
        # coords: <t,h,w,c_coord>
        # latents: <t,h,w,c_latent>

        # premap 
        if not self.premap_mode ==None: 
            x = self.premap_layer(coords)
        else: 
            x = coords

        # pass it through  the nf network 
        for i in range(len(self.net1) -1):
            x = self.net1[i](x) + self.net2[i](latents)
            x = self.nl(x)
        x = self.net1[-1](x)
        return x 


    def disable_gradient(self):
        for param in self.parameters():
            param.requires_grad = False


class SIRENAutodecoder_film_extra_in(SIRENAutodecoder_film):
    def forward(self, coord, latents):
        coord = torch.concat(
            [torch.ones_like(coord[0][...,:1])*coord[1], coord[0]], dim = -1
        )
        return super().forward(coord, latents)

# fn auto decoder: FILM
class FNAutodecoder_film(nn.Module):
    '''
    fourier network with author decoding 
    '''
    def __init__(self, in_coord_features, in_latent_features, out_features, num_hidden_layers, hidden_features,
                 bias=True, output_act=False,input_scale=256.0,weight_scale=1.0,alpha=6.0,beta=1.0
                 ,premap_mode = None, **kwargs):
        super().__init__()
        self.premap_mode = premap_mode
        if not self.premap_mode ==None: 
            self.premap_layer = FeatureMapping(in_coord_features,mode = premap_mode, **kwargs)
            in_coord_features = self.premap_layer.dim # update the nf in features     
            print('premap dim:',self.premap_layer.dim)
        # assemble the MFN
        self.net1 = nn.ModuleList(
            [nn.Linear(hidden_features, hidden_features, bias) for _ in range(num_hidden_layers)]+[nn.Linear(hidden_features, out_features)]
        )

        # assemble the FILM
        self.net2 = nn.ModuleList([nn.Linear(in_latent_features, hidden_features,bias = False ) for _ in range(num_hidden_layers+1)])

        # initialize MFN
        for lin in self.net1:
            in_dim = lin.weight.shape[1]
            lin.weight.data.uniform_(
                -np.sqrt(weight_scale / in_dim),
                np.sqrt(weight_scale / in_dim),
            )

        # initialize FILM
        for lin in self.net2:
            in_dim = lin.weight.shape[1]
            lin.weight.data.uniform_(
                -np.sqrt(weight_scale / in_dim),
                np.sqrt(weight_scale / in_dim),
            )

        # initialize filters 
        self.filters = nn.ModuleList(
            [
                FourierLayer(in_coord_features, hidden_features, input_scale / np.sqrt(num_hidden_layers + 1))
                for _ in range(num_hidden_layers + 1)
            ]
        )


    def forward(self, coords, latents):
        # coords: <t,h,w,c_coord>
        # latents: <t,h,w,c_latent>
        
        # premap 
        if not self.premap_mode ==None: 
            x0 = self.premap_layer(coords)
        else: 
            x0 = coords

        x = self.filters[0](x0)*(self.net2[0](latents))

        # pass it through  the nf network 
        for i in range(1,len(self.filters)):
            x = self.filters[i](x0)*(self.net1[i-1](x)+self.net2[i](latents))
        x = self.net1[-1](x)
        return x 


# auto decoder: FILM
class GNAutodecoder_film(nn.Module):
    '''
    siren network with author decoding 
    '''
    def __init__(self, in_coord_features, in_latent_features, out_features, num_hidden_layers, hidden_features,
                 bias=True, output_act=False,input_scale=256.0,weight_scale=1.0,alpha=6.0,beta=1.0
                 ,premap_mode = None, **kwargs):
        super().__init__()
        self.premap_mode = premap_mode
        if not self.premap_mode ==None: 
            self.premap_layer = FeatureMapping(in_coord_features,mode = premap_mode, **kwargs)
            in_coord_features = self.premap_layer.dim # update the nf in features     
            print('premap dim:',self.premap_layer.dim)
        # assemble the MFN
        self.net1 = nn.ModuleList(
            [nn.Linear(hidden_features, hidden_features, bias) for _ in range(num_hidden_layers)]+[nn.Linear(hidden_features, out_features)]
        )

        # assemble the FILM
        self.net2 = nn.ModuleList([nn.Linear(in_latent_features, hidden_features,bias = False ) for _ in range(num_hidden_layers+1)])

        # initialize MFN
        for lin in self.net1:
            in_dim = lin.weight.shape[1]
            lin.weight.data.uniform_(
                -np.sqrt(weight_scale / in_dim),
                np.sqrt(weight_scale / in_dim),
            )

        # initialize FILM
        for lin in self.net2:
            in_dim = lin.weight.shape[1]
            lin.weight.data.uniform_(
                -np.sqrt(weight_scale / in_dim),
                np.sqrt(weight_scale / in_dim),
            )

        # initialize filters 
        self.filters = nn.ModuleList(
            [
                GaborLayer(
                    in_coord_features,
                    hidden_features,
                    input_scale / np.sqrt(num_hidden_layers + 1),
                    alpha / (num_hidden_layers + 1),
                    beta,
                )
                for _ in range(num_hidden_layers + 1)
            ]
        )

    def forward(self, coords, latents):
        # coords: <t,h,w,c_coord>
        # latents: <t,h,w,c_latent>
        
        # premap 
        if not self.premap_mode ==None: 
            x0 = self.premap_layer(coords)
        else: 
            x0 = coords

        x = self.filters[0](x0)*(self.net2[0](latents))

        # pass it through  the nf network 
        for i in range(1,len(self.filters)):
            x = self.filters[i](x0)*(self.net1[i-1](x)+self.net2[i](latents))
        x = self.net1[-1](x)
        return x 


class SIRENAutodecoder_fp(nn.Module):
    '''
    siren network with author decoding 
    '''
    def __init__(self, hyper_latent_features,
                 nf_in_features, out_features , nf_num_hidden_layers, nf_hidden_features, hyper_nonlinearity = 'sine',nf_nonlinearity= 'sine',
                 omega_0_e = DEFAULT_W0, omega_0 = DEFAULT_W0,
                 premap_mode = None, **kwargs):
        super().__init__()
        use_sn = kwargs.pop('use_sn', False)
        sn_coeff = kwargs.pop('sn_coeff', 1.0)
        sn_power_iter = kwargs.pop('sn_power_iter', 1)
        self.premap_mode = premap_mode
        if not self.premap_mode ==None: 
            self.premap_layer = FeatureMapping(nf_in_features,mode = premap_mode, **kwargs)
            nf_in_features = self.premap_layer.dim # update the nf in features 

        self.nf_in_features = nf_in_features
        self.out_features = out_features
        self.nf_num_hidden_layers = nf_num_hidden_layers
        self.nf_hidden_features = nf_hidden_features

        self.hyper_latent_features = hyper_latent_features
        self.hyper_out_features = self.calculate_num_parameters(nf_in_features, out_features , nf_num_hidden_layers, nf_hidden_features)
        self.omega_0_e = omega_0_e 
        self.omega_0  = omega_0 
        self.first_layer_init = None
        # Dictionary that maps nonlinearity name to the respective function, initialization, and, if applicable,
        # special first-layer initialization scheme
        # different layers has different initialization schemes: 
        nls_and_inits = {'sine':(Sine(), sine_init, first_layer_sine_init), # act name, init func, first layer init func 
                         'relu':(nn.ReLU(inplace=True), init_weights_normal, None),
                         'sigmoid':(nn.Sigmoid(), init_weights_xavier, None),
                         'tanh':(nn.Tanh(), init_weights_xavier, None),
                         'selu':(nn.SELU(inplace=True), init_weights_selu, None),
                         'softplus':(nn.Softplus(), init_weights_normal, None),
                         'elu':(nn.ELU(inplace=True), init_weights_elu, None),
                         'swish':(Swish(), init_weights_xavier, None),
                         }

        self.nf_nl, _,_ = nls_and_inits[nf_nonlinearity]
        # set omega:
        self.nf_nl.w0 = omega_0
        # print('init hyper dims: ', hyper_latent_features,self.hyper_out_features)
        self.hyper_net_last = BatchLinear(hyper_latent_features,self.hyper_out_features, use_sn=use_sn, sn_coeff=sn_coeff, sn_power_iter=sn_power_iter)
        # print('init hyper weight shape: ', self.hyper_net_last.weight.data.shape)
        self.init_hyper_layer()

    def forward(self, coords, latents):
        # coords: <t,h,w,c_coord> or <1,h,w,c_coord> 
        # latents: <t,h,w,c_priors> or <t,1,1,c_priors>
        params = self.hyper_net_last(latents) # <t,1,1,params>
        print('param shape:', params.shape)
        
        # fill x into nf net, full projection 
        cursors = [0]
        cursors.append(cursors[-1] + self.nf_in_features * self.nf_hidden_features)
        w1 = params[..., cursors[-2]:cursors[-1]].reshape(params.shape[:-1]+(self.nf_in_features,self.nf_hidden_features))
        hidden_ws = []
        for i in range(self.nf_num_hidden_layers):
            cursors.append(cursors[-1] + self.nf_hidden_features*self.nf_hidden_features)
            hidden_ws.append(params[..., cursors[-2]: cursors[-1]].reshape(params.shape[:-1]+(self.nf_hidden_features,self.nf_hidden_features)))
        
        cursors.append(cursors[-1] + self.nf_hidden_features*self.out_features)
        w2 = params[..., cursors[-2]:cursors[-1]].reshape(params.shape[:-1]+(self.nf_hidden_features,self.out_features))
        
        # fill bias
        cursors.append(cursors[-1] + self.nf_hidden_features)
        b1 = params[..., cursors[-2]: cursors[-1]]

        hidden_bs = []
        for i in range(self.nf_num_hidden_layers):
            cursors.append(cursors[-1] + self.nf_hidden_features)
            hidden_bs.append(params[..., cursors[-2]: cursors[-1]])
        
        b2 = params[..., cursors[-1]:]
        
        # propagate through the nf net, implementing SIREN
        if not self.premap_mode ==None: 
            out = self.premap_layer(coords)
        else: 
            out = coords

        # pass it through the nf network 
        # here out:<1,h,w,c_coord>; w1: <t,1,1,c_oords, hidden_dim>; b1: <t,1,1,hidden_dim>
        out = self.nf_nl(torch.einsum('thwi,thwij->thwj', out, w1) + b1)

        for i in range(self.nf_num_hidden_layers):
            # out = torch.sin(self.omega_0 * torch.einsum('thwi,thwij->thwj', out, hidden_ws[i]) + hidden_bs[i])
            # here out:<t,h,w,hidden_dim>; w1: <t,1,1,hidden_dim, hidden_dim>; b1: <t,1,1,hidden_dim>
            out = self.nf_nl(torch.einsum('thwi,thwij->thwj', out, hidden_ws[i]) + hidden_bs[i])
        
        print('ensum shapes: ' , out.shape, w2.shape, b2.shape)
        # here out:<t,h,w,hidden_dim>; w1: <t,1,1,hidden_dim, out_dim>; b2: <t,1,1,out_dim>
        out = torch.einsum('thwi,thwij->thwj', out, w2) + b2
        return out
    
    
    def init_hyper_layer(self):
        # init weights
        w_init = nn.init.uniform_(torch.empty(self.hyper_out_features, self.hyper_latent_features),a=-np.sqrt(6.0 / self.hyper_latent_features) * 1e-2, b=np.sqrt(6.0 / self.hyper_latent_features) * 1e-2)
        # init bias
        tmp = torch.ones((self.hyper_out_features))
        tmp[:self.nf_in_features * self.nf_hidden_features] = tmp[:self.nf_in_features * self.nf_hidden_features] * 1.0 / self.nf_in_features  # 1st layer weights
        temp_num_weights =self.nf_in_features * self.nf_hidden_features+ self.nf_num_hidden_layers*self.nf_hidden_features*self.nf_hidden_features + self.nf_hidden_features*self.out_features
        tmp[self.nf_in_features * self.nf_hidden_features:temp_num_weights] = tmp[self.nf_in_features * self.nf_hidden_features:temp_num_weights] * \
                                                            np.sqrt(6.0 /self.nf_hidden_features) / self.omega_0_e  # other layer weights
        tmp[temp_num_weights:] = 1.0 / self.nf_hidden_features  # all biases
        b_init = torch.distributions.uniform.Uniform(low= -tmp,high= tmp).sample().squeeze(-1)   # uniformly distributed in the range

        with torch.no_grad():
            self.hyper_net_last.weight.data = w_init 
            self.hyper_net_last.bias.data = b_init 

    @staticmethod
    def calculate_num_parameters(nf_in_features, out_features , nf_num_hidden_layers, nf_hidden_features):
        return (nf_in_features+1) * nf_hidden_features+ nf_num_hidden_layers*(nf_hidden_features+1)*nf_hidden_features + (nf_hidden_features+1)*out_features

    def disable_gradient(self):
        for param in self.parameters():
            param.requires_grad = False
            
# siren auto decoder: modified FILM
class SIRENAutodecoder_mdf_film(nn.Module):
    '''
    siren network with auto decoding 
    mdf mean the conditioning is full projection 
    '''
    def __init__(self, in_coord_features, in_latent_features, out_features, num_hidden_layers, hidden_features,
                 outermost_linear=False, nonlinearity='sine', weight_init=None,bias_init=None,
                 premap_mode = None, **kwargs):
        super().__init__()
        use_sn = kwargs.pop('use_sn', False)
        sn_coeff = kwargs.pop('sn_coeff', 1.0)
        sn_power_iter = kwargs.pop('sn_power_iter', 1)
        self.premap_mode = premap_mode
        if not self.premap_mode ==None: 
            self.premap_layer = FeatureMapping(in_coord_features,mode = premap_mode, **kwargs)
            in_coord_features = self.premap_layer.dim # update the nf in features     

        self.first_layer_init = None
                        
        self.nl, nl_weight_init, first_layer_init = NLS_AND_INITS[nonlinearity]

        if weight_init is not None:  # Overwrite weight init if passed
            self.weight_init = weight_init
        else:
            self.weight_init = nl_weight_init # those are default init funcs 

        # create the net for the nf 
        self.nf_net = nn.ModuleList([BatchLinear(in_coord_features,hidden_features)] + 
                                  [BatchLinear(hidden_features,hidden_features) for i in range(num_hidden_layers)] + 
                                  [BatchLinear(hidden_features,out_features)])

        # create the net for the weights and bias, the hypernet it self has no bias. 
        self.hw_net = nn.ModuleList([BatchLinear(in_latent_features,in_coord_features*hidden_features,bias = False, use_sn=use_sn, sn_coeff=sn_coeff, sn_power_iter=sn_power_iter)]+
                                    [BatchLinear(in_latent_features,hidden_features*hidden_features,bias = False, use_sn=use_sn, sn_coeff=sn_coeff, sn_power_iter=sn_power_iter) for i in range(num_hidden_layers)])
                                    # [BatchLinear(in_latent_features,hidden_features*out_features,bias = False)])
        self.hb_net = nn.ModuleList([BatchLinear(in_latent_features,hidden_features,bias = False, use_sn=use_sn, sn_coeff=sn_coeff, sn_power_iter=sn_power_iter) for i in range(num_hidden_layers+1)])
                                    #  [BatchLinear(in_latent_features,out_features*out_features,bias = False)])

        if self.weight_init is not None:
            self.nf_net.apply(self.weight_init)

        if first_layer_init is not None: # Apply special initialization to first layer, if applicable.
            self.nf_net[0].apply(first_layer_init)
        self.init_hyper_layer()     

    def init_hyper_layer(self):
        # init weights
        self.hw_net.apply(init_weights_uniform_siren_scale)
        self.hb_net.apply(init_weights_uniform_siren_scale)

    def forward(self, coords, latents):
        # _, h_size, w_size, coord_size = coords.shape
        t_size = latents.shape[0]
        # coords: <t,h,w,c_coord> or <1, h,w,c_coords>
        # latents: <t,h,w,c_latent> or <t, 1,1, coords>
        
        # premap 
        if not self.premap_mode ==None: 
            x = self.premap_layer(coords)
        else: 
            x = coords

        # pass it through  the nf network 
        for i in range(len(self.nf_net) -1):

            x = (
                    self.nf_net[i](x) + 
                    torch.einsum(
                        'thwi,thwji->thwj', 
                        x, 
                        self.hw_net[i](latents).reshape((t_size, 1, 1)+self.nf_net[i].weight.shape)
                    )
                    + self.hb_net[i](latents)
                )
            
            x = self.nl(x)

        x = self.nf_net[-1](x)

        return x 
    

class SIRENAutodecoder_mdf_film_extra_in(SIRENAutodecoder_mdf_film):
    def forward(self, coord, latents):
        coord = torch.concat(
            [torch.ones_like(coord[0][...,:1])*coord[1], coord[0]], dim = -1
        )
        return super().forward(coord, latents)

        
