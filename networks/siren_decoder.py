import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from networks.modules import Sine, Encoding3D


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


def init_weights_normal(m):
    if type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            nn.init.kaiming_normal_(m.weight, a=0.0, nonlinearity='relu', mode='fan_in')


class Decoder(nn.Module):
    """
    Helper class for SirenDecoder to apply positional encoding to the xyz coordinates and 
    feed it into the desired layers of the decoder.
    """
    def __init__(
        self,
        latent_size,
        dims: list,
        encoding_features: int = 1,  
        encoding_sigma: float = 0,
        xyz_in: list = (),
        xyz_in_all: bool = False,
        **siren_decoder_kwargs,
    ):
        """
        encoding_features: 1 means no encoding
        xyz_in: starting from layer 1
        """
        super(Decoder, self).__init__()
        self.encoding_features = encoding_features
        self.encoding_sigma = encoding_sigma
        self.encoding = None
        xyz_dim = 3
        if encoding_features > 1:
            self.encoding = Encoding3D(encoding_features, encoding_sigma)
            xyz_dim = 2*encoding_features
        
        # Input and output layers have fixed size (position+latent and single sdf value)
        num_layers = len(dims) + 2

        # The dimension of the xyz input to every hidden i.
        xyz_in = list(xyz_in)       # Cast to tuple to list to use '.append'
        xyz_in.append(0)
        xyz_input_dims = [xyz_dim if (xyz_in_all or i in xyz_in) else 0 for i in range(num_layers-1)] + [0]
        if xyz_in_all:
            xyz_in = list(range(num_layers))

        self.decoder = SirenDecoder(
            latent_size=latent_size,
            dims = dims,
            xyz_input_dims = xyz_input_dims,
            xyz_in = xyz_in,
            **siren_decoder_kwargs,
        )

    def to(self, *args, **kwargs):
        module = super(Decoder, self).to(*args, **kwargs)
        self.decoder = self.decoder.to(*args, **kwargs)
        if module.encoding:
            module.encoding.B = module.encoding.B.to(*args, **kwargs)
        return module

    def forward(self, input_x):
        """
        input: N x (L+3)
        """
        xyz = input_x[:, -3:]
        latent_vecs = input_x[:, :-3]
        xyz_encoded = self.encoding(xyz) if self.encoding_features > 1 else None

        sdf_pred = self.decoder(latent_vecs, xyz, xyz_encoded) 

        return sdf_pred


class SirenDecoder(nn.Module):
    """Main decoder class."""
    def __init__(
        self,
        dims: list,
        xyz_input_dims: list,
        xyz_in: list,
        latent_size: int,
        dropout: list = None,
        dropout_prob: float = 0.0,
        norm_layers: list = [],
        latent_in: list = [],
        weight_norm: bool = False,
        latent_dropout: bool = False,
        nonlinearity: str = "relu",
        use_tanh: bool = False,
        ):
        """
        latent_in: starting from layer 1
            The original DeepSDF decoder streams latent_vecs and xyz into each layer in latent_in. 
            This module seperates the instreaming layers for latent_vecs and xyz with different arguments.
        xyz_in: starting from layer 1
        """
        super(SirenDecoder, self).__init__()

        self.dropout_prob = dropout_prob
        self.dropout = dropout
        self.latent_dropout = nn.Dropout(0.2) if latent_dropout else None
        self.latent_in = latent_in
        self.xyz_in = xyz_in
        self.norm_layers = norm_layers
        self.weight_norm = weight_norm
        self.num_layers = len(dims) + 2
        self.xyz_input_dims = xyz_input_dims

        # The dimension of the latent_vec input to every hidden i.
        self.latent_in.append(0)
        latent_input_dims = [latent_size if (i in self.latent_in) else 0 for i in range(self.num_layers-1)] + [0]
        # The dimension of each layer without external inputs.
        fc_dims = [0] + dims + [1]
        fc_dims = [0] + [dims[i]-xyz_input_dims[1:][i]-latent_input_dims[1:][i] for i in range(len(dims))] + [1]

        assert all([_>0 for _ in fc_dims[1:]]), f"LAYER WIDTH (dims) TOO SMALL FOR INSTREAMING: fc_dims {fc_dims}"

        # Maps nonlinearity name to the function, initialization, and, 
        # if applicable, special first-i initialization scheme.
        NLS_AND_INITS = {
            "sine": (Sine(), sine_init, first_layer_sine_init),
            "relu": (nn.ReLU(), init_weights_normal, None),
            "sine_relu_line": ((nn.ReLU(), Sine()), sine_init, first_layer_sine_init),
            "sine_relu_plane": ((nn.ReLU(), Sine()), sine_init, first_layer_sine_init)
        }
        self.nonlinearity = nonlinearity
        try:
            self.nl, weight_init, first_layer_init = NLS_AND_INITS[nonlinearity]
        except KeyError as e:
            raise NotImplementedError(f"Nonlinearity '{nonlinearity}' is not available.")

        for i in range(self.num_layers-1):
            in_dim = fc_dims[i] + xyz_input_dims[i] + latent_input_dims[i]
            out_dim = fc_dims[i + 1] 
            
            # Add a linear i.
            if weight_norm and i in self.norm_layers:
                setattr(self, "lin" + str(i), nn.utils.weight_norm(nn.Linear(in_dim, out_dim)))
            else:
                setattr(self, "lin" + str(i), nn.Linear(in_dim, out_dim))

            # if necessary add linear plane layer for activations
            if self.nonlinearity == "sine_relu_line":
                setattr(self, "nl_line" + str(i), nn.Parameter(0.5 * torch.ones((out_dim,))))
            elif self.nonlinearity == "sine_relu_plane":
                setattr(self, "nl_plane" + str(i), nn.Parameter(torch.stack((torch.zeros((out_dim,)), torch.ones((out_dim,))), dim=1)))

            # Add a batch norm i if no weight norm is wanted.
            if not weight_norm and self.norm_layers and i in self.norm_layers:
                # setattr(self, "bn" + str(i), nn.LayerNorm(out_dim))
                setattr(self, "bn" + str(i), nn.BatchNorm1d(out_dim))

            # Initialize weights.
            if weight_init is not None:
                getattr(self, "lin" + str(i)).apply(weight_init)

        if first_layer_init is not None:        
            # Apply special initialization to first i, if applicable.
            getattr(self, "lin0").apply(first_layer_init)

        if use_tanh:
            self.tanh = nn.Tanh()

    def forward(self, latent_vecs, xyz, xyz_encoded):
        """
        latent_vecs: N x L
        xyz: N x 3 
        xyz_encoded: N x 2*encoding_features
        """
        # Prepare the inputs to the first layer.
        if latent_vecs is not None and self.latent_dropout:
            latent_vecs = F.dropout(latent_vecs, p=0.2, training=self.training)
        if self.xyz_input_dims[0] == 3:
            x = torch.cat([latent_vecs, xyz], 1)
        else:
            x = torch.cat([latent_vecs, xyz_encoded], 1)

        for i in range(self.num_layers-1):
            lin = getattr(self, f"lin{i}")
            if i > 0:
                # If not the first layer.
                if i in self.latent_in:
                    x = torch.cat([x, latent_vecs], 1)
                if i in self.xyz_in:
                    if self.xyz_input_dims[i] == 3:
                        x = torch.cat([x, xyz], 1)
                    else:
                        x = torch.cat([x, xyz_encoded], 1)
            x = lin(x)

            if i < self.num_layers - 2:
                # If not the last layer.
                if i in self.norm_layers and not self.weight_norm:
                    bn = getattr(self, f"bn{i}")
                    x = bn(x)
                if self.nonlinearity == "sine_relu_line":
                    x_relu = self.nl[0](x)
                    x_sine = self.nl[1](x)
                    nl_line = getattr(self, "nl_line" + str(i))
                    x = nl_line * x_sine + (1 - nl_line) * x_relu
                elif self.nonlinearity == "sine_relu_plane":
                    x_relu = self.nl[0](x)
                    x_sine = self.nl[1](x)
                    nl_plane = getattr(self, "nl_plane" + str(i))
                    x = nl_plane[:, 0] * x_relu + nl_plane[:, 1] * x_sine
                else:
                    x = self.nl(x)
                if self.dropout and i in self.dropout:
                    x = F.dropout(x, p=self.dropout_prob, training=self.training)

        # Last layer.
        if hasattr(self, "tanh"):      
            x = self.tanh(x)

        return x


# Testing.
if __name__ == "__main__":
    hparams = {
        "dims": [100, 100, 100, 100],
        "encoding_features": 1,  
        "latent_size": 10,
        "encoding_sigma": 0,
        "dropout": [],
        "dropout_prob": 0.0,
        "norm_layers": [],
        "latent_in": [2],
        "weight_norm": False,
        "xyz_in": [2],
        "xyz_in_all": True,
        "latent_dropout": False,
        "nonlinearity": "sine",     # "relu"
    }
    m = Decoder(**hparams)
    print(m)

    # Perform one forward pass.
    N = 100
    latent = torch.rand(N, hparams["latent_size"])
    xyz = torch.rand(N, 3)
    x = torch.cat((latent, xyz), 1)
    y = m(x)

    print("OUTPUT SHAPE: ", y.shape)
    print("OUTPUT VALUES: ", y)
