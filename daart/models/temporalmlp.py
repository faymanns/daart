"""Temporal MLP model implemented in PyTorch."""

import numpy as np
from sklearn.metrics import accuracy_score
import torch
from torch import nn
from daart.models.base import BaseModel

# to ignore imports for sphix-autoapidoc
__all__ = ['TemporalMLP']


class TemporalMLP(BaseModel):
    """MLP network with initial 1D convolution layer."""

    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.encoder = None
        self.classifier = None
        self.classifier_weak = None
        self.predictor = None
        self.build_model()

    def __str__(self):
        """Pretty print model architecture."""
        format_str = '\nTemporalMLP architecture\n'
        format_str += '------------------------\n'
        format_str += 'Encoder:\n'
        for i, module in enumerate(self.encoder):
            format_str += str('    {}: {}\n'.format(i, module))
        format_str += 'Classifier:\n'
        for i, module in enumerate(self.classifier):
            format_str += str('    {}: {}\n'.format(i, module))
        if self.predictor is not None:
            format_str += 'Predictor:\n'
            for i, module in enumerate(self.predictor):
                format_str += str('    {}: {}\n'.format(i, module))
        return format_str

    def build_model(self):
        """Construct the model using hparams."""

        self.encoder = nn.ModuleList()

        global_layer_num = 0

        # -------------------------------------------------------------
        # first layer is 1d conv for incorporating past/future activity
        # -------------------------------------------------------------

        in_size = self.hparams['input_size']
        out_size = self.hparams['n_hid_units']
        layer = nn.Conv1d(
            in_channels=in_size,
            out_channels=out_size,
            kernel_size=self.hparams['n_lags'] * 2 + 1,  # window around t
            padding=self.hparams['n_lags'])  # same output
        name = str('conv1d_layer_%02i' % global_layer_num)
        self.encoder.add_module(name, layer)

        # add activation
        if self.hparams['n_hid_layers'] == 0:
            activation = None  # cross entropy loss handles this
        else:
            if self.hparams['activation'] == 'linear':
                activation = None
            elif self.hparams['activation'] == 'relu':
                activation = nn.ReLU()
            elif self.hparams['activation'] == 'lrelu':
                activation = nn.LeakyReLU(0.05)
            elif self.hparams['activation'] == 'sigmoid':
                activation = nn.Sigmoid()
            elif self.hparams['activation'] == 'tanh':
                activation = nn.Tanh()
            else:
                raise ValueError(
                    '"%s" is an invalid activation function' % self.hparams['activation'])

        if activation:
            name = '%s_%02i' % (self.hparams['activation'], global_layer_num)
            self.encoder.add_module(name, activation)

        # update layer info
        global_layer_num += 1
        in_size = out_size

        # -------------------------------------------------------------
        # remaining layers
        # -------------------------------------------------------------
        # loop over hidden layers (0 layers <-> linear model)
        for i_layer in range(self.hparams['n_hid_layers']):

            # add layer
            layer = nn.Linear(in_features=in_size, out_features=out_size)
            name = str('dense_layer_%02i' % global_layer_num)
            self.encoder.add_module(name, layer)

            # add activation
            if i_layer == self.hparams['n_hid_layers'] - 1:
                activation = None  # cross entropy loss handles this
            else:
                if self.hparams['activation'] == 'linear':
                    activation = None
                elif self.hparams['activation'] == 'relu':
                    activation = nn.ReLU()
                elif self.hparams['activation'] == 'lrelu':
                    activation = nn.LeakyReLU(0.05)
                elif self.hparams['activation'] == 'sigmoid':
                    activation = nn.Sigmoid()
                elif self.hparams['activation'] == 'tanh':
                    activation = nn.Tanh()
                else:
                    raise ValueError(
                        '"%s" is an invalid activation function' % self.hparams['activation'])

            if activation:
                name = '%s_%02i' % (self.hparams['activation'], global_layer_num)
                self.encoder.add_module(name, activation)

            # update layer info
            global_layer_num += 1
            in_size = out_size

        final_encoder_size = out_size

        # -------------------------------------------------------------
        # classifier: single linear layer
        # -------------------------------------------------------------
        # linear classifier (hand labels)
        self.classifier = self._build_classifier(global_layer_num=global_layer_num)

        # linear classifier (heuristic labels)
        self.classifier_weak = self._build_classifier(global_layer_num=global_layer_num)

        # update layer info
        global_layer_num += 1

        # -------------------------------------------------------------
        # decoding layers for next step prediction
        # -------------------------------------------------------------
        in_size = final_encoder_size
        if self.hparams.get('lambda_pred', 0) > 0:

            self.predictor = nn.ModuleList()

            # loop over hidden layers (0 layers <-> linear model)
            for i_layer in range(self.hparams['n_hid_layers'] + 1):

                if i_layer == self.hparams['n_hid_layers']:
                    out_size = self.hparams['input_size']
                else:
                    out_size = self.hparams['n_hid_units']

                # add layer
                layer = nn.Linear(in_features=in_size, out_features=out_size)
                name = str('dense_layer_%02i' % global_layer_num)
                self.predictor.add_module(name, layer)

                # add activation
                if i_layer == self.hparams['n_hid_layers']:
                    # no activation for final layer
                    activation = None
                else:
                    if self.hparams['activation'] == 'linear':
                        activation = None
                    elif self.hparams['activation'] == 'relu':
                        activation = nn.ReLU()
                    elif self.hparams['activation'] == 'lrelu':
                        activation = nn.LeakyReLU(0.05)
                    elif self.hparams['activation'] == 'sigmoid':
                        activation = nn.Sigmoid()
                    elif self.hparams['activation'] == 'tanh':
                        activation = nn.Tanh()
                    else:
                        raise ValueError(
                            '"%s" is an invalid activation function' % self.hparams['activation'])

                if activation:
                    name = '%s_%02i' % (self.hparams['activation'], global_layer_num)
                    self.predictor.add_module(name, activation)

                # update layer info
                global_layer_num += 1
                in_size = out_size

    def _build_classifier(self, global_layer_num):

        classifier = nn.Sequential()

        in_size = self.hparams['n_hid_units']
        out_size = self.hparams['output_size']

        # add layer (cross entropy loss handles activation)
        layer = nn.Linear(in_features=in_size, out_features=out_size)
        name = str('dense(classification)_layer_%02i' % global_layer_num)
        classifier.add_module(name, layer)

        return classifier

    def forward(self, x, **kwargs):
        """Process input data.

        Parameters
        ----------
        x : torch.Tensor object
            input data

        Returns
        -------
        dict
            - 'labels' (torch.Tensor): model classification
            - 'prediction' (torch.Tensor): one-step-ahead prediction
            - 'embedding' (torch.Tensor): behavioral embedding used for classification/prediction

        """

        # push data through encoder to get latent embedding
        for name, layer in self.encoder.named_children():

            if name == 'conv1d_layer_00':
                # input is batch x in_channels x time
                # output is batch x out_channels x time

                # x = T x N (T = 500, N = 16)
                # x.transpose(1, 0) -> x = N x T
                # x.unsqueeze(0) -> x = 1 x N x T
                # x = layer(x) -> x = 1 x M x T
                # x.squeeze() -> x = M x T
                # x.transpose(1, 0) -> x = T x M

                x = layer(x.transpose(1, 0).unsqueeze(0)).squeeze().transpose(1, 0)
            else:
                x = layer(x)

        # push embedding through classifier to get labels
        z = self.classifier(x)
        if self.hparams.get('lambda_weak', 0) > 0:
            z_weak = self.classifier_weak(x)
        else:
            z_weak = None

        # push embedding through predictor network to get data at subsequent time points
        if self.hparams.get('lambda_pred', 0) > 0:
            y = x
            for name, layer in self.predictor.named_children():
                y = layer(y)
        else:
            y = None

        return {'labels': z, 'labels_weak': z_weak, 'prediction': y, 'embedding': x}
