import torch
import torch.nn as nn
#import torch.nn.functional as F
import logging


class EmailPartDetectorLinearModel(nn.Module):
    def __init__(self, in_feat=1024, out_class=27, num_lin_lrs=None,
                 dropout=0.0, gpu=-1, activation='TANH', is_debug=False):
        super(EmailPartDetectorLinearModel, self).__init__()

        self.gpu = gpu
        self.is_debug = is_debug
        self.layers = num_lin_lrs
        self.input_nodes = in_feat
        self.output_nodes = out_class
        self.dropout = dropout
        modules = []
        # create a MLP with given layers (list of values)
        if num_lin_lrs is not None and len(num_lin_lrs) > 0:
            # size of previous layer in NN
            nprev = in_feat
            for nh in num_lin_lrs:  # number of hidden nodes in the configuration
                if nh > 0:
                    modules.append(nn.Linear(nprev, nh))
                    nprev = nh
                    # after each linear layer add activation function
                    if activation == 'TANH':
                        modules.append(nn.Tanh())
                        print('-{:d}t'.format(nh), end='')
                    elif activation == 'RELU':
                        modules.append(nn.ReLU())
                        print('-{:d}r'.format(nh), end='')
                    else:
                        raise Exception('Unrecognized activation {activation}')
                    # after each activation add dropout if specified
                    if dropout > 0:
                        modules.append(nn.Dropout(p=dropout))
            # add output layer
            modules.append(nn.Linear(nprev, out_class))
            print('-{:d}, dropout={:.1f}'.format(out_class, dropout))
        else:  # shallow NN
            # add dropuot layer
            if dropout > 0:
                modules.append(nn.Dropout(p=dropout))
            # add output layer
            modules.append(nn.Linear(in_feat, out_class))
            print(' - mlp %d-%d'.format(in_feat, out_class))
        # add all layers to the sequential model
        self.mlp = nn.Sequential(*modules)
        # optimize for GPU
        if self.gpu >= 0:
            self.device = torch.device("cuda")
            self.mlp = self.mlp.to(self.device)  # cuda()

    def forward(self, x):
        if self.is_debug:
            logging.debug("The shape of x is {}".format(x.shape))
            print("The shape of x is {}".format(x.shape))
        # optimize for GPU
        if self.gpu >= 0:
            x = x.to(self.device)
        ret = self.mlp(x)
        if self.is_debug:
            logging.warning("The shape after forward pass is {}".format(ret.shape))
            print("The shape after forward pass is {}".format(ret.shape))
        return ret


    def save(self, filepath, class_to_numb, numb_to_class):
        """Save the model to a file.
            Args:
                        filepath: the path of the file.
            """
        torch.save({
             'state_dict': self.state_dict(),
             'num_lin_lrs': self.layers,
             'out_class': self.output_nodes,
             'dropout': self.dropout,
             'class_to_numb': class_to_numb,
             'numb_to_class': numb_to_class
         }, filepath)

    @classmethod
    def load(cls, filepath):
        """Load the model from a file.
            Args:
                      filepath: the path of the file.
        """
        checkpoint = torch.load(filepath, map_location={'cuda:0': 'cpu'})
        class_to_numb = checkpoint['class_to_numb']
        numb_to_class = checkpoint['numb_to_class']
        num_lin_lrs = checkpoint['num_lin_lrs']
        out_classes = checkpoint['out_class']
        dropout = checkpoint['dropout']
        obj = EmailPartDetectorLinearModel(out_class=out_classes, num_lin_lrs=num_lin_lrs, dropout=dropout)
        obj.load_state_dict(checkpoint['state_dict'])
        return obj, class_to_numb, numb_to_class
