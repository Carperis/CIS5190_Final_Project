import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class Classifier(nn.Module):
    def __init__(self,
                 freq_network,
                 seq_network,
                 combined_input,
                 combined_dim,
                 num_classes,
                 n_layers,
                 skip_in=(4,),
                 weight_norm=True):
        super(Classifier, self).__init__()
        self.num_layers = n_layers
        self.skip_in = skip_in

        # Frequential Network
        self.freq = freq_network
        # Sequential Network
        self.seq = seq_network
        # Combined classification layers
        dims = [combined_input] + [combined_dim for _ in range(n_layers)] + [num_classes]
        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]
            lin = nn.Linear(dims[l], out_dim)
            if weight_norm:
                lin = nn.utils.weight_norm(lin)
            setattr(self, "lin" + str(l), lin)
        self.activation = nn.ReLU

    def forward(self, input_ids, attention_mask, tfidf_features):

        seq_feature = self.seq(input_ids, attention_mask)
        freq_feature = self.freq(tfidf_features)
        # Concat features
        inputs = torch.cat((seq_feature, freq_feature), dim=1)  # Shape: (batch_size, 128)
        x = inputs
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            if l in self.skip_in:
                x = torch.cat([x, inputs], 1) / np.sqrt(2)
            x = lin(x)

            if l < self.num_layers - 2:
                x = self.activation(x)

        # Output layer
        prob = F.sigmoid(x)

        return prob