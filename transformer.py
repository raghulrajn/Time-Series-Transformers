from torch.utils.data import Dataset
import torch
import torch.nn as nn
import math
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TimeSeriesDataset(Dataset):
    """
    A custom Dataset class for handling time series data.

    Attributes:
        input_sequences (list of np.array): A list of input sequences, each of shape (seq_len, input_dim).
        output_sequences (list of np.array): A list of corresponding output sequences, each of shape (seq_len, output_dim).
    
    Methods:
        __len__(): Returns the number of samples in the dataset.
        __getitem__(idx): Returns the input and output sequences at the specified index as a dictionary.
    """
    def __init__(self, input_sequences, output_sequences):
        """
        Initializes the TimeSeriesDataset.

        Args:
            input_sequences (list of np.array): A list of input sequences, each of shape (seq_len, input_dim).
            output_sequences (list of np.array): A list of corresponding output sequences, each of shape (seq_len, output_dim).
        """
        self.input_sequences = input_sequences
        self.output_sequences = output_sequences

    def __len__(self):
        """
        Returns the number of samples in the dataset.

        Returns:
            int: The number of samples in the dataset.
        """
        return len(self.input_sequences)

    def __getitem__(self, idx):
        """
        Returns the input and output sequences at the specified index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            dict: A dictionary containing:
                - 'input_sequence' (torch.Tensor): The input sequence at the specified index.
                - 'output_sequence' (torch.Tensor): The corresponding output sequence at the specified index.
        """
        return {
            'input_sequence': torch.tensor(self.input_sequences[idx], dtype=torch.float32),
            'output_sequence': torch.tensor(self.output_sequences[idx], dtype=torch.float32)
        }
    

class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer models.

    Attributes:
        pe (torch.Tensor): The positional encoding matrix.
    
    Methods:
        forward(x): Adds positional encoding to the input tensor.
    """
    def __init__(self, d_model, max_len=5000):
        """
        Initializes the PositionalEncoding module.

        Args:
            d_model (int): The dimension of the model's internal representation.
            max_len (int, optional): The maximum length of the input sequences. Default is 5000.
        """
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        pe = torch.zeros(max_len, d_model, device = device)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, device=device).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Adds positional encoding to the input tensor.

        Args:
            x (torch.Tensor): The input tensor of shape (seq_len, batch_size, d_model).

        Returns:
            torch.Tensor: The input tensor with added positional encoding of shape (seq_len, batch_size, d_model).
        """
        x = x + self.pe[:x.size(0), :, :x.size(2)]
        return x

class TimeSeriesTransformer(nn.Module):
    """
    A Transformer model for time series prediction.

    Attributes:
        input_projection (nn.Linear): Linear layer to project input dimensions to the model dimension.
        output_projection (nn.Linear): Linear layer to project output dimensions to the model dimension.
        positional_encoding (PositionalEncoding): Positional encoding module to add positional information to the inputs.
        transformer (nn.Transformer): Transformer model consisting of encoder and decoder layers.
        fc (nn.Linear): Final linear layer to project the model output to the desired output dimension.
    
    Methods:
        forward(src, tgt): Forward pass through the model.
    """
    def __init__(self, input_dim, output_dim, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, max_len=5000):
        """
        Initializes the TimeSeriesTransformer.

        Args:
            input_dim (int): The number of input features.
            output_dim (int): The number of output features.
            d_model (int): The dimension of the model's internal representation.
            nhead (int): The number of heads in the multihead attention mechanisms.
            num_encoder_layers (int): The number of encoder layers in the transformer.
            num_decoder_layers (int): The number of decoder layers in the transformer.
            dim_feedforward (int): The dimension of the feedforward network model.
            max_len (int, optional): The maximum length of the input sequences. Default is 5000.
        """
        super(TimeSeriesTransformer, self).__init__()
        self.input_projection = nn.Linear(input_dim, d_model).to(device)
        self.output_projection = nn.Linear(output_dim, d_model).to(device)
        self.positional_encoding = PositionalEncoding(d_model, max_len).to(device)
        self.transformer = nn.Transformer(d_model=d_model, nhead=nhead, num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers, dim_feedforward=dim_feedforward)
        self.fc = nn.Linear(d_model, output_dim).to(device)

    def forward(self, src, tgt):
        """
        Forward pass through the model.

        Args:
            src (torch.Tensor): The source sequence of shape (seq_len, batch_size, input_dim).
            tgt (torch.Tensor): The target sequence of shape (seq_len, batch_size, output_dim).

        Returns:
            torch.Tensor: The output of the model of shape (seq_len, batch_size, output_dim).
        """
        src = self.input_projection(src)
        tgt = self.output_projection(tgt)
        src = self.positional_encoding(src)
        tgt = self.positional_encoding(tgt)
        transformer_output = self.transformer(src, tgt)
        output = self.fc(transformer_output)
        return output
