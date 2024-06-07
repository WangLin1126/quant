import torch.nn as nn
import torch
import torch.nn.functional as F
import math
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_dim, num_layers, output_dim):
        super(LSTM, self).__init__()

        self.lstm = nn.GRU(input_size, hidden_dim, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        output, _ = self.lstm(x)
        output = output[:,-1,:]
        output = self.linear(output)
        return output

class gru(nn.Module):
    def __init__(self, input_size, hidden_dim, num_layers, output_dim):
        super(gru, self).__init__()

        self.lstm = nn.GRU(input_size, hidden_dim, num_layers, batch_first=True)
        self.batchnorm = nn.BatchNorm1d(hidden_dim)  
        self.linear = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()  

    def forward(self, x):
        output, _ = self.lstm(x)
        output = output[:, -1, :]
        output = self.batchnorm(output)  
        output = self.linear(output)
        output = self.sigmoid(output)  
        return output

class CrossGRU(nn.Module):
    def __init__(self, input_size, hidden_dim, num_layers, output_dim, dropout_rate, c=30, num_heads=2, device='cpu'):
        super(CrossGRU, self).__init__()
        self.head_dim = hidden_dim // num_heads
        assert self.head_dim * num_heads == hidden_dim, "Hidden dimension must be divisible by the number of heads"

        self.lstm = nn.GRU(input_size, hidden_dim, num_layers, batch_first=True)
        self.gate = nn.Linear(hidden_dim, 1)
        self.mlp1 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.batchnorm = nn.BatchNorm1d(hidden_dim)  
        self.mlp2 = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()  
        self.R = torch.randn(c, hidden_dim, device=device)

        self.query_transforms = nn.ModuleList([nn.Linear(hidden_dim, self.head_dim).to(device) for _ in range(num_heads)])
        self.key_transforms = nn.ModuleList([nn.Linear(hidden_dim, self.head_dim).to(device) for _ in range(num_heads)])
        self.value_transforms = nn.ModuleList([nn.Linear(hidden_dim, self.head_dim).to(device) for _ in range(num_heads)])

    def attention(self, query, key, value):
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        p_attn = F.softmax(scores, dim=-1)
        return torch.matmul(p_attn, value), p_attn

    def multi_head_attention(self, query, key, value):
        head_outputs = []
        attentions = []
        for q, k, v in zip(self.query_transforms, self.key_transforms, self.value_transforms):
            q_transformed = q(query)
            k_transformed = k(key)
            v_transformed = v(value)
            out, attn = self.attention(q_transformed, k_transformed, v_transformed)
            head_outputs.append(out)
            attentions.append(attn)
        out_concat = torch.cat(head_outputs, dim=-1)
        return out_concat, attentions

    def forward(self, x):
        output, _ = self.lstm(x)
        S = output[:, -1, :]
        B, _ = self.multi_head_attention(self.R, S, S)  
        S_prime, _ = self.multi_head_attention(S, B, B)
        alpha = torch.sigmoid(self.gate(S))
        alpha = alpha.expand_as(S)
        S_mix = alpha * S_prime + (1 - alpha) * S
        output = F.relu(self.mlp1(S_mix))
        output = self.dropout(output)
        S_mix = output + S_mix
        output = self.mlp2(S_mix)
        output = output.view(-1)
        output = (output - output.mean()) / (output.std()+1e-8)
        return output

class MLP(nn.Module):
    """
    A module for a Multi-Layer Perceptron with variable number of hidden layers.
    """
    def __init__(self, input_dim, hidden_dims, output_dim):
        super(MLP, self).__init__()
        # Construct the MLP layers based on the provided list of hidden dimensions
        layers = []
        for dim in hidden_dims:
            layers.append(nn.Linear(input_dim, dim))
            layers.append(nn.ReLU())  # Adding ReLU activation after each hidden layer
            input_dim = dim  # Update the input dimension for the next layer
        
        # Add the final output layer
        layers.append(nn.Linear(input_dim, output_dim))
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Forward pass through the network, applying softmax to the output.
        """
        output = self.mlp(x)
        return output

class CrossGRUMLP(nn.Module):
    def __init__(self, input_size, hidden_dim, num_layers, output_dim, dropout_rate, c=30, num_heads=2, device='cpu'):
        super(CrossGRUMLP, self).__init__()
        self.head_dim = hidden_dim // num_heads
        assert self.head_dim * num_heads == hidden_dim, "Hidden dimension must be divisible by the number of heads"

        self.lstm = nn.GRU(input_size, hidden_dim, num_layers, batch_first=True)
        self.gate = MLP(hidden_dim, [64,64],  hidden_dim)
        self.mlp1 = MLP(hidden_dim, [64,64],  hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.batchnorm1 = nn.BatchNorm1d(hidden_dim)  
        self.batchnorm2 = nn.BatchNorm1d(hidden_dim)  
        self.mlp2 = MLP(hidden_dim, [64,64], hidden_dim)
        self.R = torch.randn(c, hidden_dim, device=device)
        self.linear = nn.Linear(hidden_dim , output_dim)
        self.sigmoid = nn.Sigmoid()  
        self.query_transforms = nn.ModuleList([nn.Linear(hidden_dim, self.head_dim).to(device) for _ in range(num_heads)])
        self.key_transforms = nn.ModuleList([nn.Linear(hidden_dim, self.head_dim).to(device) for _ in range(num_heads)])
        self.value_transforms = nn.ModuleList([nn.Linear(hidden_dim, self.head_dim).to(device) for _ in range(num_heads)])

    def attention(self, query, key, value):
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        p_attn = F.softmax(scores, dim=-1)
        return torch.matmul(p_attn, value), p_attn

    def multi_head_attention(self, query, key, value):
        head_outputs = []
        attentions = []
        for q, k, v in zip(self.query_transforms, self.key_transforms, self.value_transforms):
            q_transformed = q(query)
            k_transformed = k(key)
            v_transformed = v(value)
            out, attn = self.attention(q_transformed, k_transformed, v_transformed)
            head_outputs.append(out)
            attentions.append(attn)
        out_concat = torch.cat(head_outputs, dim=-1)
        return out_concat, attentions

    def forward(self, x):
        output, _ = self.lstm(x)
        S = output[:, -1, :]
        B, _ = self.multi_head_attention(self.R, S, S)  
        S_prime, _ = self.multi_head_attention(S, B, B)
        alpha = F.softmax(self.gate(S), dim = 1)
        S_mix = alpha * S_prime +  S
        output = self.mlp1(S_mix)
        output = self.batchnorm1(output)
        output = self.dropout(output)
        output = self.mlp2(output)
        output = self.batchnorm2(output)
        output = output + S_mix
        output = output.mean(dim = -1)
        output = output.view(-1)
        output = (output - output.mean()) / (output.std()+1e-6)
        return output 
    
class CrossGRUMLPRes(nn.Module):
    def __init__(self, input_size, hidden_dim, num_layers, output_dim, dropout_rate, c=30, num_heads=2, device='cpu'):
        super(CrossGRUMLPRes, self).__init__()
        self.cross_gru_mlp = CrossGRUMLP(input_size, hidden_dim, num_layers, input_size, dropout_rate, c, num_heads, device)
    def forward(self, x):
        cross_gru_mlp_output = self.cross_gru_mlp(x)
        residual_output = F.softmax(cross_gru_mlp_output, dim = 0) * F.softmax(x[:,-1,:], dim = 0)
        
        return torch.mean(residual_output, dim = -1)

class multigru(nn.Module):
    def __init__(self, input_size, hidden_dim, num_layers, output_dim, dropout_rate, c=200, num_heads=2, device='cpu'):
        super(multigru, self).__init__()
        self.head_dim = hidden_dim // num_heads
        assert self.head_dim * num_heads == hidden_dim, "Hidden dimension must be divisible by the number of heads"

        self.lstm = nn.GRU(input_size, hidden_dim, num_layers, batch_first=True)
        self.gate = nn.Linear(hidden_dim, 1)
        self.mlp = nn.Linear(hidden_dim, hidden_dim)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.batchnorm1 = nn.BatchNorm1d(hidden_dim)  
        self.batchnorm2 = nn.BatchNorm1d(hidden_dim)  
        self.mlp1 = nn.Linear(hidden_dim, output_dim)
        self.mlp2 = nn.Linear(hidden_dim, output_dim)
        self.sigmoid1 = nn.Sigmoid()  
        self.sigmoid2 = nn.Sigmoid()  
        self.R = torch.randn(c, hidden_dim, device=device)

        self.query_transforms = nn.ModuleList([nn.Linear(hidden_dim, self.head_dim).to(device) for _ in range(num_heads)])
        self.key_transforms = nn.ModuleList([nn.Linear(hidden_dim, self.head_dim).to(device) for _ in range(num_heads)])
        self.value_transforms = nn.ModuleList([nn.Linear(hidden_dim, self.head_dim).to(device) for _ in range(num_heads)])

    def attention(self, query, key, value):
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        p_attn = F.softmax(scores, dim=-1)
        return torch.matmul(p_attn, value), p_attn

    def multi_head_attention(self, query, key, value):
        head_outputs = []
        attentions = []
        for q, k, v in zip(self.query_transforms, self.key_transforms, self.value_transforms):
            q_transformed = q(query)
            k_transformed = k(key)
            v_transformed = v(value)
            out, attn = self.attention(q_transformed, k_transformed, v_transformed)
            head_outputs.append(out)
            attentions.append(attn)
        out_concat = torch.cat(head_outputs, dim=-1)
        return out_concat, attentions

    def forward(self, x1, x2):
        output, _ = self.lstm(x1)
        S = output[:, -1, :]
        B, _ = self.multi_head_attention(self.R, S, S)  
        S_prime, _ = self.multi_head_attention(S, B, B)
        alpha = torch.sigmoid(self.gate(S))
        alpha = alpha.expand_as(S)
        S_mix = alpha * S_prime + (1 - alpha) * S
        x = F.relu(self.mlp(S_mix))
        x = self.dropout1(x)
        S_mix = x + S_mix
        output = self.batchnorm1(S_mix)  
        output = self.mlp1(S_mix)
        output1 = self.sigmoid1(output)  

        output, _ = self.lstm(x2)
        S = output[:, -1, :]
        B, _ = self.multi_head_attention(self.R, S, S)  
        S_prime, _ = self.multi_head_attention(S, B, B)
        alpha = torch.sigmoid(self.gate(S))
        alpha = alpha.expand_as(S)
        S_mix = alpha * S_prime + (1 - alpha) * S
        x = F.relu(self.mlp(S_mix))
        x = self.dropout2(x)
        S_mix = x + S_mix
        output = self.batchnorm2(S_mix)  
        output = self.mlp2(S_mix)
        output2 = self.sigmoid2(output)  
        part1 = torch.exp(-output1.sum())
        part2 = torch.exp(-output2.sum())
        alpha = part1 / (part1+part2+1e-8)
        return alpha * output1 + (1-alpha) * output2
    
class multigru_independent(nn.Module):
    def __init__(self, input_size1, input_size2, hidden_dim, num_layers, output_dim, dropout_rate, c=30, num_heads=2, device='cpu'):
        super(multigru_independent, self).__init__()
        self.cross1 = CrossGRU(input_size1, hidden_dim, num_layers, output_dim, dropout_rate, c, num_heads, device)
        self.cross2 = CrossGRU(input_size2, hidden_dim, num_layers, output_dim, dropout_rate, c, num_heads, device)
    def forward(self, x1, x2):
        output1 = self.cross1(x1)
        output2 = self.cross2(x2)
        part1 = torch.exp(-output1.sum())
        part2 = torch.exp(-output2.sum())
        alpha = part1 / (part1+part2+1e-8)
        return alpha * output1 + (1-alpha) * output2
    
class MultiGRU_Independent(nn.Module):
    def __init__(self, input_sizes, hidden_dim, num_layers, output_dim, dropout_rate, c=30, num_heads=2, device='cpu'):
        super(MultiGRU_Independent, self).__init__()
        self.cross_grus = nn.ModuleList([
            CrossGRU(input_size, hidden_dim, num_layers, output_dim, dropout_rate, c, num_heads, device)
            for input_size in input_sizes])
    def forward(self, inputs):
        if len(inputs) != len(self.cross_grus):
            print(len(inputs),len(self.cross_grus))
            raise ValueError("Number of inputs does not match the number of initialized CrossGRU modules.")
        outputs = []  
        for input_, cross_gru in zip(inputs, self.cross_grus):
            output = cross_gru(input_)
            outputs.append(output)

        return torch.mean(torch.stack(outputs), dim=0)