import torch
from torch import nn, Tensor
import math
import numpy as np
# from torchinfo import summary

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
class PositionalEncoding1D(nn.Module):
    def __init__(self, model_dim: int, sequence_length: int):
        super(PositionalEncoding1D, self).__init__()
        self.model_dim = model_dim
        self.position = torch.arange(sequence_length).unsqueeze(1)
        self.angle_rads = self.get_angles(self.position, torch.arange(model_dim).unsqueeze(0), model_dim)
        self.angle_rads[:, 0::2] = torch.sin(self.angle_rads[:, 0::2])
        self.angle_rads[:, 1::2] = torch.cos(self.angle_rads[:, 1::2])
        pe = self.angle_rads.unsqueeze(0)
        self.register_buffer('pe', pe)

    def get_angles(self, pos, i, d_model):
        angle_rates = 1 / torch.pow(10000, (2 * (i//2)) / torch.tensor(d_model).float())
        return pos * angle_rates

    def forward(self, x: Tensor) -> Tensor:
        return x + self.pe[:, :x.size(1), :]

class PositionalEncoding2D(nn.Module):
    def __init__(self, model_dim: int, height: int, width:int):
        super(PositionalEncoding2D, self).__init__()
        assert model_dim % 2 == 0
        self.model_dim = model_dim // 2
        self.row_pos = torch.arange(height).unsqueeze(1).repeat(1, width).view(-1, 1)
        self.col_pos = torch.arange(width).unsqueeze(0).repeat(height, 1).view(-1, 1)
        self.angle_rads_row = self.get_angles(self.row_pos, torch.arange(self.model_dim).unsqueeze(0), self.model_dim)
        self.angle_rads_col = self.get_angles(self.col_pos, torch.arange(self.model_dim).unsqueeze(0), self.model_dim)
        self.angle_rads_row[:, 0::2] = torch.sin(self.angle_rads_row[:, 0::2])
        self.angle_rads_row[:, 1::2] = torch.cos(self.angle_rads_row[:, 1::2])
        self.angle_rads_col[:, 0::2] = torch.sin(self.angle_rads_col[:, 0::2])
        self.angle_rads_col[:, 1::2] = torch.cos(self.angle_rads_col[:, 1::2])
        pe = torch.cat([self.angle_rads_row, self.angle_rads_col], dim=1).unsqueeze(0)
        self.register_buffer('pe', pe)

    def get_angles(self, pos, i, d_model):
        angle_rates = 1 / torch.pow(10000, (2 * (i//2)) / torch.tensor(d_model).float())
        return pos * angle_rates

    def forward(self, x: Tensor) -> Tensor:
        return x + self.pe[:, :x.size(1), :]


class MultiHeadAttention(nn.Module):
    def __init__(self, model_dim: int, num_heads: int):
        super(MultiHeadAttention, self).__init__()

        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads

        self.query_linear = nn.Linear(model_dim, model_dim)
        self.key_linear = nn.Linear(model_dim, model_dim)
        self.value_linear = nn.Linear(model_dim, model_dim)
        self.fc_out = nn.Linear(model_dim, model_dim)

    def scaled_dot_product_attention(self, query: Tensor, key: Tensor, value: Tensor, mask: Tensor = None) -> Tensor:

        attn_scores = torch.matmul(query, key.transpose(-2, -1))

        dk = key.size(-1)
        scaled_attention_logits = attn_scores / torch.sqrt(torch.tensor([dk]).to(device))

        if mask is not None:
            scaled_attention_logits += (mask * -1e9)
        
        attn_weights = nn.functional.softmax(scaled_attention_logits, dim = -1)
        output = torch.matmul(attn_weights, value)

        return output

    def forward(self, query: Tensor, key: Tensor, value: Tensor, mask: Tensor = None) -> Tensor:
        N = query.shape[0]
        query = self.query_linear(query).view(N, -1, self.num_heads, self.head_dim)
        key = self.key_linear(key).view(N, -1, self.num_heads, self.head_dim)
        value = self.value_linear(value).view(N, -1, self.num_heads, self.head_dim)

        query = query.permute(0, 2, 1, 3)
        key = key.permute(0, 2, 1, 3)
        value = value.permute(0, 2, 1, 3)

        attn_output = self.scaled_dot_product_attention(query, key, value, mask)
        out = attn_output.permute(0, 2, 1, 3).contiguous().view(N, -1, self.model_dim)

        out = self.fc_out(out)
        return out
    

class EncoderBlock(nn.Module):
    def __init__(self, model_dim: int, num_heads: int, feed_forward_dim: int = 2048, dropout: float = 0.1):
        super(EncoderBlock, self).__init__()

        self.mha = MultiHeadAttention(model_dim, num_heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(model_dim, feed_forward_dim),
            nn.ReLU(),
            nn.Linear(feed_forward_dim, model_dim)
        )

        self.layer_norm1 = nn.LayerNorm(model_dim, eps = 1e-6)
        self.layer_norm2 = nn.LayerNorm(model_dim, eps = 1e-6)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: Tensor, mask = None) -> Tensor:
        attention_output = self.mha(x, x, x, mask)
        attention_output = self.dropout1(attention_output)
        x = self.layer_norm1(x + attention_output)

        feed_forward_output = self.feed_forward(x)
        feed_forward_output = self.dropout2(feed_forward_output)
        output = self.layer_norm2(x + feed_forward_output)

        return output
    
class DecoderBlock(nn.Module):
    def __init__(self, model_dim: int, num_heads: int, feed_forward_dim: int = 2048, dropout: float = 0.1):
        super(DecoderBlock, self).__init__()

        self.mha1 = MultiHeadAttention(model_dim, num_heads)
        self.mha2 = MultiHeadAttention(model_dim, num_heads)

        self.feed_forward = nn.Sequential(
            nn.Linear(model_dim, feed_forward_dim),
            nn.ReLU(),
            nn.Linear(feed_forward_dim, model_dim)
        )

        self.layer_norm1 = nn.LayerNorm(model_dim, eps = 1e-6)
        self.layer_norm2 = nn.LayerNorm(model_dim, eps = 1e-6)
        self.layer_norm3 = nn.LayerNorm(model_dim, eps = 1e-6)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x: Tensor, enc_output: Tensor, look_ahead_mask: Tensor = None, padding_mask: Tensor = None) -> Tensor:
        attn_output1 = self.mha1(x, x, x, look_ahead_mask)
        attn_output1 = self.dropout1(attn_output1)  
        x = self.layer_norm1(attn_output1 + x)

        attn_output2 = self.mha2(x, enc_output, enc_output, padding_mask)
        attn_output2 = self.dropout2(attn_output2)
        x = self.layer_norm2(attn_output2 + x)

        ffn_output = self.feed_forward(x)
        ffn_output = self.dropout3(ffn_output)
        output = self.layer_norm3(ffn_output + x)

        return output
    
class Encoder(nn.Module):
    def __init__(self, num_layers: int, model_dim: int, input_dim: int, num_heads: int, height: int, width: int, feed_forward_dim: int = 2048, dropout: float = 0.1):
        super(Encoder, self).__init__()
        self.model_dim = model_dim
        self.num_layers = num_layers
        self.embedding = nn.Sequential(
            nn.Linear(input_dim, model_dim),
            nn.ReLU(),
        )
        self.pos_enc = PositionalEncoding2D(model_dim, height, width)
        self.layers = nn.ModuleList([
            EncoderBlock(model_dim, num_heads, feed_forward_dim, dropout)
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        x = self.embedding(x)
        x = self.pos_enc(x)
        x = self.dropout(x)
        for i in range(self.num_layers):
            x = self.layers[i](x, mask)
        return x

class Decoder(nn.Module):
    def __init__(self, num_layers: int, model_dim: int, num_heads: int, feed_forward_dim: int, vocab_size: int, sequence_length: int, dropout: float = 0.1):
        super(Decoder, self).__init__()
        self.model_dim = model_dim
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, model_dim)
        self.pos_enc = PositionalEncoding1D(model_dim, sequence_length)
        self.layers = nn.ModuleList([
            DecoderBlock(model_dim, num_heads, feed_forward_dim, dropout)
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: Tensor, enc_output: Tensor, enc_mask: Tensor = None, look_ahead_mask: Tensor = None) -> Tensor:
        x = self.embedding(x)
        x *= torch.sqrt(torch.Tensor([self.model_dim]).to(device))
        x = self.pos_enc(x)
        x = self.dropout(x)
        for i in range(self.num_layers):
            x = self.layers[i](x, enc_output, enc_mask, look_ahead_mask)
        return x
    
class TransformerModel(nn.Module):

    def __init__(self, model_dim: int, input_dim: int, num_heads: int, num_layers: int, vocab_size:int, sequence_length: int = 34, height: int = 7, width: int = 7, feed_forward_dim: int = 2048, dropout: float = 0.1):
        super(TransformerModel, self).__init__()
        self.model_dim = model_dim
        self.encoder = Encoder(num_layers, model_dim, input_dim, num_heads, height, width, feed_forward_dim, dropout)
        self.decoder = Decoder(num_layers, model_dim, num_heads, feed_forward_dim, vocab_size, sequence_length, dropout)
        self.final_layer = nn.Linear(model_dim, vocab_size)

    def padding_mask(self, input):
        input = (input == 0).float()
        return input.unsqueeze(1).unsqueeze(2)

    def look_ahead_mask(self, shape):
        mask = 1 - torch.tril(torch.ones(shape, shape))
        return mask
    
    def create_masks_decoder(self, target):
        target = target.to(device)
        dec_look_ahead_mask = self.look_ahead_mask(target.size(1)).to(device)
        dec_padding_mask = self.padding_mask(target).to(device)
        combined_mask = torch.max(dec_padding_mask, dec_look_ahead_mask)
        return combined_mask

    def forward(self, input: Tensor, target: Tensor, enc_mask: Tensor = None, src_mask: Tensor = None, tgt_mask: Tensor = None, test: bool = True) -> Tensor:
        dec_input_mask = self.create_masks_decoder(target)
        enc_output = self.encoder(input, enc_mask)
        dec_output = self.decoder(target, enc_output, dec_input_mask, tgt_mask) if test else self.decoder(target, enc_output, src_mask, tgt_mask)
        output = self.final_layer(dec_output)

        return output


if __name__ == '__main__':
    inp = torch.randn(32, 49, 768)
    tgt = torch.randint(0, 8357, (32, 34))
    model = TransformerModel(512, 768, 8, 4, 8357, height = 7, width = 7)
    # summary(model, input_data=[inp, tgt])
    out = model(inp, tgt)
