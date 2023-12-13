import torch
import torch.nn as nn
import math

class PegasusXAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(PegasusXAttention, self).__init__()
        # Ensure that the model dimension (d_model) is divisible by the number of heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        # Initialize dimensions
        self.d_model = d_model # Model's dimension
        self.num_heads = num_heads # Number of attention heads
        self.d_k = d_model // num_heads # Dimension of each head's key, query, and value
        
        # Linear layers for transforming inputs
        self.W_q = nn.Linear(d_model, d_model) # Query transformation
        self.W_k = nn.Linear(d_model, d_model) # Key transformation
        self.W_v = nn.Linear(d_model, d_model) # Value transformation
        self.W_o = nn.Linear(d_model, d_model) # Output transformation
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        # Calculate attention scores
        attn_scores = torch.matmul(Q, K.transpose(-2, -1))
        
        # Apply mask if provided (useful for preventing attention to certain parts like padding)
        if mask is not None:
            # shape of mask: batch_size, tgt_len, src_len
            attn_scores = attn_scores + mask
        
        # Softmax is applied to obtain attention probabilities
        attn_probs = torch.softmax(attn_scores, dim=-1)
        
        # Multiply by values to obtain the final output
        output = torch.matmul(attn_probs, V)
        return output
        
    def split_heads(self, x):
        # Reshape the input to have num_heads for multi-head attention
        batch_size, seq_length, _ = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
        
    def combine_heads(self, x):
        # Combine the multiple heads back to original shape
        batch_size, _, seq_length, _ = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)
        
    def forward(self, Q, K, V, mask=None):
        # Apply linear transformations and split heads
        Q = self.split_heads(self.W_q(Q) / math.sqrt(self.d_k))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))
        
        # Perform scaled dot-product attention
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # Combine heads and apply output transformation
        output = self.W_o(self.combine_heads(attn_output))
        return output

class PegasusXGlobalLocalAttention(nn.Module):
    def __init__(self, d_model, num_heads, block_size, global_len, padded_seq_len):
        super(PegasusXGlobalLocalAttention, self).__init__()
        # Ensure that the model dimension (d_model) is divisible by the number of heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        # Initialize dimensions
        self.d_model = d_model # Model's dimension
        self.num_heads = num_heads # Number of attention heads
        self.d_k = d_model // num_heads # Dimension of each head's key, query, and value
        self.block_size = block_size
        self.padded_seq_len = padded_seq_len
        self.num_blocks = self.padded_seq_len // block_size
        self.global_len = global_len
        
        # Linear layers for transforming inputs
        self.W_q = nn.Linear(d_model, d_model) # Query transformation
        self.W_k = nn.Linear(d_model, d_model) # Key transformation
        self.W_v = nn.Linear(d_model, d_model) # Value transformation
        self.W_o = nn.Linear(d_model, d_model) # Output transformation
        
    def split_heads(self, x):
        # Reshape the input to have num_heads for multi-head attention
        batch_size, seq_length, _ = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
        
    def combine_heads(self, x):
        # Combine the multiple heads back to original shape
        batch_size, _, seq_length, _ = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)
        
    def compute_global_attention(self, global_k, global_q, global_v, local_k, local_v, mask):
        global_and_local_k = torch.cat([global_k, local_k], dim=2)
        global_and_local_v = torch.cat([global_v, local_v], dim=2)
        extended_mask = nn.functional.pad(mask, pad=(self.global_len, 0))

        attn_weights = torch.einsum("BHGF,BHXF->BHGX", global_q, global_and_local_k) / math.sqrt(self.d_k)
        attn_weights = attn_weights + extended_mask[:, None, None, :]
        attn_probs = nn.functional.softmax(attn_weights, dim=-1)

        attn_output = torch.einsum("BHGX,BHXF->BHGF", attn_probs, global_and_local_v)
        return attn_output

    def compute_local_attention(self, local_k, local_q, local_v, global_k, global_v, mask):
        blocked_local_k = local_k.view(self.batch_size, self.num_heads, self.num_blocks, self.block_size, self.d_k)
        blocked_local_q = local_q.view(self.batch_size, self.num_heads, self.num_blocks, self.block_size, self.d_k)
        blocked_local_v = local_v.view(self.batch_size, self.num_heads, self.num_blocks, self.block_size, self.d_k)

        attn_local2global = torch.enisum("BHNKF,BHGF->BHNKG", blocked_local_q, global_k)
        attn_local2local = torch.einsum("BHNKF,BHNXF->BHNKX", blocked_local_q, blocked_local_k)

        extended_mask = nn.functional.pad(
            mask.view(self.batch_size, self.num_blocks, self.block_size),
            pad=(self.global_len, 0)
        )

        attn_weights = torch.cat((attn_local2global, attn_local2local), dim=-1)
        attn_weights = attn_weights + extended_mask[:, None, :, None, :]
        attn_probs = nn.functional.softmax(attn_weights, dim=-1)

        global_v = global_v.unsqueeze(2).expand(-1,-1,self.num_blocks, -1, -1)
        global_and_local_v = torch.cat((global_v, blocked_local_v), dim=3)

        attn_output = torch.einsum("BHNKX,BHNXF->BHNKF", attn_probs, global_and_local_v)
        return attn_output

    def forward(self, Q, K, V, G, mask=None):
        # Apply linear transformations and split heads
        local_q = self.split_heads(self.W_q(Q) / math.sqrt(self.d_k) )
        local_k = self.split_heads(self.W_k(K))
        local_v = self.split_heads(self.W_v(V))

        global_q = self.split_heads(self.W_q(G) / math.sqrt(self.d_k))
        global_k = self.split_heads(self.W_k(G))
        global_v = self.split_heads(self.W_v(G))

        local_output = self.compute_local_attention(local_k, local_q, local_v, global_k, global_v, mask)
        global_output = self.compute_global_attention(global_k, global_q, global_v, local_k, local_v, mask)

        local_output = local_output.permute(0, 2, 3, 1, 4).contiguous()
        # [batch_size, padded_seq_len, hidden_dim]
        local_output = local_output.view(self.batch_size, self.padded_seq_len, self.d_model)

        # Combine heads and apply output transformation
        local_output = self.W_o(local_output)
        global_output = self.W_o(self.combine_heads(global_output))
        return local_output, global_output
    
class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class PegasusXEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, block_size, num_global_tokens, 
                 padded_seq_len, d_ff, dropout, stagger_blocks = False):
        super(PegasusXEncoderLayer, self).__init__()
        self.self_attn = PegasusXGlobalLocalAttention(d_model, num_heads, block_size, num_global_tokens, padded_seq_len)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.stagger_blocks = stagger_blocks
        
    def pad_local_tokens(self, hidden_states, mask, block_size, mask_value):
        pad_size = block_size // 2
        
        padded_hidden_states = nn.functional.pad(
            hidden_states,
            pad=(0, 0, pad_size, pad_size),
        )
        padded_mask = nn.functional.pad(
            mask,
            pad=(pad_size, pad_size),
            value=mask_value,
        )
        return padded_hidden_states, padded_mask

    def forward(self, hidden_states, global_hidden_states, mask):
        if self.stagger_blocks:
            mask_value = torch.finfo(torch.float32).min
            hidden_states, mask = self.pad_local_tokens(hidden_states=hidden_states, mask=mask, 
                                                        block_size=self.block_size, mask_value = mask_value)
        
        local_attn_output, global_attn_output = self.self_attn(hidden_states, hidden_states, 
                                                               hidden_states, global_hidden_states, mask)
        if self.stagger_blocks:
            pad_size = self.block_size // 2
            local_attn_output = local_attn_output[:, pad_size:-pad_size, :]
        
        hidden_states = self.norm1(hidden_states + self.dropout(local_attn_output))
        ff_output = self.feed_forward(hidden_states)
        hidden_states = self.norm2(hidden_states + self.dropout(ff_output))

        global_hidden_states = self.norm1(global_hidden_states + self.dropout(global_attn_output))
        ff_output = self.feed_forward(global_hidden_states)
        global_hidden_states = self.norm2(global_hidden_states + self.dropout(ff_output))

        return (hidden_states, global_hidden_states)

class PegasusXDecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(PegasusXDecoderLayer, self).__init__()
        self.self_attn = PegasusXAttention(d_model, num_heads)
        self.cross_attn = PegasusXAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, enc_output, src_mask, tgt_mask):
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x

class PegasusXModel(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, src_num_layers, tgt_num_layers, 
                 block_size, num_global_tokens, d_ff, src_padded_seq_len, 
                 tgt_padded_seq_len, dropout, masked_prediction=False):
        super(PegasusXModel, self).__init__()
        self.masked_prediction = masked_prediction
        self.num_global_tokens = num_global_tokens
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.embed_global = nn.Embedding(num_global_tokens, d_model)
        self.src_positional_encoding = PositionalEncoding(d_model, src_padded_seq_len)
        self.tgt_positional_encoding = PositionalEncoding(d_model, tgt_padded_seq_len)

        self.encoder_layers = nn.ModuleList([PegasusXEncoderLayer(d_model, num_heads, block_size, num_global_tokens, 
                                                                  src_padded_seq_len, d_ff, dropout, 
                                                                  stagger_blocks=(i%2==1)) for i in range(src_num_layers)])
        self.decoder_layers = nn.ModuleList([PegasusXDecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(tgt_num_layers)])

        self.fc = nn.Linear(d_model, tgt_vocab_size)
        self.masked_prediction = masked_prediction
        if self.masked_prediction:
            self.fc1 = nn.Linear(d_model, d_model)
            self.relu = nn.ReLu()
            self.norm = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def generate_mask(self, src_attn_mask, tgt_attn_mask):
        src_attn_mask = src_attn_mask.to(dtype=torch.float32)
        mask_min_value = torch.finfo(torch.float32).min
        src_attn_mask = 1.0 - src_attn_mask
        src_attn_mask = src_attn_mask.masked_fill(
            src_attn_mask.to(torch.bool),
            mask_min_value,
        )

        tgt_attn_mask = tgt_attn_mask.unsqueeze(1).unsqueeze(3)
        tgt_seq_length = tgt_attn_mask.size(1)
        nopeak_mask = (1 - torch.triu(torch.ones(1, tgt_seq_length, tgt_seq_length), diagonal=1)).bool()
        tgt_attn_mask = tgt_attn_mask & nopeak_mask
        tgt_attn_mask = 1.0 - tgt_attn_mask
        tgt_attn_mask = tgt_attn_mask.masked_fill(
            tgt_attn_mask.to(torch.bool),
            mask_min_value,
        )
        return src_attn_mask, tgt_attn_mask

    def forward(self, src, tgt):
        src_attn_mask, tgt_attn_mask = self.generate_mask(src, tgt)
        src_embedded = self.dropout(self.src_positional_encoding(self.encoder_embedding(src)))
        tgt_embedded = self.dropout(self.tgt_positional_encoding(self.decoder_embedding(tgt)))
        batch_size=1
        global_hidden_states = self.dropout(self.embed_global(
            torch.arange(self.num_global_tokens, device='cuda')[None].expand(batch_size, -1)
        ))

        enc_output = src_embedded
        for enc_layer in self.encoder_layers:
            enc_output, global_hidden_states = enc_layer(enc_output, global_hidden_states, src_attn_mask)

        dec_output = tgt_embedded
        src_attn_mask = src_attn_mask[:,None, None, :]
        tgt_attn_mask = tgt_attn_mask[:, None, :, :]
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, enc_output, src_attn_mask, tgt_attn_mask)

        output = self.fc(dec_output)
        if self.masked_prediction:
            output_logits = self.fc(self.norm(self.relu(self.fc1(enc_output))))
        else:
            output_logits = None
        return enc_output, output, output_logits