import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence

# =========================================================
# 模型保存与加载
# =========================================================

def save_model(model, path="transformer_decoder.pth"):
    """保存模型的 state_dict"""
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

def load_model(model_class, path, **kwargs):
    """加载模型的 state_dict"""
    model = model_class(**kwargs)
    # map_location="cpu" 确保加载到 CPU，方便跨设备使用
    model.load_state_dict(torch.load(path, map_location="cpu")) 
    print(f"Model loaded from {path}")
    return model

def sample_next_token(logits, temperature=1.0, top_k=50):
    """使用 Top-k 采样从 logits 中选择下一个 token"""
    logits = logits / temperature

    # Top-k
    values, indices = torch.topk(logits, top_k)
    probs = torch.softmax(values, dim=-1)

    next_idx = torch.multinomial(probs, num_samples=1)
    next_token = indices.gather(-1, next_idx)

    return next_token

# =========================================================
# 模型定义
# =========================================================

class MultiHeadAttention(nn.Module):
    """多头自注意力机制"""
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads # 每个头的维度
        
        # 线性层用于生成 Q, K, V
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model) # 输出线性层

    def forward(self, Q, K, V, mask=None):
        bsz = Q.shape[0]
        
        # 1. 线性变换并分割成多头
        # [B, L, D] -> [B, L, H, d_k] -> [B, H, L, d_k] (H: num_heads)
        Q = self.W_q(Q).reshape(bsz, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(K).reshape(bsz, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(V).reshape(bsz, -1, self.num_heads, self.d_k).transpose(1, 2)

        # 2. 计算 Attention Score
        # scores: [B, H, L_q, L_k]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # 3. 应用 Mask (针对 Decoder 的 Look-Ahead Mask 或 Padding Mask)
        if mask is not None:
            # 必须转换为 float32 才能与 float32 的 scores 进行 masked_fill
            mask = mask.to(dtype=torch.float32) 
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # 4. Softmax 得到注意力权重
        attn = torch.softmax(scores, dim=-1)
        
        # 5. 加权求和
        # output: [B, H, L_q, d_k]
        output = torch.matmul(attn, V)
        
        # 6. 拼接多头并进行最终线性变换
        # [B, L_q, H, d_k] -> [B, L_q, D]
        output = output.transpose(1, 2).reshape(bsz, -1, self.d_model)
        return self.W_o(output)


class PositionwiseFeedForward(nn.Module):
    """前馈网络 (FFN)"""
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear2(torch.relu(self.linear1(x)))


class PositionalEncoding(nn.Module):
    """正弦位置编码"""
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        # 计算分母 1/10000^(2i/d_model)
        div = torch.exp(-math.log(10000.0) * torch.arange(0, d_model, 2).float() / d_model)
        
        # 偶数位置使用 sin
        pe[:, 0::2] = torch.sin(pos * div) 
        # 奇数位置使用 cos
        pe[:, 1::2] = torch.cos(pos * div)
        
        # 注册为 buffer，不会作为模型参数训练
        self.register_buffer('pe', pe.unsqueeze(0)) 

    def forward(self, x):
        # 将位置编码加到输入 embedding 上
        return x + self.pe[:, :x.size(1), :]


class DecoderLayer(nn.Module):
    """Transformer Decoder 的单层"""
    def __init__(self, d_model, num_heads, d_ff, use_ffn=True, use_norm=True, use_attn=True):
        super().__init__()
        self.use_ffn = use_ffn
        self.use_norm = use_norm
        self.use_attn = use_attn

        if use_attn:
            # 掩码自注意力（Decoder 独有）
            self.self_attn = MultiHeadAttention(d_model, num_heads) 
            # 交叉注意力 (如果作为完整的 Seq2Seq Decoder，这里需要)
            self.enc_dec_attn = MultiHeadAttention(d_model, num_heads) 
        if use_ffn:
            self.ffn = PositionwiseFeedForward(d_model, d_ff)

        # Layer Normalization
        if use_norm:
            self.norm1 = nn.LayerNorm(d_model) # 自注意力后
            self.norm2 = nn.LayerNorm(d_model) # 交叉注意力后
            self.norm3 = nn.LayerNorm(d_model) # FFN 后

    def forward(self, x, enc_output=None, tgt_mask=None, memory_mask=None):
        # 1. Self-Attention Block
        if self.use_attn:
            # 输入 Q, K, V 都是 x (自注意力)
            attn_out1 = self.self_attn(x, x, x, mask=tgt_mask) 
            x = x + attn_out1 # 残差连接
            if self.use_norm: x = self.norm1(x) # LayerNorm

        # 2. Encoder-Decoder Attention Block (如果提供了 Encoder 输出)
        if self.use_attn and enc_output is not None:
            # Q 是 Decoder 的输出 x，K 和 V 是 Encoder 的输出 enc_output
            attn_out2 = self.enc_dec_attn(x, enc_output, enc_output, mask=memory_mask)
            x = x + attn_out2 # 残差连接
            if self.use_norm: x = self.norm2(x) # LayerNorm
            
        # 3. Feed Forward Network Block
        if self.use_ffn:
            ffn_out = self.ffn(x)
            x = x + ffn_out # 残差连接
            if self.use_norm: x = self.norm3(x) # LayerNorm
            
        return x


class TransformerDecoder(nn.Module):
    """完整的 Transformer Decoder (用于语言模型)"""
    def __init__(self, num_layers, num_heads, vocab_size, d_ff, d_model,
                 max_len=5000, use_ffn=True, use_norm=True, use_pos=True, use_attn=True):
        super().__init__()
        self.use_pos = use_pos
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        if use_pos:
            self.pos_encoding = PositionalEncoding(d_model, max_len)

        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, use_ffn=use_ffn, use_norm=use_norm, use_attn=use_attn)
            for _ in range(num_layers)
        ])
        
        # 最后的线性层，将 d_model 映射回 vocab_size
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, x, enc_output=None, tgt_mask=None):
        # 1. Embedding
        x = self.embedding(x)
        
        # 2. Positional Encoding
        if self.use_pos:
            x = self.pos_encoding(x)
            
        # 3. Decoder Layers
        for layer in self.layers:
            # 在语言模型任务中，通常 enc_output=None，只使用自注意力
            x = layer(x, enc_output, tgt_mask) 
            
        # 4. Final Output Projection
        return self.fc_out(x)