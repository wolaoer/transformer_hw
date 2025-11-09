import math
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd
from modelscope.msdatasets import MsDataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from modelscope import AutoTokenizer
import os # 用于创建模型保存目录

# 从 model.py 导入所需的类和函数
from model import TransformerDecoder, save_model

# 确保 Qwen Tokenizer 可用
tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen-tokenizer",
    use_fast=True
)

# =========================================================
# 1 数据加载与处理
# =========================================================
def load_wikitext2_modelscope():
    """加载 WikiText-2 数据集并进行分词"""
    print("Loading WikiText-2-v1 from ModelScope ...")

    # 加载数据集
    train_ds = MsDataset.load('modelscope/wikitext', subset_name='wikitext-2-v1', split='train', trust_remote_code=True)
    val_ds = MsDataset.load('modelscope/wikitext', subset_name='wikitext-2-v1', split='validation', trust_remote_code=True)
    test_ds = MsDataset.load('modelscope/wikitext', subset_name='wikitext-2-v1', split='test', trust_remote_code=True)

    def tokenize(example):
        """对单个文本样本进行分词"""
        encoded = tokenizer(
            example['text'],
            truncation=False,
            add_special_tokens=False
        )
        ids = encoded["input_ids"]
        # 返回 LongTensor
        return torch.tensor(ids, dtype=torch.long) 

    # 分词处理所有数据
    train_data = [tokenize(ex) for ex in train_ds]
    val_data = [tokenize(ex) for ex in val_ds]
    test_data = [tokenize(ex) for ex in test_ds]

    # 过滤空样本
    train_data = [t for t in train_data if t.numel() > 0]
    val_data = [t for t in val_data if t.numel() > 0]
    test_data = [t for t in test_data if t.numel() > 0]

    return train_data, val_data, test_data


def create_look_ahead_mask(size):
    """创建 Look-Ahead Mask (上三角为 0)"""
    # tril: 下三角矩阵，即对角线和对角线以下为 1
    mask = torch.tril(torch.ones(size, size)) 
    # [1, 1, size, size]
    return mask.unsqueeze(0).unsqueeze(0)


def collate_fn(batch):
    """DataLoader 的 batch 整理函数，用于对序列进行 Padding"""
    # padding_value=0 是因为 Qwen tokenizer 的 padding token ID 是 0
    return pad_sequence(batch, batch_first=True, padding_value=0) 


# =========================================================
# 2 训练与评估函数
# =========================================================
def train_model(model, data_loader, optimizer, criterion, device):
    """单次训练循环"""
    model.train()
    total_loss = 0
    for batch in data_loader:
        batch = batch.to(device)
        # 输入是 [0, L-1]，目标是 [1, L]
        src = batch[:, :-1] 
        tgt = batch[:, 1:] 
        
        # 创建 Look-Ahead Mask 并转移到设备
        mask = create_look_ahead_mask(src.size(1)).to(device) 
        logits = model(src, tgt_mask=mask)

        # 计算 Loss (展平处理)
        loss = criterion(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1)) 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(data_loader)


def evaluate_model(model, data_loader, criterion, device):
    """模型评估，计算平均 Loss 和 Perplexity"""
    model.eval()
    total_loss = 0.0
    valid_batches = 0

    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            src = batch[:, :-1]
            tgt = batch[:, 1:]
            mask = create_look_ahead_mask(src.size(1)).to(device)

            logits = model(src, tgt_mask=mask)
            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))

            # 防止 loss 出现 NaN
            if torch.isnan(loss): 
                continue

            total_loss += loss.item()
            valid_batches += 1

    avg_loss = total_loss / max(valid_batches, 1)
    # Perplexity = exp(Loss)
    perplexity = math.exp(avg_loss) if avg_loss < 20 else float('inf')  
    return avg_loss, perplexity


# =========================================================
# 3 主程序：执行消融实验
# =========================================================
def run_ablation():
    """执行 Transformer Decoder 的消融实验"""
    
    # 硬件配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 模型和训练参数
    vocab_size = tokenizer.vocab_size
    d_model = 128
    num_heads = 4
    d_ff = 512
    num_layers = 2
    batch_size = 16
    epochs = 10

    # 1. 加载数据
    train_data, val_data, test_data = load_wikitext2_modelscope()

    # 2. 创建 DataLoader
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    # 3. 定义消融实验配置
    experiments = {
        "Baseline (Full)": dict(use_pos=True, use_attn=True),
        "No PosEncoding": dict(use_pos=False, use_attn=True), # 移除位置编码
        "No Attention": dict(use_pos=True, use_attn=False) # 移除注意力机制 (只剩 FFN 和 Norm)
    }

    results = {}

    # 4. 运行每个实验
    for name, cfg in experiments.items():
        print(f"\nRunning experiment: {name}")
        
        # 初始化模型
        model = TransformerDecoder(
            num_layers=num_layers,
            num_heads=num_heads,
            vocab_size=vocab_size,
            d_ff=d_ff,
            d_model=d_model,
            use_pos=cfg["use_pos"],
            use_ffn=True, 
            use_norm=True,
            use_attn=cfg["use_attn"]
        ).to(device)

        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        # 忽略 padding token (ID=0) 的损失
        criterion = nn.CrossEntropyLoss(ignore_index=0) 

        train_losses, val_losses, val_ppls = [], [], []

        # 训练循环
        for epoch in range(epochs):
            train_loss = train_model(model, train_loader, optimizer, criterion, device)
            val_loss, perplexity = evaluate_model(model, val_loader, criterion, device)
            print(f"  Epoch {epoch+1}/{epochs} | Train={train_loss:.4f} | Val={val_loss:.4f} | PPL={perplexity:.2f}")

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            val_ppls.append(perplexity)

        # 保存模型
        # 对文件名中的特殊字符进行清理
        model_save_path = f"../weight/{name.replace(' ','_').replace('(','').replace(')','')}.pth"
        save_model(model, model_save_path) 

        # 最终测试集评估
        test_loss, perplexity = evaluate_model(model, test_loader, criterion, device)
        results[name] = dict(train=train_losses, val=val_losses, ppl=val_ppls, test=test_loss)
        print(f" Test Loss ({name}) = {test_loss:.4f} | Perplexity = {perplexity:.2f}")
        
    # 5. 绘制和输出结果
    
    # --- 绘制 Loss 曲线 ---
    plt.figure(figsize=(9,6))
    for name, res in results.items():
        plt.plot(res["train"], label=f"{name} Train")
        plt.plot(res["val"], linestyle='--', label=f"{name} Val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train / Val Loss (Ablation Study)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("loss_curve.png")
    plt.show()

    # --- 绘制 Perplexity 曲线 ---
    plt.figure(figsize=(9,6))
    for name, res in results.items():
        plt.plot(res["ppl"], label=f"{name} Val PPL")
    plt.xlabel("Epoch")
    plt.ylabel("Perplexity")
    plt.title("Validation Perplexity Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig("perplexity_curve.png")
    plt.show()

    # --- 输出 Test Loss 表 ---
    df = pd.DataFrame({
        "Model": results.keys(),
        "Test Loss": [f"{v['test']:.4f}" for v in results.values()]
    })
    print("\nFinal Test Loss:")
    print(df.to_markdown(index=False))


if __name__ == "__main__":
    run_ablation()