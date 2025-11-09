import torch
import random
from modelscope import AutoTokenizer

# 从 model.py 和 train.py 导入所需函数和类
from model import TransformerDecoder, sample_next_token, load_model
from train import load_wikitext2_modelscope, create_look_ahead_mask

# 确保 Qwen Tokenizer 可用
tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen-tokenizer",
    use_fast=True
)


@torch.no_grad()
def gen_text(model_path, num_samples=5, max_new_tokens=50, device="cuda"):
    """
    加载模型并进行文本生成。
    注意：这里的模型参数需要与训练时保存的模型参数匹配 (这里示例用了一组较小的参数，
          您需要根据实际保存的模型调整，或者在 load_model 时传入正确的 kwargs)。
    """
    print(f"\n Loading model for text generation: {model_path}")

    vocab_size = tokenizer.vocab_size
    
    # **重要提醒**: 请确保这些参数 (num_layers, d_model, d_ff, num_heads) 
    # 与您在 run_ablation 中训练和保存的模型参数一致！
    model_kwargs = dict(
        num_layers=2, # 假设训练时是 6 层
        num_heads=4,
        vocab_size=vocab_size,
        d_ff=512, 
        d_model=128, 
        use_pos=True, # 假设加载的是 Baseline
        use_ffn=True,
        use_norm=True,
        use_attn=True
    )
    
    # 使用 load_model 函数加载
    model = load_model(
        TransformerDecoder, 
        model_path, 
        **model_kwargs
    )
    
    model.to(device)
    model.eval()

    # 1. 加载测试集
    _, _, test_data = load_wikitext2_modelscope()
    print(f" Test set loaded, total {len(test_data)} samples.")

    # 2. 过滤有效样本 (非空样本)
    valid_samples = []
    attempts = 0
    while len(valid_samples) < num_samples and attempts < 200:
        attempts += 1
        sample = random.choice(test_data)
        # 确保样本至少有一个 token
        if sample.numel() >= 1: 
            valid_samples.append(sample)

    if len(valid_samples) == 0:
        print("❌ No valid samples found (all empty).")
        return

    # 3. 自回归生成
    for idx, sample in enumerate(valid_samples):
        # 限制 prompt 长度
        sample = sample[:50] 
        input_ids = sample.unsqueeze(0).to(device)

        print(f"\n=== Sample {idx+1} ===")
        print("Prompt:")
        # 打印原始 prompt
        print(tokenizer.decode(input_ids[0].tolist(), skip_special_tokens=True)) 

        generated = input_ids.clone()

        # 自回归循环
        for _ in range(max_new_tokens):
            if generated.size(1) == 0:
                break

            # 创建 Look-Ahead Mask
            mask = create_look_ahead_mask(generated.size(1)).to(device)
            # 模型预测
            logits = model(generated, tgt_mask=mask) 

            # 采样下一个 token (只使用最后一个位置的 logits)
            next_token = sample_next_token(logits[:, -1, :], temperature=0.8, top_k=50) 
            generated = torch.cat([generated, next_token], dim=1)

        # 打印最终生成的文本
        text = tokenizer.decode(generated[0].tolist(), skip_special_tokens=True)
        print("\nGenerated Text:")
        print(text)


if __name__ == "__main__":
    # 替换为 run_ablation 训练后保存的模型路径
    # 例如: "Baseline_(Full).pth"
    path_to_model = "/data/chengkaiwang/Project/code/transformer/weight/Baseline_(Full).pth" 
    
    # 确保路径存在且模型已训练保存
    if not torch.cuda.is_available():
        device_to_use = "cpu"
    else:
        device_to_use = "cuda"
        
    print("If you see an error about missing file, please run train.py first to generate the model files.")
    
    gen_text(
        model_path=path_to_model, 
        num_samples=3, 
        max_new_tokens=50, 
        device=device_to_use
    )