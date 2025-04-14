import torch
from sae_lens import SAE
from transformer_lens import HookedTransformer
from data.gasp.data6 import goals, suffixes

device = "cuda:4" if torch.cuda.is_available() else "cpu"  

torch.set_grad_enabled(False)
model = HookedTransformer.from_pretrained(
    "meta-llama/Meta-Llama-3-8B",
    device= device
)
model = model.to(device)

sae, cfg_dict, sparsity = SAE.from_pretrained(
    release = "llama-3-8b-it-res-jh",
    sae_id = "blocks.25.hook_resid_post",
    device = device
)
sae = sae.to(device)

sentences_original = [f"{sentence}." for sentence in goals]

sentences_with_prefix = [ ]
for goal, suffix_list in zip(goals, suffixes):
    for suffix in suffix_list:
        sentences_with_prefix.append(f'{goal} {suffix}')


def analyze_latent_separation(sae, sentences_original, sentences_with_prefix, model):
    # 获取两组句子的residual stream
    def get_residual_streams(sentences):
        residuals = []
        for sent in sentences:
            tokens = model.to_tokens(sent)
            _, cache = model.run_with_cache(tokens)
            last_token_residual = cache['resid_post', model.cfg.n_layers-1][0, -1, :]
            residuals.append(last_token_residual)
        return torch.stack(residuals)
    
    # 获取residual streams
    original_residuals = get_residual_streams(sentences_original)
    prefix_residuals = get_residual_streams(sentences_with_prefix)
    
    # 使用SAE编码获取latent activations
    original_latents = sae.encode(original_residuals)
    prefix_latents = sae.encode(prefix_residuals)
    
    # 计算每个latent的激活频率 - 转换布尔值为浮点数
    f_original = torch.mean((original_latents > 0).float(), dim=0)
    f_prefix = torch.mean((prefix_latents > 0).float(), dim=0)
    
    # 计算separation scores
    separation_scores = f_prefix - f_original
    
    # 找出最显著的latents，但限制在有效范围内
    n_latents = sae.W_dec.shape[1]  # 获取实际的latents数量
    print(n_latents)
    top_k = min(500, n_latents)  # 确保不超过可用的latents数量
    top_latents_idx1 = torch.argsort(separation_scores[:n_latents], descending=True)[:top_k]
    top_latents_idx2 = torch.argsort(separation_scores[:n_latents], descending=False)[:top_k]
    
    results1 = {
        'separation_scores': separation_scores[:n_latents],  # 只保留有效范围内的分数
        'top_latents': top_latents_idx1,
        'top_scores': separation_scores[top_latents_idx1],
        'f_original': f_original[top_latents_idx1],
        'f_prefix': f_prefix[top_latents_idx1]
    }

    results2 = {
        'separation_scores': separation_scores[:n_latents],  # 只保留有效范围内的分数
        'top_latents': top_latents_idx2,
        'top_scores': separation_scores[top_latents_idx2],
        'f_original': f_original[top_latents_idx2],
        'f_prefix': f_prefix[top_latents_idx2]
    }
    return results1, results2

# 打印结果并保存到log文件
def print_results(results1, results2):
    # 保存到log文件
    with open("./results/gasp6.txt", "w") as log_file:
        log_file.write("Top 100 latents with highest separation scores (ascending order):\n")
        log_file.write("Format: Latent_idx: Score (f_prefix, f_original)\n")
        log_file.write("-" * 100 + "\n")
        # 将 zip 转换为 list
        zipped_results1 = list(zip(
            results1['top_latents'],
            results1['top_scores'],
            results1['f_prefix'],
            results1['f_original']
        ))

        # 使用列表进行遍历
        for i, (latent_idx, score, f_w, f_wo) in enumerate(zipped_results1[:100]):
                log_file.write(f"{i+1}. Latent {latent_idx:4d}: {score:6.3f} ({f_w:6.3f}, {f_wo:6.3f})\n")

        log_file.write("\nTop 100 latents with lowest separation scores (descending order):\n")
        log_file.write("Format: Latent_idx: Score (f_prefix, f_original)\n")
        log_file.write("-" * 100 + "\n")
        # 将 zip 转换为 list
        zipped_results2 = list(zip(
            results2['top_latents'],
            results2['top_scores'],
            results2['f_prefix'],
            results2['f_original']
        ))
        # 使用列表进行遍历
        for i, (latent_idx, score, f_w, f_wo) in enumerate(zipped_results2[:100]):
                log_file.write(f"{i+1}. Latent {latent_idx:4d}: {score:6.3f} ({f_w:6.3f}, {f_wo:6.3f})\n")

# 调用分析
results1,results2 = analyze_latent_separation(sae, sentences_original, sentences_with_prefix, model)
print_results(results1,results2)