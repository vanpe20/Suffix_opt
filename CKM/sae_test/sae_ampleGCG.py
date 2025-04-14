import torch
from sae_lens import SAE
from transformer_lens import HookedTransformer
import json

# 目标文件路径
file_path = "/common/home/km1558/szr/CKM/AmpleGCG_Generated_Suffixes/AdvBench/llama2-chat-7b/test_prompter_test_checkpoint_decode_group_beam_search_group_100_numreturn_100_promptway_prompter_q_p_model_plus/llama2-7b-chat_max_new_tokens_100/targetlm.jsonl" 

# 用于存储 q 和 p
original = []
add_suffix = []

# 读取 JSONL 文件并提取 q 和 p 字段
with open(file_path, 'r') as file:
    for line in file:
        data = json.loads(line)  # 将每行内容转换为字典
        if 'q' in data and 'p' in data:
            original.append(data['q'])  # 提取 q 字段并添加到 original 列表
            add_suffix.append(data['p'])  # 提取 p 字段并添加到 add_suffix 列表

device = "cuda:2" if torch.cuda.is_available() else "cpu"  

torch.set_grad_enabled(False)
model = HookedTransformer.from_pretrained(
    "google/gemma-2b-it",
    device= device
)
model = model.to(device)

sae, cfg_dict, sparsity = SAE.from_pretrained(
    release = "gemma-2b-it-res-jb",
    sae_id = "blocks.12.hook_resid_post" ,
    device = device
)
sae = sae.to(device)

# 打印查看结果
original = original[::100]
# original = original[50:]
add_suffix = add_suffix[::100]
# add_suffix = add_suffix[50:]
# print("original:", original)  # 打印前5个 q 值
# print("add_suffix:", add_suffix)  # 打印前5个 p 值

# 生成两组句子
sentences_original = original
sentences_with_suffix = [f'{p} {g}' for p, g in zip(original, add_suffix)]


def analyze_latent_separation(sae, sentences_original, sentences_with_suffix, model):
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
    suffix_residuals = get_residual_streams(sentences_with_suffix)
    
    # 使用SAE编码获取latent activations
    original_latents = sae.encode(original_residuals)
    suffix_latents = sae.encode(suffix_residuals)
    
    # 计算每个latent的激活频率 - 转换布尔值为浮点数
    f_original = torch.mean((original_latents > 0).float(), dim=0)
    f_suffix = torch.mean((suffix_latents > 0).float(), dim=0)
    
    # 计算separation scores
    separation_scores = f_suffix - f_original
    
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
        'f_suffix': f_suffix[top_latents_idx1]
    }

    results2 = {
        'separation_scores': separation_scores[:n_latents],  # 只保留有效范围内的分数
        'top_latents': top_latents_idx2,
        'top_scores': separation_scores[top_latents_idx2],
        'f_original': f_original[top_latents_idx2],
        'f_suffix': f_suffix[top_latents_idx2]
    }
    return results1, results2

# 打印结果并保存到log文件
def print_results(results1, results2):
    # 保存到log文件
    with open("./results2/jailbreak/AmpleGCG-plus.txt", "a") as log_file:
        log_file.write("Top 500 latents with highest separation scores (ascending order):\n")
        log_file.write("Format: Latent_idx: Score (f_suffix, f_original)\n")
        log_file.write("-" * 500 + "\n")
        # 将 zip 转换为 list
        zipped_results1 = list(zip(
            results1['top_latents'],
            results1['top_scores'],
            results1['f_suffix'],
            results1['f_original']
        ))

        # 使用列表进行遍历
        for i, (latent_idx, score, f_w, f_wo) in enumerate(zipped_results1[:500]):
                log_file.write(f"{i+1}. Latent {latent_idx:4d}: {score:6.3f} ({f_w:6.3f}, {f_wo:6.3f})\n")

        log_file.write("\nTop 500 latents with lowest separation scores (descending order):\n")
        log_file.write("Format: Latent_idx: Score (f_suffix, f_original)\n")
        log_file.write("-" * 500 + "\n")
        # 将 zip 转换为 list
        zipped_results2 = list(zip(
            results2['top_latents'],
            results2['top_scores'],
            results2['f_suffix'],
            results2['f_original']
        ))
        # 使用列表进行遍历
        for i, (latent_idx, score, f_w, f_wo) in enumerate(zipped_results2[:500]):
                log_file.write(f"{i+1}. Latent {latent_idx:4d}: {score:6.3f} ({f_w:6.3f}, {f_wo:6.3f})\n")

# 调用分析
results1,results2 = analyze_latent_separation(sae, sentences_original, sentences_with_suffix, model)
print_results(results1,results2)