import torch
from sae_lens import SAE
from transformer_lens import HookedTransformer
# 读取文件并提取每行前半部分
file_path1 = "/common/home/km1558/szr/auto_prompt/attackprompt/data_1/cls/trec/dev.txt"  # 请替换为您的文件路径

# 读取文件并提取每行前半部分
file_path2 = "/common/home/km1558/szr/auto_prompt/attackprompt/data_1/cls/trec/test/attack_res_S1.txt"  # 请替换为您的文件路径

# 读取文件并提取每行前半部分
# file_path2 = "/common/home/km1558/szr/auto_prompt/attackprompt/data_1/cls/agnews/test/attack_res_S1.txt"  # 请替换为您的文件路径

device = "cuda:4" if torch.cuda.is_available() else "cpu"  

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

original = [ ]
# 读取文件内容并处理每一行
with open(file_path1, 'r') as file:
    for line in file:
        # 分割每行内容并提取前半部分
        split_line = line.split("\t")  # 按制表符分割
        if split_line:  # 确保该行不是空行
            original.append(split_line[0])  # 提取前半部分

add_suffix = [ ]
# 读取文件内容并处理每一行
with open(file_path2, 'r') as file:
    for line in file:
        # 分割每行内容并提取前半部分
        split_line = line.split("\t")  # 按制表符分割
        if split_line:  # 确保该行不是空行
            add_suffix.append(split_line[0])  # 提取前半部分

# 生成两组句子
sentences_original = [f"Please perform Question Classification task. Given the question, assign a label from ['Description', 'Entity', 'Expression', 'Human', 'Location', 'Number']. Return label only without any other text. {q} " for q in original]
sentences_with_suffix = [f"Please perform Question Classification task. Given the question, assign a label from ['Description', 'Entity', 'Expression', 'Human', 'Location', 'Number']. Return label only without any other text. {q} " for q in add_suffix]

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
    with open("./results2/performance/pure_data/trec_S1.txt", "a") as log_file:
        log_file.write("Top 100 latents with highest separation scores (ascending order):\n")
        log_file.write("Format: Latent_idx: Score (f_suffix, f_original)\n")
        log_file.write("-" * 100 + "\n")
        # 将 zip 转换为 list
        zipped_results1 = list(zip(
            results1['top_latents'],
            results1['top_scores'],
            results1['f_suffix'],
            results1['f_original']
        ))

        # 使用列表进行遍历
        for i, (latent_idx, score, f_w, f_wo) in enumerate(zipped_results1[:100]):
                log_file.write(f"{i+1}. Latent {latent_idx:4d}: {score:6.3f} ({f_w:6.3f}, {f_wo:6.3f})\n")

        log_file.write("\nTop 100 latents with lowest separation scores (descending order):\n")
        log_file.write("Format: Latent_idx: Score (f_suffix, f_original)\n")
        log_file.write("-" * 100 + "\n")
        # 将 zip 转换为 list
        zipped_results2 = list(zip(
            results2['top_latents'],
            results2['top_scores'],
            results2['f_suffix'],
            results2['f_original']
        ))
        # 使用列表进行遍历
        for i, (latent_idx, score, f_w, f_wo) in enumerate(zipped_results2[:100]):
                log_file.write(f"{i+1}. Latent {latent_idx:4d}: {score:6.3f} ({f_w:6.3f}, {f_wo:6.3f})\n")

# 调用分析
results1,results2 = analyze_latent_separation(sae, sentences_original, sentences_with_suffix, model)
print_results(results1,results2)