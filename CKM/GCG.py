import nanogcg
import torch
import csv

from nanogcg import GCGConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "meta-llama/Meta-Llama-3-8B"
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda:1")
tokenizer = AutoTokenizer.from_pretrained(model_id)
import csv

# 打开CSV文件
with open('/common/home/km1558/szr/CKM/enhanced/llm-attacks-main/data/advbench/harmful_behaviors.csv', mode='r') as file:
    # 读取CSV内容
    csv_reader = csv.reader(file)
    
    # 跳过标题行
    next(csv_reader)

    # 每次读取一行并打印
    for row in csv_reader:
        goal, target = row  
        print(f"Goal: {goal}")
        print(f"Target: {target}")
        print("-" * 40)  # 用于分隔不同的输出

        config = GCGConfig(
            num_steps=500,
            search_width=64,
            topk=64,
            seed=42,
            verbosity="WARNING"
        )

        result = nanogcg.run(model, tokenizer, goal, target, config)
        # 输出到txt文件
        output_file = "/common/home/km1558/szr/CKM/result_output.txt"
        with open(output_file, 'a') as f:
            f.write(f"{goal}, {result.best_string}\n")  # 用逗号分隔 Message 和 Best String

        print(f"Results have been written to {output_file}")
print("DONE!")