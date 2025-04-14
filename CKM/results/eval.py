extracted_content = []

# 读取文件并处理每行
with open('/common/home/km1558/szr/CKM/results2/performance/together/S1_num.txt', 'r') as file:
    for line in file:
        # 分割字符串，取冒号前的部分
        if ':' in line:
            before_colon = line.split(',')[0]
            extracted_content.append(before_colon.strip())  # 去除前后的空格

# 保存提取内容到新文件
with open('/common/home/km1558/szr/CKM/results2/performance/together/S1_num.txt', 'w') as output_file:
    for content in extracted_content:
        output_file.write(content + '\n')

import re

# 目标文件路径
file_path = '/common/home/km1558/szr/CKM/results2/performance/together/S1_num.txt'
output_file_path = '/common/home/km1558/szr/CKM/results2/performance/together/S1_num.txt'

# 用于存储 Latent 后的数字
latent_numbers = []

# 读取文件并提取 Latent 后的数字
with open(file_path, 'r') as file:
    for line in file:
        # 查找 "Latent" 后的数字
        match = re.search(r'Latent\s+(\d+)', line)
        if match:
            # 提取数字并添加到列表
            latent_numbers.append(match.group(1))

# 将提取的数字保存到新的文件
with open(output_file_path, 'w') as output_file:
    output_file.write("\n".join(latent_numbers))

print(f"提取的数字已保存到 {output_file_path}")