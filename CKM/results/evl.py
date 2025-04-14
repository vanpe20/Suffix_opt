from collections import Counter

# 假设文件名为 'neurons.txt'，请根据实际文件名调整
file_name = '/common/home/km1558/szr/CKM/results2/performance/together/S1_num.txt'

# 读取文件并提取所有的神经元编号
with open(file_name, 'r') as file:
    # 假设每一行是一个编号，且编号是数字
    neuron_ids = [line.strip() for line in file.readlines()]

# 使用Counter统计每个神经元编号的出现频度
neuron_counts = Counter(neuron_ids)

# 将结果按频度从高到低排序
sorted_neuron_counts = neuron_counts.most_common()

# 输出排序后的结果
for neuron, count in sorted_neuron_counts:
    print(f"Latent: {neuron}, Frequency: {count}")