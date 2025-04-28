import torch
import numpy as np
from torch.autograd import Variable
# from Generator.networks import Generator_net  # 确保你已经导入了正确的生成器模型类
from Generator.networks1 import Generator_net

# 加载生成器模型
def load_generator(model_path, n_chars, seq_len, batch_size, hidden):
    generator = Generator_net(n_chars, seq_len, batch_size, hidden)
    generator.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    generator.eval()
    return generator

# 采样生成序列
def sample_sequences(generator, num_samples, seq_len, device='cpu'):
    noise = Variable(torch.randn(num_samples, 128)).to(device)
    with torch.no_grad():
        generated_sequences = generator(noise)
    return generated_sequences.cpu().numpy()

# 保存生成的序列到文件
def save_sequences(sequences, output_file, charmap):
    with open(output_file, 'w') as f:
        for seq in sequences:
            seq_str = ''.join([charmap[int(char.argmax())] for char in seq])  # 使用charmap映射字符
            f.write(seq_str + '\n')

if __name__ == "__main__":
    model_path = '/Generator/checkpoint/G.pth'  # 生成器模型保存的路径
    output_file = '/Generator/samples/SC_short/G.txt'  # 保存生成序列的文件路径
    num_samples = 20000  # 需要生成的序列数量
    n_chars = 4  # 字符数量
    seq_len = 80  # 序列长度
    batch_size = 64  # 批量大小
    hidden = 512  # 隐藏层大小

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载生成器模型
    generator = load_generator(model_path, n_chars, seq_len, batch_size, hidden).to(device)

    # 采样生成序列
    generated_sequences = sample_sequences(generator, num_samples, seq_len, device)

    # 字符映射表
    charmap = {0: 'A', 1: 'T', 2: 'C', 3: 'G'}

    # 保存生成的序列到文件
    save_sequences(generated_sequences, output_file, charmap)

    print(f"生成的序列已保存到 {output_file}")
