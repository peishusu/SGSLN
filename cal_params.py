import torch
from fvcore.nn import FlopCountAnalysis, parameter_count

from models.FHLCDNet import FHLCDNet  # 根据你模型的实际路径来调整
from models.HSANet import HSANet
from models.LSTMNet import CDXLSTM

# 构造模型并移动到GPU
model = FHLCDNet().cuda()
model.eval()

# 构造输入张量（双输入）
A = torch.randn(1, 3, 256, 256).cuda()
B = torch.randn(1, 3, 256, 256).cuda()
# 这个是针对 CDXLSTM网络的输入参数
# H, W = 256, 256
# A = [
#     torch.randn(1, 40,  H//2,  W//2).cuda(),
#     torch.randn(1, 80,  H//4,  W//4).cuda(),
#     torch.randn(1, 192, H//8,  W//8).cuda(),
#     torch.randn(1, 384, H//16, W//16).cuda(),
# ]
# B = [
#     torch.randn(1, 40,  H//2,  W//2).cuda(),
#     torch.randn(1, 80,  H//4,  W//4).cuda(),
#     torch.randn(1, 192, H//8,  W//8).cuda(),
#     torch.randn(1, 384, H//16, W//16).cuda(),
# ]

# FLOPs 计算（单位：次乘加 = MACs）
flops = FlopCountAnalysis(model, (A, B))
total_flops = flops.total() / 1e9  # 转换为 G (Giga FLOPs)

# 参数量计算（单位：个参数）
params = sum(p.numel() for p in model.parameters()) / 1e6  # 转换为 M (Million Params)

# 打印结果
print("Params (M): {:.2f} M".format(params))
print("FLOPs  (G): {:.2f} G".format(total_flops))
