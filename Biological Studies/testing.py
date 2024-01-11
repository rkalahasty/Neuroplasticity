import torch

tensor1 = torch.randn(4, 4)
tensor2 = torch.randn(4, 1)
print(torch.matmul(tensor1, tensor2).size())