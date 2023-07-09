import torch
from torchsummary import summary

def get_lr(optimizer):
   for params_group in optimizer.params_groups:
      return param_group['lr']

def get_device():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

def print_summary(model, input_size):
    summary(model, input_size)
