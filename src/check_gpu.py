import torch


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device:", device)
if device.type == 'cuda':
    print("GPU name:", torch.cuda.get_device_name(0))
    print("Allocated memory:", round(torch.cuda.memory_allocated(0)/1024**3, 1), "GB")
    print("Total memory:", round(torch.cuda.get_device_properties(0).total_memory/1024**3, 1), "GB")
