
import numba
from numba import cuda

import torch


print("Cuda King")

'''

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using device:", device)
    print("Device name:", torch.cuda.get_device_name(0))
    print("CUDA cores estimate not directly available in PyTorch")
else:
    print("CUDA not available")






from numba import cuda

# Dictionary mapping compute capability to cores per SM
cc_cores_per_SM_dict = {
    (2, 0): 32, (2, 1): 48,
    (3, 0): 192, (3, 5): 192, (3, 7): 192,
    (5, 0): 128, (5, 2): 128,
    (6, 0): 64,  (6, 1): 128,
    (7, 0): 64,  (7, 5): 64,
    (8, 0): 64,  (8, 6): 128, (8, 9): 128,
    (9, 0): 128
}

# Get current device
device = cuda.get_current_device()
cc = device.compute_capability
sms = device.MULTIPROCESSOR_COUNT
cores_per_sm = cc_cores_per_SM_dict.get(cc, 0)
total_cores = sms * cores_per_sm

print(f"GPU Name: {device.name}")
print(f"Compute Capability: {cc}")
print(f"Streaming Multiprocessors: {sms}")
print(f"Estimated CUDA Cores: {total_cores}")



'''
# CUDA kernel
@cuda.jit
def vector_add(a, b, result):
    idx = cuda.grid(1)
    if idx < a.size:
        result[idx] = a[idx] + b[idx]

# Host code
n = 1000000
a = np.ones(n, dtype=np.float32)
b = np.ones(n, dtype=np.float32)
result = np.zeros(n, dtype=np.float32)

# Transfer data to the device
a_device = cuda.to_device(a)
b_device = cuda.to_device(b)
result_device = cuda.to_device(result)

# Configure the blocks
threads_per_block = 256
blocks_per_grid = (n + threads_per_block - 1) // threads_per_block

# Launch the kernel
vector_add[blocks_per_grid, threads_per_block](a_device, b_device, result_device)

# Copy the result back to the host
result_device.copy_to_host(result)

print(result[:10])  # Print first 10 results

