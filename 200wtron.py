import numpy as np
import numba.cuda as cuda

# 定义一个需要加速的函数
@cuda.jit
def my_kernel(io_array):
    pos = cuda.grid(1)
    if pos < io_array.size:
        io_array[pos] *= 2  # 对数组中的每个元素乘以2

# 创建一个输入数组
data = np.ones(1000000)

# 将输入数组从主机内存复制到设备内存
data_device = cuda.to_device(data)

# 定义块和网格大小
threads_per_block = 1024
blocks_per_grid = (data.size + (threads_per_block - 1)) // threads_per_block

# 调用内核函数
my_kernel[blocks_per_grid, threads_per_block](data_device)

# 将输出数组从设备内存复制回主机内存
data_result = data_device.copy_to_host()

# 打印结果
print(data_result)

# 查看GPU占用率
device = cuda.get_current_device()
print(f"GPU占用率: {cuda.profile_start()}")
print(f"GPU占用率: {device.compute_capability}")
print(f"GPU占用率: {device.max_threads_per_block}")
print(f"GPU占用率: {device.max_block_dim_x}")
print(f"GPU占用率: {device.max_grid_dim_x}")
print(f"GPU占用率: {cuda.profile_stop()}")
