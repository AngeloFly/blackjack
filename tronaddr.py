import base58
import binascii
import ecdsa
import hashlib
import cupy as cp
from numba import cuda

@cuda.jit
def generate_tron_address_kernel(h_tron_address, h_private_key):
    # 获取线程索引
    thread_idx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x

    # 生成随机私钥
    private_key = cp.random.bytes(32)

    # 获取公钥
    signing_key = ecdsa.SigningKey.from_string(private_key, curve=ecdsa.SECP256k1)
    verifying_key = signing_key.get_verifying_key()

    # 添加字节前缀并哈希
    public_key_bytes = (b"\x04" + verifying_key.to_string())
    sha256_hash = hashlib.sha256(public_key_bytes).digest()

    # 计算ripemd160哈希
    ripemd160_hasher = hashlib.new("ripemd160")
    ripemd160_hasher.update(sha256_hash)
    ripemd160_hash = ripemd160_hasher.digest()

    # 添加字节前缀
    prefix_ripemd160_hash = (b"\x41" + ripemd160_hash)

    # 计算两次sha256哈希
    sha256_hasher = hashlib.sha256(prefix_ripemd160_hash)
    sha256_hash = sha256_hasher.digest()

    sha256_hasher2 = hashlib.sha256(sha256_hash)
    sha256_hash2 = sha256_hasher2.digest()

    # 添加校验和并base58编码
    address = prefix_ripemd160_hash + sha256_hash2[:4]
    tron_address = base58.b58encode(address)

    # 将结果存储在cupy数组中
    h_tron_address[thread_idx] = tron_address.decode('utf-8')
    h_private_key[thread_idx] = binascii.hexlify(private_key).decode('utf-8')

# 创建cupy数组来存储结果
threads_per_block = 1024
blocks_per_grid = 30
h_tron_address = cp.empty(threads_per_block * blocks_per_grid, dtype=cp.dtype('U34'))
h_private_key = cp.empty(threads_per_block * blocks_per_grid, dtype=cp.dtype('S32'))

# 在GPU上运行生成Tron地址的函数
generate_tron_address_kernel[blocks_per_grid, threads_per_block](h_tron_address, h_private_key)

# 找到第一个非空地址和私钥
for i in range(len(h_tron_address)):
    if h_tron_address[i] != '':
        print(len(h_tron_address[i]))
        break
