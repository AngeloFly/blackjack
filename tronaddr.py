import base58
import os
import binascii
import ecdsa
import hashlib
import time
import numba as nb
from numba import cuda
import numpy as np

from concurrent.futures import ThreadPoolExecutor
logging.basicConfig(filename='tron.log', level=logging.INFO)
logging.basicConfig(format='%(asctime)s %(message)s')

@cuda.jit(device=True)
def sha256(data):
    # CUDA设备函数，计算sha256哈希值
    hash = hashlib.sha256(data).digest()
    return hash

@cuda.jit(device=True)
def ripemd160(data):
    # CUDA设备函数，计算ripemd160哈希值
    hash = hashlib.new("ripemd160")
    hash.update(data)
    return hash.digest()

@cuda.jit(device=True)
def prefix(data, prefix):
    # CUDA设备函数，添加字节前缀
    return prefix + data

@cuda.jit(device=True)
def encode_base58(data):
    # CUDA设备函数，base58编码
    return base58.b58encode(data)

@cuda.jit(device=True)
def generate_tron_address_private_key():
    # CUDA设备函数，生成随机私钥
    return os.urandom(32)

@cuda.jit(device=True)
def get_public_key_bytes(private_key):
    # CUDA设备函数，获取公钥
    signing_key = ecdsa.SigningKey.from_string(private_key, curve=ecdsa.SECP256k1)
    verifying_key = signing_key.get_verifying_key()
    return b"\x04" + verifying_key.to_string()

@cuda.jit(device=True)
def generate_tron_address_kernel(d_tron_address):
    # CUDA设备函数，生成Tron地址
    i = cuda.grid(1)

    if i < d_tron_address.shape[0]:
        # 生成随机私钥
        private_key = generate_tron_address_private_key()

        # 获取公钥
        public_key_bytes = get_public_key_bytes(private_key)

        # 添加字节前缀并哈希
        sha256_hash = sha256(public_key_bytes)

        # 计算ripemd160哈希
        ripemd160_hash = ripemd160(sha256_hash)

        # 添加字节前缀
        prefix_ripemd160_hash = prefix(ripemd160_hash, b"\x41")

        # 计算两次sha256哈希
        sha256_hash = sha256(prefix_ripemd160_hash)
        sha256_hash2 = sha256(sha256_hash)

        # 添加校验和并base58编码
        address = prefix_ripemd160_hash + sha256_hash2[:4]
        tron_address = encode_base58(address)

        # 将地址存储在设备数组中
        d_tron_address[i] = tron_address

def generate_tron_address():
    # 使用Numba和CUDA进行GPU加速的Tron地址生成函数
    threads_per_block = 1024
    blocks_per_grid = 30
    d_tron_address = cuda.device_array(threads_per_block * blocks_per_grid, dtype=np.str_)
    generate_tron_address_kernel[blocks_per_grid, threads_per_block](d_tron_address)

    # 将结果复制回主机内存
    h_tron_address = d_tron_address.copy_to_host()

    # 返回第一个非空地址
    for address in h_tron_address:
        if address != '':
            return address, binascii.hexlify(private_key).decode('utf-8')

    return None

print(len(generate_tron_address()[0]))
