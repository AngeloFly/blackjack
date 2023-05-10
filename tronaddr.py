import os
import binascii
import ecdsa
import hashlib
import torch
import numpy as np
import base58

def generate_tron_address():
    # 生成随机私钥
    private_key = torch.tensor(np.frombuffer(os.urandom(32), dtype=np.uint8)).cuda()

    # 获取公钥
    signing_key = ecdsa.SigningKey.from_string(private_key.cpu().numpy(), curve=ecdsa.SECP256k1)
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

    return tron_address.decode('utf-8'),binascii.hexlify(private_key.cpu().numpy()).decode('utf-8')

print(len(generate_tron_address()[0]))
