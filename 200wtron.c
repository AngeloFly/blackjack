#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <openssl/sha.h>
#include <cuda_runtime.h>

#define ADDRESS_LENGTH 34

// 生成随机数
__global__ void generate_random_bytes(unsigned char *bytes, int length) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < length) {
        bytes[idx] = rand() % 256;
    }
}

// 计算SHA256哈希
__global__ void sha256(unsigned char *data, int length, unsigned char *hash) {
    SHA256_CTX sha256_ctx;
    SHA256_Init(&sha256_ctx);
    SHA256_Update(&sha256_ctx, data, length);
    SHA256_Final(hash, &sha256_ctx);
}

// 生成TRON地址
char *generate_tron_address() {
    char *address = (char*) malloc((ADDRESS_LENGTH + 1) * sizeof(char));
    if (address == NULL) {
        return NULL;
    }
    
    // 生成随机数
    unsigned char *random_bytes_host = (unsigned char*) malloc(32 * sizeof(unsigned char));
    unsigned char *random_bytes_device;
    cudaMalloc((void**) &random_bytes_device, 32 * sizeof(unsigned char));
    srand(time(NULL));
    cudaMemcpy(random_bytes_device, random_bytes_host, 32 * sizeof(unsigned char), cudaMemcpyHostToDevice);
    int block_size = 256;
    int grid_size = (32 + block_size - 1) / block_size;
    generate_random_bytes<<<grid_size, block_size>>>(random_bytes_device, 32);
    cudaMemcpy(random_bytes_host, random_bytes_device, 32 * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    
    // 计算SHA256哈希
    unsigned char *hash_host = (unsigned char*) malloc(SHA256_DIGEST_LENGTH * sizeof(unsigned char));
    unsigned char *hash_device;
    cudaMalloc((void**) &hash_device, SHA256_DIGEST_LENGTH * sizeof(unsigned char));
    cudaMemcpy(hash_device, hash_host, SHA256_DIGEST_LENGTH * sizeof(unsigned char), cudaMemcpyHostToDevice);
    sha256<<<1, 1>>>(random_bytes_device, 32, hash_device);
    cudaMemcpy(hash_host, hash_device, SHA256_DIGEST_LENGTH * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    
    // 取哈希前20字节
    unsigned char *prefix = (unsigned char*) malloc(20 * sizeof(unsigned char));
    memcpy(prefix, hash_host, 20);
    
    // 拼接地址
    sprintf(address, "T%c%s", '9', prefix);
    
    free(random_bytes_host);
    cudaFree(random_bytes_device);
    free(hash_host);
    cudaFree(hash_device);
    free(prefix);
    
    return address;
}

int main() {
    char *address = generate_tron_address();
    printf("TRON Address: %s\n", address);
    free(address);
    return 0;
}
