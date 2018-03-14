#include <iostream>
#include <math.h>
#include "threefish.cu"

__device__
void next(uint8_t *n) {
  for (int i = 0;; i++) {
    n[i]++;
    if (n[i] <= 'z') {
      break;
    } else {
      n[i] = 'a';
    }
  }
}

__global__
void search(int* minValue) {
  const uint64_t target[16] = {0x8082a05f5fa94d5b,0xc818f444df7998fc,0x7d75b724a42bf1f9,0x4f4c0daefbbd2be0,0x04fec50cc81793df,0x97f26c46739042c6,0xf6d2dd9959c2b806,0x877b97cc75440d54,0x8f9bf123e07b75f4,0x88b7862872d73540,0xf99ca716e96d8269,0x247d34d49cc74cc9,0x73a590233eaa67b5,0x4066675e8aa473a3,0xe7c5e19701c79cc7,0xb65818ca53fb02f9};

  const uint8_t alp[27] = "abcdefghijklmnopqrstuvwxyz";
  uint8_t cur[33];
  for (int i = 0; i <= 32; i++) {
    cur[i] = alp[threadIdx.x];
  }
  while(*minValue > 200) {
    uint64_t hash[16];
    skeinhash1024x1024(cur, 32, hash);

    int c = 0;
    for (int j = 0; j < Nw; j++) {
      uint64_t tmp = target[j] ^ hash[j];
      for (int k = 0; k < 64; k++) {
        if(tmp & 1) {
          c++;
        }
        tmp >>= 1;
      }
    }
    if(c < *minValue) {
      *minValue = c;
      printf("(%d) %s\n", c, cur);
    }
    next(cur);
  }
}

int main(void) {
  int* min;

  // Allocate Unified Memory – accessible from CPU or GPU
  cudaMallocManaged(&min, sizeof(int));

  *min = 420;

  printf("Starting search...\n");
  search <<<1, 27>>> (min);

  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();

  // Free memory
  cudaFree(min);

  return 0;
}
