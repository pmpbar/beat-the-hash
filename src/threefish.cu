#include <stdio.h>
#include <stdint.h>
#include <unistd.h>
#include <string.h>
#include <inttypes.h>
#include <assert.h>
#include <stdlib.h>

#define Nw 16
#define Nb (Nw*8)
#define Nr 80

  __device__
inline int permute(uint64_t state[Nw]) {
  uint64_t orig[Nw];
  memcpy(orig, state, sizeof(uint64_t)*Nw);
  state[0] = orig[0];
  state[1] = orig[9];
  state[2] = orig[2];
  state[3] = orig[13];
  state[4] = orig[6];
  state[5] = orig[11];
  state[6] = orig[4];
  state[7] = orig[15];
  state[8] = orig[10];
  state[9] = orig[7];
  state[10] = orig[12];
  state[11] = orig[3];
  state[12] = orig[14];
  state[13] = orig[5];
  state[14] = orig[8];
  state[15] = orig[1];

  return 0;
}
__device__
const int Rdj[8][Nw/2] = {
  {24, 13,  8, 47,  8, 17, 22, 37},
  {38, 19, 10, 55, 49, 18, 23, 52},
  {33,  4, 51, 13, 34, 41, 59, 17},
  { 5, 20, 48, 41, 47, 28, 16, 25},
  {41,  9, 37, 31, 12, 47, 44, 30},
  {16, 34, 56, 51,  4, 53, 42, 41},
  {31, 44, 47, 46, 19, 42, 44, 25},
  { 9, 48, 35, 52, 23, 31, 37, 20}
};


#ifdef DEBUG
#define debugPrintf(...) printf(__VA_ARGS__)
#define debugPrintMsg(var) \
  do {\
    for(int _i = 0; _i < Nw; _i++) {\
      printf("%016" PRIx64 " ", var[_i]);\
    }\
    printf("\n");\
  } while (0)
#else
#define debugPrintf(...)
#define debugPrintMsg(...)
#endif

#define printMsg(var) \
  do {\
    for(int _i = 0; _i < Nw; _i++) {\
      printf("%016" PRIx64 " ", (var)[_i]);\
    }\
    printf("\n");\
  } while (0)


  __device__
uint64_t rotl(uint64_t in, int numBits) {
  uint64_t out = in << numBits;
  uint64_t tmp = in >> (64 - numBits);
  out |= tmp;
  return out;
}

  __device__
int mix(uint64_t* y0, uint64_t* y1, uint64_t x0, uint64_t x1, int d, int j) {
  *y0 = x0+x1;
  *y1 = rotl(x1, Rdj[d%8][j]) ^ *y0;

  return 0;
}

  __device__
int threefish(uint64_t key[Nw], uint64_t tweak[2], uint64_t plaintext[Nw]) {
  uint64_t subkeyTable[Nr/4+1][Nw];

  // t is one word longer than tweak, with an extra tweak value
  uint64_t t[3];
  memcpy(t, tweak, sizeof(uint64_t)*2);
  t[2] = tweak[0] ^ tweak[1];

  // k is one word longer than key, with an extra key word
  uint64_t k[Nw+1];
  memcpy(k, key, sizeof(uint64_t)*Nw);
  // k[Nw] = 6148914691236517205; // 2^64 / 3
  k[Nw] = 0x1BD11BDAA9FC1A22;
  // printf("%016" PRIx64 "\n", k[Nw]);
  for(int i = 0; i < Nw; i++) {
    k[Nw] ^= key[i];
  }

  // generate subkey table
  debugPrintf("Subkeys:\n");
  for(int s = 0; s < Nr/4+1; s++) {
    int i;
    for(i = 0; i < Nw-3; i++) {
      subkeyTable[s][i] = k[(s+i) % (Nw+1)];
    }
    subkeyTable[s][i] = k[(s+i) % (Nw+1)] + t[s%3];
    i++;
    subkeyTable[s][i] = k[(s+i) % (Nw+1)] + t[(s+1)%3];
    i++;
    subkeyTable[s][i] = k[(s+i) % (Nw+1)] + s;

    debugPrintMsg(subkeyTable[s]);
  }

  for(int d = 0; d < Nr; d++) {
    if (d % 4 == 0) { // Add subkey
      for(int i = 0; i < Nw; i++)
        plaintext[i] += subkeyTable[d/4][i];
    }

    // Mix
    for(int j = 0; j < Nw/2; j++) {
      mix(&plaintext[2*j], &plaintext[2*j+1], plaintext[2*j], plaintext[2*j+1], d, j);
    }

    // permutation
    permute(plaintext);
  }

  for(int i = 0; i < Nw; i++) {
    plaintext[i] += subkeyTable[Nr/4][i];
  }

  return 0;
}


// Unique Block Iteration (UBI)
  __device__
int ubi(uint64_t G[Nw], uint8_t* M, int Mlen, uint64_t Ts[2]) {
  uint64_t block[Nw];
  uint64_t pt[Nw];
  int Moffset = 0;

  uint64_t tweak[2];

  int numBlocks = Mlen/Nb + (Mlen%Nb != 0);

  for(int i = 0; i < numBlocks; i++) {
    int bytesThisMsg = 0;

    // Make current block from input M.
    if(Moffset + Nb < Mlen) {
      memcpy(block, M+Moffset, Nb);
      bytesThisMsg = Nb;
    } else { // if M isn't an integral number of blocks
      memset(block, 0, Nb);
      bytesThisMsg = Mlen - Moffset;
      memcpy(block, M+Moffset, bytesThisMsg);
    }

    // My threefish implementation overwrites the input block with its output.
    // backing up the input block in pt.
    memcpy(pt, block, Nb);

    // Handle tweak 128bit math.
    tweak[1] = Ts[1];
    tweak[0] = Ts[0] + Moffset + bytesThisMsg;

    // // Check for carry.
    // if( (Ts[0] > Moffset + bytesThisMsg && UINT64_MAX - Ts[0] < Moffset + bytesThisMsg) ||
    // 	(Ts[0] < Moffset + bytesThisMsg && UINT64_MAX - Moffset + bytesThisMsg < Ts[0]))
    // {
    // 	tweak[1]++;
    // }

    // check first and last blocks
    if(i == 0) {
      tweak[1] |= 0x4000000000000000;
    }

    if(i == numBlocks - 1) {
      tweak[1] |= 0x8000000000000000;
    }

    // actually run the encryption
    threefish(G, tweak, block);

    // xor the output from threefish with the plaintext input.
    for(int i = 0; i < Nw; i++) {
      block[i] ^= pt[i];
    }

    // set starting value for next iteration
    memcpy(G, block, Nb);

    Moffset += Nb;
  }

  return 0;
}

__device__
int skeinhash1024x1024(uint8_t* bytes, int len, uint64_t out[Nw]) {
  uint64_t Kprime[Nw] = {0};

  uint8_t configStr[32] = "SHA3";
  *((uint16_t*)(configStr+4)) = (uint16_t)1;
  *((uint16_t*)(configStr+6)) = (uint16_t)0;
  *((uint64_t*)(configStr+8)) = (uint64_t)1024;
  configStr[16] = 0;
  configStr[17] = 0;
  configStr[18] = 0;
  for(int i = 19; i < 32; i++) {
    configStr[i] = 0;
  }

  uint64_t typeCfg[2] = {0};
  uint64_t typeMsg[2] = {0};
  uint64_t typeOut[2] = {0};
  typeCfg[1] |= ((uint64_t)4)<<56;
  typeMsg[1] |= ((uint64_t)48)<<56;
  typeOut[1] |= ((uint64_t)63)<<56;

  ubi(Kprime, configStr, 32, typeCfg);
  uint64_t* G0 = Kprime;

  ubi(G0, bytes, len, typeMsg);
  uint64_t* G1 = G0;

  uint64_t zero = 0;
  ubi(G1, (uint8_t*)&zero, 8, typeOut);
  uint64_t* H = G1;

  memcpy(out, H, Nb);

  return 0;
}
