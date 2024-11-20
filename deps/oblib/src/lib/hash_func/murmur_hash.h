// (C) 2010-2016 Alibaba Group Holding Limited.
//
// Authors:
// Normalizer:

#ifndef OCEANBASE_LIB_HASH_MURMURHASH_
#define OCEANBASE_LIB_HASH_MURMURHASH_

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <string>

namespace oceanbase
{
namespace common
{

inline uint64_t murmurhash64A(const void *key, int32_t len, uint64_t seed)
{
  const uint64_t multiply = 0xc6a4a7935bd1e995;
  const int rotate = 47;

  uint64_t ret = seed ^ (len * multiply);

  const uint64_t *data = (const uint64_t *)key;
  const uint64_t *end = data + (len / 8);
  // for (; len >= 8; len -= 8) {
  //   uint64_t val = *data;
  //   val *= multiply;
  //   val ^= val >> rotate;
  //   val *= multiply;
  //   ret ^= val;
  //   ret *= multiply;
  //   ++data;
  // }
  // 循环展开
  // 在处理对齐数据（len >= 8）时，可以对循环进行展开，以减少分支和指针运算。
  // 使用64更慢,GPT给出的理由如下:
  // 处理 8 个 uint64_t 时，需要 8 个变量（val1 到 val8）占用 CPU 寄存器，同时还需要寄存器保存 ret、multiply 和 rotate 等额外信息。如果寄存器数量不足，CPU 会使用堆栈或临时内存来存储这些变量，导致额外的 寄存器溢出开销（register spill overhead）。
  // 现代 CPU 通常有 16 个通用寄存器（64 位 x86 架构的 AVX2 指令集）。如果使用了较多的寄存器，编译器可能需要将某些变量存入内存，并在需要时加载回来，从而引入额外的内存访问。
  for (; len >= 32; len -= 32) {
    uint64_t val1 = data[0];
    uint64_t val2 = data[1];
    uint64_t val3 = data[2];
    uint64_t val4 = data[3];

    val1 *= multiply; val1 ^= val1 >> rotate; val1 *= multiply;
    val2 *= multiply; val2 ^= val2 >> rotate; val2 *= multiply;
    val3 *= multiply; val3 ^= val3 >> rotate; val3 *= multiply;
    val4 *= multiply; val4 ^= val4 >> rotate; val4 *= multiply;

    ret ^= val1; ret *= multiply;
    ret ^= val2; ret *= multiply;
    ret ^= val3; ret *= multiply;
    ret ^= val4; ret *= multiply;

    data += 4;
  }

  const unsigned char *data2 = (const unsigned char *)data;
  while (len > 0) {
    --len;
    ret ^= uint64_t(data2[len]) << (len * 8);
    if (0 == len) {
      ret *= multiply;
    }
  }
  ret ^= ret >> rotate;
  ret *= multiply;
  ret ^= ret >> rotate;

  return ret;
}

// The MurmurHash 2 from Austin Appleby, faster and better mixed (but weaker
// crypto-wise with one pair of obvious differential) than both Lookup3 and
// SuperFastHash. Not-endian neutral for speed.
uint32_t murmurhash2(const void *data, int32_t len, uint32_t hash);

// MurmurHash2, 64-bit versions, by Austin Appleby
// The same caveats as 32-bit MurmurHash2 apply here - beware of alignment
// and endian-ness issues if used across multiple platforms.
// 64-bit hash for 64-bit platforms
uint64_t murmurhash64A(const void *key, int32_t len, uint64_t seed);

//public function, please only use this one
inline uint64_t murmurhash(const void *data, int32_t len, uint64_t hash)
{
  return murmurhash64A(data, len, hash);
}

inline uint64_t appname_hash(const void *data, int32_t len, uint64_t hash)
{
  return murmurhash64A(data, len, hash);
}

uint32_t fnv_hash2(const void *data, int32_t len, uint32_t hash);
}//namespace common
}//namespace oceanbase
#endif // OCEANBASE_LIB_HASH_MURMURHASH_
