

// Copyright 2024-present the vsag project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <x86intrin.h>
#include <vector>
#include <iostream>

namespace vsag {

#define PORTABLE_ALIGN32 __attribute__((aligned(32)))
#define PORTABLE_ALIGN64 __attribute__((aligned(64)))

float
L2SqrSIMD16ExtAVX512(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
    float* pVect1 = (float*)pVect1v;
    float* pVect2 = (float*)pVect2v;
    size_t qty = *((size_t*)qty_ptr);
    float PORTABLE_ALIGN64 TmpRes[16];
    size_t qty16 = qty >> 4;

    const float* pEnd1 = pVect1 + (qty16 << 4);

    __m512 diff, v1, v2;
    __m512 sum = _mm512_set1_ps(0);

    while (pVect1 < pEnd1) {
        v1 = _mm512_loadu_ps(pVect1);
        pVect1 += 16;
        v2 = _mm512_loadu_ps(pVect2);
        pVect2 += 16;
        diff = _mm512_sub_ps(v1, v2);
        sum = _mm512_fmadd_ps(diff, diff, sum);
        // sum = _mm512_add_ps(sum, _mm512_mul_ps(diff, diff));
    }

    _mm512_store_ps(TmpRes, sum);
    float res = TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3] + TmpRes[4] + TmpRes[5] + TmpRes[6] +
                TmpRes[7] + TmpRes[8] + TmpRes[9] + TmpRes[10] + TmpRes[11] + TmpRes[12] +
                TmpRes[13] + TmpRes[14] + TmpRes[15];

    return (res);
}

float
InnerProductSIMD16ExtAVX512(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
    float PORTABLE_ALIGN64 TmpRes[16];
    float* pVect1 = (float*)pVect1v;
    float* pVect2 = (float*)pVect2v;
    size_t qty = *((size_t*)qty_ptr);

    size_t qty16 = qty / 16;

    const float* pEnd1 = pVect1 + 16 * qty16;

    __m512 sum512 = _mm512_set1_ps(0);

    while (pVect1 < pEnd1) {
        //_mm_prefetch((char*)(pVect2 + 16), _MM_HINT_T0);

        __m512 v1 = _mm512_loadu_ps(pVect1);
        pVect1 += 16;
        __m512 v2 = _mm512_loadu_ps(pVect2);
        pVect2 += 16;
        sum512 = _mm512_add_ps(sum512, _mm512_mul_ps(v1, v2));
    }

    _mm512_store_ps(TmpRes, sum512);
    float sum = TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3] + TmpRes[4] + TmpRes[5] + TmpRes[6] +
                TmpRes[7] + TmpRes[8] + TmpRes[9] + TmpRes[10] + TmpRes[11] + TmpRes[12] +
                TmpRes[13] + TmpRes[14] + TmpRes[15];

    return sum;
}

}  // namespace vsag

// 通过vid和id的关系 直接将vid转成id
void VidToIDAVX512(const void* vec, int64_t value_to_subtract, const size_t& n) {
    size_t * result = (size_t*)vec; 
    size_t i = 0;

    // 使用 AVX512 处理 64 位整数（每次处理 8 个元素）
    __m512i sub_value = _mm512_set1_epi64(value_to_subtract);

    for (; i <= n - 8; i += 8) {
        // 加载 8 个 64 位整数到 AVX512 寄存器中
        __m512i data = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(&result[i]));

        // 每个元素减去常数值
        __m512i result_data = _mm512_sub_epi64(data, sub_value);

        // 存储结果回到内存
        _mm512_storeu_si512(reinterpret_cast<__m512i*>(&result[i]), result_data);
    }

    // 处理剩余的元素
    for (; i < n; ++i) {
        result[i] -= value_to_subtract;
    }
}


// std::vector<int8_t> floatToint8SIMD(const float* input, const size_t& size) {
//     // size_t size = input.size();
//     std::vector<int8_t> output(size);

//     for (size_t i = 0; i < size; i += 16) { // 每次处理16个元素
//         __m512 values = _mm512_loadu_ps(&input[i]);            // 加载16个float
//         __m512i integers = _mm512_cvtps_epi32(values);         // 转为int32
//         __m128i packed = _mm512_cvtusepi32_epi8(integers);     // 压缩到uint8_t
//         _mm_storeu_si128((__m128i*)&output[i], packed);        // 存储结果
//     }
//     return output;
// }