module;
#include <immintrin.h>

module foye.algorithm;
import foye.foye_core;
import std;

namespace fy
{
    template<BasicArithmetic Element_t>
    extended_t<Element_t> dot_product(const Element_t* a, const Element_t* b, usize count) noexcept
    {
        std::unreachable();
    }

    template<> f64 dot_product(const f16* a, const f16* b, usize count) noexcept
    {
        constexpr usize CACHE_LINE_BYTES = 64;
        constexpr usize F16_SIMD_LANES = 8;
        constexpr usize BATCHES_PER_LOOP = 4;
        constexpr usize STRIDE_ELEMENTS = BATCHES_PER_LOOP * F16_SIMD_LANES;
        constexpr usize PREFETCH_OFFSET_BYTES = STRIDE_ELEMENTS * sizeof(f16);
        constexpr usize BLOCK_32_ELEMENTS = 32;
        constexpr usize BLOCK_16_ELEMENTS = 16;
        constexpr usize BLOCK_8_ELEMENTS = 8;

        f64 result = 0.0;
        usize i = 0;

        while (i < count &&
            (reinterpret_cast<uintptr_t>(&a[i]) % CACHE_LINE_BYTES ||
                reinterpret_cast<uintptr_t>(&b[i]) % CACHE_LINE_BYTES))
        {
            result += static_cast<f64>(static_cast<f32>(a[i])) * static_cast<f32>(b[i]);
            ++i;
        }

        __m256 acc0 = _mm256_setzero_ps();
        __m256 acc1 = _mm256_setzero_ps();
        __m256 acc2 = _mm256_setzero_ps();
        __m256 acc3 = _mm256_setzero_ps();

        for (; i + STRIDE_ELEMENTS <= count; i += STRIDE_ELEMENTS)
        {
            _mm_prefetch(reinterpret_cast<const char*>(&a[i]) + PREFETCH_OFFSET_BYTES, _MM_HINT_T0);
            _mm_prefetch(reinterpret_cast<const char*>(&b[i]) + PREFETCH_OFFSET_BYTES, _MM_HINT_T0);

            __m128i a_batch0 = _mm_load_si128(reinterpret_cast<const __m128i*>(a + i + 0 * F16_SIMD_LANES));
            __m128i b_batch0 = _mm_load_si128(reinterpret_cast<const __m128i*>(b + i + 0 * F16_SIMD_LANES));
            __m256 a_converted0 = _mm256_cvtph_ps(a_batch0);
            __m256 b_converted0 = _mm256_cvtph_ps(b_batch0);
            acc0 = _mm256_fmadd_ps(a_converted0, b_converted0, acc0);

            __m128i a_batch1 = _mm_load_si128(reinterpret_cast<const __m128i*>(a + i + 1 * F16_SIMD_LANES));
            __m128i b_batch1 = _mm_load_si128(reinterpret_cast<const __m128i*>(b + i + 1 * F16_SIMD_LANES));
            __m256 a_converted1 = _mm256_cvtph_ps(a_batch1);
            __m256 b_converted1 = _mm256_cvtph_ps(b_batch1);
            acc1 = _mm256_fmadd_ps(a_converted1, b_converted1, acc1);

            __m128i a_batch2 = _mm_load_si128(reinterpret_cast<const __m128i*>(a + i + 2 * F16_SIMD_LANES));
            __m128i b_batch2 = _mm_load_si128(reinterpret_cast<const __m128i*>(b + i + 2 * F16_SIMD_LANES));
            __m256 a_converted2 = _mm256_cvtph_ps(a_batch2);
            __m256 b_converted2 = _mm256_cvtph_ps(b_batch2);
            acc2 = _mm256_fmadd_ps(a_converted2, b_converted2, acc2);

            __m128i a_batch3 = _mm_load_si128(reinterpret_cast<const __m128i*>(a + i + 3 * F16_SIMD_LANES));
            __m128i b_batch3 = _mm_load_si128(reinterpret_cast<const __m128i*>(b + i + 3 * F16_SIMD_LANES));
            __m256 a_converted3 = _mm256_cvtph_ps(a_batch3);
            __m256 b_converted3 = _mm256_cvtph_ps(b_batch3);
            acc3 = _mm256_fmadd_ps(a_converted3, b_converted3, acc3);
        }

        acc0 = _mm256_add_ps(acc0, acc1);
        acc2 = _mm256_add_ps(acc2, acc3);
        acc0 = _mm256_add_ps(acc0, acc2);

        usize remaining = count - i;
        if (remaining >= BLOCK_32_ELEMENTS)
        {
            __m128i a_tail0 = _mm_load_si128(reinterpret_cast<const __m128i*>(a + i));
            __m128i b_tail0 = _mm_load_si128(reinterpret_cast<const __m128i*>(b + i));
            __m256 a_tail0_f32 = _mm256_cvtph_ps(a_tail0);
            __m256 b_tail0_f32 = _mm256_cvtph_ps(b_tail0);
            __m256 product0 = _mm256_mul_ps(a_tail0_f32, b_tail0_f32);
            acc0 = _mm256_add_ps(acc0, product0);
            i += F16_SIMD_LANES * 2;

            __m128i a_tail1 = _mm_load_si128(reinterpret_cast<const __m128i*>(a + i));
            __m128i b_tail1 = _mm_load_si128(reinterpret_cast<const __m128i*>(b + i));
            __m256 a_tail1_f32 = _mm256_cvtph_ps(a_tail1);
            __m256 b_tail1_f32 = _mm256_cvtph_ps(b_tail1);
            __m256 product1 = _mm256_mul_ps(a_tail1_f32, b_tail1_f32);
            acc0 = _mm256_add_ps(acc0, product1);
            i += F16_SIMD_LANES * 2;
            remaining -= BLOCK_32_ELEMENTS;
        }

        if (remaining >= BLOCK_16_ELEMENTS)
        {
            __m128i a_tail = _mm_load_si128(reinterpret_cast<const __m128i*>(a + i));
            __m128i b_tail = _mm_load_si128(reinterpret_cast<const __m128i*>(b + i));
            __m256 a_tail_f32 = _mm256_cvtph_ps(a_tail);
            __m256 b_tail_f32 = _mm256_cvtph_ps(b_tail);
            __m256 product = _mm256_mul_ps(a_tail_f32, b_tail_f32);
            acc0 = _mm256_add_ps(acc0, product);
            i += BLOCK_16_ELEMENTS;
            remaining -= BLOCK_16_ELEMENTS;
        }

        if (remaining >= BLOCK_8_ELEMENTS)
        {
            __m128i a_half = _mm_loadu_si64(a + i);
            __m128i b_half = _mm_loadu_si64(b + i);
            __m256 a_half_f32 = _mm256_cvtph_ps(a_half);
            __m256 b_half_f32 = _mm256_cvtph_ps(b_half);
            __m256 product = _mm256_mul_ps(a_half_f32, b_half_f32);
            acc0 = _mm256_add_ps(acc0, product);
            i += BLOCK_8_ELEMENTS;
            remaining -= BLOCK_8_ELEMENTS;
        }

        __m128 sum128 = _mm_add_ps(_mm256_castps256_ps128(acc0), _mm256_extractf128_ps(acc0, 1));
        sum128 = _mm_hadd_ps(sum128, sum128);
        sum128 = _mm_hadd_ps(sum128, sum128);
        result += _mm_cvtss_f32(sum128);

        for (; i < count; ++i)
        {
            result += static_cast<f64>(static_cast<f32>(a[i])) * static_cast<f32>(b[i]);
        }

        return result;
    }

    template<> f64 dot_product(const f32* a, const f32* b, usize count) noexcept
    {
        constexpr usize CACHE_LINE_BYTES = 64;
        constexpr usize F32_SIMD_LANES = 8;
        constexpr usize STRIDE_ELEMENTS = F32_SIMD_LANES * F32_SIMD_LANES;
        constexpr usize PREFETCH_OFFSET_BYTES = STRIDE_ELEMENTS * sizeof(f32);
        constexpr usize ADDITIONAL_PREFETCH_OFFSET = 128;
        constexpr usize BLOCK_32_ELEMENTS = 32;
        constexpr usize BLOCK_8_ELEMENTS = F32_SIMD_LANES;
        constexpr usize BLOCK_4_ELEMENTS = 4;

        constexpr usize SUB_BLOCK = 8;

        f64 result = 0.0;
        usize i = 0;

        while (i < count &&
            (reinterpret_cast<uintptr_t>(&a[i]) % CACHE_LINE_BYTES ||
                reinterpret_cast<uintptr_t>(&b[i]) % CACHE_LINE_BYTES))
        {
            result += static_cast<f64>(a[i]) * b[i];
            ++i;
        }

        __m256 acc0 = _mm256_setzero_ps();
        __m256 acc1 = _mm256_setzero_ps();
        __m256 acc2 = _mm256_setzero_ps();
        __m256 acc3 = _mm256_setzero_ps();
        __m256 acc4 = _mm256_setzero_ps();
        __m256 acc5 = _mm256_setzero_ps();
        __m256 acc6 = _mm256_setzero_ps();
        __m256 acc7 = _mm256_setzero_ps();

        for (; i <= count - STRIDE_ELEMENTS; i += STRIDE_ELEMENTS)
        {
            const char* a_prefetch = reinterpret_cast<const char*>(&a[i]);
            const char* b_prefetch = reinterpret_cast<const char*>(&b[i]);

            _mm_prefetch(a_prefetch + PREFETCH_OFFSET_BYTES, _MM_HINT_T0);
            _mm_prefetch(b_prefetch + PREFETCH_OFFSET_BYTES, _MM_HINT_T0);

            _mm_prefetch(a_prefetch + PREFETCH_OFFSET_BYTES + ADDITIONAL_PREFETCH_OFFSET, _MM_HINT_T0);
            _mm_prefetch(b_prefetch + PREFETCH_OFFSET_BYTES + ADDITIONAL_PREFETCH_OFFSET, _MM_HINT_T0);

            __m256 va0 = _mm256_load_ps(a + i + 0 * F32_SIMD_LANES);
            __m256 vb0 = _mm256_load_ps(b + i + 0 * F32_SIMD_LANES);
            acc0 = _mm256_fmadd_ps(va0, vb0, acc0);

            __m256 va1 = _mm256_load_ps(a + i + 1 * F32_SIMD_LANES);
            __m256 vb1 = _mm256_load_ps(b + i + 1 * F32_SIMD_LANES);
            acc1 = _mm256_fmadd_ps(va1, vb1, acc1);

            __m256 va2 = _mm256_load_ps(a + i + 2 * F32_SIMD_LANES);
            __m256 vb2 = _mm256_load_ps(b + i + 2 * F32_SIMD_LANES);
            acc2 = _mm256_fmadd_ps(va2, vb2, acc2);

            __m256 va3 = _mm256_load_ps(a + i + 3 * F32_SIMD_LANES);
            __m256 vb3 = _mm256_load_ps(b + i + 3 * F32_SIMD_LANES);
            acc3 = _mm256_fmadd_ps(va3, vb3, acc3);

            __m256 va4 = _mm256_load_ps(a + i + 4 * F32_SIMD_LANES);
            __m256 vb4 = _mm256_load_ps(b + i + 4 * F32_SIMD_LANES);
            acc4 = _mm256_fmadd_ps(va4, vb4, acc4);

            __m256 va5 = _mm256_load_ps(a + i + 5 * F32_SIMD_LANES);
            __m256 vb5 = _mm256_load_ps(b + i + 5 * F32_SIMD_LANES);
            acc5 = _mm256_fmadd_ps(va5, vb5, acc5);

            __m256 va6 = _mm256_load_ps(a + i + 6 * F32_SIMD_LANES);
            __m256 vb6 = _mm256_load_ps(b + i + 6 * F32_SIMD_LANES);
            acc6 = _mm256_fmadd_ps(va6, vb6, acc6);

            __m256 va7 = _mm256_load_ps(a + i + 7 * F32_SIMD_LANES);
            __m256 vb7 = _mm256_load_ps(b + i + 7 * F32_SIMD_LANES);
            acc7 = _mm256_fmadd_ps(va7, vb7, acc7);
        }

        __m256 sum01 = _mm256_add_ps(acc0, acc1);
        __m256 sum23 = _mm256_add_ps(acc2, acc3);
        __m256 sum45 = _mm256_add_ps(acc4, acc5);
        __m256 sum67 = _mm256_add_ps(acc6, acc7);

        __m256 sum0123 = _mm256_add_ps(sum01, sum23);
        __m256 sum4567 = _mm256_add_ps(sum45, sum67);

        __m256 total_acc = _mm256_add_ps(sum0123, sum4567);

        usize remaining = count - i;
        if (remaining >= BLOCK_32_ELEMENTS)
        {
            __m256 tail0 = _mm256_load_ps(a + i + 0 * SUB_BLOCK);
            __m256 tail1 = _mm256_load_ps(a + i + 1 * SUB_BLOCK);
            __m256 tail2 = _mm256_load_ps(a + i + 2 * SUB_BLOCK);
            __m256 tail3 = _mm256_load_ps(a + i + 3 * SUB_BLOCK);

            __m256 vmr0 = _mm256_load_ps(b + i + 0 * SUB_BLOCK);
            __m256 vmr1 = _mm256_load_ps(b + i + 1 * SUB_BLOCK);
            __m256 vmr2 = _mm256_load_ps(b + i + 2 * SUB_BLOCK);
            __m256 vmr3 = _mm256_load_ps(b + i + 3 * SUB_BLOCK);

            tail0 = _mm256_mul_ps(tail0, vmr0);
            tail1 = _mm256_mul_ps(tail1, vmr1);
            tail2 = _mm256_mul_ps(tail2, vmr2);
            tail3 = _mm256_mul_ps(tail3, vmr3);

            __m256 tail_sum = _mm256_add_ps(_mm256_add_ps(tail0, tail1), _mm256_add_ps(tail2, tail3));
            total_acc = _mm256_add_ps(total_acc, tail_sum);
            i += BLOCK_32_ELEMENTS;
            remaining -= BLOCK_32_ELEMENTS;
        }

        __m128 sse_acc = _mm_add_ps(
            _mm256_castps256_ps128(total_acc),
            _mm256_extractf128_ps(total_acc, 1));

        if (remaining >= BLOCK_8_ELEMENTS)
        {
            __m256 tail = _mm256_load_ps(a + i);
            tail = _mm256_mul_ps(tail, _mm256_load_ps(b + i));
            __m128 tail_low = _mm256_castps256_ps128(tail);
            __m128 tail_high = _mm256_extractf128_ps(tail, 1);
            sse_acc = _mm_add_ps(sse_acc, _mm_add_ps(tail_low, tail_high));
            i += BLOCK_8_ELEMENTS;
            remaining -= BLOCK_8_ELEMENTS;
        }

        if (remaining >= BLOCK_4_ELEMENTS)
        {
            __m128 a4 = _mm_load_ps(a + i);
            __m128 b4 = _mm_load_ps(b + i);
            __m128 product = _mm_mul_ps(a4, b4);
            sse_acc = _mm_add_ps(sse_acc, product);
            i += BLOCK_4_ELEMENTS;
            remaining -= BLOCK_4_ELEMENTS;
        }

        sse_acc = _mm_hadd_ps(sse_acc, sse_acc);
        sse_acc = _mm_hadd_ps(sse_acc, sse_acc);
        result += _mm_cvtss_f32(sse_acc);

        for (; i < count; ++i)
        {
            result += static_cast<f64>(a[i]) * b[i];
        }

        return result;
    }

    template<> f64 dot_product(const f64* a, const f64* b, usize count) noexcept
    {
        constexpr usize F64_ALIGNMENT = 32;
        constexpr usize F64_SIMD_LANES = 4;
        constexpr usize BATCHES_PER_LOOP = 8;
        constexpr usize STRIDE_ELEMENTS = BATCHES_PER_LOOP * F64_SIMD_LANES;
        constexpr usize PREFETCH_DISTANCE = 512;
        constexpr usize BLOCK_16_ELEMENTS = 16;
        constexpr usize BLOCK_8_ELEMENTS = 8;
        constexpr usize BLOCK_4_ELEMENTS = F64_SIMD_LANES;

        f64 result = 0.0;
        usize i = 0;

        if (count >= F64_SIMD_LANES)
        {
            const uintptr_t a_addr = reinterpret_cast<uintptr_t>(a);
            const uintptr_t b_addr = reinterpret_cast<uintptr_t>(b);
            const usize a_misalign = (F64_ALIGNMENT - (a_addr % F64_ALIGNMENT)) % F64_ALIGNMENT;
            const usize b_misalign = (F64_ALIGNMENT - (b_addr % F64_ALIGNMENT)) % F64_ALIGNMENT;
            const usize max_misalign = std::max(a_misalign, b_misalign) / sizeof(f64);
            const usize head_end = std::min(max_misalign, count);

            for (; i < head_end; ++i)
            {
                result += a[i] * b[i];
            }
        }

        __m256d acc0 = _mm256_setzero_pd();
        __m256d acc1 = _mm256_setzero_pd();
        __m256d acc2 = _mm256_setzero_pd();
        __m256d acc3 = _mm256_setzero_pd();
        __m256d acc4 = _mm256_setzero_pd();
        __m256d acc5 = _mm256_setzero_pd();
        __m256d acc6 = _mm256_setzero_pd();
        __m256d acc7 = _mm256_setzero_pd();

        for (; i <= count - STRIDE_ELEMENTS; i += STRIDE_ELEMENTS)
        {
            _mm_prefetch(reinterpret_cast<const char*>(a + i) + PREFETCH_DISTANCE, _MM_HINT_T0);
            _mm_prefetch(reinterpret_cast<const char*>(b + i) + PREFETCH_DISTANCE, _MM_HINT_T0);

            __m256d vmul_l_0 = _mm256_load_pd(a + i + 0 * F64_SIMD_LANES);
            __m256d vmul_l_1 = _mm256_load_pd(a + i + 1 * F64_SIMD_LANES);
            __m256d vmul_l_2 = _mm256_load_pd(a + i + 2 * F64_SIMD_LANES);
            __m256d vmul_l_3 = _mm256_load_pd(a + i + 3 * F64_SIMD_LANES);
            __m256d vmul_l_4 = _mm256_load_pd(a + i + 4 * F64_SIMD_LANES);
            __m256d vmul_l_5 = _mm256_load_pd(a + i + 5 * F64_SIMD_LANES);
            __m256d vmul_l_6 = _mm256_load_pd(a + i + 6 * F64_SIMD_LANES);
            __m256d vmul_l_7 = _mm256_load_pd(a + i + 7 * F64_SIMD_LANES);

            __m256d vmul_r_0 = _mm256_load_pd(b + i + 0 * F64_SIMD_LANES);
            __m256d vmul_r_1 = _mm256_load_pd(b + i + 1 * F64_SIMD_LANES);
            __m256d vmul_r_2 = _mm256_load_pd(b + i + 2 * F64_SIMD_LANES);
            __m256d vmul_r_3 = _mm256_load_pd(b + i + 3 * F64_SIMD_LANES);
            __m256d vmul_r_4 = _mm256_load_pd(b + i + 4 * F64_SIMD_LANES);
            __m256d vmul_r_5 = _mm256_load_pd(b + i + 5 * F64_SIMD_LANES);
            __m256d vmul_r_6 = _mm256_load_pd(b + i + 6 * F64_SIMD_LANES);
            __m256d vmul_r_7 = _mm256_load_pd(b + i + 7 * F64_SIMD_LANES);

            acc0 = _mm256_fmadd_pd(vmul_l_0, vmul_r_0, acc0);
            acc1 = _mm256_fmadd_pd(vmul_l_1, vmul_r_1, acc1);
            acc2 = _mm256_fmadd_pd(vmul_l_2, vmul_r_2, acc2);
            acc3 = _mm256_fmadd_pd(vmul_l_3, vmul_r_3, acc3);
            acc4 = _mm256_fmadd_pd(vmul_l_4, vmul_r_4, acc4);
            acc5 = _mm256_fmadd_pd(vmul_l_5, vmul_r_5, acc5);
            acc6 = _mm256_fmadd_pd(vmul_l_6, vmul_r_6, acc6);
            acc7 = _mm256_fmadd_pd(vmul_l_7, vmul_r_7, acc7);
        }

        acc0 = _mm256_add_pd(acc0, acc1);
        acc2 = _mm256_add_pd(acc2, acc3);
        acc4 = _mm256_add_pd(acc4, acc5);
        acc6 = _mm256_add_pd(acc6, acc7);
        acc0 = _mm256_add_pd(acc0, acc2);
        acc4 = _mm256_add_pd(acc4, acc6);
        __m256d total_acc = _mm256_add_pd(acc0, acc4);

        usize remaining = count - i;
        if (remaining >= BLOCK_16_ELEMENTS)
        {
            __m256d vmul_l_0 = _mm256_load_pd(a + i + 0);
            __m256d vmul_l_1 = _mm256_load_pd(a + i + 4);
            __m256d vmul_l_2 = _mm256_load_pd(a + i + 8);
            __m256d vmul_l_3 = _mm256_load_pd(a + i + 12);

            __m256d vmul_r_0 = _mm256_load_pd(b + i + 0);
            __m256d vmul_r_1 = _mm256_load_pd(b + i + 4);
            __m256d vmul_r_2 = _mm256_load_pd(b + i + 8);
            __m256d vmul_r_3 = _mm256_load_pd(b + i + 12);

            total_acc = _mm256_fmadd_pd(vmul_l_0, vmul_r_0, total_acc);
            total_acc = _mm256_fmadd_pd(vmul_l_1, vmul_r_1, total_acc);
            total_acc = _mm256_fmadd_pd(vmul_l_2, vmul_r_2, total_acc);
            total_acc = _mm256_fmadd_pd(vmul_l_3, vmul_r_3, total_acc);

            i += BLOCK_16_ELEMENTS;
            remaining -= BLOCK_16_ELEMENTS;
        }

        if (remaining >= BLOCK_8_ELEMENTS)
        {
            __m256d vmul_l_0 = _mm256_load_pd(a + i + 0);
            __m256d vmul_l_1 = _mm256_load_pd(a + i + 4);

            __m256d vmul_r_0 = _mm256_load_pd(b + i + 0);
            __m256d vmul_r_1 = _mm256_load_pd(b + i + 4);

            total_acc = _mm256_fmadd_pd(vmul_l_0, vmul_r_0, total_acc);
            total_acc = _mm256_fmadd_pd(vmul_l_1, vmul_r_1, total_acc);
            i += BLOCK_8_ELEMENTS;
            remaining -= BLOCK_8_ELEMENTS;
        }

        if (remaining >= BLOCK_4_ELEMENTS)
        {
            __m256d vmul_l_0 = _mm256_load_pd(a + i);
            __m256d vmul_r_1 = _mm256_load_pd(b + i);

            total_acc = _mm256_fmadd_pd(vmul_l_0, vmul_r_1, total_acc);
            i += BLOCK_4_ELEMENTS;
            remaining -= BLOCK_4_ELEMENTS;
        }

        __m128d low = _mm256_castpd256_pd128(total_acc);
        __m128d high = _mm256_extractf128_pd(total_acc, 1);
        low = _mm_add_pd(low, high);
        __m128d sum = _mm_add_sd(low, _mm_unpackhi_pd(low, low));
        result += _mm_cvtsd_f64(sum);

        for (; i + 3 < count; i += 4)
        {
            result += a[i] * b[i] + a[i + 1] * b[i + 1] + a[i + 2] * b[i + 2] + a[i + 3] * b[i + 3];
        }

        for (; i < count; ++i)
        {
            result += a[i] * b[i];
        }

        return result;
    }
}

namespace fy
{
    template<> i64 dot_product(const i8* a, const i8* b, usize count) noexcept
    {
        alignas(64) i32 sum { 0 };
        const i8* end = a + count;
        const i8* end256 = a + (count & ~usize(255));

        __m256i acc0 = _mm256_setzero_si256();
        __m256i acc1 = _mm256_setzero_si256();
        __m256i acc2 = _mm256_setzero_si256();
        __m256i acc3 = _mm256_setzero_si256();
        __m256i acc4 = _mm256_setzero_si256();
        __m256i acc5 = _mm256_setzero_si256();
        __m256i acc6 = _mm256_setzero_si256();
        __m256i acc7 = _mm256_setzero_si256();

        while (a < end256)
        {
            _mm_prefetch(reinterpret_cast<const char*>(a) + 512, _MM_HINT_T0);
            _mm_prefetch(reinterpret_cast<const char*>(b) + 512, _MM_HINT_T0);

            for (usize j = 0; j < 8; ++j)
            {
                __m256i a_vec = _mm256_load_si256(reinterpret_cast<const __m256i*>(a + j * 32));
                __m256i b_vec = _mm256_load_si256(reinterpret_cast<const __m256i*>(b + j * 32));

                __m128i a_low = _mm256_castsi256_si128(a_vec);
                __m128i a_high = _mm256_extracti128_si256(a_vec, 1);
                __m128i b_low = _mm256_castsi256_si128(b_vec);
                __m128i b_high = _mm256_extracti128_si256(b_vec, 1);

                __m256i a_low_16 = _mm256_cvtepi8_epi16(a_low);
                __m256i a_high_16 = _mm256_cvtepi8_epi16(a_high);
                __m256i b_low_16 = _mm256_cvtepi8_epi16(b_low);
                __m256i b_high_16 = _mm256_cvtepi8_epi16(b_high);

                __m256i prod_low = _mm256_madd_epi16(a_low_16, b_low_16);
                __m256i prod_high = _mm256_madd_epi16(a_high_16, b_high_16);

                switch (j % 4)
                {
                    case 0: acc0 = _mm256_add_epi32(acc0, prod_low); acc1 = _mm256_add_epi32(acc1, prod_high); break;
                    case 1: acc2 = _mm256_add_epi32(acc2, prod_low); acc3 = _mm256_add_epi32(acc3, prod_high); break;
                    case 2: acc4 = _mm256_add_epi32(acc4, prod_low); acc5 = _mm256_add_epi32(acc5, prod_high); break;
                    case 3: acc6 = _mm256_add_epi32(acc6, prod_low); acc7 = _mm256_add_epi32(acc7, prod_high); break;
                }
            }

            a += 256;
            b += 256;
        }

        acc0 = _mm256_add_epi32(acc0, acc1);
        acc2 = _mm256_add_epi32(acc2, acc3);
        acc4 = _mm256_add_epi32(acc4, acc5);
        acc6 = _mm256_add_epi32(acc6, acc7);
        acc0 = _mm256_add_epi32(acc0, acc2);
        acc4 = _mm256_add_epi32(acc4, acc6);
        __m256i sum256 = _mm256_add_epi32(acc0, acc4);

        __m128i sum128 = _mm_add_epi32(_mm256_castsi256_si128(sum256), _mm256_extracti128_si256(sum256, 1));
        sum128 = _mm_add_epi32(sum128, _mm_shuffle_epi32(sum128, _MM_SHUFFLE(1, 0, 3, 2)));
        sum += _mm_extract_epi32(sum128, 0) + _mm_extract_epi32(sum128, 1);

        if (a < end)
        {
            __m256i tail_acc = _mm256_setzero_si256();
            usize remain = end - a;
            const i8* end32 = a + (remain & ~usize(31));

            while (a < end32)
            {
                __m256i a_vec = _mm256_load_si256(reinterpret_cast<const __m256i*>(a));
                __m256i b_vec = _mm256_load_si256(reinterpret_cast<const __m256i*>(b));

                __m128i a_low = _mm256_castsi256_si128(a_vec);
                __m128i a_high = _mm256_extracti128_si256(a_vec, 1);
                __m128i b_low = _mm256_castsi256_si128(b_vec);
                __m128i b_high = _mm256_extracti128_si256(b_vec, 1);

                __m256i a_low_16 = _mm256_cvtepi8_epi16(a_low);
                __m256i a_high_16 = _mm256_cvtepi8_epi16(a_high);
                __m256i b_low_16 = _mm256_cvtepi8_epi16(b_low);
                __m256i b_high_16 = _mm256_cvtepi8_epi16(b_high);

                __m256i prod_low = _mm256_madd_epi16(a_low_16, b_low_16);
                __m256i prod_high = _mm256_madd_epi16(a_high_16, b_high_16);

                tail_acc = _mm256_add_epi32(tail_acc, prod_low);
                tail_acc = _mm256_add_epi32(tail_acc, prod_high);

                a += 32;
                b += 32;
            }

            __m128i t = _mm_add_epi32(_mm256_castsi256_si128(tail_acc), _mm256_extracti128_si256(tail_acc, 1));
            t = _mm_hadd_epi32(t, t);
            sum += _mm_extract_epi32(t, 0) + _mm_extract_epi32(t, 1);

            while (a < end)
            {
                sum += static_cast<i32>(*a) * static_cast<i32>(*b);
                ++a;
                ++b;
            }
        }

        return sum;
    }

    template<> u64 dot_product(const u8* a, const u8* b, usize count) noexcept
    {
        alignas(64) u64 sum { 0 };
        const u8* end = a + count;
        const u8* end256 = a + (count & ~255);

        __m256i acc0 = _mm256_setzero_si256();
        __m256i acc1 = _mm256_setzero_si256();
        __m256i acc2 = _mm256_setzero_si256();
        __m256i acc3 = _mm256_setzero_si256();
        __m256i acc4 = _mm256_setzero_si256();
        __m256i acc5 = _mm256_setzero_si256();
        __m256i acc6 = _mm256_setzero_si256();
        __m256i acc7 = _mm256_setzero_si256();

        while (a < end256)
        {
            _mm_prefetch(reinterpret_cast<const char*>(a) + 512, _MM_HINT_T0);
            _mm_prefetch(reinterpret_cast<const char*>(b) + 512, _MM_HINT_T0);

            for (usize j = 0; j < 8; ++j)
            {
                __m256i a_vec = _mm256_load_si256(reinterpret_cast<const __m256i*>(a + j * 32));
                __m256i b_vec = _mm256_load_si256(reinterpret_cast<const __m256i*>(b + j * 32));

                __m128i a_low = _mm256_castsi256_si128(a_vec);
                __m128i a_high = _mm256_extracti128_si256(a_vec, 1);
                __m256i a_low_16 = _mm256_cvtepu8_epi16(a_low);
                __m256i a_high_16 = _mm256_cvtepu8_epi16(a_high);

                __m128i b_low = _mm256_castsi256_si128(b_vec);
                __m128i b_high = _mm256_extracti128_si256(b_vec, 1);
                __m256i b_low_16 = _mm256_cvtepu8_epi16(b_low);
                __m256i b_high_16 = _mm256_cvtepu8_epi16(b_high);

                __m256i prod_low = _mm256_madd_epi16(a_low_16, b_low_16);
                __m256i prod_high = _mm256_madd_epi16(a_high_16, b_high_16);

                switch (j % 4)
                {
                    case 0: acc0 = _mm256_add_epi32(acc0, prod_low); acc1 = _mm256_add_epi32(acc1, prod_high); break;
                    case 1: acc2 = _mm256_add_epi32(acc2, prod_low); acc3 = _mm256_add_epi32(acc3, prod_high); break;
                    case 2: acc4 = _mm256_add_epi32(acc4, prod_low); acc5 = _mm256_add_epi32(acc5, prod_high); break;
                    case 3: acc6 = _mm256_add_epi32(acc6, prod_low); acc7 = _mm256_add_epi32(acc7, prod_high); break;
                }
            }
            a += 256;
            b += 256;
        }

        acc0 = _mm256_add_epi32(acc0, acc1);
        acc2 = _mm256_add_epi32(acc2, acc3);
        acc4 = _mm256_add_epi32(acc4, acc5);
        acc6 = _mm256_add_epi32(acc6, acc7);
        acc0 = _mm256_add_epi32(acc0, acc2);
        acc4 = _mm256_add_epi32(acc4, acc6);
        __m256i sum256 = _mm256_add_epi32(acc0, acc4);

        __m128i sum128 = _mm_add_epi32(_mm256_castsi256_si128(sum256), _mm256_extracti128_si256(sum256, 1));
        sum128 = _mm_add_epi32(sum128, _mm_shuffle_epi32(sum128, _MM_SHUFFLE(1, 0, 3, 2)));
        sum += _mm_extract_epi32(sum128, 0) + _mm_extract_epi32(sum128, 1);

        if (a < end)
        {
            __m256i tail_acc = _mm256_setzero_si256();
            usize remain = end - a;
            const u8* end32 = a + (remain & ~31);

            while (a < end32)
            {
                __m256i a_vec = _mm256_load_si256(reinterpret_cast<const __m256i*>(a));
                __m256i b_vec = _mm256_load_si256(reinterpret_cast<const __m256i*>(b));

                __m128i a_low = _mm256_castsi256_si128(a_vec);
                __m128i a_high = _mm256_extracti128_si256(a_vec, 1);
                __m256i a_low_16 = _mm256_cvtepu8_epi16(a_low);
                __m256i a_high_16 = _mm256_cvtepu8_epi16(a_high);

                __m128i b_low = _mm256_castsi256_si128(b_vec);
                __m128i b_high = _mm256_extracti128_si256(b_vec, 1);
                __m256i b_low_16 = _mm256_cvtepu8_epi16(b_low);
                __m256i b_high_16 = _mm256_cvtepu8_epi16(b_high);

                __m256i prod_low = _mm256_madd_epi16(a_low_16, b_low_16);
                __m256i prod_high = _mm256_madd_epi16(a_high_16, b_high_16);

                tail_acc = _mm256_add_epi32(tail_acc, prod_low);
                tail_acc = _mm256_add_epi32(tail_acc, prod_high);

                a += 32;
                b += 32;
            }

            __m128i t = _mm_add_epi32(_mm256_castsi256_si128(tail_acc), _mm256_extracti128_si256(tail_acc, 1));
            t = _mm_hadd_epi32(t, t);
            sum += _mm_extract_epi32(t, 0) + _mm_extract_epi32(t, 1);

            while (a < end)
            {
                sum += static_cast<u32>(*a) * static_cast<u32>(*b);
                ++a;
                ++b;
            }
        }

        return sum;
    }

    template<> i64 dot_product(const i16* a, const i16* b, usize count) noexcept
    {
        alignas(64) i64 sum = 0;
        const i16* end = a + count;
        const i16* end128 = a + (count & ~127);

        __m256i acc0 = _mm256_setzero_si256();
        __m256i acc1 = _mm256_setzero_si256();
        __m256i acc2 = _mm256_setzero_si256();
        __m256i acc3 = _mm256_setzero_si256();

        while (a < end128)
        {
            _mm_prefetch(reinterpret_cast<const char*>(a) + 512, _MM_HINT_T0);
            _mm_prefetch(reinterpret_cast<const char*>(b) + 512, _MM_HINT_T0);

            for (usize j = 0; j < 8; ++j)
            {
                __m256i a_vec = _mm256_load_si256(reinterpret_cast<const __m256i*>(a));
                __m256i b_vec = _mm256_load_si256(reinterpret_cast<const __m256i*>(b));
                a += 16;
                b += 16;

                __m256i prod = _mm256_madd_epi16(a_vec, b_vec);

                __m128i prod_lo = _mm256_castsi256_si128(prod);
                __m128i prod_hi = _mm256_extracti128_si256(prod, 1);

                __m256i prod_lo64 = _mm256_cvtepi32_epi64(prod_lo);
                __m256i prod_hi64 = _mm256_cvtepi32_epi64(prod_hi);

                if (j % 2 == 0)
                {
                    acc0 = _mm256_add_epi64(acc0, prod_lo64);
                    acc1 = _mm256_add_epi64(acc1, prod_hi64);
                }
                else
                {
                    acc2 = _mm256_add_epi64(acc2, prod_lo64);
                    acc3 = _mm256_add_epi64(acc3, prod_hi64);
                }
            }
        }

        acc0 = _mm256_add_epi64(acc0, acc1);
        acc2 = _mm256_add_epi64(acc2, acc3);
        acc0 = _mm256_add_epi64(acc0, acc2);

        __m128i sum128 = _mm_add_epi64(
            _mm256_castsi256_si128(acc0),
            _mm256_extracti128_si256(acc0, 1)
        );
        sum += _mm_extract_epi64(sum128, 0) + _mm_extract_epi64(sum128, 1);

        if (a < end)
        {
            __m256i tail_acc = _mm256_setzero_si256();
            const i16* end16 = a + ((end - a) & ~15);

            while (a < end16)
            {
                __m256i a_vec = _mm256_load_si256(reinterpret_cast<const __m256i*>(a));
                __m256i b_vec = _mm256_load_si256(reinterpret_cast<const __m256i*>(b));
                a += 16;
                b += 16;

                __m256i prod = _mm256_madd_epi16(a_vec, b_vec);
                __m128i prod_lo = _mm256_castsi256_si128(prod);
                __m128i prod_hi = _mm256_extracti128_si256(prod, 1);

                tail_acc = _mm256_add_epi64(tail_acc, _mm256_cvtepi32_epi64(prod_lo));
                tail_acc = _mm256_add_epi64(tail_acc, _mm256_cvtepi32_epi64(prod_hi));
            }

            __m128i tail128 = _mm_add_epi64(
                _mm256_castsi256_si128(tail_acc),
                _mm256_extracti128_si256(tail_acc, 1)
            );
            sum += _mm_extract_epi64(tail128, 0) + _mm_extract_epi64(tail128, 1);

            while (a < end)
            {
                sum += static_cast<i64>(*a) * static_cast<i64>(*b);
                ++a;
                ++b;
            }
        }

        return sum;
    }

    template<> u64 dot_product(const u16* a, const u16* b, usize count) noexcept
    {
        alignas(64) u64 sum = 0;
        const u16* end = a + count;
        const u16* end256 = a + (count & ~255);

        __m256i acc0 = _mm256_setzero_si256();
        __m256i acc1 = _mm256_setzero_si256();
        __m256i acc2 = _mm256_setzero_si256();
        __m256i acc3 = _mm256_setzero_si256();
        __m256i acc4 = _mm256_setzero_si256();
        __m256i acc5 = _mm256_setzero_si256();
        __m256i acc6 = _mm256_setzero_si256();
        __m256i acc7 = _mm256_setzero_si256();

        while (a < end256)
        {
            _mm_prefetch(reinterpret_cast<const char*>(a) + 1024, _MM_HINT_T0);
            _mm_prefetch(reinterpret_cast<const char*>(b) + 1024, _MM_HINT_T0);

            for (usize j = 0; j < 8; ++j)
            {
                __m256i a_vec = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a));
                __m256i b_vec = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b));
                a += 16;
                b += 16;

                __m128i a_low128 = _mm256_castsi256_si128(a_vec);
                __m128i a_high128 = _mm256_extracti128_si256(a_vec, 1);
                __m128i b_low128 = _mm256_castsi256_si128(b_vec);
                __m128i b_high128 = _mm256_extracti128_si256(b_vec, 1);

                __m256i a_low = _mm256_cvtepu16_epi32(a_low128);
                __m256i a_high = _mm256_cvtepu16_epi32(a_high128);
                __m256i b_low = _mm256_cvtepu16_epi32(b_low128);
                __m256i b_high = _mm256_cvtepu16_epi32(b_high128);

                __m256i prod_low = _mm256_mullo_epi32(a_low, b_low);
                __m256i prod_high = _mm256_mullo_epi32(a_high, b_high);

                __m256i pl_lo = _mm256_cvtepu32_epi64(_mm256_castsi256_si128(prod_low));
                __m256i pl_hi = _mm256_cvtepu32_epi64(_mm256_extracti128_si256(prod_low, 1));
                __m256i ph_lo = _mm256_cvtepu32_epi64(_mm256_castsi256_si128(prod_high));
                __m256i ph_hi = _mm256_cvtepu32_epi64(_mm256_extracti128_si256(prod_high, 1));

                switch (j % 4)
                {
                    case 0:
                        acc0 = _mm256_add_epi64(acc0, pl_lo);
                        acc1 = _mm256_add_epi64(acc1, pl_hi);
                        acc2 = _mm256_add_epi64(acc2, ph_lo);
                        acc3 = _mm256_add_epi64(acc3, ph_hi);
                        break;
                    case 1:
                        acc4 = _mm256_add_epi64(acc4, pl_lo);
                        acc5 = _mm256_add_epi64(acc5, pl_hi);
                        acc6 = _mm256_add_epi64(acc6, ph_lo);
                        acc7 = _mm256_add_epi64(acc7, ph_hi);
                        break;
                    case 2:
                        acc0 = _mm256_add_epi64(acc0, pl_lo);
                        acc1 = _mm256_add_epi64(acc1, pl_hi);
                        acc2 = _mm256_add_epi64(acc2, ph_lo);
                        acc3 = _mm256_add_epi64(acc3, ph_hi);
                        break;
                    case 3:
                        acc4 = _mm256_add_epi64(acc4, pl_lo);
                        acc5 = _mm256_add_epi64(acc5, pl_hi);
                        acc6 = _mm256_add_epi64(acc6, ph_lo);
                        acc7 = _mm256_add_epi64(acc7, ph_hi);
                        break;
                }
            }
        }

        acc0 = _mm256_add_epi64(acc0, acc1);
        acc2 = _mm256_add_epi64(acc2, acc3);
        acc4 = _mm256_add_epi64(acc4, acc5);
        acc6 = _mm256_add_epi64(acc6, acc7);
        acc0 = _mm256_add_epi64(acc0, acc2);
        acc4 = _mm256_add_epi64(acc4, acc6);
        __m256i sum256 = _mm256_add_epi64(acc0, acc4);

        __m128i sum128 = _mm_add_epi64(
            _mm256_castsi256_si128(sum256),
            _mm256_extracti128_si256(sum256, 1)
        );
        sum += _mm_extract_epi64(sum128, 0) + _mm_extract_epi64(sum128, 1);

        if (a < end)
        {
            __m256i tail_acc = _mm256_setzero_si256();
            usize remain = end - a;

            if (remain >= 16)
            {
                __m256i a_vec = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a));
                __m256i b_vec = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b));
                a += 16;
                b += 16;

                __m128i a_low128 = _mm256_castsi256_si128(a_vec);
                __m128i a_high128 = _mm256_extracti128_si256(a_vec, 1);
                __m128i b_low128 = _mm256_castsi256_si128(b_vec);
                __m128i b_high128 = _mm256_extracti128_si256(b_vec, 1);

                __m256i a_low = _mm256_cvtepu16_epi32(a_low128);
                __m256i a_high = _mm256_cvtepu16_epi32(a_high128);
                __m256i b_low = _mm256_cvtepu16_epi32(b_low128);
                __m256i b_high = _mm256_cvtepu16_epi32(b_high128);

                __m256i prod_low = _mm256_mullo_epi32(a_low, b_low);
                __m256i prod_high = _mm256_mullo_epi32(a_high, b_high);

                tail_acc = _mm256_add_epi64(tail_acc, _mm256_cvtepu32_epi64(_mm256_castsi256_si128(prod_low)));
                tail_acc = _mm256_add_epi64(tail_acc, _mm256_cvtepu32_epi64(_mm256_extracti128_si256(prod_low, 1)));
                tail_acc = _mm256_add_epi64(tail_acc, _mm256_cvtepu32_epi64(_mm256_castsi256_si128(prod_high)));
                tail_acc = _mm256_add_epi64(tail_acc, _mm256_cvtepu32_epi64(_mm256_extracti128_si256(prod_high, 1)));
                remain -= 16;
            }

            if (remain >= 8)
            {
                __m128i a_vec = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a));
                __m128i b_vec = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b));
                a += 8;
                b += 8;

                __m256i a_ext = _mm256_cvtepu16_epi32(a_vec);
                __m256i b_ext = _mm256_cvtepu16_epi32(b_vec);
                __m256i prod = _mm256_mullo_epi32(a_ext, b_ext);

                tail_acc = _mm256_add_epi64(tail_acc, _mm256_cvtepu32_epi64(_mm256_castsi256_si128(prod)));
                tail_acc = _mm256_add_epi64(tail_acc, _mm256_cvtepu32_epi64(_mm256_extracti128_si256(prod, 1)));
                remain -= 8;
            }

            while (remain-- > 0)
            {
                sum += static_cast<u64>(*a) * static_cast<u64>(*b);
                ++a;
                ++b;
            }

            __m128i tail128 = _mm_add_epi64(
                _mm256_castsi256_si128(tail_acc),
                _mm256_extracti128_si256(tail_acc, 1)
            );
            sum += _mm_extract_epi64(tail128, 0) + _mm_extract_epi64(tail128, 1);
        }

        return sum;
    }

    template<> i64 dot_product(const i32* a, const i32* b, usize count) noexcept
    {
        alignas(64) i64 sum = 0;
        const i32* end = a + count;
        const i32* end256 = a + (count & ~255);

        __m256i acc0 = _mm256_setzero_si256();
        __m256i acc1 = _mm256_setzero_si256();
        __m256i acc2 = _mm256_setzero_si256();
        __m256i acc3 = _mm256_setzero_si256();
        __m256i acc4 = _mm256_setzero_si256();
        __m256i acc5 = _mm256_setzero_si256();
        __m256i acc6 = _mm256_setzero_si256();
        __m256i acc7 = _mm256_setzero_si256();

        while (a < end256)
        {
            _mm_prefetch(reinterpret_cast<const char*>(a) + 512, _MM_HINT_T0);
            _mm_prefetch(reinterpret_cast<const char*>(b) + 512, _MM_HINT_T0);

            for (usize j = 0; j < 8; ++j)
            {
                __m256i a_vec = _mm256_load_si256(reinterpret_cast<const __m256i*>(a));
                __m256i b_vec = _mm256_load_si256(reinterpret_cast<const __m256i*>(b));
                a += 8;
                b += 8;

                __m128i a_low = _mm256_castsi256_si128(a_vec);
                __m256i a_low_ext = _mm256_cvtepi32_epi64(a_low);
                __m128i b_low = _mm256_castsi256_si128(b_vec);
                __m256i b_low_ext = _mm256_cvtepi32_epi64(b_low);
                __m256i prod_low = _mm256_mul_epi32(a_low_ext, b_low_ext);

                __m128i a_high = _mm256_extracti128_si256(a_vec, 1);
                __m256i a_high_ext = _mm256_cvtepi32_epi64(a_high);
                __m128i b_high = _mm256_extracti128_si256(b_vec, 1);
                __m256i b_high_ext = _mm256_cvtepi32_epi64(b_high);
                __m256i prod_high = _mm256_mul_epi32(a_high_ext, b_high_ext);

                switch (j % 4)
                {
                    case 0:
                        acc0 = _mm256_add_epi64(acc0, prod_low);
                        acc1 = _mm256_add_epi64(acc1, prod_high);
                        break;
                    case 1:
                        acc2 = _mm256_add_epi64(acc2, prod_low);
                        acc3 = _mm256_add_epi64(acc3, prod_high);
                        break;
                    case 2:
                        acc4 = _mm256_add_epi64(acc4, prod_low);
                        acc5 = _mm256_add_epi64(acc5, prod_high);
                        break;
                    case 3:
                        acc6 = _mm256_add_epi64(acc6, prod_low);
                        acc7 = _mm256_add_epi64(acc7, prod_high);
                        break;
                }
            }
        }

        acc0 = _mm256_add_epi64(acc0, acc1);
        acc2 = _mm256_add_epi64(acc2, acc3);
        acc4 = _mm256_add_epi64(acc4, acc5);
        acc6 = _mm256_add_epi64(acc6, acc7);
        acc0 = _mm256_add_epi64(acc0, acc2);
        acc4 = _mm256_add_epi64(acc4, acc6);
        __m256i sum256 = _mm256_add_epi64(acc0, acc4);

        __m128i sum_low = _mm256_castsi256_si128(sum256);
        __m128i sum_high = _mm256_extracti128_si256(sum256, 1);
        __m128i sum128 = _mm_add_epi64(sum_low, sum_high);
        sum += _mm_extract_epi64(sum128, 0) + _mm_extract_epi64(sum128, 1);

        if (a < end)
        {
            __m256i tail_acc = _mm256_setzero_si256();
            usize remain = end - a;
            const i32* end32 = a + (remain & ~7);

            while (a < end32)
            {
                __m256i a_vec = _mm256_load_si256(reinterpret_cast<const __m256i*>(a));
                __m256i b_vec = _mm256_load_si256(reinterpret_cast<const __m256i*>(b));
                a += 8;
                b += 8;

                __m128i a_low = _mm256_castsi256_si128(a_vec);
                __m256i a_low_ext = _mm256_cvtepi32_epi64(a_low);
                __m128i b_low = _mm256_castsi256_si128(b_vec);
                __m256i b_low_ext = _mm256_cvtepi32_epi64(b_low);
                __m256i prod_low = _mm256_mul_epi32(a_low_ext, b_low_ext);

                __m128i a_high = _mm256_extracti128_si256(a_vec, 1);
                __m256i a_high_ext = _mm256_cvtepi32_epi64(a_high);
                __m128i b_high = _mm256_extracti128_si256(b_vec, 1);
                __m256i b_high_ext = _mm256_cvtepi32_epi64(b_high);
                __m256i prod_high = _mm256_mul_epi32(a_high_ext, b_high_ext);

                tail_acc = _mm256_add_epi64(tail_acc, prod_low);
                tail_acc = _mm256_add_epi64(tail_acc, prod_high);
            }

            __m128i tail_low = _mm256_castsi256_si128(tail_acc);
            __m128i tail_high = _mm256_extracti128_si256(tail_acc, 1);
            __m128i tail_sum = _mm_add_epi64(tail_low, tail_high);
            sum += _mm_extract_epi64(tail_sum, 0) + _mm_extract_epi64(tail_sum, 1);

            while (a < end)
            {
                sum += static_cast<i64>(*a) * static_cast<i64>(*b);
                ++a;
                ++b;
            }
        }

        return sum;
    }

    template<> u64 dot_product(const u32* a, const u32* b, usize count) noexcept
    {
        alignas(64) u64 sum = 0;
        const u32* end = a + count;
        const u32* end256 = a + (count & ~255);

        __m256i acc0 = _mm256_setzero_si256();
        __m256i acc1 = _mm256_setzero_si256();
        __m256i acc2 = _mm256_setzero_si256();
        __m256i acc3 = _mm256_setzero_si256();
        __m256i acc4 = _mm256_setzero_si256();
        __m256i acc5 = _mm256_setzero_si256();
        __m256i acc6 = _mm256_setzero_si256();
        __m256i acc7 = _mm256_setzero_si256();

        while (a < end256)
        {
            _mm_prefetch(reinterpret_cast<const char*>(a) + 512, _MM_HINT_T0);
            _mm_prefetch(reinterpret_cast<const char*>(b) + 512, _MM_HINT_T0);

            for (usize j = 0; j < 8; ++j)
            {
                __m256i a_vec = _mm256_load_si256(reinterpret_cast<const __m256i*>(a));
                __m256i b_vec = _mm256_load_si256(reinterpret_cast<const __m256i*>(b));
                a += 8;
                b += 8;

                __m128i a_low128 = _mm256_castsi256_si128(a_vec);
                __m256i a_low = _mm256_cvtepu32_epi64(a_low128);
                __m128i b_low128 = _mm256_castsi256_si128(b_vec);
                __m256i b_low = _mm256_cvtepu32_epi64(b_low128);
                __m256i prod_low = _mm256_mul_epu32(a_low, b_low);

                __m128i a_high128 = _mm256_extracti128_si256(a_vec, 1);
                __m256i a_high = _mm256_cvtepu32_epi64(a_high128);
                __m128i b_high128 = _mm256_extracti128_si256(b_vec, 1);
                __m256i b_high = _mm256_cvtepu32_epi64(b_high128);
                __m256i prod_high = _mm256_mul_epu32(a_high, b_high);

                switch (j % 4)
                {
                    case 0:
                        acc0 = _mm256_add_epi64(acc0, prod_low);
                        acc1 = _mm256_add_epi64(acc1, prod_high);
                        break;
                    case 1:
                        acc2 = _mm256_add_epi64(acc2, prod_low);
                        acc3 = _mm256_add_epi64(acc3, prod_high);
                        break;
                    case 2:
                        acc4 = _mm256_add_epi64(acc4, prod_low);
                        acc5 = _mm256_add_epi64(acc5, prod_high);
                        break;
                    case 3:
                        acc6 = _mm256_add_epi64(acc6, prod_low);
                        acc7 = _mm256_add_epi64(acc7, prod_high);
                        break;
                }
            }
        }

        acc0 = _mm256_add_epi64(acc0, acc1);
        acc2 = _mm256_add_epi64(acc2, acc3);
        acc4 = _mm256_add_epi64(acc4, acc5);
        acc6 = _mm256_add_epi64(acc6, acc7);
        acc0 = _mm256_add_epi64(acc0, acc2);
        acc4 = _mm256_add_epi64(acc4, acc6);
        __m256i sum256 = _mm256_add_epi64(acc0, acc4);

        __m128i sum_low = _mm256_castsi256_si128(sum256);
        __m128i sum_high = _mm256_extracti128_si256(sum256, 1);
        sum += _mm_extract_epi64(sum_low, 0) + _mm_extract_epi64(sum_low, 1);
        sum += _mm_extract_epi64(sum_high, 0) + _mm_extract_epi64(sum_high, 1);

        if (a < end)
        {
            __m256i tail_acc = _mm256_setzero_si256();
            usize remain = end - a;
            const u32* end32 = a + (remain & ~7);

            while (a < end32)
            {
                __m256i a_vec = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a));
                __m256i b_vec = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b));
                a += 8;
                b += 8;

                __m128i a_low = _mm256_castsi256_si128(a_vec);
                __m256i a_low_ext = _mm256_cvtepu32_epi64(a_low);
                __m128i b_low = _mm256_castsi256_si128(b_vec);
                __m256i b_low_ext = _mm256_cvtepu32_epi64(b_low);
                __m256i prod_low = _mm256_mul_epu32(a_low_ext, b_low_ext);

                __m128i a_high = _mm256_extracti128_si256(a_vec, 1);
                __m256i a_high_ext = _mm256_cvtepu32_epi64(a_high);
                __m128i b_high = _mm256_extracti128_si256(b_vec, 1);
                __m256i b_high_ext = _mm256_cvtepu32_epi64(b_high);
                __m256i prod_high = _mm256_mul_epu32(a_high_ext, b_high_ext);

                tail_acc = _mm256_add_epi64(tail_acc, prod_low);
                tail_acc = _mm256_add_epi64(tail_acc, prod_high);
            }

            __m128i t_low = _mm256_castsi256_si128(tail_acc);
            __m128i t_high = _mm256_extracti128_si256(tail_acc, 1);
            sum += _mm_extract_epi64(t_low, 0) + _mm_extract_epi64(t_low, 1);
            sum += _mm_extract_epi64(t_high, 0) + _mm_extract_epi64(t_high, 1);

            while (a < end)
            {
                sum += static_cast<u64>(*a) * static_cast<u64>(*b);
                ++a;
                ++b;
            }
        }

        return sum;
    }

    template<> i64 dot_product(const i64* a, const i64* b, usize count) noexcept
    {
        alignas(64) i64 sum = 0;
        const i64* end = a + count;
        const i64* end256 = a + (count & ~255);

        i64 acc0 = 0;
        i64 acc1 = 0;
        i64 acc2 = 0;
        i64 acc3 = 0;
        i64 acc4 = 0;
        i64 acc5 = 0;
        i64 acc6 = 0;
        i64 acc7 = 0;

        while (a < end256)
        {
            _mm_prefetch(reinterpret_cast<const char*>(a) + 512, _MM_HINT_T0);
            _mm_prefetch(reinterpret_cast<const char*>(b) + 512, _MM_HINT_T0);

            for (usize j = 0; j < 32; ++j)
            {
                acc0 += a[0] * b[0];
                acc1 += a[1] * b[1];
                acc2 += a[2] * b[2];
                acc3 += a[3] * b[3];
                acc4 += a[4] * b[4];
                acc5 += a[5] * b[5];
                acc6 += a[6] * b[6];
                acc7 += a[7] * b[7];
                a += 8;
                b += 8;
            }
        }

        sum = acc0 + acc1 + acc2 + acc3 + acc4 + acc5 + acc6 + acc7;

        if (a < end)
        {
            i64 tail_acc = 0;
            usize remain = end - a;
            const i64* end8 = a + (remain & ~7);

            while (a < end8)
            {
                tail_acc += a[0] * b[0];
                tail_acc += a[1] * b[1];
                tail_acc += a[2] * b[2];
                tail_acc += a[3] * b[3];
                tail_acc += a[4] * b[4];
                tail_acc += a[5] * b[5];
                tail_acc += a[6] * b[6];
                tail_acc += a[7] * b[7];
                a += 8;
                b += 8;
            }

            while (a < end)
            {
                tail_acc += *a * *b;
                ++a;
                ++b;
            }

            sum += tail_acc;
        }

        return sum;
    }


    template<> u64 dot_product(const u64* a, const u64* b, usize count) noexcept
    {
        alignas(64) u64 sum = 0;
        const u64* end = a + count;
        const u64* end256 = a + (count & ~255);

        u64 acc0 = 0;
        u64 acc1 = 0;
        u64 acc2 = 0;
        u64 acc3 = 0;
        u64 acc4 = 0;
        u64 acc5 = 0;
        u64 acc6 = 0;
        u64 acc7 = 0;

        while (a < end256)
        {
            _mm_prefetch(reinterpret_cast<const char*>(a) + 512, _MM_HINT_T0);
            _mm_prefetch(reinterpret_cast<const char*>(b) + 512, _MM_HINT_T0);

            for (usize j = 0; j < 32; ++j)
            {
                acc0 += a[0] * b[0];
                acc1 += a[1] * b[1];
                acc2 += a[2] * b[2];
                acc3 += a[3] * b[3];
                acc4 += a[4] * b[4];
                acc5 += a[5] * b[5];
                acc6 += a[6] * b[6];
                acc7 += a[7] * b[7];
                a += 8;
                b += 8;
            }
        }

        sum = acc0 + acc1 + acc2 + acc3 + acc4 + acc5 + acc6 + acc7;

        if (a < end)
        {
            u64 tail_acc = 0;
            usize remain = end - a;
            const u64* end8 = a + (remain & ~7);

            while (a < end8)
            {
                tail_acc += a[0] * b[0];
                tail_acc += a[1] * b[1];
                tail_acc += a[2] * b[2];
                tail_acc += a[3] * b[3];
                tail_acc += a[4] * b[4];
                tail_acc += a[5] * b[5];
                tail_acc += a[6] * b[6];
                tail_acc += a[7] * b[7];
                a += 8;
                b += 8;
            }

            while (a < end)
            {
                tail_acc += *a * *b;
                ++a;
                ++b;
            }

            sum += tail_acc;
        }

        return sum;
    }
}