module;
#include <immintrin.h>
#include <intrin.h>
module foye.algorithm;
import foye.foye_core;
import foye.simd;
import std;

#pragma warning(disable: 4309)
#pragma warning(disable: 4244)

namespace fy
{
	template<BasicArithmetic Element_t>
	extended_t<Element_t> sum(const Element_t* src, usize count) noexcept
	{
		std::unreachable();
	}

    template<BasicArithmetic Element_t>
    static extended_t<Element_t> summation_8bits__(const Element_t* src, usize count) noexcept
    {
        constexpr usize max_safe_chunks = 8;
        const Element_t* ptr = src;
        usize remaining = count;
        extended_t<Element_t> total = 0;

        const uintptr_t align_mask = 31;
        const uintptr_t misalignment = reinterpret_cast<uintptr_t>(ptr) & align_mask;
        if (misalignment != 0)
        {
            const usize align_skip = 32 - misalignment;
            const usize process = std::min(align_skip, remaining);
            for (usize i = 0; i < process; ++i)
            {
                total += static_cast<extended_t<Element_t>>(ptr[i]);
            }
            ptr += process;
            remaining -= process;
        }

        alignas(64) __m256i acc[4] = {
            _mm256_setzero_si256(),
            _mm256_setzero_si256(),
            _mm256_setzero_si256(),
            _mm256_setzero_si256()
        };

        usize chunks = 0;

        while (remaining >= 128)
        {
            _mm_prefetch(reinterpret_cast<const char*>(ptr) + 512, _MM_HINT_T0);

            const __m256i v0 = _mm256_load_si256(reinterpret_cast<const __m256i*>(ptr));
            const __m256i v1 = _mm256_load_si256(reinterpret_cast<const __m256i*>(ptr + 32));
            const __m256i v2 = _mm256_load_si256(reinterpret_cast<const __m256i*>(ptr + 64));
            const __m256i v3 = _mm256_load_si256(reinterpret_cast<const __m256i*>(ptr + 96));

            if constexpr (std::is_unsigned_v<Element_t>)
            {
                acc[0] = _mm256_add_epi16(acc[0], _mm256_cvtepu8_epi16(_mm256_castsi256_si128(v0)));
                acc[1] = _mm256_add_epi16(acc[1], _mm256_cvtepu8_epi16(_mm256_extracti128_si256(v0, 1)));
                acc[2] = _mm256_add_epi16(acc[2], _mm256_cvtepu8_epi16(_mm256_castsi256_si128(v1)));
                acc[3] = _mm256_add_epi16(acc[3], _mm256_cvtepu8_epi16(_mm256_extracti128_si256(v1, 1)));

                acc[0] = _mm256_add_epi16(acc[0], _mm256_cvtepu8_epi16(_mm256_castsi256_si128(v2)));
                acc[1] = _mm256_add_epi16(acc[1], _mm256_cvtepu8_epi16(_mm256_extracti128_si256(v2, 1)));
                acc[2] = _mm256_add_epi16(acc[2], _mm256_cvtepu8_epi16(_mm256_castsi256_si128(v3)));
                acc[3] = _mm256_add_epi16(acc[3], _mm256_cvtepu8_epi16(_mm256_extracti128_si256(v3, 1)));
            }
            else
            {
                acc[0] = _mm256_add_epi16(acc[0], _mm256_cvtepi8_epi16(_mm256_castsi256_si128(v0)));
                acc[1] = _mm256_add_epi16(acc[1], _mm256_cvtepi8_epi16(_mm256_extracti128_si256(v0, 1)));
                acc[2] = _mm256_add_epi16(acc[2], _mm256_cvtepi8_epi16(_mm256_castsi256_si128(v1)));
                acc[3] = _mm256_add_epi16(acc[3], _mm256_cvtepi8_epi16(_mm256_extracti128_si256(v1, 1)));

                acc[0] = _mm256_add_epi16(acc[0], _mm256_cvtepi8_epi16(_mm256_castsi256_si128(v2)));
                acc[1] = _mm256_add_epi16(acc[1], _mm256_cvtepi8_epi16(_mm256_extracti128_si256(v2, 1)));
                acc[2] = _mm256_add_epi16(acc[2], _mm256_cvtepi8_epi16(_mm256_castsi256_si128(v3)));
                acc[3] = _mm256_add_epi16(acc[3], _mm256_cvtepi8_epi16(_mm256_extracti128_si256(v3, 1)));
            }

            ptr += 128;
            remaining -= 128;
            chunks += 4;

            if (chunks >= max_safe_chunks)
            {
                for (__m256i& vec : acc)
                {
                    const __m256i ones = _mm256_set1_epi16(1);
                    const __m256i sum32 = _mm256_madd_epi16(vec, ones);

                    __m128i sum128 = _mm_add_epi32(
                        _mm256_castsi256_si128(sum32),
                        _mm256_extracti128_si256(sum32, 1)
                    );
                    sum128 = _mm_add_epi32(sum128, _mm_srli_si128(sum128, 8));
                    sum128 = _mm_add_epi32(sum128, _mm_srli_si128(sum128, 4));
                    total += _mm_cvtsi128_si32(sum128);
                }

                acc[0] = acc[1] = acc[2] = acc[3] = _mm256_setzero_si256();
                chunks = 0;
            }
        }

        while (remaining >= 32)
        {
            const __m256i v = _mm256_load_si256(reinterpret_cast<const __m256i*>(ptr));

            if constexpr (std::is_unsigned_v<Element_t>)
            {
                acc[0] = _mm256_add_epi16(acc[0], _mm256_cvtepu8_epi16(_mm256_castsi256_si128(v)));
                acc[1] = _mm256_add_epi16(acc[1], _mm256_cvtepu8_epi16(_mm256_extracti128_si256(v, 1)));
            }
            else
            {
                acc[0] = _mm256_add_epi16(acc[0], _mm256_cvtepi8_epi16(_mm256_castsi256_si128(v)));
                acc[1] = _mm256_add_epi16(acc[1], _mm256_cvtepi8_epi16(_mm256_extracti128_si256(v, 1)));
            }

            ptr += 32;
            remaining -= 32;
            chunks++;

            if (chunks >= max_safe_chunks)
            {
                for (usize pacc = 0; pacc < 2; ++pacc)
                {
                    const __m256i ones = _mm256_set1_epi16(1);
                    const __m256i sum32 = _mm256_madd_epi16(acc[pacc], ones);

                    __m128i sum128 = _mm_add_epi32(
                        _mm256_castsi256_si128(sum32),
                        _mm256_extracti128_si256(sum32, 1)
                    );
                    sum128 = _mm_add_epi32(sum128, _mm_srli_si128(sum128, 8));
                    sum128 = _mm_add_epi32(sum128, _mm_srli_si128(sum128, 4));
                    total += _mm_cvtsi128_si32(sum128);
                }

                acc[0] = acc[1] = _mm256_setzero_si256();
                chunks = 0;
            }
        }

        const __m256i ones = _mm256_set1_epi16(1);
        for (const __m256i& vec : acc)
        {
            const __m256i sum32 = _mm256_madd_epi16(vec, ones);
            __m128i sum128 = _mm_add_epi32(
                _mm256_castsi256_si128(sum32),
                _mm256_extracti128_si256(sum32, 1)
            );
            sum128 = _mm_add_epi32(sum128, _mm_srli_si128(sum128, 8));
            sum128 = _mm_add_epi32(sum128, _mm_srli_si128(sum128, 4));
            total += _mm_cvtsi128_si32(sum128);
        }

        constexpr usize tail_sizes[5] = { 16, 8, 4, 2, 1 };
        for (usize size : tail_sizes)
        {
            if (remaining >= static_cast<usize>(size))
            {
                for (usize i = 0; i < size; ++i)
                {
                    total += static_cast<extended_t<Element_t>>(ptr[i]);
                }
                ptr += size;
                remaining -= size;
            }
        }

        _mm256_zeroupper();
        return total;
    }

    template<BasicArithmetic Element_t>
    static extended_t<Element_t> summation_16bits__(const Element_t* src, usize count) noexcept
    {
        const Element_t* ptr = src;
        usize remaining = count;
        extended_t<Element_t> total = 0;

        const uintptr_t align_mask = 31;
        const uintptr_t misalignment = reinterpret_cast<uintptr_t>(ptr) & align_mask;
        if (misalignment != 0)
        {
            const usize align_skip_bytes = 32 - misalignment;
            const usize align_skip_elements = align_skip_bytes / sizeof(Element_t);
            const usize process = std::min(align_skip_elements, remaining);
            for (usize i = 0; i < process; ++i)
            {
                total += static_cast<extended_t<Element_t>>(ptr[i]);
            }
            ptr += process;
            remaining -= process;
        }

        alignas(64) __m256i acc[4] = {
            _mm256_setzero_si256(),
            _mm256_setzero_si256(),
            _mm256_setzero_si256(),
            _mm256_setzero_si256()
        };

        usize chunks = 0;
        constexpr usize max_safe_chunks = 4096;

        while (remaining >= 64)
        {
            _mm_prefetch(reinterpret_cast<const char*>(ptr) + 512, _MM_HINT_T0);

            const __m256i v0 = _mm256_load_si256(reinterpret_cast<const __m256i*>(ptr));
            const __m256i v1 = _mm256_load_si256(reinterpret_cast<const __m256i*>(ptr + 16));
            const __m256i v2 = _mm256_load_si256(reinterpret_cast<const __m256i*>(ptr + 32));
            const __m256i v3 = _mm256_load_si256(reinterpret_cast<const __m256i*>(ptr + 48));

            {
                __m128i low = _mm256_extracti128_si256(v0, 0);
                __m128i high = _mm256_extracti128_si256(v0, 1);
                if constexpr (std::is_unsigned_v<Element_t>)
                {
                    acc[0] = _mm256_add_epi32(acc[0], _mm256_cvtepu16_epi32(low));
                    acc[1] = _mm256_add_epi32(acc[1], _mm256_cvtepu16_epi32(high));
                }
                else
                {
                    acc[0] = _mm256_add_epi32(acc[0], _mm256_cvtepi16_epi32(low));
                    acc[1] = _mm256_add_epi32(acc[1], _mm256_cvtepi16_epi32(high));
                }
            }

            {
                __m128i low = _mm256_extracti128_si256(v1, 0);
                __m128i high = _mm256_extracti128_si256(v1, 1);
                if constexpr (std::is_unsigned_v<Element_t>)
                {
                    acc[2] = _mm256_add_epi32(acc[2], _mm256_cvtepu16_epi32(low));
                    acc[3] = _mm256_add_epi32(acc[3], _mm256_cvtepu16_epi32(high));
                }
                else
                {
                    acc[2] = _mm256_add_epi32(acc[2], _mm256_cvtepi16_epi32(low));
                    acc[3] = _mm256_add_epi32(acc[3], _mm256_cvtepi16_epi32(high));
                }
            }

            {
                __m128i low = _mm256_extracti128_si256(v2, 0);
                __m128i high = _mm256_extracti128_si256(v2, 1);
                if constexpr (std::is_unsigned_v<Element_t>)
                {
                    acc[0] = _mm256_add_epi32(acc[0], _mm256_cvtepu16_epi32(low));
                    acc[1] = _mm256_add_epi32(acc[1], _mm256_cvtepu16_epi32(high));
                }
                else
                {
                    acc[0] = _mm256_add_epi32(acc[0], _mm256_cvtepi16_epi32(low));
                    acc[1] = _mm256_add_epi32(acc[1], _mm256_cvtepi16_epi32(high));
                }
            }

            {
                __m128i low = _mm256_extracti128_si256(v3, 0);
                __m128i high = _mm256_extracti128_si256(v3, 1);
                if constexpr (std::is_unsigned_v<Element_t>)
                {
                    acc[2] = _mm256_add_epi32(acc[2], _mm256_cvtepu16_epi32(low));
                    acc[3] = _mm256_add_epi32(acc[3], _mm256_cvtepu16_epi32(high));
                }
                else
                {
                    acc[2] = _mm256_add_epi32(acc[2], _mm256_cvtepi16_epi32(low));
                    acc[3] = _mm256_add_epi32(acc[3], _mm256_cvtepi16_epi32(high));
                }
            }

            ptr += 64;
            remaining -= 64;
            chunks += 4;

            if (chunks >= max_safe_chunks)
            {
                for (__m256i& vec : acc)
                {
                    __m128i low = _mm256_castsi256_si128(vec);
                    __m128i high = _mm256_extracti128_si256(vec, 1);
                    __m128i sum = _mm_add_epi32(low, high);
                    sum = _mm_hadd_epi32(sum, sum);
                    sum = _mm_hadd_epi32(sum, sum);
                    total += _mm_extract_epi32(sum, 0);
                    vec = _mm256_setzero_si256();
                }
                chunks = 0;
            }
        }

        while (remaining >= 16)
        {
            const __m256i v = _mm256_load_si256(reinterpret_cast<const __m256i*>(ptr));

            __m128i low = _mm256_extracti128_si256(v, 0);
            __m128i high = _mm256_extracti128_si256(v, 1);
            __m256i low_ext, high_ext;
            
            if constexpr (std::is_unsigned_v<Element_t>)
            {
                low_ext = _mm256_cvtepu16_epi32(low);
                high_ext = _mm256_cvtepu16_epi32(high);
            }
            else
            {
                low_ext = _mm256_cvtepi16_epi32(low);
                high_ext = _mm256_cvtepi16_epi32(high);
            }

            acc[0] = _mm256_add_epi32(acc[0], low_ext);
            acc[1] = _mm256_add_epi32(acc[1], high_ext);

            ptr += 16;
            remaining -= 16;
            chunks += 1;

            if (chunks >= max_safe_chunks)
            {
                for (usize i = 0; i < 2; ++i)
                {
                    __m128i low = _mm256_castsi256_si128(acc[i]);
                    __m128i high = _mm256_extracti128_si256(acc[i], 1);
                    __m128i sum = _mm_add_epi32(low, high);
                    sum = _mm_hadd_epi32(sum, sum);
                    sum = _mm_hadd_epi32(sum, sum);
                    total += _mm_extract_epi32(sum, 0);
                    acc[i] = _mm256_setzero_si256();
                }
                chunks = 0;
            }
        }

        for (const __m256i& vec : acc)
        {
            __m128i low = _mm256_castsi256_si128(vec);
            __m128i high = _mm256_extracti128_si256(vec, 1);
            __m128i sum = _mm_add_epi32(low, high);
            sum = _mm_hadd_epi32(sum, sum);
            sum = _mm_hadd_epi32(sum, sum);
            total += _mm_extract_epi32(sum, 0);
        }

        constexpr usize tail_sizes[] = { 8, 4, 2, 1 };
        for (usize size : tail_sizes)
        {
            if (remaining >= size)
            {
                for (usize i = 0; i < size; ++i)
                {
                    total += static_cast<extended_t<Element_t>>(ptr[i]);
                }
                ptr += size;
                remaining -= size;
            }
        }

        _mm256_zeroupper();
        return total;
    }

    template<BasicArithmetic Element_t>
    static extended_t<Element_t> summation_32bits__(const Element_t* src, usize count) noexcept
    {
        const Element_t* ptr = src;
        usize remaining = count;
        extended_t<Element_t> total = 0;

        const uintptr_t align_mask = 31;
        const uintptr_t misalignment = reinterpret_cast<uintptr_t>(ptr) & align_mask;
        if (misalignment != 0)
        {
            const usize align_skip_bytes = 32 - misalignment;
            const usize align_skip_elements = align_skip_bytes / sizeof(Element_t);
            const usize process = std::min(align_skip_elements, remaining);
            for (usize i = 0; i < process; ++i)
            {
                total += static_cast<extended_t<Element_t>>(ptr[i]);
            }
            ptr += process;
            remaining -= process;
        }

        alignas(32) __m256i acc[4] = {
            _mm256_setzero_si256(),
            _mm256_setzero_si256(),
            _mm256_setzero_si256(),
            _mm256_setzero_si256()
        };

        usize chunks = 0;
        constexpr usize max_safe_chunks = 4096;

        while (remaining >= 16)
        {
            _mm_prefetch(reinterpret_cast<const char*>(ptr) + 512, _MM_HINT_T0);

            const __m256i v0 = _mm256_load_si256(reinterpret_cast<const __m256i*>(ptr));
            const __m256i v1 = _mm256_load_si256(reinterpret_cast<const __m256i*>(ptr + 8));

            __m128i v0_low = _mm256_castsi256_si128(v0);
            __m128i v0_high = _mm256_extracti128_si256(v0, 1);
            __m256i v0_low64, v0_high64;
            if constexpr (std::is_unsigned_v<Element_t>)
            {
                v0_low64 = _mm256_cvtepu32_epi64(v0_low);
                v0_high64 = _mm256_cvtepu32_epi64(v0_high);
            }
            else
            {
                v0_low64 = _mm256_cvtepi32_epi64(v0_low);
                v0_high64 = _mm256_cvtepi32_epi64(v0_high);
            }

            __m128i v1_low = _mm256_castsi256_si128(v1);
            __m128i v1_high = _mm256_extracti128_si256(v1, 1);
            __m256i v1_low64, v1_high64;
            if constexpr (std::is_unsigned_v<Element_t>)
            {
                v1_low64 = _mm256_cvtepu32_epi64(v1_low);
                v1_high64 = _mm256_cvtepu32_epi64(v1_high);
            }
            else
            {
                v1_low64 = _mm256_cvtepi32_epi64(v1_low);
                v1_high64 = _mm256_cvtepi32_epi64(v1_high);
            }

            acc[0] = _mm256_add_epi64(acc[0], v0_low64);
            acc[1] = _mm256_add_epi64(acc[1], v0_high64);
            acc[2] = _mm256_add_epi64(acc[2], v1_low64);
            acc[3] = _mm256_add_epi64(acc[3], v1_high64);

            ptr += 16;
            remaining -= 16;
            chunks += 2;

            if (chunks >= max_safe_chunks)
            {
                for (__m256i& a : acc)
                {
                    __m128i low = _mm256_castsi256_si128(a);
                    __m128i high = _mm256_extracti128_si256(a, 1);
                    __m128i sum = _mm_add_epi64(low, high);
                    sum = _mm_add_epi64(sum, _mm_unpackhi_epi64(sum, sum));
                    total += _mm_extract_epi64(sum, 0);
                    a = _mm256_setzero_si256();
                }
                chunks = 0;
            }
        }

        while (remaining >= 8)
        {
            const __m256i v = _mm256_load_si256(reinterpret_cast<const __m256i*>(ptr));

            __m128i v_low = _mm256_castsi256_si128(v);
            __m128i v_high = _mm256_extracti128_si256(v, 1);
            __m256i v_low64, v_high64;
            if constexpr (std::is_unsigned_v<Element_t>)
            {
                v_low64 = _mm256_cvtepu32_epi64(v_low);
                v_high64 = _mm256_cvtepu32_epi64(v_high);
            }
            else
            {
                v_low64 = _mm256_cvtepi32_epi64(v_low);
                v_high64 = _mm256_cvtepi32_epi64(v_high);
            }

            acc[0] = _mm256_add_epi64(acc[0], v_low64);
            acc[1] = _mm256_add_epi64(acc[1], v_high64);

            ptr += 8;
            remaining -= 8;
            chunks += 1;

            if (chunks >= max_safe_chunks)
            {
                for (__m256i& a : acc)
                {
                    __m128i low = _mm256_castsi256_si128(a);
                    __m128i high = _mm256_extracti128_si256(a, 1);
                    __m128i sum = _mm_add_epi64(low, high);
                    sum = _mm_add_epi64(sum, _mm_unpackhi_epi64(sum, sum));
                    total += _mm_extract_epi64(sum, 0);
                    a = _mm256_setzero_si256();
                }
                chunks = 0;
            }
        }

        for (const __m256i& a : acc)
        {
            __m128i low = _mm256_castsi256_si128(a);
            __m128i high = _mm256_extracti128_si256(a, 1);
            __m128i sum = _mm_add_epi64(low, high);
            sum = _mm_add_epi64(sum, _mm_unpackhi_epi64(sum, sum));
            total += _mm_extract_epi64(sum, 0);
        }

        constexpr usize tail_sizes[] = { 4, 2, 1 };
        for (usize size : tail_sizes)
        {
            if (remaining >= size)
            {
                for (usize i = 0; i < size; ++i)
                {
                    total += static_cast<extended_t<Element_t>>(ptr[i]);
                }
                ptr += size;
                remaining -= size;
            }
        }

        _mm256_zeroupper();
        return total;
    }

    template<> i64 sum<i8>(const i8* src, usize count) noexcept
    {
        return summation_8bits__<i8>(src, count);
    }

    template<> u64 sum<u8>(const u8* src, usize count) noexcept
    {
        return summation_8bits__<u8>(src, count);
    }

    template<> i64 sum<i16>(const i16* src, usize count) noexcept
    {
        return summation_16bits__<i16>(src, count);
    }

    template<> u64 sum<u16>(const u16* src, usize count) noexcept
    {
        return summation_16bits__<u16>(src, count);
    }

    template<> i64 sum<i32>(const i32* src, usize count) noexcept
    {
        return summation_32bits__<i32>(src, count);
    }

    template<> u64 sum<u32>(const u32* src, usize count) noexcept
    {
        return summation_32bits__<u32>(src, count);
    }

    template<BasicArithmetic Element_t>
    static extended_t<Element_t> summation_64bits__(const Element_t* src, usize count) noexcept
    {
        extended_t<Element_t> res{ 0 };
        for (usize i = 0; i < count; ++i)
        {
            res += src[i];
        }

        return res;
    }

    template<> i64 sum<i64>(const i64* src, usize count) noexcept
    {
        return summation_64bits__<i64>(src, count);
    }

    template<> u64 sum<u64>(const u64* src, usize count) noexcept
    {
        return summation_64bits__<u64>(src, count);
    }

    template<> f64 sum<f16>(const f16* src, usize count) noexcept
    {
        const f16* ptr = src;
        usize remaining = count;
        f32 total = 0.0f;

        const uintptr_t align_mask = 31;
        const uintptr_t misalignment = reinterpret_cast<uintptr_t>(ptr) & align_mask;
        if (misalignment != 0)
        {
            const usize align_skip_bytes = 32 - misalignment;
            const usize align_skip_elements = align_skip_bytes / sizeof(f16);
            const usize process = std::min(align_skip_elements, remaining);
            for (usize i = 0; i < process; ++i)
            {
                total += static_cast<f32>(ptr[i]);
            }
            ptr += process;
            remaining -= process;
        }

        const u16* simd_ptr = reinterpret_cast<const u16*>(ptr);

        alignas(32) __m256 acc[4] = {
            _mm256_setzero_ps(),
            _mm256_setzero_ps(),
            _mm256_setzero_ps(),
            _mm256_setzero_ps()
        };
        
        constexpr usize elements_per_chunk = 32;
        constexpr usize max_safe_chunks = 4096;
        usize chunks = 0;

        while (remaining >= elements_per_chunk)
        {
            _mm_prefetch(reinterpret_cast<const char*>(simd_ptr) + 512, _MM_HINT_T0);

            const __m128i pack0 = _mm_load_si128(reinterpret_cast<const __m128i*>(simd_ptr));
            const __m128i pack1 = _mm_load_si128(reinterpret_cast<const __m128i*>(simd_ptr + 8));
            const __m128i pack2 = _mm_load_si128(reinterpret_cast<const __m128i*>(simd_ptr + 16));
            const __m128i pack3 = _mm_load_si128(reinterpret_cast<const __m128i*>(simd_ptr + 24));

            acc[0] = _mm256_add_ps(acc[0], _mm256_cvtph_ps(pack0));
            acc[1] = _mm256_add_ps(acc[1], _mm256_cvtph_ps(pack1));
            acc[2] = _mm256_add_ps(acc[2], _mm256_cvtph_ps(pack2));
            acc[3] = _mm256_add_ps(acc[3], _mm256_cvtph_ps(pack3));

            simd_ptr += elements_per_chunk;
            remaining -= elements_per_chunk;
            chunks += 4;

            if (chunks >= max_safe_chunks)
            {
                alignas(32) f32 buffer[8];
                __m256 sum = _mm256_add_ps(
                    _mm256_add_ps(acc[0], acc[1]),
                    _mm256_add_ps(acc[2], acc[3]));

                _mm256_store_ps(buffer, sum);
                for (int i = 0; i < 8; ++i) total += buffer[i];
                acc[0] = acc[1] = acc[2] = acc[3] = _mm256_setzero_ps();
                chunks = 0;
            }
        }

        while (remaining >= 16)
        {
            const __m128i pack = _mm_load_si128(reinterpret_cast<const __m128i*>(simd_ptr));
            acc[0] = _mm256_add_ps(acc[0], _mm256_cvtph_ps(pack));
            simd_ptr += 16;
            remaining -= 16;
            chunks += 1;

            if (chunks >= max_safe_chunks)
            {
                alignas(32) f32 buffer[8];
                _mm256_store_ps(buffer, acc[0]);
                for (int i = 0; i < 8; ++i) total += buffer[i];
                acc[0] = _mm256_setzero_ps();
                chunks = 0;
            }
        }

        alignas(32) f32 buffer[8];
        __m256 sum = _mm256_add_ps(
            _mm256_add_ps(acc[0], acc[1]),
            _mm256_add_ps(acc[2], acc[3]));

        _mm256_store_ps(buffer, sum);
        for (usize i = 0; i < 8; ++i)
        {
            total += buffer[i];
        }

        const f16* tail_ptr = reinterpret_cast<const f16*>(simd_ptr);
        for (usize i = 0; i < remaining; ++i)
        {
            total += static_cast<f32>(tail_ptr[i]);
        }

        _mm256_zeroupper();
        return static_cast<f64>(total);
    }
    template<> f64 sum<f32>(const f32* src, usize count) noexcept
    {
        const f32* ptr = src;
        usize remaining = count;
        f64 total = 0.0;

        const uintptr_t align_mask = 31;
        const uintptr_t misalignment = reinterpret_cast<uintptr_t>(ptr) & align_mask;
        if (misalignment != 0)
        {
            const usize align_skip_bytes = 32 - misalignment;
            const usize align_skip_elements = align_skip_bytes / sizeof(f32);
            const usize process = std::min(align_skip_elements, remaining);
            for (usize i = 0; i < process; ++i)
            {
                total += static_cast<f64>(ptr[i]);
            }
            ptr += process;
            remaining -= process;
        }

        alignas(64) __m256d acc[4] = {
            _mm256_setzero_pd(),
            _mm256_setzero_pd(),
            _mm256_setzero_pd(),
            _mm256_setzero_pd()
        };

        usize chunks = 0;
        constexpr usize max_safe_chunks = 4096;

        while (remaining >= 32)
        {
            _mm_prefetch(reinterpret_cast<const char*>(ptr) + 512, _MM_HINT_T0);

            const __m256 v0 = _mm256_load_ps(ptr);
            const __m256 v1 = _mm256_load_ps(ptr + 8);
            const __m256 v2 = _mm256_load_ps(ptr + 16);
            const __m256 v3 = _mm256_load_ps(ptr + 24);

            acc[0] = _mm256_add_pd(acc[0], _mm256_cvtps_pd(_mm256_castps256_ps128(v0)));
            acc[1] = _mm256_add_pd(acc[1], _mm256_cvtps_pd(_mm256_extractf128_ps(v0, 1)));
            acc[2] = _mm256_add_pd(acc[2], _mm256_cvtps_pd(_mm256_castps256_ps128(v1)));
            acc[3] = _mm256_add_pd(acc[3], _mm256_cvtps_pd(_mm256_extractf128_ps(v1, 1)));
            acc[0] = _mm256_add_pd(acc[0], _mm256_cvtps_pd(_mm256_castps256_ps128(v2)));
            acc[1] = _mm256_add_pd(acc[1], _mm256_cvtps_pd(_mm256_extractf128_ps(v2, 1)));
            acc[2] = _mm256_add_pd(acc[2], _mm256_cvtps_pd(_mm256_castps256_ps128(v3)));
            acc[3] = _mm256_add_pd(acc[3], _mm256_cvtps_pd(_mm256_extractf128_ps(v3, 1)));

            ptr += 32;
            remaining -= 32;
            chunks += 4;

            if (chunks >= max_safe_chunks)
            {
                __m256d sum01 = _mm256_add_pd(acc[0], acc[1]);
                __m256d sum23 = _mm256_add_pd(acc[2], acc[3]);
                __m256d sum = _mm256_add_pd(sum01, sum23);

                __m128d low = _mm256_castpd256_pd128(sum);
                __m128d high = _mm256_extractf128_pd(sum, 1);
                low = _mm_add_pd(low, high);
                low = _mm_hadd_pd(low, low);
                total += _mm_cvtsd_f64(low);

                acc[0] = acc[1] = acc[2] = acc[3] = _mm256_setzero_pd();
                chunks = 0;
            }
        }

        while (remaining >= 8)
        {
            const __m256 v = _mm256_load_ps(ptr);
            acc[0] = _mm256_add_pd(acc[0], _mm256_cvtps_pd(_mm256_castps256_ps128(v)));
            acc[1] = _mm256_add_pd(acc[1], _mm256_cvtps_pd(_mm256_extractf128_ps(v, 1)));
            ptr += 8;
            remaining -= 8;
            chunks += 1;

            if (chunks >= max_safe_chunks)
            {
                __m256d sum = _mm256_add_pd(acc[0], acc[1]);
                __m128d low = _mm256_castpd256_pd128(sum);
                __m128d high = _mm256_extractf128_pd(sum, 1);
                low = _mm_add_pd(low, high);
                low = _mm_hadd_pd(low, low);
                total += _mm_cvtsd_f64(low);

                acc[0] = acc[1] = _mm256_setzero_pd();
                chunks = 0;
            }
        }

        __m256d sum01 = _mm256_add_pd(acc[0], acc[1]);
        __m256d sum23 = _mm256_add_pd(acc[2], acc[3]);
        __m256d sum = _mm256_add_pd(sum01, sum23);
        __m128d low = _mm256_castpd256_pd128(sum);
        __m128d high = _mm256_extractf128_pd(sum, 1);
        low = _mm_add_pd(low, high);
        low = _mm_hadd_pd(low, low);
        total += _mm_cvtsd_f64(low);

        constexpr usize tail_sizes[] = { 4, 2, 1 };
        for (usize size : tail_sizes)
        {
            if (remaining >= size)
            {
                if (size == 4)
                {
                    __m128 v = _mm_loadu_ps(ptr);
                    __m256d vd = _mm256_cvtps_pd(v);
                    __m128d low = _mm256_castpd256_pd128(vd);
                    __m128d high = _mm256_extractf128_pd(vd, 1);
                    low = _mm_add_pd(low, high);
                    total += _mm_cvtsd_f64(low) + _mm_cvtsd_f64(_mm_unpackhi_pd(low, low));
                }
                else
                {
                    for (usize i = 0; i < size; ++i)
                    {
                        total += static_cast<f64>(ptr[i]);
                    }
                }
                ptr += size;
                remaining -= size;
            }
        }

        _mm256_zeroupper();
        return total;
    }

    template<> f64 sum<f64>(const f64* src, usize count) noexcept
    {
        const f64* ptr = src;
        usize remaining = count;
        f64 total = 0.0;

        const uintptr_t align_mask = 31;
        const uintptr_t misalignment = reinterpret_cast<uintptr_t>(ptr) & align_mask;
        if (misalignment != 0)
        {
            const usize align_skip_bytes = 32 - misalignment;
            const usize align_skip_elements = align_skip_bytes / sizeof(f64);
            const usize process = std::min(align_skip_elements, remaining);
            for (usize i = 0; i < process; ++i)
            {
                total += ptr[i];
            }
            ptr += process;
            remaining -= process;
        }

        alignas(32) __m256d acc[4] = {
            _mm256_setzero_pd(),
            _mm256_setzero_pd(),
            _mm256_setzero_pd(),
            _mm256_setzero_pd()
        };

        usize chunks = 0;
        constexpr usize max_safe_chunks = 4096;

        while (remaining >= 16)
        {
            _mm_prefetch(reinterpret_cast<const char*>(ptr) + 512, _MM_HINT_T0);

            const __m256d v0 = _mm256_load_pd(ptr);
            const __m256d v1 = _mm256_load_pd(ptr + 4);
            const __m256d v2 = _mm256_load_pd(ptr + 8);
            const __m256d v3 = _mm256_load_pd(ptr + 12);

            acc[0] = _mm256_add_pd(acc[0], v0);
            acc[1] = _mm256_add_pd(acc[1], v1);
            acc[2] = _mm256_add_pd(acc[2], v2);
            acc[3] = _mm256_add_pd(acc[3], v3);

            ptr += 16;
            remaining -= 16;
            chunks += 4;

            if (chunks >= max_safe_chunks)
            {
                __m256d sum01 = _mm256_add_pd(acc[0], acc[1]);
                __m256d sum23 = _mm256_add_pd(acc[2], acc[3]);
                __m256d sum = _mm256_add_pd(sum01, sum23);

                __m128d low = _mm256_castpd256_pd128(sum);
                __m128d high = _mm256_extractf128_pd(sum, 1);
                low = _mm_add_pd(low, high);
                low = _mm_hadd_pd(low, low);
                total += _mm_cvtsd_f64(low);

                acc[0] = acc[1] = acc[2] = acc[3] = _mm256_setzero_pd();
                chunks = 0;
            }
        }

        while (remaining >= 8)
        {
            const __m256d v0 = _mm256_load_pd(ptr);
            const __m256d v1 = _mm256_load_pd(ptr + 4);
            acc[0] = _mm256_add_pd(acc[0], v0);
            acc[1] = _mm256_add_pd(acc[1], v1);
            ptr += 8;
            remaining -= 8;
            chunks += 2;

            if (chunks >= max_safe_chunks)
            {
                __m256d sum = _mm256_add_pd(acc[0], acc[1]);
                __m128d low = _mm256_castpd256_pd128(sum);
                __m128d high = _mm256_extractf128_pd(sum, 1);
                low = _mm_add_pd(low, high);
                low = _mm_hadd_pd(low, low);
                total += _mm_cvtsd_f64(low);

                acc[0] = acc[1] = _mm256_setzero_pd();
                chunks = 0;
            }
        }

        __m256d sum01 = _mm256_add_pd(acc[0], acc[1]);
        __m256d sum23 = _mm256_add_pd(acc[2], acc[3]);
        __m256d sum = _mm256_add_pd(sum01, sum23);
        __m128d low = _mm256_castpd256_pd128(sum);
        __m128d high = _mm256_extractf128_pd(sum, 1);
        low = _mm_add_pd(low, high);
        low = _mm_hadd_pd(low, low);
        total += _mm_cvtsd_f64(low);

        constexpr usize tail_sizes[] = { 4, 2, 1 };
        for (usize size : tail_sizes)
        {
            if (remaining >= size)
            {
                if (size == 4)
                {
                    __m256d v = _mm256_loadu_pd(ptr);
                    __m128d low = _mm256_castpd256_pd128(v);
                    __m128d high = _mm256_extractf128_pd(v, 1);
                    low = _mm_add_pd(low, high);
                    total += _mm_cvtsd_f64(low) + _mm_cvtsd_f64(_mm_unpackhi_pd(low, low));
                }
                else if (size == 2)
                {
                    __m128d v = _mm_loadu_pd(ptr);
                    total += _mm_cvtsd_f64(v) + _mm_cvtsd_f64(_mm_unpackhi_pd(v, v));
                }
                else
                {
                    total += ptr[0];
                }
                ptr += size;
                remaining -= size;
            }
        }

        _mm256_zeroupper();
        return total;
    }
}

namespace fy
{
    template<BasicArithmetic Element_t>
    Element_t median(const Element_t* src, usize count) noexcept
    {
        std::unreachable();
    }

    template<typename Element_t>
    static Element_t convert_idx(u8 idx) noexcept
    {
        if constexpr (std::is_signed_v<Element_t>)
        {
            return static_cast<Element_t>(idx - 128);
        }
        else
        {
            return static_cast<Element_t>(idx);
        }
    }

    template <BasicArithmetic Element_t>
    static Element_t median_8bits__(const Element_t* src, usize count) noexcept
    {
        alignas(32) u64 counts[256] = { 0 };
        const Element_t* ptr = src;
        usize processed = 0;

        for (; processed + 32 <= count; processed += 32)
        {
            __m256i chunk = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr));
            ptr += 32;

            alignas(32) Element_t buffer[32];
            _mm256_store_si256(reinterpret_cast<__m256i*>(buffer), chunk);

            for (usize i = 0; i < 32; ++i)
            {
                u8 idx = 0;
                if constexpr (std::is_signed_v<Element_t>)
                {
                    idx = static_cast<u8>(buffer[i] + 128);
                }
                else
                {
                    idx = static_cast<u8>(buffer[i]);
                }
                counts[idx]++;
            }
        }

        for (; processed < count; ++processed)
        {
            u8 idx = 0;
            if constexpr (std::is_signed_v<Element_t>)
            {
                idx = static_cast<u8>(*ptr + 128);
            }
            else
            {
                idx = static_cast<u8>(*ptr);
            }
            counts[idx]++;
            ptr++;
        }

        const usize mid = count / 2;
        const bool is_odd = (count & 1);
        u64 accumulated = 0;
        u8 lower_idx = 0, upper_idx = 0;

        for (usize idx = 0; idx < 256; ++idx)
        {
            const u64 cnt = counts[idx];
            if (cnt == 0)
            {
                continue;
            }

            const u64 new_acc = accumulated + cnt;

            if (is_odd)
            {
                if (new_acc > mid)
                {
                    return convert_idx<Element_t>(idx);
                }
            }
            else
            {
                if (accumulated <= mid - 1 && new_acc > mid - 1)
                {
                    lower_idx = static_cast<u8>(idx);
                }
                if (accumulated <= mid && new_acc > mid)
                {
                    upper_idx = static_cast<u8>(idx);
                    const Element_t lower = convert_idx<Element_t>(lower_idx);
                    const Element_t upper = convert_idx<Element_t>(upper_idx);
                    return static_cast<Element_t>((lower + upper) / 2);
                }
            }
            accumulated = new_acc;
        }

        std::unreachable();
    }

    template<> u8 median<u8>(const u8* src, usize count) noexcept
    {
        return median_8bits__<u8>(src, count);
    }

    template<> i8 median<i8>(const i8* src, usize count) noexcept
    {
        return median_8bits__<i8>(src, count);
    }

    template<typename Element_t>
    static constexpr Element_t convert_idx_16(u16 idx) noexcept
    {
        if constexpr (std::is_signed_v<Element_t>)
        {
            return static_cast<Element_t>(idx - 32768);
        }
        else
        {
            return static_cast<Element_t>(idx);
        }
    }

    template<BasicArithmetic Element_t>
    static Element_t median_16bits__(const Element_t* src, usize count) noexcept
    {
        constexpr usize count_16bits = 65536;

        alignas(32) thread_local u64 counts[count_16bits] = { 0 };
        std::memset(counts, 0, sizeof(counts));

        const Element_t* ptr = src;
        usize processed = 0;

        for (; processed + 16 <= count; processed += 16)
        {
            __m256i chunk = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr));
            ptr += 16;

            alignas(32) Element_t buffer[16];
            _mm256_store_si256(reinterpret_cast<__m256i*>(buffer), chunk);

            for (usize i = 0; i < 16; ++i)
            {
                u16 idx = 0;
                if constexpr (std::is_signed_v<Element_t>)
                {
                    idx = static_cast<u16>(buffer[i] + 32768);
                }
                else
                {
                    idx = static_cast<u16>(buffer[i]);
                }
                counts[idx]++;
            }
        }

        for (; processed < count; ++processed)
        {
            u16 idx = 0;
            if constexpr (std::is_signed_v<Element_t>)
            {
                idx = static_cast<u16>(*ptr + 32768);
            }
            else
            {
                idx = static_cast<u16>(*ptr);
            }
            counts[idx]++;
            ptr++;
        }

        const usize mid = count / 2;
        const bool is_odd = (count & 1);
        u64 accumulated = 0;
        u16 lower_idx = 0, upper_idx = 0;

        for (u32 idx = 0; idx < 65536; ++idx)
        {
            const u64 cnt = counts[idx];
            if (cnt == 0)
            {
                continue;
            }

            const u64 new_acc = accumulated + cnt;

            if (is_odd)
            {
                if (new_acc > mid)
                {
                    return convert_idx_16<Element_t>(static_cast<u16>(idx));
                }
            }
            else
            {
                if (accumulated <= mid - 1 && new_acc > mid - 1)
                {
                    lower_idx = static_cast<u16>(idx);
                }
                if (accumulated <= mid && new_acc > mid)
                {
                    upper_idx = static_cast<u16>(idx);
                    const auto lower = convert_idx_16<Element_t>(lower_idx);
                    const auto upper = convert_idx_16<Element_t>(upper_idx);
                    return static_cast<Element_t>((lower + upper) / 2);
                }
            }

            accumulated = new_acc;
            if (accumulated > mid)
            {
                break;
            }
        }

        std::unreachable();
    }

    template<> u16 median<u16>(const u16* src, usize count) noexcept
    {
        return median_16bits__<u16>(src, count);
    }

    template<> i16 median<i16>(const i16* src, usize count) noexcept
    {
        return median_16bits__<i16>(src, count);
    }

    template<BasicArithmetic Element_t>
    static usize hoare_partition__(Element_t* arr, usize low, usize high)
    {
        const Element_t pivot = std::clamp<Element_t>(
            arr[low],
            arr[low + (high - low) / 2],
            arr[high]
        );

        usize i = low - 1;
        usize j = high + 1;

        while (true)
        {
            do 
            {
                ++i; 
            } while (arr[i] < pivot);

            do 
            {
                --j;
            } while (arr[j] > pivot);

            if (i >= j)
            {
                return j;
            }

            std::swap(arr[i], arr[j]);
        }
    }

    template<BasicArithmetic Element_t>
    static Element_t quickselect_dispatch__(Element_t* arr, usize left, usize right, usize k)
    {
        constexpr usize INSERTION_THRESHOLD = 64;

        while (right > left)
        {
            if (right - left < INSERTION_THRESHOLD)
            {
                for (usize i = left + 1; i <= right; ++i)
                {
                    Element_t key = arr[i];
                    usize j = i;
                    while (j > left && arr[j - 1] > key)
                    {
                        arr[j] = arr[j - 1];
                        --j;
                    }
                    arr[j] = key;
                }
                return arr[k];
            }

            usize pivot_idx = hoare_partition__(arr, left, right);

            if (k <= pivot_idx)
            {
                right = pivot_idx;
            }
            else
            {
                left = pivot_idx + 1;
            }
        }
        return arr[left];
    }

    template<BasicArithmetic Element_t>
    static Element_t quickselect(const Element_t* src, usize size)
    {
        Element_t* workspace = memory_alloc<Element_t>(size);
        std::memcpy(workspace, src, size * sizeof(Element_t));

        const usize mid = size / 2;
        Element_t res{};
        if (size % 2 == 1)
        {
            res = quickselect_dispatch__(workspace, 0, size - 1, mid);
        }
        else
        {
            const Element_t lower = quickselect_dispatch__(workspace, 0, size - 1, mid - 1);
            Element_t upper = std::numeric_limits<Element_t>::max();

            for (usize i = mid; i < size; ++i)
            {
                if (workspace[i] < upper)
                {
                    upper = workspace[i];
                }
            }
            res = (lower + upper) / 2;
        }

        memory_free(reinterpret_cast<void*>(workspace));
        return res;
    }

    template<> u32 median<u32>(const u32* src, usize count) noexcept
    {
        return quickselect<u32>(src, count);
    }

    template<> i32 median<i32>(const i32* src, usize count) noexcept
    {
        return quickselect<i32>(src, count);
    }

    template<> u64 median<u64>(const u64* src, usize count) noexcept
    {
        return quickselect<u64>(src, count);
    }

    template<> i64 median<i64>(const i64* src, usize count) noexcept
    {
        return quickselect<i64>(src, count);
    }

    template<> f32 median<f32>(const f32* src, usize count) noexcept
    {
        return quickselect<f32>(src, count);
    }

    template<> f64 median<f64>(const f64* src, usize count) noexcept
    {
        return quickselect<f64>(src, count);
    }
}

namespace fy
{
    template<typename Lo_t, typename Up_t>
    Lo_t acc_safe_s1_ubits__(const Lo_t* src_ptr, usize count) noexcept
    {
        __assume(count <= std::numeric_limits<Up_t>::max() / std::numeric_limits<Lo_t>::max());
        using namespace simd;
        auto_zeroupper instance{};

        AVX_t<Up_t> acc0 = v_broadcast_zero<AVX_t<Up_t>>();
        AVX_t<Up_t> acc1 = v_broadcast_zero<AVX_t<Up_t>>();
        AVX_t<Up_t> lowUP, highLO;

        const Lo_t* ptr = src_ptr;
        usize i = 0;

        for (; i + (AVX_t<Lo_t>::batch_size - 1) < count; i += AVX_t<Lo_t>::batch_size)
        {
            AVX_t<Lo_t> data(ptr + i);
            v_expand(data, lowUP, highLO);

            acc0 = v_add(acc0, lowUP);
            acc1 = v_add(acc1, highLO);
        }

        Up_t total_sum = 0;
        total_sum += v_reduce_sum(AVX_t<Up_t>(acc0));
        total_sum += v_reduce_sum(AVX_t<Up_t>(acc1));

        for (; i < count; ++i)
        {
            total_sum += ptr[i];
        }

        if constexpr (std::is_signed_v<Lo_t>)
        {
            if (total_sum >= 0)
            {
                return static_cast<Lo_t>((total_sum + count / 2) / static_cast<Up_t>(count));
            }
            else
            {
                return static_cast<Lo_t>((total_sum - count / 2) / static_cast<Up_t>(count));
            }
        }
        else
        {
            return static_cast<Lo_t>((total_sum + count / 2) / count);
        }
    }

    template<typename Lo_t, typename Up_t, typename Up2_t>
    Lo_t acc_safe_s2_ubits__(const Lo_t* src_ptr, usize count) noexcept
    {
        using namespace simd;

        auto_zeroupper instance{};

        constexpr usize first_stage_acc = std::numeric_limits<Up_t>::max() / std::numeric_limits<Lo_t>::max() - 1;
        const Lo_t* ptr = src_ptr;
        usize i = 0;
        
        AVX_t<Up2_t> acc32[4] = {
            v_broadcast_zero<AVX_t<Up2_t>>(),
            v_broadcast_zero<AVX_t<Up2_t>>(),
            v_broadcast_zero<AVX_t<Up2_t>>(),
            v_broadcast_zero<AVX_t<Up2_t>>()
        };

        for (; i + first_stage_acc <= count; i += first_stage_acc)
        {
            const Lo_t* current = ptr + i;

            AVX_t<Up_t> acc16[2] = {
                v_broadcast_zero<AVX_t<Up_t>>(),
                v_broadcast_zero<AVX_t<Up_t>>()
            };

            for (usize j = 0; j < first_stage_acc; j += AVX_t<Lo_t>::batch_size)
            {
                AVX_t<Lo_t> data(current + j);
                AVX_t<Up_t> low16, high16;
                v_expand(data, low16, high16);
                
                acc16[0] = v_add(acc16[0], low16);
                acc16[1] = v_add(acc16[1], high16);
            }

            AVX_t<Up2_t> low32_low, low32_high, high32_low, high32_high;
            v_expand(acc16[0], low32_low, low32_high);
            v_expand(acc16[1], high32_low, high32_high);

            acc32[0] = v_add(acc32[0], low32_low);
            acc32[1] = v_add(acc32[1], low32_high);
            acc32[2] = v_add(acc32[2], high32_low);
            acc32[3] = v_add(acc32[3], high32_high);
        }

        for (; i + AVX_t<Lo_t>::batch_size <= count; i += AVX_t<Lo_t>::batch_size)
        {
            AVX_t<Lo_t> data(ptr + i);
            AVX_t<Up_t> low16, high16;
            v_expand(data, low16, high16);

            AVX_t<Up2_t> low32_low, low32_high, high32_low, high32_high;
            v_expand(low16, low32_low, low32_high);
            v_expand(high16, high32_low, high32_high);

            acc32[0] = v_add(acc32[0], low32_low);
            acc32[1] = v_add(acc32[1], low32_high);
            acc32[2] = v_add(acc32[2], high32_low);
            acc32[3] = v_add(acc32[3], high32_high);
        }

        Up2_t total_sum = 0;
        for (; i < count; ++i)
        {
            total_sum += ptr[i];
        }

        for (usize idx = 0; idx < 4; ++idx)
        {
            total_sum += v_reduce_sum(acc32[idx]);
        }

        if constexpr (std::is_signed_v<Lo_t>)
        {
            if (total_sum >= 0)
            {
                return static_cast<Lo_t>((total_sum + count / 2) / static_cast<Up2_t>(count));
            }
            else
            {
                return static_cast<Lo_t>((total_sum - count / 2) / static_cast<Up2_t>(count));
            }
        }
        else
        {
            return static_cast<Lo_t>((total_sum + count / 2) / count);
        }
    }

    u8 acc_safe_s3_8bits__(const u8* src_ptr, usize count) noexcept
    {
        using namespace simd;

        auto_zeroupper instance{};

        constexpr usize stage1_size = std::numeric_limits<u16>::max() / std::numeric_limits<u8>::max() - 1;
        constexpr usize stage2_size = std::numeric_limits<u32>::max() / std::numeric_limits<u16>::max() - 1;

        const u8* ptr = src_ptr;
        usize i = 0;

        constexpr usize acc64_count = 8;
        v_uint64x4 acc64[acc64_count];
        for (usize idx = 0; idx < acc64_count; ++idx)
        {
            acc64[idx] = v_broadcast_zero<v_uint64x4>();
        }

        constexpr usize acc32_count = 4;
        v_uint32x8 acc32[acc32_count];
        for (usize idx = 0; idx < acc32_count; ++idx)
        {
            acc32[idx] = v_broadcast_zero<v_uint32x8>();
        }

        for (; i + stage2_size <= count; i += stage2_size)
        {
            v_uint16x16 acc16_low = v_broadcast_zero<v_uint16x16>();
            v_uint16x16 acc16_high = v_broadcast_zero<v_uint16x16>();

            const u8* block_start = ptr + i;
            usize inner_i = 0;

            for (; inner_i + stage1_size <= stage2_size; inner_i += stage1_size)
            {
                const u8* sub_block = block_start + inner_i;

                for (usize j = 0; j < stage1_size; j += v_uint8x32::batch_size)
                {
                    v_uint8x32 data(sub_block + j);
                    v_uint16x16 low16, high16;
                    v_expand(data, low16, high16);

                    acc16_low = v_add(acc16_low, low16);
                    acc16_high = v_add(acc16_high, high16);
                }

                v_uint32x8 expanded32[4];
                v_expand(acc16_low, expanded32[0], expanded32[1]);
                v_expand(acc16_high, expanded32[2], expanded32[3]);

                for (usize idx = 0; idx < 4; ++idx)
                {
                    acc32[idx] = v_add(acc32[idx], expanded32[idx]);
                }

                acc16_low = v_broadcast_zero<v_uint16x16>();
                acc16_high = v_broadcast_zero<v_uint16x16>();
            }

            for (; inner_i + v_uint8x32::batch_size <= stage2_size; inner_i += v_uint8x32::batch_size)
            {
                v_uint8x32 data(block_start + inner_i);
                v_uint16x16 low16, high16;
                v_expand(data, low16, high16);

                v_uint32x8 expanded32[4];
                v_expand(low16, expanded32[0], expanded32[1]);
                v_expand(high16, expanded32[2], expanded32[3]);

                for (usize idx = 0; idx < 4; ++idx)
                {
                    acc32[idx] = v_add(acc32[idx], expanded32[idx]);
                }
            }

            v_uint64x4 vu64temp[acc64_count];
            v_expand(acc32[0], vu64temp[0], vu64temp[1]);
            v_expand(acc32[1], vu64temp[2], vu64temp[3]);
            v_expand(acc32[2], vu64temp[4], vu64temp[5]);
            v_expand(acc32[3], vu64temp[6], vu64temp[7]);

            for (usize idx = 0; idx < acc64_count; ++idx)
            {
                acc64[idx] = v_add(acc64[idx], vu64temp[idx]);
            }

            for (usize idx = 0; idx < acc32_count; ++idx)
            {
                acc32[idx] = v_broadcast_zero<v_uint32x8>();
            }

            u32 scalar_sum = 0;
            for (; inner_i < stage2_size; inner_i++)
            {
                scalar_sum += block_start[inner_i];
            }

            acc64[0] = v_add(acc64[0], v_uint64x4(scalar_sum));
        }

        for (usize idx = 0; idx < acc32_count; ++idx)
        {
            acc32[idx] = v_broadcast_zero<v_uint32x8>();
        }

        for (; i + stage1_size <= count; i += stage1_size)
        {
            v_uint16x16 acc16_low = v_broadcast_zero<v_uint16x16>();
            v_uint16x16 acc16_high = v_broadcast_zero<v_uint16x16>();

            const u8* block_start = ptr + i;

            for (usize j = 0; j < stage1_size; j += v_uint8x32::batch_size)
            {
                v_uint8x32 data(block_start + j);
                v_uint16x16 low16, high16;
                v_expand(data, low16, high16);

                acc16_low = v_add(acc16_low, low16);
                acc16_high = v_add(acc16_high, high16);
            }

            v_uint32x8 temp32[4];
            v_expand(acc16_low, temp32[0], temp32[1]);
            v_expand(acc16_high, temp32[2], temp32[3]);

            for (usize idx = 0; idx < 4; ++idx)
            {
                acc32[idx] = v_add(acc32[idx], temp32[idx]);
            }
        }

        for (; i + v_uint8x32::batch_size <= count; i += v_uint8x32::batch_size)
        {
            v_uint8x32 data(ptr + i);
            v_uint16x16 temp16[2];
            v_expand(data, temp16[0], temp16[1]);

            v_uint32x8 temp32[4];
            v_expand(temp16[0], temp32[0], temp32[1]);
            v_expand(temp16[1], temp32[2], temp32[3]);

            for (usize idx = 0; idx < 4; ++idx)
            {
                acc32[idx] = v_add(acc32[idx], temp32[idx]);
            }
        }

        v_uint64x4 temp64[8];
        v_expand(acc32[0], temp64[0], temp64[1]);
        v_expand(acc32[1], temp64[2], temp64[3]);
        v_expand(acc32[2], temp64[4], temp64[5]);
        v_expand(acc32[3], temp64[6], temp64[7]);

        for (usize idx = 0; idx < acc64_count; ++idx)
        {
            acc64[idx] = v_add(acc64[idx], temp64[idx]);
        }

        u64 scalar_sum = 0;
        for (; i < count; i++)
        {
            scalar_sum += ptr[i];
        }

        u64 total_sum = scalar_sum;
        for (usize idx = 0; idx < acc64_count; ++idx)
        {
            total_sum += v_reduce_sum(acc64[idx]);
        }

        return (total_sum + count / 2) / count;
    }
}

namespace fy
{
    template<integral_arithmetic Element_t>
    Element_t mean(const Element_t* src, usize count) noexcept
    {
        std::unreachable();
    }

#define ENABLE_128Bit_ACC 0

    template<> u8 mean(const u8* src, usize count) noexcept
    {
        constexpr usize acc_u16_threshold = std::numeric_limits<u16>::max() / std::numeric_limits<u8>::max();
        constexpr usize acc_u32_threshold = std::numeric_limits<u32>::max() / std::numeric_limits<u8>::max();
        constexpr usize acc_u64_threshold = std::numeric_limits<u64>::max() / std::numeric_limits<u8>::max();

             if (count <= acc_u16_threshold) { return acc_safe_s1_ubits__<u8, u16>(src, count); }
        else if (count <= acc_u32_threshold) { return acc_safe_s2_ubits__<u8, u16, u32>(src, count); }
        else 
#if ENABLE_128Bit_ACC
                 if (count <= acc_u64_threshold)
#endif
                 {
                     return acc_safe_s3_8bits__(src, count);
                 }
#if ENABLE_128Bit_ACC
        else [[unlikely]]
        {
            u128 res{ 0 };
            for (usize i = 0; i < count; ++i)
            {
                res = res + u128(src[i]);
            }

            return static_cast<u8>((res + count / 2) / count);
        }
#endif
    }

    template<> i8 mean(const i8* src, usize count) noexcept
    {
        constexpr usize acc_u16_threshold = std::numeric_limits<i16>::max() / std::numeric_limits<i8>::max();
        constexpr usize acc_u32_threshold = std::numeric_limits<i32>::max() / std::numeric_limits<i8>::max();
        constexpr usize acc_u64_threshold = std::numeric_limits<i64>::max() / std::numeric_limits<i8>::max();

             if (count <= acc_u16_threshold) { return acc_safe_s1_ubits__<i8, i16>(src, count); }
        else if (count <= acc_u32_threshold) { return acc_safe_s2_ubits__<i8, i16, i32>(src, count); }
//        else
//#if ENABLE_128Bit_ACC
//            if (count <= acc_u64_threshold)
//#endif
//            {
//                return acc_safe_s3_8bits__<i8, i16, i32, i64>(src, count);
//            }
//#if ENABLE_128Bit_ACC
            else [[unlikely]]
            {
                u128 res{ 0 };
                for (usize i = 0; i < count; ++i)
                {
                    res = res + u128(src[i]);
                }

                return static_cast<u8>((res + count / 2) / count);
            }
//#endif
    }

    template<> u16 mean(const u16* src, usize count) noexcept
    {
        constexpr usize acc_u32_threshold = std::numeric_limits<u32>::max() / std::numeric_limits<u16>::max();
        constexpr usize acc_u64_threshold = std::numeric_limits<u64>::max() / std::numeric_limits<u16>::max();

             if (count <= acc_u32_threshold) { return acc_safe_s1_ubits__<u16, u32>(src, count); }
        else 
#if ENABLE_128Bit_ACC
                 if (count <= acc_u64_threshold) 
#endif
                 {
                     return acc_safe_s2_ubits__<u16, u32, u64>(src, count);
                 }
#if ENABLE_128Bit_ACC
        else [[unlikely]]
        {
            u128 res{ 0 };
            for (usize i = 0; i < count; ++i)
            {
                res = res + u128(src[i]);
            }

            return static_cast<u8>((res + count / 2) / count);
        }
#endif
    }

}

