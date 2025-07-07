module;
#include <immintrin.h>
#include <intrin.h>
module foye.algorithm;
import foye.foye_core;
import foye.farray;
import std;

namespace fy
{
    template<BasicArithmetic Element_t>
    void arg_where(Element_t to_find, const Element_t* src, usize count, farray<usize>& resbuf) noexcept
    {
        std::unreachable();
    }

    template<> void arg_where(u8 to_find, const u8* src, usize count, farray<usize>& resbuf) noexcept
    {
        const __m256i target = _mm256_set1_epi8(to_find);
        usize i = 0;

        for (; i + 32 <= count; i += 32)
        {
            __m256i chunk = _mm256_load_si256(reinterpret_cast<const __m256i*>(src + i));
            __m256i cmp = _mm256_cmpeq_epi8(chunk, target);
            u32 mask = std::bit_cast<u32>(_mm256_movemask_epi8(cmp));

            while (mask)
            {
                unsigned long bit_pos;
                _BitScanForward(&bit_pos, mask);
                resbuf.push_back(i + static_cast<usize>(bit_pos));
                mask &= mask - 1;
            }
        }

        for (; i < count; ++i)
        {
            if (src[i] == to_find)
            {
                resbuf.push_back(i);
            }
        }
    }

    template<> void arg_where(u16 to_find, const u16* src, usize count, farray<usize>& resbuf) noexcept
    {
        const __m256i target = _mm256_set1_epi16(static_cast<i16>(to_find));
        usize i = 0;

        for (; i + 16 <= count; i += 16)
        {
            __m256i chunk = _mm256_load_si256(reinterpret_cast<const __m256i*>(src + i));
            __m256i cmp = _mm256_cmpeq_epi16(chunk, target);

            __m256i cmp_lo = _mm256_castsi128_si256(_mm256_castsi256_si128(cmp));
            __m256i cmp_hi = _mm256_castsi128_si256(_mm256_extractf128_si256(cmp, 1));

            __m256i packed = _mm256_packs_epi16(cmp_lo, cmp_hi);
            u32 mask = std::bit_cast<u32>(_mm256_movemask_epi8(packed)) & 0xFFFF;

            while (mask)
            {
                unsigned long bit_pos;
                _BitScanForward(&bit_pos, mask);
                resbuf.push_back(i + static_cast<usize>(bit_pos));
                mask &= mask - 1;
            }
        }

        for (; i < count; ++i)
        {
            if (src[i] == to_find)
            {
                resbuf.push_back(i);
            }
        }
    }

    template<> void arg_where(u32 to_find, const u32* src, usize count, farray<usize>& resbuf) noexcept
    {
        const __m256i target = _mm256_set1_epi32(static_cast<i32>(to_find));
        usize i = 0;

        for (; i + 8 <= count; i += 8)
        {
            __m256i chunk = _mm256_load_si256(reinterpret_cast<const __m256i*>(src + i));
            __m256i cmp = _mm256_cmpeq_epi32(chunk, target);
            u32 mask = std::bit_cast<u32>(_mm256_movemask_ps(_mm256_castsi256_ps(cmp)));

            while (mask)
            {
                unsigned long bit_pos;
                _BitScanForward(&bit_pos, mask);
                resbuf.push_back(i + static_cast<usize>(bit_pos));
                mask &= mask - 1;
            }
        }

        for (; i < count; ++i)
        {
            if (src[i] == to_find)
            {
                resbuf.push_back(i);
            }
        }
    }

    template<> void arg_where(u64 to_find, const u64* src, usize count, farray<usize>& resbuf) noexcept
    {
        const __m256i target = _mm256_set1_epi64x(static_cast<i64>(to_find));
        usize i = 0;

        for (; i + 4 <= count; i += 4)
        {
            __m256i chunk = _mm256_load_si256(reinterpret_cast<const __m256i*>(src + i));
            __m256i cmp = _mm256_cmpeq_epi64(chunk, target);
            u32 mask = std::bit_cast<u32>(_mm256_movemask_pd(_mm256_castsi256_pd(cmp)));

            while (mask)
            {
                unsigned long bit_pos;
                _BitScanForward(&bit_pos, mask);
                resbuf.push_back(i + static_cast<usize>(bit_pos));
                mask &= mask - 1;
            }
        }

        for (; i < count; ++i)
        {
            if (src[i] == to_find)
            {
                resbuf.push_back(i);
            }
        }
    }

    template<> void arg_where(i8 to_find, const i8* src, usize count, farray<usize>& resbuf) noexcept
    {
        return arg_where(std::bit_cast<u8>(to_find), reinterpret_cast<const u8*>(src), count, resbuf);
    }

    template<> void arg_where(i16 to_find, const i16* src, usize count, farray<usize>& resbuf) noexcept
    {
        return arg_where(std::bit_cast<u16>(to_find), reinterpret_cast<const u16*>(src), count, resbuf);
    }

    template<> void arg_where(i32 to_find, const i32* src, usize count, farray<usize>& resbuf) noexcept
    {
        return arg_where(std::bit_cast<u32>(to_find), reinterpret_cast<const u32*>(src), count, resbuf);
    }

    template<> void arg_where(i64 to_find, const i64* src, usize count, farray<usize>& resbuf) noexcept
    {
        return arg_where(std::bit_cast<u64>(to_find), reinterpret_cast<const u64*>(src), count, resbuf);
    }

    template<> void arg_where(f16 to_find, const f16* src, usize count, farray<usize>& resbuf) noexcept
    {
        return arg_where(std::bit_cast<u16>(to_find), reinterpret_cast<const u16*>(src), count, resbuf);
    }

    template<> void arg_where(f32 to_find, const f32* src, usize count, farray<usize>& resbuf) noexcept
    {
        return arg_where(std::bit_cast<u32>(to_find), reinterpret_cast<const u32*>(src), count, resbuf);
    }

    template<> void arg_where(f64 to_find, const f64* src, usize count, farray<usize>& resbuf) noexcept
    {
        return arg_where(std::bit_cast<u64>(to_find), reinterpret_cast<const u64*>(src), count, resbuf);
    }
}

namespace fy
{
    template<Floating_arithmetic Element_t>
    void arg_where(Element_t to_find, const Element_t* src, usize count, farray<usize>& resbuf, Element_t epsilon) noexcept
    {
        std::unreachable();
    }

    template<> void arg_where(f16 to_find, const f16* src, usize count, farray<usize>& resbuf, f16 epsilon) noexcept
    {
        const f32 target_f32 = static_cast<f32>(to_find);
        const f32 epsilon_f32 = std::max(
            static_cast<f32>(epsilon),
            static_cast<f32>(limits_floating<f16>::epsilon())
        );
        const __m256 target = _mm256_set1_ps(target_f32);
        const __m256 eps_vec = _mm256_set1_ps(epsilon_f32);
        const __m256 sign_mask = _mm256_set1_ps(-0.0f);

        usize i = 0;

        for (; i + 16 <= count; i += 16)
        {
            __m128i hbits1 = _mm_load_si128(reinterpret_cast<const __m128i*>(src + i));
            __m128i hbits2 = _mm_load_si128(reinterpret_cast<const __m128i*>(src + i + 8));

            __m256 vec1 = _mm256_cvtph_ps(hbits1);
            __m256 vec2 = _mm256_cvtph_ps(hbits2);

            __m256 diff1 = _mm256_sub_ps(vec1, target);
            __m256 abs1 = _mm256_andnot_ps(sign_mask, diff1);
            __m256 cmp1 = _mm256_cmp_ps(abs1, eps_vec, _CMP_LE_OQ);
            u32 mask1 = _mm256_movemask_ps(cmp1);

            __m256 diff2 = _mm256_sub_ps(vec2, target);
            __m256 abs2 = _mm256_andnot_ps(sign_mask, diff2);
            __m256 cmp2 = _mm256_cmp_ps(abs2, eps_vec, _CMP_LE_OQ);
            u32 mask2 = _mm256_movemask_ps(cmp2);

            u64 mask = (static_cast<u64>(mask2) << 8) | mask1;

            while (mask)
            {
                unsigned long bit_pos;
                _BitScanForward64(&bit_pos, mask);
                resbuf.push_back(i + static_cast<usize>(bit_pos));
                mask &= mask - 1;
            }
        }

        for (; i + 8 <= count; i += 8)
        {
            __m128i hbits = _mm_load_si128(reinterpret_cast<const __m128i*>(src + i));
            __m256 vec = _mm256_cvtph_ps(hbits);

            __m256 diff = _mm256_sub_ps(vec, target);
            __m256 abs_diff = _mm256_andnot_ps(sign_mask, diff);
            __m256 cmp = _mm256_cmp_ps(abs_diff, eps_vec, _CMP_LE_OQ);
            u32 mask = _mm256_movemask_ps(cmp);

            while (mask)
            {
                unsigned long bit_pos;
                _BitScanForward(&bit_pos, mask);
                resbuf.push_back(i + static_cast<usize>(bit_pos));
                mask &= mask - 1;
            }
        }

        for (; i < count; ++i)
        {
            const f32 val = static_cast<f32>(src[i]);
            if (std::abs(val - target_f32) <= epsilon_f32 && !src[i].isNaN())
            {
                resbuf.push_back(i);
            }
        }
    }

    template<> void arg_where(f32 to_find, const f32* src, usize count, farray<usize>& resbuf, f32 epsilon) noexcept
    {
        const __m256 target = _mm256_set1_ps(to_find);
        const __m256 eps_vec = _mm256_set1_ps(epsilon);
        const __m256 sign_mask = _mm256_set1_ps(-0.0f);
        usize i = 0;

        for (; i + 16 <= count; i += 16)
        {
            __m256 chunk1 = _mm256_loadu_ps(src + i);
            __m256 chunk2 = _mm256_loadu_ps(src + i + 8);

            __m256 diff1 = _mm256_sub_ps(chunk1, target);
            __m256 diff2 = _mm256_sub_ps(chunk2, target);
            __m256 abs_diff1 = _mm256_andnot_ps(sign_mask, diff1);
            __m256 abs_diff2 = _mm256_andnot_ps(sign_mask, diff2);

            __m256 cmp1 = _mm256_cmp_ps(abs_diff1, eps_vec, _CMP_LE_OQ);
            __m256 cmp2 = _mm256_cmp_ps(abs_diff2, eps_vec, _CMP_LE_OQ);

            u32 mask1 = std::bit_cast<u32>(_mm256_movemask_ps(cmp1));
            u32 mask2 = std::bit_cast<u32>(_mm256_movemask_ps(cmp2));
            u64 mask = static_cast<u64>(mask1) | (static_cast<u64>(mask2) << 8);

            while (mask)
            {
                unsigned long bit_pos;
                _BitScanForward64(&bit_pos, mask);
                resbuf.push_back(i + static_cast<usize>(bit_pos));
                mask &= mask - 1;
            }
        }

        for (; i + 8 <= count; i += 8)
        {
            __m256 chunk = _mm256_loadu_ps(src + i);
            __m256 diff = _mm256_sub_ps(chunk, target);
            __m256 abs_diff = _mm256_andnot_ps(sign_mask, diff);
            __m256 cmp = _mm256_cmp_ps(abs_diff, eps_vec, _CMP_LE_OQ);
            u32 mask = std::bit_cast<u32>(_mm256_movemask_ps(cmp));

            while (mask)
            {
                unsigned long bit_pos;
                _BitScanForward(&bit_pos, mask);
                resbuf.push_back(i + static_cast<usize>(bit_pos));
                mask &= mask - 1;
            }
        }

        for (; i < count; ++i)
        {
            if (std::abs(src[i] - to_find) <= epsilon)
            {
                resbuf.push_back(i);
            }
        }
    }

    template<> void arg_where(f64 to_find, const f64* src, usize count, farray<usize>& resbuf, f64 epsilon) noexcept
    {
        const __m256d target = _mm256_set1_pd(to_find);
        const __m256d eps_vec = _mm256_set1_pd(epsilon);
        const __m256d neg_zero = _mm256_set1_pd(-0.0);
        usize i = 0;

        for (; i + 8 <= count; i += 8)
        {
            __m256d chunk1 = _mm256_load_pd(src + i);
            __m256d chunk2 = _mm256_load_pd(src + i + 4);

            __m256d diff1 = _mm256_sub_pd(chunk1, target);
            __m256d diff2 = _mm256_sub_pd(chunk2, target);
            __m256d abs_diff1 = _mm256_andnot_pd(neg_zero, diff1);
            __m256d abs_diff2 = _mm256_andnot_pd(neg_zero, diff2);

            __m256d cmp1 = _mm256_cmp_pd(abs_diff1, eps_vec, _CMP_LE_OQ);
            __m256d cmp2 = _mm256_cmp_pd(abs_diff2, eps_vec, _CMP_LE_OQ);

            u64 mask = static_cast<u64>(_mm256_movemask_pd(cmp1)) | (static_cast<u64>(_mm256_movemask_pd(cmp2)) << 4);

            while (mask)
            {
                unsigned long bit_pos;
                _BitScanForward64(&bit_pos, mask);
                resbuf.push_back(i + static_cast<usize>(bit_pos));
                mask &= mask - 1;
            }
        }

        for (; i + 4 <= count; i += 4)
        {
            __m256d chunk = _mm256_load_pd(src + i);
            __m256d diff = _mm256_sub_pd(chunk, target);
            __m256d abs_diff = _mm256_andnot_pd(neg_zero, diff);
            __m256d cmp = _mm256_cmp_pd(abs_diff, eps_vec, _CMP_LE_OQ);
            u64 mask = static_cast<u64>(_mm256_movemask_pd(cmp));

            while (mask)
            {
                unsigned long bit_pos;
                _BitScanForward64(&bit_pos, mask);
                resbuf.push_back(i + static_cast<usize>(bit_pos));
                mask &= mask - 1;
            }
        }

        for (; i < count; ++i)
        {
            if (std::abs(src[i] - to_find) <= epsilon)
            {
                resbuf.push_back(i);
            }
        }
    }
}

namespace fy
{
    template<BasicArithmetic Element_t>
    void argset_where(const Element_t* const to_find, usize count_to_find, const Element_t* src, usize count_count, farray<usize>& resbuf) noexcept
    {
        std::unreachable();
    }

    template<typename Bits8_t>
    static void argset_where_8bit__(const Bits8_t* const to_find, usize count_to_find, const Bits8_t* src, usize count_src, farray<usize>& resbuf) noexcept
    {
        constexpr usize threshold_for_initiating_vectorized_loop_comparison = 4;
        constexpr usize bits_permask = 32;

        if (count_to_find <= threshold_for_initiating_vectorized_loop_comparison)
        {
            usize i = 0;
            for (; i + 32 <= count_src; i += 32)
            {
                __m256i src_chunk = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + i));
                u32 combined_mask = 0;

                for (usize j = 0; j < count_to_find; ++j)
                {
                    __m256i target = _mm256_set1_epi8(to_find[j]);
                    __m256i cmp = _mm256_cmpeq_epi8(src_chunk, target);
                    combined_mask |= static_cast<u32>(_mm256_movemask_epi8(cmp));
                }

                while (combined_mask)
                {
                    unsigned long bit_pos;
                    _BitScanForward(&bit_pos, combined_mask);
                    resbuf.push_back(i + static_cast<usize>(bit_pos));
                    combined_mask &= combined_mask - 1;
                }
            }

            for (; i < count_src; ++i)
            {
                for (usize j = 0; j < count_to_find; ++j)
                {
                    if (src[i] == to_find[j])
                    {
                        resbuf.push_back(i);
                        break;
                    }
                }
            }
        }
        else
        {
            u8 mask_arr[bits_permask] = { 0 };
            for (usize i = 0; i < count_to_find; ++i)
            {
                i8 val = to_find[i];
                u8 v = static_cast<Bits8_t>(val);
                mask_arr[v >> 3] |= (1 << (v & 7));
            }

            usize i = 0;
            for (; i + bits_permask <= count_src; i += bits_permask)
            {
                const Bits8_t* temp = src + i;

                u32 mask = 0;
                for (usize j = 0; j < bits_permask; ++j)
                {
                    Bits8_t val = temp[j];
                    u8 v = static_cast<u8>(val);
                    if ((mask_arr[v >> 3] >> (v & 7)) & 1)
                    {
                        mask |= (1 << j);
                    }
                }

                while (mask)
                {
                    unsigned long bit_pos;
                    _BitScanForward(&bit_pos, mask);
                    resbuf.push_back(i + static_cast<usize>(bit_pos));
                    mask &= mask - 1;
                }
            }

            for (; i < count_src; ++i)
            {
                i8 val = src[i];
                u8 v = static_cast<u8>(val);
                if ((mask_arr[v >> 3] >> (v & 7)) & 1)
                {
                    resbuf.push_back(i);
                }
            }
        }
    }

    template<> void argset_where<u8>(const u8* const to_find, usize count_to_find, const u8* src, usize count_src, farray<usize>& resbuf) noexcept
    {
        return argset_where_8bit__<u8>(to_find, count_to_find, src, count_src, resbuf);
    }

    template<> void argset_where<i8>(const i8* const to_find, usize count_to_find, const i8* src, usize count_src, farray<usize>& resbuf) noexcept
    {
        return argset_where_8bit__<i8>(to_find, count_to_find, src, count_src, resbuf);
    }

    template<typename Bits16_t>
    static void argset_where_16bit__(const Bits16_t* const to_find, usize count_to_find, const Bits16_t* src, usize count_src, farray<usize>& resbuf) noexcept
    {
        constexpr usize threshold_for_vectorized = 4;

        if (count_to_find <= threshold_for_vectorized)
        {
            usize i = 0;
            for (; i + 16 <= count_src; i += 16)
            {
                __m256i src_chunk = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + i));
                u32 combined_mask = 0;

                for (usize j = 0; j < count_to_find; ++j)
                {
                    __m256i target = _mm256_set1_epi16(to_find[j]);
                    __m256i cmp = _mm256_cmpeq_epi16(src_chunk, target);
                    u32 mask = _mm256_movemask_epi8(cmp);
                    combined_mask |= (mask & 0xAAAAAAAA) >> 1;
                }

                while (combined_mask)
                {
                    unsigned long bit_pos;
                    _BitScanForward(&bit_pos, combined_mask);
                    resbuf.push_back(i + static_cast<usize>(bit_pos));
                    combined_mask &= combined_mask - 1;
                }
            }

            for (; i < count_src; ++i)
            {
                for (usize j = 0; j < count_to_find; ++j)
                {
                    if (src[i] == to_find[j])
                    {
                        resbuf.push_back(i);
                        break;
                    }
                }
            }
        }
        else
        {
            alignas(32) u8 mask_arr[8192] = { 0 };

            for (usize i = 0; i < count_to_find; ++i)
            {
                u16 val = static_cast<u16>(to_find[i]);
                mask_arr[val >> 3] |= (1 << (val & 7));
            }

            usize i = 0;
            for (; i + 16 <= count_src; i += 16)
            {
                const Bits16_t* block = src + i;
                u32 mask = 0;

                for (usize j = 0; j < 16; ++j)
                {
                    u16 val = static_cast<u16>(block[j]);
                    if ((mask_arr[val >> 3] >> (val & 7)) & 1)
                    {
                        mask |= (1 << j);
                    }
                }

                while (mask)
                {
                    unsigned long bit_pos;
                    _BitScanForward(&bit_pos, mask);
                    resbuf.push_back(i + static_cast<usize>(bit_pos));
                    mask &= mask - 1;
                }
            }

            for (; i < count_src; ++i)
            {
                u16 val = static_cast<u16>(src[i]);
                if ((mask_arr[val >> 3] >> (val & 7)) & 1)
                {
                    resbuf.push_back(i);
                }
            }
        }
    }

    template<> void argset_where<u16>(const u16* const to_find, usize count_to_find, const u16* src, usize count_src, farray<usize>& resbuf) noexcept
    {
        return argset_where_16bit__<u16>(to_find, count_to_find, src, count_src, resbuf);
    }

    template<> void argset_where<i16>(const i16* const to_find, usize count_to_find, const i16* src, usize count_src, farray<usize>& resbuf) noexcept
    {
        return argset_where_16bit__<i16>(to_find, count_to_find, src, count_src, resbuf);
    }

    template<> void argset_where<f16>(const f16* const to_find, usize count_to_find, const f16* src, usize count_src, farray<usize>& resbuf) noexcept
    {
        return argset_where_16bit__<f16>(to_find, count_to_find, src, count_src, resbuf);
    }

    template<typename Bits32_t>
    static void argset_where_32bit__(const Bits32_t* const to_find, usize count_to_find, const Bits32_t* src, usize count_src, farray<usize>& resbuf) noexcept
    {
        constexpr usize threshold_for_vectorized = 4;
        if (count_to_find <= threshold_for_vectorized)
        {
            usize i = 0;
            for (; i + 8 <= count_src; i += 8)
            {
                __m256i src_chunk = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + i));
                u32 combined_mask = 0;

                for (usize j = 0; j < count_to_find; ++j)
                {
                    __m256i target = _mm256_set1_epi32(static_cast<int32_t>(to_find[j]));
                    __m256i cmp = _mm256_cmpeq_epi32(src_chunk, target);

                    u32 mask = static_cast<u32>(_mm256_movemask_ps(_mm256_castsi256_ps(cmp)));
                    combined_mask |= mask;
                }

                while (combined_mask)
                {
                    unsigned long bit_pos;
                    _BitScanForward(&bit_pos, combined_mask);
                    resbuf.push_back(i + static_cast<usize>(bit_pos));
                    combined_mask &= combined_mask - 1;
                }
            }

            for (; i < count_src; ++i)
            {
                for (usize j = 0; j < count_to_find; ++j)
                {
                    if (src[i] == to_find[j])
                    {
                        resbuf.push_back(i);
                        break;
                    }
                }
            }
        }
        else
        {
            std::unordered_set<u32> lookup_set;
            for (usize i = 0; i < count_to_find; ++i)
            {
                lookup_set.insert(static_cast<u32>(to_find[i]));
            }

            usize i = 0;
            for (; i + 8 <= count_src; i += 8)
            {
                const Bits32_t* block = src + i;
                u32 mask = 0;

                for (usize j = 0; j < 8; ++j)
                {
                    if (lookup_set.count(static_cast<u32>(block[j])))
                    {
                        mask |= (1 << j);
                    }
                }

                while (mask)
                {
                    unsigned long bit_pos;
                    _BitScanForward(&bit_pos, mask);
                    resbuf.push_back(i + static_cast<usize>(bit_pos));
                    mask &= mask - 1;
                }
            }

            for (; i < count_src; ++i)
            {
                if (lookup_set.count(static_cast<u32>(src[i])))
                {
                    resbuf.push_back(i);
                }
            }
        }
    }

    template<> void argset_where<u32>(const u32* const to_find, usize count_to_find, const u32* src, usize count_src, farray<usize>& resbuf) noexcept
    {
        return argset_where_32bit__<u32>(to_find, count_to_find, src, count_src, resbuf);
    }

    template<> void argset_where<i32>(const i32* const to_find, usize count_to_find, const i32* src, usize count_src, farray<usize>& resbuf) noexcept
    {
        return argset_where_32bit__<i32>(to_find, count_to_find, src, count_src, resbuf);
    }

    template<> void argset_where<f32>(const f32* const to_find, usize count_to_find, const f32* src, usize count_src, farray<usize>& resbuf) noexcept
    {
        return argset_where_32bit__<f32>(to_find, count_to_find, src, count_src, resbuf);
    }

    template<typename Bits64_t>
    static void argset_where_64bit__(const Bits64_t* const to_find, usize count_to_find, const Bits64_t* src, usize count_src, farray<usize>& resbuf) noexcept
    {
        constexpr usize threshold_for_vectorized = 4;
        constexpr usize elements_per_vector = 4;

        if (count_to_find <= threshold_for_vectorized)
        {
            usize i = 0;
            for (; i + elements_per_vector <= count_src; i += elements_per_vector)
            {
                __m256i src_chunk = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + i));
                u32 combined_mask = 0;

                for (usize j = 0; j < count_to_find; ++j)
                {
                    __m256i target = _mm256_set1_epi64x(static_cast<i64>(to_find[j]));
                    __m256i cmp = _mm256_cmpeq_epi64(src_chunk, target);

                    __m256d cmp_dbl = _mm256_castsi256_pd(cmp);
                    u32 mask = _mm256_movemask_pd(cmp_dbl);
                    combined_mask |= mask;
                }

                while (combined_mask)
                {
                    unsigned long bit_pos;
                    _BitScanForward(&bit_pos, combined_mask);
                    resbuf.push_back(i + static_cast<usize>(bit_pos));
                    combined_mask &= combined_mask - 1;
                }
            }

            for (; i < count_src; ++i)
            {
                for (usize j = 0; j < count_to_find; ++j)
                {
                    if (src[i] == to_find[j])
                    {
                        resbuf.push_back(i);
                        break;
                    }
                }
            }
        }
        else
        {
            std::unordered_set<u64> lookup_set;
            for (usize i = 0; i < count_to_find; ++i)
            {
                lookup_set.insert(static_cast<u64>(to_find[i]));
            }

            for (usize i = 0; i < count_src; ++i)
            {
                if (lookup_set.count(static_cast<u64>(src[i])))
                {
                    resbuf.push_back(i);
                }
            }
        }
    }

    template<> void argset_where<u64>(const u64* const to_find, usize count_to_find, const u64* src, usize count_src, farray<usize>& resbuf) noexcept
    {
        return argset_where_64bit__<u64>(to_find, count_to_find, src, count_src, resbuf);
    }

    template<> void argset_where<i64>(const i64* const to_find, usize count_to_find, const i64* src, usize count_src, farray<usize>& resbuf) noexcept
    {
        return argset_where_64bit__<i64>(to_find, count_to_find, src, count_src, resbuf);
    }

    template<> void argset_where<f64>(const f64* const to_find, usize count_to_find, const f64* src, usize count_src, farray<usize>& resbuf) noexcept
    {
        return argset_where_64bit__<f64>(to_find, count_to_find, src, count_src, resbuf);
    }
}


namespace fy
{
    template<Floating_arithmetic Element_t>
    void argset_where(
        const Element_t* const to_find, usize count_to_find,
        const Element_t* src,     usize count_src,
        farray<usize>& resbuf,
        Element_t epsilon) noexcept
    {
        std::unreachable();
    }

    template<> void argset_where<f32>(
        const f32* const to_find, usize count_to_find,
        const f32* src,           usize count_src,
        farray<usize>& resbuf,
        f32 epsilon) noexcept
    {
        constexpr usize threshold_for_vectorized = 8;

        if (count_to_find <= threshold_for_vectorized)
        {
            const __m256 eps_vec = _mm256_set1_ps(epsilon);
            const __m256 sign_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));

            usize i = 0;
            for (; i + 8 <= count_src; i += 8)
            {
                __m256 src_vec = _mm256_loadu_ps(src + i);
                __m256 any_match = _mm256_setzero_ps();

                for (usize j = 0; j < count_to_find; ++j)
                {
                    __m256 target = _mm256_set1_ps(to_find[j]);
                    __m256 diff = _mm256_sub_ps(src_vec, target);
                    __m256 abs_diff = _mm256_and_ps(diff, sign_mask);
                    __m256 cmp = _mm256_cmp_ps(abs_diff, eps_vec, _CMP_LE_OQ);
                    any_match = _mm256_or_ps(any_match, cmp);
                }

                u32 mask = static_cast<u32>(_mm256_movemask_ps(any_match));
                while (mask)
                {
                    unsigned long bit_pos;
                    _BitScanForward(&bit_pos, mask);
                    resbuf.push_back(i + bit_pos);
                    mask &= mask - 1;
                }
            }

            for (; i < count_src; ++i)
            {
                const f32 x = src[i];
                for (usize j = 0; j < count_to_find; ++j)
                {
                    if (std::abs(x - to_find[j]) <= epsilon)
                    {
                        resbuf.push_back(i);
                        break;
                    }
                }
            }
        }
        else
        {
            std::vector<f32> sorted_to_find(to_find, to_find + count_to_find);
            std::sort(sorted_to_find.begin(), sorted_to_find.end());

            for (usize i = 0; i < count_src; ++i)
            {
                const f32 x = src[i];
                const f32 low = x - epsilon;
                const f32 high = x + epsilon;

                auto it = std::lower_bound(sorted_to_find.begin(), sorted_to_find.end(), low);

                if (it != sorted_to_find.end() && *it <= high)
                {
                    resbuf.push_back(i);
                }
            }
        }
    }
}