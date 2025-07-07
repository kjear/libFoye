module;
#include <immintrin.h>
#include <intrin.h>
module foye.algorithm;
import foye.foye_core;
import foye.farray;
import foye.simd;
import std;

#pragma warning(disable: 4309)
#pragma warning(disable: 4244)

namespace fy
{
    template<BasicArithmetic Element_t>
    usize count_diff(const Element_t* cmp_0, const Element_t* cmp_1, usize count) noexcept
    {
        usize res = 0;
        constexpr usize stride = sizeof(__m256i);
        constexpr usize elements_per_stride = stride / sizeof(Element_t);

        const Element_t* end = cmp_0 + (count / elements_per_stride) * elements_per_stride;
        for (; cmp_0 < end; cmp_0 += elements_per_stride, cmp_1 += elements_per_stride)
        {
            __m256i vec0 = _mm256_load_si256(reinterpret_cast<const __m256i*>(cmp_0));
            __m256i vec1 = _mm256_load_si256(reinterpret_cast<const __m256i*>(cmp_1));
            
            __m256i cmp;
            if constexpr (sizeof(Element_t) == sizeof(u8))
            {
                cmp = _mm256_cmpeq_epi8(vec0, vec1);
            }
            else if constexpr (sizeof(Element_t) == sizeof(u16))
            {
                cmp = _mm256_cmpeq_epi16(vec0, vec1);
            }
            else if constexpr (sizeof(Element_t) == sizeof(u32))
            {
                cmp = _mm256_cmpeq_epi32(vec0, vec1);
            }
            else if constexpr (sizeof(Element_t) == sizeof(u64))
            {
                cmp = _mm256_cmpeq_epi64(vec0, vec1);
            }

            u32 mask = static_cast<u32>(_mm256_movemask_epi8(cmp));
            res += elements_per_stride - (_mm_popcnt_u32(mask) / (stride / elements_per_stride));
        }

        usize remaining = count % elements_per_stride;
        for (usize i = 0; i < remaining; ++i)
        {
            res += (cmp_0[i] != cmp_1[i]);
        }

        return res;
    }

    template<> usize count_diff<f16>(const f16* cmp_0, const f16* cmp_1, usize count) noexcept
    {
        return count_diff<u16>(reinterpret_cast<const u16*>(cmp_0), reinterpret_cast<const u16*>(cmp_1), count);
    }

    template<> usize count_diff<f32>(const f32* cmp_0, const f32* cmp_1, usize count) noexcept
    {
        return count_diff<u32>(reinterpret_cast<const u32*>(cmp_0), reinterpret_cast<const u32*>(cmp_1), count);
    }

    template<> usize count_diff<f64>(const f64* cmp_0, const f64* cmp_1, usize count) noexcept
    {
        return count_diff<u64>(reinterpret_cast<const u64*>(cmp_0), reinterpret_cast<const u64*>(cmp_1), count);
    }

    template<BasicArithmetic Element_t>
    bool any_diff(const Element_t* cmp_0, const Element_t* cmp_1, usize count) noexcept
    {
        constexpr usize kAVX2BytesPerVector = 32;
        constexpr usize elements_per_vector = kAVX2BytesPerVector / sizeof(Element_t);
        constexpr u32 kFullMask = 0xFFFFFFFF;

        const Element_t* end = cmp_0 + count;
        const Element_t* avx2_end = cmp_0 + (count / elements_per_vector) * elements_per_vector;

        while (cmp_0 < avx2_end)
        {
            __m256i vec0 = _mm256_load_si256(reinterpret_cast<const __m256i*>(cmp_0));
            __m256i vec1 = _mm256_load_si256(reinterpret_cast<const __m256i*>(cmp_1));

            __m256i cmp_result;
            if constexpr (sizeof(Element_t) == sizeof(u8))
            {
                cmp_result = _mm256_cmpeq_epi8(vec0, vec1);
            }
            else if constexpr (sizeof(Element_t) == sizeof(u16))
            {
                cmp_result = _mm256_cmpeq_epi16(vec0, vec1);
            }
            else if constexpr (sizeof(Element_t) == sizeof(u32))
            {
                cmp_result = _mm256_cmpeq_epi32(vec0, vec1);
            }
            else if constexpr (sizeof(Element_t) == sizeof(u64))
            {
                cmp_result = _mm256_cmpeq_epi64(vec0, vec1);
            }


            u32 mask = static_cast<u32>(_mm256_movemask_epi8(cmp_result));

            if (mask != kFullMask)
            {
                return true;
            }

            cmp_0 += elements_per_vector;
            cmp_1 += elements_per_vector;
        }

        for (; cmp_0 < end; ++cmp_0, ++cmp_1)
        {
            if (*cmp_0 != *cmp_1)
            {
                return true;
            }
        }

        return false;
    }

    template<> bool any_diff<f16>(const f16* cmp_0, const f16* cmp_1, usize count) noexcept
    {
        return any_diff<u16>(reinterpret_cast<const u16*>(cmp_0), reinterpret_cast<const u16*>(cmp_1), count);
    }

    template<> bool any_diff<f32>(const f32* cmp_0, const f32* cmp_1, usize count) noexcept
    {
        return any_diff<u32>(reinterpret_cast<const u32*>(cmp_0), reinterpret_cast<const u32*>(cmp_1), count);
    }

    template<> bool any_diff<f64>(const f64* cmp_0, const f64* cmp_1, usize count) noexcept
    {
        return any_diff<u64>(reinterpret_cast<const u64*>(cmp_0), reinterpret_cast<const u64*>(cmp_1), count);
    }

    template<BasicArithmetic Element_t>
    bool any_same(const Element_t* cmp0, const Element_t* cmp1, size_t count) noexcept
    {
        constexpr size_t kAVX2BytesPerVector = 32;
        constexpr size_t elements_per_vector = kAVX2BytesPerVector / sizeof(Element_t);

        const Element_t* end = cmp0 + count;
        const Element_t* avx2_end = cmp0 + (count / elements_per_vector) * elements_per_vector;

        while (cmp0 < avx2_end)
        {
            __m256i vec0 = _mm256_load_si256(reinterpret_cast<const __m256i*>(cmp0));
            __m256i vec1 = _mm256_load_si256(reinterpret_cast<const __m256i*>(cmp1));

            __m256i cmp_result;
            if constexpr (sizeof(Element_t) == 1)
            {
                cmp_result = _mm256_cmpeq_epi8(vec0, vec1);
            }
            else if constexpr (sizeof(Element_t) == 2)
            {
                cmp_result = _mm256_cmpeq_epi16(vec0, vec1);
            }
            else if constexpr (sizeof(Element_t) == 4)
            {
                cmp_result = _mm256_cmpeq_epi32(vec0, vec1);
            }
            else if constexpr (sizeof(Element_t) == 8)
            {
                cmp_result = _mm256_cmpeq_epi64(vec0, vec1);
            }

            u32 mask = static_cast<u32>(_mm256_movemask_epi8(cmp_result));
            if (mask != 0)
            {
                return true;
            }

            cmp0 += elements_per_vector;
            cmp1 += elements_per_vector;
        }

        for (; cmp0 < end; ++cmp0, ++cmp1)
        {
            if (*cmp0 == *cmp1)
            {
                return true;
            }
        }

        return false;
    }

    template<> bool any_same<f16>(const f16* cmp_0, const f16* cmp_1, usize count) noexcept
    {
        return any_same<u16>(reinterpret_cast<const u16*>(cmp_0), reinterpret_cast<const u16*>(cmp_1), count);
    }

    template<> bool any_same<f32>(const f32* cmp_0, const f32* cmp_1, usize count) noexcept
    {
        return any_same<u32>(reinterpret_cast<const u32*>(cmp_0), reinterpret_cast<const u32*>(cmp_1), count);
    }

    template<> bool any_same<f64>(const f64* cmp_0, const f64* cmp_1, usize count) noexcept
    {
        return any_same<u64>(reinterpret_cast<const u64*>(cmp_0), reinterpret_cast<const u64*>(cmp_1), count);
    }


    template usize count_diff<u8>(const u8*, const u8*, usize) noexcept;
    template usize count_diff<u16>(const u16*, const u16*, usize) noexcept;
    template usize count_diff<u32>(const u32*, const u32*, usize) noexcept;
    template usize count_diff<u64>(const u64*, const u64*, usize) noexcept;
    template usize count_diff<i8>(const i8*, const i8*, usize) noexcept;
    template usize count_diff<i16>(const i16*, const i16*, usize) noexcept;
    template usize count_diff<i32>(const i32*, const i32*, usize) noexcept;
    template usize count_diff<i64>(const i64*, const i64*, usize) noexcept;

    template bool any_diff<u8>(const u8*, const u8*, usize) noexcept;
    template bool any_diff<u16>(const u16*, const u16*, usize) noexcept;
    template bool any_diff<u32>(const u32*, const u32*, usize) noexcept;
    template bool any_diff<u64>(const u64*, const u64*, usize) noexcept;
    template bool any_diff<i8>(const i8*, const i8*, usize) noexcept;
    template bool any_diff<i16>(const i16*, const i16*, usize) noexcept;
    template bool any_diff<i32>(const i32*, const i32*, usize) noexcept;
    template bool any_diff<i64>(const i64*, const i64*, usize) noexcept;

    template bool any_same<u8>(const u8*, const u8*, usize) noexcept;
    template bool any_same<u16>(const u16*, const u16*, usize) noexcept;
    template bool any_same<u32>(const u32*, const u32*, usize) noexcept;
    template bool any_same<u64>(const u64*, const u64*, usize) noexcept;
    template bool any_same<i8>(const i8*, const i8*, usize) noexcept;
    template bool any_same<i16>(const i16*, const i16*, usize) noexcept;
    template bool any_same<i32>(const i32*, const i32*, usize) noexcept;
    template bool any_same<i64>(const i64*, const i64*, usize) noexcept;
}

namespace fy
{
    template<typename Element_t, typename scalar_op> struct bitwise_cmp_Invoker
    {
        Element_t mask_equal_val;
        Element_t mask_nonequal_val;
        bool use_fastpath;

        bitwise_cmp_Invoker(Element_t _mask_equal_val = mask_equal<Element_t>, Element_t _mask_nonequal_val = mask_nonequal<Element_t>) noexcept
            : mask_equal_val(_mask_equal_val)
            , mask_nonequal_val(_mask_nonequal_val)
            , use_fastpath((mask_equal_val == static_cast<i8>(~0)) && (mask_nonequal_val == 0)) { }

        template<typename vcmp>
        void operator () (const Element_t* cmp_0, Element_t cmp_1, Element_t* res_mask, usize count, vcmp&& vcmpexpr) noexcept
        {
            constexpr usize alignment = 32;
            constexpr usize vec_size = alignment / sizeof(Element_t);

            const uptr_t res_addr = reinterpret_cast<uptr_t>(res_mask);
            const uptr_t align_mask = alignment - 1;

            scalar_op sop{};

            usize i = 0;
            if (res_addr & align_mask)
            {
                const usize align_count = std::min(count, alignment - (res_addr & align_mask));

                for (; i < align_count; ++i)
                {
                    res_mask[i] = (sop(cmp_0[i], cmp_1)) ? mask_equal_val : mask_nonequal_val;
                }

                cmp_0 += align_count;
                res_mask += align_count;
                count -= align_count;
                i = 0;
            }

            const usize aligned_count = count & ~(vec_size - 1);

            using namespace simd;
            AVX_t<Element_t> v_cmp1(cmp_1);
            if (use_fastpath)
            {
                for (; i < aligned_count; i += vec_size)
                {
                    (vcmpexpr(AVX_t<Element_t>(cmp_0 + i), v_cmp1)).download(res_mask + i);
                }
            }
            else
            {
                const AVX_t<Element_t> v_equal(mask_equal_val);
                const AVX_t<Element_t> v_neq(mask_nonequal_val);
                for (; i < aligned_count; i += vec_size)
                {
                    AVX_t<Element_t> mask = vcmpexpr(AVX_t<Element_t>(cmp_0 + i), v_cmp1);
                    mask = v_bitwise_OR<AVX_t<Element_t>>(v_bitwise_AND<AVX_t<Element_t>>(mask, v_equal), v_bitwise_ANDNOT<AVX_t<Element_t>>(mask, v_neq));
                    mask.download(res_mask + i);
                }
            }

            for (; i < count; ++i)
            {
                res_mask[i] = (sop(cmp_0[i], cmp_1)) ? mask_equal_val : mask_nonequal_val;
            }
        }



        template<typename vcmp>
        void operator () (const Element_t* cmp_0, const Element_t* cmp_1, Element_t* res_mask, usize count, vcmp&& vcmpexpr) noexcept
        {
            constexpr usize alignment = 32;
            constexpr usize vec_size = alignment / sizeof(Element_t);

            const uptr_t res_addr = reinterpret_cast<uptr_t>(res_mask);
            const uptr_t align_mask = alignment - 1;

            scalar_op sop{};

            usize i = 0;
            if (res_addr & align_mask)
            {
                const usize align_count = std::min(count, alignment - (res_addr & align_mask));

                for (; i < align_count; ++i)
                {
                    res_mask[i] = (sop(cmp_0[i], cmp_1[i])) ? mask_equal_val : mask_nonequal_val;
                }

                cmp_0 += align_count;
                cmp_1 += align_count;
                res_mask += align_count;
                count -= align_count;
                i = 0;
            }

            const usize aligned_count = count & ~(vec_size - 1);

            using namespace simd;
            if (use_fastpath)
            {
                for (; i < aligned_count; i += vec_size)
                {
                    (vcmpexpr(AVX_t<Element_t>(cmp_0 + i), AVX_t<Element_t>(cmp_1 + i))).download(res_mask + i);
                }
            }
            else
            {
                const AVX_t<Element_t> v_equal(mask_equal_val);
                const AVX_t<Element_t> v_neq(mask_nonequal_val);
                for (; i < aligned_count; i += vec_size)
                {
                    AVX_t<Element_t> mask = vcmpexpr(AVX_t<Element_t>(cmp_0 + i), AVX_t<Element_t>(cmp_1 + i));
                    mask = v_bitwise_OR<AVX_t<Element_t>>(v_bitwise_AND<AVX_t<Element_t>>(mask, v_equal), v_bitwise_ANDNOT<AVX_t<Element_t>>(mask, v_neq));
                    mask.download(res_mask + i);
                }
            }

            for (; i < count; ++i)
            {
                res_mask[i] = (sop(cmp_0[i], cmp_1[i])) ? mask_equal_val : mask_nonequal_val;
            }
        }
    };

    template<BasicArithmetic Element_t> void equal(const Element_t* cmp_0, const Element_t* cmp_1, Element_t* res_mask, usize count, 
        Element_t mask_equal_val, Element_t mask_nonequal_val) noexcept
    {
        bitwise_cmp_Invoker<Element_t, std::equal_to<Element_t>> invoker(mask_equal_val, mask_nonequal_val);
        invoker(cmp_0, cmp_1, res_mask, count,
            []<simd::VectorType Vec_t>(const Vec_t & left, const Vec_t & right) -> Vec_t { return simd::v_eq(left, right); }
        );
    }

    template<BasicArithmetic Element_t> void not_equal(const Element_t* cmp_0, const Element_t* cmp_1, Element_t* res_mask, usize count,
        Element_t mask_equal_val, Element_t mask_nonequal_val) noexcept
    {
        bitwise_cmp_Invoker<Element_t, std::not_equal_to<Element_t>> invoker(mask_equal_val, mask_nonequal_val);
        invoker(cmp_0, cmp_1, res_mask, count,
            []<simd::VectorType Vec_t>(const Vec_t & left, const Vec_t & right) -> Vec_t { return simd::v_ne(left, right); }
        );
    }

    template<BasicArithmetic Element_t> void less(const Element_t* cmp_0, const Element_t* cmp_1, Element_t* res_mask, usize count,
        Element_t mask_equal_val, Element_t mask_nonequal_val) noexcept
    {
        bitwise_cmp_Invoker<Element_t, std::less<Element_t>> invoker(mask_equal_val, mask_nonequal_val);
        invoker(cmp_0, cmp_1, res_mask, count,
            []<simd::VectorType Vec_t>(const Vec_t & left, const Vec_t & right) -> Vec_t { return simd::v_lt(left, right); }
        );
    }

    template<BasicArithmetic Element_t> void less_equal(const Element_t* cmp_0, const Element_t* cmp_1, Element_t* res_mask, usize count,
        Element_t mask_equal_val, Element_t mask_nonequal_val) noexcept
    {
        bitwise_cmp_Invoker<Element_t, std::less_equal<Element_t>> invoker(mask_equal_val, mask_nonequal_val);
        invoker(cmp_0, cmp_1, res_mask, count,
            []<simd::VectorType Vec_t>(const Vec_t & left, const Vec_t & right) -> Vec_t { return simd::v_le(left, right); }
        );
    }

    template<BasicArithmetic Element_t> void greater(const Element_t* cmp_0, const Element_t* cmp_1, Element_t* res_mask, usize count,
        Element_t mask_equal_val, Element_t mask_nonequal_val) noexcept
    {
        bitwise_cmp_Invoker<Element_t, std::greater<Element_t>> invoker(mask_equal_val, mask_nonequal_val);
        invoker(cmp_0, cmp_1, res_mask, count,
            []<simd::VectorType Vec_t>(const Vec_t & left, const Vec_t & right) -> Vec_t { return simd::v_gt(left, right); }
        );
    }

    template<BasicArithmetic Element_t> void greater_equal(const Element_t* cmp_0, const Element_t* cmp_1, Element_t* res_mask, usize count,
        Element_t mask_equal_val, Element_t mask_nonequal_val) noexcept
    {
        bitwise_cmp_Invoker<Element_t, std::greater_equal<Element_t>> invoker(mask_equal_val, mask_nonequal_val);
        invoker(cmp_0, cmp_1, res_mask, count,
            []<simd::VectorType Vec_t>(const Vec_t & left, const Vec_t & right) -> Vec_t { return simd::v_ge(left, right); }
        );
    }

    template<BasicArithmetic Element_t>
    void equal(const Element_t* cmp_0, Element_t cmp_1, Element_t* res_mask, usize count, Element_t mask_equal_val, Element_t mask_nonequal_val) noexcept
    {
        bitwise_cmp_Invoker<Element_t, std::equal_to<Element_t>> invoker(mask_equal_val, mask_nonequal_val);
        invoker(cmp_0, cmp_1, res_mask, count,
            []<simd::VectorType Vec_t>(const Vec_t & left, const Vec_t & right) -> Vec_t { return simd::v_eq(left, right); }
        );
    }

    template<BasicArithmetic Element_t>
    void not_equal(const Element_t* cmp_0, Element_t cmp_1, Element_t* res_mask, usize count, Element_t mask_equal_val, Element_t mask_nonequal_val) noexcept
    {
        bitwise_cmp_Invoker<Element_t, std::not_equal_to<Element_t>> invoker(mask_equal_val, mask_nonequal_val);
        invoker(cmp_0, cmp_1, res_mask, count,
            []<simd::VectorType Vec_t>(const Vec_t & left, const Vec_t & right) -> Vec_t { return simd::v_ne(left, right); }
        );
    }

    template<BasicArithmetic Element_t>
    void less(const Element_t* cmp_0, Element_t cmp_1, Element_t* res_mask, usize count, Element_t mask_equal_val, Element_t mask_nonequal_val) noexcept
    {
        bitwise_cmp_Invoker<Element_t, std::less<Element_t>> invoker(mask_equal_val, mask_nonequal_val);
        invoker(cmp_0, cmp_1, res_mask, count,
            []<simd::VectorType Vec_t>(const Vec_t & left, const Vec_t & right) -> Vec_t { return simd::v_lt(left, right); }
        );
    }

    template<BasicArithmetic Element_t>
    void less_equal(const Element_t* cmp_0, Element_t cmp_1, Element_t* res_mask, usize count, Element_t mask_equal_val, Element_t mask_nonequal_val) noexcept
    {
        bitwise_cmp_Invoker<Element_t, std::less_equal<Element_t>> invoker(mask_equal_val, mask_nonequal_val);
        invoker(cmp_0, cmp_1, res_mask, count,
            []<simd::VectorType Vec_t>(const Vec_t & left, const Vec_t & right) -> Vec_t { return simd::v_le(left, right); }
        );
    }

    template<BasicArithmetic Element_t>
    void greater(const Element_t* cmp_0, Element_t cmp_1, Element_t* res_mask, usize count, Element_t mask_equal_val, Element_t mask_nonequal_val) noexcept
    {
        bitwise_cmp_Invoker<Element_t, std::greater<Element_t>> invoker(mask_equal_val, mask_nonequal_val);
        invoker(cmp_0, cmp_1, res_mask, count,
            []<simd::VectorType Vec_t>(const Vec_t & left, const Vec_t & right) -> Vec_t { return simd::v_gt(left, right); }
        );
    }

    template<BasicArithmetic Element_t>
    void greater_equal(const Element_t* cmp_0, Element_t cmp_1, Element_t* res_mask, usize count, Element_t mask_equal_val, Element_t mask_nonequal_val) noexcept
    {
        bitwise_cmp_Invoker<Element_t, std::greater_equal<Element_t>> invoker(mask_equal_val, mask_nonequal_val);
        invoker(cmp_0, cmp_1, res_mask, count,
            []<simd::VectorType Vec_t>(const Vec_t & left, const Vec_t & right) -> Vec_t { return simd::v_ge(left, right); }
        );
    }

    template void equal<u8>(const u8*, const u8*, u8*, usize, u8, u8) noexcept;
    template void equal<u16>(const u16*, const u16*, u16*, usize, u16, u16) noexcept;
    template void equal<u32>(const u32*, const u32*, u32*, usize, u32, u32) noexcept;
    template void equal<u64>(const u64*, const u64*, u64*, usize, u64, u64) noexcept;
    template void equal<i8>(const i8*, const i8*, i8*, usize, i8, i8) noexcept;
    template void equal<i16>(const i16*, const i16*, i16*, usize, i16, i16) noexcept;
    template void equal<i32>(const i32*, const i32*, i32*, usize, i32, i32) noexcept;
    template void equal<i64>(const i64*, const i64*, i64*, usize, i64, i64) noexcept;
    template void equal<f16>(const f16*, const f16*, f16*, usize, f16, f16) noexcept;
    template void equal<f32>(const f32*, const f32*, f32*, usize, f32, f32) noexcept;
    template void equal<f64>(const f64*, const f64*, f64*, usize, f64, f64) noexcept;

    template void not_equal<u8>(const u8*, const u8*, u8*, usize, u8, u8) noexcept;
    template void not_equal<u16>(const u16*, const u16*, u16*, usize, u16, u16) noexcept;
    template void not_equal<u32>(const u32*, const u32*, u32*, usize, u32, u32) noexcept;
    template void not_equal<u64>(const u64*, const u64*, u64*, usize, u64, u64) noexcept;
    template void not_equal<i8>(const i8*, const i8*, i8*, usize, i8, i8) noexcept;
    template void not_equal<i16>(const i16*, const i16*, i16*, usize, i16, i16) noexcept;
    template void not_equal<i32>(const i32*, const i32*, i32*, usize, i32, i32) noexcept;
    template void not_equal<i64>(const i64*, const i64*, i64*, usize, i64, i64) noexcept;
    template void not_equal<f16>(const f16*, const f16*, f16*, usize, f16, f16) noexcept;
    template void not_equal<f32>(const f32*, const f32*, f32*, usize, f32, f32) noexcept;
    template void not_equal<f64>(const f64*, const f64*, f64*, usize, f64, f64) noexcept;

    template void less<u8>(const u8*, const u8*, u8*, usize, u8, u8) noexcept;
    template void less<u16>(const u16*, const u16*, u16*, usize, u16, u16) noexcept;
    template void less<u32>(const u32*, const u32*, u32*, usize, u32, u32) noexcept;
    template void less<u64>(const u64*, const u64*, u64*, usize, u64, u64) noexcept;
    template void less<i8>(const i8*, const i8*, i8*, usize, i8, i8) noexcept;
    template void less<i16>(const i16*, const i16*, i16*, usize, i16, i16) noexcept;
    template void less<i32>(const i32*, const i32*, i32*, usize, i32, i32) noexcept;
    template void less<i64>(const i64*, const i64*, i64*, usize, i64, i64) noexcept;
    template void less<f16>(const f16*, const f16*, f16*, usize, f16, f16) noexcept;
    template void less<f32>(const f32*, const f32*, f32*, usize, f32, f32) noexcept;
    template void less<f64>(const f64*, const f64*, f64*, usize, f64, f64) noexcept;

    template void less_equal<u8>(const u8*, const u8*, u8*, usize, u8, u8) noexcept;
    template void less_equal<u16>(const u16*, const u16*, u16*, usize, u16, u16) noexcept;
    template void less_equal<u32>(const u32*, const u32*, u32*, usize, u32, u32) noexcept;
    template void less_equal<u64>(const u64*, const u64*, u64*, usize, u64, u64) noexcept;
    template void less_equal<i8>(const i8*, const i8*, i8*, usize, i8, i8) noexcept;
    template void less_equal<i16>(const i16*, const i16*, i16*, usize, i16, i16) noexcept;
    template void less_equal<i32>(const i32*, const i32*, i32*, usize, i32, i32) noexcept;
    template void less_equal<i64>(const i64*, const i64*, i64*, usize, i64, i64) noexcept;
    template void less_equal<f16>(const f16*, const f16*, f16*, usize, f16, f16) noexcept;
    template void less_equal<f32>(const f32*, const f32*, f32*, usize, f32, f32) noexcept;
    template void less_equal<f64>(const f64*, const f64*, f64*, usize, f64, f64) noexcept;

    template void greater<u8>(const u8*, const u8*, u8*, usize, u8, u8) noexcept;
    template void greater<u16>(const u16*, const u16*, u16*, usize, u16, u16) noexcept;
    template void greater<u32>(const u32*, const u32*, u32*, usize, u32, u32) noexcept;
    template void greater<u64>(const u64*, const u64*, u64*, usize, u64, u64) noexcept;
    template void greater<i8>(const i8*, const i8*, i8*, usize, i8, i8) noexcept;
    template void greater<i16>(const i16*, const i16*, i16*, usize, i16, i16) noexcept;
    template void greater<i32>(const i32*, const i32*, i32*, usize, i32, i32) noexcept;
    template void greater<i64>(const i64*, const i64*, i64*, usize, i64, i64) noexcept;
    template void greater<f16>(const f16*, const f16*, f16*, usize, f16, f16) noexcept;
    template void greater<f32>(const f32*, const f32*, f32*, usize, f32, f32) noexcept;
    template void greater<f64>(const f64*, const f64*, f64*, usize, f64, f64) noexcept;

    template void greater_equal<u8>(const u8*, const u8*, u8*, usize, u8, u8) noexcept;
    template void greater_equal<u16>(const u16*, const u16*, u16*, usize, u16, u16) noexcept;
    template void greater_equal<u32>(const u32*, const u32*, u32*, usize, u32, u32) noexcept;
    template void greater_equal<u64>(const u64*, const u64*, u64*, usize, u64, u64) noexcept;
    template void greater_equal<i8>(const i8*, const i8*, i8*, usize, i8, i8) noexcept;
    template void greater_equal<i16>(const i16*, const i16*, i16*, usize, i16, i16) noexcept;
    template void greater_equal<i32>(const i32*, const i32*, i32*, usize, i32, i32) noexcept;
    template void greater_equal<i64>(const i64*, const i64*, i64*, usize, i64, i64) noexcept;
    template void greater_equal<f16>(const f16*, const f16*, f16*, usize, f16, f16) noexcept;
    template void greater_equal<f32>(const f32*, const f32*, f32*, usize, f32, f32) noexcept;
    template void greater_equal<f64>(const f64*, const f64*, f64*, usize, f64, f64) noexcept;

    template void equal<u8>(const u8*, u8, u8*, usize, u8, u8) noexcept;
    template void equal<u16>(const u16*, u16, u16*, usize, u16, u16) noexcept;
    template void equal<u32>(const u32*, u32, u32*, usize, u32, u32) noexcept;
    template void equal<u64>(const u64*, u64, u64*, usize, u64, u64) noexcept;
    template void equal<i8>(const i8*, i8, i8*, usize, i8, i8) noexcept;
    template void equal<i16>(const i16*, i16, i16*, usize, i16, i16) noexcept;
    template void equal<i32>(const i32*, i32, i32*, usize, i32, i32) noexcept;
    template void equal<i64>(const i64*, i64, i64*, usize, i64, i64) noexcept;
    template void equal<f16>(const f16*, f16, f16*, usize, f16, f16) noexcept;
    template void equal<f32>(const f32*, f32, f32*, usize, f32, f32) noexcept;
    template void equal<f64>(const f64*, f64, f64*, usize, f64, f64) noexcept;

    template void not_equal<u8>(const u8*, u8, u8*, usize, u8, u8) noexcept;
    template void not_equal<u16>(const u16*, u16, u16*, usize, u16, u16) noexcept;
    template void not_equal<u32>(const u32*, u32, u32*, usize, u32, u32) noexcept;
    template void not_equal<u64>(const u64*, u64, u64*, usize, u64, u64) noexcept;
    template void not_equal<i8>(const i8*, i8, i8*, usize, i8, i8) noexcept;
    template void not_equal<i16>(const i16*, i16, i16*, usize, i16, i16) noexcept;
    template void not_equal<i32>(const i32*, i32, i32*, usize, i32, i32) noexcept;
    template void not_equal<i64>(const i64*, i64, i64*, usize, i64, i64) noexcept;
    template void not_equal<f16>(const f16*, f16, f16*, usize, f16, f16) noexcept;
    template void not_equal<f32>(const f32*, f32, f32*, usize, f32, f32) noexcept;
    template void not_equal<f64>(const f64*, f64, f64*, usize, f64, f64) noexcept;

    template void less<u8>(const u8*, u8, u8*, usize, u8, u8) noexcept;
    template void less<u16>(const u16*, u16, u16*, usize, u16, u16) noexcept;
    template void less<u32>(const u32*, u32, u32*, usize, u32, u32) noexcept;
    template void less<u64>(const u64*, u64, u64*, usize, u64, u64) noexcept;
    template void less<i8>(const i8*, i8, i8*, usize, i8, i8) noexcept;
    template void less<i16>(const i16*, i16, i16*, usize, i16, i16) noexcept;
    template void less<i32>(const i32*, i32, i32*, usize, i32, i32) noexcept;
    template void less<i64>(const i64*, i64, i64*, usize, i64, i64) noexcept;
    template void less<f16>(const f16*, f16, f16*, usize, f16, f16) noexcept;
    template void less<f32>(const f32*, f32, f32*, usize, f32, f32) noexcept;
    template void less<f64>(const f64*, f64, f64*, usize, f64, f64) noexcept;

    template void less_equal<u8>(const u8*, u8, u8*, usize, u8, u8) noexcept;
    template void less_equal<u16>(const u16*, u16, u16*, usize, u16, u16) noexcept;
    template void less_equal<u32>(const u32*, u32, u32*, usize, u32, u32) noexcept;
    template void less_equal<u64>(const u64*, u64, u64*, usize, u64, u64) noexcept;
    template void less_equal<i8>(const i8*, i8, i8*, usize, i8, i8) noexcept;
    template void less_equal<i16>(const i16*, i16, i16*, usize, i16, i16) noexcept;
    template void less_equal<i32>(const i32*, i32, i32*, usize, i32, i32) noexcept;
    template void less_equal<i64>(const i64*, i64, i64*, usize, i64, i64) noexcept;
    template void less_equal<f16>(const f16*, f16, f16*, usize, f16, f16) noexcept;
    template void less_equal<f32>(const f32*, f32, f32*, usize, f32, f32) noexcept;
    template void less_equal<f64>(const f64*, f64, f64*, usize, f64, f64) noexcept;

    template void greater<u8>(const u8*, u8, u8*, usize, u8, u8) noexcept;
    template void greater<u16>(const u16*, u16, u16*, usize, u16, u16) noexcept;
    template void greater<u32>(const u32*, u32, u32*, usize, u32, u32) noexcept;
    template void greater<u64>(const u64*, u64, u64*, usize, u64, u64) noexcept;
    template void greater<i8>(const i8*, i8, i8*, usize, i8, i8) noexcept;
    template void greater<i16>(const i16*, i16, i16*, usize, i16, i16) noexcept;
    template void greater<i32>(const i32*, i32, i32*, usize, i32, i32) noexcept;
    template void greater<i64>(const i64*, i64, i64*, usize, i64, i64) noexcept;
    template void greater<f16>(const f16*, f16, f16*, usize, f16, f16) noexcept;
    template void greater<f32>(const f32*, f32, f32*, usize, f32, f32) noexcept;
    template void greater<f64>(const f64*, f64, f64*, usize, f64, f64) noexcept;

    template void greater_equal<u8>(const u8*, u8, u8*, usize, u8, u8) noexcept;
    template void greater_equal<u16>(const u16*, u16, u16*, usize, u16, u16) noexcept;
    template void greater_equal<u32>(const u32*, u32, u32*, usize, u32, u32) noexcept;
    template void greater_equal<u64>(const u64*, u64, u64*, usize, u64, u64) noexcept;
    template void greater_equal<i8>(const i8*, i8, i8*, usize, i8, i8) noexcept;
    template void greater_equal<i16>(const i16*, i16, i16*, usize, i16, i16) noexcept;
    template void greater_equal<i32>(const i32*, i32, i32*, usize, i32, i32) noexcept;
    template void greater_equal<i64>(const i64*, i64, i64*, usize, i64, i64) noexcept;
    template void greater_equal<f16>(const f16*, f16, f16*, usize, f16, f16) noexcept;
    template void greater_equal<f32>(const f32*, f32, f32*, usize, f32, f32) noexcept;
    template void greater_equal<f64>(const f64*, f64, f64*, usize, f64, f64) noexcept;
}

namespace fy
{
    template<integral_arithmetic Element_t>
    void close(const Element_t* cmp_0, const Element_t* cmp_1, Element_t* res_mask, usize count, 
        Element_t tolerance,
        Element_t mask_matched_val, 
        Element_t mask_nomatched_val) noexcept
    {
        for (usize i = 0; i < count; ++i)
        {
            Element_t src_0 = cmp_0[i];
            Element_t src_1 = cmp_1[i];
            bool is_close = false;

            if constexpr (std::is_unsigned_v<Element_t>)
            {
                is_close = (src_0 >= src_1) ?
                    (src_0 - src_1 <= tolerance) :
                    (src_1 - src_0 <= tolerance);
            }
            else
            {
                if (src_0 >= src_1)
                {
                    if (src_1 >= 0 || src_0 <= std::numeric_limits<Element_t>::max() + src_1)
                    {
                        Element_t diff = src_0 - src_1;
                        is_close = (diff <= tolerance);
                    }
                    else
                    {
                        is_close = false;
                    }
                }
                else
                {
                    if (src_0 >= 0 || src_1 <= std::numeric_limits<Element_t>::max() + src_0)
                    {
                        Element_t diff = src_1 - src_0;
                        is_close = (diff <= tolerance);
                    }
                    else
                    {
                        is_close = false;
                    }
                }
            }

            res_mask[i] = is_close ? mask_matched_val : mask_nomatched_val;
        }
    }

    template<typename Element_t> void dispatch_close_u8u16____(
        const Element_t* cmp_0, const Element_t* cmp_1, Element_t* res_mask, usize count,
        Element_t tolerance, Element_t mask_matched_val, Element_t mask_nomatched_val) noexcept
    {
        static_assert(std::is_same_v<Element_t, u8> || std::is_same_v<Element_t, u16>);
        if (tolerance == Element_t{ 0 })
        {
            return equal<Element_t>(cmp_0, cmp_1, res_mask, count,
                mask_matched_val, mask_nomatched_val);
        }
        
        const simd::AVX_t<Element_t> v_tol(tolerance);
        const simd::AVX_t<Element_t> v_zero = simd::v_broadcast_zero<simd::AVX_t<Element_t>>();

        const bool default_masks = (mask_matched_val == mask_equal<Element_t>) &&
            (mask_nomatched_val == mask_nonequal<Element_t>);

        usize i = 0;
        const uintptr_t res_mask_ptr = reinterpret_cast<uintptr_t>(res_mask);
        const usize misalign = res_mask_ptr & 0x1F;

        if (misalign > 0)
        {
            const usize align_bytes = 32 - misalign;
            const usize align_elems = (align_bytes + sizeof(Element_t) - 1) / sizeof(Element_t);

            const usize head_end = i + std::min(align_elems, count);

            for (; i < head_end; ++i)
            {
                Element_t a = cmp_0[i];
                Element_t b = cmp_1[i];
                Element_t diff = (a >= b) ? (a - b) : (b - a);
                res_mask[i] = (diff <= tolerance) ? mask_matched_val : mask_nomatched_val;
            }
        }

        const simd::AVX_t<Element_t> v_matched(mask_matched_val);
        const simd::AVX_t<Element_t> v_nomatched(mask_nomatched_val);

        for (; i + (simd::AVX_t<Element_t>::batch_size - 1) < count; i += simd::AVX_t<Element_t>::batch_size)
        {
            const simd::AVX_t<Element_t> a(cmp_0 + i);
            const simd::AVX_t<Element_t> b(cmp_1 + i);

            const simd::AVX_t<Element_t> diff_ab = simd::v_subs(a, b);
            const simd::AVX_t<Element_t> diff_ba = simd::v_subs(b, a);
            const simd::AVX_t<Element_t> abs_diff = simd::v_bitwise_OR(diff_ab, diff_ba);

            const simd::AVX_t<Element_t> cmp = simd::v_subs(abs_diff, v_tol);
            const simd::AVX_t<Element_t> mask = simd::v_eq(cmp, v_zero);

            simd::AVX_t<Element_t> res;
            if (default_masks)
            {
                res = mask;
            }
            else
            {
                res = simd::AVX_t<Element_t>(_mm256_blendv_epi8(v_nomatched.data, v_matched.data, mask.data));
            }
            res.streamback(res_mask + i);
        }

        _mm_sfence();

        for (; i < count; ++i)
        {
            Element_t a = cmp_0[i];
            Element_t b = cmp_1[i];
            Element_t diff = (a >= b) ? (a - b) : (b - a);
            res_mask[i] = (diff <= tolerance) ? mask_matched_val : mask_nomatched_val;
        }

    }

    template <> void close(const u8* cmp_0, const u8* cmp_1, u8* res_mask, usize count,
        u8 tolerance, u8 mask_matched_val, u8 mask_nomatched_val) noexcept
    {
        return dispatch_close_u8u16____<u8>(
            cmp_0, cmp_1, res_mask,
            count, tolerance, mask_matched_val, mask_nomatched_val
        );
    }

    template <> void close(const u16* cmp_0, const u16* cmp_1, u16* res_mask, usize count,
        u16 tolerance, u16 mask_matched_val, u16 mask_nomatched_val) noexcept
    {
        return dispatch_close_u8u16____<u16>(
            cmp_0, cmp_1, res_mask,
            count, tolerance, mask_matched_val, mask_nomatched_val
        );
    }

    template <> void close(const u32* cmp_0, const u32* cmp_1, u32* res_mask, usize count,
        u32 tolerance, u32 mask_matched_val, u32 mask_nomatched_val) noexcept
    {
        if (tolerance == u32{ 0 })
        {
            return equal(cmp_0, cmp_1, res_mask, count,
                mask_matched_val, mask_nomatched_val);
        }

        const __m256i v_tol = _mm256_set1_epi32(tolerance);
        const __m256i v_zero = _mm256_setzero_si256();

        const bool default_masks = (mask_matched_val == mask_equal<u32>) &&
            (mask_nomatched_val == mask_nonequal<u32>);

        usize i = 0;
        const uintptr_t res_mask_ptr = reinterpret_cast<uintptr_t>(res_mask);
        const usize misalign = res_mask_ptr & 0x1F;

        if (misalign > 0)
        {
            const usize align_bytes = 32 - misalign;
            const usize align_elems = (align_bytes + sizeof(u32) - 1) / sizeof(u32);
            const usize head_end = std::min(align_elems, count);

            for (; i < head_end; ++i)
            {
                u32 a = cmp_0[i];
                u32 b = cmp_1[i];
                u32 diff = (a >= b) ? (a - b) : (b - a);
                res_mask[i] = (diff <= tolerance) ? mask_matched_val : mask_nomatched_val;
            }
        }

        const __m256i v_matched = _mm256_set1_epi32(mask_matched_val);
        const __m256i v_nomatched = _mm256_set1_epi32(mask_nomatched_val);

        for (; i + 7 < count; i += 8)
        {
            __m256i a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(cmp_0 + i));
            __m256i b = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(cmp_1 + i));

            __m256i max_val = _mm256_max_epu32(a, b);
            __m256i min_val = _mm256_min_epu32(a, b);
            __m256i abs_diff = _mm256_sub_epi32(max_val, min_val);

            __m256i cmp_result = _mm256_sub_epi32(abs_diff, v_tol);
            __m256i underflow_mask = _mm256_cmpgt_epi32(v_zero, cmp_result);
            __m256i is_less_or_equal = _mm256_or_si256(
                _mm256_cmpeq_epi32(cmp_result, v_zero),
                underflow_mask
            );

            __m256i result;
            if (default_masks)
            {
                result = is_less_or_equal;
            }
            else
            {
                result = _mm256_blendv_epi8(
                    v_nomatched,
                    v_matched,
                    is_less_or_equal
                );
            }
            _mm256_stream_si256(reinterpret_cast<__m256i*>(res_mask + i), result);
        }

        for (; i < count; ++i)
        {
            u32 a = cmp_0[i];
            u32 b = cmp_1[i];
            u32 diff = (a >= b) ? (a - b) : (b - a);
            res_mask[i] = (diff <= tolerance) ? mask_matched_val : mask_nomatched_val;
        }
    }

    template void close<u64>(const u64*, const u64*, u64*, usize, u64, u64, u64) noexcept;

    template void close<i16>(const i16*, const i16*, i16*, usize, i16, i16, i16) noexcept;
    template void close<i32>(const i32*, const i32*, i32*, usize, i32, i32, i32) noexcept;
    template void close<i64>(const i64*, const i64*, i64*, usize, i64, i64, i64) noexcept;
}

namespace fy
{
    template<Floating_arithmetic Element_t>
    void close(
        const Element_t* cmp_0, const Element_t* cmp_1, Element_t* res_mask, usize count,
        Element_t eqval, Element_t neqval,
        f64 rtol, f64 atol, bool nan_is_eq) noexcept
    {
        std::unreachable();
    }

    template <> void close<f16>(
        const f16* cmp_0, const f16* cmp_1, f16* res_mask, usize count,
        f16 eqval, f16 neqval,
        f64 rtol, f64 atol, bool nan_is_eq) noexcept
    {
        const f32 rtol_f = static_cast<f32>(rtol);
        const f32 atol_f = static_cast<f32>(atol);

        const f32 eqval_f = static_cast<f32>(eqval);
        const f32 neqval_f = static_cast<f32>(neqval);

        const __m256 v_eqval = _mm256_set1_ps(eqval_f);
        const __m256 v_neqval = _mm256_set1_ps(neqval_f);
        const __m256 v_rtol = _mm256_set1_ps(rtol_f);
        const __m256 v_atol = _mm256_set1_ps(atol_f);
        const __m256 v_sign_mask = _mm256_set1_ps(-0.0f);

        usize i = 0;
        const usize aligned_count = count & ~static_cast<usize>(7);

        for (; i < aligned_count; i += 8)
        {
            const __m128i v_cmp0 = _mm_load_si128(reinterpret_cast<const __m128i*>(cmp_0 + i));
            const __m128i v_cmp1 = _mm_load_si128(reinterpret_cast<const __m128i*>(cmp_1 + i));

            __m256 a = _mm256_cvtph_ps(v_cmp0);
            __m256 b = _mm256_cvtph_ps(v_cmp1);

            __m256 a_nan = _mm256_cmp_ps(a, a, _CMP_NEQ_UQ);
            __m256 b_nan = _mm256_cmp_ps(b, b, _CMP_NEQ_UQ);
            __m256 nan_mask = _mm256_or_ps(a_nan, b_nan);

            __m256 result;
            if (nan_is_eq)
            {
                __m256 both_nan = _mm256_and_ps(a_nan, b_nan);
                result = _mm256_blendv_ps(v_neqval, v_eqval, both_nan);
            }
            else
            {
                result = v_neqval;
            }

            __m256 diff = _mm256_sub_ps(a, b);
            diff = _mm256_andnot_ps(v_sign_mask, diff);
            __m256 abs_b = _mm256_andnot_ps(v_sign_mask, b);

            __m256 tol = _mm256_fmadd_ps(v_rtol, abs_b, v_atol);
            __m256 close_mask = _mm256_cmp_ps(diff, tol, _CMP_LE_OQ);

            __m256 valid_result = _mm256_blendv_ps(v_neqval, v_eqval, close_mask);
            result = _mm256_blendv_ps(valid_result, result, nan_mask);

            __m128i res_f16 = _mm256_cvtps_ph(result, _MM_FROUND_CUR_DIRECTION);
            _mm_store_si128(reinterpret_cast<__m128i*>(res_mask + i), res_f16);
        }

        for (; i < count; ++i)
        {
            const f32 a = static_cast<f32>(cmp_0[i]);
            const f32 b = static_cast<f32>(cmp_1[i]);

            if (std::isnan(a) || std::isnan(b))
            {
                res_mask[i] = (nan_is_eq && std::isnan(a) && std::isnan(b)) ? eqval : neqval;
            }
            else
            {
                const f32 diff = std::abs(a - b);
                const f32 abs_b = std::abs(b);
                const bool is_close = diff <= (rtol_f * abs_b + atol_f);
                res_mask[i] = is_close ? eqval : neqval;
            }
        }
    }

    template <> void close<f32>(
        const f32* cmp_0, const f32* cmp_1, f32* res_mask, usize count,
        f32 eqval, f32 neqval,
        f64 rtol, f64 atol, bool nan_is_eq) noexcept
    {
        const f32 rtol_f = static_cast<f32>(rtol);
        const f32 atol_f = static_cast<f32>(atol);

        const __m256 v_eqval = _mm256_set1_ps(eqval);
        const __m256 v_neqval = _mm256_set1_ps(neqval);
        const __m256 v_rtol = _mm256_set1_ps(rtol_f);
        const __m256 v_atol = _mm256_set1_ps(atol_f);
        const __m256 v_sign_mask = _mm256_set1_ps(-0.0f);
        const __m256 v_nan_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));

        usize i = 0;
        const usize aligned_count = count & ~static_cast<usize>(7);

        for (; i < aligned_count; i += 8)
        {
            __m256 a = _mm256_load_ps(cmp_0 + i);
            __m256 b = _mm256_load_ps(cmp_1 + i);

            __m256 a_nan = _mm256_cmp_ps(a, a, _CMP_NEQ_UQ);
            __m256 b_nan = _mm256_cmp_ps(b, b, _CMP_NEQ_UQ);
            __m256 nan_mask = _mm256_or_ps(a_nan, b_nan);

            __m256 result;
            if (nan_is_eq)
            {
                __m256 both_nan = _mm256_and_ps(a_nan, b_nan);
                result = _mm256_blendv_ps(v_neqval, v_eqval, both_nan);
            }
            else
            {
                result = v_neqval;
            }

            __m256 diff = _mm256_sub_ps(a, b);
            diff = _mm256_andnot_ps(v_sign_mask, diff);
            __m256 abs_b = _mm256_andnot_ps(v_sign_mask, b);

            __m256 tol = _mm256_fmadd_ps(v_rtol, abs_b, v_atol);
            __m256 close_mask = _mm256_cmp_ps(diff, tol, _CMP_LE_OQ);

            __m256 valid_result = _mm256_blendv_ps(v_neqval, v_eqval, close_mask);
            result = _mm256_blendv_ps(valid_result, result, nan_mask);

            _mm256_store_ps(res_mask + i, result);
        }

        for (; i < count; ++i)
        {
            const f32 a = cmp_0[i];
            const f32 b = cmp_1[i];

            if (std::isnan(a) || std::isnan(b))
            {
                res_mask[i] = (nan_is_eq && std::isnan(a) && std::isnan(b)) ? eqval : neqval;
            }
            else
            {
                const f32 diff = std::abs(a - b);
                const f32 abs_b = std::abs(b);
                const bool is_close = diff <= (rtol_f * abs_b + atol_f);
                res_mask[i] = is_close ? eqval : neqval;
            }
        }
    }

    template <> void close<f64>(
        const f64* cmp_0, const f64* cmp_1, f64* res_mask, usize count,
        f64 eqval, f64 neqval,
        f64 rtol, f64 atol, bool nan_is_eq) noexcept
    {
        const __m256d v_eqval = _mm256_set1_pd(eqval);
        const __m256d v_neqval = _mm256_set1_pd(neqval);
        const __m256d v_rtol = _mm256_set1_pd(rtol);
        const __m256d v_atol = _mm256_set1_pd(atol);
        const __m256d v_sign_mask = _mm256_set1_pd(-0.0);
        const __m256d v_all_nan = _mm256_castsi256_pd(_mm256_set1_epi64x(0x7FFFFFFFFFFFFFFF));

        usize i = 0;
        const usize aligned_count = count & ~static_cast<usize>(3);

        for (; i < aligned_count; i += 4)
        {
            __m256d a = _mm256_load_pd(cmp_0 + i);
            __m256d b = _mm256_load_pd(cmp_1 + i);

            __m256d a_nan = _mm256_cmp_pd(a, a, _CMP_NEQ_UQ);
            __m256d b_nan = _mm256_cmp_pd(b, b, _CMP_NEQ_UQ);
            __m256d nan_mask = _mm256_or_pd(a_nan, b_nan);

            __m256d result;
            if (nan_is_eq)
            {
                __m256d both_nan = _mm256_and_pd(a_nan, b_nan);
                result = _mm256_blendv_pd(v_neqval, v_eqval, both_nan);
            }
            else
            {
                result = v_neqval;
            }

            __m256d diff = _mm256_sub_pd(a, b);
            diff = _mm256_andnot_pd(v_sign_mask, diff);
            __m256d abs_b = _mm256_andnot_pd(v_sign_mask, b);

            __m256d tol = _mm256_fmadd_pd(v_rtol, abs_b, v_atol);
            __m256d close_mask = _mm256_cmp_pd(diff, tol, _CMP_LE_OQ);
            __m256d valid_result = _mm256_blendv_pd(v_neqval, v_eqval, close_mask);

            result = _mm256_blendv_pd(valid_result, result, nan_mask);
            _mm256_store_pd(res_mask + i, result);
        }

        for (; i < count; ++i)
        {
            const f64 a = cmp_0[i];
            const f64 b = cmp_1[i];

            if (std::isnan(a) || std::isnan(b))
            {
                res_mask[i] = (nan_is_eq && std::isnan(a) && std::isnan(b)) ? eqval : neqval;
            }
            else
            {
                const f64 diff = std::abs(a - b);
                const f64 abs_b = std::abs(b);
                const bool is_close = diff <= (rtol * abs_b + atol);
                res_mask[i] = is_close ? eqval : neqval;
            }
        }
    }
}

namespace fy
{
    template<BasicArithmetic Element_t>
    usize count_repeat(const Element_t* ptr, Element_t repeat, usize count) noexcept
    {
        using namespace simd;
        usize res{ 0 };

        usize i = 0;

        const AVX_t<Element_t> target_vec = (repeat == 0) ?
            v_broadcast_zero<AVX_t<Element_t>>() :
            AVX_t<Element_t>(repeat);

        for (; i + AVX_t<Element_t>::batch_size <= count; i += AVX_t<Element_t>::batch_size)
        {
            const AVX_t<Element_t> data_vec(ptr + i);
            u32 mask = _mm256_movemask_epi8(v_eq(data_vec, target_vec).data);
            res += std::popcount(mask);
        }
        
        for (; i < count; ++i)
        {
            res += (ptr[i] == repeat ? 1 : 0);
        }

        return res;
    }

    template usize count_repeat<u8>(const u8*, u8, usize) noexcept;
    template usize count_repeat<u16>(const u16*, u16, usize) noexcept;
    template usize count_repeat<u32>(const u32*, u32, usize) noexcept;
    template usize count_repeat<u64>(const u64*, u64, usize) noexcept;

    template usize count_repeat<i8>(const i8*, i8, usize) noexcept;
    template usize count_repeat<i16>(const i16*, i16, usize) noexcept;
    template usize count_repeat<i32>(const i32*, i32, usize) noexcept;
    template usize count_repeat<i64>(const i64*, i64, usize) noexcept;

    template<> usize count_repeat<f16>(const f16* ptr, f16 repeat, usize count) noexcept
    {
        return count_repeat<u16>(reinterpret_cast<const u16*>(ptr), std::bit_cast<u16>(repeat), count);
    }

    template<> usize count_repeat<f32>(const f32* ptr, f32 repeat, usize count) noexcept
    {
        return count_repeat<u32>(reinterpret_cast<const u32*>(ptr), std::bit_cast<u32>(repeat), count);
    }

    template<> usize count_repeat<f64>(const f64* ptr, f64 repeat, usize count) noexcept
    {
        return count_repeat<u64>(reinterpret_cast<const u64*>(ptr), std::bit_cast<u64>(repeat), count);
    }
}


namespace fy
{
    template<BasicArithmetic Element_t>
    void compare(const Element_t* src, usize count, Element_t* dst, Element_t to_compare_scalar,
        Element_t mask_lez_val, Element_t mask_eqz_val, Element_t mask_gtz_val)
    {
        for (usize i = 0; i < count; ++i)
        {
            Element_t val = src[i];
            if (val < to_compare_scalar)
            {
                dst[i] = mask_lez_val;
            }
            else if (val > to_compare_scalar)
            {
                dst[i] = mask_gtz_val;
            }
            else if (val == to_compare_scalar)
            {
                dst[i] = mask_eqz_val;
            }
        }
    }

    template<BasicArithmetic Element_t>
    void compare_scalar__(
        const Element_t* src, usize count, Element_t* dst, Element_t to_compare_scalar,
        Element_t mask_lez_val, Element_t mask_eqz_val, Element_t mask_gtz_val)
    {
        for (usize i = 0; i < count; ++i)
        {
            Element_t val = src[i];
            if (val < to_compare_scalar)
            {
                dst[i] = mask_lez_val;
            }
            else if (val > to_compare_scalar)
            {
                dst[i] = mask_gtz_val;
            }
            else if (val == to_compare_scalar)
            {
                dst[i] = mask_eqz_val;
            }
        }
    }

    template<> void compare<i8>(const i8* src, usize count, i8* dst, i8 to_compare_scalar,
        i8 mask_lez_val, i8 mask_eqz_val, i8 mask_gtz_val)
    {
        const __m256i v_cmp = _mm256_set1_epi8(to_compare_scalar);
        const __m256i v_lez = _mm256_set1_epi8(mask_lez_val);
        const __m256i v_eqz = _mm256_set1_epi8(mask_eqz_val);
        const __m256i v_gtz = _mm256_set1_epi8(mask_gtz_val);

        usize i = 0;
        const usize simd_chunk = count - (count % 32);
        for (; i < simd_chunk; i += 32)
        {
            __m256i data = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + i));

            __m256i mask_lt = _mm256_cmpgt_epi8(v_cmp, data);
            __m256i mask_eq = _mm256_cmpeq_epi8(data, v_cmp);
            __m256i mask_gt = _mm256_cmpgt_epi8(data, v_cmp);

            __m256i result = _mm256_blendv_epi8(v_gtz, v_lez, mask_lt);
            result = _mm256_blendv_epi8(result, v_eqz, mask_eq);

            _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + i), result);
        }

        return compare_scalar__(src + i, count - i, dst + i, to_compare_scalar,
            mask_lez_val, mask_eqz_val, mask_gtz_val);
    }
}

namespace fy
{
    template<Unsigned_integral_arithmetic Element_t>
    void abs_diff_unsigned_dispatch__(const Element_t* src_0, const Element_t* src_1, Element_t* dst, usize count) noexcept
    {
        using namespace simd;
        using simd_t = AVX_t<Element_t>;
        constexpr usize stride = simd_t::batch_size;
        usize i = 0;

        for (; i + stride <= count; i += stride)
        {
            simd_t a(src_0 + i);
            simd_t b(src_1 + i);

            simd_t max = v_max_replace(a, b);
            simd_t min = v_min_replace(a, b);
            simd_t result = v_sub(max, min);
            result.download(dst + i);
        }

        for (; i < count; ++i)
        {
            Element_t a = src_0[i];
            Element_t b = src_1[i];
            dst[i] = a > b ? a - b : b - a;
        }
    }

    template<Signed_integral_arithmetic Element_t>
    void abs_diff_signed_dispatch__(const Element_t* src_0, const Element_t* src_1, Element_t* dst, usize count) noexcept
    {
        constexpr Element_t maxval = std::numeric_limits<Element_t>::max();

        for (size_t i = 0; i < count; ++i)
        {
            const Element_t a = src_0[i];
            const Element_t b = src_1[i];

            if (a == b)
            {
                dst[i] = 0;
            }
            else if (a >= 0)
            {
                if (b >= 0)
                {
                    dst[i] = (a > b) ? (a - b) : (b - a);
                }
                else
                {
                    if (a > maxval + b)
                    {
                        dst[i] = a;
                        dst[i] -= b;
                    }
                    else
                    {
                        dst[i] = a - b;
                    }
                }
            }
            else
            {
                if (b < 0)
                {
                    dst[i] = (b > a) ? (b - a) : (a - b);
                }
                else
                {
                    if (b > maxval + a)
                    {
                        dst[i] = b;
                        dst[i] -= a;
                    }
                    else
                    {
                        dst[i] = b - a;
                    }
                }
            }
        }
    }


    template<BasicArithmetic Element_t>
    void abs_diff(const Element_t* src_0, const Element_t* src_1, Element_t* dst, usize count) noexcept
    {
        if constexpr (std::is_unsigned_v<Element_t>)
        {
            return abs_diff_unsigned_dispatch__(src_0, src_1, dst, count);
        }
    }

    template void abs_diff<u8>(const u8*, const u8*, u8*, usize) noexcept;
    template void abs_diff<u16>(const u16*, const u16*, u16*, usize) noexcept;
    template void abs_diff<u32>(const u32*, const u32*, u32*, usize) noexcept;
    template void abs_diff<u64>(const u64*, const u64*, u64*, usize) noexcept;
}