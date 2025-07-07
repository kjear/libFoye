module;
#include <immintrin.h>

module foye.algorithm;
import foye.foye_core;
import foye.simd;
import std;

namespace fy
{
    template<BasicArithmetic Element_t, typename Expr>
    void advanced_mathematics_dispacher__(const Element_t* input, Element_t* output, usize count, Expr&& expr) noexcept
    {
        if constexpr (std::is_unsigned_v<Element_t>)
        {
            std::memcpy(output, input, count * sizeof(Element_t));
            return;
        }

        constexpr usize vec_size = sizeof(__m256) / sizeof(Element_t);
        uintptr_t output_addr = reinterpret_cast<uintptr_t>(output);
        const uintptr_t align_mask = 31;

        usize unaligned_count = output_addr & align_mask
            ? vec_size - (output_addr & align_mask) / sizeof(Element_t)
            : 0;

        unaligned_count = (unaligned_count > count) ? count : unaligned_count;

        if (unaligned_count > 0)
        {
            expr(simd::AVX_t<Element_t>::loadu(input)).downloadu(output);

            input += unaligned_count;
            output += unaligned_count;
            count -= unaligned_count;
        }

        usize aligned_count = count & ~(vec_size - 1);

        usize i = 0;
        for (; i + vec_size <= count; i += vec_size)
        {
            expr(simd::AVX_t<Element_t>(input + i)).streamback(output + i);
        }

        usize remaining = count - aligned_count;
        if ((count - aligned_count) > 0)
        {
            alignas(32) Element_t tail_vec[vec_size];
            std::memcpy(tail_vec, input + aligned_count, remaining * sizeof(Element_t));
            expr(simd::AVX_t<Element_t>(tail_vec)).streamback(tail_vec);
            std::memcpy(output + aligned_count, tail_vec, remaining * sizeof(Element_t));
        }
    }
}

namespace fy
{
    template<BasicArithmetic Element_t>
    void avg(const Element_t* input_0, const Element_t* input_1, Element_t* output, usize count) noexcept
    {
        if constexpr (!std::is_same_v<Element_t, u8> || !std::is_same_v<Element_t, u16>)
        {
            for (usize i = 0; i < count; ++i)
            {
                extended_t<Element_t> temp = extended_t<Element_t>(input_0[i]) + extended_t<Element_t>(input_1[i]) + 1;
                output[i] = Element_t(temp / 2);
            }
            return;
        }

        constexpr usize vec_size = sizeof(__m256) / sizeof(Element_t);
        uintptr_t output_addr = reinterpret_cast<uintptr_t>(output);
        const uintptr_t align_mask = 31;

        usize unaligned_count = output_addr & align_mask
            ? vec_size - (output_addr & align_mask) / sizeof(Element_t)
            : 0;

        unaligned_count = (unaligned_count > count) ? count : unaligned_count;

        if (unaligned_count > 0)
        {
            simd::v_avg(
                simd::AVX_t<Element_t>::loadu(input_0),
                simd::AVX_t<Element_t>::loadu(input_1)
            ).downloadu(output);

            input_0 += unaligned_count;
            input_1 += unaligned_count;
            output += unaligned_count;
            count -= unaligned_count;
        }

        usize aligned_count = count & ~(vec_size - 1);

        usize i = 0;
        for (; i + vec_size <= count; i += vec_size)
        {
            simd::v_avg(
                simd::AVX_t<Element_t>(input_0 + i),
                simd::AVX_t<Element_t>(input_1 + i)
            ).streamback(output + i);
        }

        usize remaining = count - aligned_count;
        if (remaining > 0)
        {
            input_0 += aligned_count;
            input_1 += aligned_count;
            output += aligned_count;

            simd::v_avg(
                simd::AVX_t<Element_t>::loadu(input_0),
                simd::AVX_t<Element_t>::loadu(input_1)
            ).downloadu(output);
        }
    }

    template void avg<u8>(const u8*, const u8*, u8*, usize) noexcept;
    template void avg<u16>(const u16*, const u16*, u16*, usize) noexcept;
    template void avg<u32>(const u32*, const u32*, u32*, usize) noexcept;
    template void avg<u64>(const u64*, const u64*, u64*, usize) noexcept;

    template void avg<i8>(const i8*, const i8*, i8*, usize) noexcept;
    template void avg<i16>(const i16*, const i16*, i16*, usize) noexcept;
    template void avg<i32>(const i32*, const i32*, i32*, usize) noexcept;
    template void avg<i64>(const i64*, const i64*, i64*, usize) noexcept;

    template void avg<f16>(const f16*, const f16*, f16*, usize) noexcept;
    template void avg<f32>(const f32*, const f32*, f32*, usize) noexcept;
    template void avg<f64>(const f64*, const f64*, f64*, usize) noexcept;
}

namespace fy
{
    template<BasicArithmetic Element_t>
    void abs(const Element_t* input, Element_t* output, usize count) noexcept
    {
        return advanced_mathematics_dispacher__<Element_t>(input, output, count, 
            [ ](const simd::AVX_t<Element_t>& a) 
            {
                return simd::v_abs(a); 
            } 
        );
    }

    template<Floating_arithmetic Element_t>
    void exp(const Element_t* input, Element_t* output, usize count) noexcept
    {
        return advanced_mathematics_dispacher__<Element_t>(input, output, count,
            [ ](const simd::AVX_t<Element_t>& a)
            {
                return simd::v_exp(a);
            }
        );
    }

    template<Floating_arithmetic Element_t>
    void exp2(const Element_t* input, Element_t* output, usize count) noexcept
    {
        return advanced_mathematics_dispacher__<Element_t>(input, output, count,
            [ ](const simd::AVX_t<Element_t>& a)
            {
                return simd::v_exp2(a);
            }
        );
    }

    template<Floating_arithmetic Element_t>
    void exp10(const Element_t* input, Element_t* output, usize count) noexcept
    {
        return advanced_mathematics_dispacher__<Element_t>(input, output, count,
            [ ](const simd::AVX_t<Element_t>& a)
            {
                return simd::v_exp10(a);
            }
        );
    }

    template<Floating_arithmetic Element_t>
    void log(const Element_t* input, Element_t* output, usize count) noexcept
    {
        return advanced_mathematics_dispacher__<Element_t>(input, output, count,
            [ ](const simd::AVX_t<Element_t>& a)
            {
                return simd::v_log(a);
            }
        );
    }

    template<Floating_arithmetic Element_t>
    void log2(const Element_t* input, Element_t* output, usize count) noexcept
    {
        return advanced_mathematics_dispacher__<Element_t>(input, output, count,
            [ ](const simd::AVX_t<Element_t>& a)
            {
                return simd::v_log2(a);
            }
        );
    }

    template<Floating_arithmetic Element_t>
    void log10(const Element_t* input, Element_t* output, usize count) noexcept
    {
        return advanced_mathematics_dispacher__<Element_t>(input, output, count,
            [ ](const simd::AVX_t<Element_t>& a)
            {
                return simd::v_log10(a);
            }
        );
    }

    template<Floating_arithmetic Element_t>
    void sqrt(const Element_t* input, Element_t* output, usize count) noexcept
    {
        return advanced_mathematics_dispacher__<Element_t>(input, output, count,
            [ ](const simd::AVX_t<Element_t>& a)
            {
                return simd::v_sqrt(a);
            }
        );
    }

    template<Floating_arithmetic Element_t>
    void rsqrt(const Element_t* input, Element_t* output, usize count) noexcept
    {
        return advanced_mathematics_dispacher__<Element_t>(input, output, count,
            [ ](const simd::AVX_t<Element_t>& a)
            {
                return simd::v_rsqrt(a);
            }
        );
    }

    template<Floating_arithmetic Element_t>
    void rcp(const Element_t* input, Element_t* output, usize count) noexcept
    {
        return advanced_mathematics_dispacher__<Element_t>(input, output, count,
            [ ](const simd::AVX_t<Element_t>& a)
            {
                return simd::v_rcp(a);
            }
        );
    }

    template void abs<i8>(const i8*, i8*, usize) noexcept;
    template void abs<i16>(const i16*, i16*, usize) noexcept;
    template void abs<i32>(const i32*, i32*, usize) noexcept;
    template void abs<i64>(const i64*, i64*, usize) noexcept;
    template void abs<u8>(const u8*, u8*, usize) noexcept;
    template void abs<u16>(const u16*, u16*, usize) noexcept;
    template void abs<u32>(const u32*, u32*, usize) noexcept;
    template void abs<u64>(const u64*, u64*, usize) noexcept;
    template void abs<f16>(const f16*, f16*, usize) noexcept;
    template void abs<f32>(const f32*, f32*, usize) noexcept;
    template void abs<f64>(const f64*, f64*, usize) noexcept;

    template void exp<f16>(const f16*, f16*, usize) noexcept;
    template void exp<f32>(const f32*, f32*, usize) noexcept;
    template void exp<f64>(const f64*, f64*, usize) noexcept;

    template void exp2<f16>(const f16*, f16*, usize) noexcept;
    template void exp2<f32>(const f32*, f32*, usize) noexcept;
    template void exp2<f64>(const f64*, f64*, usize) noexcept;

    template void exp10<f16>(const f16*, f16*, usize) noexcept;
    template void exp10<f32>(const f32*, f32*, usize) noexcept;
    template void exp10<f64>(const f64*, f64*, usize) noexcept;

    template void log<f16>(const f16*, f16*, usize) noexcept;
    template void log<f32>(const f32*, f32*, usize) noexcept;
    template void log<f64>(const f64*, f64*, usize) noexcept;

    template void log2<f16>(const f16*, f16*, usize) noexcept;
    template void log2<f32>(const f32*, f32*, usize) noexcept;
    template void log2<f64>(const f64*, f64*, usize) noexcept;
    
    template void log10<f16>(const f16*, f16*, usize) noexcept;
    template void log10<f32>(const f32*, f32*, usize) noexcept;
    template void log10<f64>(const f64*, f64*, usize) noexcept;

    template void sqrt<f16>(const f16*, f16*, usize) noexcept;
    template void sqrt<f32>(const f32*, f32*, usize) noexcept;
    template void sqrt<f64>(const f64*, f64*, usize) noexcept;

    template void rsqrt<f16>(const f16*, f16*, usize) noexcept;
    template void rsqrt<f32>(const f32*, f32*, usize) noexcept;
    template void rsqrt<f64>(const f64*, f64*, usize) noexcept;

    template void rcp<f16>(const f16*, f16*, usize) noexcept;
    template void rcp<f32>(const f32*, f32*, usize) noexcept;
    template void rcp<f64>(const f64*, f64*, usize) noexcept;
}

namespace fy
{
    //template<Floating_arithmetic Element_t>
    //void pow(const Element_t* )
}