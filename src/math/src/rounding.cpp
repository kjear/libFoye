module;
#include <immintrin.h>

module foye.algorithm;
import foye.foye_core;
import std;

namespace fy
{
    template<Floating_arithmetic Element_t>
    void floor(const Element_t* input, Element_t* output, usize count) noexcept
    {
        std::unreachable();
    }

    template<Floating_arithmetic Element_t>
    void ceil(const Element_t* input, Element_t* output, usize count) noexcept
    {
        std::unreachable();
    }

    template<Floating_arithmetic Element_t>
    void round(const Element_t* input, Element_t* output, usize count) noexcept
    {
        std::unreachable();
    }

    template<Floating_arithmetic Element_t>
    void trunc(const Element_t* input, Element_t* output, usize count) noexcept
    {
        std::unreachable();
    }

    template<typename SIMD_t> struct operator_expr__
    {
        static constexpr auto floor = [ ](SIMD_t vin) -> SIMD_t
            {
                if constexpr (std::is_same_v<SIMD_t, __m256>)
                {
                    return _mm256_floor_ps(vin);
                }
                else { return _mm256_floor_pd(vin); }
            };

        static constexpr auto ceil = [ ](SIMD_t vin) -> SIMD_t
            {
                if constexpr (std::is_same_v<SIMD_t, __m256>)
                {
                    return _mm256_ceil_ps(vin);
                }
                else { return _mm256_ceil_pd(vin); }
            };

        static constexpr auto round = [ ](SIMD_t vin) -> SIMD_t
            {
                if constexpr (std::is_same_v<SIMD_t, __m256>)
                {
                    return _mm256_round_ps(vin, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
                }
                else { return _mm256_round_pd(vin, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC); }
            };

        static constexpr auto trunc = [ ](SIMD_t vin) -> SIMD_t
            {
                if constexpr (std::is_same_v<SIMD_t, __m256>)
                {
                    return _mm256_round_ps(vin, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
                }
                else { return _mm256_round_pd(vin, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC); }
            };
    };

    template<typename T> struct simd_traits__;

    template<> struct simd_traits__<f32>
    {
        using Type = __m256;
        static constexpr usize VEC_SIZE = 8;
        static Type load(const f32* p) { return _mm256_load_ps(p); }
        static void store(f32* p, Type v) { _mm256_store_ps(p, v); }
    };

    template<> struct simd_traits__<f64>
    {
        using Type = __m256d;
        static constexpr usize VEC_SIZE = 4;
        static Type load(const f64* p) { return _mm256_load_pd(p); }
        static void store(f64* p, Type v) { _mm256_store_pd(p, v); }
    };

    template<Floating_arithmetic Element_t, i32 simd_OP, typename Scalar_OP>
    static void begin_rounding__(const Element_t* input, Element_t* output, usize count, Scalar_OP&& scalar_op)
    {
        constexpr usize VEC_SIZE = simd_traits__<Element_t>::VEC_SIZE;
        constexpr usize ALIGNMENT = 32;
        usize i = 0;

        for (; i < count && (reinterpret_cast<uintptr_t>(input + i) % ALIGNMENT != 0 ||
            reinterpret_cast<uintptr_t>(output + i) % ALIGNMENT != 0); ++i)
        {
            output[i] = scalar_op(input[i]);
        }

        constexpr usize MAIN_CHUNK = 4 * VEC_SIZE;
        for (; i + MAIN_CHUNK <= count; i += MAIN_CHUNK)
        {
            typename simd_traits__<Element_t>::Type v0 = simd_traits__<Element_t>::load(input + i + 0 * VEC_SIZE);
            typename simd_traits__<Element_t>::Type v1 = simd_traits__<Element_t>::load(input + i + 1 * VEC_SIZE);
            typename simd_traits__<Element_t>::Type v2 = simd_traits__<Element_t>::load(input + i + 2 * VEC_SIZE);
            typename simd_traits__<Element_t>::Type v3 = simd_traits__<Element_t>::load(input + i + 3 * VEC_SIZE);

            if constexpr (simd_OP == 0)
            {
                v0 = operator_expr__<typename simd_traits__<Element_t>::Type>::floor(v0);
                v1 = operator_expr__<typename simd_traits__<Element_t>::Type>::floor(v1);
                v2 = operator_expr__<typename simd_traits__<Element_t>::Type>::floor(v2);
                v3 = operator_expr__<typename simd_traits__<Element_t>::Type>::floor(v3);
            }
            else if constexpr (simd_OP == 1)
            {
                v0 = operator_expr__<typename simd_traits__<Element_t>::Type>::ceil(v0);
                v1 = operator_expr__<typename simd_traits__<Element_t>::Type>::ceil(v1);
                v2 = operator_expr__<typename simd_traits__<Element_t>::Type>::ceil(v2);
                v3 = operator_expr__<typename simd_traits__<Element_t>::Type>::ceil(v3);
            }
            else if constexpr (simd_OP == 2)
            {
                v0 = operator_expr__<typename simd_traits__<Element_t>::Type>::round(v0);
                v1 = operator_expr__<typename simd_traits__<Element_t>::Type>::round(v1);
                v2 = operator_expr__<typename simd_traits__<Element_t>::Type>::round(v2);
                v3 = operator_expr__<typename simd_traits__<Element_t>::Type>::round(v3);
            }
            else if constexpr (simd_OP == 3)
            {
                v0 = operator_expr__<typename simd_traits__<Element_t>::Type>::trunc(v0);
                v1 = operator_expr__<typename simd_traits__<Element_t>::Type>::trunc(v1);
                v2 = operator_expr__<typename simd_traits__<Element_t>::Type>::trunc(v2);
                v3 = operator_expr__<typename simd_traits__<Element_t>::Type>::trunc(v3);
            }

            simd_traits__<Element_t>::store(output + i + 0 * VEC_SIZE, v0);
            simd_traits__<Element_t>::store(output + i + 1 * VEC_SIZE, v1);
            simd_traits__<Element_t>::store(output + i + 2 * VEC_SIZE, v2);
            simd_traits__<Element_t>::store(output + i + 3 * VEC_SIZE, v3);
        }

        for (; i + VEC_SIZE <= count; i += VEC_SIZE)
        {
            typename simd_traits__<Element_t>::Type v = simd_traits__<Element_t>::load(input + i);
                 if constexpr (simd_OP == 0) { v = operator_expr__<typename simd_traits__<Element_t>::Type>::floor(v); }
            else if constexpr (simd_OP == 1) { v = operator_expr__<typename simd_traits__<Element_t>::Type>::ceil(v); }
            else if constexpr (simd_OP == 2) { v = operator_expr__<typename simd_traits__<Element_t>::Type>::round(v); }
            else if constexpr (simd_OP == 3) { v = operator_expr__<typename simd_traits__<Element_t>::Type>::trunc(v); }

            simd_traits__<Element_t>::store(output + i, v);
        }

        for (; i < count; ++i)
        {
            output[i] = scalar_op(input[i]);
        }
    }

    template<> void floor(const f32* input, f32* output, usize count) noexcept { begin_rounding__<f32, 0>(input, output, count, std::floorf); }
    template<> void ceil(const f32* input, f32* output, usize count) noexcept { begin_rounding__<f32, 1>(input, output, count, std::ceilf); }
    template<> void round(const f32* input, f32* output, usize count) noexcept { begin_rounding__<f32, 2>(input, output, count, std::roundf); }
    template<> void trunc(const f32* input, f32* output, usize count) noexcept { begin_rounding__<f32, 3>(input, output, count, std::ceilf); }

    template<> void floor(const f64* input, f64* output, usize count) noexcept
    {
        begin_rounding__<f64, 0>(input, output, count, [ ](f64 in) -> f64 {return std::floor(in); });
    }

    template<> void ceil(const f64* input, f64* output, usize count) noexcept
    {
        begin_rounding__<f64, 1>(input, output, count, [ ](f64 in) -> f64 {return std::ceil(in); });
    }

    template<> void round(const f64* input, f64* output, usize count) noexcept
    {
        begin_rounding__<f64, 2>(input, output, count, [ ](f64 in) -> f64 {return std::round(in); });
    }

    template<> void trunc(const f64* input, f64* output, usize count) noexcept
    {
        begin_rounding__<f64, 3>(input, output, count, [ ](f64 in) -> f64 {return std::trunc(in); });
    }
}

namespace fy
{
    template<i32 simd_OP, Floating_arithmetic Element_t, typename Scalar_OP>
    static void begin_rounding_FP16__(const Element_t* input, Element_t* output, usize count, Scalar_OP&& scalar_op) noexcept
    {
        constexpr usize VEC_SIZE = 16;
        constexpr usize ALIGNMENT = 32;
        usize i = 0;

        for (; i < count && (reinterpret_cast<uintptr_t>(input + i) % ALIGNMENT != 0 ||
            reinterpret_cast<uintptr_t>(output + i) % ALIGNMENT != 0); ++i)
        {
            output[i] = static_cast<f16>(scalar_op(static_cast<f32>(input[i])));
        }

        for (; i + VEC_SIZE <= count; i += VEC_SIZE)
        {
            __m256h vraw_0 = _mm256_load_si256(reinterpret_cast<const __m256i*>(input + i + 0 * VEC_SIZE));

            __m256 vraw_00 = _mm256_cvtph_ps(_mm256_extractf128_si256(_mm256_castph_si256(vraw_0), 0));
            __m256 vraw_01 = _mm256_cvtph_ps(_mm256_extractf128_si256(_mm256_castph_si256(vraw_0), 1));

            if constexpr (simd_OP == 0) { vraw_00 = _mm256_floor_ps(vraw_00); }
            else if constexpr (simd_OP == 1) { vraw_00 = _mm256_ceil_ps(vraw_00); }
            else if constexpr (simd_OP == 2) { vraw_00 = _mm256_round_ps(vraw_00, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC); }
            else if constexpr (simd_OP == 3) { vraw_00 = _mm256_round_ps(vraw_00, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC); }

            if constexpr (simd_OP == 0) { vraw_01 = _mm256_floor_ps(vraw_01); }
            else if constexpr (simd_OP == 1) { vraw_01 = _mm256_ceil_ps(vraw_01); }
            else if constexpr (simd_OP == 2) { vraw_01 = _mm256_round_ps(vraw_01, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC); }
            else if constexpr (simd_OP == 3) { vraw_01 = _mm256_round_ps(vraw_01, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC); }

            __m128i vraw_0_lo_h = _mm256_cvtps_ph(vraw_00, _MM_FROUND_TO_NEAREST_INT);
            __m128i vraw_0_hi_h = _mm256_cvtps_ph(vraw_01, _MM_FROUND_TO_NEAREST_INT);
            __m256i vraw_0_combined = _mm256_set_m128i(vraw_0_hi_h, vraw_0_lo_h);

            _mm256_store_si256(reinterpret_cast<__m256i*>(output + i + 0 * VEC_SIZE), vraw_0_combined);
        }

        for (; i < count; ++i)
        {
            output[i] = static_cast<f16>(scalar_op(static_cast<f32>(input[i])));
        }
    }


    template<> void floor(const f16* input, f16* output, usize count) noexcept
    {
        begin_rounding_FP16__<0>(input, output, count, std::floorf);
    }

    template<> void ceil(const f16* input, f16* output, usize count) noexcept
    {
        begin_rounding_FP16__<1>(input, output, count, std::ceilf);
    }

    template<> void round(const f16* input, f16* output, usize count) noexcept
    {
        begin_rounding_FP16__<2>(input, output, count, std::roundf);
    }

    template<> void trunc(const f16* input, f16* output, usize count) noexcept
    {
        begin_rounding_FP16__<3>(input, output, count, std::truncf);
    }
}