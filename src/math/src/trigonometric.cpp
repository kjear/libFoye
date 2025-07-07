module;
#include <immintrin.h>

module foye.algorithm;
import foye.foye_core;
import std;

namespace fy
{
    enum class op_method
    {
        sin = 0, cos = 1, tan = 2, atan = 3, acos = 4, asin = 5,
        exp = 6, exp2 = 7, exp10 = 8,
        log = 9, log2 = 10, log10 = 11,
        sqrt = 12, rsqrt = 13, rcp = 14
    };

    template<i32 operation>
    static void trigonometric_FP16__(const f16* input, f16* output, usize count) noexcept
    {
        usize i = 0;

        {
            const uintptr_t in_addr = reinterpret_cast<uintptr_t>(input);
            const uintptr_t out_addr = reinterpret_cast<uintptr_t>(output);

            usize leading = 0;
            while (i < count)
            {
                const uintptr_t current_in = in_addr + i * sizeof(f16);
                const uintptr_t current_out = out_addr + i * sizeof(f16);
                if ((current_in % 32 == 0) && (current_out % 32 == 0))
                {
                    break;
                }
                leading++;
                i++;
            }

            usize processed = 0;
            while (processed < leading)
            {
                const usize batch = std::min(leading - processed, 8ULL);
                alignas(16) f16 temp_in[8] = { };
                alignas(16) f16 temp_out[8];

                for (usize j = 0; j < batch; ++j)
                {
                    temp_in[j] = input[processed + j];
                }

                __m128i h = _mm_load_si128(reinterpret_cast<const __m128i*>(temp_in));
                __m256 v = _mm256_cvtph_ps(h);

                if constexpr (operation == 0)
                {
                    v = _mm256_sin_ps(v);
                }
                else if constexpr (operation == 1)
                {
                    v = _mm256_cos_ps(v);
                }
                else if constexpr (operation == 2)
                {
                    v = _mm256_tan_ps(v);
                }
                else if constexpr (operation == 3)
                {
                    v = _mm256_atan_ps(v);
                }
                else if constexpr (operation == 4)
                {
                    v = _mm256_acos_ps(v);
                }
                else if constexpr (operation == 5)
                {
                    v = _mm256_asin_ps(v);
                }


                //else if constexpr (operation == 6)
                //{
                //    v = _mm256_exp_ps(v);
                //}
                //else if constexpr (operation == 7)
                //{
                //    v = _mm256_exp2_ps(v);
                //}
                //else if constexpr (operation == 8)
                //{
                //    v = _mm256_exp10_ps(v);
                //}
                //else if constexpr (operation == 9)
                //{
                //    v = _mm256_log_ps(v);
                //}
                //else if constexpr (operation == 10)
                //{
                //    v = _mm256_log2_ps(v);
                //}
                //else if constexpr (operation == 11)
                //{
                //    v = _mm256_log10_ps(v);
                //}
                //else if constexpr (operation == 12)
                //{
                //    v = _mm256_sqrt_ps(v);
                //}
                //else if constexpr (operation == 13)
                //{
                //    v = _mm256_rsqrt_ps(v);
                //}
                //else if constexpr (operation == 14)
                //{
                //    v = _mm256_rcp_ps(v);
                //}



                __m128i hres = _mm256_cvtps_ph(v, _MM_FROUND_TO_NEAREST_INT);
                _mm_store_si128(reinterpret_cast<__m128i*>(temp_out), hres);

                for (usize j = 0; j < batch; ++j)
                {
                    output[processed + j] = temp_out[j];
                }

                processed += batch;
            }
            i = processed;
        }

        const usize prefetch_offset = 128;
        for (; i + 16 <= count; i += 16)
        {
            _mm_prefetch(reinterpret_cast<const char*>(input + i + prefetch_offset), _MM_HINT_T0);

            __m256h vinput = _mm256_load_si256(reinterpret_cast<const __m256i*>(input + i));

            __m256 vlo = _mm256_cvtph_ps(_mm256_extractf128_si256(_mm256_castph_si256(vinput), 0));
            __m256 vhi = _mm256_cvtph_ps(_mm256_extractf128_si256(_mm256_castph_si256(vinput), 1));

            if constexpr (operation == 0)
            {
                vlo = _mm256_sin_ps(vlo);
                vhi = _mm256_sin_ps(vhi);
            }
            else if constexpr (operation == 1)
            {
                vlo = _mm256_cos_ps(vlo);
                vhi = _mm256_cos_ps(vhi);
            }
            else if constexpr (operation == 2)
            {
                vlo = _mm256_tan_ps(vlo);
                vhi = _mm256_tan_ps(vhi);
            }
            else if constexpr (operation == 3)
            {
                vlo = _mm256_atan_ps(vlo);
                vhi = _mm256_atan_ps(vhi);
            }
            else if constexpr (operation == 4)
            {
                vlo = _mm256_acos_ps(vlo);
                vhi = _mm256_acos_ps(vhi);
            }
            else if constexpr (operation == 5)
            {
                vlo = _mm256_asin_ps(vlo);
                vhi = _mm256_asin_ps(vhi);
            }

            __m128i lo_h = _mm256_cvtps_ph(vlo, _MM_FROUND_TO_NEAREST_INT);
            __m128i hi_h = _mm256_cvtps_ph(vhi, _MM_FROUND_TO_NEAREST_INT);
            __m256i combined = _mm256_set_m128i(hi_h, lo_h);

            _mm256_store_si256(reinterpret_cast<__m256i*>(output + i), combined);
        }

        if (usize remaining = count - i; remaining > 0)
        {
            alignas(32) f16 temp_in[16] = { };
            alignas(32) f16 temp_out[16];

            for (usize j = 0; j < remaining; ++j)
            {
                temp_in[j] = input[i + j];
            }

            __m256i vinput = _mm256_load_si256(reinterpret_cast<const __m256i*>(temp_in));
            __m256 vlo = _mm256_cvtph_ps(_mm256_extractf128_si256(vinput, 0));
            __m256 vhi = _mm256_cvtph_ps(_mm256_extractf128_si256(vinput, 1));

            if constexpr (operation == 0)
            {
                vlo = _mm256_sin_ps(vlo);
                vhi = _mm256_sin_ps(vhi);
            }
            else if constexpr (operation == 1)
            {
                vlo = _mm256_cos_ps(vlo);
                vhi = _mm256_cos_ps(vhi);
            }
            else if constexpr (operation == 2)
            {
                vlo = _mm256_tan_ps(vlo);
                vhi = _mm256_tan_ps(vhi);
            }
            else if constexpr (operation == 3)
            {
                vlo = _mm256_atan_ps(vlo);
                vhi = _mm256_atan_ps(vhi);
            }
            else if constexpr (operation == 4)
            {
                vlo = _mm256_acos_ps(vlo);
                vhi = _mm256_acos_ps(vhi);
            }
            else if constexpr (operation == 5)
            {
                vlo = _mm256_asin_ps(vlo);
                vhi = _mm256_asin_ps(vhi);
            }

            __m128i lo_h = _mm256_cvtps_ph(vlo, _MM_FROUND_TO_NEAREST_INT);
            __m128i hi_h = _mm256_cvtps_ph(vhi, _MM_FROUND_TO_NEAREST_INT);
            __m256i combined = _mm256_set_m128i(hi_h, lo_h);
            _mm256_store_si256(reinterpret_cast<__m256i*>(temp_out), combined);

            for (usize j = 0; j < remaining; ++j)
            {
                output[i + j] = temp_out[j];
            }
        }
    }

    template<i32 operation>
    static void trigonometric_FP32__(const f32* input, f32* output, usize count) noexcept
    {
        usize i = 0;

        {
            const uintptr_t in_addr = reinterpret_cast<uintptr_t>(input);
            const uintptr_t out_addr = reinterpret_cast<uintptr_t>(output);

            usize leading = 0;
            while (i < count)
            {
                const uintptr_t current_in = in_addr + i * sizeof(f32);
                const uintptr_t current_out = out_addr + i * sizeof(f32);
                if ((current_in % 32 == 0) && (current_out % 32 == 0))
                {
                    break;
                }
                leading++;
                i++;
            }

            usize processed = 0;
            while (processed < leading)
            {
                const usize batch = (leading - processed) > 8 ? 8 : (leading - processed);
                alignas(32) f32 temp_in[8] = { 0 };
                alignas(32) f32 temp_out[8];

                for (usize j = 0; j < batch; ++j)
                {
                    temp_in[j] = input[processed + j];
                }

                __m256 vin = _mm256_load_ps(temp_in);
                __m256 vres;
                if constexpr (operation == 0)
                {
                    vres = _mm256_sin_ps(vin);
                }
                else if constexpr (operation == 1)
                {
                    vres = _mm256_cos_ps(vin);
                }
                else if constexpr (operation == 2)
                {
                    vres = _mm256_tan_ps(vin);
                }
                else if constexpr (operation == 3)
                {
                    vres = _mm256_atan_ps(vin);
                }
                else if constexpr (operation == 4)
                {
                    vres = _mm256_acos_ps(vin);
                }
                else if constexpr (operation == 5)
                {
                    vres = _mm256_asin_ps(vin);
                }

                _mm256_store_ps(temp_out, vres);

                for (usize j = 0; j < batch; ++j)
                {
                    output[processed + j] = temp_out[j];
                }

                processed += batch;
            }
            i = processed;
        }

        const usize prefetch_offset = 64;
        for (; i + 32 <= count; i += 32)
        {
            _mm_prefetch(reinterpret_cast<const char*>(input + i + prefetch_offset), _MM_HINT_T0);

            __m256 vin0 = _mm256_load_ps(input + i);
            __m256 vin1 = _mm256_load_ps(input + i + 8);
            __m256 vin2 = _mm256_load_ps(input + i + 16);
            __m256 vin3 = _mm256_load_ps(input + i + 24);

            __m256 vres0;
            __m256 vres1;
            __m256 vres2;
            __m256 vres3;

            if constexpr (operation == 0)
            {
                vres0 = _mm256_sin_ps(vin0);
                vres1 = _mm256_sin_ps(vin1);
                vres2 = _mm256_sin_ps(vin2);
                vres3 = _mm256_sin_ps(vin3);
            }
            else if constexpr (operation == 1)
            {
                vres0 = _mm256_cos_ps(vin0);
                vres1 = _mm256_cos_ps(vin1);
                vres2 = _mm256_cos_ps(vin2);
                vres3 = _mm256_cos_ps(vin3);
            }
            else if constexpr (operation == 2)
            {
                vres0 = _mm256_tan_ps(vin0);
                vres1 = _mm256_tan_ps(vin1);
                vres2 = _mm256_tan_ps(vin2);
                vres3 = _mm256_tan_ps(vin3);
            }
            else if constexpr (operation == 3)
            {
                vres0 = _mm256_atan_ps(vin0);
                vres1 = _mm256_atan_ps(vin1);
                vres2 = _mm256_atan_ps(vin2);
                vres3 = _mm256_atan_ps(vin3);
            }
            else if constexpr (operation == 4)
            {
                vres0 = _mm256_acos_ps(vin0);
                vres1 = _mm256_acos_ps(vin1);
                vres2 = _mm256_acos_ps(vin2);
                vres3 = _mm256_acos_ps(vin3);
            }
            else if constexpr (operation == 5)
            {
                vres0 = _mm256_asin_ps(vin0);
                vres1 = _mm256_asin_ps(vin1);
                vres2 = _mm256_asin_ps(vin2);
                vres3 = _mm256_asin_ps(vin3);
            }

            _mm256_store_ps(output + i, vres0);
            _mm256_store_ps(output + i + 8, vres1);
            _mm256_store_ps(output + i + 16, vres2);
            _mm256_store_ps(output + i + 24, vres3);
        }

        if (usize remaining = count - i; remaining > 0)
        {
            alignas(32) f32 temp_in[8] = { 0 };
            alignas(32) f32 temp_out[8];

            for (usize j = 0; j < remaining; ++j)
            {
                temp_in[j] = input[i + j];
            }

            __m256 vin = _mm256_load_ps(temp_in);
            __m256 vres;
            if constexpr (operation == 0)
            {
                vres = _mm256_sin_ps(vin);
            }
            else if constexpr (operation == 1)
            {
                vres = _mm256_cos_ps(vin);
            }
            else if constexpr (operation == 2)
            {
                vres = _mm256_tan_ps(vin);
            }
            else if constexpr (operation == 3)
            {
                vres = _mm256_atan_ps(vin);
            }
            else if constexpr (operation == 4)
            {
                vres = _mm256_acos_ps(vin);
            }
            else if constexpr (operation == 5)
            {
                vres = _mm256_asin_ps(vin);
            }

            _mm256_store_ps(temp_out, vres);

            for (usize j = 0; j < remaining; ++j)
            {
                output[i + j] = temp_out[j];
            }
        }
    }

    template<i32 operation>
    static void trigonometric_FP64__(const f64* input, f64* output, usize count) noexcept
    {
        usize i = 0;

        {
            const uintptr_t in_addr = reinterpret_cast<uintptr_t>(input);
            const uintptr_t out_addr = reinterpret_cast<uintptr_t>(output);

            usize leading = 0;
            while (i < count)
            {
                const uintptr_t current_in = in_addr + i * sizeof(f64);
                const uintptr_t current_out = out_addr + i * sizeof(f64);

                if ((current_in % 32 == 0) && (current_out % 32 == 0))
                {
                    break;
                }
                leading++;
                i++;
            }

            usize processed = 0;
            while (processed < leading)
            {
                const usize batch = (leading - processed) > 4 ? 4 : (leading - processed);
                alignas(32) f64 temp_in[4] = { 0 };
                alignas(32) f64 temp_out[4];

                for (usize j = 0; j < batch; ++j)
                {
                    temp_in[j] = input[processed + j];
                }

                __m256d vin = _mm256_load_pd(temp_in);
                __m256d vres;

                if constexpr (operation == 0)
                {
                    vres = _mm256_sin_pd(vin);
                }
                else if constexpr (operation == 1)
                {
                    vres = _mm256_cos_pd(vin);
                }
                else if constexpr (operation == 2)
                {
                    vres = _mm256_tan_pd(vin);
                }
                else if constexpr (operation == 3)
                {
                    vres = _mm256_atan_pd(vin);
                }
                else if constexpr (operation == 4)
                {
                    vres = _mm256_acos_pd(vin);
                }
                else if constexpr (operation == 5)
                {
                    vres = _mm256_asin_pd(vin);
                }

                _mm256_store_pd(temp_out, vres);

                for (usize j = 0; j < batch; ++j)
                {
                    output[processed + j] = temp_out[j];
                }

                processed += batch;
            }
            i = processed;
        }

        const usize prefetch_offset = 32;
        for (; i + 16 <= count; i += 16)
        {
            _mm_prefetch(reinterpret_cast<const char*>(input + i + prefetch_offset), _MM_HINT_T0);

            __m256d vin0 = _mm256_load_pd(input + i);
            __m256d vin1 = _mm256_load_pd(input + i + 4);
            __m256d vin2 = _mm256_load_pd(input + i + 8);
            __m256d vin3 = _mm256_load_pd(input + i + 12);

            __m256d vres0;
            __m256d vres1;
            __m256d vres2;
            __m256d vres3;

            if constexpr (operation == 0)
            {
                vres0 = _mm256_sin_pd(vin0);
                vres1 = _mm256_sin_pd(vin1);
                vres2 = _mm256_sin_pd(vin2);
                vres3 = _mm256_sin_pd(vin3);
            }
            else if constexpr (operation == 1)
            {
                vres0 = _mm256_cos_pd(vin0);
                vres1 = _mm256_cos_pd(vin1);
                vres2 = _mm256_cos_pd(vin2);
                vres3 = _mm256_cos_pd(vin3);
            }
            else if constexpr (operation == 2)
            {
                vres0 = _mm256_tan_pd(vin0);
                vres1 = _mm256_tan_pd(vin1);
                vres2 = _mm256_tan_pd(vin2);
                vres3 = _mm256_tan_pd(vin3);
            }
            else if constexpr (operation == 3)
            {
                vres0 = _mm256_atan_pd(vin0);
                vres1 = _mm256_atan_pd(vin1);
                vres2 = _mm256_atan_pd(vin2);
                vres3 = _mm256_atan_pd(vin3);
            }
            else if constexpr (operation == 4)
            {
                vres0 = _mm256_acos_pd(vin0);
                vres1 = _mm256_acos_pd(vin1);
                vres2 = _mm256_acos_pd(vin2);
                vres3 = _mm256_acos_pd(vin3);
            }
            else if constexpr (operation == 5)
            {
                vres0 = _mm256_asin_pd(vin0);
                vres1 = _mm256_asin_pd(vin1);
                vres2 = _mm256_asin_pd(vin2);
                vres3 = _mm256_asin_pd(vin3);
            }

            _mm256_store_pd(output + i, vres0);
            _mm256_store_pd(output + i + 4, vres1);
            _mm256_store_pd(output + i + 8, vres2);
            _mm256_store_pd(output + i + 12, vres3);
        }

        if (usize remaining = count - i; remaining > 0)
        {
            usize processed = 0;
            while (processed < remaining)
            {
                const usize batch = (remaining - processed) > 4 ? 4 : (remaining - processed);
                alignas(32) f64 temp_in[4] = { 0 };
                alignas(32) f64 temp_out[4];

                for (usize j = 0; j < batch; ++j)
                {
                    temp_in[j] = input[i + processed + j];
                }

                __m256d vin = _mm256_load_pd(temp_in);
                __m256d vres;
                if constexpr (operation == 0)
                {
                    vres = _mm256_sin_pd(vin);
                }
                else if constexpr (operation == 1)
                {
                    vres = _mm256_cos_pd(vin);
                }
                else if constexpr (operation == 2)
                {
                    vres = _mm256_tan_pd(vin);
                }
                else if constexpr (operation == 3)
                {
                    vres = _mm256_atan_pd(vin);
                }
                else if constexpr (operation == 4)
                {
                    vres = _mm256_acos_pd(vin);
                }
                else if constexpr (operation == 5)
                {
                    vres = _mm256_asin_pd(vin);
                }

                _mm256_store_pd(temp_out, vres);

                for (usize j = 0; j < batch; ++j)
                {
                    output[i + processed + j] = temp_out[j];
                }

                processed += batch;
            }
        }
    }
}


namespace fy
{
    template<Floating_arithmetic Element_t>
    void sine(const Element_t* input, Element_t* output, usize count) noexcept
    {
        std::unreachable();
    }

    template<> void sine(const f16* input, f16* output, usize count) noexcept
    {
        return trigonometric_FP16__<0>(input, output, count);
    }
    
    template<> void sine(const f32* input, f32* output, usize count) noexcept
    {
        return trigonometric_FP32__<0>(input, output, count);
    }
	
    template<> void sine(const f64* input, f64* output, usize count) noexcept
    {
        return trigonometric_FP64__<0>(input, output, count);
    }
}

namespace fy
{
    template<Floating_arithmetic Element_t>
    void cosine(const Element_t* input, Element_t* output, usize count) noexcept
    {
        std::unreachable();
    }

    template<> void cosine(const f16* input, f16* output, usize count) noexcept
    {
        return trigonometric_FP16__<1>(input, output, count);
    }

    template<> void cosine(const f32* input, f32* output, usize count) noexcept
    {
        return trigonometric_FP32__<1>(input, output, count);
    }

    template<> void cosine(const f64* input, f64* output, usize count) noexcept
    {
        return trigonometric_FP64__<1>(input, output, count);
    }
}

namespace fy
{
    template<Floating_arithmetic Element_t>
    void tangent(const Element_t* input, Element_t* output, usize count) noexcept
    {
        std::unreachable();
    }

    template<> void tangent(const f16* input, f16* output, usize count) noexcept
    {
        return trigonometric_FP16__<2>(input, output, count);
    }

    template<> void tangent(const f32* input, f32* output, usize count) noexcept
    {
        return trigonometric_FP32__<2>(input, output, count);
    }

    template<> void tangent(const f64* input, f64* output, usize count) noexcept
    {
        return trigonometric_FP64__<2>(input, output, count);
    }
}

namespace fy
{
    template<Floating_arithmetic Element_t>
    void arctangent(const Element_t* input, Element_t* output, usize count) noexcept
    {
        std::unreachable();
    }

    template<> void arctangent(const f16* input, f16* output, usize count) noexcept
    {
        return trigonometric_FP16__<3>(input, output, count);
    }

    template<> void arctangent(const f32* input, f32* output, usize count) noexcept
    {
        return trigonometric_FP32__<3>(input, output, count);
    }

    template<> void arctangent(const f64* input, f64* output, usize count) noexcept
    {
        return trigonometric_FP64__<3>(input, output, count);
    }

}

namespace fy
{
    template<Floating_arithmetic Element_t>
    void arccosine(const Element_t* input, Element_t* output, usize count) noexcept
    {
        std::unreachable();
    }

    template<> void arccosine(const f16* input, f16* output, usize count) noexcept
    {
        return trigonometric_FP16__<4>(input, output, count);
    }

    template<> void arccosine(const f32* input, f32* output, usize count) noexcept
    {
        return trigonometric_FP32__<4>(input, output, count);
    }

    template<> void arccosine(const f64* input, f64* output, usize count) noexcept
    {
        return trigonometric_FP64__<4>(input, output, count);
    }
}

namespace fy
{
    template<Floating_arithmetic Element_t>
    void arcsine(const Element_t* input, Element_t* output, usize count) noexcept
    {
        std::unreachable();
    }

    template<> void arcsine(const f16* input, f16* output, usize count) noexcept
    {
        return trigonometric_FP16__<5>(input, output, count);
    }

    template<> void arcsine(const f32* input, f32* output, usize count) noexcept
    {
        return trigonometric_FP32__<5>(input, output, count);
    }

    template<> void arcsine(const f64* input, f64* output, usize count) noexcept
    {
        return trigonometric_FP64__<5>(input, output, count);
    }
}