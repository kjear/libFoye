module;
#include <immintrin.h>
#include <intrin.h>

module foye.algorithm;
import foye.foye_core;
import foye.simd;
import std;

namespace fy
{
    template<typename AVX_t, usize N> struct v_transpose_Invoker final
    {
        using input = const AVX_t&;
        using output = AVX_t&;

        v_transpose_Invoker() { std::unreachable(); }
        void operator () (input i0, input i1, input i2, input i3, input i4, input i5, input i6, input i7,
            output o0, output o1, output o2, output o3, output o4, output o5, output o6, output o7) const noexcept
        {
            std::unreachable();
        }
    };

    template<typename T> struct v_transpose_Invoker<T, sizeof(u8)>
    {
        using input = const T&;
        using output = T&;

        v_transpose_Invoker() = default;

        void operator()(input i0, input i1, input i2, input i3, input i4, input i5, input i6, input i7,
            output o0, output o1, output o2, output o3, output o4, output o5, output o6, output o7) const noexcept
        {
            __m256i t0 = _mm256_unpacklo_epi8(static_cast<__m256i>(i0.data), static_cast<__m256i>(i1.data));
            __m256i t1 = _mm256_unpackhi_epi8(static_cast<__m256i>(i0.data), static_cast<__m256i>(i1.data));
            __m256i t2 = _mm256_unpacklo_epi8(static_cast<__m256i>(i2.data), static_cast<__m256i>(i3.data));
            __m256i t3 = _mm256_unpackhi_epi8(static_cast<__m256i>(i2.data), static_cast<__m256i>(i3.data));
            __m256i t4 = _mm256_unpacklo_epi8(static_cast<__m256i>(i4.data), static_cast<__m256i>(i5.data));
            __m256i t5 = _mm256_unpackhi_epi8(static_cast<__m256i>(i4.data), static_cast<__m256i>(i5.data));
            __m256i t6 = _mm256_unpacklo_epi8(static_cast<__m256i>(i6.data), static_cast<__m256i>(i7.data));
            __m256i t7 = _mm256_unpackhi_epi8(static_cast<__m256i>(i6.data), static_cast<__m256i>(i7.data));

            __m256i tt0 = _mm256_unpacklo_epi16(t0, t2);
            __m256i tt1 = _mm256_unpackhi_epi16(t0, t2);
            __m256i tt2 = _mm256_unpacklo_epi16(t1, t3);
            __m256i tt3 = _mm256_unpackhi_epi16(t1, t3);
            __m256i tt4 = _mm256_unpacklo_epi16(t4, t6);
            __m256i tt5 = _mm256_unpackhi_epi16(t4, t6);
            __m256i tt6 = _mm256_unpacklo_epi16(t5, t7);
            __m256i tt7 = _mm256_unpackhi_epi16(t5, t7);

            __m256i ttt0 = _mm256_unpacklo_epi32(tt0, tt4);
            __m256i ttt1 = _mm256_unpackhi_epi32(tt0, tt4);
            __m256i ttt2 = _mm256_unpacklo_epi32(tt1, tt5);
            __m256i ttt3 = _mm256_unpackhi_epi32(tt1, tt5);
            __m256i ttt4 = _mm256_unpacklo_epi32(tt2, tt6);
            __m256i ttt5 = _mm256_unpackhi_epi32(tt2, tt6);
            __m256i ttt6 = _mm256_unpacklo_epi32(tt3, tt7);
            __m256i ttt7 = _mm256_unpackhi_epi32(tt3, tt7);

            o0.data = _mm256_permute2x128_si256(ttt0, ttt1, 0x20);
            o1.data = _mm256_permute2x128_si256(ttt2, ttt3, 0x20);
            o2.data = _mm256_permute2x128_si256(ttt4, ttt5, 0x20);
            o3.data = _mm256_permute2x128_si256(ttt6, ttt7, 0x20);
            o4.data = _mm256_permute2x128_si256(ttt0, ttt1, 0x31);
            o5.data = _mm256_permute2x128_si256(ttt2, ttt3, 0x31);
            o6.data = _mm256_permute2x128_si256(ttt4, ttt5, 0x31);
            o7.data = _mm256_permute2x128_si256(ttt6, ttt7, 0x31);
        }
    };

    template<typename T> struct v_transpose_Invoker<T, sizeof(u16)> final
    {
        using input = const T&;
        using output = T&;

        v_transpose_Invoker() = default;

        void operator () (input i0, input i1, input i2, input i3, input i4, input i5, input i6, input i7,
            output o0, output o1, output o2, output o3, output o4, output o5, output o6, output o7) const noexcept
        {
            __m256i t0 = _mm256_unpacklo_epi16(static_cast<__m256i>(i0.data), static_cast<__m256i>(i1.data));
            __m256i t1 = _mm256_unpackhi_epi16(static_cast<__m256i>(i0.data), static_cast<__m256i>(i1.data));
            __m256i t2 = _mm256_unpacklo_epi16(static_cast<__m256i>(i2.data), static_cast<__m256i>(i3.data));
            __m256i t3 = _mm256_unpackhi_epi16(static_cast<__m256i>(i2.data), static_cast<__m256i>(i3.data));
            __m256i t4 = _mm256_unpacklo_epi16(static_cast<__m256i>(i4.data), static_cast<__m256i>(i5.data));
            __m256i t5 = _mm256_unpackhi_epi16(static_cast<__m256i>(i4.data), static_cast<__m256i>(i5.data));
            __m256i t6 = _mm256_unpacklo_epi16(static_cast<__m256i>(i6.data), static_cast<__m256i>(i7.data));
            __m256i t7 = _mm256_unpackhi_epi16(static_cast<__m256i>(i6.data), static_cast<__m256i>(i7.data));

            __m256i tt0 = _mm256_unpacklo_epi32(t0, t2);
            __m256i tt1 = _mm256_unpackhi_epi32(t0, t2);
            __m256i tt2 = _mm256_unpacklo_epi32(t1, t3);
            __m256i tt3 = _mm256_unpackhi_epi32(t1, t3);
            __m256i tt4 = _mm256_unpacklo_epi32(t4, t6);
            __m256i tt5 = _mm256_unpackhi_epi32(t4, t6);
            __m256i tt6 = _mm256_unpacklo_epi32(t5, t7);
            __m256i tt7 = _mm256_unpackhi_epi32(t5, t7);

            __m256i ttt0 = _mm256_unpacklo_epi64(tt0, tt4);
            __m256i ttt1 = _mm256_unpackhi_epi64(tt0, tt4);
            __m256i ttt2 = _mm256_unpacklo_epi64(tt1, tt5);
            __m256i ttt3 = _mm256_unpackhi_epi64(tt1, tt5);
            __m256i ttt4 = _mm256_unpacklo_epi64(tt2, tt6);
            __m256i ttt5 = _mm256_unpackhi_epi64(tt2, tt6);
            __m256i ttt6 = _mm256_unpacklo_epi64(tt3, tt7);
            __m256i ttt7 = _mm256_unpackhi_epi64(tt3, tt7);

            o0.data = static_cast<typename T::vector_t>(_mm256_permute2x128_si256(ttt0, ttt1, 0x20));
            o1.data = static_cast<typename T::vector_t>(_mm256_permute2x128_si256(ttt2, ttt3, 0x20));
            o2.data = static_cast<typename T::vector_t>(_mm256_permute2x128_si256(ttt4, ttt5, 0x20));
            o3.data = static_cast<typename T::vector_t>(_mm256_permute2x128_si256(ttt6, ttt7, 0x20));
            o4.data = static_cast<typename T::vector_t>(_mm256_permute2x128_si256(ttt0, ttt1, 0x31));
            o5.data = static_cast<typename T::vector_t>(_mm256_permute2x128_si256(ttt2, ttt3, 0x31));
            o6.data = static_cast<typename T::vector_t>(_mm256_permute2x128_si256(ttt4, ttt5, 0x31));
            o7.data = static_cast<typename T::vector_t>(_mm256_permute2x128_si256(ttt6, ttt7, 0x31));
        }
    };

    template<typename T> struct v_transpose_Invoker<T, sizeof(u32)> final
    {
        using input = const T&;
        using output = T&;

        v_transpose_Invoker() = default;

        void operator()(input i0, input i1, input i2, input i3, input i4, input i5, input i6, input i7,
            output o0, output o1, output o2, output o3, output o4, output o5, output o6, output o7) const noexcept 
            requires std::is_integral_v<typename T::scalar_t>
        {
            __m256i t0 = _mm256_unpacklo_epi32(static_cast<__m256i>(i0.data), static_cast<__m256i>(i1.data));
            __m256i t1 = _mm256_unpackhi_epi32(static_cast<__m256i>(i0.data), static_cast<__m256i>(i1.data));
            __m256i t2 = _mm256_unpacklo_epi32(static_cast<__m256i>(i2.data), static_cast<__m256i>(i3.data));
            __m256i t3 = _mm256_unpackhi_epi32(static_cast<__m256i>(i2.data), static_cast<__m256i>(i3.data));
            __m256i t4 = _mm256_unpacklo_epi32(static_cast<__m256i>(i4.data), static_cast<__m256i>(i5.data));
            __m256i t5 = _mm256_unpackhi_epi32(static_cast<__m256i>(i4.data), static_cast<__m256i>(i5.data));
            __m256i t6 = _mm256_unpacklo_epi32(static_cast<__m256i>(i6.data), static_cast<__m256i>(i7.data));
            __m256i t7 = _mm256_unpackhi_epi32(static_cast<__m256i>(i6.data), static_cast<__m256i>(i7.data));

            __m256i tt0 = _mm256_unpacklo_epi64(t0, t2);
            __m256i tt1 = _mm256_unpackhi_epi64(t0, t2);
            __m256i tt2 = _mm256_unpacklo_epi64(t1, t3);
            __m256i tt3 = _mm256_unpackhi_epi64(t1, t3);
            __m256i tt4 = _mm256_unpacklo_epi64(t4, t6);
            __m256i tt5 = _mm256_unpackhi_epi64(t4, t6);
            __m256i tt6 = _mm256_unpacklo_epi64(t5, t7);
            __m256i tt7 = _mm256_unpackhi_epi64(t5, t7);

            o0.data = _mm256_permute2x128_si256(tt0, tt4, 0x20);
            o1.data = _mm256_permute2x128_si256(tt1, tt5, 0x20);
            o2.data = _mm256_permute2x128_si256(tt2, tt6, 0x20);
            o3.data = _mm256_permute2x128_si256(tt3, tt7, 0x20);
            o4.data = _mm256_permute2x128_si256(tt0, tt4, 0x31);
            o5.data = _mm256_permute2x128_si256(tt1, tt5, 0x31);
            o6.data = _mm256_permute2x128_si256(tt2, tt6, 0x31);
            o7.data = _mm256_permute2x128_si256(tt3, tt7, 0x31);
        }

        void operator()(input i0, input i1, input i2, input i3, input i4, input i5, input i6, input i7,
            output o0, output o1, output o2, output o3, output o4, output o5, output o6, output o7) const noexcept 
            requires std::is_floating_point_v<typename T::scalar_t>
        {
            __m256 t0 = _mm256_unpacklo_ps(static_cast<__m256>(i0.data), static_cast<__m256>(i1.data));
            __m256 t1 = _mm256_unpackhi_ps(static_cast<__m256>(i0.data), static_cast<__m256>(i1.data));
            __m256 t2 = _mm256_unpacklo_ps(static_cast<__m256>(i2.data), static_cast<__m256>(i3.data));
            __m256 t3 = _mm256_unpackhi_ps(static_cast<__m256>(i2.data), static_cast<__m256>(i3.data));
            __m256 t4 = _mm256_unpacklo_ps(static_cast<__m256>(i4.data), static_cast<__m256>(i5.data));
            __m256 t5 = _mm256_unpackhi_ps(static_cast<__m256>(i4.data), static_cast<__m256>(i5.data));
            __m256 t6 = _mm256_unpacklo_ps(static_cast<__m256>(i6.data), static_cast<__m256>(i7.data));
            __m256 t7 = _mm256_unpackhi_ps(static_cast<__m256>(i6.data), static_cast<__m256>(i7.data));

            __m256 tt0 = _mm256_shuffle_ps(t0, t2, _MM_SHUFFLE(1, 0, 1, 0));
            __m256 tt1 = _mm256_shuffle_ps(t0, t2, _MM_SHUFFLE(3, 2, 3, 2));
            __m256 tt2 = _mm256_shuffle_ps(t1, t3, _MM_SHUFFLE(1, 0, 1, 0));
            __m256 tt3 = _mm256_shuffle_ps(t1, t3, _MM_SHUFFLE(3, 2, 3, 2));
            __m256 tt4 = _mm256_shuffle_ps(t4, t6, _MM_SHUFFLE(1, 0, 1, 0));
            __m256 tt5 = _mm256_shuffle_ps(t4, t6, _MM_SHUFFLE(3, 2, 3, 2));
            __m256 tt6 = _mm256_shuffle_ps(t5, t7, _MM_SHUFFLE(1, 0, 1, 0));
            __m256 tt7 = _mm256_shuffle_ps(t5, t7, _MM_SHUFFLE(3, 2, 3, 2));

            o0.data = _mm256_permute2f128_ps(tt0, tt4, 0x20);
            o1.data = _mm256_permute2f128_ps(tt1, tt5, 0x20);
            o2.data = _mm256_permute2f128_ps(tt2, tt6, 0x20);
            o3.data = _mm256_permute2f128_ps(tt3, tt7, 0x20);
            o4.data = _mm256_permute2f128_ps(tt0, tt4, 0x31);
            o5.data = _mm256_permute2f128_ps(tt1, tt5, 0x31);
            o6.data = _mm256_permute2f128_ps(tt2, tt6, 0x31);
            o7.data = _mm256_permute2f128_ps(tt3, tt7, 0x31);
        }
    };

    template<typename T> struct v_transpose_Invoker<T, sizeof(u64)> final
    {
        using input = const T&;
        using output = T&;
        using scalar_t = typename T::scalar_t;

        v_transpose_Invoker() = default;

        void operator()(input i0, input i1, input i2, input i3, input i4, input i5, input i6, input i7,
            output o0, output o1, output o2, output o3, output o4, output o5, output o6, output o7) const noexcept requires std::integral<scalar_t>
        {
            __m256i t0 = _mm256_unpacklo_epi64(static_cast<__m256i>(i0.data), static_cast<__m256i>(i1.data));
            __m256i t1 = _mm256_unpackhi_epi64(static_cast<__m256i>(i0.data), static_cast<__m256i>(i1.data));
            __m256i t2 = _mm256_unpacklo_epi64(static_cast<__m256i>(i2.data), static_cast<__m256i>(i3.data));
            __m256i t3 = _mm256_unpackhi_epi64(static_cast<__m256i>(i2.data), static_cast<__m256i>(i3.data));

            __m256i tt0 = _mm256_permute2x128_si256(t0, t2, 0x20);
            __m256i tt1 = _mm256_permute2x128_si256(t1, t3, 0x20);
            __m256i tt2 = _mm256_permute2x128_si256(t0, t2, 0x31);
            __m256i tt3 = _mm256_permute2x128_si256(t1, t3, 0x31);

            __m256i t4 = _mm256_unpacklo_epi64(static_cast<__m256i>(i4.data), static_cast<__m256i>(i5.data));
            __m256i t5 = _mm256_unpackhi_epi64(static_cast<__m256i>(i4.data), static_cast<__m256i>(i5.data));
            __m256i t6 = _mm256_unpacklo_epi64(static_cast<__m256i>(i6.data), static_cast<__m256i>(i7.data));
            __m256i t7 = _mm256_unpackhi_epi64(static_cast<__m256i>(i6.data), static_cast<__m256i>(i7.data));

            __m256i tt4 = _mm256_permute2x128_si256(t4, t6, 0x20);
            __m256i tt5 = _mm256_permute2x128_si256(t5, t7, 0x20);
            __m256i tt6 = _mm256_permute2x128_si256(t4, t6, 0x31);
            __m256i tt7 = _mm256_permute2x128_si256(t5, t7, 0x31);

            o0.data = tt0;
            o1.data = tt1;
            o2.data = tt2;
            o3.data = tt3;
            o4.data = tt4;
            o5.data = tt5;
            o6.data = tt6;
            o7.data = tt7;
        }

        void operator()(input i0, input i1, input i2, input i3, input i4, input i5, input i6, input i7,
            output o0, output o1, output o2, output o3, output o4, output o5, output o6, output o7) 
            const noexcept requires std::same_as<scalar_t, f64>
        {
            __m256d t0 = _mm256_unpacklo_pd(static_cast<__m256d>(i0.data), static_cast<__m256d>(i1.data));
            __m256d t1 = _mm256_unpackhi_pd(static_cast<__m256d>(i0.data), static_cast<__m256d>(i1.data));
            __m256d t2 = _mm256_unpacklo_pd(static_cast<__m256d>(i2.data), static_cast<__m256d>(i3.data));
            __m256d t3 = _mm256_unpackhi_pd(static_cast<__m256d>(i2.data), static_cast<__m256d>(i3.data));

            __m256d tt0 = _mm256_permute2f128_pd(t0, t2, 0x20);
            __m256d tt1 = _mm256_permute2f128_pd(t1, t3, 0x20);
            __m256d tt2 = _mm256_permute2f128_pd(t0, t2, 0x31);
            __m256d tt3 = _mm256_permute2f128_pd(t1, t3, 0x31);

            __m256d t4 = _mm256_unpacklo_pd(static_cast<__m256d>(i4.data), static_cast<__m256d>(i5.data));
            __m256d t5 = _mm256_unpackhi_pd(static_cast<__m256d>(i4.data), static_cast<__m256d>(i5.data));
            __m256d t6 = _mm256_unpacklo_pd(static_cast<__m256d>(i6.data), static_cast<__m256d>(i7.data));
            __m256d t7 = _mm256_unpackhi_pd(static_cast<__m256d>(i6.data), static_cast<__m256d>(i7.data));

            __m256d tt4 = _mm256_permute2f128_pd(t4, t6, 0x20);
            __m256d tt5 = _mm256_permute2f128_pd(t5, t7, 0x20);
            __m256d tt6 = _mm256_permute2f128_pd(t4, t6, 0x31);
            __m256d tt7 = _mm256_permute2f128_pd(t5, t7, 0x31);

            o0.data = tt0;
            o1.data = tt1;
            o2.data = tt2;
            o3.data = tt3;
            o4.data = tt4;
            o5.data = tt5;
            o6.data = tt6;
            o7.data = tt7;
        }
    };

    template<BasicArithmetic Element_t>
    void transpose_(const Element_t* src_ptr, Element_t* dst_ptr, usize src_rows, usize src_cols) noexcept
    {
        constexpr usize block_size = simd::AVX_t<Element_t>::batch_size;

        const usize row_blocks = src_rows / block_size;
        const usize col_blocks = src_cols / block_size;

        v_transpose_Invoker<simd::AVX_t<Element_t>, sizeof(typename simd::AVX_t<Element_t>::scalar_t)> invoker_8x8;

        alignas(32) simd::AVX_t<Element_t> block[block_size];
        alignas(32) simd::AVX_t<Element_t> transposed[block_size];

        for (usize i = 0; i < row_blocks; ++i)
        {
            for (usize j = 0; j < col_blocks; ++j)
            {
                if (j + 1 < col_blocks)
                {
                    for (usize k = 0; k < block_size; ++k)
                    {
                        _mm_prefetch(
                            reinterpret_cast<const char*>(&src_ptr[(i * block_size + k) * src_cols + (j + 1) * block_size]),
                            _MM_HINT_T0
                        );
                    }
                }

                for (usize k = 0; k < block_size; ++k)
                {
                    block[k] = simd::AVX_t<Element_t>(&src_ptr[(i * block_size + k) * src_cols + j * block_size]);
                }

                invoker_8x8(
                    block[0], block[1], block[2], block[3],
                    block[4], block[5], block[6], block[7],

                    transposed[0], transposed[1], transposed[2], transposed[3],
                    transposed[4], transposed[5], transposed[6], transposed[7]
                );

                for (usize k = 0; k < block_size; ++k)
                {
                    transposed[k].download(&dst_ptr[(j * block_size + k) * src_rows + i * block_size]);
                }

                if (i + 1 < row_blocks)
                {
                    for (usize k = 0; k < block_size; ++k)
                    {
                        _mm_prefetch(
                            reinterpret_cast<const char*>(&dst_ptr[(j * block_size + k) * src_rows + (i + 1) * block_size]),
                            _MM_HINT_T0
                        );
                    }
                }
            }
        }

        for (usize i = 0; i < src_rows; ++i)
        {
            for (usize j = col_blocks * block_size; j < src_cols; ++j)
            {
                dst_ptr[j * src_rows + i] = src_ptr[i * src_cols + j];
            }
        }

        for (usize i = row_blocks * block_size; i < src_rows; ++i)
        {
            for (usize j = 0; j < col_blocks * block_size; ++j)
            {
                dst_ptr[j * src_rows + i] = src_ptr[i * src_cols + j];
            }
        }
    }
}