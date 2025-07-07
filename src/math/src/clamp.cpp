module;
#include <immintrin.h>
#include <intrin.h>

module foye.algorithm;
import foye.foye_core;
import foye.simd;
import std;

namespace fy
{
	template<BasicArithmetic Element_t>
	void clamp(const Element_t* src_ptr, Element_t* dst_ptr, usize count, Element_t minVal, Element_t maxVal) noexcept
	{
        const bool min_reach_min = (minVal == std::numeric_limits<Element_t>::lowest());
        const bool max_reach_max = (maxVal == std::numeric_limits<Element_t>::max());

        if (min_reach_min && max_reach_max)
        {
            std::memcpy(dst_ptr, src_ptr, count);
            return;
        }
        if (minVal == maxVal)
        {
            std::memset(dst_ptr, minVal, count);
            return;
        }

        const usize align = reinterpret_cast<uintptr_t>(src_ptr) % 32;
        if (align != 0)
        {
            const usize prefix = std::min<usize>(32 - align, count);
            for (usize i = 0; i < prefix; ++i)
            {
                Element_t val = src_ptr[i];
                dst_ptr[i] = (val < minVal) ? minVal : (val > maxVal) ? maxVal : val;
            }
            src_ptr += prefix;
            dst_ptr += prefix;
            count -= prefix;
        }

        using namespace simd;
        constexpr usize batch_size = AVX_t<Element_t>::batch_size;

        if (count >= batch_size)
        {
            AVX_t<Element_t> vmin(minVal);
            AVX_t<Element_t> vmax(maxVal);

            const usize blocks = count / batch_size;
            const Element_t* end = src_ptr + blocks * batch_size;

            if (min_reach_min)
            {
                while (src_ptr < end)
                {
                    AVX_t<Element_t> data(src_ptr);
                    src_ptr += batch_size;
                    data = v_min_replace<AVX_t<Element_t>>(data, vmax);
                    data.streamback(dst_ptr);
                    dst_ptr += batch_size;
                }
            }
            else if (max_reach_max)
            {
                while (src_ptr < end)
                {
                    AVX_t<Element_t> data(src_ptr);
                    src_ptr += batch_size;
                    data = v_max_replace<AVX_t<Element_t>>(data, vmin);
                    data.streamback(dst_ptr);
                    dst_ptr += batch_size;
                }
            }
            else
            {
                while (src_ptr < end)
                {
                    AVX_t<Element_t> data(src_ptr);
                    src_ptr += batch_size;
                    data = v_max_replace<AVX_t<Element_t>>(data, vmin);
                    data = v_min_replace<AVX_t<Element_t>>(data, vmax);
                    data.streamback(dst_ptr);
                    dst_ptr += batch_size;
                }
            }

            _mm_sfence();
        }

        count %= batch_size;
        for (usize i = 0; i < count; i++)
        {
            Element_t val = src_ptr[i];
            dst_ptr[i] = (val < minVal) ? minVal : (val > maxVal) ? maxVal : val;
        }
	}

    template void clamp(const u8*, u8*, usize, u8, u8) noexcept;
    template void clamp(const u16*, u16*, usize, u16, u16) noexcept;
    template void clamp(const u32*, u32*, usize, u32, u32) noexcept;
    template void clamp(const u64*, u64*, usize, u64, u64) noexcept;

    template void clamp(const i8*, i8*, usize, i8, i8) noexcept;
    template void clamp(const i16*, i16*, usize, i16, i16) noexcept;
    template void clamp(const i32*, i32*, usize, i32, i32) noexcept;
    template void clamp(const i64*, i64*, usize, i64, i64) noexcept;

    template void clamp(const f16*, f16*, usize, f16, f16) noexcept;
    template void clamp(const f32*, f32*, usize, f32, f32) noexcept;
    template void clamp(const f64*, f64*, usize, f64, f64) noexcept;
}