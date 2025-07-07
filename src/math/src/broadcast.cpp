module;
#include <immintrin.h>
#include <intrin.h>
module foye.algorithm;
import foye.foye_core;
import foye.farray;
import foye.simd;
import std;

namespace fy
{
	template<BasicArithmetic Element_t>
	void setall(Element_t* ptr, Element_t toset, usize count) noexcept
	{
        if constexpr (sizeof(Element_t) == 1)
        {
            if (toset == Element_t{ 0 })
            {
                std::memset(ptr, 0, count * sizeof(Element_t));
                return;
            }
        }

		using namespace simd;
		using simd_t = AVX_t<Element_t>;

        constexpr usize elements_per_stride = simd_t::batch_size;

        usize i = 0;
        if (toset == Element_t(0))
        {
            const simd_t zero = v_broadcast_zero<simd_t>();
            for (; i + simd_t::batch_size <= count; i += simd_t::batch_size)
            {
                _mm256_store_si256(reinterpret_cast<__m256i*>(ptr + i), zero.data);
            }
        }
        else
        {
            const simd_t v_toset(toset);
            for (; i + simd_t::batch_size <= count; i += simd_t::batch_size)
            {
                v_toset.download(ptr + i);
            }
        }

        for (; i < count; ++i)
        {
            ptr[i] = toset;
        }
	}

    template void setall<u8>(u8* ptr, u8 toset, usize count) noexcept;
    template void setall<u16>(u16* ptr, u16 toset, usize count) noexcept;
    template void setall<u32>(u32* ptr, u32 toset, usize count) noexcept;
    template void setall<u64>(u64* ptr, u64 toset, usize count) noexcept;
    template void setall<i8>(i8* ptr, i8 toset, usize count) noexcept;
    template void setall<i16>(i16* ptr, i16 toset, usize count) noexcept;
    template void setall<i32>(i32* ptr, i32 toset, usize count) noexcept;
    template void setall<i64>(i64* ptr, i64 toset, usize count) noexcept;
}