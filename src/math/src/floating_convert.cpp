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
	template<> void convert<f32, bfloat16>(const bfloat16* from, f32* to, usize count) noexcept
	{
		usize i = 0;
		for (; i + 8 <= count; i += 8)
		{
			using namespace simd;
			v_bfloat16x8 vsrc(from + i);
			v_float32x8 vdst = v_convert<v_float32x8>(vsrc);
			vdst.streamback(to + i);
		}

		for (; i < count; ++i)
		{
			to[i] = static_cast<f32>(from[i]);
		}
	}


	template<> void convert<bfloat16, f32>(const f32* from, bfloat16* to, usize count) noexcept
	{
		constexpr u32 EXPONENT_MASK = 0x7F800000;
		constexpr u32 MANTISSA_MASK = 0x007FFFFF;
		constexpr u32 LOW_16_BITS_MASK = 0x0000FFFF;
		constexpr u32 QUIET_NAN_BIT = 0x00000001;

		usize i = 0;

		using namespace simd;

		//const v_uint32x8 v_exp_mask(EXPONENT_MASK);
		//const v_uint32x8 v_mant_mask(MANTISSA_MASK);
		//const v_uint32x8 v_low16_bit_mask(LOW_16_BITS_MASK);
		//const v_uint32x8 v_quiet_bit(QUIET_NAN_BIT);
		//const v_uint32x8 v_zero = v_brocast_zero<v_uint32x8>();
		for (; i + 8 <= count; i += 8)
		{
			v_float32x8 vsrc(from + i);
			v_bfloat16x8 vdst = v_convert<v_bfloat16x8>(vsrc);
			vdst.streamback(to + i);
		}

		for (; i < count; ++i)
		{
			to[i] = bfloat16(from[i]);
		}
	}
}

//
//constexpr void scalar_float32_to_bfloat16__(const f32* src, bfloat16* dst, usize count)
//{
//	for (usize i = 0; i < count; ++i)
//	{
//		dst[i] = bfloat16(src[i]);
//	}
//}
//
//void bf16_fp32_saclar_base(const bfloat16* src, f32* dst, usize count)
//{
//	for (usize i = 0; i < count; ++i)
//	{
//		dst[i] = static_cast<f32>(src[i]);
//	}
//}
//
//int main()
//{
//	farray<f32> src(640 * 640 + 14);
//
//	rand_MT19937 rand;
//	src.for_each([&](f32& v) {v = rand.uniform(-1., 1.); });
//
//	farray<bfloat16> bf16_0(src.size());
//	farray<bfloat16> bf16_1(src.size());
//
//	farray<f32> res_0(src.size());
//	farray<f32> res_1(src.size());
//
//	{
//		auto [cost_0, cost_1] = bench_mark<100>(
//			[&]() -> void
//			{
//				scalar_float32_to_bfloat16__(src.data(), bf16_0.data(), src.size());
//			},
//			[&]() -> void
//			{
//				fy::convert(src.data(), bf16_1.data(), src.size());
//			}
//		);
//
//		if (!(std::memcmp(bf16_0.data(), bf16_1.data(), src.size() * sizeof(bfloat16)) == 0))
//		{
//			throw std::runtime_error("convert fp32 to bf16 result not match");
//		}
//
//		fy::println("fp32 => bf16 scalar: ", cost_0, "ms");
//		fy::println("fp32 => bf16 simd:   ", cost_1, "ms");
//	}
//
//	fy::print('\n');
//
//	{
//		auto [cost_0, cost_1] = bench_mark<100>(
//			[&]() -> void
//			{
//				bf16_fp32_saclar_base(bf16_0.data(), res_0.data(), src.size());
//			},
//			[&]() -> void
//			{
//				fy::convert(bf16_1.data(), res_1.data(), src.size()); //
//			}
//		);
//
//		if (!(std::memcmp(res_0.data(), res_1.data(), src.size() * sizeof(bfloat16)) == 0))
//		{
//			throw std::runtime_error("convert bf16 to fp32 result not match");
//		}
//
//		fy::println("bf16 => fp32 scalar: ", cost_0, "ms");
//		fy::println("bf16 => fp32 simd:   ", cost_1, "ms");
//	}
//
//	fy::println(src.subarray(0, 6));
//	fy::println(res_0.subarray(0, 6));
//	fy::println(res_1.subarray(0, 6));
//
//	/*fy::println(res_0.subarray(0, 6));
//	fy::println(res_1.subarray(0, 6));
//
//	fy::println(
//		std::memcmp(res_0.data(), res_1.data(), src.size() * sizeof(bfloat16)) == 0
//	);*/
//
//
//}
