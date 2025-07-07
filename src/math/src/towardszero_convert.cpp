module;
#include <immintrin.h>
#include <intrin.h>

module foye.algorithm;
import foye.foye_core;
import std;

#pragma warning(disable: 4309)
#pragma warning(disable: 4244)

namespace fy
{
	template<BasicArithmetic Src_t, BasicArithmetic Dst_t>
	void cast_towardszero(const Src_t* src_ptr, Dst_t* dst_ptr, usize count) noexcept
	{
		std::unreachable();
	}

	template<bool src_signed, bool dst_signed> struct convert_8bits_to_16bits_integral_Invoker
	{
		static inline constexpr usize cache_line_size = 64;
		static inline constexpr usize simd_stride = 64;
		static inline constexpr usize prefetch_ahead = 4 * cache_line_size;

		convert_8bits_to_16bits_integral_Invoker() = default;

		using src_t = std::conditional_t<src_signed, i8, u8>;
		using dst_t = std::conditional_t<dst_signed, i16, u16>;

		static void process(const src_t* src_ptr, dst_t* dst_ptr, usize count) noexcept
		{
			uintptr_t base_addr = reinterpret_cast<uintptr_t>(dst_ptr);
			usize i = 0;

			while (i < count && (reinterpret_cast<uintptr_t>(dst_ptr + i) % 16) != 0)
			{
				dst_ptr[i] = static_cast<dst_t>(src_ptr[i]);
				i++;
			}

			const src_t* src_aligned = src_ptr + i;
			dst_t* dst_aligned = dst_ptr + i;
			const usize main_loop_count = (count - i) / (simd_stride);
			const src_t* main_loop_end = src_aligned + main_loop_count * simd_stride;

			for (; src_aligned < main_loop_end; src_aligned += simd_stride, dst_aligned += simd_stride)
			{
				_mm_prefetch(reinterpret_cast<const char*>(src_aligned) + prefetch_ahead, _MM_HINT_T0);

				const __m128i block0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src_aligned));
				const __m128i block1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src_aligned + 16));
				const __m128i block2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src_aligned + 32));
				const __m128i block3 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src_aligned + 48));

				const __m128i block0_hi = _mm_unpackhi_epi64(block0, block0);
				const __m128i block1_hi = _mm_unpackhi_epi64(block1, block1);
				const __m128i block2_hi = _mm_unpackhi_epi64(block2, block2);
				const __m128i block3_hi = _mm_unpackhi_epi64(block3, block3);

				__m128i lo0;
				__m128i hi0;
				__m128i lo1;
				__m128i hi1;
				__m128i lo2;
				__m128i hi2;
				__m128i lo3;
				__m128i hi3;

				if constexpr (src_signed)
				{
					lo0 = _mm_cvtepi8_epi16(block0);
					hi0 = _mm_cvtepi8_epi16(block0_hi);
					lo1 = _mm_cvtepi8_epi16(block1);
					hi1 = _mm_cvtepi8_epi16(block1_hi);
					lo2 = _mm_cvtepi8_epi16(block2);
					hi2 = _mm_cvtepi8_epi16(block2_hi);
					lo3 = _mm_cvtepi8_epi16(block3);
					hi3 = _mm_cvtepi8_epi16(block3_hi);
				}
				else
				{
					lo0 = _mm_cvtepu8_epi16(block0);
					hi0 = _mm_cvtepu8_epi16(block0_hi);
					lo1 = _mm_cvtepu8_epi16(block1);
					hi1 = _mm_cvtepu8_epi16(block1_hi);
					lo2 = _mm_cvtepu8_epi16(block2);
					hi2 = _mm_cvtepu8_epi16(block2_hi);
					lo3 = _mm_cvtepu8_epi16(block3);
					hi3 = _mm_cvtepu8_epi16(block3_hi);
				}

				_mm_stream_si128(reinterpret_cast<__m128i*>(dst_aligned), lo0);
				_mm_stream_si128(reinterpret_cast<__m128i*>(dst_aligned + 8), hi0);
				_mm_stream_si128(reinterpret_cast<__m128i*>(dst_aligned + 16), lo1);
				_mm_stream_si128(reinterpret_cast<__m128i*>(dst_aligned + 24), hi1);
				_mm_stream_si128(reinterpret_cast<__m128i*>(dst_aligned + 32), lo2);
				_mm_stream_si128(reinterpret_cast<__m128i*>(dst_aligned + 40), hi2);
				_mm_stream_si128(reinterpret_cast<__m128i*>(dst_aligned + 48), lo3);
				_mm_stream_si128(reinterpret_cast<__m128i*>(dst_aligned + 56), hi3);
			}

			usize processed = main_loop_count * simd_stride;
			usize remaining = count - i - processed;
			const src_t* src_remaining = src_aligned;
			dst_t* dst_remaining = dst_aligned;

			while (remaining >= 16)
			{
				const __m128i block = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src_remaining));
				__m128i hi, lo;
				if constexpr (src_signed)
				{
					lo = _mm_cvtepi8_epi16(block);
					hi = _mm_cvtepi8_epi16(_mm_unpackhi_epi64(block, block));
				}
				else
				{
					lo = _mm_cvtepu8_epi16(block);
					hi = _mm_cvtepu8_epi16(_mm_unpackhi_epi64(block, block));
				}

				_mm_stream_si128(reinterpret_cast<__m128i*>(dst_remaining), lo);
				_mm_stream_si128(reinterpret_cast<__m128i*>(dst_remaining + 8), hi);

				src_remaining += 16;
				dst_remaining += 16;
				remaining -= 16;
			}

			for (; remaining > 0; --remaining)
			{
				*dst_remaining++ = static_cast<dst_t>(*src_remaining++);
			}

			_mm_sfence();
		}
	};

	template<> void cast_towardszero<u8, i8>(const u8* src_ptr, i8* dst_ptr, usize count) noexcept
	{
		constexpr usize cache_line_size = 64;
		constexpr usize simd_stride = 64;
		constexpr usize prefetch_ahead = 4 * cache_line_size;

		const uintptr_t base_addr = reinterpret_cast<uintptr_t>(dst_ptr);
		const usize misalignment = base_addr % 16;
		usize i = 0;

		if (misalignment > 0)
		{
			const usize align_count = std::min(16 - misalignment, count);
			for (; i < align_count; ++i)
			{
				dst_ptr[i] = static_cast<i8>(src_ptr[i]);
			}
		}

		const u8* src_aligned = src_ptr + i;
		i8* dst_aligned = dst_ptr + i;
		const usize main_loop_count = (count - i) / simd_stride;
		const u8* main_loop_end = src_aligned + main_loop_count * simd_stride;

		for (; src_aligned < main_loop_end; src_aligned += simd_stride, dst_aligned += simd_stride)
		{
			_mm_prefetch(reinterpret_cast<const char*>(src_aligned) + prefetch_ahead, _MM_HINT_T0);

			const __m128i block0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src_aligned));
			const __m128i block1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src_aligned + 16));
			const __m128i block2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src_aligned + 32));
			const __m128i block3 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src_aligned + 48));

			_mm_stream_si128(reinterpret_cast<__m128i*>(dst_aligned), block0);
			_mm_stream_si128(reinterpret_cast<__m128i*>(dst_aligned + 16), block1);
			_mm_stream_si128(reinterpret_cast<__m128i*>(dst_aligned + 32), block2);
			_mm_stream_si128(reinterpret_cast<__m128i*>(dst_aligned + 48), block3);
		}

		usize processed = main_loop_count * simd_stride;
		usize remaining = count - i - processed;

		while (remaining >= 16)
		{
			const __m128i block = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src_aligned));
			_mm_stream_si128(reinterpret_cast<__m128i*>(dst_aligned), block);
			src_aligned += 16;
			dst_aligned += 16;
			remaining -= 16;
		}

		for (; remaining > 0; --remaining)
		{
			*dst_aligned++ = static_cast<i8>(*src_aligned++);
		}

		_mm_sfence();
	}

	template<> void cast_towardszero<u8, i16>(const u8* src_ptr, i16* dst_ptr, usize count) noexcept
	{
		convert_8bits_to_16bits_integral_Invoker<false, true>::process(src_ptr, dst_ptr, count);
	}

	template<> void cast_towardszero<u8, i32>(const u8* src_ptr, i32* dst_ptr, usize count) noexcept
	{
		constexpr usize cache_line_size = 64;
		constexpr usize simd_stride = 64;
		constexpr usize prefetch_ahead = 4 * cache_line_size;

		const uintptr_t base_addr = reinterpret_cast<uintptr_t>(dst_ptr);
		const usize misalignment = (base_addr % 32) / sizeof(i32);
		usize i = 0;

		if (misalignment > 0)
		{
			const usize align_count = std::min(static_cast<usize>(32 / sizeof(i32) - misalignment), count);
			for (; i < align_count; ++i)
			{
				dst_ptr[i] = static_cast<i32>(src_ptr[i]);
			}
		}

		const u8* src_aligned = src_ptr + i;
		i32* dst_aligned = dst_ptr + i;
		const usize main_loop_count = (count - i) / simd_stride;
		const u8* main_loop_end = src_aligned + main_loop_count * simd_stride;

		for (; src_aligned < main_loop_end; src_aligned += simd_stride, dst_aligned += simd_stride)
		{
			_mm_prefetch(reinterpret_cast<const char*>(src_aligned) + prefetch_ahead, _MM_HINT_T0);

			const __m128i block0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src_aligned));
			const __m128i block1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src_aligned + 16));
			const __m128i block2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src_aligned + 32));
			const __m128i block3 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src_aligned + 48));

			__m256i b0_low = _mm256_cvtepu8_epi32(block0);
			__m256i b0_high = _mm256_cvtepu8_epi32(_mm_srli_si128(block0, 8));
			_mm256_stream_si256(reinterpret_cast<__m256i*>(dst_aligned), b0_low);
			_mm256_stream_si256(reinterpret_cast<__m256i*>(dst_aligned + 8), b0_high);

			__m256i b1_low = _mm256_cvtepu8_epi32(block1);
			__m256i b1_high = _mm256_cvtepu8_epi32(_mm_srli_si128(block1, 8));
			_mm256_stream_si256(reinterpret_cast<__m256i*>(dst_aligned + 16), b1_low);
			_mm256_stream_si256(reinterpret_cast<__m256i*>(dst_aligned + 24), b1_high);

			__m256i b2_low = _mm256_cvtepu8_epi32(block2);
			__m256i b2_high = _mm256_cvtepu8_epi32(_mm_srli_si128(block2, 8));
			_mm256_stream_si256(reinterpret_cast<__m256i*>(dst_aligned + 32), b2_low);
			_mm256_stream_si256(reinterpret_cast<__m256i*>(dst_aligned + 40), b2_high);

			__m256i b3_low = _mm256_cvtepu8_epi32(block3);
			__m256i b3_high = _mm256_cvtepu8_epi32(_mm_srli_si128(block3, 8));
			_mm256_stream_si256(reinterpret_cast<__m256i*>(dst_aligned + 48), b3_low);
			_mm256_stream_si256(reinterpret_cast<__m256i*>(dst_aligned + 56), b3_high);
		}

		usize remaining = count - (main_loop_count * simd_stride + i);
		for (usize j = 0; j < remaining; ++j)
		{
			dst_aligned[j] = static_cast<i32>(src_aligned[j]);
		}

		_mm_sfence();
	}

	template<> void cast_towardszero<u8, i64>(const u8* src_ptr, i64* dst_ptr, usize count) noexcept
	{
		constexpr usize cache_line_size = 64;
		constexpr usize simd_stride = 32;
		constexpr usize prefetch_ahead = 4 * cache_line_size;

		const uintptr_t base_addr = reinterpret_cast<uintptr_t>(dst_ptr);
		const usize misalignment = (base_addr % 32) / sizeof(i64);
		usize i = 0;

		if (misalignment > 0)
		{
			const usize align_count = std::min(static_cast<usize>(32 / sizeof(i64) - misalignment), count);
			for (; i < align_count; ++i)
			{
				dst_ptr[i] = static_cast<i64>(src_ptr[i]);
			}
		}

		const u8* src_aligned = src_ptr + i;
		i64* dst_aligned = dst_ptr + i;
		const usize main_loop_count = (count - i) / simd_stride;
		const u8* main_loop_end = src_aligned + main_loop_count * simd_stride;

		for (; src_aligned < main_loop_end; src_aligned += simd_stride, dst_aligned += simd_stride)
		{
			_mm_prefetch(reinterpret_cast<const char*>(src_aligned) + prefetch_ahead, _MM_HINT_T0);

			for (usize batch = 0; batch < 4; ++batch)
			{
				const u8* batch_src = src_aligned + batch * 8;
				i64* batch_dst = dst_aligned + batch * 8;

				const __m128i chunk = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(batch_src));
				const __m256i u32_vec = _mm256_cvtepu8_epi32(chunk);

				const __m256i i64_low = _mm256_cvtepu32_epi64(_mm256_extracti128_si256(u32_vec, 0));
				const __m256i i64_high = _mm256_cvtepu32_epi64(_mm256_extracti128_si256(u32_vec, 1));

				_mm256_stream_si256(reinterpret_cast<__m256i*>(batch_dst), i64_low);
				_mm256_stream_si256(reinterpret_cast<__m256i*>(batch_dst + 4), i64_high);
			}
		}

		const usize remaining = count - (main_loop_count * simd_stride + i);
		for (usize j = 0; j < remaining; ++j)
		{
			dst_aligned[j] = static_cast<i64>(src_aligned[j]);
		}

		_mm_sfence();
	}

	template<> void cast_towardszero<u8, u8>(const u8* src_ptr, u8* dst_ptr, usize count) noexcept
	{
		std::memcpy(dst_ptr, src_ptr, count * sizeof(u8));
	}

	template<> void cast_towardszero<u8, u16>(const u8* src_ptr, u16* dst_ptr, usize count) noexcept
	{
		convert_8bits_to_16bits_integral_Invoker<false, false>::process(src_ptr, dst_ptr, count);
	}

	template<> void cast_towardszero<u8, u32>(const u8* src_ptr, u32* dst_ptr, usize count) noexcept
	{
		constexpr usize cache_line_size = 64;
		constexpr usize simd_stride = 64;
		constexpr usize prefetch_ahead = 4 * cache_line_size;

		usize i = 0;

		while (i < count && (reinterpret_cast<uintptr_t>(dst_ptr + i) % 16) != 0)
		{
			dst_ptr[i] = static_cast<u32>(src_ptr[i]);
			i++;
		}

		const u8* src_aligned = src_ptr + i;
		u32* dst_aligned = dst_ptr + i;
		const usize main_loop_count = (count - i) / simd_stride;
		const u8* main_loop_end = src_aligned + main_loop_count * simd_stride;

		for (; src_aligned < main_loop_end; src_aligned += simd_stride, dst_aligned += simd_stride)
		{
			_mm_prefetch(reinterpret_cast<const char*>(src_aligned) + prefetch_ahead, _MM_HINT_T0);

			const __m128i block0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src_aligned));
			const __m128i block1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src_aligned + 16));
			const __m128i block2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src_aligned + 32));
			const __m128i block3 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src_aligned + 48));

			_mm_stream_si128(reinterpret_cast<__m128i*>(dst_aligned + 0), _mm_cvtepu8_epi32(block0));
			_mm_stream_si128(reinterpret_cast<__m128i*>(dst_aligned + 4), _mm_cvtepu8_epi32(_mm_srli_si128(block0, 4)));
			_mm_stream_si128(reinterpret_cast<__m128i*>(dst_aligned + 8), _mm_cvtepu8_epi32(_mm_srli_si128(block0, 8)));
			_mm_stream_si128(reinterpret_cast<__m128i*>(dst_aligned + 12), _mm_cvtepu8_epi32(_mm_srli_si128(block0, 12)));

			_mm_stream_si128(reinterpret_cast<__m128i*>(dst_aligned + 16), _mm_cvtepu8_epi32(block1));
			_mm_stream_si128(reinterpret_cast<__m128i*>(dst_aligned + 20), _mm_cvtepu8_epi32(_mm_srli_si128(block1, 4)));
			_mm_stream_si128(reinterpret_cast<__m128i*>(dst_aligned + 24), _mm_cvtepu8_epi32(_mm_srli_si128(block1, 8)));
			_mm_stream_si128(reinterpret_cast<__m128i*>(dst_aligned + 28), _mm_cvtepu8_epi32(_mm_srli_si128(block1, 12)));

			_mm_stream_si128(reinterpret_cast<__m128i*>(dst_aligned + 32), _mm_cvtepu8_epi32(block2));
			_mm_stream_si128(reinterpret_cast<__m128i*>(dst_aligned + 36), _mm_cvtepu8_epi32(_mm_srli_si128(block2, 4)));
			_mm_stream_si128(reinterpret_cast<__m128i*>(dst_aligned + 40), _mm_cvtepu8_epi32(_mm_srli_si128(block2, 8)));
			_mm_stream_si128(reinterpret_cast<__m128i*>(dst_aligned + 44), _mm_cvtepu8_epi32(_mm_srli_si128(block2, 12)));

			_mm_stream_si128(reinterpret_cast<__m128i*>(dst_aligned + 48), _mm_cvtepu8_epi32(block3));
			_mm_stream_si128(reinterpret_cast<__m128i*>(dst_aligned + 52), _mm_cvtepu8_epi32(_mm_srli_si128(block3, 4)));
			_mm_stream_si128(reinterpret_cast<__m128i*>(dst_aligned + 56), _mm_cvtepu8_epi32(_mm_srli_si128(block3, 8)));
			_mm_stream_si128(reinterpret_cast<__m128i*>(dst_aligned + 60), _mm_cvtepu8_epi32(_mm_srli_si128(block3, 12)));
		}

		usize processed = main_loop_count * simd_stride;
		usize remaining = count - i - processed;
		const u8* src_remaining = src_aligned;
		u32* dst_remaining = dst_aligned;

		while (remaining >= 16)
		{
			const __m128i block = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src_remaining));

			_mm_stream_si128(reinterpret_cast<__m128i*>(dst_remaining + 0), _mm_cvtepu8_epi32(block));
			_mm_stream_si128(reinterpret_cast<__m128i*>(dst_remaining + 4), _mm_cvtepu8_epi32(_mm_srli_si128(block, 4)));
			_mm_stream_si128(reinterpret_cast<__m128i*>(dst_remaining + 8), _mm_cvtepu8_epi32(_mm_srli_si128(block, 8)));
			_mm_stream_si128(reinterpret_cast<__m128i*>(dst_remaining + 12), _mm_cvtepu8_epi32(_mm_srli_si128(block, 12)));

			src_remaining += 16;
			dst_remaining += 16;
			remaining -= 16;
		}

		for (; remaining > 0; --remaining)
		{
			*dst_remaining++ = static_cast<u32>(*src_remaining++);
		}

		_mm_sfence();
	}

	template<> void cast_towardszero<u8, u64>(const u8* src_ptr, u64* dst_ptr, usize count) noexcept
	{
		constexpr usize cache_line_size = 64;
		constexpr usize simd_stride = 32;
		constexpr usize prefetch_ahead = 4 * cache_line_size;

		const uintptr_t base_addr = reinterpret_cast<uintptr_t>(dst_ptr);
		const usize misalignment = (base_addr % 32) / sizeof(u64);
		usize i = 0;

		if (misalignment > 0)
		{
			const usize align_count = std::min(static_cast<usize>(32 / sizeof(u64) - misalignment), count);
			for (; i < align_count; ++i)
			{
				dst_ptr[i] = static_cast<u64>(src_ptr[i]);
			}
		}

		const u8* src_aligned = src_ptr + i;
		u64* dst_aligned = dst_ptr + i;
		const usize main_loop_count = (count - i) / simd_stride;
		const u8* main_loop_end = src_aligned + main_loop_count * simd_stride;

		for (; src_aligned < main_loop_end; src_aligned += simd_stride, dst_aligned += simd_stride)
		{
			_mm_prefetch(reinterpret_cast<const char*>(src_aligned) + prefetch_ahead, _MM_HINT_T0);

			for (usize batch = 0; batch < 4; ++batch)
			{
				const u8* batch_src = src_aligned + batch * 8;
				u64* batch_dst = dst_aligned + batch * 8;

				const __m128i chunk = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(batch_src));
				const __m256i u32_vec = _mm256_cvtepu8_epi32(chunk);

				const __m256i u64_low = _mm256_cvtepu32_epi64(_mm256_extracti128_si256(u32_vec, 0));
				const __m256i u64_high = _mm256_cvtepu32_epi64(_mm256_extracti128_si256(u32_vec, 1));

				_mm256_stream_si256(reinterpret_cast<__m256i*>(batch_dst), u64_low);
				_mm256_stream_si256(reinterpret_cast<__m256i*>(batch_dst + 4), u64_high);
			}
		}

		const usize remaining = count - (main_loop_count * simd_stride + i);
		for (usize j = 0; j < remaining; ++j)
		{
			dst_aligned[j] = static_cast<u64>(src_aligned[j]);
		}

		_mm_sfence();
	}

	template<> void cast_towardszero<u8, f16>(const u8* src_ptr, f16* dst_ptr, usize count) noexcept
	{
		constexpr usize cache_line_size = 64;
		constexpr usize simd_stride = 64;
		constexpr usize prefetch_ahead = 4 * cache_line_size;

		const uintptr_t base_addr = reinterpret_cast<uintptr_t>(dst_ptr);
		const usize misalignment = (base_addr % 32) / sizeof(f16);
		usize i = 0;

		if (misalignment > 0)
		{
			const usize align_count = std::min(static_cast<usize>(32 / sizeof(f16) - misalignment), count);

			for (; i + 8 <= align_count; i += 8)
			{
				__m128i chunk = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(src_ptr + i));
				__m256i u16_vec = _mm256_cvtepu8_epi16(chunk);
				__m128i u16_low = _mm256_extracti128_si256(u16_vec, 0);

				__m256i u32_low = _mm256_cvtepu16_epi32(u16_low);
				__m256 f32_low = _mm256_cvtepi32_ps(u32_low);

				__m128i h = _mm256_cvtps_ph(f32_low, _MM_FROUND_TO_NEAREST_INT);
				_mm_storeu_si128(reinterpret_cast<__m128i*>(dst_ptr + i), h);
			}

			for (; i < align_count; ++i)
			{
				dst_ptr[i] = f16(static_cast<f32>(src_ptr[i]));
			}
		}

		const u8* src_aligned = src_ptr + i;
		f16* dst_aligned = dst_ptr + i;
		const usize main_loop_count = (count - i) / simd_stride;
		const u8* main_loop_end = src_aligned + main_loop_count * simd_stride;

		for (; src_aligned < main_loop_end; src_aligned += simd_stride, dst_aligned += simd_stride)
		{
			_mm_prefetch(reinterpret_cast<const char*>(src_aligned) + prefetch_ahead, _MM_HINT_T0);

			for (usize pb = 0; pb < 4; ++pb)
			{
				const __m128i block_src = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src_aligned + (pb * 16)));
				const __m256i u16_block = _mm256_cvtepu8_epi16(block_src);

				const __m128i u16_low = _mm256_extracti128_si256(u16_block, 0);
				const __m128i u16_high = _mm256_extracti128_si256(u16_block, 1);

				const __m256i u32_low = _mm256_cvtepu16_epi32(u16_low);
				const __m256i u32_high = _mm256_cvtepu16_epi32(u16_high);

				const __m256 f32_low = _mm256_cvtepi32_ps(u32_low);
				const __m256 f32_high = _mm256_cvtepi32_ps(u32_high);

				const __m128i h_low = _mm256_cvtps_ph(f32_low, _MM_FROUND_TO_NEAREST_INT);
				const __m128i h_high = _mm256_cvtps_ph(f32_high, _MM_FROUND_TO_NEAREST_INT);

				__m256i vres = _mm256_set_m128i(h_high, h_low);
				_mm256_stream_si256(reinterpret_cast<__m256i*>(dst_aligned + (pb * 16)), vres);
			}
		}

		usize processed = main_loop_count * simd_stride;
		usize remaining = count - i - processed;

		if (remaining >= 8)
		{
			__m128i chunk = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(src_aligned));
			__m256i u16_vec = _mm256_cvtepu8_epi16(chunk);
			__m128i u16_low = _mm256_extracti128_si256(u16_vec, 0);

			__m256i u32_low = _mm256_cvtepu16_epi32(u16_low);
			__m256 f32_low = _mm256_cvtepi32_ps(u32_low);

			__m128i h = _mm256_cvtps_ph(f32_low, _MM_FROUND_TO_NEAREST_INT);
			_mm_storeu_si128(reinterpret_cast<__m128i*>(dst_aligned), h);
			src_aligned += 8;
			dst_aligned += 8;
			remaining -= 8;
		}

		for (; remaining > 0; --remaining)
		{
			*dst_aligned++ = f16(static_cast<f32>(*src_aligned++));
		}

		_mm_sfence();
	}

	template<> void cast_towardszero<u8, f32>(const u8* src_ptr, f32* dst_ptr, usize count) noexcept
	{
		constexpr usize cache_line_size = 64;
		constexpr usize simd_stride = 64;
		constexpr usize prefetch_ahead = 4 * cache_line_size;

		const uintptr_t base_addr = reinterpret_cast<uintptr_t>(dst_ptr);
		const usize misalignment = (base_addr % 32) / sizeof(f32);
		usize i = 0;

		if (misalignment > 0)
		{
			const usize align_count = std::min(static_cast<usize>(32 / sizeof(f32) - misalignment), count);

			for (; i + 4 <= align_count; i += 4)
			{
				const __m128i chunk = _mm_cvtsi32_si128(*reinterpret_cast<const u32*>(src_ptr + i));
				const __m256i ints = _mm256_cvtepu8_epi32(chunk);
				const __m256 floats = _mm256_cvtepi32_ps(ints);
				_mm_storeu_ps(dst_ptr + i, _mm256_castps256_ps128(floats));
			}

			for (; i < align_count; ++i)
			{
				dst_ptr[i] = static_cast<f32>(src_ptr[i]);
			}
		}

		const u8* src_aligned = src_ptr + i;
		f32* dst_aligned = dst_ptr + i;
		const usize main_loop_count = (count - i) / simd_stride;
		const u8* main_loop_end = src_aligned + main_loop_count * simd_stride;

		for (; src_aligned < main_loop_end; src_aligned += simd_stride, dst_aligned += simd_stride)
		{
			_mm_prefetch(reinterpret_cast<const char*>(src_aligned) + prefetch_ahead, _MM_HINT_T0);

			const __m128i block0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src_aligned));
			const __m128i block1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src_aligned + 16));
			const __m128i block2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src_aligned + 32));
			const __m128i block3 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src_aligned + 48));

			_mm256_stream_ps(dst_aligned + 0, _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(block0)));
			_mm256_stream_ps(dst_aligned + 8, _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128(block0, 8))));

			_mm256_stream_ps(dst_aligned + 16, _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(block1)));
			_mm256_stream_ps(dst_aligned + 24, _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128(block1, 8))));

			_mm256_stream_ps(dst_aligned + 32, _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(block2)));
			_mm256_stream_ps(dst_aligned + 40, _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128(block2, 8))));

			_mm256_stream_ps(dst_aligned + 48, _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(block3)));
			_mm256_stream_ps(dst_aligned + 56, _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128(block3, 8))));
		}

		usize processed = main_loop_count * simd_stride;
		usize remaining = count - i - processed;

		if (remaining >= 8)
		{
			const __m128i chunk = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(src_aligned));
			const __m256i ints = _mm256_cvtepu8_epi32(chunk);
			_mm256_storeu_ps(dst_aligned, _mm256_cvtepi32_ps(ints));
			src_aligned += 8;
			dst_aligned += 8;
			remaining -= 8;
		}

		if (remaining > 0)
		{
			while (remaining >= 4)
			{
				dst_aligned[0] = src_aligned[0];
				dst_aligned[1] = src_aligned[1];
				dst_aligned[2] = src_aligned[2];
				dst_aligned[3] = src_aligned[3];
				src_aligned += 4;
				dst_aligned += 4;
				remaining -= 4;
			}

			while (remaining-- > 0)
			{
				*dst_aligned++ = *src_aligned++;
			}
		}

		_mm_sfence();
	}

	template<> void cast_towardszero<u8, f64>(const u8* src_ptr, f64* dst_ptr, usize count) noexcept
	{
		constexpr usize cache_line_size = 64;
		constexpr usize simd_stride = 64;
		constexpr usize output_stride = 64;
		constexpr usize prefetch_ahead = 4 * cache_line_size;

		const uintptr_t base_addr = reinterpret_cast<uintptr_t>(dst_ptr);
		const usize misalignment = (base_addr % 32) / sizeof(f64);
		usize i = 0;

		if (misalignment > 0)
		{
			const usize align_count = std::min(32 / sizeof(f64) - misalignment, count);

			for (; i + 4 <= align_count; i += 4)
			{
				const __m128i chunk = _mm_cvtsi32_si128(*reinterpret_cast<const u32*>(src_ptr + i));
				const __m256i ints = _mm256_cvtepu8_epi32(chunk);
				const __m128i ints_lo = _mm256_castsi256_si128(ints);
				const __m256d doubles = _mm256_cvtepi32_pd(ints_lo);
				_mm256_storeu_pd(dst_ptr + i, doubles);
			}

			for (; i < align_count; ++i)
			{
				dst_ptr[i] = static_cast<f64>(src_ptr[i]);
			}
		}

		const u8* src_aligned = src_ptr + i;
		f64* dst_aligned = dst_ptr + i;
		const usize main_loop_count = (count - i) / simd_stride;
		const u8* main_loop_end = src_aligned + main_loop_count * simd_stride;

		for (; src_aligned < main_loop_end; src_aligned += simd_stride, dst_aligned += output_stride)
		{
			_mm_prefetch(reinterpret_cast<const char*>(src_aligned) + prefetch_ahead, _MM_HINT_T0);

			for (usize pd = 0; pd < 4; ++pd)
			{
				const __m128i block = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src_aligned + (pd * 16)));

				__m128i chunk0 = _mm_cvtsi32_si128(_mm_extract_epi32(block, 0));
				__m256i ints0 = _mm256_cvtepu8_epi32(chunk0);
				_mm256_stream_pd((dst_aligned + (pd * 16)) + 0, _mm256_cvtepi32_pd(_mm256_castsi256_si128(ints0)));

				__m128i chunk1 = _mm_cvtsi32_si128(_mm_extract_epi32(block, 1));
				__m256i ints1 = _mm256_cvtepu8_epi32(chunk1);
				_mm256_stream_pd((dst_aligned + (pd * 16)) + 4, _mm256_cvtepi32_pd(_mm256_castsi256_si128(ints1)));

				__m128i chunk2 = _mm_cvtsi32_si128(_mm_extract_epi32(block, 2));
				__m256i ints2 = _mm256_cvtepu8_epi32(chunk2);
				_mm256_stream_pd((dst_aligned + (pd * 16)) + 8, _mm256_cvtepi32_pd(_mm256_castsi256_si128(ints2)));

				__m128i chunk3 = _mm_cvtsi32_si128(_mm_extract_epi32(block, 3));
				__m256i ints3 = _mm256_cvtepu8_epi32(chunk3);
				_mm256_stream_pd((dst_aligned + (pd * 16)) + 12, _mm256_cvtepi32_pd(_mm256_castsi256_si128(ints3)));
			}
		}

		usize processed = main_loop_count * simd_stride;
		usize remaining = count - i - processed;

		if (remaining >= 8)
		{
			const __m128i chunk = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(src_aligned));
			const __m256i ints = _mm256_cvtepu8_epi32(chunk);

			const __m256d dbl_lo = _mm256_cvtepi32_pd(_mm256_castsi256_si128(ints));
			const __m256d dbl_hi = _mm256_cvtepi32_pd(_mm256_extracti128_si256(ints, 1));

			_mm256_storeu_pd(dst_aligned, dbl_lo);
			_mm256_storeu_pd(dst_aligned + 4, dbl_hi);

			src_aligned += 8;
			dst_aligned += 8;
			remaining -= 8;
		}

		for (; remaining > 0; --remaining)
		{
			*dst_aligned++ = static_cast<f64>(*src_aligned++);
		}

		_mm_sfence();
	}

	template<> void cast_towardszero<u16, i8>(const u16* src_ptr, i8* dst_ptr, usize count) noexcept
	{
		constexpr usize cache_line_size = 64;
		constexpr usize simd_stride = 64;
		constexpr usize prefetch_ahead = 4 * cache_line_size;

		usize i = 0;

		while (i < count && (reinterpret_cast<uintptr_t>(dst_ptr + i) % 16) != 0)
		{
			dst_ptr[i] = static_cast<i8>(src_ptr[i]);
			i++;
		}

		const u16* src_aligned = src_ptr + i;
		i8* dst_aligned = dst_ptr + i;
		const usize main_loop_count = (count - i) / simd_stride;
		const u16* main_loop_end = src_aligned + main_loop_count * simd_stride;

		const __m128i mask_lo8 = _mm_set1_epi16(0x00FF);

		for (; src_aligned < main_loop_end; src_aligned += simd_stride, dst_aligned += simd_stride)
		{
			_mm_prefetch(reinterpret_cast<const char*>(src_aligned) + prefetch_ahead, _MM_HINT_T0);

			__m128i block0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src_aligned + 0));
			__m128i block1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src_aligned + 8));
			__m128i block2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src_aligned + 16));
			__m128i block3 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src_aligned + 24));
			__m128i block4 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src_aligned + 32));
			__m128i block5 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src_aligned + 40));
			__m128i block6 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src_aligned + 48));
			__m128i block7 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src_aligned + 56));

			block0 = _mm_and_si128(block0, mask_lo8);
			block1 = _mm_and_si128(block1, mask_lo8);
			block2 = _mm_and_si128(block2, mask_lo8);
			block3 = _mm_and_si128(block3, mask_lo8);
			block4 = _mm_and_si128(block4, mask_lo8);
			block5 = _mm_and_si128(block5, mask_lo8);
			block6 = _mm_and_si128(block6, mask_lo8);
			block7 = _mm_and_si128(block7, mask_lo8);
			
			__m128i packed0 = _mm_packus_epi16(block0, block1);
			__m128i packed1 = _mm_packus_epi16(block2, block3);
			__m128i packed2 = _mm_packus_epi16(block4, block5);
			__m128i packed3 = _mm_packus_epi16(block6, block7);

			_mm_stream_si128(reinterpret_cast<__m128i*>(dst_aligned + 0), packed0);
			_mm_stream_si128(reinterpret_cast<__m128i*>(dst_aligned + 16), packed1);
			_mm_stream_si128(reinterpret_cast<__m128i*>(dst_aligned + 32), packed2);
			_mm_stream_si128(reinterpret_cast<__m128i*>(dst_aligned + 48), packed3);
		}

		usize processed = main_loop_count * simd_stride;
		usize remaining = count - i - processed;
		const u16* src_remaining = src_aligned;
		i8* dst_remaining = dst_aligned;

		while (remaining >= 16)
		{
			__m128i block0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src_remaining + 0));
			__m128i block1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src_remaining + 8));
			block0 = _mm_and_si128(block0, mask_lo8);
			block1 = _mm_and_si128(block1, mask_lo8);
			__m128i packed = _mm_packus_epi16(block0, block1);
			_mm_storeu_si128(reinterpret_cast<__m128i*>(dst_remaining), packed);
			src_remaining += 16;
			dst_remaining += 16;
			remaining -= 16;
		}

		while (remaining-- > 0)
		{
			*dst_remaining++ = static_cast<i8>(*src_remaining++);
		}

		_mm_sfence();
	}

	template<> void cast_towardszero<u16, i16>(const u16* src_ptr, i16* dst_ptr, usize count) noexcept
	{
		constexpr usize cache_line_size = 64;
		constexpr usize simd_stride = 64;
		constexpr usize prefetch_ahead = 4 * cache_line_size;

		usize i = 0;

		while (i < count && (reinterpret_cast<uintptr_t>(dst_ptr + i) % 32) != 0)
		{
			dst_ptr[i] = static_cast<i16>(src_ptr[i]);
			i++;
		}

		const u16* src_aligned = src_ptr + i;
		i16* dst_aligned = dst_ptr + i;
		const usize main_loop_count = (count - i) / simd_stride;
		const u16* main_loop_end = src_aligned + main_loop_count * simd_stride;

		for (; src_aligned < main_loop_end; src_aligned += simd_stride, dst_aligned += simd_stride)
		{
			_mm_prefetch(reinterpret_cast<const char*>(src_aligned) + prefetch_ahead, _MM_HINT_T0);

			__m256i block0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src_aligned + 0));
			__m256i block1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src_aligned + 16));
			__m256i block2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src_aligned + 32));
			__m256i block3 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src_aligned + 48));

			_mm256_stream_si256(reinterpret_cast<__m256i*>(dst_aligned + 0), block0);
			_mm256_stream_si256(reinterpret_cast<__m256i*>(dst_aligned + 16), block1);
			_mm256_stream_si256(reinterpret_cast<__m256i*>(dst_aligned + 32), block2);
			_mm256_stream_si256(reinterpret_cast<__m256i*>(dst_aligned + 48), block3);
		}

		usize processed = main_loop_count * simd_stride;
		usize remaining = count - i - processed;
		const u16* src_remaining = src_aligned;
		i16* dst_remaining = dst_aligned;

		while (remaining >= 16)
		{
			__m256i block = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src_remaining));
			_mm256_storeu_si256(reinterpret_cast<__m256i*>(dst_remaining), block);
			src_remaining += 16;
			dst_remaining += 16;
			remaining -= 16;
		}

		while (remaining >= 8)
		{
			__m128i block = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src_remaining));
			_mm_storeu_si128(reinterpret_cast<__m128i*>(dst_remaining), block);
			src_remaining += 8;
			dst_remaining += 8;
			remaining -= 8;
		}

		while (remaining-- > 0)
		{
			*dst_remaining++ = static_cast<i16>(*src_remaining++);
		}

		_mm_sfence();
	}

	template<> void cast_towardszero<u16, i32>(const u16* src_ptr, i32* dst_ptr, usize count) noexcept
	{
		constexpr usize elements_per_vector = 16;
		constexpr usize simd_stride = elements_per_vector * 2;
		constexpr usize cache_line_size = 64;
		constexpr usize prefetch_ahead = 4 * cache_line_size;

		usize i = 0;

		while (i < count && (reinterpret_cast<uintptr_t>(dst_ptr + i) % 32) != 0)
		{
			dst_ptr[i] = static_cast<i32>(src_ptr[i]);
			i++;
		}

		const u16* src_aligned = src_ptr + i;
		i32* dst_aligned = dst_ptr + i;
		const usize main_loop_count = (count - i) / simd_stride;
		const u16* main_loop_end = src_aligned + main_loop_count * simd_stride;

		for (; src_aligned < main_loop_end; src_aligned += simd_stride, dst_aligned += simd_stride)
		{
			_mm_prefetch(reinterpret_cast<const char*>(src_aligned) + prefetch_ahead, _MM_HINT_T0);

			__m256i src0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src_aligned));
			__m256i src1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src_aligned + 16));

			__m256i dst0 = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(src0));
			__m256i dst1 = _mm256_cvtepu16_epi32(_mm256_extracti128_si256(src0, 1));
			__m256i dst2 = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(src1));
			__m256i dst3 = _mm256_cvtepu16_epi32(_mm256_extracti128_si256(src1, 1));

			_mm256_stream_si256(reinterpret_cast<__m256i*>(dst_aligned + 0), dst0);
			_mm256_stream_si256(reinterpret_cast<__m256i*>(dst_aligned + 8), dst1);
			_mm256_stream_si256(reinterpret_cast<__m256i*>(dst_aligned + 16), dst2);
			_mm256_stream_si256(reinterpret_cast<__m256i*>(dst_aligned + 24), dst3);
		}

		usize processed = main_loop_count * simd_stride;
		usize remaining = count - i - processed;
		const u16* src_remaining = src_aligned;
		i32* dst_remaining = dst_aligned;

		while (remaining >= 16)
		{
			__m256i src = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src_remaining));
			__m256i dst0 = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(src));
			__m256i dst1 = _mm256_cvtepu16_epi32(_mm256_extracti128_si256(src, 1));
			_mm256_storeu_si256(reinterpret_cast<__m256i*>(dst_remaining), dst0);
			_mm256_storeu_si256(reinterpret_cast<__m256i*>(dst_remaining + 8), dst1);
			src_remaining += 16;
			dst_remaining += 16;
			remaining -= 16;
		}

		while (remaining >= 8)
		{
			__m128i src = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src_remaining));
			__m256i dst = _mm256_cvtepu16_epi32(src);
			_mm256_storeu_si256(reinterpret_cast<__m256i*>(dst_remaining), dst);
			src_remaining += 8;
			dst_remaining += 8;
			remaining -= 8;
		}

		while (remaining-- > 0)
		{
			*dst_remaining++ = static_cast<i32>(*src_remaining++);
		}

		_mm_sfence();
	}

	template<> void cast_towardszero<u16, i64>(const u16* src_ptr, i64* dst_ptr, usize count) noexcept
	{
		constexpr usize elements_per_vector = 4;
		constexpr usize simd_stride = 4 * elements_per_vector;
		constexpr usize cache_line_size = 64;
		constexpr usize prefetch_ahead = 4 * cache_line_size;

		usize i = 0;

		while (i < count && (reinterpret_cast<uintptr_t>(dst_ptr + i) % 32) != 0)
		{
			dst_ptr[i] = static_cast<i64>(src_ptr[i]);
			i++;
		}

		const u16* src_aligned = src_ptr + i;
		i64* dst_aligned = dst_ptr + i;
		const usize main_loop_count = (count - i) / simd_stride;
		const u16* main_loop_end = src_aligned + main_loop_count * simd_stride;

		for (; src_aligned < main_loop_end; src_aligned += simd_stride, dst_aligned += simd_stride)
		{
			_mm_prefetch(reinterpret_cast<const char*>(src_aligned) + prefetch_ahead, _MM_HINT_T0);

			__m256i src = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src_aligned));

			__m256i lo16 = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(src));
			__m256i hi16 = _mm256_cvtepu16_epi32(_mm256_extracti128_si256(src, 1));

			__m256i dst0 = _mm256_cvtepu32_epi64(_mm256_castsi256_si128(lo16));
			__m256i dst1 = _mm256_cvtepu32_epi64(_mm256_extracti128_si256(lo16, 1));
			__m256i dst2 = _mm256_cvtepu32_epi64(_mm256_castsi256_si128(hi16));
			__m256i dst3 = _mm256_cvtepu32_epi64(_mm256_extracti128_si256(hi16, 1));

			_mm256_stream_si256(reinterpret_cast<__m256i*>(dst_aligned + 0), dst0);
			_mm256_stream_si256(reinterpret_cast<__m256i*>(dst_aligned + 4), dst1);
			_mm256_stream_si256(reinterpret_cast<__m256i*>(dst_aligned + 8), dst2);
			_mm256_stream_si256(reinterpret_cast<__m256i*>(dst_aligned + 12), dst3);
		}

		usize processed = main_loop_count * simd_stride;
		usize remaining = count - i - processed;
		const u16* src_remaining = src_aligned;
		i64* dst_remaining = dst_aligned;

		while (remaining >= 4)
		{
			__m128i src = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src_remaining));
			__m256i tmp = _mm256_cvtepu16_epi32(src);
			__m256i dst0 = _mm256_cvtepu32_epi64(_mm256_castsi256_si128(tmp));
			__m256i dst1 = _mm256_cvtepu32_epi64(_mm256_extracti128_si256(tmp, 1));
			_mm256_storeu_si256(reinterpret_cast<__m256i*>(dst_remaining), dst0);
			_mm256_storeu_si256(reinterpret_cast<__m256i*>(dst_remaining + 4), dst1);
			src_remaining += 4;
			dst_remaining += 4;
			remaining -= 4;
		}

		while (remaining-- > 0)
		{
			*dst_remaining++ = static_cast<i64>(*src_remaining++);
		}

		_mm_sfence();
	}

	template<> void cast_towardszero<u16, u8>(const u16* src_ptr, u8* dst_ptr, usize count) noexcept
	{
		constexpr usize cache_line_size = 64;
		constexpr usize simd_stride = 64;
		constexpr usize prefetch_ahead = 4 * cache_line_size;

		usize i = 0;

		while (i < count && (reinterpret_cast<uintptr_t>(dst_ptr + i) % 16) != 0)
		{
			dst_ptr[i] = static_cast<u8>(src_ptr[i]);
			i++;
		}

		const u16* src_aligned = src_ptr + i;
		u8* dst_aligned = dst_ptr + i;
		const usize main_loop_count = (count - i) / simd_stride;
		const u16* main_loop_end = src_aligned + main_loop_count * simd_stride;

		const __m128i mask_lo8 = _mm_set1_epi16(0x00FF);

		for (; src_aligned < main_loop_end; src_aligned += simd_stride, dst_aligned += simd_stride)
		{
			_mm_prefetch(reinterpret_cast<const char*>(src_aligned) + prefetch_ahead, _MM_HINT_T0);

			__m128i block0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src_aligned + 0));
			__m128i block1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src_aligned + 8));
			__m128i block2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src_aligned + 16));
			__m128i block3 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src_aligned + 24));
			__m128i block4 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src_aligned + 32));
			__m128i block5 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src_aligned + 40));
			__m128i block6 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src_aligned + 48));
			__m128i block7 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src_aligned + 56));

			block0 = _mm_and_si128(block0, mask_lo8);
			block1 = _mm_and_si128(block1, mask_lo8);
			block2 = _mm_and_si128(block2, mask_lo8);
			block3 = _mm_and_si128(block3, mask_lo8);
			block4 = _mm_and_si128(block4, mask_lo8);
			block5 = _mm_and_si128(block5, mask_lo8);
			block6 = _mm_and_si128(block6, mask_lo8);
			block7 = _mm_and_si128(block7, mask_lo8);

			__m128i packed0 = _mm_packus_epi16(block0, block1);
			__m128i packed1 = _mm_packus_epi16(block2, block3);
			__m128i packed2 = _mm_packus_epi16(block4, block5);
			__m128i packed3 = _mm_packus_epi16(block6, block7);

			_mm_stream_si128(reinterpret_cast<__m128i*>(dst_aligned + 0), packed0);
			_mm_stream_si128(reinterpret_cast<__m128i*>(dst_aligned + 16), packed1);
			_mm_stream_si128(reinterpret_cast<__m128i*>(dst_aligned + 32), packed2);
			_mm_stream_si128(reinterpret_cast<__m128i*>(dst_aligned + 48), packed3);
		}

		usize processed = main_loop_count * simd_stride;
		usize remaining = count - i - processed;
		const u16* src_remaining = src_aligned;
		u8* dst_remaining = dst_aligned;

		while (remaining >= 16)
		{
			__m128i block0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src_remaining + 0));
			__m128i block1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src_remaining + 8));
			block0 = _mm_and_si128(block0, mask_lo8);
			block1 = _mm_and_si128(block1, mask_lo8);
			__m128i packed = _mm_packus_epi16(block0, block1);
			_mm_storeu_si128(reinterpret_cast<__m128i*>(dst_remaining), packed);
			src_remaining += 16;
			dst_remaining += 16;
			remaining -= 16;
		}

		while (remaining-- > 0)
		{
			*dst_remaining++ = static_cast<u8>(*src_remaining++);
		}

		_mm_sfence();
	}

	template<> void cast_towardszero<u16, u16>(const u16* src_ptr, u16* dst_ptr, usize count) noexcept
	{
		std::memcpy(dst_ptr, src_ptr, count * sizeof(u16));
	}

	template<> void cast_towardszero<u16, u32>(const u16* src_ptr, u32* dst_ptr, usize count) noexcept
	{
		constexpr usize elements_per_vector = 16;
		constexpr usize simd_stride = elements_per_vector * 2;
		constexpr usize cache_line_size = 64;
		constexpr usize prefetch_ahead = 4 * cache_line_size;

		usize i = 0;

		while (i < count && (reinterpret_cast<uintptr_t>(dst_ptr + i) % 32) != 0)
		{
			dst_ptr[i] = static_cast<u32>(src_ptr[i]);
			i++;
		}

		const u16* src_aligned = src_ptr + i;
		u32* dst_aligned = dst_ptr + i;
		const usize main_loop_count = (count - i) / simd_stride;
		const u16* main_loop_end = src_aligned + main_loop_count * simd_stride;

		for (; src_aligned < main_loop_end; src_aligned += simd_stride, dst_aligned += simd_stride)
		{
			_mm_prefetch(reinterpret_cast<const char*>(src_aligned) + prefetch_ahead, _MM_HINT_T0);

			__m256i src0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src_aligned));
			__m256i src1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src_aligned + 16));

			__m256i dst0 = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(src0));
			__m256i dst1 = _mm256_cvtepu16_epi32(_mm256_extracti128_si256(src0, 1));
			__m256i dst2 = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(src1));
			__m256i dst3 = _mm256_cvtepu16_epi32(_mm256_extracti128_si256(src1, 1));

			_mm256_stream_si256(reinterpret_cast<__m256i*>(dst_aligned + 0), dst0);
			_mm256_stream_si256(reinterpret_cast<__m256i*>(dst_aligned + 8), dst1);
			_mm256_stream_si256(reinterpret_cast<__m256i*>(dst_aligned + 16), dst2);
			_mm256_stream_si256(reinterpret_cast<__m256i*>(dst_aligned + 24), dst3);
		}

		usize processed = main_loop_count * simd_stride;
		usize remaining = count - i - processed;
		const u16* src_remaining = src_aligned;
		u32* dst_remaining = dst_aligned;

		while (remaining >= 16)
		{
			__m256i src = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src_remaining));
			__m256i dst0 = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(src));
			__m256i dst1 = _mm256_cvtepu16_epi32(_mm256_extracti128_si256(src, 1));
			_mm256_storeu_si256(reinterpret_cast<__m256i*>(dst_remaining), dst0);
			_mm256_storeu_si256(reinterpret_cast<__m256i*>(dst_remaining + 8), dst1);
			src_remaining += 16;
			dst_remaining += 16;
			remaining -= 16;
		}

		while (remaining >= 8)
		{
			__m128i src = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src_remaining));
			__m256i dst = _mm256_cvtepu16_epi32(src);
			_mm256_storeu_si256(reinterpret_cast<__m256i*>(dst_remaining), dst);
			src_remaining += 8;
			dst_remaining += 8;
			remaining -= 8;
		}

		while (remaining-- > 0)
		{
			*dst_remaining++ = static_cast<u32>(*src_remaining++);
		}

		_mm_sfence();
	}

	template<> void cast_towardszero<u16, u64>(const u16* src_ptr, u64* dst_ptr, usize count) noexcept
	{
		constexpr usize elements_per_vector = 16;
		constexpr usize simd_stride = elements_per_vector;
		constexpr usize cache_line_size = 64;
		constexpr usize prefetch_ahead = 4 * cache_line_size;

		usize i = 0;

		while (i < count && (reinterpret_cast<uintptr_t>(dst_ptr + i) % 32) != 0)
		{
			dst_ptr[i] = static_cast<u64>(src_ptr[i]);
			i++;
		}

		const u16* src_aligned = src_ptr + i;
		u64* dst_aligned = dst_ptr + i;
		const usize main_loop_count = (count - i) / simd_stride;
		const u16* main_loop_end = src_aligned + main_loop_count * simd_stride;

		for (; src_aligned < main_loop_end; src_aligned += simd_stride, dst_aligned += simd_stride)
		{
			_mm_prefetch(reinterpret_cast<const char*>(src_aligned) + prefetch_ahead, _MM_HINT_T0);

			__m256i src = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src_aligned));

			__m256i lo32 = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(src));
			__m256i hi32 = _mm256_cvtepu16_epi32(_mm256_extracti128_si256(src, 1));

			__m256i dst0 = _mm256_cvtepu32_epi64(_mm256_castsi256_si128(lo32));
			__m256i dst1 = _mm256_cvtepu32_epi64(_mm256_extracti128_si256(lo32, 1));
			__m256i dst2 = _mm256_cvtepu32_epi64(_mm256_castsi256_si128(hi32));
			__m256i dst3 = _mm256_cvtepu32_epi64(_mm256_extracti128_si256(hi32, 1));

			_mm256_stream_si256(reinterpret_cast<__m256i*>(dst_aligned + 0), dst0);
			_mm256_stream_si256(reinterpret_cast<__m256i*>(dst_aligned + 4), dst1);
			_mm256_stream_si256(reinterpret_cast<__m256i*>(dst_aligned + 8), dst2);
			_mm256_stream_si256(reinterpret_cast<__m256i*>(dst_aligned + 12), dst3);
		}

		usize processed = main_loop_count * simd_stride;
		usize remaining = count - i - processed;
		const u16* src_remaining = src_aligned;
		u64* dst_remaining = dst_aligned;

		while (remaining >= 4)
		{
			__m128i src = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(src_remaining));
			__m128i u32_vec = _mm_cvtepu16_epi32(src);
			__m256i u64_vec = _mm256_cvtepu32_epi64(u32_vec);
			_mm256_storeu_si256(reinterpret_cast<__m256i*>(dst_remaining), u64_vec);
			src_remaining += 4;
			dst_remaining += 4;
			remaining -= 4;
		}
		
		while (remaining-- > 0)
		{
			*dst_remaining++ = static_cast<u64>(*src_remaining++);
		}

		_mm_sfence();
	}

	template<> void cast_towardszero<u16, f16>(const u16* src_ptr, f16* dst_ptr, usize count) noexcept
	{
		constexpr usize cache_line_size = 64;
		constexpr usize simd_stride = 32;
		constexpr usize prefetch_ahead = 4 * cache_line_size;

		usize i = 0;

		while (i < count && (reinterpret_cast<uintptr_t>(dst_ptr + i) % 32) != 0)
		{
			dst_ptr[i] = static_cast<f16>(static_cast<f32>(src_ptr[i]));
			i++;
		}

		const u16* src_aligned = src_ptr + i;
		f16* dst_aligned = dst_ptr + i;
		const usize main_loop_count = (count - i) / simd_stride;
		const u16* main_loop_end = src_aligned + main_loop_count * simd_stride;

		for (; src_aligned < main_loop_end; src_aligned += simd_stride, dst_aligned += simd_stride)
		{
			_mm_prefetch(reinterpret_cast<const char*>(src_aligned) + prefetch_ahead, _MM_HINT_T0);

			__m256i src0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src_aligned));
			__m256i src1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src_aligned + 16));

			__m256i u32_0 = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(src0));
			__m256i u32_1 = _mm256_cvtepu16_epi32(_mm256_extracti128_si256(src0, 1));
			__m256i u32_2 = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(src1));
			__m256i u32_3 = _mm256_cvtepu16_epi32(_mm256_extracti128_si256(src1, 1));

			__m256 f32_0 = _mm256_cvtepi32_ps(u32_0);
			__m256 f32_1 = _mm256_cvtepi32_ps(u32_1);
			__m256 f32_2 = _mm256_cvtepi32_ps(u32_2);
			__m256 f32_3 = _mm256_cvtepi32_ps(u32_3);

			__m128i h0 = _mm256_cvtps_ph(f32_0, _MM_FROUND_TO_NEAREST_INT);
			__m128i h1 = _mm256_cvtps_ph(f32_1, _MM_FROUND_TO_NEAREST_INT);
			__m128i h2 = _mm256_cvtps_ph(f32_2, _MM_FROUND_TO_NEAREST_INT);
			__m128i h3 = _mm256_cvtps_ph(f32_3, _MM_FROUND_TO_NEAREST_INT);

			_mm256_stream_si256(reinterpret_cast<__m256i*>(dst_aligned), _mm256_set_m128i(h1, h0));
			_mm256_stream_si256(reinterpret_cast<__m256i*>(dst_aligned + 16), _mm256_set_m128i(h3, h2));
		}

		usize processed = main_loop_count * simd_stride;
		usize remaining = count - i - processed;
		const u16* src_remaining = src_aligned;
		f16* dst_remaining = dst_aligned;

		while (remaining >= 16)
		{
			__m256i src = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src_remaining));

			__m256i u32_lo = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(src));
			__m256i u32_hi = _mm256_cvtepu16_epi32(_mm256_extracti128_si256(src, 1));

			__m256 f32_lo = _mm256_cvtepi32_ps(u32_lo);
			__m256 f32_hi = _mm256_cvtepi32_ps(u32_hi);

			__m128i h_lo = _mm256_cvtps_ph(f32_lo, _MM_FROUND_TO_NEAREST_INT);
			__m128i h_hi = _mm256_cvtps_ph(f32_hi, _MM_FROUND_TO_NEAREST_INT);

			_mm_storeu_si128(reinterpret_cast<__m128i*>(dst_remaining), h_lo);
			_mm_storeu_si128(reinterpret_cast<__m128i*>(dst_remaining + 8), h_hi);

			src_remaining += 16;
			dst_remaining += 16;
			remaining -= 16;
		}

		if (remaining >= 8)
		{
			__m128i src = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src_remaining));
			__m256i u32_vec = _mm256_cvtepu16_epi32(src);
			__m256 f32_vec = _mm256_cvtepi32_ps(u32_vec);
			__m128i h = _mm256_cvtps_ph(f32_vec, _MM_FROUND_TO_NEAREST_INT);
			_mm_storeu_si128(reinterpret_cast<__m128i*>(dst_remaining), h);

			src_remaining += 8;
			dst_remaining += 8;
			remaining -= 8;
		}

		if (remaining >= 4)
		{
			__m128i src = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(src_remaining));
			__m128i u32_vec = _mm_cvtepu16_epi32(src);
			__m128 f32_vec = _mm_cvtepi32_ps(u32_vec);
			__m128i h = _mm_cvtps_ph(f32_vec, _MM_FROUND_TO_NEAREST_INT);
			_mm_storel_epi64(reinterpret_cast<__m128i*>(dst_remaining), h);

			src_remaining += 4;
			dst_remaining += 4;
			remaining -= 4;
		}

		for (; remaining > 0; --remaining)
		{
			*dst_remaining++ = static_cast<f16>(static_cast<f32>(*src_remaining++));
		}

		_mm_sfence();
	}

	template<> void cast_towardszero<u16, f32>(const u16* src_ptr, f32* dst_ptr, usize count) noexcept
	{
		constexpr usize cache_line_size = 64;
		constexpr usize simd_stride = 32;
		constexpr usize prefetch_ahead = 4 * cache_line_size;

		usize i = 0;

		while (i < count && (reinterpret_cast<uintptr_t>(dst_ptr + i) % 32) != 0)
		{
			dst_ptr[i] = static_cast<f32>(src_ptr[i]);
			i++;
		}

		const u16* src_aligned = src_ptr + i;
		f32* dst_aligned = dst_ptr + i;
		const usize main_loop_count = (count - i) / simd_stride;
		const u16* main_loop_end = src_aligned + main_loop_count * simd_stride;

		for (; src_aligned < main_loop_end; src_aligned += simd_stride, dst_aligned += simd_stride)
		{
			_mm_prefetch(reinterpret_cast<const char*>(src_aligned) + prefetch_ahead, _MM_HINT_T0);

			__m256i src0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src_aligned + 0));
			__m256i src1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src_aligned + 16));

			__m256i u32_0 = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(src0));
			__m256i u32_1 = _mm256_cvtepu16_epi32(_mm256_extracti128_si256(src0, 1));
			__m256i u32_2 = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(src1));
			__m256i u32_3 = _mm256_cvtepu16_epi32(_mm256_extracti128_si256(src1, 1));

			__m256 f32_0 = _mm256_cvtepi32_ps(u32_0);
			__m256 f32_1 = _mm256_cvtepi32_ps(u32_1);
			__m256 f32_2 = _mm256_cvtepi32_ps(u32_2);
			__m256 f32_3 = _mm256_cvtepi32_ps(u32_3);

			_mm256_stream_ps(dst_aligned + 0, f32_0);
			_mm256_stream_ps(dst_aligned + 8, f32_1);
			_mm256_stream_ps(dst_aligned + 16, f32_2);
			_mm256_stream_ps(dst_aligned + 24, f32_3);
		}

		usize processed = main_loop_count * simd_stride;
		usize remaining = count - i - processed;
		const u16* src_remaining = src_aligned;
		f32* dst_remaining = dst_aligned;

		while (remaining >= 16)
		{
			__m256i src = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src_remaining));
			__m256i lo = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(src));
			__m256i hi = _mm256_cvtepu16_epi32(_mm256_extracti128_si256(src, 1));
			__m256 f32_lo = _mm256_cvtepi32_ps(lo);
			__m256 f32_hi = _mm256_cvtepi32_ps(hi);
			_mm256_storeu_ps(dst_remaining, f32_lo);
			_mm256_storeu_ps(dst_remaining + 8, f32_hi);
			src_remaining += 16;
			dst_remaining += 16;
			remaining -= 16;
		}

		if (remaining >= 8)
		{
			__m128i src = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src_remaining));
			__m256i u32_vec = _mm256_cvtepu16_epi32(src);
			__m256 f32_vec = _mm256_cvtepi32_ps(u32_vec);
			_mm256_storeu_ps(dst_remaining, f32_vec);
			src_remaining += 8;
			dst_remaining += 8;
			remaining -= 8;
		}

		if (remaining >= 4)
		{
			__m128i src = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(src_remaining));
			__m128i u32_vec = _mm_cvtepu16_epi32(src);
			__m128 f32_vec = _mm_cvtepi32_ps(u32_vec);
			_mm_storeu_ps(dst_remaining, f32_vec);
			src_remaining += 4;
			dst_remaining += 4;
			remaining -= 4;
		}

		while (remaining-- > 0)
		{
			*dst_remaining++ = static_cast<f32>(*src_remaining++);
		}

		_mm_sfence();
	}

	template<> void cast_towardszero<u16, f64>(const u16* src_ptr, f64* dst_ptr, usize count) noexcept
	{
		constexpr usize cache_line_size = 64;
		constexpr usize simd_stride = 8;
		constexpr usize prefetch_ahead = 4 * cache_line_size;

		usize i = 0;

		while (i < count && (reinterpret_cast<uintptr_t>(dst_ptr + i) % 32) != 0)
		{
			dst_ptr[i] = static_cast<f64>(src_ptr[i]);
			i++;
		}

		const u16* src_aligned = src_ptr + i;
		f64* dst_aligned = dst_ptr + i;
		const usize main_loop_count = (count - i) / simd_stride;
		const u16* main_loop_end = src_aligned + main_loop_count * simd_stride;

		for (; src_aligned < main_loop_end; src_aligned += simd_stride, dst_aligned += simd_stride)
		{
			_mm_prefetch(reinterpret_cast<const char*>(src_aligned) + prefetch_ahead, _MM_HINT_T0);

			__m128i src_u16 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src_aligned));

			__m256i src_u32 = _mm256_cvtepu16_epi32(src_u16);

			__m256 src_f32 = _mm256_cvtepi32_ps(src_u32);

			__m256d dst_f64_lo = _mm256_cvtps_pd(_mm256_castps256_ps128(src_f32));
			__m256d dst_f64_hi = _mm256_cvtps_pd(_mm256_extractf128_ps(src_f32, 1));

			_mm256_stream_pd(dst_aligned, dst_f64_lo);
			_mm256_stream_pd(dst_aligned + 4, dst_f64_hi);
		}

		usize processed = main_loop_count * simd_stride;
		usize remaining = count - i - processed;
		const u16* src_remaining = src_aligned;
		f64* dst_remaining = dst_aligned;

		if (remaining >= 4)
		{
			__m128i src_u16 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(src_remaining));

			__m128i src_u32 = _mm_cvtepu16_epi32(src_u16);

			__m128 src_f32 = _mm_cvtepi32_ps(src_u32);

			__m256d dst_f64 = _mm256_cvtps_pd(src_f32);

			_mm256_storeu_pd(dst_remaining, dst_f64);

			src_remaining += 4;
			dst_remaining += 4;
			remaining -= 4;
		}

		while (remaining-- > 0)
		{
			*dst_remaining++ = static_cast<f64>(*src_remaining++);
		}

		_mm_sfence();
	}

	template<> void cast_towardszero<u32, i8>(const u32* src_ptr, i8* dst_ptr, usize count) noexcept
	{
		constexpr usize cache_line_size = 64;
		constexpr usize simd_stride = 64;
		constexpr usize prefetch_ahead = 4 * cache_line_size;

		usize i = 0;

		while (i < count && (reinterpret_cast<uintptr_t>(dst_ptr + i) % 16) != 0)
		{
			dst_ptr[i] = static_cast<i8>(src_ptr[i]);
			i++;
		}

		const u32* src_aligned = src_ptr + i;
		i8* dst_aligned = dst_ptr + i;
		const usize main_loop_count = (count - i) / simd_stride;
		const u32* main_loop_end = src_aligned + main_loop_count * simd_stride;

		const __m128i shuffle_mask = _mm_set_epi8(
			0x80, 0x80, 0x80, 0x80,
			0x80, 0x80, 0x80, 0x80,
			0x80, 0x80, 0x80, 0x80,
			12, 8, 4, 0
		);

		for (; src_aligned < main_loop_end; src_aligned += simd_stride, dst_aligned += simd_stride)
		{
			_mm_prefetch(reinterpret_cast<const char*>(src_aligned) + prefetch_ahead, _MM_HINT_T0);

			for (usize j = 0; j < simd_stride; j += 16)
			{
				const u32* src = src_aligned + j;
				i8* dst = dst_aligned + j;

				__m128i block0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + 0));
				__m128i block1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + 4));
				__m128i block2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + 8));
				__m128i block3 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + 12));

				__m128i bytes0 = _mm_shuffle_epi8(block0, shuffle_mask);
				__m128i bytes1 = _mm_shuffle_epi8(block1, shuffle_mask);
				__m128i bytes2 = _mm_shuffle_epi8(block2, shuffle_mask);
				__m128i bytes3 = _mm_shuffle_epi8(block3, shuffle_mask);

				bytes1 = _mm_slli_si128(bytes1, 4);
				bytes2 = _mm_slli_si128(bytes2, 8);
				bytes3 = _mm_slli_si128(bytes3, 12);
				__m128i merged = _mm_or_si128(_mm_or_si128(bytes0, bytes1), _mm_or_si128(bytes2, bytes3));

				_mm_stream_si128(reinterpret_cast<__m128i*>(dst), merged);
			}
		}

		usize processed = main_loop_count * simd_stride;
		usize remaining = count - i - processed;
		const u32* src_remaining = src_aligned;
		i8* dst_remaining = dst_aligned;

		while (remaining >= 16)
		{
			__m128i block0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src_remaining + 0));
			__m128i block1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src_remaining + 4));
			__m128i block2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src_remaining + 8));
			__m128i block3 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src_remaining + 12));

			__m128i bytes0 = _mm_shuffle_epi8(block0, shuffle_mask);
			__m128i bytes1 = _mm_shuffle_epi8(block1, shuffle_mask);
			__m128i bytes2 = _mm_shuffle_epi8(block2, shuffle_mask);
			__m128i bytes3 = _mm_shuffle_epi8(block3, shuffle_mask);

			bytes1 = _mm_slli_si128(bytes1, 4);
			bytes2 = _mm_slli_si128(bytes2, 8);
			bytes3 = _mm_slli_si128(bytes3, 12);
			__m128i merged = _mm_or_si128(_mm_or_si128(bytes0, bytes1), _mm_or_si128(bytes2, bytes3));

			_mm_storeu_si128(reinterpret_cast<__m128i*>(dst_remaining), merged);
			src_remaining += 16;
			dst_remaining += 16;
			remaining -= 16;
		}

		while (remaining-- > 0)
		{
			*dst_remaining++ = static_cast<i8>(*src_remaining++);
		}

		_mm_sfence();
	}

	template<> void cast_towardszero<u32, i16>(const u32* src_ptr, i16* dst_ptr, usize count) noexcept
	{
		constexpr usize cache_line_size = 64;
		constexpr usize simd_stride = 32;
		constexpr usize prefetch_ahead = 4 * cache_line_size;

		usize i = 0;

		while (i < count && (reinterpret_cast<uintptr_t>(dst_ptr + i) % 16) != 0)
		{
			dst_ptr[i] = static_cast<i16>(src_ptr[i]);
			i++;
		}

		const u32* src_aligned = src_ptr + i;
		i16* dst_aligned = dst_ptr + i;
		const usize main_loop_count = (count - i) / simd_stride;
		const u32* main_loop_end = src_aligned + main_loop_count * simd_stride;

		const __m128i mask32 = _mm_set1_epi32(0x0000FFFF);

		for (; src_aligned < main_loop_end; src_aligned += simd_stride, dst_aligned += simd_stride)
		{
			_mm_prefetch(reinterpret_cast<const char*>(src_aligned) + prefetch_ahead, _MM_HINT_T0);

			for (usize j = 0; j < simd_stride; j += 8)
			{
				const u32* src = src_aligned + j;
				i16* dst = dst_aligned + j;

				__m128i block0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + 0));
				__m128i block1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + 4));

				block0 = _mm_and_si128(block0, mask32);
				block1 = _mm_and_si128(block1, mask32);

				__m128i packed = _mm_packus_epi32(block0, block1);

				_mm_stream_si128(reinterpret_cast<__m128i*>(dst), packed);
			}
		}

		usize processed = main_loop_count * simd_stride;
		usize remaining = count - i - processed;
		const u32* src_remaining = src_aligned;
		i16* dst_remaining = dst_aligned;

		while (remaining >= 8)
		{
			__m128i block0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src_remaining + 0));
			__m128i block1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src_remaining + 4));

			block0 = _mm_and_si128(block0, mask32);
			block1 = _mm_and_si128(block1, mask32);

			__m128i packed = _mm_packus_epi32(block0, block1);
			_mm_storeu_si128(reinterpret_cast<__m128i*>(dst_remaining), packed);

			src_remaining += 8;
			dst_remaining += 8;
			remaining -= 8;
		}

		while (remaining-- > 0)
		{
			*dst_remaining++ = static_cast<i16>(*src_remaining++);
		}

		_mm_sfence();
	}

	template<> void cast_towardszero<u32, i32>(const u32* src_ptr, i32* dst_ptr, usize count) noexcept
	{
		constexpr usize cache_line_size = 64;
		constexpr usize simd_stride = 32;
		constexpr usize prefetch_ahead = 4 * cache_line_size;

		usize i = 0;

		while (i < count && (reinterpret_cast<uintptr_t>(dst_ptr + i) % 32) != 0)
		{
			dst_ptr[i] = static_cast<i32>(src_ptr[i]);
			i++;
		}

		const u32* src_aligned = src_ptr + i;
		i32* dst_aligned = dst_ptr + i;
		const usize main_loop_count = (count - i) / simd_stride;
		const u32* main_loop_end = src_aligned + main_loop_count * simd_stride;

		for (; src_aligned < main_loop_end; src_aligned += simd_stride, dst_aligned += simd_stride)
		{
			_mm_prefetch(reinterpret_cast<const char*>(src_aligned) + prefetch_ahead, _MM_HINT_T0);

			__m256i block0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src_aligned + 0));
			__m256i block1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src_aligned + 8));

			_mm256_stream_si256(reinterpret_cast<__m256i*>(dst_aligned + 0), block0);
			_mm256_stream_si256(reinterpret_cast<__m256i*>(dst_aligned + 8), block1);

			__m256i block2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src_aligned + 16));
			__m256i block3 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src_aligned + 24));

			_mm256_stream_si256(reinterpret_cast<__m256i*>(dst_aligned + 16), block2);
			_mm256_stream_si256(reinterpret_cast<__m256i*>(dst_aligned + 24), block3);
		}

		usize processed = main_loop_count * simd_stride;
		usize remaining = count - i - processed;
		const u32* src_remaining = src_aligned;
		i32* dst_remaining = dst_aligned;

		while (remaining >= 8)
		{
			__m256i block = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src_remaining));
			_mm256_stream_si256(reinterpret_cast<__m256i*>(dst_remaining), block);
			src_remaining += 8;
			dst_remaining += 8;
			remaining -= 8;
		}

		while (remaining-- > 0)
		{
			*dst_remaining++ = static_cast<i32>(*src_remaining++);
		}

		_mm_sfence();
	}

	template<> void cast_towardszero<u32, i64>(const u32* src_ptr, i64* dst_ptr, usize count) noexcept
	{
		constexpr usize cache_line_size = 64;
		constexpr usize simd_stride = 32;
		constexpr usize prefetch_ahead = 4 * cache_line_size;

		usize i = 0;

		while (i < count && (reinterpret_cast<uintptr_t>(dst_ptr + i) % 32) != 0)
		{
			dst_ptr[i] = static_cast<i64>(src_ptr[i]);
			i++;
		}

		const u32* src_aligned = src_ptr + i;
		i64* dst_aligned = dst_ptr + i;
		const usize main_loop_count = (count - i) / simd_stride;
		const u32* main_loop_end = src_aligned + main_loop_count * simd_stride;

		for (; src_aligned < main_loop_end; src_aligned += simd_stride, dst_aligned += simd_stride)
		{
			_mm_prefetch(reinterpret_cast<const char*>(src_aligned) + prefetch_ahead, _MM_HINT_T0);

			for (usize j = 0; j < simd_stride; j += 8)
			{
				const u32* src = src_aligned + j;
				i64* dst = dst_aligned + j;

				__m256i input = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src));

				__m128i low = _mm256_extracti128_si256(input, 0);
				__m256i low_ext = _mm256_cvtepu32_epi64(low);
				_mm256_stream_si256(reinterpret_cast<__m256i*>(dst), low_ext);

				__m128i high = _mm256_extracti128_si256(input, 1);
				__m256i high_ext = _mm256_cvtepu32_epi64(high);
				_mm256_stream_si256(reinterpret_cast<__m256i*>(dst + 4), high_ext);
			}
		}

		usize processed = main_loop_count * simd_stride;
		usize remaining = count - i - processed;
		const u32* src_remaining = src_aligned;
		i64* dst_remaining = dst_aligned;

		while (remaining >= 8)
		{
			__m256i input = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src_remaining));
			__m128i low = _mm256_extracti128_si256(input, 0);
			__m128i high = _mm256_extracti128_si256(input, 1);
			__m256i low_ext = _mm256_cvtepu32_epi64(low);
			__m256i high_ext = _mm256_cvtepu32_epi64(high);
			_mm256_storeu_si256(reinterpret_cast<__m256i*>(dst_remaining), low_ext);
			_mm256_storeu_si256(reinterpret_cast<__m256i*>(dst_remaining + 4), high_ext);
			src_remaining += 8;
			dst_remaining += 8;
			remaining -= 8;
		}

		while (remaining-- > 0)
		{
			*dst_remaining++ = static_cast<i64>(*src_remaining++);
		}

		_mm_sfence();
	}

	template<> void cast_towardszero<u32, u8>(const u32* src_ptr, u8* dst_ptr, usize count) noexcept
	{
		constexpr usize cache_line_size = 64;
		constexpr usize simd_stride = 64;
		constexpr usize prefetch_ahead = 4 * cache_line_size;

		usize i = 0;

		while (i < count && (reinterpret_cast<uintptr_t>(dst_ptr + i) % 16) != 0)
		{
			dst_ptr[i] = static_cast<u8>(src_ptr[i]);
			i++;
		}

		const u32* src_aligned = src_ptr + i;
		u8* dst_aligned = dst_ptr + i;
		const usize main_loop_count = (count - i) / simd_stride;
		const u32* main_loop_end = src_aligned + main_loop_count * simd_stride;

		const __m128i shuffle_mask = _mm_set_epi8(
			0x80, 0x80, 0x80, 0x80,
			0x80, 0x80, 0x80, 0x80,
			0x80, 0x80, 0x80, 0x80,
			12, 8, 4, 0
		);

		for (; src_aligned < main_loop_end; src_aligned += simd_stride, dst_aligned += simd_stride)
		{
			_mm_prefetch(reinterpret_cast<const char*>(src_aligned) + prefetch_ahead, _MM_HINT_T0);

			for (usize j = 0; j < simd_stride; j += 16)
			{
				const u32* src = src_aligned + j;
				u8* dst = dst_aligned + j;

				__m128i block0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + 0));
				__m128i block1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + 4));
				__m128i block2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + 8));
				__m128i block3 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + 12));

				__m128i bytes0 = _mm_shuffle_epi8(block0, shuffle_mask);
				__m128i bytes1 = _mm_shuffle_epi8(block1, shuffle_mask);
				__m128i bytes2 = _mm_shuffle_epi8(block2, shuffle_mask);
				__m128i bytes3 = _mm_shuffle_epi8(block3, shuffle_mask);

				bytes1 = _mm_slli_si128(bytes1, 4);
				bytes2 = _mm_slli_si128(bytes2, 8);
				bytes3 = _mm_slli_si128(bytes3, 12);
				__m128i merged = _mm_or_si128(_mm_or_si128(bytes0, bytes1), _mm_or_si128(bytes2, bytes3));

				_mm_stream_si128(reinterpret_cast<__m128i*>(dst), merged);
			}
		}

		usize processed = main_loop_count * simd_stride;
		usize remaining = count - i - processed;
		const u32* src_remaining = src_aligned;
		u8* dst_remaining = dst_aligned;

		while (remaining >= 16)
		{
			__m128i block0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src_remaining + 0));
			__m128i block1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src_remaining + 4));
			__m128i block2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src_remaining + 8));
			__m128i block3 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src_remaining + 12));

			__m128i bytes0 = _mm_shuffle_epi8(block0, shuffle_mask);
			__m128i bytes1 = _mm_shuffle_epi8(block1, shuffle_mask);
			__m128i bytes2 = _mm_shuffle_epi8(block2, shuffle_mask);
			__m128i bytes3 = _mm_shuffle_epi8(block3, shuffle_mask);

			bytes1 = _mm_slli_si128(bytes1, 4);
			bytes2 = _mm_slli_si128(bytes2, 8);
			bytes3 = _mm_slli_si128(bytes3, 12);
			__m128i merged = _mm_or_si128(_mm_or_si128(bytes0, bytes1), _mm_or_si128(bytes2, bytes3));

			_mm_storeu_si128(reinterpret_cast<__m128i*>(dst_remaining), merged);
			src_remaining += 16;
			dst_remaining += 16;
			remaining -= 16;
		}

		while (remaining-- > 0)
		{
			*dst_remaining++ = static_cast<u8>(*src_remaining++);
		}

		_mm_sfence();
	}

	template<> void cast_towardszero<u32, u16>(const u32* src_ptr, u16* dst_ptr, usize count) noexcept
	{
		constexpr usize cache_line_size = 64;
		constexpr usize simd_stride = 32;
		constexpr usize prefetch_ahead = 4 * cache_line_size;

		usize i = 0;

		while (i < count && (reinterpret_cast<uintptr_t>(dst_ptr + i) % 16) != 0)
		{
			dst_ptr[i] = static_cast<u16>(src_ptr[i]);
			i++;
		}

		const u32* src_aligned = src_ptr + i;
		u16* dst_aligned = dst_ptr + i;
		const usize main_loop_count = (count - i) / simd_stride;
		const u32* main_loop_end = src_aligned + main_loop_count * simd_stride;

		const __m128i mask32 = _mm_set1_epi32(0x0000FFFF);

		for (; src_aligned < main_loop_end; src_aligned += simd_stride, dst_aligned += simd_stride)
		{
			_mm_prefetch(reinterpret_cast<const char*>(src_aligned) + prefetch_ahead, _MM_HINT_T0);

			for (usize j = 0; j < simd_stride; j += 8)
			{
				const u32* src = src_aligned + j;
				u16* dst = dst_aligned + j;

				__m128i block0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + 0));
				__m128i block1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + 4));

				block0 = _mm_and_si128(block0, mask32);
				block1 = _mm_and_si128(block1, mask32);

				__m128i packed = _mm_packus_epi32(block0, block1);

				_mm_stream_si128(reinterpret_cast<__m128i*>(dst), packed);
			}
		}

		usize processed = main_loop_count * simd_stride;
		usize remaining = count - i - processed;
		const u32* src_remaining = src_aligned;
		u16* dst_remaining = dst_aligned;

		while (remaining >= 8)
		{
			__m128i block0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src_remaining + 0));
			__m128i block1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src_remaining + 4));

			block0 = _mm_and_si128(block0, mask32);
			block1 = _mm_and_si128(block1, mask32);

			__m128i packed = _mm_packus_epi32(block0, block1);
			_mm_storeu_si128(reinterpret_cast<__m128i*>(dst_remaining), packed);

			src_remaining += 8;
			dst_remaining += 8;
			remaining -= 8;
		}

		while (remaining-- > 0)
		{
			*dst_remaining++ = static_cast<u16>(*src_remaining++);
		}

		_mm_sfence();
	}

	template<> void cast_towardszero<u32, u32>(const u32* src_ptr, u32* dst_ptr, usize count) noexcept
	{
		std::memcpy(dst_ptr, src_ptr, count * sizeof(u32));
	}

	template<> void cast_towardszero<u32, u64>(const u32* src_ptr, u64* dst_ptr, usize count) noexcept
	{
		constexpr usize cache_line_size = 64;
		constexpr usize simd_stride = 32;
		constexpr usize prefetch_ahead = 4 * cache_line_size;

		usize i = 0;

		while (i < count && (reinterpret_cast<uintptr_t>(dst_ptr + i) % 32) != 0)
		{
			dst_ptr[i] = static_cast<u64>(src_ptr[i]);
			i++;
		}

		const u32* src_aligned = src_ptr + i;
		u64* dst_aligned = dst_ptr + i;
		const usize main_loop_count = (count - i) / simd_stride;
		const u32* main_loop_end = src_aligned + main_loop_count * simd_stride;

		for (; src_aligned < main_loop_end; src_aligned += simd_stride, dst_aligned += simd_stride)
		{
			_mm_prefetch(reinterpret_cast<const char*>(src_aligned) + prefetch_ahead, _MM_HINT_T0);

			for (usize j = 0; j < simd_stride; j += 8)
			{
				const u32* src = src_aligned + j;
				u64* dst = dst_aligned + j;

				__m256i input = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src));

				__m128i low = _mm256_extracti128_si256(input, 0);
				__m256i low_ext = _mm256_cvtepu32_epi64(low);
				_mm256_stream_si256(reinterpret_cast<__m256i*>(dst), low_ext);

				__m128i high = _mm256_extracti128_si256(input, 1);
				__m256i high_ext = _mm256_cvtepu32_epi64(high);
				_mm256_stream_si256(reinterpret_cast<__m256i*>(dst + 4), high_ext);
			}
		}

		usize processed = main_loop_count * simd_stride;
		usize remaining = count - i - processed;
		const u32* src_remaining = src_aligned;
		u64* dst_remaining = dst_aligned;

		while (remaining >= 8)
		{
			__m256i input = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src_remaining));
			__m128i low = _mm256_extracti128_si256(input, 0);
			__m128i high = _mm256_extracti128_si256(input, 1);
			__m256i low_ext = _mm256_cvtepu32_epi64(low);
			__m256i high_ext = _mm256_cvtepu32_epi64(high);
			_mm256_storeu_si256(reinterpret_cast<__m256i*>(dst_remaining), low_ext);
			_mm256_storeu_si256(reinterpret_cast<__m256i*>(dst_remaining + 4), high_ext);
			src_remaining += 8;
			dst_remaining += 8;
			remaining -= 8;
		}

		while (remaining-- > 0)
		{
			*dst_remaining++ = static_cast<u64>(*src_remaining++);
		}

		_mm_sfence();
	}

	template<> void cast_towardszero<u32, f16>(const u32* src_ptr, f16* dst_ptr, usize count) noexcept
	{
		constexpr usize cache_line_size = 64;
		constexpr usize simd_stride = 16;
		constexpr usize prefetch_ahead = 4 * cache_line_size;

		usize i = 0;

		while (i < count && (reinterpret_cast<uintptr_t>(dst_ptr + i) % 32) != 0)
		{
			dst_ptr[i] = static_cast<f16>(static_cast<float>(src_ptr[i]));
			i++;
		}

		const u32* src_aligned = src_ptr + i;
		f16* dst_aligned = dst_ptr + i;
		const usize main_loop_count = (count - i) / simd_stride;
		const u32* main_loop_end = src_aligned + main_loop_count * simd_stride;

		const __m256i mask_low = _mm256_set1_epi32(0xFFFF);
		const __m256 scale = _mm256_set1_ps(65536.0f);

		for (; src_aligned < main_loop_end; src_aligned += simd_stride, dst_aligned += simd_stride)
		{
			_mm_prefetch(reinterpret_cast<const char*>(src_aligned) + prefetch_ahead, _MM_HINT_T0);

			__m256i src0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src_aligned));
			__m256i src1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src_aligned + 8));

			__m256 f32_0, f32_1;
			{
				__m256i hi = _mm256_srli_epi32(src0, 16);
				__m256i lo = _mm256_and_si256(src0, mask_low);
				__m256 hi_f = _mm256_cvtepi32_ps(hi);
				__m256 lo_f = _mm256_cvtepi32_ps(lo);
				f32_0 = _mm256_fmadd_ps(hi_f, scale, lo_f);
			}

			{
				__m256i hi = _mm256_srli_epi32(src1, 16);
				__m256i lo = _mm256_and_si256(src1, mask_low);
				__m256 hi_f = _mm256_cvtepi32_ps(hi);
				__m256 lo_f = _mm256_cvtepi32_ps(lo);
				f32_1 = _mm256_fmadd_ps(hi_f, scale, lo_f);
			}

			__m128i h0 = _mm256_cvtps_ph(f32_0, _MM_FROUND_TO_NEAREST_INT);
			__m128i h1 = _mm256_cvtps_ph(f32_1, _MM_FROUND_TO_NEAREST_INT);

			_mm256_stream_si256(reinterpret_cast<__m256i*>(dst_aligned),
				_mm256_set_m128i(h1, h0));
		}

		usize processed = main_loop_count * simd_stride;
		usize remaining = count - i - processed;
		const u32* src_remaining = src_aligned;
		f16* dst_remaining = dst_aligned;

		while (remaining >= 8)
		{
			__m256i src = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src_remaining));

			__m256i hi = _mm256_srli_epi32(src, 16);
			__m256i lo = _mm256_and_si256(src, mask_low);
			__m256 hi_f = _mm256_cvtepi32_ps(hi);
			__m256 lo_f = _mm256_cvtepi32_ps(lo);
			__m256 f32_vec = _mm256_fmadd_ps(hi_f, scale, lo_f);

			__m128i h = _mm256_cvtps_ph(f32_vec, _MM_FROUND_TO_NEAREST_INT);
			_mm_storeu_si128(reinterpret_cast<__m128i*>(dst_remaining), h);
			src_remaining += 8;
			dst_remaining += 8;
			remaining -= 8;
		}

		if (remaining >= 4)
		{
			__m128i src = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src_remaining));
			__m256i src_256 = _mm256_castsi128_si256(src);

			__m256i hi = _mm256_srli_epi32(src_256, 16);
			__m256i lo = _mm256_and_si256(src_256, mask_low);
			__m256 hi_f = _mm256_cvtepi32_ps(hi);
			__m256 lo_f = _mm256_cvtepi32_ps(lo);
			__m256 f32_vec = _mm256_fmadd_ps(hi_f, scale, lo_f);

			__m128i h = _mm256_cvtps_ph(f32_vec, _MM_FROUND_TO_NEAREST_INT);
			_mm_storel_epi64(reinterpret_cast<__m128i*>(dst_remaining), h);
			src_remaining += 4;
			dst_remaining += 4;
			remaining -= 4;
		}

		for (; remaining > 0; remaining--)
		{
			*dst_remaining++ = static_cast<f16>(static_cast<float>(*src_remaining++));
		}

		_mm_sfence();
	}

	template<> void cast_towardszero<u32, f32>(const u32* src_ptr, f32* dst_ptr, usize count) noexcept
	{
		constexpr usize cache_line_size = 64;
		constexpr usize simd_stride = 8;
		constexpr usize prefetch_ahead = 4 * cache_line_size;

		usize i = 0;

		while (i < count && (reinterpret_cast<uintptr_t>(dst_ptr + i) % 32) != 0)
		{
			dst_ptr[i] = static_cast<f32>(src_ptr[i]);
			i++;
		}

		const u32* src_aligned = src_ptr + i;
		f32* dst_aligned = dst_ptr + i;
		const usize main_loop_count = (count - i) / simd_stride;
		const u32* main_loop_end = src_aligned + main_loop_count * simd_stride;

		const __m256 scale = _mm256_set1_ps(65536.0f);
		const __m256i mask_low = _mm256_set1_epi32(0xFFFF);

		for (; src_aligned < main_loop_end; src_aligned += simd_stride, dst_aligned += simd_stride)
		{
			_mm_prefetch(reinterpret_cast<const char*>(src_aligned) + prefetch_ahead, _MM_HINT_T0);

			__m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src_aligned));

			__m256i hi = _mm256_srli_epi32(v, 16);
			__m256i lo = _mm256_and_si256(v, mask_low);

			__m256 hi_f = _mm256_cvtepi32_ps(hi);
			__m256 lo_f = _mm256_cvtepi32_ps(lo);

			__m256 result = _mm256_fmadd_ps(hi_f, scale, lo_f);

			_mm256_stream_ps(dst_aligned, result);
		}

		usize processed = main_loop_count * simd_stride;
		usize remaining = count - i - processed;
		const u32* src_remaining = src_aligned;
		f32* dst_remaining = dst_aligned;

		while (remaining >= 8)
		{
			__m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src_remaining));
			__m256i hi = _mm256_srli_epi32(v, 16);
			__m256i lo = _mm256_and_si256(v, mask_low);
			__m256 hi_f = _mm256_cvtepi32_ps(hi);
			__m256 lo_f = _mm256_cvtepi32_ps(lo);
			__m256 result = _mm256_fmadd_ps(hi_f, scale, lo_f);
			_mm256_storeu_ps(dst_remaining, result);
			src_remaining += 8;
			dst_remaining += 8;
			remaining -= 8;
		}

		while (remaining-- > 0)
		{
			*dst_remaining++ = static_cast<f32>(*src_remaining++);
		}

		_mm_sfence();
	}

	template<> void cast_towardszero<u32, f64>(const u32* src_ptr, f64* dst_ptr, usize count) noexcept
	{
		constexpr usize cache_line_size = 64;
		constexpr usize simd_stride = 8;
		constexpr usize prefetch_ahead = 4 * cache_line_size;

		usize i = 0;

		while (i < count && (reinterpret_cast<uintptr_t>(dst_ptr + i) % 32) != 0)
		{
			dst_ptr[i] = static_cast<f64>(src_ptr[i]);
			i++;
		}

		const u32* src_aligned = src_ptr + i;
		f64* dst_aligned = dst_ptr + i;
		const usize main_loop_count = (count - i) / simd_stride;
		const u32* main_loop_end = src_aligned + main_loop_count * simd_stride;

		const __m256d two_pow_32 = _mm256_set1_pd(4294967296.0);

		for (; src_aligned < main_loop_end; src_aligned += simd_stride, dst_aligned += simd_stride)
		{
			_mm_prefetch(reinterpret_cast<const char*>(src_aligned) + prefetch_ahead, _MM_HINT_T0);

			__m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src_aligned));

			__m128i v_low = _mm256_castsi256_si128(v);
			__m256d f_low = _mm256_cvtepi32_pd(v_low);
			__m256d cmp_low = _mm256_cmp_pd(f_low, _mm256_setzero_pd(), _CMP_LT_OQ);
			__m256d corrected_low = _mm256_add_pd(f_low, _mm256_and_pd(cmp_low, two_pow_32));
			_mm256_stream_pd(dst_aligned, corrected_low);

			__m128i v_high = _mm256_extractf128_si256(v, 1);
			__m256d f_high = _mm256_cvtepi32_pd(v_high);
			__m256d cmp_high = _mm256_cmp_pd(f_high, _mm256_setzero_pd(), _CMP_LT_OQ);
			__m256d corrected_high = _mm256_add_pd(f_high, _mm256_and_pd(cmp_high, two_pow_32));
			_mm256_stream_pd(dst_aligned + 4, corrected_high);
		}

		usize processed = main_loop_count * simd_stride;
		usize remaining = count - i - processed;
		const u32* src_remaining = src_aligned;
		f64* dst_remaining = dst_aligned;

		while (remaining >= 4)
		{
			__m128i v = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src_remaining));
			__m256d f = _mm256_cvtepi32_pd(v);
			__m256d cmp = _mm256_cmp_pd(f, _mm256_setzero_pd(), _CMP_LT_OQ);
			__m256d corrected = _mm256_add_pd(f, _mm256_and_pd(cmp, two_pow_32));
			_mm256_storeu_pd(dst_remaining, corrected);
			src_remaining += 4;
			dst_remaining += 4;
			remaining -= 4;
		}

		while (remaining-- > 0)
		{
			*dst_remaining++ = static_cast<f64>(*src_remaining++);
		}

		_mm_sfence();
	}


	template<> void cast_towardszero<i8, u8>(const i8* src_ptr, u8* dst_ptr, usize count) noexcept
	{
		constexpr usize cache_line_size = 64;
		constexpr usize simd_stride = 64;
		constexpr usize prefetch_ahead = 4 * cache_line_size;

		const uintptr_t base_addr = reinterpret_cast<uintptr_t>(dst_ptr);
		const usize misalignment = base_addr % 32;
		usize i = 0;

		if (misalignment > 0)
		{
			const usize align_count = std::min(32 - misalignment, count);
			for (; i < align_count; ++i)
			{
				dst_ptr[i] = static_cast<u8>(src_ptr[i]);
			}
		}

		const i8* src_aligned = src_ptr + i;
		u8* dst_aligned = dst_ptr + i;
		const usize main_loop_count = (count - i) / simd_stride;
		const i8* main_loop_end = src_aligned + main_loop_count * simd_stride;

		for (; src_aligned < main_loop_end; src_aligned += simd_stride, dst_aligned += simd_stride)
		{
			_mm_prefetch(reinterpret_cast<const char*>(src_aligned) + prefetch_ahead, _MM_HINT_T0);

			const __m256i block0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src_aligned));
			const __m256i block1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src_aligned + 32));

			_mm256_stream_si256(reinterpret_cast<__m256i*>(dst_aligned), block0);
			_mm256_stream_si256(reinterpret_cast<__m256i*>(dst_aligned + 32), block1);
		}

		usize processed = main_loop_count * simd_stride;
		usize remaining = count - i - processed;

		while (remaining >= 32)
		{
			const __m256i block = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src_aligned));
			_mm256_stream_si256(reinterpret_cast<__m256i*>(dst_aligned), block);
			src_aligned += 32;
			dst_aligned += 32;
			remaining -= 32;
		}

		if (remaining >= 16)
		{
			const __m128i block = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src_aligned));
			_mm_stream_si128(reinterpret_cast<__m128i*>(dst_aligned), block);
			src_aligned += 16;
			dst_aligned += 16;
			remaining -= 16;
		}

		for (; remaining > 0; --remaining)
		{
			*dst_aligned++ = static_cast<u8>(*src_aligned++);
		}

		_mm_sfence();
	}

	template<> void cast_towardszero<i8, u16>(const i8* src_ptr, u16* dst_ptr, usize count) noexcept
	{
		convert_8bits_to_16bits_integral_Invoker<true, false>::process(src_ptr, dst_ptr, count);
	}

	template<> void cast_towardszero<i8, u32>(const i8* src_ptr, u32* dst_ptr, usize count) noexcept
	{
		constexpr usize cache_line_size = 64;
		constexpr usize simd_stride = 64;
		constexpr usize prefetch_ahead = 4 * cache_line_size;

		usize i = 0;

		while (i < count && (reinterpret_cast<uintptr_t>(dst_ptr + i) % 16) != 0)
		{
			dst_ptr[i] = static_cast<u32>(src_ptr[i]);
			i++;
		}

		const i8* src_aligned = src_ptr + i;
		u32* dst_aligned = dst_ptr + i;
		const usize main_loop_count = (count - i) / simd_stride;
		const i8* main_loop_end = src_aligned + main_loop_count * simd_stride;

		for (; src_aligned < main_loop_end; src_aligned += simd_stride, dst_aligned += simd_stride)
		{
			_mm_prefetch(reinterpret_cast<const char*>(src_aligned) + prefetch_ahead, _MM_HINT_T0);

			const __m128i block0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src_aligned));
			const __m128i block1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src_aligned + 16));
			const __m128i block2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src_aligned + 32));
			const __m128i block3 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src_aligned + 48));

			__m128i b0_p0 = _mm_cvtepi8_epi32(block0);
			__m128i b0_p1 = _mm_cvtepi8_epi32(_mm_srli_si128(block0, 4));
			__m128i b0_p2 = _mm_cvtepi8_epi32(_mm_srli_si128(block0, 8));
			__m128i b0_p3 = _mm_cvtepi8_epi32(_mm_srli_si128(block0, 12));

			__m128i b1_p0 = _mm_cvtepi8_epi32(block1);
			__m128i b1_p1 = _mm_cvtepi8_epi32(_mm_srli_si128(block1, 4));
			__m128i b1_p2 = _mm_cvtepi8_epi32(_mm_srli_si128(block1, 8));
			__m128i b1_p3 = _mm_cvtepi8_epi32(_mm_srli_si128(block1, 12));

			__m128i b2_p0 = _mm_cvtepi8_epi32(block2);
			__m128i b2_p1 = _mm_cvtepi8_epi32(_mm_srli_si128(block2, 4));
			__m128i b2_p2 = _mm_cvtepi8_epi32(_mm_srli_si128(block2, 8));
			__m128i b2_p3 = _mm_cvtepi8_epi32(_mm_srli_si128(block2, 12));

			__m128i b3_p0 = _mm_cvtepi8_epi32(block3);
			__m128i b3_p1 = _mm_cvtepi8_epi32(_mm_srli_si128(block3, 4));
			__m128i b3_p2 = _mm_cvtepi8_epi32(_mm_srli_si128(block3, 8));
			__m128i b3_p3 = _mm_cvtepi8_epi32(_mm_srli_si128(block3, 12));

			_mm_stream_si128(reinterpret_cast<__m128i*>(dst_aligned + 0), b0_p0);
			_mm_stream_si128(reinterpret_cast<__m128i*>(dst_aligned + 4), b0_p1);
			_mm_stream_si128(reinterpret_cast<__m128i*>(dst_aligned + 8), b0_p2);
			_mm_stream_si128(reinterpret_cast<__m128i*>(dst_aligned + 12), b0_p3);

			_mm_stream_si128(reinterpret_cast<__m128i*>(dst_aligned + 16), b1_p0);
			_mm_stream_si128(reinterpret_cast<__m128i*>(dst_aligned + 20), b1_p1);
			_mm_stream_si128(reinterpret_cast<__m128i*>(dst_aligned + 24), b1_p2);
			_mm_stream_si128(reinterpret_cast<__m128i*>(dst_aligned + 28), b1_p3);

			_mm_stream_si128(reinterpret_cast<__m128i*>(dst_aligned + 32), b2_p0);
			_mm_stream_si128(reinterpret_cast<__m128i*>(dst_aligned + 36), b2_p1);
			_mm_stream_si128(reinterpret_cast<__m128i*>(dst_aligned + 40), b2_p2);
			_mm_stream_si128(reinterpret_cast<__m128i*>(dst_aligned + 44), b2_p3);

			_mm_stream_si128(reinterpret_cast<__m128i*>(dst_aligned + 48), b3_p0);
			_mm_stream_si128(reinterpret_cast<__m128i*>(dst_aligned + 52), b3_p1);
			_mm_stream_si128(reinterpret_cast<__m128i*>(dst_aligned + 56), b3_p2);
			_mm_stream_si128(reinterpret_cast<__m128i*>(dst_aligned + 60), b3_p3);
		}

		usize processed = main_loop_count * simd_stride;
		usize remaining = count - i - processed;
		const i8* src_remaining = src_aligned;
		u32* dst_remaining = dst_aligned;

		while (remaining >= 16)
		{
			const __m128i block = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src_remaining));
			__m128i p0 = _mm_cvtepi8_epi32(block);
			__m128i p1 = _mm_cvtepi8_epi32(_mm_srli_si128(block, 4));
			__m128i p2 = _mm_cvtepi8_epi32(_mm_srli_si128(block, 8));
			__m128i p3 = _mm_cvtepi8_epi32(_mm_srli_si128(block, 12));

			_mm_stream_si128(reinterpret_cast<__m128i*>(dst_remaining + 0), p0);
			_mm_stream_si128(reinterpret_cast<__m128i*>(dst_remaining + 4), p1);
			_mm_stream_si128(reinterpret_cast<__m128i*>(dst_remaining + 8), p2);
			_mm_stream_si128(reinterpret_cast<__m128i*>(dst_remaining + 12), p3);

			src_remaining += 16;
			dst_remaining += 16;
			remaining -= 16;
		}

		for (; remaining > 0; --remaining)
		{
			*dst_remaining++ = static_cast<u32>(*src_remaining++);
		}

		_mm_sfence();
	}

	template<> void cast_towardszero<i8, u64>(const i8* src_ptr, u64* dst_ptr, usize count) noexcept
	{
		constexpr usize cache_line_size = 64;
		constexpr usize simd_stride = 32;
		constexpr usize prefetch_ahead = 4 * cache_line_size;

		const uintptr_t base_addr = reinterpret_cast<uintptr_t>(dst_ptr);
		const usize misalignment = (base_addr % 32) / sizeof(u64);
		usize i = 0;

		if (misalignment > 0)
		{
			const usize align_count = std::min(static_cast<usize>(32 / sizeof(u64) - misalignment), count);
			for (; i < align_count; ++i)
			{
				dst_ptr[i] = static_cast<u64>(src_ptr[i]);
			}
		}

		const i8* src_aligned = src_ptr + i;
		u64* dst_aligned = dst_ptr + i;
		const usize main_loop_count = (count - i) / simd_stride;
		const i8* main_loop_end = src_aligned + main_loop_count * simd_stride;

		for (; src_aligned < main_loop_end; src_aligned += simd_stride, dst_aligned += simd_stride)
		{
			_mm_prefetch(reinterpret_cast<const char*>(src_aligned) + prefetch_ahead, _MM_HINT_T0);

			for (usize batch = 0; batch < 4; ++batch)
			{
				const i8* batch_src = src_aligned + batch * 8;
				u64* batch_dst = dst_aligned + batch * 8;

				const __m128i chunk = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(batch_src));
				const __m256i u32_vec = _mm256_cvtepi8_epi32(chunk);

				const __m256i u64_low = _mm256_cvtepi32_epi64(_mm256_extracti128_si256(u32_vec, 0));
				const __m256i u64_high = _mm256_cvtepi32_epi64(_mm256_extracti128_si256(u32_vec, 1));

				_mm256_stream_si256(reinterpret_cast<__m256i*>(batch_dst), u64_low);
				_mm256_stream_si256(reinterpret_cast<__m256i*>(batch_dst + 4), u64_high);
			}
		}

		const usize remaining = count - (main_loop_count * simd_stride + i);
		for (usize j = 0; j < remaining; ++j)
		{
			dst_aligned[j] = static_cast<u64>(src_aligned[j]);
		}

		_mm_sfence();
	}

	template<> void cast_towardszero<i8, i8>(const i8* src_ptr, i8* dst_ptr, usize count) noexcept
	{
		std::memcpy(dst_ptr, src_ptr, count * sizeof(i8));
	}

	template<> void cast_towardszero<i8, i16>(const i8* src_ptr, i16* dst_ptr, usize count) noexcept
	{
		convert_8bits_to_16bits_integral_Invoker<true, true>::process(src_ptr, dst_ptr, count);
	}

	template<> void cast_towardszero<i8, i32>(const i8* src_ptr, i32* dst_ptr, usize count) noexcept
	{
		constexpr usize cache_line_size = 64;
		constexpr usize simd_stride = 64;
		constexpr usize prefetch_ahead = 4 * cache_line_size;

		usize i = 0;

		while (i < count && (reinterpret_cast<uintptr_t>(dst_ptr + i) % 16) != 0)
		{
			dst_ptr[i] = static_cast<i32>(src_ptr[i]);
			i++;
		}

		const i8* src_aligned = src_ptr + i;
		i32* dst_aligned = dst_ptr + i;
		const usize main_loop_count = (count - i) / simd_stride;
		const i8* main_loop_end = src_aligned + main_loop_count * simd_stride;

		for (; src_aligned < main_loop_end; src_aligned += simd_stride, dst_aligned += simd_stride)
		{
			_mm_prefetch(reinterpret_cast<const char*>(src_aligned) + prefetch_ahead, _MM_HINT_T0);

			const __m128i block0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src_aligned));
			const __m128i block1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src_aligned + 16));
			const __m128i block2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src_aligned + 32));
			const __m128i block3 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src_aligned + 48));

			__m128i b0_p0 = _mm_cvtepi8_epi32(block0);
			__m128i b0_p1 = _mm_cvtepi8_epi32(_mm_srli_si128(block0, 4));
			__m128i b0_p2 = _mm_cvtepi8_epi32(_mm_srli_si128(block0, 8));
			__m128i b0_p3 = _mm_cvtepi8_epi32(_mm_srli_si128(block0, 12));

			__m128i b1_p0 = _mm_cvtepi8_epi32(block1);
			__m128i b1_p1 = _mm_cvtepi8_epi32(_mm_srli_si128(block1, 4));
			__m128i b1_p2 = _mm_cvtepi8_epi32(_mm_srli_si128(block1, 8));
			__m128i b1_p3 = _mm_cvtepi8_epi32(_mm_srli_si128(block1, 12));

			__m128i b2_p0 = _mm_cvtepi8_epi32(block2);
			__m128i b2_p1 = _mm_cvtepi8_epi32(_mm_srli_si128(block2, 4));
			__m128i b2_p2 = _mm_cvtepi8_epi32(_mm_srli_si128(block2, 8));
			__m128i b2_p3 = _mm_cvtepi8_epi32(_mm_srli_si128(block2, 12));

			__m128i b3_p0 = _mm_cvtepi8_epi32(block3);
			__m128i b3_p1 = _mm_cvtepi8_epi32(_mm_srli_si128(block3, 4));
			__m128i b3_p2 = _mm_cvtepi8_epi32(_mm_srli_si128(block3, 8));
			__m128i b3_p3 = _mm_cvtepi8_epi32(_mm_srli_si128(block3, 12));

			_mm_stream_si128(reinterpret_cast<__m128i*>(dst_aligned + 0), b0_p0);
			_mm_stream_si128(reinterpret_cast<__m128i*>(dst_aligned + 4), b0_p1);
			_mm_stream_si128(reinterpret_cast<__m128i*>(dst_aligned + 8), b0_p2);
			_mm_stream_si128(reinterpret_cast<__m128i*>(dst_aligned + 12), b0_p3);

			_mm_stream_si128(reinterpret_cast<__m128i*>(dst_aligned + 16), b1_p0);
			_mm_stream_si128(reinterpret_cast<__m128i*>(dst_aligned + 20), b1_p1);
			_mm_stream_si128(reinterpret_cast<__m128i*>(dst_aligned + 24), b1_p2);
			_mm_stream_si128(reinterpret_cast<__m128i*>(dst_aligned + 28), b1_p3);

			_mm_stream_si128(reinterpret_cast<__m128i*>(dst_aligned + 32), b2_p0);
			_mm_stream_si128(reinterpret_cast<__m128i*>(dst_aligned + 36), b2_p1);
			_mm_stream_si128(reinterpret_cast<__m128i*>(dst_aligned + 40), b2_p2);
			_mm_stream_si128(reinterpret_cast<__m128i*>(dst_aligned + 44), b2_p3);

			_mm_stream_si128(reinterpret_cast<__m128i*>(dst_aligned + 48), b3_p0);
			_mm_stream_si128(reinterpret_cast<__m128i*>(dst_aligned + 52), b3_p1);
			_mm_stream_si128(reinterpret_cast<__m128i*>(dst_aligned + 56), b3_p2);
			_mm_stream_si128(reinterpret_cast<__m128i*>(dst_aligned + 60), b3_p3);
		}

		usize processed = main_loop_count * simd_stride;
		usize remaining = count - i - processed;
		const i8* src_remaining = src_aligned;
		i32* dst_remaining = dst_aligned;

		while (remaining >= 16)
		{
			const __m128i block = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src_remaining));
			__m128i p0 = _mm_cvtepi8_epi32(block);
			__m128i p1 = _mm_cvtepi8_epi32(_mm_srli_si128(block, 4));
			__m128i p2 = _mm_cvtepi8_epi32(_mm_srli_si128(block, 8));
			__m128i p3 = _mm_cvtepi8_epi32(_mm_srli_si128(block, 12));

			_mm_stream_si128(reinterpret_cast<__m128i*>(dst_remaining + 0), p0);
			_mm_stream_si128(reinterpret_cast<__m128i*>(dst_remaining + 4), p1);
			_mm_stream_si128(reinterpret_cast<__m128i*>(dst_remaining + 8), p2);
			_mm_stream_si128(reinterpret_cast<__m128i*>(dst_remaining + 12), p3);

			src_remaining += 16;
			dst_remaining += 16;
			remaining -= 16;
		}

		for (; remaining > 0; --remaining)
		{
			*dst_remaining++ = static_cast<i32>(*src_remaining++);
		}

		_mm_sfence();
	}

	template<> void cast_towardszero<i8, i64>(const i8* src_ptr, i64* dst_ptr, usize count) noexcept
	{
		constexpr usize cache_line_size = 64;
		constexpr usize simd_stride = 32;
		constexpr usize prefetch_ahead = 4 * cache_line_size;

		const uintptr_t base_addr = reinterpret_cast<uintptr_t>(dst_ptr);
		const usize misalignment = (base_addr % 32) / sizeof(i64);
		usize i = 0;

		if (misalignment > 0)
		{
			const usize align_count = std::min(static_cast<usize>(32 / sizeof(i64) - misalignment), count);
			for (; i < align_count; ++i)
			{
				dst_ptr[i] = static_cast<i64>(src_ptr[i]);
			}
		}

		const i8* src_aligned = src_ptr + i;
		i64* dst_aligned = dst_ptr + i;
		const usize main_loop_count = (count - i) / simd_stride;
		const i8* main_loop_end = src_aligned + main_loop_count * simd_stride;

		for (; src_aligned < main_loop_end; src_aligned += simd_stride, dst_aligned += simd_stride)
		{
			_mm_prefetch(reinterpret_cast<const char*>(src_aligned) + prefetch_ahead, _MM_HINT_T0);

			for (usize batch = 0; batch < 4; ++batch)
			{
				const i8* batch_src = src_aligned + batch * 8;
				i64* batch_dst = dst_aligned + batch * 8;

				const __m128i chunk = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(batch_src));
				const __m256i i32_vec = _mm256_cvtepi8_epi32(chunk);

				const __m256i i64_low = _mm256_cvtepi32_epi64(_mm256_extracti128_si256(i32_vec, 0));
				const __m256i i64_high = _mm256_cvtepi32_epi64(_mm256_extracti128_si256(i32_vec, 1));

				_mm256_stream_si256(reinterpret_cast<__m256i*>(batch_dst), i64_low);
				_mm256_stream_si256(reinterpret_cast<__m256i*>(batch_dst + 4), i64_high);
			}
		}

		const usize remaining = count - (main_loop_count * simd_stride + i);
		for (usize j = 0; j < remaining; ++j)
		{
			dst_aligned[j] = static_cast<i64>(src_aligned[j]);
		}

		_mm_sfence();
	}

	template<> void cast_towardszero<i8, f16>(const i8* src_ptr, f16* dst_ptr, usize count) noexcept
	{
		constexpr usize cache_line_size = 64;
		constexpr usize simd_stride = 64;
		constexpr usize prefetch_ahead = 4 * cache_line_size;

		const uintptr_t base_addr = reinterpret_cast<uintptr_t>(dst_ptr);
		const usize misalignment = (base_addr % 32) / sizeof(f16);
		usize i = 0;

		if (misalignment > 0)
		{
			const usize align_count = std::min(static_cast<usize>(32 / sizeof(f16) - misalignment), count);

			for (; i + 8 <= align_count; i += 8)
			{
				__m128i chunk = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(src_ptr + i));
				__m256i i16_vec = _mm256_cvtepi8_epi16(chunk);
				__m128i i16_low = _mm256_extracti128_si256(i16_vec, 0);

				__m256i i32_low = _mm256_cvtepi16_epi32(i16_low);
				__m256 f32_low = _mm256_cvtepi32_ps(i32_low);

				__m128i h = _mm256_cvtps_ph(f32_low, _MM_FROUND_TO_NEAREST_INT);
				_mm_storeu_si128(reinterpret_cast<__m128i*>(dst_ptr + i), h);
			}

			for (; i < align_count; ++i)
			{
				dst_ptr[i] = static_cast<f16>(static_cast<f32>(src_ptr[i]));
			}
		}

		const i8* src_aligned = src_ptr + i;
		f16* dst_aligned = dst_ptr + i;
		const usize main_loop_count = (count - i) / simd_stride;
		const i8* main_loop_end = src_aligned + main_loop_count * simd_stride;

		for (; src_aligned < main_loop_end; src_aligned += simd_stride, dst_aligned += simd_stride)
		{
			_mm_prefetch(reinterpret_cast<const char*>(src_aligned) + prefetch_ahead, _MM_HINT_T0);

			for (usize pb = 0; pb < 4; ++pb)
			{
				const __m128i block_src = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src_aligned + pb * 16));
				const __m256i i16_block = _mm256_cvtepi8_epi16(block_src);

				const __m128i i16_low = _mm256_extracti128_si256(i16_block, 0);
				const __m128i i16_high = _mm256_extracti128_si256(i16_block, 1);

				const __m256i i32_low = _mm256_cvtepi16_epi32(i16_low);
				const __m256i i32_high = _mm256_cvtepi16_epi32(i16_high);

				const __m256 f32_low = _mm256_cvtepi32_ps(i32_low);
				const __m256 f32_high = _mm256_cvtepi32_ps(i32_high);

				const __m128i h_low = _mm256_cvtps_ph(f32_low, _MM_FROUND_TO_NEAREST_INT);
				const __m128i h_high = _mm256_cvtps_ph(f32_high, _MM_FROUND_TO_NEAREST_INT);

				__m256i vres = _mm256_set_m128i(h_high, h_low);
				_mm256_stream_si256(reinterpret_cast<__m256i*>(dst_aligned + pb * 16), vres);
			}
		}

		usize processed = main_loop_count * simd_stride;
		usize remaining = count - i - processed;

		if (remaining >= 8)
		{
			__m128i chunk = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(src_aligned));
			__m256i i16_vec = _mm256_cvtepi8_epi16(chunk);
			__m128i i16_low = _mm256_extracti128_si256(i16_vec, 0);

			__m256i i32_low = _mm256_cvtepi16_epi32(i16_low);
			__m256 f32_low = _mm256_cvtepi32_ps(i32_low);

			__m128i h = _mm256_cvtps_ph(f32_low, _MM_FROUND_TO_NEAREST_INT);
			_mm_storeu_si128(reinterpret_cast<__m128i*>(dst_aligned), h);
			src_aligned += 8;
			dst_aligned += 8;
			remaining -= 8;
		}

		for (; remaining > 0; --remaining)
		{
			*dst_aligned++ = static_cast<f16>(static_cast<f32>(*src_aligned++));
		}

		_mm_sfence();
	}

	template<> void cast_towardszero<i8, f32>(const i8* src_ptr, f32* dst_ptr, usize count) noexcept
	{
		constexpr usize cache_line_size = 64;
		constexpr usize simd_stride = 64;
		constexpr usize prefetch_ahead = 4 * cache_line_size;

		const uintptr_t base_addr = reinterpret_cast<uintptr_t>(dst_ptr);
		const usize misalignment = (base_addr % 32) / sizeof(f32);
		usize i = 0;

		if (misalignment > 0)
		{
			const usize align_count = std::min(32 / sizeof(f32) - misalignment, count);

			for (; i + 4 <= align_count; i += 4)
			{
				__m128i chunk = _mm_cvtsi32_si128(*reinterpret_cast<const int*>(src_ptr + i));
				__m256i i32_vec = _mm256_cvtepi8_epi32(chunk);
				__m256 f32_vec = _mm256_cvtepi32_ps(i32_vec);
				_mm_storeu_ps(dst_ptr + i, _mm256_castps256_ps128(f32_vec));
			}

			for (; i < align_count; ++i)
			{
				dst_ptr[i] = static_cast<f32>(src_ptr[i]);
			}
		}

		const i8* src_aligned = src_ptr + i;
		f32* dst_aligned = dst_ptr + i;
		const usize main_loop_count = (count - i) / simd_stride;
		const i8* main_loop_end = src_aligned + main_loop_count * simd_stride;

		for (; src_aligned < main_loop_end; src_aligned += simd_stride, dst_aligned += simd_stride)
		{
			_mm_prefetch(reinterpret_cast<const char*>(src_aligned) + prefetch_ahead, _MM_HINT_T0);

			const __m128i block0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src_aligned));
			const __m128i block1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src_aligned + 16));
			const __m128i block2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src_aligned + 32));
			const __m128i block3 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src_aligned + 48));

			_mm256_stream_ps(dst_aligned + 0, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(block0)));
			_mm256_stream_ps(dst_aligned + 8, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(block0, 8))));

			_mm256_stream_ps(dst_aligned + 16, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(block1)));
			_mm256_stream_ps(dst_aligned + 24, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(block1, 8))));

			_mm256_stream_ps(dst_aligned + 32, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(block2)));
			_mm256_stream_ps(dst_aligned + 40, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(block2, 8))));

			_mm256_stream_ps(dst_aligned + 48, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(block3)));
			_mm256_stream_ps(dst_aligned + 56, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(block3, 8))));
		}

		usize processed = main_loop_count * simd_stride;
		usize remaining = count - i - processed;

		if (remaining >= 8)
		{
			__m128i chunk = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(src_aligned));
			__m256i i32_vec = _mm256_cvtepi8_epi32(chunk);
			_mm256_storeu_ps(dst_aligned, _mm256_cvtepi32_ps(i32_vec));
			src_aligned += 8;
			dst_aligned += 8;
			remaining -= 8;
		}

		for (; remaining > 0; --remaining)
		{
			*dst_aligned++ = static_cast<f32>(*src_aligned++);
		}

		_mm_sfence();
	}

	template<> void cast_towardszero<i8, f64>(const i8* src_ptr, f64* dst_ptr, usize count) noexcept
	{
		constexpr usize cache_line_size = 64;
		constexpr usize simd_stride = 64;
		constexpr usize output_stride = 64;
		constexpr usize prefetch_ahead = 4 * cache_line_size;

		const uintptr_t base_addr = reinterpret_cast<uintptr_t>(dst_ptr);
		const usize misalignment = (base_addr % 32) / sizeof(f64);
		usize i = 0;

		if (misalignment > 0)
		{
			const usize align_count = std::min(32 / sizeof(f64) - misalignment, count);

			for (; i + 4 <= align_count; i += 4)
			{
				__m128i chunk = _mm_cvtsi32_si128(*reinterpret_cast<const int*>(src_ptr + i));
				__m256i i32_vec = _mm256_cvtepi8_epi32(chunk);
				__m256d f64_vec = _mm256_cvtepi32_pd(_mm256_castsi256_si128(i32_vec));
				_mm256_storeu_pd(dst_ptr + i, f64_vec);
			}

			for (; i < align_count; ++i)
			{
				dst_ptr[i] = static_cast<f64>(src_ptr[i]);
			}
		}

		const i8* src_aligned = src_ptr + i;
		f64* dst_aligned = dst_ptr + i;
		const usize main_loop_count = (count - i) / simd_stride;
		const i8* main_loop_end = src_aligned + main_loop_count * simd_stride;

		for (; src_aligned < main_loop_end; src_aligned += simd_stride, dst_aligned += output_stride)
		{
			_mm_prefetch(reinterpret_cast<const char*>(src_aligned) + prefetch_ahead, _MM_HINT_T0);

			for (usize pd = 0; pd < 4; ++pd)
			{
				const __m128i block = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src_aligned + pd * 16));

				__m128i chunk0 = _mm_cvtsi32_si128(_mm_extract_epi32(block, 0));
				__m256i i32_vec0 = _mm256_cvtepi8_epi32(chunk0);
				_mm256_stream_pd(dst_aligned + pd * 16 + 0, _mm256_cvtepi32_pd(_mm256_castsi256_si128(i32_vec0)));

				__m128i chunk1 = _mm_cvtsi32_si128(_mm_extract_epi32(block, 1));
				__m256i i32_vec1 = _mm256_cvtepi8_epi32(chunk1);
				_mm256_stream_pd(dst_aligned + pd * 16 + 4, _mm256_cvtepi32_pd(_mm256_castsi256_si128(i32_vec1)));

				__m128i chunk2 = _mm_cvtsi32_si128(_mm_extract_epi32(block, 2));
				__m256i i32_vec2 = _mm256_cvtepi8_epi32(chunk2);
				_mm256_stream_pd(dst_aligned + pd * 16 + 8, _mm256_cvtepi32_pd(_mm256_castsi256_si128(i32_vec2)));

				__m128i chunk3 = _mm_cvtsi32_si128(_mm_extract_epi32(block, 3));
				__m256i i32_vec3 = _mm256_cvtepi8_epi32(chunk3);
				_mm256_stream_pd(dst_aligned + pd * 16 + 12, _mm256_cvtepi32_pd(_mm256_castsi256_si128(i32_vec3)));
			}
		}

		usize processed = main_loop_count * simd_stride;
		usize remaining = count - i - processed;

		if (remaining >= 8)
		{
			__m128i chunk = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(src_aligned));
			__m256i i32_vec = _mm256_cvtepi8_epi32(chunk);

			__m256d dbl_lo = _mm256_cvtepi32_pd(_mm256_castsi256_si128(i32_vec));
			__m256d dbl_hi = _mm256_cvtepi32_pd(_mm256_extracti128_si256(i32_vec, 1));

			_mm256_storeu_pd(dst_aligned, dbl_lo);
			_mm256_storeu_pd(dst_aligned + 4, dbl_hi);
			src_aligned += 8;
			dst_aligned += 8;
			remaining -= 8;
		}

		for (; remaining > 0; --remaining)
		{
			*dst_aligned++ = static_cast<f64>(*src_aligned++);
		}

		_mm_sfence();
	}


	template<> void cast_towardszero<i16, i8>(const i16* src_ptr, i8* dst_ptr, usize count) noexcept
	{
		constexpr usize cache_line_size = 64;
		constexpr usize simd_stride = 64;
		constexpr usize prefetch_ahead = 4 * cache_line_size;

		usize i = 0;

		while (i < count && (reinterpret_cast<uintptr_t>(dst_ptr + i) % 32) != 0)
		{
			dst_ptr[i] = static_cast<i8>(src_ptr[i]);
			i++;
		}

		const i16* src_aligned = src_ptr + i;
		i8* dst_aligned = dst_ptr + i;
		const usize main_loop_count = (count - i) / simd_stride;
		const i16* main_loop_end = src_aligned + main_loop_count * simd_stride;

		for (; src_aligned < main_loop_end; src_aligned += simd_stride, dst_aligned += simd_stride)
		{
			_mm_prefetch(reinterpret_cast<const char*>(src_aligned) + prefetch_ahead, _MM_HINT_T0);

			__m256i in0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src_aligned + 0));
			__m256i in1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src_aligned + 16));
			__m256i in2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src_aligned + 32));
			__m256i in3 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src_aligned + 48));

			const __m256i mask_lo8 = _mm256_set1_epi16(0x00FF);
			in0 = _mm256_and_si256(in0, mask_lo8);
			in1 = _mm256_and_si256(in1, mask_lo8);
			in2 = _mm256_and_si256(in2, mask_lo8);
			in3 = _mm256_and_si256(in3, mask_lo8);

			__m256i packed0 = _mm256_packus_epi16(in0, in1);
			__m256i packed1 = _mm256_packus_epi16(in2, in3);

			packed0 = _mm256_permute4x64_epi64(packed0, 0b11011000);
			packed1 = _mm256_permute4x64_epi64(packed1, 0b11011000);

			_mm256_stream_si256(reinterpret_cast<__m256i*>(dst_aligned + 0), packed0);
			_mm256_stream_si256(reinterpret_cast<__m256i*>(dst_aligned + 32), packed1);
		}

		usize processed = main_loop_count * simd_stride;
		usize remaining = count - i - processed;
		const i16* src_remaining = src_aligned;
		i8* dst_remaining = dst_aligned;

		while (remaining >= 16)
		{
			__m128i block0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src_remaining));
			__m128i block1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src_remaining + 8));

			const __m128i mask_lo8 = _mm_set1_epi16(0x00FF);
			block0 = _mm_and_si128(block0, mask_lo8);
			block1 = _mm_and_si128(block1, mask_lo8);
			__m128i packed = _mm_packus_epi16(block0, block1);

			_mm_storeu_si128(reinterpret_cast<__m128i*>(dst_remaining), packed);
			src_remaining += 16;
			dst_remaining += 16;
			remaining -= 16;
		}

		while (remaining-- > 0)
		{
			*dst_remaining++ = static_cast<i8>(*src_remaining++);
		}

		_mm_sfence();
	}

	template<> void cast_towardszero<i16, i16>(const i16* src_ptr, i16* dst_ptr, usize count) noexcept
	{
		std::memcpy(dst_ptr, src_ptr, count * sizeof(i16));
	}

	template<> void cast_towardszero<i16, i32>(const i16* src_ptr, i32* dst_ptr, usize count) noexcept
	{
		constexpr usize simd_stride = 8;
		constexpr usize cache_line_size = 64;
		constexpr usize prefetch_ahead = 4 * cache_line_size;

		usize i = 0;

		while (i < count && (reinterpret_cast<uintptr_t>(dst_ptr + i) % 32) != 0)
		{
			dst_ptr[i] = static_cast<i32>(src_ptr[i]);
			i++;
		}

		const i16* src_aligned = src_ptr + i;
		i32* dst_aligned = dst_ptr + i;
		const usize main_loop_count = (count - i) / simd_stride;
		const i16* main_loop_end = src_aligned + main_loop_count * simd_stride;

		for (; src_aligned < main_loop_end; src_aligned += simd_stride, dst_aligned += simd_stride)
		{
			_mm_prefetch(reinterpret_cast<const char*>(src_aligned) + prefetch_ahead, _MM_HINT_T0);

			__m128i chunk = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src_aligned));
			__m256i extended = _mm256_cvtepi16_epi32(chunk);

			_mm256_stream_si256(reinterpret_cast<__m256i*>(dst_aligned), extended);
		}

		usize remaining = count - i - main_loop_count * simd_stride;
		const i16* src_remaining = src_aligned;
		i32* dst_remaining = dst_aligned;

		while (remaining >= 4)
		{
			__m128i chunk = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(src_remaining));
			__m128i extended = _mm_cvtepi16_epi32(chunk);
			_mm_storeu_si128(reinterpret_cast<__m128i*>(dst_remaining), extended);
			src_remaining += 4;
			dst_remaining += 4;
			remaining -= 4;
		}

		while (remaining-- > 0)
		{
			*dst_remaining++ = static_cast<i32>(*src_remaining++);
		}

		_mm_sfence();
	}

	template<> void cast_towardszero<i16, i64>(const i16* src_ptr, i64* dst_ptr, usize count) noexcept
	{
		constexpr usize simd_stride = 8;
		constexpr usize cache_line_size = 64;
		constexpr usize prefetch_ahead = 4 * cache_line_size;

		usize i = 0;

		while (i < count && (reinterpret_cast<uintptr_t>(dst_ptr + i) % 64) != 0)
		{
			dst_ptr[i] = static_cast<i64>(src_ptr[i]);
			i++;
		}

		const i16* src_aligned = src_ptr + i;
		i64* dst_aligned = dst_ptr + i;
		const usize main_loop_count = (count - i) / simd_stride;
		const i16* main_loop_end = src_aligned + main_loop_count * simd_stride;

		for (; src_aligned < main_loop_end; src_aligned += simd_stride, dst_aligned += simd_stride)
		{
			_mm_prefetch(reinterpret_cast<const char*>(src_aligned) + prefetch_ahead, _MM_HINT_T0);

			__m128i chunk = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src_aligned));

			__m256i low = _mm256_cvtepi16_epi64(chunk);
			__m256i high = _mm256_cvtepi16_epi64(_mm_srli_si128(chunk, 8));

			_mm256_stream_si256(reinterpret_cast<__m256i*>(dst_aligned + 0), low);
			_mm256_stream_si256(reinterpret_cast<__m256i*>(dst_aligned + 4), high);
		}

		usize remaining = count - i - main_loop_count * simd_stride;
		const i16* src_remaining = src_aligned;
		i64* dst_remaining = dst_aligned;

		while (remaining-- > 0)
		{
			*dst_remaining++ = static_cast<i64>(*src_remaining++);
		}

		_mm_sfence();
	}

	template<> void cast_towardszero<i16, u8>(const i16* src_ptr, u8* dst_ptr, usize count) noexcept
	{
		constexpr usize cache_line_size = 64;
		constexpr usize simd_stride = 64;
		constexpr usize prefetch_ahead = 4 * cache_line_size;

		usize i = 0;

		while (i < count && (reinterpret_cast<uintptr_t>(dst_ptr + i) % 32) != 0)
		{
			dst_ptr[i] = static_cast<u8>(src_ptr[i]);
			i++;
		}

		const i16* src_aligned = src_ptr + i;
		u8* dst_aligned = dst_ptr + i;
		const usize main_loop_count = (count - i) / simd_stride;
		const i16* main_loop_end = src_aligned + main_loop_count * simd_stride;

		for (; src_aligned < main_loop_end; src_aligned += simd_stride, dst_aligned += simd_stride)
		{
			_mm_prefetch(reinterpret_cast<const char*>(src_aligned) + prefetch_ahead, _MM_HINT_T0);

			__m256i in0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src_aligned + 0));
			__m256i in1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src_aligned + 16));
			__m256i in2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src_aligned + 32));
			__m256i in3 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src_aligned + 48));

			const __m256i mask = _mm256_set1_epi16(0x00FF);
			in0 = _mm256_and_si256(in0, mask);
			in1 = _mm256_and_si256(in1, mask);
			in2 = _mm256_and_si256(in2, mask);
			in3 = _mm256_and_si256(in3, mask);

			__m256i packed0 = _mm256_packus_epi16(in0, in1);
			__m256i packed1 = _mm256_packus_epi16(in2, in3);
			packed0 = _mm256_permute4x64_epi64(packed0, 0b11011000);
			packed1 = _mm256_permute4x64_epi64(packed1, 0b11011000);

			_mm256_stream_si256(reinterpret_cast<__m256i*>(dst_aligned), packed0);
			_mm256_stream_si256(reinterpret_cast<__m256i*>(dst_aligned + 32), packed1);
		}

		usize processed = main_loop_count * simd_stride;
		usize remaining = count - i - processed;
		const i16* src_remaining = src_aligned;
		u8* dst_remaining = dst_aligned;

		while (remaining >= 16)
		{
			__m128i chunk0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src_remaining));
			__m128i chunk1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src_remaining + 8));

			const __m128i mask = _mm_set1_epi16(0x00FF);
			chunk0 = _mm_and_si128(chunk0, mask);
			chunk1 = _mm_and_si128(chunk1, mask);

			__m128i packed = _mm_packus_epi16(chunk0, chunk1);
			_mm_stream_si128(reinterpret_cast<__m128i*>(dst_remaining), packed);

			src_remaining += 16;
			dst_remaining += 16;
			remaining -= 16;
		}

		while (remaining-- > 0)
		{
			*dst_remaining++ = static_cast<u8>(*src_remaining++);
		}

		_mm_sfence();
	}

	template<> void cast_towardszero<i16, u16>(const i16* src_ptr, u16* dst_ptr, usize count) noexcept
	{
		std::memcpy(dst_ptr, src_ptr, count * sizeof(i16));
	}

	template<> void cast_towardszero<i16, u32>(const i16* src_ptr, u32* dst_ptr, usize count) noexcept
	{
		constexpr usize simd_stride = 8;
		constexpr usize cache_line_size = 64;
		constexpr usize prefetch_ahead = 4 * cache_line_size;

		usize i = 0;

		while (i < count && (reinterpret_cast<uintptr_t>(dst_ptr + i) % 32) != 0)
		{
			dst_ptr[i] = static_cast<u32>(static_cast<i32>(src_ptr[i]));
			i++;
		}

		const i16* src_aligned = src_ptr + i;
		u32* dst_aligned = dst_ptr + i;
		const usize main_loop_count = (count - i) / simd_stride;
		const i16* main_loop_end = src_aligned + main_loop_count * simd_stride;

		for (; src_aligned < main_loop_end; src_aligned += simd_stride, dst_aligned += simd_stride)
		{
			_mm_prefetch(reinterpret_cast<const char*>(src_aligned) + prefetch_ahead, _MM_HINT_T0);

			__m128i chunk = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src_aligned));
			__m256i extended = _mm256_cvtepi16_epi32(chunk);

			_mm256_stream_si256(reinterpret_cast<__m256i*>(dst_aligned), extended);
		}

		usize remaining = count - i - main_loop_count * simd_stride;
		const i16* src_remaining = src_aligned;
		u32* dst_remaining = dst_aligned;

		while (remaining >= 4)
		{
			__m128i chunk = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(src_remaining));
			__m128i extended = _mm_cvtepi16_epi32(chunk);
			_mm_storeu_si128(reinterpret_cast<__m128i*>(dst_remaining), extended);

			src_remaining += 4;
			dst_remaining += 4;
			remaining -= 4;
		}

		while (remaining-- > 0)
		{
			*dst_remaining++ = static_cast<u32>(static_cast<i32>(*src_remaining++));
		}

		_mm_sfence();
	}

	template<> void cast_towardszero<i16, u64>(const i16* src_ptr, u64* dst_ptr, usize count) noexcept
	{
		constexpr usize simd_stride = 8;
		constexpr usize cache_line_size = 64;
		constexpr usize prefetch_ahead = 4 * cache_line_size;

		usize i = 0;

		while (i < count && (reinterpret_cast<uintptr_t>(dst_ptr + i) % 64) != 0)
		{
			dst_ptr[i] = static_cast<u64>(static_cast<i64>(src_ptr[i]));
			i++;
		}

		const i16* src_aligned = src_ptr + i;
		u64* dst_aligned = dst_ptr + i;
		const usize main_loop_count = (count - i) / simd_stride;
		const i16* main_loop_end = src_aligned + main_loop_count * simd_stride;

		for (; src_aligned < main_loop_end; src_aligned += simd_stride, dst_aligned += simd_stride)
		{
			_mm_prefetch(reinterpret_cast<const char*>(src_aligned) + prefetch_ahead, _MM_HINT_T0);

			__m128i chunk = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src_aligned));

			__m128i low_i32 = _mm_cvtepi16_epi32(chunk);
			__m256i low_i64 = _mm256_cvtepi32_epi64(low_i32);

			__m128i high_i32 = _mm_cvtepi16_epi32(_mm_srli_si128(chunk, 8));
			__m256i high_i64 = _mm256_cvtepi32_epi64(high_i32);

			_mm256_stream_si256(reinterpret_cast<__m256i*>(dst_aligned), low_i64);
			_mm256_stream_si256(reinterpret_cast<__m256i*>(dst_aligned + 4), high_i64);
		}

		usize remaining = count - i - main_loop_count * simd_stride;
		const i16* src_remaining = src_aligned;
		u64* dst_remaining = dst_aligned;

		while (remaining-- > 0)
		{
			*dst_remaining++ = static_cast<u64>(static_cast<i64>(*src_remaining++));
		}

		_mm_sfence();
	}

	template<> void cast_towardszero<i16, f16>(const i16* src_ptr, f16* dst_ptr, usize count) noexcept
	{
		constexpr usize cache_line_size = 64;
		constexpr usize simd_stride = 32;
		constexpr usize prefetch_ahead = 4 * cache_line_size;

		const uintptr_t base_addr = reinterpret_cast<uintptr_t>(dst_ptr);
		const usize misalignment = (base_addr % 32) / sizeof(f16);
		usize i = 0;

		if (misalignment > 0)
		{
			const usize align_count = std::min(static_cast<usize>(32 / sizeof(f16) - misalignment), count);
			for (; i + 8 <= align_count; i += 8)
			{
				__m128i chunk = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src_ptr + i));
				__m128i low_i32 = _mm_cvtepi16_epi32(chunk);
				__m128i high_i32 = _mm_cvtepi16_epi32(_mm_srli_si128(chunk, 8));
				__m256i i32_vec = _mm256_set_m128i(high_i32, low_i32);
				__m256 f32_vec = _mm256_cvtepi32_ps(i32_vec);
				__m128i h = _mm256_cvtps_ph(f32_vec, _MM_FROUND_TO_NEAREST_INT);
				_mm_storeu_si128(reinterpret_cast<__m128i*>(dst_ptr + i), h);
			}
			for (; i < align_count; ++i)
			{
				dst_ptr[i] = static_cast<f16>(src_ptr[i]);
			}
		}

		const i16* src_aligned = src_ptr + i;
		f16* dst_aligned = dst_ptr + i;
		const usize main_loop_count = (count - i) / simd_stride;
		const i16* main_loop_end = src_aligned + main_loop_count * simd_stride;

		for (; src_aligned < main_loop_end; src_aligned += simd_stride, dst_aligned += simd_stride)
		{
			_mm_prefetch(reinterpret_cast<const char*>(src_aligned) + prefetch_ahead, _MM_HINT_T0);

			for (usize pb = 0; pb < 4; ++pb)
			{
				__m128i chunk = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src_aligned + pb * 8));
				__m128i low_i32 = _mm_cvtepi16_epi32(chunk);
				__m128i high_i32 = _mm_cvtepi16_epi32(_mm_srli_si128(chunk, 8));
				__m256i i32_vec = _mm256_set_m128i(high_i32, low_i32);
				__m256 f32_vec = _mm256_cvtepi32_ps(i32_vec);
				__m128i h = _mm256_cvtps_ph(f32_vec, _MM_FROUND_TO_NEAREST_INT);
				_mm_stream_si128(reinterpret_cast<__m128i*>(dst_aligned + pb * 8), h);
			}
		}

		usize processed = main_loop_count * simd_stride;
		usize remaining = count - i - processed;

		if (remaining >= 8)
		{
			__m128i chunk = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src_aligned));
			__m128i low_i32 = _mm_cvtepi16_epi32(chunk);
			__m128i high_i32 = _mm_cvtepi16_epi32(_mm_srli_si128(chunk, 8));
			__m256i i32_vec = _mm256_set_m128i(high_i32, low_i32);
			__m256 f32_vec = _mm256_cvtepi32_ps(i32_vec);
			__m128i h = _mm256_cvtps_ph(f32_vec, _MM_FROUND_TO_NEAREST_INT);
			_mm_storeu_si128(reinterpret_cast<__m128i*>(dst_aligned), h);
			src_aligned += 8;
			dst_aligned += 8;
			remaining -= 8;
		}

		for (; remaining > 0; --remaining)
		{
			*dst_aligned++ = static_cast<f16>(*src_aligned++);
		}

		_mm_sfence();
	}

	template<> void cast_towardszero<i16, f32>(const i16* src_ptr, f32* dst_ptr, usize count) noexcept
	{
		constexpr usize cache_line_size = 64;
		constexpr usize simd_stride = 32;
		constexpr usize prefetch_ahead = 4 * cache_line_size;

		const uintptr_t base_addr = reinterpret_cast<uintptr_t>(dst_ptr);
		const usize misalignment = (base_addr % 32) / sizeof(f32);
		usize i = 0;

		if (misalignment > 0)
		{
			const usize align_count = std::min(static_cast<usize>(32 / sizeof(f32) - misalignment), count);
			for (; i + 8 <= align_count; i += 8)
			{
				__m128i chunk = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src_ptr + i));
				__m128i low_i32 = _mm_cvtepi16_epi32(chunk);
				__m128i high_i32 = _mm_cvtepi16_epi32(_mm_srli_si128(chunk, 8));
				__m256i i32_vec = _mm256_set_m128i(high_i32, low_i32);
				__m256 f32_vec = _mm256_cvtepi32_ps(i32_vec);
				_mm256_storeu_ps(dst_ptr + i, f32_vec);
			}
			for (; i < align_count; ++i)
			{
				dst_ptr[i] = static_cast<f32>(src_ptr[i]);
			}
		}

		const i16* src_aligned = src_ptr + i;
		f32* dst_aligned = dst_ptr + i;
		const usize main_loop_count = (count - i) / simd_stride;
		const i16* main_loop_end = src_aligned + main_loop_count * simd_stride;

		for (; src_aligned < main_loop_end; src_aligned += simd_stride, dst_aligned += simd_stride)
		{
			_mm_prefetch(reinterpret_cast<const char*>(src_aligned) + prefetch_ahead, _MM_HINT_T0);

			for (usize pb = 0; pb < 4; ++pb)
			{
				__m128i chunk = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src_aligned + pb * 8));
				__m128i low_i32 = _mm_cvtepi16_epi32(chunk);
				__m128i high_i32 = _mm_cvtepi16_epi32(_mm_srli_si128(chunk, 8));
				__m256i i32_vec = _mm256_set_m128i(high_i32, low_i32);
				__m256 f32_vec = _mm256_cvtepi32_ps(i32_vec);
				_mm256_stream_ps(dst_aligned + pb * 8, f32_vec);
			}
		}

		usize processed = main_loop_count * simd_stride;
		usize remaining = count - i - processed;

		if (remaining >= 8)
		{
			__m128i chunk = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src_aligned));
			__m128i low_i32 = _mm_cvtepi16_epi32(chunk);
			__m128i high_i32 = _mm_cvtepi16_epi32(_mm_srli_si128(chunk, 8));
			__m256i i32_vec = _mm256_set_m128i(high_i32, low_i32);
			__m256 f32_vec = _mm256_cvtepi32_ps(i32_vec);
			_mm256_storeu_ps(dst_aligned, f32_vec);
			src_aligned += 8;
			dst_aligned += 8;
			remaining -= 8;
		}

		for (; remaining > 0; --remaining)
		{
			*dst_aligned++ = static_cast<f32>(*src_aligned++);
		}

		_mm_sfence();
	}

	template<> void cast_towardszero<i16, f64>(const i16* src_ptr, f64* dst_ptr, usize count) noexcept
	{
		constexpr usize cache_line_size = 64;
		constexpr usize simd_stride = 16;
		constexpr usize prefetch_ahead = 4 * cache_line_size;

		const uintptr_t base_addr = reinterpret_cast<uintptr_t>(dst_ptr);
		const usize misalignment = (base_addr % 32) / sizeof(f64);
		usize i = 0;

		if (misalignment > 0)
		{
			const usize align_count = std::min(static_cast<usize>(32 / sizeof(f64) - misalignment), count);
			for (; i + 4 <= align_count; i += 4)
			{
				__m128i chunk = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(src_ptr + i));
				__m128i i32_vec = _mm_cvtepi16_epi32(chunk);
				__m256d f64_vec = _mm256_cvtepi32_pd(i32_vec);
				_mm256_storeu_pd(dst_ptr + i, f64_vec);
			}
			for (; i < align_count; ++i)
			{
				dst_ptr[i] = static_cast<f64>(src_ptr[i]);
			}
		}

		const i16* src_aligned = src_ptr + i;
		f64* dst_aligned = dst_ptr + i;
		const usize main_loop_count = (count - i) / simd_stride;
		const i16* main_loop_end = src_aligned + main_loop_count * simd_stride;

		for (; src_aligned < main_loop_end; src_aligned += simd_stride, dst_aligned += simd_stride)
		{
			_mm_prefetch(reinterpret_cast<const char*>(src_aligned) + prefetch_ahead, _MM_HINT_T0);

			for (usize pd = 0; pd < 4; ++pd)
			{
				__m128i chunk = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(src_aligned + pd * 4));
				__m128i i32_vec = _mm_cvtepi16_epi32(chunk);
				__m256d f64_vec = _mm256_cvtepi32_pd(i32_vec);
				_mm256_stream_pd(dst_aligned + pd * 4, f64_vec);
			}
		}

		usize processed = main_loop_count * simd_stride;
		usize remaining = count - i - processed;

		if (remaining >= 4)
		{
			__m128i chunk = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(src_aligned));
			__m128i i32_vec = _mm_cvtepi16_epi32(chunk);
			__m256d f64_vec = _mm256_cvtepi32_pd(i32_vec);
			_mm256_storeu_pd(dst_aligned, f64_vec);
			src_aligned += 4;
			dst_aligned += 4;
			remaining -= 4;
		}

		for (; remaining > 0; --remaining)
		{
			*dst_aligned++ = static_cast<f64>(*src_aligned++);
		}

		_mm_sfence();
	}

	template<> void cast_towardszero<i32, i8>(const i32* src_ptr, i8* dst_ptr, usize count) noexcept
	{
		constexpr usize cache_line_size = 64;
		constexpr usize simd_stride = 64;
		constexpr usize prefetch_ahead = 4 * cache_line_size;

		usize i = 0;

		while (i < count && (reinterpret_cast<uintptr_t>(dst_ptr + i) % 16) != 0)
		{
			dst_ptr[i] = static_cast<i8>(src_ptr[i]);
			i++;
		}

		const i32* src_aligned = src_ptr + i;
		i8* dst_aligned = dst_ptr + i;
		const usize main_loop_count = (count - i) / simd_stride;
		const i32* main_loop_end = src_aligned + main_loop_count * simd_stride;

		const __m128i shuffle_mask = _mm_set_epi8(
			0x80, 0x80, 0x80, 0x80,
			0x80, 0x80, 0x80, 0x80,
			0x80, 0x80, 0x80, 0x80,
			12, 8, 4, 0
		);

		for (; src_aligned < main_loop_end; src_aligned += simd_stride, dst_aligned += simd_stride)
		{
			_mm_prefetch(reinterpret_cast<const char*>(src_aligned) + prefetch_ahead, _MM_HINT_T0);

			for (usize j = 0; j < simd_stride; j += 16)
			{
				const i32* src = src_aligned + j;
				i8* dst = dst_aligned + j;

				__m128i block0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + 0));
				__m128i block1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + 4));
				__m128i block2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + 8));
				__m128i block3 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + 12));

				__m128i bytes0 = _mm_shuffle_epi8(block0, shuffle_mask);
				__m128i bytes1 = _mm_shuffle_epi8(block1, shuffle_mask);
				__m128i bytes2 = _mm_shuffle_epi8(block2, shuffle_mask);
				__m128i bytes3 = _mm_shuffle_epi8(block3, shuffle_mask);
				
				bytes1 = _mm_slli_si128(bytes1, 4);
				bytes2 = _mm_slli_si128(bytes2, 8);
				bytes3 = _mm_slli_si128(bytes3, 12);
				__m128i merged = _mm_or_si128(_mm_or_si128(bytes0, bytes1), _mm_or_si128(bytes2, bytes3));

				_mm_stream_si128(reinterpret_cast<__m128i*>(dst), merged);
			}
		}

		usize processed = main_loop_count * simd_stride;
		usize remaining = count - i - processed;
		const i32* src_remaining = src_aligned;
		i8* dst_remaining = dst_aligned;

		while (remaining >= 16)
		{
			__m128i block0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src_remaining + 0));
			__m128i block1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src_remaining + 4));
			__m128i block2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src_remaining + 8));
			__m128i block3 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src_remaining + 12));

			__m128i bytes0 = _mm_shuffle_epi8(block0, shuffle_mask);
			__m128i bytes1 = _mm_shuffle_epi8(block1, shuffle_mask);
			__m128i bytes2 = _mm_shuffle_epi8(block2, shuffle_mask);
			__m128i bytes3 = _mm_shuffle_epi8(block3, shuffle_mask);

			bytes1 = _mm_slli_si128(bytes1, 4);
			bytes2 = _mm_slli_si128(bytes2, 8);
			bytes3 = _mm_slli_si128(bytes3, 12);
			__m128i merged = _mm_or_si128(_mm_or_si128(bytes0, bytes1), _mm_or_si128(bytes2, bytes3));

			_mm_stream_si128(reinterpret_cast<__m128i*>(dst_remaining), merged);
			src_remaining += 16;
			dst_remaining += 16;
			remaining -= 16;
		}

		while (remaining-- > 0)
		{
			*dst_remaining++ = static_cast<i8>(*src_remaining++);
		}

		_mm_sfence();
	}

	template<> void cast_towardszero<i32, i16>(const i32* src_ptr, i16* dst_ptr, usize count) noexcept
	{
		constexpr usize cache_line_size = 64;
		constexpr usize simd_stride = 32;
		constexpr usize prefetch_ahead = 4 * cache_line_size;

		usize i = 0;

		while (i < count && (reinterpret_cast<uintptr_t>(dst_ptr + i) % 16) != 0)
		{
			dst_ptr[i] = static_cast<i16>(src_ptr[i]);
			i++;
		}

		const i32* src_aligned = src_ptr + i;
		i16* dst_aligned = dst_ptr + i;
		const usize main_loop_count = (count - i) / simd_stride;
		const i32* main_loop_end = src_aligned + main_loop_count * simd_stride;

		const __m128i mask = _mm_set1_epi32(0x0000FFFF);
		const __m128i shuffle_mask = _mm_set_epi8(
			0x80, 0x80, 0x80, 0x80,
			0x80, 0x80, 0x80, 0x80,
			13, 12, 9, 8, 5, 4, 1, 0
		);

		for (; src_aligned < main_loop_end; src_aligned += simd_stride, dst_aligned += simd_stride)
		{
			_mm_prefetch(reinterpret_cast<const char*>(src_aligned) + prefetch_ahead, _MM_HINT_T0);

			for (usize j = 0; j < simd_stride; j += 8)
			{
				const i32* src = src_aligned + j;
				i16* dst = dst_aligned + j;

				__m128i block0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + 0));
				__m128i block1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + 4));

				block0 = _mm_and_si128(block0, mask);
				block1 = _mm_and_si128(block1, mask);

				__m128i shuffled0 = _mm_shuffle_epi8(block0, shuffle_mask);
				__m128i shuffled1 = _mm_shuffle_epi8(block1, shuffle_mask);

				__m128i packed = _mm_unpacklo_epi64(shuffled0, shuffled1);

				_mm_stream_si128(reinterpret_cast<__m128i*>(dst), packed);
			}
		}

		usize processed = main_loop_count * simd_stride;
		usize remaining = count - i - processed;
		const i32* src_remaining = src_aligned;
		i16* dst_remaining = dst_aligned;

		while (remaining-- > 0)
		{
			*dst_remaining++ = static_cast<i16>(*src_remaining++);
		}

		_mm_sfence();
	}


	template<> void cast_towardszero<i32, i32>(const i32* src_ptr, i32* dst_ptr, usize count) noexcept
	{
		std::memcpy(dst_ptr, src_ptr, count * sizeof(i32));
	}

	template<> void cast_towardszero<i32, i64>(const i32* src_ptr, i64* dst_ptr, usize count) noexcept
	{
		constexpr usize cache_line_size = 64;
		constexpr usize simd_stride = 32;
		constexpr usize prefetch_ahead = 4 * cache_line_size;

		usize i = 0;

		while (i < count && (reinterpret_cast<uintptr_t>(dst_ptr + i) % 32) != 0)
		{
			dst_ptr[i] = static_cast<i64>(src_ptr[i]);
			i++;
		}

		const i32* src_aligned = src_ptr + i;
		i64* dst_aligned = dst_ptr + i;
		const usize main_loop_count = (count - i) / simd_stride;
		const i32* main_loop_end = src_aligned + main_loop_count * simd_stride;

		for (; src_aligned < main_loop_end; src_aligned += simd_stride, dst_aligned += simd_stride)
		{
			_mm_prefetch(reinterpret_cast<const char*>(src_aligned) + prefetch_ahead, _MM_HINT_T0);

			for (usize j = 0; j < simd_stride; j += 8)
			{
				const i32* src = src_aligned + j;
				i64* dst = dst_aligned + j;

				__m256i input = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src));

				__m128i low = _mm256_extracti128_si256(input, 0);
				__m256i low_ext = _mm256_cvtepi32_epi64(low);
				_mm256_stream_si256(reinterpret_cast<__m256i*>(dst), low_ext);

				__m128i high = _mm256_extracti128_si256(input, 1);
				__m256i high_ext = _mm256_cvtepi32_epi64(high);
				_mm256_stream_si256(reinterpret_cast<__m256i*>(dst + 4), high_ext);
			}
		}

		usize processed = main_loop_count * simd_stride;
		usize remaining = count - i - processed;
		const i32* src_remaining = src_aligned;
		i64* dst_remaining = dst_aligned;

		while (remaining >= 8)
		{
			__m256i input = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src_remaining));
			__m128i low = _mm256_extracti128_si256(input, 0);
			__m128i high = _mm256_extracti128_si256(input, 1);
			__m256i low_ext = _mm256_cvtepi32_epi64(low);
			__m256i high_ext = _mm256_cvtepi32_epi64(high);
			_mm256_stream_si256(reinterpret_cast<__m256i*>(dst_remaining), low_ext);
			_mm256_stream_si256(reinterpret_cast<__m256i*>(dst_remaining + 4), high_ext);
			src_remaining += 8;
			dst_remaining += 8;
			remaining -= 8;
		}

		while (remaining-- > 0)
		{
			*dst_remaining++ = static_cast<i64>(*src_remaining++);
		}

		_mm_sfence();
	}

	template<> void cast_towardszero<i32, u8>(const i32* src_ptr, u8* dst_ptr, usize count) noexcept
	{
		constexpr usize cache_line_size = 64;
		constexpr usize simd_stride = 64;
		constexpr usize prefetch_ahead = 4 * cache_line_size;

		usize i = 0;

		while (i < count && (reinterpret_cast<uintptr_t>(dst_ptr + i) % 16) != 0)
		{
			dst_ptr[i] = static_cast<u8>(src_ptr[i]);
			i++;
		}

		const i32* src_aligned = src_ptr + i;
		u8* dst_aligned = dst_ptr + i;
		const usize main_loop_count = (count - i) / simd_stride;
		const i32* main_loop_end = src_aligned + main_loop_count * simd_stride;

		const __m128i shuffle_mask = _mm_set_epi8(
			0x80, 0x80, 0x80, 0x80,
			0x80, 0x80, 0x80, 0x80,
			0x80, 0x80, 0x80, 0x80,
			12, 8, 4, 0
		);

		for (; src_aligned < main_loop_end; src_aligned += simd_stride, dst_aligned += simd_stride)
		{
			_mm_prefetch(reinterpret_cast<const char*>(src_aligned) + prefetch_ahead, _MM_HINT_T0);

			for (usize j = 0; j < simd_stride; j += 16)
			{
				const i32* src = src_aligned + j;
				u8* dst = dst_aligned + j;

				__m128i block0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + 0));
				__m128i block1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + 4));
				__m128i block2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + 8));
				__m128i block3 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + 12));

				__m128i bytes0 = _mm_shuffle_epi8(block0, shuffle_mask);
				__m128i bytes1 = _mm_shuffle_epi8(block1, shuffle_mask);
				__m128i bytes2 = _mm_shuffle_epi8(block2, shuffle_mask);
				__m128i bytes3 = _mm_shuffle_epi8(block3, shuffle_mask);

				bytes1 = _mm_slli_si128(bytes1, 4);
				bytes2 = _mm_slli_si128(bytes2, 8);
				bytes3 = _mm_slli_si128(bytes3, 12);
				__m128i merged = _mm_or_si128(_mm_or_si128(bytes0, bytes1), _mm_or_si128(bytes2, bytes3));

				_mm_stream_si128(reinterpret_cast<__m128i*>(dst), merged);
			}
		}

		usize processed = main_loop_count * simd_stride;
		usize remaining = count - i - processed;
		const i32* src_remaining = src_aligned;
		u8* dst_remaining = dst_aligned;

		while (remaining >= 16)
		{
			__m128i block0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src_remaining + 0));
			__m128i block1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src_remaining + 4));
			__m128i block2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src_remaining + 8));
			__m128i block3 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src_remaining + 12));

			__m128i bytes0 = _mm_shuffle_epi8(block0, shuffle_mask);
			__m128i bytes1 = _mm_shuffle_epi8(block1, shuffle_mask);
			__m128i bytes2 = _mm_shuffle_epi8(block2, shuffle_mask);
			__m128i bytes3 = _mm_shuffle_epi8(block3, shuffle_mask);

			bytes1 = _mm_slli_si128(bytes1, 4);
			bytes2 = _mm_slli_si128(bytes2, 8);
			bytes3 = _mm_slli_si128(bytes3, 12);
			__m128i merged = _mm_or_si128(_mm_or_si128(bytes0, bytes1), _mm_or_si128(bytes2, bytes3));

			_mm_stream_si128(reinterpret_cast<__m128i*>(dst_remaining), merged);
			src_remaining += 16;
			dst_remaining += 16;
			remaining -= 16;
		}

		while (remaining-- > 0)
		{
			*dst_remaining++ = static_cast<u8>(*src_remaining++);
		}

		_mm_sfence();
	}

	template<> void cast_towardszero<i32, u16>(const i32* src_ptr, u16* dst_ptr, usize count) noexcept
	{
		constexpr usize cache_line_size = 64;
		constexpr usize simd_stride = 32;
		constexpr usize prefetch_ahead = 4 * cache_line_size;

		usize i = 0;

		while (i < count && (reinterpret_cast<uintptr_t>(dst_ptr + i) % 16) != 0)
		{
			dst_ptr[i] = static_cast<u16>(src_ptr[i]);
			i++;
		}

		const i32* src_aligned = src_ptr + i;
		u16* dst_aligned = dst_ptr + i;
		const usize main_loop_count = (count - i) / simd_stride;
		const i32* main_loop_end = src_aligned + main_loop_count * simd_stride;

		const __m128i mask = _mm_set1_epi32(0x0000FFFF);
		const __m128i shuffle_mask = _mm_set_epi8(
			0x80, 0x80, 0x80, 0x80,
			0x80, 0x80, 0x80, 0x80,
			13, 12, 9, 8, 5, 4, 1, 0
		);

		for (; src_aligned < main_loop_end; src_aligned += simd_stride, dst_aligned += simd_stride)
		{
			_mm_prefetch(reinterpret_cast<const char*>(src_aligned) + prefetch_ahead, _MM_HINT_T0);

			for (usize j = 0; j < simd_stride; j += 8)
			{
				const i32* src = src_aligned + j;
				u16* dst = dst_aligned + j;

				__m128i block0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + 0));
				__m128i block1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + 4));

				block0 = _mm_and_si128(block0, mask);
				block1 = _mm_and_si128(block1, mask);

				__m128i shuffled0 = _mm_shuffle_epi8(block0, shuffle_mask);
				__m128i shuffled1 = _mm_shuffle_epi8(block1, shuffle_mask);

				__m128i packed = _mm_unpacklo_epi64(shuffled0, shuffled1);

				_mm_stream_si128(reinterpret_cast<__m128i*>(dst), packed);
			}
		}

		usize processed = main_loop_count * simd_stride;
		usize remaining = count - i - processed;
		const i32* src_remaining = src_aligned;
		u16* dst_remaining = dst_aligned;

		while (remaining-- > 0)
		{
			*dst_remaining++ = static_cast<u16>(*src_remaining++);
		}

		_mm_sfence();
	}

	template<> void cast_towardszero<i32, u32>(const i32* src_ptr, u32* dst_ptr, usize count) noexcept
	{
		std::memcpy(dst_ptr, src_ptr, count * sizeof(i32));
	}

	template<> void cast_towardszero<i32, u64>(const i32* src_ptr, u64* dst_ptr, usize count) noexcept
	{
		constexpr usize cache_line_size = 64;
		constexpr usize simd_stride = 32;
		constexpr usize prefetch_ahead = 4 * cache_line_size;

		usize i = 0;

		while (i < count && (reinterpret_cast<uintptr_t>(dst_ptr + i) % 32) != 0)
		{
			dst_ptr[i] = static_cast<u64>(src_ptr[i]);
			i++;
		}

		const i32* src_aligned = src_ptr + i;
		u64* dst_aligned = dst_ptr + i;
		const usize main_loop_count = (count - i) / simd_stride;
		const i32* main_loop_end = src_aligned + main_loop_count * simd_stride;

		for (; src_aligned < main_loop_end; src_aligned += simd_stride, dst_aligned += simd_stride)
		{
			_mm_prefetch(reinterpret_cast<const char*>(src_aligned) + prefetch_ahead, _MM_HINT_T0);

			for (usize j = 0; j < simd_stride; j += 8)
			{
				const i32* src = src_aligned + j;
				u64* dst = dst_aligned + j;

				__m256i input = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src));

				__m128i low = _mm256_extracti128_si256(input, 0);
				__m256i low_ext = _mm256_cvtepi32_epi64(low);
				_mm256_stream_si256(reinterpret_cast<__m256i*>(dst), low_ext);

				__m128i high = _mm256_extracti128_si256(input, 1);
				__m256i high_ext = _mm256_cvtepi32_epi64(high);
				_mm256_stream_si256(reinterpret_cast<__m256i*>(dst + 4), high_ext);
			}
		}

		usize processed = main_loop_count * simd_stride;
		usize remaining = count - i - processed;
		const i32* src_remaining = src_aligned;
		u64* dst_remaining = dst_aligned;

		while (remaining >= 8)
		{
			__m256i input = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src_remaining));
			__m128i low = _mm256_extracti128_si256(input, 0);
			__m128i high = _mm256_extracti128_si256(input, 1);
			__m256i low_ext = _mm256_cvtepi32_epi64(low);
			__m256i high_ext = _mm256_cvtepi32_epi64(high);
			_mm256_stream_si256(reinterpret_cast<__m256i*>(dst_remaining), low_ext);
			_mm256_stream_si256(reinterpret_cast<__m256i*>(dst_remaining + 4), high_ext);
			src_remaining += 8;
			dst_remaining += 8;
			remaining -= 8;
		}

		while (remaining-- > 0)
		{
			*dst_remaining++ = static_cast<u64>(*src_remaining++);
		}

		_mm_sfence();
	}

	template<> void cast_towardszero<i32, f16>(const i32* src_ptr, f16* dst_ptr, usize count) noexcept
	{
		constexpr usize cache_line_size = 64;
		constexpr usize simd_stride = 16;
		constexpr usize prefetch_ahead = 4 * cache_line_size;

		usize i = 0;

		while (i < count && (reinterpret_cast<uintptr_t>(dst_ptr + i) % 32) != 0)
		{
			dst_ptr[i] = static_cast<f16>(static_cast<f32>(src_ptr[i]));
			i++;
		}

		const i32* src_aligned = src_ptr + i;
		f16* dst_aligned = dst_ptr + i;
		const usize main_loop_count = (count - i) / simd_stride;
		const i32* main_loop_end = src_aligned + main_loop_count * simd_stride;

		for (; src_aligned < main_loop_end; src_aligned += simd_stride, dst_aligned += simd_stride)
		{
			_mm_prefetch(reinterpret_cast<const char*>(src_aligned) + prefetch_ahead, _MM_HINT_T0);

			__m256i vec0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src_aligned));
			__m256i vec1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src_aligned + 8));

			__m256 f32_0 = _mm256_cvtepi32_ps(vec0);
			__m256 f32_1 = _mm256_cvtepi32_ps(vec1);

			__m128i h0 = _mm256_cvtps_ph(f32_0, _MM_FROUND_TO_NEAREST_INT);
			__m128i h1 = _mm256_cvtps_ph(f32_1, _MM_FROUND_TO_NEAREST_INT);

			_mm256_stream_si256(reinterpret_cast<__m256i*>(dst_aligned), _mm256_set_m128i(h1, h0));
		}

		usize processed = main_loop_count * simd_stride;
		usize remaining = count - i - processed;
		const i32* src_remaining = src_aligned;
		f16* dst_remaining = dst_aligned;

		while (remaining >= 8)
		{
			__m256i vec = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src_remaining));
			__m256 f32_vec = _mm256_cvtepi32_ps(vec);
			__m128i h = _mm256_cvtps_ph(f32_vec, _MM_FROUND_TO_NEAREST_INT);
			_mm_stream_si128(reinterpret_cast<__m128i*>(dst_remaining), h);
			src_remaining += 8;
			dst_remaining += 8;
			remaining -= 8;
		}

		for (; remaining > 0; remaining--)
		{
			*dst_remaining++ = static_cast<f16>(static_cast<f32>(*src_remaining++));
		}

		_mm_sfence();
	}

	template<> void cast_towardszero<i32, f32>(const i32* src_ptr, f32* dst_ptr, usize count) noexcept
	{
		constexpr usize cache_line_size = 64;
		constexpr usize simd_stride = 8;
		constexpr usize prefetch_ahead = 4 * cache_line_size;

		usize i = 0;

		while (i < count && (reinterpret_cast<uintptr_t>(dst_ptr + i) % 32) != 0)
		{
			dst_ptr[i] = static_cast<f32>(src_ptr[i]);
			i++;
		}

		const i32* src_aligned = src_ptr + i;
		f32* dst_aligned = dst_ptr + i;
		const usize main_loop_count = (count - i) / simd_stride;
		const i32* main_loop_end = src_aligned + main_loop_count * simd_stride;

		for (; src_aligned < main_loop_end; src_aligned += simd_stride, dst_aligned += simd_stride)
		{
			_mm_prefetch(reinterpret_cast<const char*>(src_aligned) + prefetch_ahead, _MM_HINT_T0);

			__m256i int_vec = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src_aligned));
			__m256 float_vec = _mm256_cvtepi32_ps(int_vec);
			_mm256_stream_ps(dst_aligned, float_vec);
		}

		usize processed = main_loop_count * simd_stride;
		usize remaining = count - i - processed;
		const i32* src_remaining = src_aligned;
		f32* dst_remaining = dst_aligned;

		if (remaining >= 4)
		{
			__m128i int128 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src_remaining));
			__m128 float128 = _mm_cvtepi32_ps(int128);
			_mm_stream_ps(dst_remaining, float128);
			src_remaining += 4;
			dst_remaining += 4;
			remaining -= 4;
		}

		for (; remaining > 0; remaining--)
		{
			*dst_remaining++ = static_cast<f32>(*src_remaining++);
		}

		_mm_sfence();
	}

	template<> void cast_towardszero<i32, f64>(const i32* src_ptr, f64* dst_ptr, usize count) noexcept
	{
		constexpr usize cache_line_size = 64;
		constexpr usize simd_stride = 4;
		constexpr usize prefetch_ahead = 4 * cache_line_size;

		usize i = 0;

		while (i < count && (reinterpret_cast<uintptr_t>(dst_ptr + i) % 32) != 0)
		{
			dst_ptr[i] = static_cast<f64>(src_ptr[i]);
			i++;
		}

		const i32* src_aligned = src_ptr + i;
		f64* dst_aligned = dst_ptr + i;
		const usize main_loop_count = (count - i) / simd_stride;
		const i32* main_loop_end = src_aligned + main_loop_count * simd_stride;

		for (; src_aligned < main_loop_end; src_aligned += simd_stride, dst_aligned += simd_stride)
		{
			_mm_prefetch(reinterpret_cast<const char*>(src_aligned) + prefetch_ahead, _MM_HINT_T0);

			__m128i int_vec = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src_aligned));
			__m256d double_vec = _mm256_cvtepi32_pd(int_vec);
			_mm256_stream_pd(dst_aligned, double_vec);
		}

		usize processed = main_loop_count * simd_stride;
		usize remaining = count - i - processed;
		const i32* src_remaining = src_aligned;
		f64* dst_remaining = dst_aligned;

		while (remaining-- > 0)
		{
			*dst_remaining++ = static_cast<f64>(*src_remaining++);
		}

		_mm_sfence();
	}
}

namespace fy
{
	template<> void cast_towardszero<u64, i8>(const u64* src_ptr, i8* dst_ptr, usize count) noexcept
	{
		constexpr usize cache_line_size = 64;
		constexpr usize prefetch_ahead = 4 * cache_line_size;

		const __m128i lane_mask = _mm_set_epi8(
			0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
			0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 8, 0);

		const __m256i shuffle_mask = _mm256_set_m128i(lane_mask, lane_mask);

		const __m128i reorder_mask = _mm_set_epi8(
			0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
			0x80, 0x80, 0x80, 0x80, 3, 1, 2, 0);

		usize i = 0;

		while (i < count && (reinterpret_cast<uintptr_t>(dst_ptr + i) % 16) != 0)
		{
			dst_ptr[i] = static_cast<i8>(src_ptr[i]);
			i++;
		}

		const usize main_loop_count = (count - i) / 16;
		for (usize k = 0; k < main_loop_count; k++)
		{
			_mm_prefetch(reinterpret_cast<const char*>(src_ptr + i + 16 * prefetch_ahead / sizeof(u64)), _MM_HINT_T0);

			__m256i src0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src_ptr + i));
			__m256i src1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src_ptr + i + 4));
			__m256i src2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src_ptr + i + 8));
			__m256i src3 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src_ptr + i + 12));

			__m256i vi0 = _mm256_shuffle_epi8(src0, shuffle_mask);
			__m256i vi1 = _mm256_shuffle_epi8(src1, shuffle_mask);
			__m256i vi2 = _mm256_shuffle_epi8(src2, shuffle_mask);
			__m256i vi3 = _mm256_shuffle_epi8(src3, shuffle_mask);

			__m128i low0 = _mm256_castsi256_si128(vi0);
			__m128i high0 = _mm256_extracti128_si256(vi0, 1);
			__m128i temp0 = _mm_unpacklo_epi8(low0, high0);
			__m128i final0 = _mm_shuffle_epi8(temp0, reorder_mask);

			__m128i low1 = _mm256_castsi256_si128(vi1);
			__m128i high1 = _mm256_extracti128_si256(vi1, 1);
			__m128i temp1 = _mm_unpacklo_epi8(low1, high1);
			__m128i final1 = _mm_shuffle_epi8(temp1, reorder_mask);

			__m128i low2 = _mm256_castsi256_si128(vi2);
			__m128i high2 = _mm256_extracti128_si256(vi2, 1);
			__m128i temp2 = _mm_unpacklo_epi8(low2, high2);
			__m128i final2 = _mm_shuffle_epi8(temp2, reorder_mask);

			__m128i low3 = _mm256_castsi256_si128(vi3);
			__m128i high3 = _mm256_extracti128_si256(vi3, 1);
			__m128i temp3 = _mm_unpacklo_epi8(low3, high3);
			__m128i final3 = _mm_shuffle_epi8(temp3, reorder_mask);

			i32 s0 = _mm_cvtsi128_si32(final0);
			i32 s1 = _mm_cvtsi128_si32(final1);
			i32 s2 = _mm_cvtsi128_si32(final2);
			i32 s3 = _mm_cvtsi128_si32(final3);
			__m128i output = _mm_set_epi32(s3, s2, s1, s0);

			_mm_stream_si128(reinterpret_cast<__m128i*>(dst_ptr + i), output);

			i += 16;
		}

		while (i < count)
		{
			dst_ptr[i] = static_cast<i8>(src_ptr[i]);
			i++;
		}

		_mm_sfence();
	}

	template<> void cast_towardszero<u64, i16>(const u64* src_ptr, i16* dst_ptr, usize count) noexcept
	{
		constexpr usize cache_line_size = 64;
		constexpr usize prefetch_ahead = 4 * cache_line_size;

		const __m128i lane_mask = _mm_set_epi8(
			0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
			13, 12, 9, 8, 5, 4, 1, 0);

		usize i = 0;

		while (i < count && (reinterpret_cast<uintptr_t>(dst_ptr + i) % 16) != 0)
		{
			dst_ptr[i] = static_cast<i16>(src_ptr[i]);
			i++;
		}

		const usize main_loop_count = (count - i) / 8;
		for (usize k = 0; k < main_loop_count; k++)
		{
			_mm_prefetch(reinterpret_cast<const char*>(src_ptr + i + 8 * prefetch_ahead / sizeof(u64)), _MM_HINT_T0);

			__m256i src0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src_ptr + i));
			__m128i low0 = _mm256_castsi256_si128(src0);
			__m128i high0 = _mm256_extracti128_si256(src0, 1);
			__m128i temp0 = _mm_shuffle_epi32(low0, _MM_SHUFFLE(3, 1, 2, 0));
			__m128i temp0b = _mm_shuffle_epi32(high0, _MM_SHUFFLE(3, 1, 2, 0));
			__m128i temp0_32 = _mm_unpacklo_epi64(temp0, temp0b);

			__m256i src1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src_ptr + i + 4));
			__m128i low1 = _mm256_castsi256_si128(src1);
			__m128i high1 = _mm256_extracti128_si256(src1, 1);
			__m128i temp1 = _mm_shuffle_epi32(low1, _MM_SHUFFLE(3, 1, 2, 0));
			__m128i temp1b = _mm_shuffle_epi32(high1, _MM_SHUFFLE(3, 1, 2, 0));
			__m128i temp1_32 = _mm_unpacklo_epi64(temp1, temp1b);

			__m128i final0 = _mm_shuffle_epi8(temp0_32, lane_mask);
			__m128i final1 = _mm_shuffle_epi8(temp1_32, lane_mask);

			__m128i output = _mm_set_epi64x(_mm_cvtsi128_si64(final1), _mm_cvtsi128_si64(final0));

			_mm_stream_si128(reinterpret_cast<__m128i*>(dst_ptr + i), output);

			i += 8;
		}

		while (i < count)
		{
			dst_ptr[i] = static_cast<i16>(src_ptr[i]);
			i++;
		}

		_mm_sfence();
	}

	template<> void cast_towardszero<u64, i32>(const u64* src_ptr, i32* dst_ptr, usize count) noexcept
	{
		constexpr usize cache_line_size = 64;
		constexpr usize prefetch_ahead = 4 * cache_line_size;

		const __m256i permute_mask = _mm256_set_epi32(0, 0, 0, 0, 6, 4, 2, 0);

		usize i = 0;

		while (i < count && (reinterpret_cast<uintptr_t>(dst_ptr + i) % 16) != 0)
		{
			dst_ptr[i] = static_cast<i32>(src_ptr[i]);
			i++;
		}

		const usize main_loop_count = (count - i) / 8;
		for (usize k = 0; k < main_loop_count; k++)
		{
			_mm_prefetch(reinterpret_cast<const char*>(src_ptr + i + 8 * prefetch_ahead / sizeof(u64)), _MM_HINT_T0);

			__m256i src0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src_ptr + i));
			__m256i src1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src_ptr + i + 4));

			__m256i permuted0 = _mm256_permutevar8x32_epi32(src0, permute_mask);
			__m256i permuted1 = _mm256_permutevar8x32_epi32(src1, permute_mask);

			__m128i low0 = _mm256_castsi256_si128(permuted0);
			__m128i low1 = _mm256_castsi256_si128(permuted1);
			__m256i combined = _mm256_set_m128i(low1, low0);

			_mm256_stream_si256(reinterpret_cast<__m256i*>(dst_ptr + i), combined);

			i += 8;
		}

		while (i < count)
		{
			dst_ptr[i] = static_cast<i32>(src_ptr[i]);
			i++;
		}

		_mm_sfence();
	}

	template<> void cast_towardszero<u64, i64>(const u64* src_ptr, i64* dst_ptr, usize count) noexcept
	{
		constexpr usize cache_line_size = 64;
		constexpr usize prefetch_ahead = 4 * cache_line_size;

		usize i = 0;

		while (i < count && (reinterpret_cast<uintptr_t>(dst_ptr + i) % 32) != 0)
		{
			dst_ptr[i] = static_cast<i64>(src_ptr[i]);
			i++;
		}

		const usize main_loop_count = (count - i) / 4;
		for (usize k = 0; k < main_loop_count; k++)
		{
			_mm_prefetch(reinterpret_cast<const char*>(src_ptr + i + 4 * prefetch_ahead / sizeof(u64)), _MM_HINT_T0);

			__m256i src = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src_ptr + i));
			_mm256_stream_si256(reinterpret_cast<__m256i*>(dst_ptr + i), src);

			i += 4;
		}

		while (i < count)
		{
			dst_ptr[i] = static_cast<i64>(src_ptr[i]);
			i++;
		}

		_mm_sfence();
	}

	template<> void cast_towardszero<u64, u8>(const u64* src_ptr, u8* dst_ptr, usize count) noexcept
	{
		constexpr usize cache_line_size = 64;
		constexpr usize prefetch_ahead = 4 * cache_line_size;

		const __m128i lane_mask = _mm_set_epi8(
			0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
			0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 8, 0);

		const __m256i shuffle_mask = _mm256_set_m128i(lane_mask, lane_mask);

		const __m128i reorder_mask = _mm_set_epi8(
			0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
			0x80, 0x80, 0x80, 0x80, 3, 1, 2, 0);

		usize i = 0;

		while (i < count && (reinterpret_cast<uintptr_t>(dst_ptr + i) % 16) != 0)
		{
			dst_ptr[i] = static_cast<u8>(src_ptr[i]);
			i++;
		}

		const usize main_loop_count = (count - i) / 16;
		for (usize k = 0; k < main_loop_count; k++)
		{
			_mm_prefetch(reinterpret_cast<const char*>(src_ptr + i + 16 * prefetch_ahead / sizeof(u64)), _MM_HINT_T0);

			__m256i src0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src_ptr + i));
			__m256i src1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src_ptr + i + 4));
			__m256i src2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src_ptr + i + 8));
			__m256i src3 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src_ptr + i + 12));

			__m256i vi0 = _mm256_shuffle_epi8(src0, shuffle_mask);
			__m256i vi1 = _mm256_shuffle_epi8(src1, shuffle_mask);
			__m256i vi2 = _mm256_shuffle_epi8(src2, shuffle_mask);
			__m256i vi3 = _mm256_shuffle_epi8(src3, shuffle_mask);

			__m128i low0 = _mm256_castsi256_si128(vi0);
			__m128i high0 = _mm256_extracti128_si256(vi0, 1);
			__m128i temp0 = _mm_unpacklo_epi8(low0, high0);
			__m128i final0 = _mm_shuffle_epi8(temp0, reorder_mask);

			__m128i low1 = _mm256_castsi256_si128(vi1);
			__m128i high1 = _mm256_extracti128_si256(vi1, 1);
			__m128i temp1 = _mm_unpacklo_epi8(low1, high1);
			__m128i final1 = _mm_shuffle_epi8(temp1, reorder_mask);

			__m128i low2 = _mm256_castsi256_si128(vi2);
			__m128i high2 = _mm256_extracti128_si256(vi2, 1);
			__m128i temp2 = _mm_unpacklo_epi8(low2, high2);
			__m128i final2 = _mm_shuffle_epi8(temp2, reorder_mask);

			__m128i low3 = _mm256_castsi256_si128(vi3);
			__m128i high3 = _mm256_extracti128_si256(vi3, 1);
			__m128i temp3 = _mm_unpacklo_epi8(low3, high3);
			__m128i final3 = _mm_shuffle_epi8(temp3, reorder_mask);
			
			i32 s0 = _mm_cvtsi128_si32(final0);
			i32 s1 = _mm_cvtsi128_si32(final1);
			i32 s2 = _mm_cvtsi128_si32(final2);
			i32 s3 = _mm_cvtsi128_si32(final3);
			__m128i output = _mm_set_epi32(s3, s2, s1, s0);

			_mm_stream_si128(reinterpret_cast<__m128i*>(dst_ptr + i), output);

			i += 16;
		}

		while (i < count)
		{
			dst_ptr[i] = static_cast<u8>(src_ptr[i]);
			i++;
		}

		_mm_sfence();
	}

	template<> void cast_towardszero<u64, u16>(const u64* src_ptr, u16* dst_ptr, usize count) noexcept
	{
		constexpr usize cache_line_size = 64;
		constexpr usize prefetch_ahead = 4 * cache_line_size;

		const __m128i lane_mask = _mm_set_epi8(
			0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
			13, 12, 9, 8, 5, 4, 1, 0);

		usize i = 0;

		while (i < count && (reinterpret_cast<uintptr_t>(dst_ptr + i) % 16) != 0)
		{
			dst_ptr[i] = static_cast<u16>(src_ptr[i]);
			i++;
		}

		const usize main_loop_count = (count - i) / 8;
		for (usize k = 0; k < main_loop_count; k++)
		{
			_mm_prefetch(reinterpret_cast<const char*>(src_ptr + i + 8 * prefetch_ahead / sizeof(u64)), _MM_HINT_T0);

			__m256i src0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src_ptr + i));
			__m128i low0 = _mm256_castsi256_si128(src0);
			__m128i high0 = _mm256_extracti128_si256(src0, 1);
			__m128i temp0 = _mm_shuffle_epi32(low0, _MM_SHUFFLE(3, 1, 2, 0));
			__m128i temp0b = _mm_shuffle_epi32(high0, _MM_SHUFFLE(3, 1, 2, 0));
			__m128i temp0_32 = _mm_unpacklo_epi64(temp0, temp0b);

			__m256i src1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src_ptr + i + 4));
			__m128i low1 = _mm256_castsi256_si128(src1);
			__m128i high1 = _mm256_extracti128_si256(src1, 1);
			__m128i temp1 = _mm_shuffle_epi32(low1, _MM_SHUFFLE(3, 1, 2, 0));
			__m128i temp1b = _mm_shuffle_epi32(high1, _MM_SHUFFLE(3, 1, 2, 0));
			__m128i temp1_32 = _mm_unpacklo_epi64(temp1, temp1b);

			__m128i final0 = _mm_shuffle_epi8(temp0_32, lane_mask);
			__m128i final1 = _mm_shuffle_epi8(temp1_32, lane_mask);

			__m128i output = _mm_set_epi64x(_mm_cvtsi128_si64(final1), _mm_cvtsi128_si64(final0));
			_mm_stream_si128(reinterpret_cast<__m128i*>(dst_ptr + i), output);

			i += 8;
		}

		while (i < count)
		{
			dst_ptr[i] = static_cast<u16>(src_ptr[i]);
			i++;
		}

		_mm_sfence();
	}

	template<> void cast_towardszero<u64, u32>(const u64* src_ptr, u32* dst_ptr, usize count) noexcept
	{
		constexpr usize cache_line_size = 64;
		constexpr usize prefetch_ahead = 4 * cache_line_size;

		const __m256i permute_mask = _mm256_set_epi32(0, 0, 0, 0, 6, 4, 2, 0);

		usize i = 0;

		while (i < count && (reinterpret_cast<uintptr_t>(dst_ptr + i) % 32) != 0)
		{
			dst_ptr[i] = static_cast<u32>(src_ptr[i]);
			i++;
		}

		const usize main_loop_count = (count - i) / 8;
		for (usize k = 0; k < main_loop_count; k++)
		{
			_mm_prefetch(reinterpret_cast<const char*>(src_ptr + i + 8 * prefetch_ahead / sizeof(u64)), _MM_HINT_T0);

			__m256i src0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src_ptr + i));
			__m256i src1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src_ptr + i + 4));

			__m256i permuted0 = _mm256_permutevar8x32_epi32(src0, permute_mask);
			__m256i permuted1 = _mm256_permutevar8x32_epi32(src1, permute_mask);

			__m128i low0 = _mm256_castsi256_si128(permuted0);
			__m128i low1 = _mm256_castsi256_si128(permuted1);
			__m256i combined = _mm256_set_m128i(low1, low0);

			_mm256_stream_si256(reinterpret_cast<__m256i*>(dst_ptr + i), combined);

			i += 8;
		}

		while (i < count)
		{
			dst_ptr[i] = static_cast<u32>(src_ptr[i]);
			i++;
		}

		_mm_sfence();
	}

	template<> void cast_towardszero<u64, u64>(const u64* src_ptr, u64* dst_ptr, usize count) noexcept
	{
		std::memcpy(dst_ptr, src_ptr, count * sizeof(u64));
	}

	template<> void cast_towardszero<u64, f16>(const u64* src_ptr, f16* dst_ptr, usize count) noexcept
	{
		for (usize i = 0; i < count; ++i) dst_ptr[i] = static_cast<f16>(src_ptr[i]);
	}

	template<> void cast_towardszero<u64, f32>(const u64* src_ptr, f32* dst_ptr, usize count) noexcept
	{
		for (usize i = 0; i < count; ++i) dst_ptr[i] = static_cast<f32>(src_ptr[i]);
	}

	template<> void cast_towardszero<u64, f64>(const u64* src_ptr, f64* dst_ptr, usize count) noexcept
	{
		for (usize i = 0; i < count; ++i) dst_ptr[i] = static_cast<f64>(src_ptr[i]);
	}
}

namespace fy
{
	template<> void cast_towardszero<i64, i8>(const i64* src_ptr, i8* dst_ptr, usize count) noexcept
	{
		constexpr usize cache_line_size = 64;
		constexpr usize prefetch_ahead = 4 * cache_line_size;

		const __m128i lane_mask = _mm_set_epi8(
			0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
			0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 8, 0);

		const __m256i shuffle_mask = _mm256_set_m128i(lane_mask, lane_mask);

		const __m128i reorder_mask = _mm_set_epi8(
			0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
			0x80, 0x80, 0x80, 0x80, 3, 1, 2, 0);

		usize i = 0;

		while (i < count && (reinterpret_cast<uintptr_t>(dst_ptr + i) % 16) != 0)
		{
			dst_ptr[i] = static_cast<i8>(src_ptr[i]);
			i++;
		}

		const usize main_loop_count = (count - i) / 16;
		for (usize k = 0; k < main_loop_count; k++)
		{
			_mm_prefetch(reinterpret_cast<const char*>(src_ptr + i + 16 * prefetch_ahead / sizeof(i64)), _MM_HINT_T0);

			__m256i src0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src_ptr + i));
			__m256i src1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src_ptr + i + 4));
			__m256i src2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src_ptr + i + 8));
			__m256i src3 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src_ptr + i + 12));

			__m256i vi0 = _mm256_shuffle_epi8(src0, shuffle_mask);
			__m256i vi1 = _mm256_shuffle_epi8(src1, shuffle_mask);
			__m256i vi2 = _mm256_shuffle_epi8(src2, shuffle_mask);
			__m256i vi3 = _mm256_shuffle_epi8(src3, shuffle_mask);

			__m128i low0 = _mm256_castsi256_si128(vi0);
			__m128i high0 = _mm256_extracti128_si256(vi0, 1);
			__m128i temp0 = _mm_unpacklo_epi8(low0, high0);
			__m128i final0 = _mm_shuffle_epi8(temp0, reorder_mask);

			__m128i low1 = _mm256_castsi256_si128(vi1);
			__m128i high1 = _mm256_extracti128_si256(vi1, 1);
			__m128i temp1 = _mm_unpacklo_epi8(low1, high1);
			__m128i final1 = _mm_shuffle_epi8(temp1, reorder_mask);

			__m128i low2 = _mm256_castsi256_si128(vi2);
			__m128i high2 = _mm256_extracti128_si256(vi2, 1);
			__m128i temp2 = _mm_unpacklo_epi8(low2, high2);
			__m128i final2 = _mm_shuffle_epi8(temp2, reorder_mask);

			__m128i low3 = _mm256_castsi256_si128(vi3);
			__m128i high3 = _mm256_extracti128_si256(vi3, 1);
			__m128i temp3 = _mm_unpacklo_epi8(low3, high3);
			__m128i final3 = _mm_shuffle_epi8(temp3, reorder_mask);

			i32 s0 = _mm_cvtsi128_si32(final0);
			i32 s1 = _mm_cvtsi128_si32(final1);
			i32 s2 = _mm_cvtsi128_si32(final2);
			i32 s3 = _mm_cvtsi128_si32(final3);
			__m128i output = _mm_set_epi32(s3, s2, s1, s0);

			_mm_stream_si128(reinterpret_cast<__m128i*>(dst_ptr + i), output);

			i += 16;
		}

		while (i < count)
		{
			dst_ptr[i] = static_cast<i8>(src_ptr[i]);
			i++;
		}

		_mm_sfence();
	}

	template<> void cast_towardszero<i64, i16>(const i64* src_ptr, i16* dst_ptr, usize count) noexcept
	{
		for (usize i = 0; i < count; ++i) dst_ptr[i] = static_cast<i16>(src_ptr[i]);
	}

	template<> void cast_towardszero<i64, i32>(const i64* src_ptr, i32* dst_ptr, usize count) noexcept
	{
		const uintptr_t dst_addr = reinterpret_cast<uintptr_t>(dst_ptr);
		constexpr usize alignment = 32;
		const usize align_offset = (alignment - (dst_addr % alignment)) % alignment;
		usize n_pre = std::min(align_offset / sizeof(i32), count);

		for (usize i = 0; i < n_pre; ++i)
		{
			dst_ptr[i] = static_cast<i32>(src_ptr[i]);
		}

		src_ptr += n_pre;
		dst_ptr += n_pre;
		usize remaining = count - n_pre;

		constexpr usize elements_per_vector = 8;
		constexpr usize unroll_factor = 4;
		constexpr usize prefetch_offset = 64;

		usize i = 0;
		const usize main_count = remaining / elements_per_vector * elements_per_vector;
		const usize unrolled_count = main_count / (elements_per_vector * unroll_factor) * (elements_per_vector * unroll_factor);

		const __m256i permute_mask = _mm256_setr_epi32(0, 2, 4, 6, 1, 3, 5, 7);

		for (; i < unrolled_count; i += elements_per_vector * unroll_factor)
		{
			_mm_prefetch(reinterpret_cast<const char*>(src_ptr + i + prefetch_offset), _MM_HINT_T0);

			for (usize j = 0; j < unroll_factor; ++j)
			{
				const usize offset = i + j * elements_per_vector;

				__m256i v0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src_ptr + offset));
				__m256i shuffled0 = _mm256_permutevar8x32_epi32(v0, permute_mask);
				__m128i res0 = _mm256_castsi256_si128(shuffled0);

				__m256i v1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src_ptr + offset + 4));
				__m256i shuffled1 = _mm256_permutevar8x32_epi32(v1, permute_mask);
				__m128i res1 = _mm256_castsi256_si128(shuffled1);

				__m256i result = _mm256_set_m128i(res1, res0);

				_mm256_stream_si256(reinterpret_cast<__m256i*>(dst_ptr + offset), result);
			}
		}

		for (; i < main_count; i += elements_per_vector)
		{
			__m256i v0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src_ptr + i));
			__m256i shuffled0 = _mm256_permutevar8x32_epi32(v0, permute_mask);
			__m128i res0 = _mm256_castsi256_si128(shuffled0);

			__m256i v1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src_ptr + i + 4));
			__m256i shuffled1 = _mm256_permutevar8x32_epi32(v1, permute_mask);
			__m128i res1 = _mm256_castsi256_si128(shuffled1);

			__m256i result = _mm256_set_m128i(res1, res0);
			_mm256_stream_si256(reinterpret_cast<__m256i*>(dst_ptr + i), result);
		}

		for (usize j = main_count; j < remaining; ++j)
		{
			dst_ptr[j] = static_cast<i32>(src_ptr[j]);
		}

		_mm_sfence();
	}

	template<> void cast_towardszero<i64, i64>(const i64* src_ptr, i64* dst_ptr, usize count) noexcept
	{
		std::memcpy(dst_ptr, src_ptr, count * sizeof(i64));
	}

	template<> void cast_towardszero<i64, u8>(const i64* src_ptr, u8* dst_ptr, usize count) noexcept
	{
		for (usize i = 0; i < count; ++i) dst_ptr[i] = static_cast<u8>(src_ptr[i]);
	}

	template<> void cast_towardszero<i64, u16>(const i64* src_ptr, u16* dst_ptr, usize count) noexcept
	{
		for (usize i = 0; i < count; ++i) dst_ptr[i] = static_cast<u16>(src_ptr[i]);
	}

	template<> void cast_towardszero<i64, u32>(const i64* src_ptr, u32* dst_ptr, usize count) noexcept
	{
		for (usize i = 0; i < count; ++i) dst_ptr[i] = static_cast<u32>(src_ptr[i]);
	}

	template<> void cast_towardszero<i64, u64>(const i64* src_ptr, u64* dst_ptr, usize count) noexcept
	{
		for (usize i = 0; i < count; ++i) dst_ptr[i] = static_cast<u64>(src_ptr[i]);
	}

	template<> void cast_towardszero<i64, f16>(const i64* src_ptr, f16* dst_ptr, usize count) noexcept
	{
		for (usize i = 0; i < count; ++i) dst_ptr[i] = static_cast<f16>(src_ptr[i]);
	}

	template<> void cast_towardszero<i64, f32>(const i64* src_ptr, f32* dst_ptr, usize count) noexcept
	{
		for (usize i = 0; i < count; ++i) dst_ptr[i] = static_cast<f32>(src_ptr[i]);
	}

	template<> void cast_towardszero<i64, f64>(const i64* src_ptr, f64* dst_ptr, usize count) noexcept
	{
		for (usize i = 0; i < count; ++i) dst_ptr[i] = static_cast<f64>(src_ptr[i]);
	}
}

namespace fy
{
	template<> void cast_towardszero<f16, i8>(const f16* src_ptr, i8* dst_ptr, usize count) noexcept
	{
		for (usize i = 0; i < count; ++i) dst_ptr[i] = static_cast<i8>(static_cast<f32>(src_ptr[i]));
	}

	template<> void cast_towardszero<f16, i16>(const f16* src_ptr, i16* dst_ptr, usize count) noexcept
	{
		for (usize i = 0; i < count; ++i) dst_ptr[i] = static_cast<i16>(static_cast<f32>(src_ptr[i]));
	}

	template<> void cast_towardszero<f16, i32>(const f16* src_ptr, i32* dst_ptr, usize count) noexcept
	{
		for (usize i = 0; i < count; ++i) dst_ptr[i] = static_cast<i32>(static_cast<f32>(src_ptr[i]));
	}

	template<> void cast_towardszero<f16, i64>(const f16* src_ptr, i64* dst_ptr, usize count) noexcept
	{
		for (usize i = 0; i < count; ++i) dst_ptr[i] = static_cast<i64>(static_cast<f32>(src_ptr[i]));
	}

	template<> void cast_towardszero<f16, u8>(const f16* src_ptr, u8* dst_ptr, usize count) noexcept
	{
		for (usize i = 0; i < count; ++i) dst_ptr[i] = static_cast<u8>(static_cast<f32>(src_ptr[i]));
	}

	template<> void cast_towardszero<f16, u16>(const f16* src_ptr, u16* dst_ptr, usize count) noexcept
	{
		for (usize i = 0; i < count; ++i) dst_ptr[i] = static_cast<u16>(static_cast<f32>(src_ptr[i]));
	}

	template<> void cast_towardszero<f16, u32>(const f16* src_ptr, u32* dst_ptr, usize count) noexcept
	{
		for (usize i = 0; i < count; ++i) dst_ptr[i] = static_cast<u32>(static_cast<f32>(src_ptr[i]));
	}

	template<> void cast_towardszero<f16, u64>(const f16* src_ptr, u64* dst_ptr, usize count) noexcept
	{
		for (usize i = 0; i < count; ++i) dst_ptr[i] = static_cast<u64>(static_cast<f32>(src_ptr[i]));
	}

	template<> void cast_towardszero<f16, f16>(const f16* src_ptr, f16* dst_ptr, usize count) noexcept
	{
		for (usize i = 0; i < count; ++i) dst_ptr[i] = static_cast<f16>(static_cast<f32>(src_ptr[i]));
	}

	template<> void cast_towardszero<f16, f32>(const f16* src_ptr, f32* dst_ptr, usize count) noexcept
	{
		for (usize i = 0; i < count; ++i) dst_ptr[i] = static_cast<f32>(static_cast<f32>(src_ptr[i]));
	}

	template<> void cast_towardszero<f16, f64>(const f16* src_ptr, f64* dst_ptr, usize count) noexcept
	{
		for (usize i = 0; i < count; ++i) dst_ptr[i] = static_cast<f64>(static_cast<f32>(src_ptr[i]));
	}
}

namespace fy
{
	template<> void cast_towardszero<f32, i8>(const f32* src_ptr, i8* dst_ptr, usize count) noexcept
	{
		constexpr usize cache_line_size = 64;
		constexpr usize simd_stride = 16;
		constexpr usize prefetch_ahead = 4 * cache_line_size;

		usize i = 0;

		while (i < count && (reinterpret_cast<uintptr_t>(dst_ptr + i) % 16) != 0)
		{
			dst_ptr[i] = static_cast<i8>(src_ptr[i]);
			i++;
		}

		const f32* src_aligned = src_ptr + i;
		i8* dst_aligned = dst_ptr + i;
		const usize main_loop_count = (count - i) / simd_stride;
		const f32* main_loop_end = src_aligned + main_loop_count * simd_stride;

		const __m128i shuffle_mask = _mm_set_epi8(
			-1, -1, -1, -1,
			-1, -1, -1, -1,
			-1, -1, -1, -1,
			0x0C, 0x08, 0x04, 0x00
		);

		for (; src_aligned < main_loop_end; src_aligned += simd_stride, dst_aligned += simd_stride)
		{
			_mm_prefetch(reinterpret_cast<const char*>(src_aligned) + prefetch_ahead, _MM_HINT_T0);

			__m128 f0 = _mm_loadu_ps(src_aligned + 0);
			__m128 f1 = _mm_loadu_ps(src_aligned + 4);
			__m128 f2 = _mm_loadu_ps(src_aligned + 8);
			__m128 f3 = _mm_loadu_ps(src_aligned + 12);

			__m128i i0 = _mm_cvttps_epi32(f0);
			__m128i i1 = _mm_cvttps_epi32(f1);
			__m128i i2 = _mm_cvttps_epi32(f2);
			__m128i i3 = _mm_cvttps_epi32(f3);

			__m128i b0 = _mm_shuffle_epi8(i0, shuffle_mask);
			__m128i b1 = _mm_shuffle_epi8(i1, shuffle_mask);
			__m128i b2 = _mm_shuffle_epi8(i2, shuffle_mask);
			__m128i b3 = _mm_shuffle_epi8(i3, shuffle_mask);

			b1 = _mm_slli_si128(b1, 4);
			b2 = _mm_slli_si128(b2, 8);
			b3 = _mm_slli_si128(b3, 12);

			__m128i packed = _mm_or_si128(
				_mm_or_si128(b0, b1),
				_mm_or_si128(b2, b3));

			_mm_stream_si128(reinterpret_cast<__m128i*>(dst_aligned), packed);
		}

		usize processed = main_loop_count * simd_stride;
		usize remaining = count - i - processed;
		const f32* src_remaining = src_aligned;
		i8* dst_remaining = dst_aligned;

		while (remaining >= 8)
		{
			__m128 f0 = _mm_loadu_ps(src_remaining + 0);
			__m128 f1 = _mm_loadu_ps(src_remaining + 4);

			__m128i i0 = _mm_cvttps_epi32(f0);
			__m128i i1 = _mm_cvttps_epi32(f1);

			__m128i b0 = _mm_shuffle_epi8(i0, shuffle_mask);
			__m128i b1 = _mm_shuffle_epi8(i1, shuffle_mask);
			b1 = _mm_slli_si128(b1, 4);

			__m128i packed = _mm_or_si128(b0, b1);
			_mm_storel_epi64(reinterpret_cast<__m128i*>(dst_remaining), packed);

			src_remaining += 8;
			dst_remaining += 8;
			remaining -= 8;
		}

		if (remaining >= 4)
		{
			__m128 f0 = _mm_loadu_ps(src_remaining);
			__m128i i0 = _mm_cvttps_epi32(f0);
			__m128i b0 = _mm_shuffle_epi8(i0, shuffle_mask);
			*reinterpret_cast<i32*>(dst_remaining) = _mm_cvtsi128_si32(b0);

			src_remaining += 4;
			dst_remaining += 4;
			remaining -= 4;
		}

		while (remaining-- > 0)
		{
			*dst_remaining++ = static_cast<i8>(*src_remaining++);
		}

		_mm_sfence();
	}
	
	template<> void cast_towardszero<f32, i16>(const f32* src_ptr, i16* dst_ptr, usize count) noexcept
	{
		constexpr usize cache_line_size = 64;
		constexpr usize simd_stride = 8;
		constexpr usize prefetch_ahead = 4 * cache_line_size;

		usize i = 0;

		while (i < count && (reinterpret_cast<uintptr_t>(dst_ptr + i) % 16) != 0)
		{
			dst_ptr[i] = static_cast<i16>(src_ptr[i]);
			i++;
		}

		const f32* src_aligned = src_ptr + i;
		i16* dst_aligned = dst_ptr + i;
		const usize main_loop_count = (count - i) / simd_stride;
		const f32* main_loop_end = src_aligned + main_loop_count * simd_stride;

		const __m128i shuffle_mask = _mm_set_epi8(
			-1, -1, -1, -1, -1, -1, -1, -1,
			13, 12, 9, 8, 5, 4, 1, 0
		);

		for (; src_aligned < main_loop_end; src_aligned += simd_stride, dst_aligned += simd_stride)
		{
			_mm_prefetch(reinterpret_cast<const char*>(src_aligned) + prefetch_ahead, _MM_HINT_T0);

			__m128 f0 = _mm_loadu_ps(src_aligned + 0);
			__m128 f1 = _mm_loadu_ps(src_aligned + 4);
			__m128i i0 = _mm_cvttps_epi32(f0);
			__m128i i1 = _mm_cvttps_epi32(f1);

			__m128i p0 = _mm_shuffle_epi8(i0, shuffle_mask);
			__m128i p1 = _mm_shuffle_epi8(i1, shuffle_mask);
			p1 = _mm_slli_si128(p1, 8);
			__m128i packed = _mm_or_si128(p0, p1);

			_mm_stream_si128(reinterpret_cast<__m128i*>(dst_aligned), packed);
		}

		usize processed = main_loop_count * simd_stride;
		usize remaining = count - i - processed;
		const f32* src_remaining = src_aligned;
		i16* dst_remaining = dst_aligned;

		if (remaining >= 4)
		{
			__m128 f = _mm_loadu_ps(src_remaining);
			__m128i i32s = _mm_cvttps_epi32(f);
			__m128i packed = _mm_shuffle_epi8(i32s, shuffle_mask);
			_mm_storel_epi64(reinterpret_cast<__m128i*>(dst_remaining), packed);
			src_remaining += 4;
			dst_remaining += 4;
			remaining -= 4;
		}

		if (remaining >= 2)
		{
			__m128 f = _mm_loadl_pi(_mm_undefined_ps(), (const __m64*)src_remaining);
			__m128i i32s = _mm_cvttps_epi32(f);
			__m128i packed = _mm_shuffle_epi8(i32s, shuffle_mask);
			*reinterpret_cast<i32*>(dst_remaining) = _mm_cvtsi128_si32(packed);
			src_remaining += 2;
			dst_remaining += 2;
			remaining -= 2;
		}

		while (remaining-- > 0)
		{
			*dst_remaining++ = static_cast<i16>(*src_remaining++);
		}

		_mm_sfence();
	}

	template<> void cast_towardszero<f32, i32>(const f32* src_ptr, i32* dst_ptr, usize count) noexcept
	{
		constexpr usize cache_line_size = 64;
		constexpr usize simd_stride = 8;
		constexpr usize prefetch_ahead = 4 * cache_line_size;

		usize i = 0;

		while (i < count && (reinterpret_cast<uintptr_t>(dst_ptr + i) % 32) != 0)
		{
			dst_ptr[i] = static_cast<i32>(src_ptr[i]);
			i++;
		}

		const f32* src_aligned = src_ptr + i;
		i32* dst_aligned = dst_ptr + i;
		const usize main_loop_count = (count - i) / simd_stride;
		const f32* main_loop_end = src_aligned + main_loop_count * simd_stride;

		for (; src_aligned < main_loop_end; src_aligned += simd_stride, dst_aligned += simd_stride)
		{
			_mm_prefetch(reinterpret_cast<const char*>(src_aligned) + prefetch_ahead, _MM_HINT_T0);

			__m256 f = _mm256_loadu_ps(src_aligned);
			__m256i i32_vec = _mm256_cvttps_epi32(f);

			_mm256_stream_si256(reinterpret_cast<__m256i*>(dst_aligned), i32_vec);
		}

		usize processed = main_loop_count * simd_stride;
		usize remaining = count - i - processed;
		const f32* src_remaining = src_aligned;
		i32* dst_remaining = dst_aligned;

		if (remaining >= 4)
		{
			__m128 f = _mm_loadu_ps(src_remaining);
			__m128i i32s = _mm_cvttps_epi32(f);
			_mm_storeu_si128(reinterpret_cast<__m128i*>(dst_remaining), i32s);
			src_remaining += 4;
			dst_remaining += 4;
			remaining -= 4;
		}

		if (remaining >= 2)
		{
			__m128 f = _mm_loadl_pi(_mm_undefined_ps(), (const __m64*)src_remaining);
			__m128i i32s = _mm_cvttps_epi32(f);
			_mm_storel_epi64(reinterpret_cast<__m128i*>(dst_remaining), i32s);
			src_remaining += 2;
			dst_remaining += 2;
			remaining -= 2;
		}

		if (remaining > 0)
		{
			*dst_remaining = static_cast<i32>(*src_remaining);
		}

		_mm_sfence();
	}

	template<> void cast_towardszero<f32, i64>(const f32* src_ptr, i64* dst_ptr, usize count) noexcept
	{
		for (usize i = 0; i < count; ++i) dst_ptr[i] = static_cast<i64>(src_ptr[i]);
	}

	template<> void cast_towardszero<f32, u8>(const f32* src_ptr, u8* dst_ptr, usize count) noexcept
	{
		constexpr usize cache_line_size = 64;
		constexpr usize simd_stride = 16;
		constexpr usize prefetch_ahead = 4 * cache_line_size;

		usize i = 0;

		while (i < count && (reinterpret_cast<uintptr_t>(dst_ptr + i) % 16) != 0)
		{
			dst_ptr[i] = static_cast<u8>(src_ptr[i]);
			i++;
		}

		const f32* src_aligned = src_ptr + i;
		u8* dst_aligned = dst_ptr + i;
		const usize main_loop_count = (count - i) / simd_stride;
		const f32* main_loop_end = src_aligned + main_loop_count * simd_stride;

		const __m128i shuffle_mask = _mm_set_epi8(
			-1, -1, -1, -1, -1, -1, -1, -1,
			-1, -1, -1, -1, 0x0C, 0x08, 0x04, 0x00
		);

		for (; src_aligned < main_loop_end; src_aligned += simd_stride, dst_aligned += simd_stride)
		{
			_mm_prefetch(reinterpret_cast<const char*>(src_aligned) + prefetch_ahead, _MM_HINT_T0);

			__m128 f0 = _mm_loadu_ps(src_aligned + 0);
			__m128 f1 = _mm_loadu_ps(src_aligned + 4);
			__m128 f2 = _mm_loadu_ps(src_aligned + 8);
			__m128 f3 = _mm_loadu_ps(src_aligned + 12);

			__m128i i0 = _mm_cvttps_epi32(f0);
			__m128i i1 = _mm_cvttps_epi32(f1);
			__m128i i2 = _mm_cvttps_epi32(f2);
			__m128i i3 = _mm_cvttps_epi32(f3);

			__m128i b0 = _mm_shuffle_epi8(i0, shuffle_mask);
			__m128i b1 = _mm_shuffle_epi8(i1, shuffle_mask);
			__m128i b2 = _mm_shuffle_epi8(i2, shuffle_mask);
			__m128i b3 = _mm_shuffle_epi8(i3, shuffle_mask);

			b1 = _mm_slli_si128(b1, 4);
			b2 = _mm_slli_si128(b2, 8);
			b3 = _mm_slli_si128(b3, 12);

			__m128i packed = _mm_or_si128(_mm_or_si128(b0, b1),
				_mm_or_si128(b2, b3));

			_mm_stream_si128(reinterpret_cast<__m128i*>(dst_aligned), packed);
		}

		usize processed = main_loop_count * simd_stride;
		usize remaining = count - i - processed;
		const f32* src_remaining = src_aligned;
		u8* dst_remaining = dst_aligned;

		while (remaining >= 8)
		{
			__m128 f0 = _mm_loadu_ps(src_remaining);
			__m128 f1 = _mm_loadu_ps(src_remaining + 4);

			__m128i i0 = _mm_cvttps_epi32(f0);
			__m128i i1 = _mm_cvttps_epi32(f1);

			__m128i b0 = _mm_shuffle_epi8(i0, shuffle_mask);
			__m128i b1 = _mm_shuffle_epi8(i1, shuffle_mask);
			b1 = _mm_slli_si128(b1, 4);

			_mm_storel_epi64(reinterpret_cast<__m128i*>(dst_remaining),
				_mm_or_si128(b0, b1));

			src_remaining += 8;
			dst_remaining += 8;
			remaining -= 8;
		}

		if (remaining >= 4)
		{
			__m128 f = _mm_loadu_ps(src_remaining);
			__m128i i32s = _mm_cvttps_epi32(f);
			__m128i packed = _mm_shuffle_epi8(i32s, shuffle_mask);
			*reinterpret_cast<i32*>(dst_remaining) = _mm_cvtsi128_si32(packed);
			src_remaining += 4;
			dst_remaining += 4;
			remaining -= 4;
		}

		while (remaining-- > 0)
		{
			*dst_remaining++ = static_cast<u8>(*src_remaining++);
		}

		_mm_sfence();
	}

	template<> void cast_towardszero<f32, u16>(const f32* src_ptr, u16* dst_ptr, usize count) noexcept
	{
		constexpr usize cache_line_size = 64;
		constexpr usize simd_stride = 8;
		constexpr usize prefetch_ahead = 4 * cache_line_size;

		usize i = 0;

		while (i < count && (reinterpret_cast<uintptr_t>(dst_ptr + i) % 16) != 0)
		{
			dst_ptr[i] = static_cast<u16>(src_ptr[i]);
			i++;
		}

		const f32* src_aligned = src_ptr + i;
		u16* dst_aligned = dst_ptr + i;
		const usize main_loop_count = (count - i) / simd_stride;
		const f32* main_loop_end = src_aligned + main_loop_count * simd_stride;

		const __m128i shuffle_mask = _mm_set_epi8(
			-1, -1, -1, -1, -1, -1, -1, -1,
			13, 12, 9, 8, 5, 4, 1, 0
		);

		for (; src_aligned < main_loop_end; src_aligned += simd_stride, dst_aligned += simd_stride)
		{
			_mm_prefetch(reinterpret_cast<const char*>(src_aligned) + prefetch_ahead, _MM_HINT_T0);

			__m128 f0 = _mm_loadu_ps(src_aligned + 0);
			__m128 f1 = _mm_loadu_ps(src_aligned + 4);

			__m128i i0 = _mm_cvttps_epi32(f0);
			__m128i i1 = _mm_cvttps_epi32(f1);

			__m128i p0 = _mm_shuffle_epi8(i0, shuffle_mask);
			__m128i p1 = _mm_shuffle_epi8(i1, shuffle_mask);
			p1 = _mm_slli_si128(p1, 8);
			__m128i packed = _mm_or_si128(p0, p1);

			_mm_stream_si128(reinterpret_cast<__m128i*>(dst_aligned), packed);
		}

		usize processed = main_loop_count * simd_stride;
		usize remaining = count - i - processed;
		const f32* src_remaining = src_aligned;
		u16* dst_remaining = dst_aligned;

		if (remaining >= 4)
		{
			__m128 f = _mm_loadu_ps(src_remaining);
			__m128i i32s = _mm_cvttps_epi32(f);
			__m128i packed = _mm_shuffle_epi8(i32s, shuffle_mask);
			_mm_storel_epi64(reinterpret_cast<__m128i*>(dst_remaining), packed);
			src_remaining += 4;
			dst_remaining += 4;
			remaining -= 4;
		}

		if (remaining >= 2)
		{
			__m128 f = _mm_loadl_pi(_mm_undefined_ps(), (const __m64*)src_remaining);
			__m128i i32s = _mm_cvttps_epi32(f);
			__m128i packed = _mm_shuffle_epi8(i32s, shuffle_mask);
			*reinterpret_cast<i32*>(dst_remaining) = _mm_cvtsi128_si32(packed);
			src_remaining += 2;
			dst_remaining += 2;
			remaining -= 2;
		}

		while (remaining-- > 0)
		{
			*dst_remaining++ = static_cast<u16>(*src_remaining++);
		}

		_mm_sfence();
	}


	template<> void cast_towardszero<f32, u32>(const f32* src_ptr, u32* dst_ptr, usize count) noexcept
	{
		constexpr usize cache_line_size = 64;
		constexpr usize simd_stride = 8;
		constexpr usize prefetch_ahead = 4 * cache_line_size;

		usize i = 0;

		while (i < count && (reinterpret_cast<uintptr_t>(dst_ptr + i) % 32) != 0)
		{
			dst_ptr[i] = static_cast<u32>(src_ptr[i]);
			i++;
		}

		const f32* src_aligned = src_ptr + i;
		u32* dst_aligned = dst_ptr + i;
		const usize main_loop_count = (count - i) / simd_stride;
		const f32* main_loop_end = src_aligned + main_loop_count * simd_stride;

		for (; src_aligned < main_loop_end; src_aligned += simd_stride, dst_aligned += simd_stride)
		{
			_mm_prefetch(reinterpret_cast<const char*>(src_aligned) + prefetch_ahead, _MM_HINT_T0);

			__m256 f = _mm256_loadu_ps(src_aligned);
			__m256i u32_vec = _mm256_cvttps_epi32(f);

			_mm256_stream_si256(reinterpret_cast<__m256i*>(dst_aligned), u32_vec);
		}

		usize processed = main_loop_count * simd_stride;
		usize remaining = count - i - processed;
		const f32* src_remaining = src_aligned;
		u32* dst_remaining = dst_aligned;

		if (remaining >= 4)
		{
			__m128 f = _mm_loadu_ps(src_remaining);
			__m128i u32s = _mm_cvttps_epi32(f);
			_mm_storeu_si128(reinterpret_cast<__m128i*>(dst_remaining), u32s);
			src_remaining += 4;
			dst_remaining += 4;
			remaining -= 4;
		}

		if (remaining >= 2)
		{
			__m128 f = _mm_loadl_pi(_mm_undefined_ps(), (const __m64*)src_remaining);
			__m128i u32s = _mm_cvttps_epi32(f);
			_mm_storel_epi64(reinterpret_cast<__m128i*>(dst_remaining), u32s);
			src_remaining += 2;
			dst_remaining += 2;
			remaining -= 2;
		}

		if (remaining > 0)
		{
			*dst_remaining = static_cast<u32>(*src_remaining);
		}

		_mm_sfence();
	}

	template<> void cast_towardszero<f32, u64>(const f32* src_ptr, u64* dst_ptr, usize count) noexcept
	{
		constexpr usize cache_line_size = 64;
		constexpr usize simd_stride = 4;
		constexpr usize prefetch_ahead = 4 * cache_line_size;

		usize i = 0;

		while (i < count && (reinterpret_cast<uintptr_t>(dst_ptr + i) % 32) != 0)
		{
			dst_ptr[i] = static_cast<u64>(src_ptr[i]);
			i++;
		}

		const f32* src_aligned = src_ptr + i;
		u64* dst_aligned = dst_ptr + i;
		const usize main_loop_count = (count - i) / simd_stride;
		const f32* main_loop_end = src_aligned + main_loop_count * simd_stride;

		for (; src_aligned < main_loop_end; src_aligned += simd_stride, dst_aligned += simd_stride)
		{
			_mm_prefetch(reinterpret_cast<const char*>(src_aligned) + prefetch_ahead, _MM_HINT_T0);

			__m128 f32x4 = _mm_loadu_ps(src_aligned);

			__m128 low2 = _mm_shuffle_ps(f32x4, f32x4, _MM_SHUFFLE(1, 0, 1, 0));
			__m128 high2 = _mm_shuffle_ps(f32x4, f32x4, _MM_SHUFFLE(3, 2, 3, 2));

			__m128d low2_f64 = _mm_cvtps_pd(low2);
			__m128d high2_f64 = _mm_cvtps_pd(high2);

			__m128i i64_low = _mm_cvttpd_epi64(low2_f64);
			__m128i i64_high = _mm_cvttpd_epi64(high2_f64);

			__m256i u64_vec = _mm256_setr_m128i(i64_low, i64_high);
			_mm256_stream_si256(reinterpret_cast<__m256i*>(dst_aligned), u64_vec);
		}

		usize processed = main_loop_count * simd_stride;
		usize remaining = count - i - processed;
		const f32* src_remaining = src_aligned;
		u64* dst_remaining = dst_aligned;

		if (remaining >= 2)
		{
			__m128 two_f32 = _mm_castpd_ps(_mm_load_sd(reinterpret_cast<const double*>(src_remaining)));
			__m128d two_f64 = _mm_cvtps_pd(two_f32);
			__m128i two_i64 = _mm_cvttpd_epi64(two_f64);
			_mm_storeu_si128(reinterpret_cast<__m128i*>(dst_remaining), two_i64);
			src_remaining += 2;
			dst_remaining += 2;
			remaining -= 2;
		}

		if (remaining > 0)
		{
			*dst_remaining = static_cast<u64>(*src_remaining);
		}

		_mm_sfence();
	}

	template<> void cast_towardszero<f32, f16>(const f32* src_ptr, f16* dst_ptr, usize count) noexcept
	{
		constexpr usize cache_line_size = 64;
		constexpr usize simd_stride = 8;
		constexpr usize prefetch_ahead = 4 * cache_line_size;

		usize i = 0;

		while (i < count && (reinterpret_cast<uintptr_t>(dst_ptr + i) % 16) != 0)
		{
			dst_ptr[i] = static_cast<f16>(src_ptr[i]);
			i++;
		}

		const f32* src_aligned = src_ptr + i;
		f16* dst_aligned = dst_ptr + i;
		const usize main_loop_count = (count - i) / simd_stride;
		const f32* main_loop_end = src_aligned + main_loop_count * simd_stride;

		for (; src_aligned < main_loop_end; src_aligned += simd_stride, dst_aligned += simd_stride)
		{
			_mm_prefetch(reinterpret_cast<const char*>(src_aligned) + prefetch_ahead, _MM_HINT_T0);

			__m256 f32_vec = _mm256_loadu_ps(src_aligned);
			__m128i f16_packed = _mm256_cvtps_ph(f32_vec, _MM_FROUND_TO_NEAREST_INT);
			_mm_stream_si128(reinterpret_cast<__m128i*>(dst_aligned), f16_packed);
		}

		usize processed = main_loop_count * simd_stride;
		usize remaining = count - i - processed;
		const f32* src_remaining = src_aligned;
		f16* dst_remaining = dst_aligned;

		if (remaining >= 4)
		{
			__m128 f32_vec = _mm_loadu_ps(src_remaining);
			__m128i f16_packed = _mm_cvtps_ph(f32_vec, _MM_FROUND_TO_NEAREST_INT);
			_mm_storel_epi64(reinterpret_cast<__m128i*>(dst_remaining), f16_packed);
			src_remaining += 4;
			dst_remaining += 4;
			remaining -= 4;
		}

		if (remaining >= 2)
		{
			__m128 f32_vec = _mm_loadl_pi(_mm_undefined_ps(), (const __m64*)src_remaining);
			__m128i f16_packed = _mm_cvtps_ph(f32_vec, _MM_FROUND_TO_NEAREST_INT);
			*reinterpret_cast<int*>(dst_remaining) = _mm_extract_epi32(f16_packed, 0);
			src_remaining += 2;
			dst_remaining += 2;
			remaining -= 2;
		}

		while (remaining-- > 0)
		{
			*dst_remaining++ = static_cast<f16>(*src_remaining++);
		}

		_mm_sfence();
	}

	template<> void cast_towardszero<f32, f32>(const f32* src_ptr, f32* dst_ptr, usize count) noexcept
	{
		std::memcpy(dst_ptr, src_ptr, count * sizeof(f32));
	}

	template<> void cast_towardszero<f32, f64>(const f32* src_ptr, f64* dst_ptr, usize count) noexcept
	{
		constexpr usize cache_line_size = 64;
		constexpr usize simd_stride = 4;
		constexpr usize prefetch_ahead = 4 * cache_line_size;

		usize i = 0;

		while (i < count && (reinterpret_cast<uintptr_t>(dst_ptr + i) % 32) != 0)
		{
			dst_ptr[i] = static_cast<f64>(src_ptr[i]);
			i++;
		}

		const f32* src_aligned = src_ptr + i;
		f64* dst_aligned = dst_ptr + i;
		const usize main_loop_count = (count - i) / simd_stride;
		const f32* main_loop_end = src_aligned + main_loop_count * simd_stride;

		for (; src_aligned < main_loop_end; src_aligned += simd_stride, dst_aligned += simd_stride)
		{
			_mm_prefetch(reinterpret_cast<const char*>(src_aligned) + prefetch_ahead, _MM_HINT_T0);

			__m128 f32_vec = _mm_loadu_ps(src_aligned);
			__m256d f64_vec = _mm256_cvtps_pd(f32_vec);
			_mm256_stream_pd(dst_aligned, f64_vec);
		}

		usize processed = main_loop_count * simd_stride;
		usize remaining = count - i - processed;
		const f32* src_remaining = src_aligned;
		f64* dst_remaining = dst_aligned;

		if (remaining >= 2)
		{
			__m128 f32_vec = _mm_loadl_pi(_mm_undefined_ps(), (const __m64*)src_remaining);
			__m128d f64_vec = _mm_cvtps_pd(f32_vec);
			_mm_stream_pd(dst_remaining, f64_vec);
			src_remaining += 2;
			dst_remaining += 2;
			remaining -= 2;
		}

		while (remaining-- > 0)
		{
			*dst_remaining++ = static_cast<f64>(*src_remaining++);
		}

		_mm_sfence();
	}
}

namespace fy
{
	template<> void cast_towardszero<f64, i8>(const f64* src_ptr, i8* dst_ptr, usize count) noexcept
	{
		constexpr usize cache_line_size = 64;
		constexpr usize simd_stride = 4;
		constexpr usize prefetch_ahead = 4 * cache_line_size;

		usize i = 0;

		while (i < count && (reinterpret_cast<uintptr_t>(dst_ptr + i) % 32) != 0)
		{
			dst_ptr[i] = static_cast<i8>(src_ptr[i]);
			i++;
		}

		const f64* src_aligned = src_ptr + i;
		i8* dst_aligned = dst_ptr + i;
		const usize main_loop_count = (count - i) / simd_stride;
		const f64* main_loop_end = src_aligned + main_loop_count * simd_stride;

		const __m256i shuffle_mask = _mm256_set_epi8(
			-1, -1, -1, -1, -1, -1, -1, -1,
			-1, -1, -1, -1, 0x0C, 0x08, 0x04, 0x00,
			-1, -1, -1, -1, -1, -1, -1, -1,
			-1, -1, -1, -1, 0x0C, 0x08, 0x04, 0x00
		);

		for (; src_aligned < main_loop_end; src_aligned += simd_stride, dst_aligned += simd_stride)
		{
			_mm_prefetch(reinterpret_cast<const char*>(src_aligned) + prefetch_ahead, _MM_HINT_T0);

			__m256d d0 = _mm256_loadu_pd(src_aligned);
			__m128i i32s = _mm256_cvttpd_epi32(d0);

			__m256i shuffled = _mm256_shuffle_epi8(_mm256_castsi128_si256(i32s), shuffle_mask);
			__m128i packed = _mm_or_si128(_mm256_castsi256_si128(shuffled),
				_mm256_extracti128_si256(shuffled, 1));

			_mm_stream_si32(reinterpret_cast<int*>(dst_aligned), _mm_cvtsi128_si32(packed));
		}

		usize remaining = count - i - main_loop_count * simd_stride;
		for (usize j = 0; j < remaining; ++j)
		{
			dst_aligned[j] = static_cast<i8>(src_aligned[j]);
		}
		_mm_sfence();
	}

	template<> void cast_towardszero<f64, i16>(const f64* src_ptr, i16* dst_ptr, usize count) noexcept
	{
		constexpr usize cache_line_size = 64;
		constexpr usize simd_stride = 4;
		constexpr usize prefetch_ahead = 4 * cache_line_size;

		usize i = 0;

		while (i < count && (reinterpret_cast<uintptr_t>(dst_ptr + i) % 32) != 0)
		{
			dst_ptr[i] = static_cast<i16>(src_ptr[i]);
			i++;
		}

		const f64* src_aligned = src_ptr + i;
		i16* dst_aligned = dst_ptr + i;
		const usize main_loop_count = (count - i) / simd_stride;
		const f64* main_loop_end = src_aligned + main_loop_count * simd_stride;

		const __m128i shuffle_mask = _mm_set_epi8(
			-1, -1, -1, -1, -1, -1, -1, -1,
			13, 12, 9, 8, 5, 4, 1, 0
		);

		for (; src_aligned < main_loop_end; src_aligned += simd_stride, dst_aligned += simd_stride)
		{
			_mm_prefetch(reinterpret_cast<const char*>(src_aligned) + prefetch_ahead, _MM_HINT_T0);

			__m256d d0 = _mm256_loadu_pd(src_aligned);
			__m128i i32s = _mm256_cvttpd_epi32(d0);

			__m128i packed_i16 = _mm_shuffle_epi8(i32s, shuffle_mask);
			uint64_t result = _mm_cvtsi128_si64(packed_i16);

			_mm_stream_si64(reinterpret_cast<long long*>(dst_aligned), result);
		}

		usize remaining = count - i - main_loop_count * simd_stride;
		for (usize j = 0; j < remaining; ++j)
		{
			dst_aligned[j] = static_cast<i16>(src_aligned[j]);
		}

		_mm_sfence();
	}

	template<> void cast_towardszero<f64, i32>(const f64* src_ptr, i32* dst_ptr, usize count) noexcept
	{
		constexpr usize cache_line_size = 64;
		constexpr usize simd_stride = 4;
		constexpr usize prefetch_ahead = 4 * cache_line_size;

		usize i = 0;

		while (i < count && (reinterpret_cast<uintptr_t>(dst_ptr + i) % 32) != 0)
		{
			dst_ptr[i] = static_cast<i32>(src_ptr[i]);
			i++;
		}

		const f64* src_aligned = src_ptr + i;
		i32* dst_aligned = dst_ptr + i;
		const usize main_loop_count = (count - i) / simd_stride;
		const f64* main_loop_end = src_aligned + main_loop_count * simd_stride;

		for (; src_aligned < main_loop_end; src_aligned += simd_stride, dst_aligned += simd_stride)
		{
			_mm_prefetch(reinterpret_cast<const char*>(src_aligned) + prefetch_ahead, _MM_HINT_T0);

			__m256d d0 = _mm256_loadu_pd(src_aligned);
			__m128i i32s = _mm256_cvttpd_epi32(d0);

			_mm_stream_si128(reinterpret_cast<__m128i*>(dst_aligned), i32s);
		}

		usize remaining = count - i - main_loop_count * simd_stride;
		for (usize j = 0; j < remaining; ++j)
		{
			dst_aligned[j] = static_cast<i32>(src_aligned[j]);
		}
		_mm_sfence();
	}

	template<> void cast_towardszero<f64, i64>(const f64* src_ptr, i64* dst_ptr, usize count) noexcept
	{
		for (usize i = 0; i < count; ++i) dst_ptr[i] = static_cast<i64>(src_ptr[i]);
	}

	template<> void cast_towardszero<f64, u8>(const f64* src_ptr, u8* dst_ptr, usize count) noexcept
	{
		for (usize i = 0; i < count; ++i) dst_ptr[i] = static_cast<u8>(src_ptr[i]);
	}

	template<> void cast_towardszero<f64, u16>(const f64* src_ptr, u16* dst_ptr, usize count) noexcept
	{
		for (usize i = 0; i < count; ++i) dst_ptr[i] = static_cast<u16>(src_ptr[i]);
	}

	template<> void cast_towardszero<f64, u32>(const f64* src_ptr, u32* dst_ptr, usize count) noexcept
	{
		for (usize i = 0; i < count; ++i) dst_ptr[i] = static_cast<u32>(src_ptr[i]);
	}

	template<> void cast_towardszero<f64, u64>(const f64* src_ptr, u64* dst_ptr, usize count) noexcept
	{
		for (usize i = 0; i < count; ++i) dst_ptr[i] = static_cast<u64>(src_ptr[i]);
	}

	template<> void cast_towardszero<f64, f16>(const f64* src_ptr, f16* dst_ptr, usize count) noexcept
	{
		for (usize i = 0; i < count; ++i) dst_ptr[i] = static_cast<f16>(src_ptr[i]);
	}

	template<> void cast_towardszero<f64, f32>(const f64* src_ptr, f32* dst_ptr, usize count) noexcept
	{
		for (usize i = 0; i < count; ++i) dst_ptr[i] = static_cast<f32>(src_ptr[i]);
	}

	template<> void cast_towardszero<f64, f64>(const f64* src_ptr, f64* dst_ptr, usize count) noexcept
	{
		for (usize i = 0; i < count; ++i) dst_ptr[i] = static_cast<f64>(src_ptr[i]);
	}
}
