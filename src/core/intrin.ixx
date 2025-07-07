module;
#include <Windows.h>
#undef min
#undef max

#include <immintrin.h>

export module foye.simd;

import foye.foye_core;
import std;

#pragma warning(disable: 4309)
#pragma warning(disable: 4244)
#pragma warning(disable: 4552)
#pragma warning(disable: 4552)
#pragma warning(disable: 4018)

export namespace fy
{
	template<BasicArithmetic T>
	i32 round(T value)
	{
		if constexpr (std::is_same_v<T, f64>)
		{
			return _mm_cvtsd_si32(_mm_set_sd(value));
		}
		else if constexpr (std::is_same_v<T, f32>)
		{
			return _mm_cvt_ss2si(_mm_set_ss(value));
		}
		else if constexpr (std::is_same_v<T, f16>)
		{
			return _mm_cvt_ss2si(_mm_set_ss(static_cast<f32>(value)));
		}
		else
		{
			static_assert(false, "Only supported for floatting type");
		}
	}

	template<BasicArithmetic T> T saturate_cast(u8 v) { return T(v); }
	template<BasicArithmetic T> T saturate_cast(i8 v) { return T(v); }

	template<BasicArithmetic T> T saturate_cast(u16 v) { return T(v); }
	template<BasicArithmetic T> T saturate_cast(i16 v) { return T(v); }

	template<BasicArithmetic T> T saturate_cast(u32 v) { return T(v); }
	template<BasicArithmetic T> T saturate_cast(i32 v) { return T(v); }

	template<BasicArithmetic T> T saturate_cast(f32 v) { return T(v); }
	template<BasicArithmetic T> T saturate_cast(f64 v) { return T(v); }

	template<BasicArithmetic T> T saturate_cast(i64 v) { return T(v); }
	template<BasicArithmetic T> T saturate_cast(u64 v) { return T(v); }

	template<BasicArithmetic T> T saturate_cast(f16 v) { return saturate_cast<T>(static_cast<f32>(v)); }

	template<>  u8 saturate_cast<u8>(i8 v) { return static_cast<u8>(std::max(static_cast<int>(v), 0)); }
	template<>  u8 saturate_cast<u8>(u16 v) { return static_cast<u8>(std::min(static_cast<u32>(v), static_cast<u32>(std::numeric_limits<u8>::max()))); }
	template<>  u8 saturate_cast<u8>(i32 v) { return static_cast<u8>((static_cast<u32>(v) <= std::numeric_limits<u8>::max()) ? v : (v > 0 ? std::numeric_limits<u8>::max() : 0)); }
	template<>  u8 saturate_cast<u8>(i16 v) { return saturate_cast<u8>(static_cast<int>(v)); }
	template<>  u8 saturate_cast<u8>(u32 v) { return static_cast<u8>(std::min(v, static_cast<u32>(std::numeric_limits<u8>::max()))); }
	template<>  u8 saturate_cast<u8>(f32 v) { return saturate_cast<u8>(fy::round(v)); }
	template<>  u8 saturate_cast<u8>(f64 v) { return saturate_cast<u8>(fy::round(v)); }
	template<>  u8 saturate_cast<u8>(i64 v) { return static_cast<u8>((static_cast<u64>(v) <= static_cast<u64>(std::numeric_limits<u8>::max())) ? v : (v > 0 ? std::numeric_limits<u8>::max() : 0)); }
	template<>  u8 saturate_cast<u8>(u64 v) { return static_cast<u8>(std::min(v, static_cast<u64>(std::numeric_limits<u8>::max()))); }

	template<>  i8 saturate_cast<i8>(u8 v) { return static_cast<i8>(std::min(static_cast<i32>(v), static_cast<i32>(std::numeric_limits<i8>::max()))); }
	template<>  i8 saturate_cast<i8>(u16 v) { return static_cast<i8>(std::min(static_cast<u32>(v), static_cast<u32>(std::numeric_limits<i8>::max()))); }
	template<>  i8 saturate_cast<i8>(i32 v) { return static_cast<i8>((static_cast<u32>(v - std::numeric_limits<i8>::lowest()) <= static_cast<u32>(std::numeric_limits<u8>::max())) ? v : (v > 0 ? std::numeric_limits<i8>::max() : std::numeric_limits<i8>::lowest())); }
	template<>  i8 saturate_cast<i8>(i16 v) { return saturate_cast<i8>(static_cast<i32>(v)); }
	template<>  i8 saturate_cast<i8>(u32 v) { return static_cast<i8>(std::min(v, static_cast<u32>(std::numeric_limits<i8>::max()))); }
	template<>  i8 saturate_cast<i8>(f32 v) { return saturate_cast<i8>(fy::round(v)); }
	template<>  i8 saturate_cast<i8>(f64 v) { return saturate_cast<i8>(fy::round(v)); }
	template<>  i8 saturate_cast<i8>(i64 v) { return static_cast<i8>((static_cast<u64>(static_cast<i64>(v) - std::numeric_limits<i8>::lowest()) <= static_cast<u64>(std::numeric_limits<u8>::max())) ? v : (v > 0 ? std::numeric_limits<i8>::max() : std::numeric_limits<i8>::lowest())); }
	template<>  i8 saturate_cast<i8>(u64 v) { return static_cast<i8>(std::min(v, static_cast<u64>(std::numeric_limits<i8>::max()))); }

	template<>  u16 saturate_cast<u16>(i8 v) { return static_cast<u16>(std::max(static_cast<i32>(v), 0)); }
	template<>  u16 saturate_cast<u16>(i16 v) { return static_cast<u16>(std::max(static_cast<i32>(v), 0)); }
	template<>  u16 saturate_cast<u16>(i32 v) { return static_cast<u16>((static_cast<u32>(v) <= static_cast<u32>(std::numeric_limits<u16>::max())) ? v : (v > 0 ? std::numeric_limits<u16>::max() : 0)); }
	template<>  u16 saturate_cast<u16>(u32 v) { return static_cast<u16>(std::min(v, static_cast<u32>(std::numeric_limits<u16>::max()))); }
	template<>  u16 saturate_cast<u16>(f32 v) { return saturate_cast<u16>(fy::round(v)); }
	template<>  u16 saturate_cast<u16>(f64 v) { return saturate_cast<u16>(fy::round(v)); }
	template<>  u16 saturate_cast<u16>(i64 v) { return static_cast<u16>((static_cast<u64>(v) <= static_cast<u64>(std::numeric_limits<u16>::max())) ? v : (v > 0 ? std::numeric_limits<u16>::max() : 0)); }
	template<>  u16 saturate_cast<u16>(u64 v) { return static_cast<u16>(std::min(v, static_cast<u64>(std::numeric_limits<u16>::max()))); }

	template<>  i16 saturate_cast<i16>(u16 v) { return static_cast<i16>(std::min(static_cast<i32>(v), static_cast<i32>(std::numeric_limits<i16>::max()))); }
	template<>  i16 saturate_cast<i16>(i32 v) { return static_cast<i16>((static_cast<u32>(v - std::numeric_limits<i16>::lowest()) <= static_cast<u32>(std::numeric_limits<u16>::max())) ? v : (v > 0 ? std::numeric_limits<i16>::max() : std::numeric_limits<i16>::lowest())); }
	template<>  i16 saturate_cast<i16>(u32 v) { return static_cast<i16>(std::min(v, static_cast<u32>(std::numeric_limits<i16>::max()))); }
	template<>  i16 saturate_cast<i16>(f32 v) { return saturate_cast<i16>(fy::round(v)); }
	template<>  i16 saturate_cast<i16>(f64 v) { return saturate_cast<i16>(fy::round(v)); }
	template<>  i16 saturate_cast<i16>(i64 v) { return static_cast<i16>((static_cast<u64>(static_cast<i64>(v) - std::numeric_limits<i16>::lowest()) <= static_cast<u64>(std::numeric_limits<u16>::max())) ? v : (v > 0 ? std::numeric_limits<i16>::max() : std::numeric_limits<i16>::lowest())); }
	template<>  i16 saturate_cast<i16>(u64 v) { return static_cast<i16>(std::min(v, static_cast<u64>(std::numeric_limits<i16>::max()))); }

	template<>  i32 saturate_cast<i32>(u32 v) { return static_cast<i32>(std::min(v, static_cast<u32>(std::numeric_limits<i32>::max()))); }
	template<>  i32 saturate_cast<i32>(i64 v) { return static_cast<i32>((static_cast<u64>(v - std::numeric_limits<i32>::lowest()) <= static_cast<u64>(std::numeric_limits<u32>::max())) ? v : (v > 0 ? std::numeric_limits<i32>::max() : std::numeric_limits<i32>::lowest())); }
	template<>  i32 saturate_cast<i32>(u64 v) { return static_cast<i32>(std::min(v, static_cast<u64>(std::numeric_limits<i32>::max()))); }
	template<>  i32 saturate_cast<i32>(f32 v) { return fy::round(v); }
	template<>  i32 saturate_cast<i32>(f64 v) { return fy::round(v); }

	template<>  u32 saturate_cast<u32>(i8 v) { return static_cast<u32>(std::max(v, static_cast<i8>(0))); }
	template<>  u32 saturate_cast<u32>(i16 v) { return static_cast<u32>(std::max(v, static_cast<i16>(0))); }
	template<>  u32 saturate_cast<u32>(i32 v) { return static_cast<u32>(std::max(v, static_cast<i32>(0))); }
	template<>  u32 saturate_cast<u32>(i64 v) { return static_cast<u32>((static_cast<u64>(v) <= static_cast<u64>(std::numeric_limits<u32>::max())) ? v : (v > 0 ? std::numeric_limits<u32>::max() : 0)); }

	template<>  u32 saturate_cast<u32>(u8 v) { return static_cast<u32>(v); }
	template<>  u32 saturate_cast<u32>(u16 v) { return static_cast<u32>(v); }
	template<>  u32 saturate_cast<u32>(u64 v) { return static_cast<u32>(std::min(v, static_cast<u64>(std::numeric_limits<u32>::max()))); }
	template<>  u32 saturate_cast<u32>(f32 v) { return static_cast<u32>(fy::round(v)); }
	template<>  u32 saturate_cast<u32>(f64 v) { return static_cast<u32>(fy::round(v)); }



	template<>  u64 saturate_cast<u64>(i8 v) { return static_cast<u64>(std::max(v, static_cast<i8>(0))); }
	template<>  u64 saturate_cast<u64>(i16 v) { return static_cast<u64>(std::max(v, static_cast<i16>(0))); }
	template<>  u64 saturate_cast<u64>(i32 v) { return static_cast<u64>(std::max(v, static_cast<i32>(0))); }
	template<>  u64 saturate_cast<u64>(i64 v) { return static_cast<u64>(std::max(v, static_cast<i64>(0))); }

	template<>  i64 saturate_cast<i64>(u64 v) { return static_cast<i64>(std::min(v, static_cast<u64>(std::numeric_limits<i64>::max()))); }
	template<>  i64 saturate_cast<i64>(f32 v)
	{
		if (v >= static_cast<f32>(std::numeric_limits<i64>::max())) return std::numeric_limits<i64>::max();
		if (v <= static_cast<f32>(std::numeric_limits<i64>::lowest())) return std::numeric_limits<i64>::lowest();
		return static_cast<i64>(fy::round(v));
	}
	template<>  i64 saturate_cast<i64>(f64 v)
	{
		if (v >= static_cast<f64>(std::numeric_limits<i64>::max())) return std::numeric_limits<i64>::max();
		if (v <= static_cast<f64>(std::numeric_limits<i64>::lowest())) return std::numeric_limits<i64>::lowest();
		return static_cast<i64>(fy::round(v));
	}

	template<> f16 saturate_cast<f16>(u8 v) { return f16(static_cast<f32>(v)); }
	template<> f16 saturate_cast<f16>(i8 v) { return f16(static_cast<f32>(v)); }
	template<> f16 saturate_cast<f16>(u16 v) { return f16(static_cast<f32>(v)); }
	template<> f16 saturate_cast<f16>(i16 v) { return f16(static_cast<f32>(v)); }
	template<> f16 saturate_cast<f16>(u32 v) { return f16(static_cast<f32>(v)); }
	template<> f16 saturate_cast<f16>(i32 v) { return f16(static_cast<f32>(v)); }
	template<> f16 saturate_cast<f16>(u64 v) { return f16(static_cast<f64>(v)); }
	template<> f16 saturate_cast<f16>(i64 v) { return f16(static_cast<f64>(v)); }
	template<> f16 saturate_cast<f16>(f32 v) { return f16(v); }
	template<> f16 saturate_cast<f16>(f64 v) { return f16(static_cast<f64>(v)); }

	template<> bf16 saturate_cast<bf16>(u8 v) { return bf16(static_cast<f32>(v)); }
	template<> bf16 saturate_cast<bf16>(i8 v) { return bf16(static_cast<f32>(v)); }
	template<> bf16 saturate_cast<bf16>(u16 v) { return bf16(static_cast<f32>(v)); }
	template<> bf16 saturate_cast<bf16>(i16 v) { return bf16(static_cast<f32>(v)); }
	template<> bf16 saturate_cast<bf16>(u32 v) { return bf16(static_cast<f32>(v)); }
	template<> bf16 saturate_cast<bf16>(i32 v) { return bf16(static_cast<f32>(v)); }
	template<> bf16 saturate_cast<bf16>(u64 v) { return bf16(static_cast<f64>(v)); }
	template<> bf16 saturate_cast<bf16>(i64 v) { return bf16(static_cast<f64>(v)); }
	template<> bf16 saturate_cast<bf16>(f32 v) { return bf16(v); }
	template<> bf16 saturate_cast<bf16>(f64 v) { return bf16(static_cast<f64>(v)); }

}

export namespace fy
{
	namespace simd
	{
		template<typename T>
		concept VectorType = requires(T t)
		{
			requires std::is_same_v<decltype(T::data), typename T::vector_t>;
			typename T::scalar_t;
			typename T::vector_t;
			typename T::upper_t;
			typename T::lower_t;

			{ T::batch_size } -> std::convertible_to<std::size_t>;
			{ t.download(std::declval<typename T::scalar_t*>()) } -> std::same_as<void>;
		};

		template<typename T>
		inline constexpr bool is_vector_v = VectorType<T>;

		template<typename T>
		inline constexpr bool is_avx_t = VectorType<T> && sizeof(T) == sizeof(__m256);

		template<typename T>
		inline constexpr bool is_sse_t = VectorType<T> && sizeof(T) == sizeof(__m128);

		template<typename T>
		concept Floating_VectorType = VectorType<T> && (std::is_floating_point_v<typename T::scalar_t> || std::is_same_v<f16, typename T::scalar_t>);

		template<typename T>
		concept Integral_VectorType = VectorType<T> && (std::is_integral_v<typename T::scalar_t>);

		template<typename T>
		concept UnsignedIntegral_VectorType = VectorType<T> && (std::is_integral_v<typename T::scalar_t> && std::is_unsigned_v<typename T::scalar_t>);

		struct auto_zeroupper
		{
			auto_zeroupper() { }
			~auto_zeroupper()
			{
				_mm256_zeroupper();
			}
		};

		__forceinline void v_zeroupper()
		{
			_mm256_zeroupper();
		}

		template<VectorType T> __forceinline T v_subs(const T&, const T&);

		template<VectorType T> __forceinline T v_add(const T&, const T&);
		template<VectorType T> __forceinline T v_sub(const T&, const T&);
		template<VectorType T> __forceinline T v_mul(const T&, const T&);
		template<VectorType T> __forceinline T v_div(const T&, const T&);

		template<VectorType T> __forceinline T v_shift_left_logical(const T&, i32);
		template<VectorType T> __forceinline T operator << (const T&, i32);
		template<VectorType T> __forceinline T& operator <<= (T&, i32);

		template<VectorType T> __forceinline T v_shift_right_arithmetic(const T&, i32);
		template<VectorType T> __forceinline T v_shift_right_logical(const T&, i32);
		template<VectorType T> __forceinline T operator >> (const T&, i32);
		template<VectorType T> __forceinline T& operator >>= (T&, i32);

		template<VectorType T> __forceinline T operator + (const T&, const T&);
		template<VectorType T> __forceinline T operator - (const T&, const T&);
		template<VectorType T> __forceinline T operator * (const T&, const T&);
		template<VectorType T> __forceinline T operator / (const T&, const T&);

		template<VectorType T> __forceinline T& operator += (T&, const T&);
		template<VectorType T> __forceinline T& operator -= (T&, const T&);
		template<VectorType T> __forceinline T& operator *= (T&, const T&);
		template<VectorType T> __forceinline T& operator /= (T&, const T&);

		template<VectorType Left, BasicArithmetic Right> __forceinline Left operator + (const Left&, Right);
		template<VectorType Left, BasicArithmetic Right> __forceinline Left operator - (const Left&, Right);
		template<VectorType Left, BasicArithmetic Right> __forceinline Left operator * (const Left&, Right);
		template<VectorType Left, BasicArithmetic Right> __forceinline Left operator / (const Left&, Right);

		template<VectorType Left, BasicArithmetic Right> __forceinline Left& operator += (Left&, Right);
		template<VectorType Left, BasicArithmetic Right> __forceinline Left& operator -= (Left&, Right);
		template<VectorType Left, BasicArithmetic Right> __forceinline Left& operator *= (Left&, Right);
		template<VectorType Left, BasicArithmetic Right> __forceinline Left& operator /= (Left&, Right);

		template<VectorType T> __forceinline T v_lt(const T& left, const T& right);
		template<VectorType T> __forceinline T v_gt(const T& left, const T& right);
		template<VectorType T> __forceinline T v_eq(const T& left, const T& right);
		template<VectorType T> __forceinline T v_ne(const T& left, const T& right);
		template<VectorType T> __forceinline T v_le(const T& left, const T& right);
		template<VectorType T> __forceinline T v_ge(const T& left, const T& right);
		template<Integral_VectorType T> __forceinline T v_remainder(const T& left, const T& right);

		template<VectorType T> __forceinline T operator < (const T& left, const T& right);
		template<VectorType T> __forceinline T operator > (const T& left, const T& right);
		template<VectorType T> __forceinline T operator == (const T& left, const T& right);
		template<VectorType T> __forceinline T operator != (const T& left, const T& right);
		template<VectorType T> __forceinline T operator <= (const T& left, const T& right);
		template<VectorType T> __forceinline T operator >= (const T& left, const T& right);
		template<Integral_VectorType T> __forceinline T operator % (const T& left, const T& right);

		template<VectorType Left, BasicArithmetic Right> __forceinline Left operator < (const Left&, Right);
		template<VectorType Left, BasicArithmetic Right> __forceinline Left operator > (const Left&, Right);
		template<VectorType Left, BasicArithmetic Right> __forceinline Left operator == (const Left&, Right);
		template<VectorType Left, BasicArithmetic Right> __forceinline Left operator != (const Left&, Right);
		template<VectorType Left, BasicArithmetic Right> __forceinline Left operator <= (const Left&, Right);
		template<VectorType Left, BasicArithmetic Right> __forceinline Left operator >= (const Left&, Right);
		template<Integral_VectorType Left, BasicArithmetic Right> __forceinline Left operator % (const Left&, Right);



		template<VectorType T> __forceinline T v_fused_mul_add(const T& mul_left, const T& mul_right, const T& add_right);
		template<VectorType T> __forceinline T v_fused_mul_sub(const T& mul_left, const T& mul_right, const T& sub_right);

		template<VectorType T> __forceinline T v_bitwise_AND(const T&, const T&);
		template<VectorType T> __forceinline T v_bitwise_OR(const T&, const T&);
		template<VectorType T> __forceinline T v_bitwise_XOR(const T&, const T&);
		template<VectorType T> __forceinline T v_bitwise_ANDNOT(const T&, const T&);

		template<Floating_VectorType T> __forceinline T v_close(const T& v0, const T& v1, const T& vtolerance);

		template<VectorType T> __forceinline T operator & (const T&, const T&);
		template<VectorType T> __forceinline T operator | (const T&, const T&);
		template<VectorType T> __forceinline T operator ^ (const T&, const T&);

		template<VectorType Dst_t, VectorType Src_t> __forceinline Dst_t v_reinterpret_convert(const Src_t&);
		template<VectorType Dst_t, VectorType Src_t> __forceinline Dst_t v_convert(const Src_t&);

		template<BasicArithmetic Scalar_t, VectorType ... Vec> __forceinline void v_load_deinterleave(const Scalar_t*, Vec& ...);
		template<BasicArithmetic Scalar_t, VectorType ... Vec> __forceinline void v_store_interleave(Scalar_t*, const Vec& ...);
		template<VectorType Dst_t, VectorType ... Src_t> Dst_t __forceinline v_merge_as(const Src_t& ... src);

		template<VectorType Src_t, Floating_VectorType... Dst_t> __forceinline void v_as_floating(const Src_t&, Dst_t&...);


		template<VectorType Src_t, VectorType Dst_t> __forceinline Dst_t v_pack(const Src_t&, const Src_t&);
		template<VectorType Src_t, VectorType Dst_t> __forceinline void v_expand(const Src_t&, Dst_t&, Dst_t&);
		template<VectorType Src_t, VectorType Dst_t> __forceinline void v_unpack(const Src_t&, Dst_t&, Dst_t&);

		template<VectorType Src_t> Src_t v_max_replace(const Src_t&, const Src_t&);
		template<VectorType Src_t> Src_t v_min_replace(const Src_t&, const Src_t&);

		template<VectorType Vec_t> __forceinline Vec_t v_abs(const Vec_t&);
		template<VectorType Vec_t> __forceinline Vec_t v_avg(const Vec_t&, const Vec_t&);

		template<Floating_VectorType Vec_t> __forceinline Vec_t v_exp(const Vec_t&);
		template<Floating_VectorType Vec_t> __forceinline Vec_t v_exp2(const Vec_t&);
		template<Floating_VectorType Vec_t> __forceinline Vec_t v_exp10(const Vec_t&);
		template<Floating_VectorType Vec_t> __forceinline Vec_t v_log(const Vec_t&);
		template<Floating_VectorType Vec_t> __forceinline Vec_t v_log2(const Vec_t&);
		template<Floating_VectorType Vec_t> __forceinline Vec_t v_log10(const Vec_t&);

		template<Floating_VectorType Vec_t> __forceinline Vec_t v_sqrt(const Vec_t&);
		template<Floating_VectorType Vec_t> __forceinline Vec_t v_rsqrt(const Vec_t&);
		template<Floating_VectorType Vec_t> __forceinline Vec_t v_rcp(const Vec_t&);

		template<VectorType Src_t> Src_t::scalar_t v_min_reduce(const Src_t&);
		template<VectorType Src_t> Src_t::scalar_t v_max_reduce(const Src_t&);

		template<VectorType T> auto v_reduce_sum(const T&);

		


		struct alignas(16) v_uint8x16;
		struct alignas(16) v_uint16x8;
		struct alignas(16) v_uint32x4;
		struct alignas(16) v_uint64x2;
		struct alignas(16) v_int8x16;
		struct alignas(16) v_int16x8;
		struct alignas(16) v_int32x4;
		struct alignas(16) v_int64x2;
		struct alignas(16) v_bfloat16x8;
		struct alignas(16) v_float16x8;
		struct alignas(16) v_float32x4;
		struct alignas(16) v_float64x2;

		struct alignas(32) v_uint8x32;
		struct alignas(32) v_uint16x16;
		struct alignas(32) v_uint32x8;
		struct alignas(32) v_uint64x4;
		struct alignas(32) v_int8x32;
		struct alignas(32) v_int16x16;
		struct alignas(32) v_int32x8;
		struct alignas(32) v_int64x4;
		struct alignas(32) v_bfloat16x16;
		struct alignas(32) v_float16x16;
		struct alignas(32) v_float32x8;
		struct alignas(32) v_float64x4;
	}
}

export namespace fy
{
	namespace simd
	{
		template<VectorType T> std::string vector_to_string(const T& vec)
		{
			std::array<typename T::scalar_t, T::batch_size> arr;
			vec.download(arr.data());

			auto to_string = [ ](const auto& elem)
				{
					if constexpr (std::is_same_v<typename T::scalar_t, u8> || std::is_same_v<typename T::scalar_t, i8>)
					{
						return std::to_string(saturate_cast<i32>(elem));
					}
					else
					{
						return std::to_string(elem);
					}
				};

			std::string result = "[";
			auto range = arr | std::views::transform(to_string);

			for (auto it = range.begin(); it != range.end(); ++it)
			{
				if (it != range.begin())
				{
					result += ", ";
				}
				result += *it;
			}

			std::string ty;
			if constexpr (std::is_unsigned_v<typename T::scalar_t>)
			{
				ty = std::string("uint");
			}
			else if constexpr (std::is_signed_v<typename T::scalar_t> && std::is_integral_v<typename T::scalar_t>)
			{
				ty = std::string("int");
			}
			else if constexpr (std::is_same_v<typename T::scalar_t, f16> || std::is_floating_point_v<typename T::scalar_t>)
			{
				ty = std::string("float");
			}
			else
			{
				std::unreachable();
			}
			ty += std::to_string(sizeof(T::scalar_t) * 8);;

			result += std::string("]");

			return result;
		}

		template<VectorType T>
		std::ostream& operator << (std::ostream& os, const T& p)
		{
			os << vector_to_string(p);
			return os;
		}

		template<BasicArithmetic T> struct vector_type_traits { using sse_type = void; using avx2_type = void; };
		template<> struct vector_type_traits<u8> { using sse_type = v_uint8x16; using avx2_type = v_uint8x32; };
		template<> struct vector_type_traits<u16> { using sse_type = v_uint16x8; using avx2_type = v_uint16x16; };
		template<> struct vector_type_traits<u32> { using sse_type = v_uint32x4; using avx2_type = v_uint32x8; };
		template<> struct vector_type_traits<u64> { using sse_type = v_uint64x2; using avx2_type = v_uint64x4; };
		template<> struct vector_type_traits<i8> { using sse_type = v_int8x16; using avx2_type = v_int8x32; };
		template<> struct vector_type_traits<i16> { using sse_type = v_int16x8; using avx2_type = v_int16x16; };
		template<> struct vector_type_traits<i32> { using sse_type = v_int32x4; using avx2_type = v_int32x8; };
		template<> struct vector_type_traits<i64> { using sse_type = v_int64x2; using avx2_type = v_int64x4; };
		template<> struct vector_type_traits<bf16> { using sse_type = v_float16x8; using avx2_type = v_bfloat16x16; };
		template<> struct vector_type_traits<f16> { using sse_type = v_float16x8; using avx2_type = v_float16x16; };
		template<> struct vector_type_traits<f32> { using sse_type = v_float32x4; using avx2_type = v_float32x8; };
		template<> struct vector_type_traits<f64> { using sse_type = v_float64x2; using avx2_type = v_float64x4; };

		template<typename T> using Instancing_vector_type_sse = typename vector_type_traits<T>::sse_type;
		template<typename T> using Instancing_vector_type_avx2 = typename vector_type_traits<T>::avx2_type;

		template<BasicArithmetic T> struct sse_vector_type_selector;
		template<> struct sse_vector_type_selector<u8> { using type = v_uint8x16; };
		template<> struct sse_vector_type_selector<u16> { using type = v_uint16x8; };
		template<> struct sse_vector_type_selector<u32> { using type = v_uint32x4; };
		template<> struct sse_vector_type_selector<u64> { using type = v_uint64x2; };
		template<> struct sse_vector_type_selector<i8> { using type = v_int8x16; };
		template<> struct sse_vector_type_selector<i16> { using type = v_int16x8; };
		template<> struct sse_vector_type_selector<i32> { using type = v_int32x4; };
		template<> struct sse_vector_type_selector<i64> { using type = v_int64x2; };
		template<> struct sse_vector_type_selector<f16> { using type = v_float16x8; };
		template<> struct sse_vector_type_selector<f32> { using type = v_float32x4; };
		template<> struct sse_vector_type_selector<f64> { using type = v_float64x2; };

		template<BasicArithmetic T> struct avx2_vector_type_selector;
		template<> struct avx2_vector_type_selector<u8> { using type = v_uint8x32; };
		template<> struct avx2_vector_type_selector<u16> { using type = v_uint16x16; };
		template<> struct avx2_vector_type_selector<u32> { using type = v_uint32x8; };
		template<> struct avx2_vector_type_selector<u64> { using type = v_uint64x4; };
		template<> struct avx2_vector_type_selector<i8> { using type = v_int8x32; };
		template<> struct avx2_vector_type_selector<i16> { using type = v_int16x16; };
		template<> struct avx2_vector_type_selector<i32> { using type = v_int32x8; };
		template<> struct avx2_vector_type_selector<i64> { using type = v_int64x4; };
		template<> struct avx2_vector_type_selector<bf16> { using type = v_bfloat16x16; };
		template<> struct avx2_vector_type_selector<f16> { using type = v_float16x16; };
		template<> struct avx2_vector_type_selector<f32> { using type = v_float32x8; };
		template<> struct avx2_vector_type_selector<f64> { using type = v_float64x4; };

		template<BasicArithmetic T> using AVX_t = typename avx2_vector_type_selector<T>::type;
		template<BasicArithmetic T> using SSE_t = typename sse_vector_type_selector<T>::type;

		template<VectorType Dst_t, VectorType Src_t>
		Dst_t v_reinterpret_convert(const Src_t& src)
		{
			static_assert(sizeof(typename Dst_t::vector_t) == sizeof(typename Src_t::vector_t));
			return Dst_t(std::bit_cast<typename Dst_t::vector_t>(src.data));
		}

		struct alignas(16) v_uint8x16
		{
			using scalar_t = u8;
			using vector_t = __m128i;
			using upper_t = v_uint8x32;
			using lower_t = void;

			enum { batch_size = 16 };
			vector_t alignas(16) data;

			void* operator new(std::size_t) = delete;
			void operator delete(void*) = delete;

			v_uint8x16() {}
			v_uint8x16(
				scalar_t v0, scalar_t v1, scalar_t v2, scalar_t v3,
				scalar_t v4, scalar_t v5, scalar_t v6, scalar_t v7,
				scalar_t v8, scalar_t v9, scalar_t v10, scalar_t v11,
				scalar_t v12, scalar_t v13, scalar_t v14, scalar_t v15)
				: data(_mm_setr_epi8(v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15))
			{
			}
			explicit v_uint8x16(const scalar_t* ptr) noexcept : data(_mm_load_si128(reinterpret_cast<const __m128i*>(ptr))) {}

			operator vector_t() noexcept { return data; }

			static v_uint8x16 loadu(const scalar_t* ptr) noexcept
			{
				return v_uint8x16(_mm_load_si128(reinterpret_cast<const __m128i*>(ptr)));
			}

			template<BasicArithmetic T>
			explicit v_uint8x16(const T val) noexcept : data(_mm_set1_epi8(static_cast<typename v_uint8x16::scalar_t>(val))) {}

			explicit v_uint8x16(vector_t data) : data(data) {}
			void download(scalar_t* ptr) const noexcept { _mm_store_si128(reinterpret_cast<__m128i*>(ptr), data); }
			void downloadu(scalar_t* ptr) const noexcept { _mm_storeu_si128(reinterpret_cast<__m128i*>(ptr), data); }
			void streamback(scalar_t* ptr) const noexcept { _mm_stream_si128(reinterpret_cast<__m128i*>(ptr), data); }
			std::string str() const noexcept { return vector_to_string(*this); }
		};

		struct alignas(16) v_uint16x8
		{
			using scalar_t = u16;
			using vector_t = __m128i;
			using upper_t = v_uint16x16;
			using lower_t = void;

			enum { batch_size = 8 };
			vector_t alignas(16) data;

			void* operator new(std::size_t) = delete;
			void operator delete(void*) = delete;

			v_uint16x8() {}
			v_uint16x8(scalar_t v0, scalar_t v1, scalar_t v2, scalar_t v3, scalar_t v4, scalar_t v5, scalar_t v6, scalar_t v7) : data(_mm_setr_epi16(v0, v1, v2, v3, v4, v5, v6, v7)) {}
			explicit v_uint16x8(const scalar_t* ptr) : data(_mm_load_si128(reinterpret_cast<const __m128i*>(ptr))) {}

			operator vector_t() noexcept { return data; }

			static v_uint16x8 loadu(const scalar_t* ptr) noexcept
			{
				return v_uint16x8(_mm_load_si128(reinterpret_cast<const __m128i*>(ptr)));
			}

			template<BasicArithmetic T>
			explicit v_uint16x8(const T val) : data(_mm_set1_epi16(static_cast<typename v_uint16x8::scalar_t>(val))) {}

			explicit v_uint16x8(vector_t data) : data(data) {}
			void download(scalar_t* ptr) const { _mm_store_si128(reinterpret_cast<__m128i*>(ptr), data); }
			void downloadu(scalar_t* ptr) const noexcept { _mm_storeu_si128(reinterpret_cast<__m128i*>(ptr), data); }
			void streamback(scalar_t* ptr) const noexcept { _mm_stream_si128(reinterpret_cast<__m128i*>(ptr), data); }
			std::string str() const { return vector_to_string(*this); }
		};

		struct alignas(16) v_uint32x4
		{
			using scalar_t = u32;
			using vector_t = __m128i;
			using upper_t = v_uint32x8;
			using lower_t = void;

			enum { batch_size = 4 };
			vector_t alignas(16) data;

			void* operator new(std::size_t) = delete;
			void operator delete(void*) = delete;

			v_uint32x4() {}
			v_uint32x4(scalar_t v0, scalar_t v1, scalar_t v2, scalar_t v3) : data(_mm_setr_epi32(v0, v1, v2, v3)) {}
			explicit v_uint32x4(const scalar_t* ptr) : data(_mm_load_si128(reinterpret_cast<const __m128i*>(ptr))) {}

			operator vector_t() noexcept { return data; }

			static v_uint32x4 loadu(const scalar_t* ptr) noexcept
			{
				return v_uint32x4(_mm_load_si128(reinterpret_cast<const __m128i*>(ptr)));
			}

			template<BasicArithmetic T>
			explicit v_uint32x4(const T val) : data(_mm_set1_epi32(static_cast<typename v_uint32x4::scalar_t>(val))) {}

			explicit v_uint32x4(vector_t data) : data(data) {}
			void download(scalar_t* ptr) const { _mm_store_si128(reinterpret_cast<__m128i*>(ptr), data); }
			void downloadu(scalar_t* ptr) const noexcept { _mm_storeu_si128(reinterpret_cast<__m128i*>(ptr), data); }
			void streamback(scalar_t* ptr) const noexcept { _mm_stream_si128(reinterpret_cast<__m128i*>(ptr), data); }
			std::string str() const { return vector_to_string(*this); }
		};

		struct alignas(16) v_uint64x2
		{
			using scalar_t = u64;
			using vector_t = __m128i;
			using upper_t = v_uint64x4;
			using lower_t = void;

			enum { batch_size = 2 };
			vector_t alignas(16) data;

			void* operator new(std::size_t) = delete;
			void operator delete(void*) = delete;

			v_uint64x2() {}
			v_uint64x2(scalar_t v0, scalar_t v1) : data(_mm_set_epi64x(v1, v0)) {}
			explicit v_uint64x2(const scalar_t* ptr) : data(_mm_load_si128(reinterpret_cast<const __m128i*>(ptr))) {}

			operator vector_t() noexcept { return data; }

			static v_uint64x2 loadu(const scalar_t* ptr) noexcept
			{
				return v_uint64x2(_mm_load_si128(reinterpret_cast<const __m128i*>(ptr)));
			}

			template<BasicArithmetic T>
			explicit v_uint64x2(const T val) : data(_mm_set1_epi64x(static_cast<typename v_uint64x2::scalar_t>(val))) {}

			explicit v_uint64x2(vector_t data) : data(data) {}
			void download(scalar_t* ptr) const { _mm_store_si128(reinterpret_cast<__m128i*>(ptr), data); }
			void downloadu(scalar_t* ptr) const noexcept { _mm_storeu_si128(reinterpret_cast<__m128i*>(ptr), data); }
			void streamback(scalar_t* ptr) const noexcept { _mm_stream_si128(reinterpret_cast<__m128i*>(ptr), data); }
			std::string str() const { return vector_to_string(*this); }
		};

		struct alignas(16) v_int8x16
		{
			using scalar_t = i8;
			using vector_t = __m128i;
			using upper_t = v_int8x32;
			using lower_t = void;

			enum { batch_size = 16 };
			vector_t alignas(16) data;

			void* operator new(std::size_t) = delete;
			void operator delete(void*) = delete;

			v_int8x16() {}
			v_int8x16(
				scalar_t v0, scalar_t v1, scalar_t v2, scalar_t v3,
				scalar_t v4, scalar_t v5, scalar_t v6, scalar_t v7,
				scalar_t v8, scalar_t v9, scalar_t v10, scalar_t v11,
				scalar_t v12, scalar_t v13, scalar_t v14, scalar_t v15)
				: data(_mm_setr_epi8(
					v0, v1, v2, v3, v4, v5, v6, v7,
					v8, v9, v10, v11, v12, v13, v14, v15))
			{
			}
			explicit v_int8x16(const scalar_t* ptr) : data(_mm_load_si128(reinterpret_cast<const __m128i*>(ptr))) {}

			operator vector_t() noexcept { return data; }

			static v_int8x16 loadu(const scalar_t* ptr) noexcept
			{
				return v_int8x16(_mm_load_si128(reinterpret_cast<const __m128i*>(ptr)));
			}

			template<BasicArithmetic T>
			explicit v_int8x16(const T val) : data(_mm_set1_epi8(static_cast<typename v_int8x16::scalar_t>(val))) {}

			explicit v_int8x16(vector_t data) : data(data) {}
			void download(scalar_t* ptr) const { _mm_store_si128(reinterpret_cast<__m128i*>(ptr), data); }
			void downloadu(scalar_t* ptr) const noexcept { _mm_storeu_si128(reinterpret_cast<__m128i*>(ptr), data); }
			void streamback(scalar_t* ptr) const noexcept { _mm_stream_si128(reinterpret_cast<__m128i*>(ptr), data); }
			std::string str() const { return vector_to_string(*this); }
		};

		struct alignas(16) v_int16x8
		{
			using scalar_t = i16;
			using vector_t = __m128i;
			using upper_t = v_int16x16;
			using lower_t = void;

			enum { batch_size = 8 };
			vector_t alignas(16) data;

			void* operator new(std::size_t) = delete;
			void operator delete(void*) = delete;

			v_int16x8() {}
			v_int16x8(scalar_t v0, scalar_t v1, scalar_t v2, scalar_t v3, scalar_t v4, scalar_t v5, scalar_t v6, scalar_t v7) : data(_mm_setr_epi16(v0, v1, v2, v3, v4, v5, v6, v7)) {}
			explicit v_int16x8(const scalar_t* ptr) : data(_mm_load_si128(reinterpret_cast<const __m128i*>(ptr))) {}

			operator vector_t() noexcept { return data; }

			static v_int16x8 loadu(const scalar_t* ptr) noexcept
			{
				return v_int16x8(_mm_load_si128(reinterpret_cast<const __m128i*>(ptr)));
			}

			template<BasicArithmetic T>
			explicit v_int16x8(const T val) : data(_mm_set1_epi16(static_cast<typename v_int16x8::scalar_t>(val))) {}

			explicit v_int16x8(vector_t data) : data(data) {}
			void download(scalar_t* ptr) const { _mm_store_si128(reinterpret_cast<__m128i*>(ptr), data); }
			void downloadu(scalar_t* ptr) const noexcept { _mm_storeu_si128(reinterpret_cast<__m128i*>(ptr), data); }
			void streamback(scalar_t* ptr) const noexcept { _mm_stream_si128(reinterpret_cast<__m128i*>(ptr), data); }
			std::string str() const { return vector_to_string(*this); }
		};

		struct alignas(16) v_int32x4
		{
			using scalar_t = i32;
			using vector_t = __m128i;
			using upper_t = v_int32x8;
			using lower_t = void;

			enum { batch_size = 4 };
			vector_t alignas(16) data;

			void* operator new(std::size_t) = delete;
			void operator delete(void*) = delete;

			v_int32x4() {}
			v_int32x4(scalar_t v0, scalar_t v1, scalar_t v2, scalar_t v3) : data(_mm_setr_epi32(v0, v1, v2, v3)) {}
			explicit v_int32x4(const scalar_t* ptr) : data(_mm_load_si128(reinterpret_cast<const __m128i*>(ptr))) {}

			operator vector_t() noexcept { return data; }

			static v_int32x4 loadu(const scalar_t* ptr) noexcept
			{
				return v_int32x4(_mm_load_si128(reinterpret_cast<const __m128i*>(ptr)));
			}

			template<BasicArithmetic T>
			explicit v_int32x4(const T val) : data(_mm_set1_epi32(static_cast<typename v_int32x4::scalar_t>(val))) {}

			explicit v_int32x4(vector_t data) : data(data) {}
			void download(scalar_t* ptr) const { _mm_store_si128(reinterpret_cast<__m128i*>(ptr), data); }
			void downloadu(scalar_t* ptr) const noexcept { _mm_storeu_si128(reinterpret_cast<__m128i*>(ptr), data); }
			void streamback(scalar_t* ptr) const noexcept { _mm_stream_si128(reinterpret_cast<__m128i*>(ptr), data); }
			std::string str() const { return vector_to_string(*this); }
		};

		struct alignas(16) v_int64x2
		{
			using scalar_t = i64;
			using vector_t = __m128i;
			using upper_t = v_int64x4;
			using lower_t = void;

			enum { batch_size = 2 };
			vector_t alignas(16) data;

			void* operator new(std::size_t) = delete;
			void operator delete(void*) = delete;

			v_int64x2() {}
			v_int64x2(scalar_t v0, scalar_t v1) : data(_mm_set_epi64x(v1, v0)) {}
			explicit v_int64x2(const scalar_t* ptr) : data(_mm_load_si128(reinterpret_cast<const __m128i*>(ptr))) {}

			operator vector_t() noexcept { return data; }

			static v_int64x2 loadu(const scalar_t* ptr) noexcept
			{
				return v_int64x2(_mm_load_si128(reinterpret_cast<const __m128i*>(ptr)));
			}

			template<BasicArithmetic T>
			explicit v_int64x2(const T val) : data(_mm_set1_epi64x(static_cast<typename v_int64x2::scalar_t>(val))) {}

			explicit v_int64x2(vector_t data) : data(data) {}
			void download(scalar_t* ptr) const { _mm_store_si128(reinterpret_cast<__m128i*>(ptr), data); }
			void downloadu(scalar_t* ptr) const noexcept { _mm_storeu_si128(reinterpret_cast<__m128i*>(ptr), data); }
			void streamback(scalar_t* ptr) const noexcept { _mm_stream_si128(reinterpret_cast<__m128i*>(ptr), data); }
			std::string str() const { return vector_to_string(*this); }
		};

		struct alignas(16) v_bfloat16x8
		{
			using scalar_t = bf16;
			using vector_t = __m128i;
			using upper_t = v_bfloat16x16;
			using lower_t = void;

			enum { batch_size = 8 };
			vector_t alignas(16) data;

			void* operator new(std::size_t) = delete;
			void operator delete(void*) = delete;

			v_bfloat16x8() {}

			explicit v_bfloat16x8(const scalar_t* arr) : data(_mm_load_si128(reinterpret_cast<const vector_t*>(arr))) {}
			explicit v_bfloat16x8(vector_t data) : data(data) {}

			template<BasicArithmetic T>
			v_bfloat16x8(
				T v0, T v1, T v2, T v3, T v4, T v5, T v6, T v7,
				T v8, T v9, T v10, T v11, T v12, T v13, T v14, T v15)
			{
				if constexpr (std::is_same_v<T, scalar_t>)
				{
					data = _mm_setr_epi16(v0.w, v1.w, v2.w, v3.w, v4.w, v5.w, v6.w, v7.w, v8.w, v9.w, v10.w, v11.w, v12.w, v13.w, v14.w, v15.w);
				}
				else
				{
					data = _mm_setr_epi16(
						scalar_t(v0).bits_, scalar_t(v1).bits_, scalar_t(v2).bits_, scalar_t(v3).bits_,
						scalar_t(v4).bits_, scalar_t(v5).bits_, scalar_t(v6).bits_, scalar_t(v7).bits_
					);
				}
			}

			static v_bfloat16x8 loadu(const scalar_t* ptr) noexcept
			{
				return v_bfloat16x8(_mm_load_si128(reinterpret_cast<const vector_t*>(ptr)));
			}

			void download(scalar_t* ptr) const { _mm_store_si128(reinterpret_cast<vector_t*>(ptr), data); }
			void downloadu(scalar_t* ptr) const { _mm_storeu_si128(reinterpret_cast<vector_t*>(ptr), data); }
			void streamback(scalar_t* ptr) const noexcept { _mm_stream_si128(reinterpret_cast<vector_t*>(ptr), data); }
			std::string str() const { return vector_to_string(*this); }
		};

		struct alignas(16) v_float16x8
		{
			using scalar_t = f16;
			using vector_t = __m128h;
			using upper_t = v_float16x16;
			using lower_t = void;

			enum { batch_size = 8 };
			vector_t alignas(16) data;

			void* operator new(std::size_t) = delete;
			void operator delete(void*) = delete;

			v_float16x8() {}

			template<BasicArithmetic T>
			v_float16x8(T v0, T v1, T v2, T v3, T v4, T v5, T v6, T v7) noexcept
			{
				if constexpr (std::is_same_v<T, f16>)
				{
					data = _mm_setr_epi16(v0.w, v1.w, v2.w, v3.w, v4.w, v5.w, v6.w, v7.w);
				}
				else
				{
					data = _mm_setr_epi16(f16(v0).w, f16(v1).w, f16(v2).w, f16(v3).w, f16(v4).w, f16(v5).w, f16(v6).w, f16(v7).w);
				}
			}

			explicit v_float16x8(const scalar_t* arr) : data(_mm_load_si128(reinterpret_cast<const __m128i*>(arr))) {}
			explicit v_float16x8(vector_t data) : data(data) {}

			static v_float16x8 loadu(const scalar_t* ptr) noexcept
			{
				return v_float16x8(_mm_load_si128(reinterpret_cast<const __m128i*>(ptr)));
			}

			operator vector_t() noexcept { return data; }

			template<BasicArithmetic T> explicit v_float16x8(const T value)
			{
				if constexpr (std::is_same_v<T, f16>)
				{
					this->data = _mm_set1_epi16(value.w);
				}
				else
				{
					f64 temp = static_cast<f64>(value);
					f16 res(temp);
					this->data = _mm_set1_epi16(res.w);
				}
			}

			void download(scalar_t* ptr) const { _mm_store_si128(reinterpret_cast<__m128i*>(ptr), data); }
			void downloadu(scalar_t* ptr) const noexcept { _mm_storeu_si128(reinterpret_cast<__m128i*>(ptr), data); }
			void streamback(scalar_t* ptr) const noexcept { _mm_stream_si128(reinterpret_cast<__m128i*>(ptr), data); }
			std::string str() const { return vector_to_string(*this); }
		};

		struct alignas(16) v_float32x4
		{
			using scalar_t = f32;
			using vector_t = __m128;
			using upper_t = v_float32x8;
			using lower_t = void;

			enum { batch_size = 4 };
			vector_t alignas(16) data;

			void* operator new(std::size_t) = delete;
			void operator delete(void*) = delete;

			v_float32x4() {}
			v_float32x4(scalar_t v0, scalar_t v1, scalar_t v2, scalar_t v3) : data(_mm_setr_ps(v0, v1, v2, v3)) {}
			explicit v_float32x4(const scalar_t* ptr) : data(_mm_load_ps(ptr)) {}

			operator vector_t() noexcept { return data; }

			static v_float32x4 loadu(const scalar_t* ptr) noexcept
			{
				return v_float32x4(_mm_load_ps(ptr));
			}

			template<BasicArithmetic T>
			explicit v_float32x4(const T val) : data(_mm_set1_ps(static_cast<typename v_float32x4::scalar_t>(val))) {}

			explicit v_float32x4(vector_t data) : data(data) {}
			void download(scalar_t* ptr) const { _mm_store_ps(ptr, data); }
			void downloadu(scalar_t* ptr) const noexcept { _mm_storeu_ps(ptr, data); }
			void streamback(scalar_t* ptr) const noexcept { _mm_stream_ps(ptr, data); }
			std::string str() const { return vector_to_string(*this); }
		};

		struct alignas(16) v_float64x2
		{
			using scalar_t = f64;
			using vector_t = __m128d;
			using upper_t = v_float64x4;
			using lower_t = void;

			enum { batch_size = 2 };
			vector_t alignas(16) data;

			void* operator new(std::size_t) = delete;
			void operator delete(void*) = delete;

			v_float64x2() {}
			v_float64x2(scalar_t v0, scalar_t v1) : data(_mm_setr_pd(v0, v1)) {}
			explicit v_float64x2(const scalar_t* ptr) : data(_mm_load_pd(ptr)) {}

			static v_float64x2 loadu(const scalar_t* ptr) noexcept
			{
				return v_float64x2(_mm_load_pd(ptr));
			}

			operator vector_t() noexcept { return data; }

			template<BasicArithmetic T>
			explicit v_float64x2(const T val) : data(_mm_set1_pd(static_cast<typename v_float64x2::scalar_t>(val))) {}

			explicit v_float64x2(vector_t data) : data(data) {}
			void download(scalar_t* ptr) const { _mm_store_pd(ptr, data); }
			void downloadu(scalar_t* ptr) const noexcept { _mm_storeu_pd(ptr, data); }
			void streamback(scalar_t* ptr) const noexcept { _mm_stream_pd(ptr, data); }
			std::string str() const { return vector_to_string(*this); }
		};

		struct alignas(32) v_uint8x32
		{
			using scalar_t = u8;
			using vector_t = __m256i;
			using upper_t = void;
			using lower_t = v_uint8x16;

			enum { batch_size = 32 };
			vector_t alignas(32) data;

			void* operator new(std::size_t) = delete;
			void operator delete(void*) = delete;

			const scalar_t& operator[](u8 index) const { return data.m256i_u8[index]; }
			scalar_t& operator[](u8 index) { return data.m256i_u8[index]; }

			v_uint8x32() {}
			v_uint8x32(
				scalar_t v0, scalar_t v1, scalar_t v2, scalar_t v3,
				scalar_t v4, scalar_t v5, scalar_t v6, scalar_t v7,
				scalar_t v8, scalar_t v9, scalar_t v10, scalar_t v11,
				scalar_t v12, scalar_t v13, scalar_t v14, scalar_t v15,
				scalar_t v16, scalar_t v17, scalar_t v18, scalar_t v19,
				scalar_t v20, scalar_t v21, scalar_t v22, scalar_t v23,
				scalar_t v24, scalar_t v25, scalar_t v26, scalar_t v27,
				scalar_t v28, scalar_t v29, scalar_t v30, scalar_t v31)
				: data(_mm256_setr_epi8(
					v0, v1, v2, v3, v4, v5, v6, v7,
					v8, v9, v10, v11, v12, v13, v14, v15,
					v16, v17, v18, v19, v20, v21, v22, v23,
					v24, v25, v26, v27, v28, v29, v30, v31))
			{
			}
			explicit v_uint8x32(const scalar_t* ptr) : data(_mm256_load_si256(reinterpret_cast<const __m256i*>(ptr))) {}

			static v_uint8x32 loadu(const scalar_t* ptr) noexcept
			{
				return v_uint8x32(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr)));
			}

			operator vector_t() noexcept { return data; }

			template<BasicArithmetic T>
			explicit v_uint8x32(const T val) : data(_mm256_set1_epi8(static_cast<typename v_uint8x32::scalar_t>(val))) {}

			explicit v_uint8x32(vector_t data) : data(data) {}
			void download(scalar_t* ptr) const { _mm256_store_si256(reinterpret_cast<__m256i*>(ptr), data); }
			void downloadu(scalar_t* ptr) const { _mm256_storeu_si256(reinterpret_cast<__m256i*>(ptr), data); }
			void streamback(scalar_t* ptr) const noexcept { _mm256_stream_si256(reinterpret_cast<__m256i*>(ptr), data); }
			std::string str() const { return vector_to_string(*this); }
		};

		struct alignas(32) v_uint16x16
		{
			using scalar_t = u16;
			using vector_t = __m256i;
			using upper_t = void;
			using lower_t = v_uint16x8;

			enum { batch_size = 16 };
			vector_t alignas(32) data;

			void* operator new(std::size_t) = delete;
			void operator delete(void*) = delete;

			const scalar_t& operator[](u8 index) const { return data.m256i_u16[index]; }
			scalar_t& operator[](u8 index) { return data.m256i_u16[index]; }

			v_uint16x16() {}
			v_uint16x16(
				scalar_t v0, scalar_t v1, scalar_t v2, scalar_t v3,
				scalar_t v4, scalar_t v5, scalar_t v6, scalar_t v7,
				scalar_t v8, scalar_t v9, scalar_t v10, scalar_t v11,
				scalar_t v12, scalar_t v13, scalar_t v14, scalar_t v15)
				: data(_mm256_setr_epi16(v0, v1, v2, v3, v4, v5, v6, v7,
					v8, v9, v10, v11, v12, v13, v14, v15))
			{
			}
			explicit v_uint16x16(const scalar_t* ptr) : data(_mm256_load_si256(reinterpret_cast<const __m256i*>(ptr))) {}

			static v_uint16x16 loadu(const scalar_t* ptr) noexcept
			{
				return v_uint16x16(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr)));
			}

			operator vector_t() noexcept { return data; }

			template<BasicArithmetic T>
			explicit v_uint16x16(const T val) : data(_mm256_set1_epi16(static_cast<typename v_uint16x16::scalar_t>(val))) {}

			explicit v_uint16x16(vector_t data) : data(data) {}
			void download(scalar_t* ptr) const { _mm256_store_si256(reinterpret_cast<__m256i*>(ptr), data); }
			void downloadu(scalar_t* ptr) const { _mm256_storeu_si256(reinterpret_cast<__m256i*>(ptr), data); }
			void streamback(scalar_t* ptr) const noexcept { _mm256_stream_si256(reinterpret_cast<__m256i*>(ptr), data); }
			std::string str() const { return vector_to_string(*this); }
		};

		struct alignas(32) v_uint32x8
		{
			using scalar_t = u32;
			using vector_t = __m256i;
			using upper_t = void;
			using lower_t = v_uint32x4;

			enum { batch_size = 8 };
			vector_t alignas(32) data;

			void* operator new(std::size_t) = delete;
			void operator delete(void*) = delete;

			const scalar_t& operator[](u8 index) const { return data.m256i_u32[index]; }
			scalar_t& operator[](u8 index) { return data.m256i_u32[index]; }

			v_uint32x8() {}
			v_uint32x8(scalar_t v0, scalar_t v1, scalar_t v2, scalar_t v3,
				scalar_t v4, scalar_t v5, scalar_t v6, scalar_t v7) : data(_mm256_setr_epi32(v0, v1, v2, v3, v4, v5, v6, v7))
			{
			}
			explicit v_uint32x8(const scalar_t* ptr) : data(_mm256_load_si256(reinterpret_cast<const __m256i*>(ptr))) {}

			operator vector_t() noexcept { return data; }

			static v_uint32x8 loadu(const scalar_t* ptr) noexcept
			{
				return v_uint32x8(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr)));
			}

			template<BasicArithmetic T>
			explicit v_uint32x8(const T val) : data(_mm256_set1_epi32(static_cast<typename v_uint32x8::scalar_t>(val))) {}

			explicit v_uint32x8(vector_t data) : data(data) {}
			void download(scalar_t* ptr) const { _mm256_store_si256(reinterpret_cast<__m256i*>(ptr), data); }
			void downloadu(scalar_t* ptr) const { _mm256_storeu_si256(reinterpret_cast<__m256i*>(ptr), data); }
			void streamback(scalar_t* ptr) const noexcept { _mm256_stream_si256(reinterpret_cast<__m256i*>(ptr), data); }
			std::string str() const { return vector_to_string(*this); }
		};

		struct alignas(32) v_uint64x4
		{
			using scalar_t = u64;
			using vector_t = __m256i;
			using upper_t = void;
			using lower_t = v_uint64x2;

			enum { batch_size = 4 };
			vector_t alignas(32) data;

			void* operator new(std::size_t) = delete;
			void operator delete(void*) = delete;

			const scalar_t& operator[](u8 index) const { return data.m256i_u64[index]; }
			scalar_t& operator[](u8 index) { return data.m256i_u64[index]; }

			v_uint64x4() {}
			v_uint64x4(scalar_t v0, scalar_t v1, scalar_t v2, scalar_t v3) : data(_mm256_setr_epi64x(v0, v1, v2, v3)) {}
			explicit v_uint64x4(const scalar_t* ptr) : data(_mm256_load_si256(reinterpret_cast<const __m256i*>(ptr))) {}

			static v_uint64x4 loadu(const scalar_t* ptr) noexcept
			{
				return v_uint64x4(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr)));
			}

			operator vector_t() noexcept { return data; }

			template<BasicArithmetic T>
			explicit v_uint64x4(const T val) : data(_mm256_set1_epi64x(static_cast<typename v_uint64x4::scalar_t>(val))) {}

			explicit v_uint64x4(vector_t data) : data(data) {}
			void download(scalar_t* ptr) const { _mm256_store_si256(reinterpret_cast<__m256i*>(ptr), data); }
			void downloadu(scalar_t* ptr) const { _mm256_storeu_si256(reinterpret_cast<__m256i*>(ptr), data); }
			void streamback(scalar_t* ptr) const noexcept { _mm256_stream_si256(reinterpret_cast<__m256i*>(ptr), data); }
			std::string str() const { return vector_to_string(*this); }
		};

		struct alignas(32) v_int8x32
		{
			using scalar_t = i8;
			using vector_t = __m256i;
			using upper_t = void;
			using lower_t = v_int8x16;

			enum { batch_size = 32 };
			vector_t alignas(32) data;

			void* operator new(std::size_t) = delete;
			void operator delete(void*) = delete;

			const scalar_t& operator[](u8 index) const { return data.m256i_i8[index]; }
			scalar_t& operator[](u8 index) { return reinterpret_cast<scalar_t&>(data.m256i_i8[index]); }

			v_int8x32() {}
			v_int8x32(
				scalar_t v0, scalar_t v1, scalar_t v2, scalar_t v3,
				scalar_t v4, scalar_t v5, scalar_t v6, scalar_t v7,
				scalar_t v8, scalar_t v9, scalar_t v10, scalar_t v11,
				scalar_t v12, scalar_t v13, scalar_t v14, scalar_t v15,
				scalar_t v16, scalar_t v17, scalar_t v18, scalar_t v19,
				scalar_t v20, scalar_t v21, scalar_t v22, scalar_t v23,
				scalar_t v24, scalar_t v25, scalar_t v26, scalar_t v27,
				scalar_t v28, scalar_t v29, scalar_t v30, scalar_t v31)
				: data(_mm256_setr_epi8(
					v0, v1, v2, v3, v4, v5, v6, v7,
					v8, v9, v10, v11, v12, v13, v14, v15,
					v16, v17, v18, v19, v20, v21, v22, v23,
					v24, v25, v26, v27, v28, v29, v30, v31))
			{
			}
			explicit v_int8x32(const scalar_t* ptr) : data(_mm256_load_si256(reinterpret_cast<const __m256i*>(ptr))) {}

			template<BasicArithmetic T>
			explicit v_int8x32(const T val) : data(_mm256_set1_epi8(static_cast<typename v_int8x32::scalar_t>(val))) {}

			static v_int8x32 loadu(const scalar_t* ptr) noexcept
			{
				return v_int8x32(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr)));
			}

			operator vector_t() noexcept { return data; }

			explicit v_int8x32(vector_t data) : data(data) {}
			void download(scalar_t* ptr) const { _mm256_store_si256(reinterpret_cast<__m256i*>(ptr), data); }
			void downloadu(scalar_t* ptr) const { _mm256_storeu_si256(reinterpret_cast<__m256i*>(ptr), data); }
			void streamback(scalar_t* ptr) const noexcept { _mm256_stream_si256(reinterpret_cast<__m256i*>(ptr), data); }
			std::string str() const { return vector_to_string(*this); }
		};

		struct alignas(32) v_int16x16
		{
			using scalar_t = i16;
			using vector_t = __m256i;
			using upper_t = void;
			using lower_t = v_int16x8;

			enum { batch_size = 16 };
			vector_t alignas(32) data;

			void* operator new(std::size_t) = delete;
			void operator delete(void*) = delete;

			const scalar_t& operator[](u8 index) const { return data.m256i_i16[index]; }
			scalar_t& operator[](u8 index) { return data.m256i_i16[index]; }

			v_int16x16() {}
			v_int16x16(
				scalar_t v0, scalar_t v1, scalar_t v2, scalar_t v3,
				scalar_t v4, scalar_t v5, scalar_t v6, scalar_t v7,
				scalar_t v8, scalar_t v9, scalar_t v10, scalar_t v11,
				scalar_t v12, scalar_t v13, scalar_t v14, scalar_t v15)
				: data(_mm256_setr_epi16(v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15))
			{
			}
			explicit v_int16x16(const scalar_t* ptr) : data(_mm256_load_si256(reinterpret_cast<const __m256i*>(ptr))) {}

			static v_int16x16 loadu(const scalar_t* ptr) noexcept
			{
				return v_int16x16(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr)));
			}

			template<BasicArithmetic T>
			explicit v_int16x16(const T val) : data(_mm256_set1_epi16(static_cast<typename v_int16x16::scalar_t>(val))) {}

			operator vector_t() noexcept { return data; }

			explicit v_int16x16(vector_t data) : data(data) {}
			void download(scalar_t* ptr) const { _mm256_store_si256(reinterpret_cast<__m256i*>(ptr), data); }
			void downloadu(scalar_t* ptr) const { _mm256_storeu_si256(reinterpret_cast<__m256i*>(ptr), data); }
			void streamback(scalar_t* ptr) const noexcept { _mm256_stream_si256(reinterpret_cast<__m256i*>(ptr), data); }
			std::string str() const { return vector_to_string(*this); }
		};

		struct alignas(32) v_int32x8
		{
			using scalar_t = i32;
			using vector_t = __m256i;
			using upper_t = void;
			using lower_t = v_int32x4;

			enum { batch_size = 8 };
			vector_t alignas(32) data;

			void* operator new(std::size_t) = delete;
			void operator delete(void*) = delete;

			const scalar_t& operator[](u8 index) const { return data.m256i_i32[index]; }
			scalar_t& operator[](u8 index) { return data.m256i_i32[index]; }

			v_int32x8() {}
			v_int32x8(scalar_t v0, scalar_t v1, scalar_t v2, scalar_t v3, scalar_t v4, scalar_t v5, scalar_t v6, scalar_t v7)
				: data(_mm256_setr_epi32(v0, v1, v2, v3, v4, v5, v6, v7))
			{
			}
			explicit v_int32x8(const scalar_t* ptr) : data(_mm256_load_si256(reinterpret_cast<const __m256i*>(ptr))) {}

			static v_int32x8 loadu(const scalar_t* ptr) noexcept
			{
				return v_int32x8(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr)));
			}

			operator vector_t() noexcept { return data; }

			template<BasicArithmetic T>
			explicit v_int32x8(const T val) : data(_mm256_set1_epi32(static_cast<typename v_int32x8::scalar_t>(val))) {}

			explicit v_int32x8(vector_t data) : data(data) {}
			void download(scalar_t* ptr) const { _mm256_store_si256(reinterpret_cast<__m256i*>(ptr), data); }
			void downloadu(scalar_t* ptr) const { _mm256_storeu_si256(reinterpret_cast<__m256i*>(ptr), data); }
			void streamback(scalar_t* ptr) const noexcept { _mm256_stream_si256(reinterpret_cast<__m256i*>(ptr), data); }
			std::string str() const { return vector_to_string(*this); }
		};

		struct alignas(32) v_int64x4
		{
			using scalar_t = i64;
			using vector_t = __m256i;
			using upper_t = void;
			using lower_t = v_int64x2;

			enum { batch_size = 4 };
			vector_t alignas(32) data;

			void* operator new(std::size_t) = delete;
			void operator delete(void*) = delete;

			const scalar_t& operator[](u8 index) const { return data.m256i_i64[index]; }
			scalar_t& operator[](u8 index) { return data.m256i_i64[index]; }

			static v_int64x4 loadu(const scalar_t* ptr) noexcept
			{
				return v_int64x4(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr)));
			}

			v_int64x4() {}
			v_int64x4(scalar_t v0, scalar_t v1, scalar_t v2, scalar_t v3) : data(_mm256_setr_epi64x(v0, v1, v2, v3)) {}
			explicit v_int64x4(const scalar_t* ptr) : data(_mm256_load_si256(reinterpret_cast<const __m256i*>(ptr))) {}

			operator vector_t() noexcept { return data; }

			template<BasicArithmetic T>
			explicit v_int64x4(const T val) : data(_mm256_set1_epi64x(static_cast<typename v_int64x4::scalar_t>(val))) {}

			explicit v_int64x4(vector_t data) : data(data) {}
			void download(scalar_t* ptr) const { _mm256_store_si256(reinterpret_cast<__m256i*>(ptr), data); }
			void downloadu(scalar_t* ptr) const { _mm256_storeu_si256(reinterpret_cast<__m256i*>(ptr), data); }
			void streamback(scalar_t* ptr) const noexcept { _mm256_stream_si256(reinterpret_cast<__m256i*>(ptr), data); }
			std::string str() const { return vector_to_string(*this); }
		};

		struct alignas(32) v_bfloat16x16
		{
			using scalar_t = bf16;
			using vector_t = __m256i;
			using upper_t = void;
			using lower_t = v_bfloat16x8;

			enum { batch_size = 16 };
			vector_t alignas(32) data;

			void* operator new(std::size_t) = delete;
			void operator delete(void*) = delete;

			const scalar_t operator[](u8 index) const { return bf16::bfloatFromBits(data.m256i_u16[index]); }
			v_bfloat16x16() {}

			explicit v_bfloat16x16(const scalar_t* arr) : data(_mm256_load_si256(reinterpret_cast<const __m256i*>(arr))) {}
			explicit v_bfloat16x16(vector_t data) : data(data) {}

			template<BasicArithmetic T>
			v_bfloat16x16(
				T v0, T v1, T v2, T v3, T v4, T v5, T v6, T v7,
				T v8, T v9, T v10, T v11, T v12, T v13, T v14, T v15)
			{
				if constexpr (std::is_same_v<T, scalar_t>)
				{
					data = _mm256_setr_epi16(v0.w, v1.w, v2.w, v3.w, v4.w, v5.w, v6.w, v7.w, v8.w, v9.w, v10.w, v11.w, v12.w, v13.w, v14.w, v15.w);
				}
				else
				{
					data = _mm256_setr_epi16(
						scalar_t(v0).bits_, scalar_t(v1).bits_, scalar_t(v2).bits_, scalar_t(v3).bits_, 
						scalar_t(v4).bits_, scalar_t(v5).bits_, scalar_t(v6).bits_, scalar_t(v7).bits_,
						scalar_t(v8).bits_, scalar_t(v9).bits_, scalar_t(v10).bits_, scalar_t(v11).bits_, 
						scalar_t(v12).bits_, scalar_t(v13).bits_, scalar_t(v14).bits_, scalar_t(v15).bits_
					);
				}
			}

			static v_bfloat16x16 loadu(const scalar_t* ptr) noexcept
			{
				return v_bfloat16x16(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr)));
			}

			void download(scalar_t* ptr) const { _mm256_store_si256(reinterpret_cast<__m256i*>(ptr), data); }
			void downloadu(scalar_t* ptr) const { _mm256_storeu_si256(reinterpret_cast<__m256i*>(ptr), data); }
			void streamback(scalar_t* ptr) const noexcept { _mm256_stream_si256(reinterpret_cast<__m256i*>(ptr), data); }
			std::string str() const { return vector_to_string(*this); }
		};

		struct alignas(32) v_float16x16
		{
			using scalar_t = f16;
			using vector_t = __m256h;
			using upper_t = void;
			using lower_t = v_float16x8;

			enum { batch_size = 16 };
			vector_t alignas(32) data;

			void* operator new(std::size_t) = delete;
			void operator delete(void*) = delete;

			const scalar_t operator[](u8 index) const { return f16::hfloatFromBits(data.m256i_u16[index]); }

			v_float16x16() {}

			static v_float16x16 loadu(const scalar_t* ptr) noexcept
			{
				return v_float16x16(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr)));
			}

			template<BasicArithmetic T>
			v_float16x16(
				T v0, T v1, T v2, T v3, T v4, T v5, T v6, T v7,
				T v8, T v9, T v10, T v11, T v12, T v13, T v14, T v15)
			{
				if constexpr (std::is_same_v<T, f16>)
				{
					data = _mm256_setr_epi16(v0.w, v1.w, v2.w, v3.w, v4.w, v5.w, v6.w, v7.w, v8.w, v9.w, v10.w, v11.w, v12.w, v13.w, v14.w, v15.w);
				}
				else
				{
					data = _mm256_setr_epi16(
						f16(v0).w, f16(v1).w, f16(v2).w, f16(v3).w, f16(v4).w, f16(v5).w, f16(v6).w, f16(v7).w,
						f16(v8).w, f16(v9).w, f16(v10).w, f16(v11).w, f16(v12).w, f16(v13).w, f16(v14).w, f16(v15).w
					);
				}
			}


			explicit v_float16x16(const scalar_t* arr) : data(_mm256_load_si256(reinterpret_cast<const __m256i*>(arr))) {}
			explicit v_float16x16(vector_t data) : data(data) {}

			operator vector_t() noexcept { return data; }

			template<BasicArithmetic T> explicit v_float16x16(const T value)
			{
				if constexpr (std::is_same_v<T, f16>)
				{
					this->data = _mm256_set1_epi16(value.w);
				}
				else
				{
					f64 temp = static_cast<f64>(value);
					f16 res(temp);
					this->data = _mm256_set1_epi16(res.w);
				}
			}

			void download(scalar_t* ptr) const { _mm256_store_si256(reinterpret_cast<__m256i*>(ptr), data); }
			void downloadu(scalar_t* ptr) const { _mm256_storeu_si256(reinterpret_cast<__m256i*>(ptr), data); }
			void streamback(scalar_t* ptr) const noexcept { _mm256_stream_si256(reinterpret_cast<__m256i*>(ptr), data); }
			std::string str() const { return vector_to_string(*this); }
		};

		struct alignas(32) v_float32x8
		{
			using scalar_t = f32;
			using vector_t = __m256;
			using upper_t = void;
			using lower_t = v_float32x4;

			enum { batch_size = 8 };
			vector_t alignas(32) data;

			void* operator new(std::size_t) = delete;
			void operator delete(void*) = delete;

			const scalar_t& operator[](u8 index) const { return data.m256_f32[index]; }
			scalar_t& operator[](u8 index) { return data.m256_f32[index]; }

			v_float32x8() {}
			v_float32x8(scalar_t v0, scalar_t v1, scalar_t v2, scalar_t v3, scalar_t v4, scalar_t v5, scalar_t v6, scalar_t v7)
				: data(_mm256_setr_ps(v0, v1, v2, v3, v4, v5, v6, v7))
			{
			}
			explicit v_float32x8(const scalar_t* ptr) : data(_mm256_load_ps(ptr)) {}

			static v_float32x8 loadu(const scalar_t* ptr) noexcept
			{
				return v_float32x8(_mm256_loadu_ps(ptr));
			}

			operator vector_t() noexcept { return data; }

			template<BasicArithmetic T>
			explicit v_float32x8(const T val) : data(_mm256_set1_ps(static_cast<typename v_float32x8::scalar_t>(val))) {}

			explicit v_float32x8(vector_t data) : data(data) {}
			void download(scalar_t* ptr) const { _mm256_store_ps(ptr, data); }
			void downloadu(scalar_t* ptr) const { _mm256_storeu_ps(ptr, data); }
			void streamback(scalar_t* ptr) const noexcept { _mm256_stream_ps(ptr, data); }
			std::string str() const { return vector_to_string(*this); }
		};

		struct alignas(32) v_float64x4
		{
			using scalar_t = f64;
			using vector_t = __m256d;
			using upper_t = void;
			using lower_t = v_float64x2;

			enum { batch_size = 4 };
			vector_t alignas(32) data;

			const scalar_t& operator[](u8 index) const { return data.m256d_f64[index]; }
			scalar_t& operator[](u8 index) { return data.m256d_f64[index]; }

			void* operator new(std::size_t) = delete;
			void operator delete(void*) = delete;

			v_float64x4() {}
			v_float64x4(scalar_t v0, scalar_t v1, scalar_t v2, scalar_t v3) : data(_mm256_setr_pd(v0, v1, v2, v3)) {}
			explicit v_float64x4(const scalar_t* ptr) : data(_mm256_load_pd(ptr)) {}

			static v_float64x4 loadu(const scalar_t* ptr) noexcept
			{
				return v_float64x4(_mm256_loadu_pd(ptr));
			}

			operator vector_t() noexcept { return data; }

			template<BasicArithmetic T>
			explicit v_float64x4(const T val) : data(_mm256_set1_pd(static_cast<typename v_float64x4::scalar_t>(val))) {}

			explicit v_float64x4(vector_t data) : data(data) {}
			void download(scalar_t* ptr) const { _mm256_store_pd(ptr, data); }
			void downloadu(scalar_t* ptr) const { _mm256_storeu_pd(ptr, data); }
			void streamback(scalar_t* ptr) const noexcept { _mm256_stream_pd(ptr, data); }
			std::string str() const { return vector_to_string(*this); }
		};
	}
}

export namespace fy
{
	namespace simd
	{
		template<VectorType T> requires (is_avx_t<T>&& std::is_integral_v<typename T::scalar_t>)
		usize v_count_mask(const T& vec)
		{
			typename T::vector_t zero_mask = (vec == v_broadcast_zero<T>());
			if constexpr (sizeof(typename T::scalar_t) == sizeof(u64))
			{
				return std::popcount(_mm256_movemask_ps(_mm256_castsi256_ps(zero_mask)));
			}
			else
			{
				return std::popcount(_mm256_movemask_epi8(zero_mask)) / sizeof(typename T::scalar_t);
			}
		}

		template<typename T> requires (std::is_floating_point_v<typename T::scalar_t> || std::is_same_v<f16, typename T::scalar_t>)
		usize v_count_mask(const T& vec)
		{
			if constexpr (std::is_same_v<typename T::scalar_t, f32>)
			{
				typename T::vector_t zero_mask = _mm256_cmp_ps(vec.data, _mm256_setzero_ps(), _CMP_EQ_OQ);
				return std::popcount(_mm256_movemask_ps(zero_mask));
			}
			else if constexpr (std::is_same_v<typename T::scalar_t, f64>)
			{
				typename T::vector_t zero_mask = _mm256_cmp_pd(vec.data, _mm256_setzero_pd(), _CMP_EQ_OQ);
				return std::popcount(_mm256_movemask_pd(zero_mask));
			}
			else if constexpr (std::is_same_v<typename T::scalar_t, f16>)
			{
				__m256 f32_data = _mm256_cvtph_ps(vec.data);
				typename T::vector_t zero_mask = _mm256_cmp_ps(f32_data, _mm256_setzero_ps(), _CMP_EQ_OQ);
				return std::popcount(_mm256_movemask_ps(zero_mask)) / 2;
			}
			else
			{
				std::unreachable();
			}
		}


		template<VectorType Vec_t>
		Vec_t v_broadcast_zero()
		{
			if constexpr (std::is_integral_v<typename Vec_t::scalar_t>)
			{
				if constexpr (sizeof(Vec_t) == sizeof(__m128))
				{
					return Vec_t(_mm_setzero_si128());
				}
				else if constexpr (sizeof(Vec_t) == sizeof(__m256))
				{
					return Vec_t(_mm256_setzero_si256());
				}
				else
				{
					std::unreachable();
				}
			}
			else if constexpr (std::is_same_v<typename Vec_t::scalar_t, f16>)
			{
				if constexpr (sizeof(Vec_t) == sizeof(__m128))
				{
					return Vec_t(_mm_setzero_ph());
				}
				else if constexpr (sizeof(Vec_t) == sizeof(__m256))
				{
					return Vec_t(_mm256_setzero_ph());
				}
				else
				{
					std::unreachable();
				}
			}
			else if constexpr (std::is_same_v<typename Vec_t::scalar_t, f32>)
			{
				if constexpr (sizeof(Vec_t) == sizeof(__m128))
				{
					return Vec_t(_mm_setzero_ps());
				}
				else if constexpr (sizeof(Vec_t) == sizeof(__m256))
				{
					return Vec_t(_mm256_setzero_ps());
				}
				else
				{
					std::unreachable();
				}
			}
			else if constexpr (std::is_same_v<Vec_t::scalar_t, f64>)
			{
				if constexpr (sizeof(Vec_t) == sizeof(__m128))
				{
					return Vec_t(_mm_setzero_pd());
				}
				else if constexpr (sizeof(Vec_t) == sizeof(__m256))
				{
					return Vec_t(_mm256_setzero_pd());
				}
				else
				{
					std::unreachable();
				}
			}
			else
			{
				std::unreachable();
			}
		}
	}
}


export namespace fy
{
	namespace simd
	{
		template<VectorType T> __forceinline T v_add(const T& left, const T& right) { std::unreachable(); }
		template<VectorType T> __forceinline T operator + (const T& left, const T& right) { return v_add(left, right); }
		template<VectorType T> __forceinline T& operator += (T& left, const T& right) { left = v_add(left, right); return left; }

		template<VectorType Left, BasicArithmetic Right> __forceinline Left operator + (const Left& left, Right right) { return v_add(left, Left(saturate_cast<typename Left::scalar_t>(right))); }
		template<VectorType Left, BasicArithmetic Right> __forceinline Left& operator += (Left& left, Right right) { left = v_add(left, Left(saturate_cast<typename Left::scalar_t>(right))); return left; }

		template<> v_uint8x16 v_add(const v_uint8x16& left, const v_uint8x16& right) { return v_uint8x16(_mm_adds_epu8(left.data, right.data)); }
		template<> v_uint16x8 v_add(const v_uint16x8& left, const v_uint16x8& right) { return v_uint16x8(_mm_adds_epu16(left.data, right.data)); }
		template<> v_uint32x4 v_add(const v_uint32x4& left, const v_uint32x4& right) { return v_uint32x4(_mm_add_epi32(left.data, right.data)); }
		template<> v_uint64x2 v_add(const v_uint64x2& left, const v_uint64x2& right) { return v_uint64x2(_mm_add_epi64(left.data, right.data)); }
		template<> v_int8x16 v_add(const v_int8x16& left, const v_int8x16& right) { return v_int8x16(_mm_adds_epi8(left.data, right.data)); }
		template<> v_int16x8 v_add(const v_int16x8& left, const v_int16x8& right) { return v_int16x8(_mm_adds_epi16(left.data, right.data)); }
		template<> v_int32x4 v_add(const v_int32x4& left, const v_int32x4& right) { return v_int32x4(_mm_add_epi32(left.data, right.data)); }
		template<> v_int64x2 v_add(const v_int64x2& left, const v_int64x2& right) { return v_int64x2(_mm_add_epi64(left.data, right.data)); }
		template<> v_float32x4 v_add(const v_float32x4& left, const v_float32x4& right) { return v_float32x4(_mm_add_ps(left.data, right.data)); }
		template<> v_float64x2 v_add(const v_float64x2& left, const v_float64x2& right) { return v_float64x2(_mm_add_pd(left.data, right.data)); }

		template<> v_uint8x32 v_add(const v_uint8x32& left, const v_uint8x32& right) { return v_uint8x32(_mm256_adds_epu8(left.data, right.data)); }
		template<> v_uint16x16 v_add(const v_uint16x16& left, const v_uint16x16& right) { return v_uint16x16(_mm256_adds_epu16(left.data, right.data)); }
		template<> v_uint32x8 v_add(const v_uint32x8& left, const v_uint32x8& right) { return v_uint32x8(_mm256_add_epi32(left.data, right.data)); }
		template<> v_uint64x4 v_add(const v_uint64x4& left, const v_uint64x4& right) { return v_uint64x4(_mm256_add_epi64(left.data, right.data)); }
		template<> v_int8x32 v_add(const v_int8x32& left, const v_int8x32& right) { return v_int8x32(_mm256_adds_epi8(left.data, right.data)); }
		template<> v_int16x16 v_add(const v_int16x16& left, const v_int16x16& right) { return v_int16x16(_mm256_adds_epi16(left.data, right.data)); }
		template<> v_int32x8 v_add(const v_int32x8& left, const v_int32x8& right) { return v_int32x8(_mm256_add_epi32(left.data, right.data)); }
		template<> v_int64x4 v_add(const v_int64x4& left, const v_int64x4& right) { return v_int64x4(_mm256_add_epi64(left.data, right.data)); }
		template<> v_float32x8 v_add(const v_float32x8& left, const v_float32x8& right) { return v_float32x8(_mm256_add_ps(left.data, right.data)); }
		template<> v_float64x4 v_add(const v_float64x4& left, const v_float64x4& right) { return v_float64x4(_mm256_add_pd(left.data, right.data)); }

		template<> v_float16x8 v_add(const v_float16x8& left, const v_float16x8& right)
		{
			__m256 a_f32 = _mm256_cvtph_ps(left.data);
			__m256 b_f32 = _mm256_cvtph_ps(right.data);
			__m256 sum_f32 = _mm256_add_ps(a_f32, b_f32);
			__m128i res = _mm256_cvtps_ph(sum_f32, 0x00);
			return v_float16x8(res);
		}

		template<> v_float16x16 v_add(const v_float16x16& left, const v_float16x16& right)
		{
			__m256 a_f32_low = _mm256_cvtph_ps(_mm256_extracti128_si256(left.data, 0));
			__m256 b_f32_low = _mm256_cvtph_ps(_mm256_extracti128_si256(right.data, 0));
			__m256 sum_f32_low = _mm256_add_ps(a_f32_low, b_f32_low);

			__m256 a_f32_high = _mm256_cvtph_ps(_mm256_extracti128_si256(left.data, 1));
			__m256 b_f32_high = _mm256_cvtph_ps(_mm256_extracti128_si256(right.data, 1));
			__m256 sum_f32_high = _mm256_add_ps(a_f32_high, b_f32_high);

			__m128i res_low = _mm256_cvtps_ph(sum_f32_low, 0);
			__m128i res_high = _mm256_cvtps_ph(sum_f32_high, 0);
			__m256i res = _mm256_inserti128_si256(_mm256_castsi128_si256(res_low), res_high, 1);
			return v_float16x16(res);
		}
	}
}

export namespace fy
{
	namespace simd
	{
		template<VectorType T> __forceinline T v_sub(const T& left, const T& right) { std::unreachable(); }
		template<VectorType T> __forceinline T operator - (const T& left, const T& right) { return v_sub(left, right); }
		template<VectorType T> __forceinline T& operator -= (T& left, const T& right) { left = v_sub(left, right); return left; }

		template<VectorType Left, BasicArithmetic Right> __forceinline Left operator - (const Left& left, Right right) { return v_sub(left, Left(saturate_cast<typename Left::scalar_t>(right))); }
		template<VectorType Left, BasicArithmetic Right> __forceinline Left& operator -= (Left& left, Right right) { left = v_sub(left, Left(saturate_cast<typename Left::scalar_t>(right))); return left; }

		template<> v_uint8x16 v_sub(const v_uint8x16& left, const v_uint8x16& right) { return v_uint8x16(_mm_subs_epu8(left.data, right.data)); }
		template<> v_uint16x8 v_sub(const v_uint16x8& left, const v_uint16x8& right) { return v_uint16x8(_mm_subs_epu16(left.data, right.data)); }
		template<> v_uint32x4 v_sub(const v_uint32x4& left, const v_uint32x4& right) { return v_uint32x4(_mm_sub_epi32(left.data, right.data)); }
		template<> v_uint64x2 v_sub(const v_uint64x2& left, const v_uint64x2& right) { return v_uint64x2(_mm_sub_epi64(left.data, right.data)); }
		template<> v_int8x16 v_sub(const v_int8x16& left, const v_int8x16& right) { return v_int8x16(_mm_subs_epi8(left.data, right.data)); }
		template<> v_int16x8 v_sub(const v_int16x8& left, const v_int16x8& right) { return v_int16x8(_mm_subs_epi16(left.data, right.data)); }
		template<> v_int32x4 v_sub(const v_int32x4& left, const v_int32x4& right) { return v_int32x4(_mm_sub_epi32(left.data, right.data)); }
		template<> v_int64x2 v_sub(const v_int64x2& left, const v_int64x2& right) { return v_int64x2(_mm_sub_epi64(left.data, right.data)); }
		template<> v_float32x4 v_sub(const v_float32x4& left, const v_float32x4& right) { return v_float32x4(_mm_sub_ps(left.data, right.data)); }
		template<> v_float64x2 v_sub(const v_float64x2& left, const v_float64x2& right) { return v_float64x2(_mm_sub_pd(left.data, right.data)); }

		template<> v_uint8x32 v_sub(const v_uint8x32& left, const v_uint8x32& right) { return v_uint8x32(_mm256_subs_epu8(left.data, right.data)); }
		template<> v_uint16x16 v_sub(const v_uint16x16& left, const v_uint16x16& right) { return v_uint16x16(_mm256_subs_epu16(left.data, right.data)); }
		template<> v_uint32x8 v_sub(const v_uint32x8& left, const v_uint32x8& right) { return v_uint32x8(_mm256_sub_epi32(left.data, right.data)); }
		template<> v_uint64x4 v_sub(const v_uint64x4& left, const v_uint64x4& right) { return v_uint64x4(_mm256_sub_epi64(left.data, right.data)); }
		template<> v_int8x32 v_sub(const v_int8x32& left, const v_int8x32& right) { return v_int8x32(_mm256_subs_epi8(left.data, right.data)); }
		template<> v_int16x16 v_sub(const v_int16x16& left, const v_int16x16& right) { return v_int16x16(_mm256_subs_epi16(left.data, right.data)); }
		template<> v_int32x8 v_sub(const v_int32x8& left, const v_int32x8& right) { return v_int32x8(_mm256_sub_epi32(left.data, right.data)); }
		template<> v_int64x4 v_sub(const v_int64x4& left, const v_int64x4& right) { return v_int64x4(_mm256_sub_epi64(left.data, right.data)); }
		template<> v_float32x8 v_sub(const v_float32x8& left, const v_float32x8& right) { return v_float32x8(_mm256_sub_ps(left.data, right.data)); }
		template<> v_float64x4 v_sub(const v_float64x4& left, const v_float64x4& right) { return v_float64x4(_mm256_sub_pd(left.data, right.data)); }

		template<> v_float16x8 v_sub(const v_float16x8& left, const v_float16x8& right)
		{
			__m256 a_f32 = _mm256_cvtph_ps(left.data);
			__m256 b_f32 = _mm256_cvtph_ps(right.data);
			__m256 sum_f32 = _mm256_sub_ps(a_f32, b_f32);
			__m128i res = _mm256_cvtps_ph(sum_f32, 0x00);
			return v_float16x8(res);
		}

		template<> v_float16x16 v_sub(const v_float16x16& left, const v_float16x16& right)
		{
			__m256 a_f32_low = _mm256_cvtph_ps(_mm256_extracti128_si256(left.data, 0));
			__m256 b_f32_low = _mm256_cvtph_ps(_mm256_extracti128_si256(right.data, 0));
			__m256 sum_f32_low = _mm256_sub_ps(a_f32_low, b_f32_low);

			__m256 a_f32_high = _mm256_cvtph_ps(_mm256_extracti128_si256(left.data, 1));
			__m256 b_f32_high = _mm256_cvtph_ps(_mm256_extracti128_si256(right.data, 1));
			__m256 sum_f32_high = _mm256_sub_ps(a_f32_high, b_f32_high);

			__m128i res_low = _mm256_cvtps_ph(sum_f32_low, 0);
			__m128i res_high = _mm256_cvtps_ph(sum_f32_high, 0);
			__m256i res = _mm256_inserti128_si256(_mm256_castsi128_si256(res_low), res_high, 1);
			return v_float16x16(res);
		}
	}
}

export namespace fy
{
	namespace simd
	{
		template<VectorType T> T v_subs(const T& a, const T& b) { return v_sub(a, b); }
	}
}

export namespace fy
{
	namespace simd
	{
		template<VectorType T> __forceinline T v_mul(const T& left, const T& right) { std::unreachable(); }
		template<VectorType T> __forceinline T operator * (const T& left, const T& right) { return v_mul(left, right); }
		template<VectorType T> __forceinline T& operator *= (T& left, const T& right) { left = v_mul(left, right); return left; }

		template<VectorType Left, BasicArithmetic Right> __forceinline Left operator * (const Left& left, Right right) { return v_mul(left, Left(saturate_cast<typename Left::scalar_t>(right))); }
		template<VectorType Left, BasicArithmetic Right> __forceinline Left& operator *= (Left& left, Right right) { left = v_mul(left, Left(saturate_cast<typename Left::scalar_t>(right))); return left; }

		template<> v_uint8x16 v_mul(const v_uint8x16& left, const v_uint8x16& right)
		{
			__m128i lo = _mm_mullo_epi16(_mm_unpacklo_epi8(left.data, _mm_setzero_si128()), _mm_unpacklo_epi8(right.data, _mm_setzero_si128()));
			__m128i hi = _mm_mullo_epi16(_mm_unpackhi_epi8(left.data, _mm_setzero_si128()), _mm_unpackhi_epi8(right.data, _mm_setzero_si128()));
			return v_uint8x16(_mm_packus_epi16(lo, hi));
		}

		template<> v_uint16x8 v_mul(const v_uint16x8& left, const v_uint16x8& right) { return v_uint16x8(_mm_mullo_epi16(left.data, right.data)); }
		template<> v_uint32x4 v_mul(const v_uint32x4& left, const v_uint32x4& right) { return v_uint32x4(_mm_mullo_epi32(left.data, right.data)); }
		template<> v_uint64x2 v_mul(const v_uint64x2& left, const v_uint64x2& right) { return v_uint64x2(_mm_set_epi64x(left.data.m128i_u64[1] * right.data.m128i_u64[1], left.data.m128i_u64[0] * right.data.m128i_u64[0])); }

		template<> v_int8x16 v_mul(const v_int8x16& left, const v_int8x16& right)
		{
			__m128i lo = _mm_mullo_epi16(_mm_srai_epi16(_mm_unpacklo_epi8(left.data, left.data), 8), _mm_srai_epi16(_mm_unpacklo_epi8(right.data, right.data), 8));
			__m128i hi = _mm_mullo_epi16(_mm_srai_epi16(_mm_unpackhi_epi8(left.data, left.data), 8), _mm_srai_epi16(_mm_unpackhi_epi8(right.data, right.data), 8));
			return v_int8x16(_mm_packs_epi16(lo, hi));
		}

		template<> v_int16x8 v_mul(const v_int16x8& left, const v_int16x8& right) { return v_int16x8(_mm_mullo_epi16(left.data, right.data)); }
		template<> v_int32x4 v_mul(const v_int32x4& left, const v_int32x4& right) { return v_int32x4(_mm_mullo_epi32(left.data, right.data)); }
		template<> v_int64x2 v_mul(const v_int64x2& left, const v_int64x2& right) { return v_int64x2(_mm_set_epi64x(left.data.m128i_i64[1] * right.data.m128i_i64[1], left.data.m128i_i64[0] * right.data.m128i_i64[0])); }

		template<> v_float16x8 v_mul(const v_float16x8& left, const v_float16x8& right)
		{
			__m256 a_f32 = _mm256_cvtph_ps(left.data);
			__m256 b_f32 = _mm256_cvtph_ps(right.data);
			__m256 sum_f32 = _mm256_mul_ps(a_f32, b_f32);
			__m128i res = _mm256_cvtps_ph(sum_f32, 0x00);
			return v_float16x8(res);
		}

		template<> v_float32x4 v_mul(const v_float32x4& left, const v_float32x4& right) { return v_float32x4(_mm_mul_ps(left.data, right.data)); }
		template<> v_float64x2 v_mul(const v_float64x2& left, const v_float64x2& right) { return v_float64x2(_mm_mul_pd(left.data, right.data)); }

		template<> v_uint8x32 v_mul(const v_uint8x32& left, const v_uint8x32& right)
		{
			__m256i left_low = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(left.data));
			__m256i left_high = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(left.data, 1));
			__m256i right_low = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(right.data));
			__m256i right_high = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(right.data, 1));

			__m256i result_low = _mm256_mullo_epi16(left_low, right_low);
			__m256i result_high = _mm256_mullo_epi16(left_high, right_high);

			__m256i result = _mm256_packus_epi16(result_low, result_high);
			__m256i replace_result = _mm256_permute4x64_epi64(result, _MM_SHUFFLE(3, 1, 2, 0));
			return v_uint8x32(replace_result);
		}

		template<> v_uint16x16 v_mul(const v_uint16x16& left, const v_uint16x16& right) { return v_uint16x16(_mm256_mullo_epi16(left.data, right.data)); }
		template<> v_uint32x8 v_mul(const v_uint32x8& left, const v_uint32x8& right) { return v_uint32x8(_mm256_mullo_epi32(left.data, right.data)); }

		template<> v_uint64x4 v_mul(const v_uint64x4& left, const v_uint64x4& right)
		{
			__m256i res{};
			for (int i = 0; i < v_uint64x4::batch_size; ++i)
			{
				res.m256i_u64[i] = left.data.m256i_u64[i] * right.data.m256i_u64[i];
			}
			return v_uint64x4(res);
		}

		template<> v_int8x32 v_mul(const v_int8x32& left, const v_int8x32& right)
		{
			__m256i left_low = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(left.data));
			__m256i left_high = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(left.data, 1));
			__m256i right_low = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(right.data));
			__m256i right_high = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(right.data, 1));

			__m256i result_low = _mm256_mullo_epi16(left_low, right_low);
			__m256i result_high = _mm256_mullo_epi16(left_high, right_high);

			__m256i result = _mm256_packs_epi16(result_low, result_high);
			__m256i replace_result = _mm256_permute4x64_epi64(result, _MM_SHUFFLE(3, 1, 2, 0));
			return v_int8x32(replace_result);
		}

		template<> v_int16x16 v_mul(const v_int16x16& left, const v_int16x16& right) { return v_int16x16(_mm256_mullo_epi16(left.data, right.data)); }
		template<> v_int32x8 v_mul(const v_int32x8& left, const v_int32x8& right) { return v_int32x8(_mm256_mullo_epi32(left.data, right.data)); }

		template<> v_int64x4 v_mul(const v_int64x4& left, const v_int64x4& right)
		{
			__m256i res{};
			for (int i = 0; i < v_int64x4::batch_size; ++i)
			{
				res.m256i_i64[i] = left.data.m256i_i64[i] * right.data.m256i_i64[i];
			}
			return v_int64x4(res);
		}

		template<> v_float16x16 v_mul(const v_float16x16& left, const v_float16x16& right)
		{
			__m256 a_f32_low = _mm256_cvtph_ps(_mm256_extracti128_si256(left.data, 0));
			__m256 b_f32_low = _mm256_cvtph_ps(_mm256_extracti128_si256(right.data, 0));
			__m256 sum_f32_low = _mm256_mul_ps(a_f32_low, b_f32_low);

			__m256 a_f32_high = _mm256_cvtph_ps(_mm256_extracti128_si256(left.data, 1));
			__m256 b_f32_high = _mm256_cvtph_ps(_mm256_extracti128_si256(right.data, 1));
			__m256 sum_f32_high = _mm256_mul_ps(a_f32_high, b_f32_high);

			__m128i res_low = _mm256_cvtps_ph(sum_f32_low, 0);
			__m128i res_high = _mm256_cvtps_ph(sum_f32_high, 0);
			__m256i res = _mm256_inserti128_si256(_mm256_castsi128_si256(res_low), res_high, 1);
			return v_float16x16(res);
		}

		template<> v_float32x8 v_mul(const v_float32x8& left, const v_float32x8& right) { return v_float32x8(_mm256_mul_ps(left.data, right.data)); }
		template<> v_float64x4 v_mul(const v_float64x4& left, const v_float64x4& right) { return v_float64x4(_mm256_mul_pd(left.data, right.data)); }
	}
}

export namespace fy
{
	namespace simd
	{
		template<VectorType T> __forceinline T v_div(const T& left, const T& right) { std::unreachable(); }
		template<VectorType T> __forceinline T operator / (const T& left, const T& right) { return v_div(left, right); }
		template<VectorType T> __forceinline T& operator /= (T& left, const T& right) { left = v_div(left, right); return left; }

		template<VectorType Left, BasicArithmetic Right> __forceinline Left operator / (const Left& left, Right right) { return v_div(left, Left(saturate_cast<typename Left::scalar_t>(right))); }
		template<VectorType Left, BasicArithmetic Right> __forceinline Left& operator /= (Left& left, Right right) { left = v_div(left, Left(saturate_cast<typename Left::scalar_t>(right))); return left; }


		template<> v_uint8x16 v_div(const v_uint8x16& left, const v_uint8x16& right) { return v_uint8x16(_mm_div_epu8(left.data, right.data)); }
		template<> v_uint16x8 v_div(const v_uint16x8& left, const v_uint16x8& right) { return v_uint16x8(_mm_div_epu16(left.data, right.data)); }
		template<> v_uint32x4 v_div(const v_uint32x4& left, const v_uint32x4& right) { return v_uint32x4(_mm_div_epu32(left.data, right.data)); }
		template<> v_uint64x2 v_div(const v_uint64x2& left, const v_uint64x2& right) { return v_uint64x2(_mm_div_epu64(left.data, right.data)); }
		template<> v_int8x16 v_div(const v_int8x16& left, const v_int8x16& right) { return v_int8x16(_mm_div_epi8(left.data, right.data)); }
		template<> v_int16x8 v_div(const v_int16x8& left, const v_int16x8& right) { return v_int16x8(_mm_div_epi16(left.data, right.data)); }
		template<> v_int32x4 v_div(const v_int32x4& left, const v_int32x4& right) { return v_int32x4(_mm_div_epi32(left.data, right.data)); }
		template<> v_int64x2 v_div(const v_int64x2& left, const v_int64x2& right) { return v_int64x2(_mm_div_epi64(left.data, right.data)); }

		template<> v_float16x8 v_div(const v_float16x8& left, const v_float16x8& right)
		{
			__m256 a_f32 = _mm256_cvtph_ps(left.data);
			__m256 b_f32 = _mm256_cvtph_ps(right.data);
			__m256 sum_f32 = _mm256_div_ps(a_f32, b_f32);
			__m128i res = _mm256_cvtps_ph(sum_f32, 0x00);
			return v_float16x8(res);
		}

		template<> v_float32x4 v_div(const v_float32x4& left, const v_float32x4& right) { return v_float32x4(_mm_div_ps(left.data, right.data)); }
		template<> v_float64x2 v_div(const v_float64x2& left, const v_float64x2& right) { return v_float64x2(_mm_div_pd(left.data, right.data)); }

		template<> v_uint8x32 v_div(const v_uint8x32& left, const v_uint8x32& right) { return v_uint8x32(_mm256_div_epu8(left.data, right.data)); }
		template<> v_uint16x16 v_div(const v_uint16x16& left, const v_uint16x16& right) { return v_uint16x16(_mm256_div_epu16(left.data, right.data)); }
		template<> v_uint32x8 v_div(const v_uint32x8& left, const v_uint32x8& right) { return v_uint32x8(_mm256_div_epu32(left.data, right.data)); }
		template<> v_uint64x4 v_div(const v_uint64x4& left, const v_uint64x4& right) { return v_uint64x4(_mm256_div_epu64(left.data, right.data)); }
		template<> v_int8x32 v_div(const v_int8x32& left, const v_int8x32& right) { return v_int8x32(_mm256_div_epi8(left.data, right.data)); }
		template<> v_int16x16 v_div(const v_int16x16& left, const v_int16x16& right) { return v_int16x16(_mm256_div_epi16(left.data, right.data)); }
		template<> v_int32x8 v_div(const v_int32x8& left, const v_int32x8& right) { return v_int32x8(_mm256_div_epi32(left.data, right.data)); }
		template<> v_int64x4 v_div(const v_int64x4& left, const v_int64x4& right) { return v_int64x4(_mm256_div_epi64(left.data, right.data)); }

		template<> v_float16x16 v_div(const v_float16x16& left, const v_float16x16& right)
		{
			__m256 a_f32_low = _mm256_cvtph_ps(_mm256_extracti128_si256(left.data, 0));
			__m256 b_f32_low = _mm256_cvtph_ps(_mm256_extracti128_si256(right.data, 0));
			__m256 sum_f32_low = _mm256_div_ps(a_f32_low, b_f32_low);

			__m256 a_f32_high = _mm256_cvtph_ps(_mm256_extracti128_si256(left.data, 1));
			__m256 b_f32_high = _mm256_cvtph_ps(_mm256_extracti128_si256(right.data, 1));
			__m256 sum_f32_high = _mm256_div_ps(a_f32_high, b_f32_high);

			__m128i res_low = _mm256_cvtps_ph(sum_f32_low, 0);
			__m128i res_high = _mm256_cvtps_ph(sum_f32_high, 0);
			__m256i res = _mm256_inserti128_si256(_mm256_castsi128_si256(res_low), res_high, 1);
			return v_float16x16(res);
		}

		template<> v_float32x8 v_div(const v_float32x8& left, const v_float32x8& right) { return v_float32x8(_mm256_div_ps(left.data, right.data)); }
		template<> v_float64x4 v_div(const v_float64x4& left, const v_float64x4& right) { return v_float64x4(_mm256_div_pd(left.data, right.data)); }
	}
}

export namespace fy
{
	namespace simd
	{
		template<VectorType T> T v_shift_left_logical(const T&, i32) { std::unreachable(); }

		template<> v_int16x8 v_shift_left_logical(const v_int16x8& src, i32 shift_amount) { return v_int16x8(_mm_slli_epi16(src.data, shift_amount)); }
		template<> v_int32x4 v_shift_left_logical(const v_int32x4& src, i32 shift_amount) { return v_int32x4(_mm_slli_epi32(src.data, shift_amount)); }
		template<> v_int64x2 v_shift_left_logical(const v_int64x2& src, i32 shift_amount) { return v_int64x2(_mm_slli_epi64(src.data, shift_amount)); }
		template<> v_uint16x8 v_shift_left_logical(const v_uint16x8& src, i32 shift_amount) { return v_uint16x8(_mm_slli_epi16(src.data, shift_amount)); }
		template<> v_uint32x4 v_shift_left_logical(const v_uint32x4& src, i32 shift_amount) { return v_uint32x4(_mm_slli_epi32(src.data, shift_amount)); }
		template<> v_uint64x2 v_shift_left_logical(const v_uint64x2& src, i32 shift_amount) { return v_uint64x2(_mm_slli_epi64(src.data, shift_amount)); }

		template<> v_int16x16 v_shift_left_logical(const v_int16x16& src, i32 shift_amount) { return v_int16x16(_mm256_slli_epi16(src.data, shift_amount)); }
		template<> v_int32x8 v_shift_left_logical(const v_int32x8& src, i32 shift_amount) { return v_int32x8(_mm256_slli_epi32(src.data, shift_amount)); }
		template<> v_int64x4 v_shift_left_logical(const v_int64x4& src, i32 shift_amount) { return v_int64x4(_mm256_slli_epi64(src.data, shift_amount)); }
		template<> v_uint16x16 v_shift_left_logical(const v_uint16x16& src, i32 shift_amount) { return v_uint16x16(_mm256_slli_epi16(src.data, shift_amount)); }
		template<> v_uint32x8 v_shift_left_logical(const v_uint32x8& src, i32 shift_amount) { return v_uint32x8(_mm256_slli_epi32(src.data, shift_amount)); }
		template<> v_uint64x4 v_shift_left_logical(const v_uint64x4& src, i32 shift_amount) { return v_uint64x4(_mm256_slli_epi64(src.data, shift_amount)); }

		template<VectorType T> T operator << (const T& src, i32 shift_amount) { return v_shift_left_logical(src, shift_amount); }

		template<VectorType T> T& operator <<= (T& src, i32 shift_amount)
		{
			src = v_shift_left_logical(src, shift_amount);
			return src;
		}
	}
}

export namespace fy
{
	namespace simd
	{
		template<VectorType T> T v_shift_right_arithmetic(const T&, i32) { std::unreachable(); }
		template<VectorType T> T v_shift_right_logical(const T&, i32) { std::unreachable(); }

		template<> v_int16x8 v_shift_right_arithmetic(const v_int16x8& src, i32 shift_amount) { return v_int16x8(_mm_srai_epi16(src.data, shift_amount)); }
		template<> v_int32x4 v_shift_right_arithmetic(const v_int32x4& src, i32 shift_amount) { return v_int32x4(_mm_srai_epi32(src.data, shift_amount)); }
		template<> v_int64x2 v_shift_right_arithmetic(const v_int64x2& src, i32 shift_amount)
		{
			__m128i mask = _mm_srai_epi32(_mm_shuffle_epi32(src.data, _MM_SHUFFLE(3, 3, 1, 1)), 31);
			__m128i shifted = _mm_srli_epi64(src.data, shift_amount);
			__m128i high_bits = _mm_slli_epi64(mask, 64 - shift_amount);
			return v_int64x2(_mm_or_si128(shifted, high_bits));
		}

		template<> v_uint16x8 v_shift_right_logical(const v_uint16x8& src, i32 shift_amount) { return v_uint16x8(_mm_srli_epi16(src.data, shift_amount)); }
		template<> v_uint32x4 v_shift_right_logical(const v_uint32x4& src, i32 shift_amount) { return v_uint32x4(_mm_srli_epi32(src.data, shift_amount)); }
		template<> v_uint64x2 v_shift_right_logical(const v_uint64x2& src, i32 shift_amount) { return v_uint64x2(_mm_srli_epi64(src.data, shift_amount)); }

		template<> v_int16x16 v_shift_right_arithmetic(const v_int16x16& src, i32 shift_amount) { return v_int16x16(_mm256_srai_epi16(src.data, shift_amount)); }
		template<> v_int32x8 v_shift_right_arithmetic(const v_int32x8& src, i32 shift_amount) { return v_int32x8(_mm256_srai_epi32(src.data, shift_amount)); }
		template<> v_int64x4 v_shift_right_arithmetic(const v_int64x4& src, i32 shift_amount)
		{
			__m256i mask = _mm256_srai_epi32(_mm256_shuffle_epi32(src.data, _MM_SHUFFLE(3, 3, 1, 1)), 31);
			__m256i shifted = _mm256_srli_epi64(src.data, shift_amount);
			__m256i high_bits = _mm256_slli_epi64(mask, 64 - shift_amount);
			return v_int64x4(_mm256_or_si256(shifted, high_bits));
		}

		template<> v_uint16x16 v_shift_right_logical(const v_uint16x16& src, i32 shift_amount) { return v_uint16x16(_mm256_srli_epi16(src.data, shift_amount)); }
		template<> v_uint32x8 v_shift_right_logical(const v_uint32x8& src, i32 shift_amount) { return v_uint32x8(_mm256_srli_epi32(src.data, shift_amount)); }
		template<> v_uint64x4 v_shift_right_logical(const v_uint64x4& src, i32 shift_amount) { return v_uint64x4(_mm256_srli_epi64(src.data, shift_amount)); }

		template<VectorType T>
		T operator >> (const T& src, i32 shift_amount)
		{
			if constexpr (std::is_signed_v<typename T::value_type>)
			{
				return v_shift_right_arithmetic(src, shift_amount);
			}
			else
			{
				return v_shift_right_logical(src, shift_amount);
			}
		}

		template<VectorType T>
		T& operator >>= (T& src, i32 shift_amount)
		{
			if constexpr (std::is_signed_v<typename T::value_type>)
			{
				src = v_shift_right_arithmetic(src, shift_amount);
			}
			else
			{
				src = v_shift_right_logical(src, shift_amount);
			}
			return src;
		}
	}

}

export namespace fy
{
	namespace simd
	{
		template<VectorType T> T v_fused_mul_add(const T& mul_left, const T& mul_right, const T& add_right)
		{
			T mul_res = v_mul<T>(mul_left, mul_right);
			T res = v_add<T>(mul_res, add_right);
			return res;
		}

		template<> v_float16x8 v_fused_mul_add(const v_float16x8& mul_left, const v_float16x8& mul_right, const v_float16x8& add_right)
		{
			__m256 mul_left_f32 = _mm256_cvtph_ps(mul_left.data);
			__m256 mul_right_f32 = _mm256_cvtph_ps(mul_right.data);
			__m256 add_right_f32 = _mm256_cvtph_ps(add_right.data);
			__m256 vres = _mm256_fmadd_ps(mul_left_f32, mul_right_f32, add_right_f32);
			__m128i res = _mm256_cvtps_ph(vres, 0x00);
			return v_float16x8(res);
		}

		//template<> v_float16x16 v_fused_mul_add(const v_float16x16& mul_left, const v_float16x16& mul_right, const v_float16x16& add_right)
		//{
		//	v_float16x8 mul_left_temp[2] = { v_float16x8(_mm256_extracti128_si256(mul_left.data, 0)), v_float16x8(_mm256_extracti128_si256(mul_left.data, 1)) };
		//	v_float16x8 mul_right_temp[2] = { v_float16x8(_mm256_extracti128_si256(mul_right.data, 0)), v_float16x8(_mm256_extracti128_si256(mul_right.data, 1)) };
		//	v_float16x8 add_right_temp[2] = { v_float16x8(_mm256_extracti128_si256(add_right.data, 0)), v_float16x8(_mm256_extracti128_si256(add_right.data, 1)) };

		//	v_float16x8 res_temp[2];

		//	for (usize i = 0; i < 2; ++i)
		//	{
		//		__m256 mul_left_f32 = _mm256_cvtph_ps(mul_left_temp[i].data);
		//		__m256 mul_right_f32 = _mm256_cvtph_ps(mul_right_temp[i].data);
		//		__m256 add_right_f32 = _mm256_cvtph_ps(add_right_temp[i].data);
		//		__m256 vres = _mm256_fmadd_ps(mul_left_f32, mul_right_f32, add_right_f32);
		//		__m128i res = _mm256_cvtps_ph(vres, 0x00);
		//		res_temp[i] = v_float16x8(res);
		//	}

		//	return v_float16x16(_mm256_inserti128_si256(_mm256_castsi128_si256(res_temp[0].data), res_temp[1].data, 1));
		//}

		template<> v_float32x4 v_fused_mul_add(const v_float32x4& mul_left, const v_float32x4& mul_right, const v_float32x4& add_right) { return v_float32x4(_mm_fmadd_ps(mul_left.data, mul_right.data, add_right.data)); }
		template<> v_float64x2 v_fused_mul_add(const v_float64x2& mul_left, const v_float64x2& mul_right, const v_float64x2& add_right) { return v_float64x2(_mm_fmadd_pd(mul_left.data, mul_right.data, add_right.data)); }
		template<> v_float32x8 v_fused_mul_add(const v_float32x8& mul_left, const v_float32x8& mul_right, const v_float32x8& add_right) { return v_float32x8(_mm256_fmadd_ps(mul_left.data, mul_right.data, add_right.data)); }
		template<> v_float64x4 v_fused_mul_add(const v_float64x4& mul_left, const v_float64x4& mul_right, const v_float64x4& add_right) { return v_float64x4(_mm256_fmadd_pd(mul_left.data, mul_right.data, add_right.data)); }
	}
}

export namespace fy
{
	namespace simd
	{
		template<VectorType T> T v_fused_mul_sub(const T& mul_left, const T& mul_right, const T& sub_right)
		{
			T mul_res = v_mul<T>(mul_left, mul_right);
			T res = v_sub<T>(mul_res, sub_right);
			return res;
		}

		template<> v_float32x4 v_fused_mul_sub(const v_float32x4& mul_left, const v_float32x4& mul_right, const v_float32x4& sub_right) { return v_float32x4(_mm_fmsub_ps(mul_left.data, mul_right.data, sub_right.data)); }
		template<> v_float64x2 v_fused_mul_sub(const v_float64x2& mul_left, const v_float64x2& mul_right, const v_float64x2& sub_right) { return v_float64x2(_mm_fmsub_pd(mul_left.data, mul_right.data, sub_right.data)); }
		template<> v_float32x8 v_fused_mul_sub(const v_float32x8& mul_left, const v_float32x8& mul_right, const v_float32x8& sub_right) { return v_float32x8(_mm256_fmsub_ps(mul_left.data, mul_right.data, sub_right.data)); }
		template<> v_float64x4 v_fused_mul_sub(const v_float64x4& mul_left, const v_float64x4& mul_right, const v_float64x4& sub_right) { return v_float64x4(_mm256_fmsub_pd(mul_left.data, mul_right.data, sub_right.data)); }
	}
}

export namespace fy
{
	namespace simd
	{
		template<VectorType T> __forceinline T v_lt(const T& left, const T& right) { std::unreachable(); }
		template<VectorType T> __forceinline T operator < (const T& left, const T& right) { return v_lt(left, right); }
		template<VectorType Left, BasicArithmetic Right> __forceinline Left operator < (const Left& left, Right right) { return v_lt(left, Left(saturate_cast<typename Left::scalar_t>(right))); }

		template<> v_uint8x16 v_lt(const v_uint8x16& left, const v_uint8x16& right) { return v_uint8x16(_mm_cmplt_epi8(left.data, right.data)); }
		template<> v_uint16x8 v_lt(const v_uint16x8& left, const v_uint16x8& right) { return v_uint16x8(_mm_cmplt_epi16(left.data, right.data)); }
		template<> v_uint32x4 v_lt(const v_uint32x4& left, const v_uint32x4& right) { return v_uint32x4(_mm_cmplt_epi32(left.data, right.data)); }

		template<> v_uint64x2 v_lt(const v_uint64x2& left, const v_uint64x2& right)
		{
			static const __m128i sign_bit = _mm_set1_epi64x(0x8000000000000000);
			__m128i l_xor = _mm_xor_si128(left.data, sign_bit);
			__m128i r_xor = _mm_xor_si128(right.data, sign_bit);
			return v_uint64x2(_mm_cmpgt_epi64(r_xor, l_xor));
		}

		template<> v_int8x16 v_lt(const v_int8x16& left, const v_int8x16& right) { return v_int8x16(_mm_cmplt_epi8(left.data, right.data)); }
		template<> v_int16x8 v_lt(const v_int16x8& left, const v_int16x8& right) { return v_int16x8(_mm_cmplt_epi16(left.data, right.data)); }
		template<> v_int32x4 v_lt(const v_int32x4& left, const v_int32x4& right) { return v_int32x4(_mm_cmplt_epi32(left.data, right.data)); }
		template<> v_int64x2 v_lt(const v_int64x2& left, const v_int64x2& right) { return v_int64x2(_mm_cmpgt_epi64(right.data, left.data)); }

		template<> v_float16x8 v_lt(const v_float16x8& left, const v_float16x8& right)
		{
			__m256 a_f32 = _mm256_cvtph_ps(left.data);
			__m256 b_f32 = _mm256_cvtph_ps(right.data);
			__m256 sum_f32 = _mm256_cmp_ps(a_f32, b_f32, _CMP_LT_OQ);
			__m128i res = _mm256_cvtps_ph(sum_f32, 0x00);
			return v_float16x8(res);
		}

		template<> v_float32x4 v_lt(const v_float32x4& left, const v_float32x4& right) { return v_float32x4(_mm_cmplt_ps(left.data, right.data)); }
		template<> v_float64x2 v_lt(const v_float64x2& left, const v_float64x2& right) { return v_float64x2(_mm_cmplt_pd(left.data, right.data)); }

		template<> v_uint8x32 v_lt(const v_uint8x32& left, const v_uint8x32& right)
		{
			__m256i diff = _mm256_sub_epi8(left.data, right.data);
			__m256i all_ones = _mm256_set1_epi8(char(-1));
			__m256i mask = _mm256_cmpgt_epi8(all_ones, diff);
			return v_uint8x32(mask);
		}

		template<> v_uint16x16 v_lt(const v_uint16x16& left, const v_uint16x16& right)
		{
			static const __m256i sign_bit = _mm256_set1_epi16(-32768);
			__m256i l_xor = _mm256_xor_si256(left.data, sign_bit);
			__m256i r_xor = _mm256_xor_si256(right.data, sign_bit);
			return v_uint16x16(_mm256_cmpgt_epi16(r_xor, l_xor));
		}

		template<> v_uint32x8 v_lt(const v_uint32x8& left, const v_uint32x8& right)
		{
			static const __m256i sign_bit = _mm256_set1_epi32(0x80000000);
			__m256i l_xor = _mm256_xor_si256(left.data, sign_bit);
			__m256i r_xor = _mm256_xor_si256(right.data, sign_bit);
			return v_uint32x8(_mm256_cmpgt_epi32(r_xor, l_xor));
		}

		template<> v_uint64x4 v_lt(const v_uint64x4& left, const v_uint64x4& right)
		{
			static const __m256i sign_bit = _mm256_set1_epi64x(0x8000000000000000);
			__m256i l_xor = _mm256_xor_si256(left.data, sign_bit);
			__m256i r_xor = _mm256_xor_si256(right.data, sign_bit);
			return v_uint64x4(_mm256_cmpgt_epi64(r_xor, l_xor));
		}

		template<> v_int8x32 v_lt(const v_int8x32& left, const v_int8x32& right) { return v_int8x32(_mm256_cmpgt_epi8(right.data, left.data)); }
		template<> v_int16x16 v_lt(const v_int16x16& left, const v_int16x16& right) { return v_int16x16(_mm256_cmpgt_epi16(right.data, left.data)); }
		template<> v_int32x8 v_lt(const v_int32x8& left, const v_int32x8& right) { return v_int32x8(_mm256_cmpgt_epi32(right.data, left.data)); }
		template<> v_int64x4 v_lt(const v_int64x4& left, const v_int64x4& right) { return v_int64x4(_mm256_cmpgt_epi64(right.data, left.data)); }

		template<> v_float16x16 v_lt(const v_float16x16& left, const v_float16x16& right)
		{
			__m256 a_f32_low = _mm256_cvtph_ps(_mm256_extracti128_si256(left.data, 0));
			__m256 b_f32_low = _mm256_cvtph_ps(_mm256_extracti128_si256(right.data, 0));
			__m256 sum_f32_low = _mm256_cmp_ps(a_f32_low, b_f32_low, _CMP_LT_OQ);

			__m256 a_f32_high = _mm256_cvtph_ps(_mm256_extracti128_si256(left.data, 1));
			__m256 b_f32_high = _mm256_cvtph_ps(_mm256_extracti128_si256(right.data, 1));
			__m256 sum_f32_high = _mm256_cmp_ps(a_f32_high, b_f32_high, _CMP_LT_OQ);

			__m128i res_low = _mm256_cvtps_ph(sum_f32_low, 0);
			__m128i res_high = _mm256_cvtps_ph(sum_f32_high, 0);
			__m256i res = _mm256_inserti128_si256(_mm256_castsi128_si256(res_low), res_high, 1);
			return v_float16x16(res);
		}

		template<> v_float32x8 v_lt(const v_float32x8& left, const v_float32x8& right) { return v_float32x8(_mm256_cmp_ps(left.data, right.data, _CMP_LT_OQ)); }
		template<> v_float64x4 v_lt(const v_float64x4& left, const v_float64x4& right) { return v_float64x4(_mm256_cmp_pd(left.data, right.data, _CMP_LT_OQ)); }
	}
}

export namespace fy
{
	namespace simd
	{
		template<Integral_VectorType T> T v_remainder(const T& left, const T& right) { std::unreachable(); }

		template<Integral_VectorType T> __forceinline T operator % (const T& left, const T& right)
		{
			return v_remainder(left, right);
		}

		template<Integral_VectorType Left, BasicArithmetic Right> __forceinline Left operator % (const Left& left, Right right)
		{
			return v_remainder(left, Left(right));
		}

		template<> v_int8x16 v_remainder(const v_int8x16& left, const v_int8x16& right) { return v_int8x16(_mm_rem_epi8(left.data, right.data)); }
		template<> v_int16x8 v_remainder(const v_int16x8& left, const v_int16x8& right) { return v_int16x8(_mm_rem_epi16(left.data, right.data)); }
		template<> v_int32x4 v_remainder(const v_int32x4& left, const v_int32x4& right) { return v_int32x4(_mm_rem_epi32(left.data, right.data)); }
		template<> v_int64x2 v_remainder(const v_int64x2& left, const v_int64x2& right) { return v_int64x2(_mm_rem_epi64(left.data, right.data)); }

		template<> v_uint8x16 v_remainder(const v_uint8x16& left, const v_uint8x16& right) { return v_uint8x16(_mm_rem_epu8(left.data, right.data)); }
		template<> v_uint16x8 v_remainder(const v_uint16x8& left, const v_uint16x8& right) { return v_uint16x8(_mm_rem_epu16(left.data, right.data)); }
		template<> v_uint32x4 v_remainder(const v_uint32x4& left, const v_uint32x4& right) { return v_uint32x4(_mm_rem_epu32(left.data, right.data)); }
		template<> v_uint64x2 v_remainder(const v_uint64x2& left, const v_uint64x2& right) { return v_uint64x2(_mm_rem_epu64(left.data, right.data)); }

		template<> v_int8x32 v_remainder(const v_int8x32& left, const v_int8x32& right) { return v_int8x32(_mm256_rem_epi8(left.data, right.data)); }
		template<> v_int16x16 v_remainder(const v_int16x16& left, const v_int16x16& right) { return v_int16x16(_mm256_rem_epi16(left.data, right.data)); }
		template<> v_int32x8 v_remainder(const v_int32x8& left, const v_int32x8& right) { return v_int32x8(_mm256_rem_epi32(left.data, right.data)); }
		template<> v_int64x4 v_remainder(const v_int64x4& left, const v_int64x4& right) { return v_int64x4(_mm256_rem_epi64(left.data, right.data)); }

		template<> v_uint8x32 v_remainder(const v_uint8x32& left, const v_uint8x32& right) { return v_uint8x32(_mm256_rem_epu8(left.data, right.data)); }
		template<> v_uint16x16 v_remainder(const v_uint16x16& left, const v_uint16x16& right) { return v_uint16x16(_mm256_rem_epu16(left.data, right.data)); }
		template<> v_uint32x8 v_remainder(const v_uint32x8& left, const v_uint32x8& right) { return v_uint32x8(_mm256_rem_epu32(left.data, right.data)); }
		template<> v_uint64x4 v_remainder(const v_uint64x4& left, const v_uint64x4& right) { return v_uint64x4(_mm256_rem_epu64(left.data, right.data)); }
	}
}



export namespace fy
{
	namespace simd
	{
		template<VectorType T> __forceinline T v_le(const T& left, const T& right) { return T(left < right | left == right); }
		template<VectorType T> __forceinline T v_ge(const T& left, const T& right) { return T(left > right | left == right); }

		template<VectorType T> __forceinline T operator <= (const T& left, const T& right) { return v_le(left, right); }
		template<VectorType T> __forceinline T operator >= (const T& left, const T& right) { return v_ge(left, right); }

		template<VectorType Left, BasicArithmetic Right> __forceinline Left operator <= (const Left& left, Right right) { return v_le(left, Left(saturate_cast<typename Left::scalar_t>(right))); }
		template<VectorType Left, BasicArithmetic Right> __forceinline Left operator >= (const Left& left, Right right) { return v_ge(left, Left(saturate_cast<typename Left::scalar_t>(right))); }
	}
}

export namespace fy
{
	namespace simd
	{
		template<typename T>
		T all_onesbits()
		{
			if constexpr (sizeof(T) == 16)
			{
				if constexpr (std::is_same_v<T, __m128i>)
				{
					return _mm_set1_epi32(-1);
				}
				else if constexpr (std::is_same_v<T, __m128h>)
				{
					return _mm_castsi128_ph(_mm_set1_epi32(-1));
				}
				else if constexpr (std::is_same_v<T, __m128>)
				{
					return _mm_castsi128_ps(_mm_set1_epi32(-1));
				}
				else if constexpr (std::is_same_v<T, __m128d>)
				{
					return _mm_castsi128_pd(_mm_set1_epi32(-1));
				}
				else
				{
					std::unreachable();
				}
			}
			else if constexpr (sizeof(T) == 32)
			{
				if constexpr (std::is_same_v<T, __m256i>)
				{
					return _mm256_set1_epi32(-1);
				}
				else if constexpr (std::is_same_v<T, __m256h>)
				{
					return _mm256_castsi256_ph(_mm256_set1_epi32(-1));
				}
				else if constexpr (std::is_same_v<T, __m256>)
				{
					return _mm256_castsi256_ps(_mm256_set1_epi32(-1));
				}
				else if constexpr (std::is_same_v<T, __m256d>)
				{
					return _mm256_castsi256_pd(_mm256_set1_epi32(-1));
				}
				else
				{
					std::unreachable();
				}
			}
			else
			{
				std::unreachable();
			}
		}

		template<VectorType T> __forceinline T v_ne(const T& left, const T& right)
		{
			static const T all_ones = T(all_onesbits<T::vector_t>());
			T eq_result = left == right;
			T ne_result = eq_result ^ all_ones;
			return ne_result;
		}

		template<VectorType T> __forceinline T operator != (const T& left, const T& right) { return v_ne(left, right); }
		template<VectorType Left, BasicArithmetic Right> __forceinline Left operator != (const Left& left, Right right) { return v_ne(left, Left(saturate_cast<typename Left::scalar_t>(right))); }
	}
}



export namespace fy
{
	namespace simd
	{
		template<VectorType T> __forceinline T v_eq(const T& left, const T& right) { std::unreachable(); }
		template<VectorType T> __forceinline T operator == (const T& left, const T& right) { return v_eq(left, right); }
		template<VectorType Left, BasicArithmetic Right> __forceinline Left operator == (const Left& left, Right right) { return v_eq(left, Left(saturate_cast<typename Left::scalar_t>(right))); }

		template<> v_uint8x16 v_eq(const v_uint8x16& left, const v_uint8x16& right) { return v_uint8x16(_mm_cmpeq_epi8(left.data, right.data)); }
		template<> v_uint16x8 v_eq(const v_uint16x8& left, const v_uint16x8& right) { return v_uint16x8(_mm_cmpeq_epi16(left.data, right.data)); }
		template<> v_uint32x4 v_eq(const v_uint32x4& left, const v_uint32x4& right) { return v_uint32x4(_mm_cmpeq_epi32(left.data, right.data)); }
		template<> v_uint64x2 v_eq(const v_uint64x2& left, const v_uint64x2& right) { return v_uint64x2(_mm_cmpeq_epi64(left.data, right.data)); }
		template<> v_int8x16 v_eq(const v_int8x16& left, const v_int8x16& right) { return v_int8x16(_mm_cmpeq_epi8(left.data, right.data)); }
		template<> v_int16x8 v_eq(const v_int16x8& left, const v_int16x8& right) { return v_int16x8(_mm_cmpeq_epi16(left.data, right.data)); }
		template<> v_int32x4 v_eq(const v_int32x4& left, const v_int32x4& right) { return v_int32x4(_mm_cmpeq_epi32(left.data, right.data)); }
		template<> v_int64x2 v_eq(const v_int64x2& left, const v_int64x2& right) { return v_int64x2(_mm_cmpeq_epi64(left.data, right.data)); }

		template<> v_float16x8 v_eq(const v_float16x8& left, const v_float16x8& right)
		{
			__m256 a_f32 = _mm256_cvtph_ps(left.data);
			__m256 b_f32 = _mm256_cvtph_ps(right.data);
			__m256 sum_f32 = _mm256_cmp_ps(a_f32, b_f32, _CMP_EQ_OQ);
			__m128i res = _mm256_cvtps_ph(sum_f32, 0x00);
			return v_float16x8(res);
		}

		template<> v_float32x4 v_eq(const v_float32x4& left, const v_float32x4& right) { return v_float32x4(_mm_cmpeq_ps(left.data, right.data)); }
		template<> v_float64x2 v_eq(const v_float64x2& left, const v_float64x2& right) { return v_float64x2(_mm_cmpeq_pd(left.data, right.data)); }

		template<> v_uint8x32 v_eq(const v_uint8x32& left, const v_uint8x32& right) { return v_uint8x32(_mm256_cmpeq_epi8(left.data, right.data)); }
		template<> v_uint16x16 v_eq(const v_uint16x16& left, const v_uint16x16& right) { return v_uint16x16(_mm256_cmpeq_epi16(left.data, right.data)); }
		template<> v_uint32x8 v_eq(const v_uint32x8& left, const v_uint32x8& right) { return v_uint32x8(_mm256_cmpeq_epi32(left.data, right.data)); }
		template<> v_uint64x4 v_eq(const v_uint64x4& left, const v_uint64x4& right) { return v_uint64x4(_mm256_cmpeq_epi64(left.data, right.data)); }
		template<> v_int8x32 v_eq(const v_int8x32& left, const v_int8x32& right) { return v_int8x32(_mm256_cmpeq_epi8(left.data, right.data)); }
		template<> v_int16x16 v_eq(const v_int16x16& left, const v_int16x16& right) { return v_int16x16(_mm256_cmpeq_epi16(left.data, right.data)); }
		template<> v_int32x8 v_eq(const v_int32x8& left, const v_int32x8& right) { return v_int32x8(_mm256_cmpeq_epi32(left.data, right.data)); }
		template<> v_int64x4 v_eq(const v_int64x4& left, const v_int64x4& right) { return v_int64x4(_mm256_cmpeq_epi64(left.data, right.data)); }

		template<> v_float16x16 v_eq(const v_float16x16& left, const v_float16x16& right)
		{
			__m256 a_f32_low = _mm256_cvtph_ps(_mm256_extracti128_si256(left.data, 0));
			__m256 b_f32_low = _mm256_cvtph_ps(_mm256_extracti128_si256(right.data, 0));
			__m256 sum_f32_low = _mm256_cmp_ps(a_f32_low, b_f32_low, _CMP_EQ_OQ);

			__m256 a_f32_high = _mm256_cvtph_ps(_mm256_extracti128_si256(left.data, 1));
			__m256 b_f32_high = _mm256_cvtph_ps(_mm256_extracti128_si256(right.data, 1));
			__m256 sum_f32_high = _mm256_cmp_ps(a_f32_high, b_f32_high, _CMP_EQ_OQ);

			__m128i res_low = _mm256_cvtps_ph(sum_f32_low, 0);
			__m128i res_high = _mm256_cvtps_ph(sum_f32_high, 0);
			__m256i res = _mm256_inserti128_si256(_mm256_castsi128_si256(res_low), res_high, 1);
			return v_float16x16(res);
		}

		template<> v_float32x8 v_eq(const v_float32x8& left, const v_float32x8& right) { return v_float32x8(_mm256_cmp_ps(left.data, right.data, _CMP_EQ_OQ)); }
		template<> v_float64x4 v_eq(const v_float64x4& left, const v_float64x4& right) { return v_float64x4(_mm256_cmp_pd(left.data, right.data, _CMP_EQ_OQ)); }
	}
}

export namespace fy
{
	namespace simd
	{
		template<VectorType T> __forceinline T v_gt(const T& left, const T& right) { std::unreachable(); }
		template<VectorType T> __forceinline T operator > (const T& left, const T& right) { return v_gt(left, right); }
		template<VectorType Left, BasicArithmetic Right> __forceinline Left operator > (const Left& left, Right right) { return v_gt(left, Left(saturate_cast<typename Left::scalar_t>(right))); }

		template<> v_uint8x16 v_gt(const v_uint8x16& left, const v_uint8x16& right)
		{
			static const __m128i offset = _mm_set1_epi8(char(0x80));
			__m128i a_signed = _mm_add_epi8(left.data, offset);
			__m128i b_signed = _mm_add_epi8(right.data, offset);
			return v_uint8x16(_mm_cmpgt_epi8(a_signed, b_signed));
		}

		template<> v_uint16x8 v_gt(const v_uint16x8& left, const v_uint16x8& right)
		{
			static const __m128i offset = _mm_set1_epi16(short(0x8000));
			__m128i a_signed = _mm_add_epi16(left.data, offset);
			__m128i b_signed = _mm_add_epi16(right.data, offset);
			return v_uint16x8(_mm_cmpgt_epi16(a_signed, b_signed));
		}

		template<> v_uint32x4 v_gt(const v_uint32x4& left, const v_uint32x4& right)
		{
			static const __m128i sign_bit = _mm_set1_epi32(int(0x80000000));
			__m128i a_signed = _mm_xor_si128(left.data, sign_bit);
			__m128i b_signed = _mm_xor_si128(right.data, sign_bit);
			return v_uint32x4(_mm_cmpgt_epi32(a_signed, b_signed));
		}

		template<> v_uint64x2 v_gt(const v_uint64x2& left, const v_uint64x2& right)
		{
			v_uint64x2 res;
			res.data.m128i_u64[0] = left.data.m128i_u64[0] > right.data.m128i_u64[0] ? std::numeric_limits<u64>::max() : 0;
			res.data.m128i_u64[1] = left.data.m128i_u64[1] > right.data.m128i_u64[1] ? std::numeric_limits<u64>::max() : 0;
			return res;
		}

		template<> v_int8x16 v_gt(const v_int8x16& left, const v_int8x16& right) { return v_int8x16(_mm_cmpgt_epi8(left.data, right.data)); }
		template<> v_int16x8 v_gt(const v_int16x8& left, const v_int16x8& right) { return v_int16x8(_mm_cmpgt_epi16(left.data, right.data)); }
		template<> v_int32x4 v_gt(const v_int32x4& left, const v_int32x4& right) { return v_int32x4(_mm_cmpgt_epi32(left.data, right.data)); }

		template<> v_int64x2 v_gt(const v_int64x2& left, const v_int64x2& right)
		{
			v_int64x2 res;
			res.data.m128i_i64[0] = left.data.m128i_i64[0] > right.data.m128i_i64[0] ? -1 : 0;
			res.data.m128i_i64[1] = left.data.m128i_i64[1] > right.data.m128i_i64[1] ? -1 : 0;
			return res;
		}

		template<> v_float16x8 v_gt(const v_float16x8& left, const v_float16x8& right)
		{
			__m256 a_f32 = _mm256_cvtph_ps(left.data);
			__m256 b_f32 = _mm256_cvtph_ps(right.data);
			__m256 sum_f32 = _mm256_cmp_ps(a_f32, b_f32, _CMP_GT_OQ);
			__m128i res = _mm256_cvtps_ph(sum_f32, 0x00);
			return v_float16x8(res);
		}

		template<> v_float32x4 v_gt(const v_float32x4& left, const v_float32x4& right) { return v_float32x4(_mm_cmpgt_ps(left.data, right.data)); }
		template<> v_float64x2 v_gt(const v_float64x2& left, const v_float64x2& right) { return v_float64x2(_mm_cmpgt_pd(left.data, right.data)); }

		template<> v_uint8x32 v_gt(const v_uint8x32& left, const v_uint8x32& right)
		{
			static const __m256i offset = _mm256_set1_epi8(char(0x80));
			__m256i a_signed = _mm256_add_epi8(left.data, offset);
			__m256i b_signed = _mm256_add_epi8(right.data, offset);
			return v_uint8x32(_mm256_cmpgt_epi8(a_signed, b_signed));
		}

		template<> v_uint16x16 v_gt(const v_uint16x16& left, const v_uint16x16& right)
		{
			static const __m256i offset = _mm256_set1_epi16(short(0x8000));
			__m256i a_signed = _mm256_add_epi16(left.data, offset);
			__m256i b_signed = _mm256_add_epi16(right.data, offset);
			return v_uint16x16(_mm256_cmpgt_epi16(a_signed, b_signed));
		}

		template<> v_uint32x8 v_gt(const v_uint32x8& left, const v_uint32x8& right)
		{
			static const __m256i sign_bit = _mm256_set1_epi32(int(0x80000000));
			__m256i a_signed = _mm256_xor_si256(left.data, sign_bit);
			__m256i b_signed = _mm256_xor_si256(right.data, sign_bit);
			return v_uint32x8(_mm256_cmpgt_epi32(a_signed, b_signed));
		}

		template<> v_uint64x4 v_gt(const v_uint64x4& left, const v_uint64x4& right)
		{
			static const __m256i sign_bit = _mm256_set1_epi64x(0x8000000000000000);
			__m256i a_signed = _mm256_xor_si256(left.data, sign_bit);
			__m256i b_signed = _mm256_xor_si256(right.data, sign_bit);
			__m256i gt = _mm256_cmpgt_epi64(a_signed, b_signed);
			return v_uint64x4(gt);
		}

		template<> v_int8x32 v_gt(const v_int8x32& left, const v_int8x32& right) { return v_int8x32(_mm256_cmpgt_epi8(left.data, right.data)); }
		template<> v_int16x16 v_gt(const v_int16x16& left, const v_int16x16& right) { return v_int16x16(_mm256_cmpgt_epi16(left.data, right.data)); }
		template<> v_int32x8 v_gt(const v_int32x8& left, const v_int32x8& right) { return v_int32x8(_mm256_cmpgt_epi32(left.data, right.data)); }
		template<> v_int64x4 v_gt(const v_int64x4& left, const v_int64x4& right) { return v_int64x4(_mm256_cmpgt_epi64(left.data, right.data)); }

		template<> v_float16x16 v_gt(const v_float16x16& left, const v_float16x16& right)
		{
			static const __m256 inverMask = _mm256_set1_ps(1.0f);

			__m256 a_f32_low = _mm256_cvtph_ps(_mm256_extracti128_si256(left.data, 0));
			__m256 b_f32_low = _mm256_cvtph_ps(_mm256_extracti128_si256(right.data, 0));
			__m256 cmp_f32_low = _mm256_cmp_ps(a_f32_low, b_f32_low, _CMP_GT_OQ);
			__m256 sum_f32_low = _mm256_and_ps(cmp_f32_low, inverMask);

			__m256 a_f32_high = _mm256_cvtph_ps(_mm256_extracti128_si256(left.data, 1));
			__m256 b_f32_high = _mm256_cvtph_ps(_mm256_extracti128_si256(right.data, 1));
			__m256 cmp_f32_high = _mm256_cmp_ps(a_f32_high, b_f32_high, _CMP_GT_OQ);
			__m256 sum_f32_high = _mm256_and_ps(cmp_f32_high, inverMask);

			__m128i res_low = _mm256_cvtps_ph(sum_f32_low, 0);
			__m128i res_high = _mm256_cvtps_ph(sum_f32_high, 0);
			__m256i res = _mm256_inserti128_si256(_mm256_castsi128_si256(res_low), res_high, 1);
			return v_float16x16(res);
		}

		template<> v_float32x8 v_gt(const v_float32x8& left, const v_float32x8& right) { return v_float32x8(_mm256_cmp_ps(left.data, right.data, _CMP_GT_OQ)); }
		template<> v_float64x4 v_gt(const v_float64x4& left, const v_float64x4& right) { return v_float64x4(_mm256_cmp_pd(left.data, right.data, _CMP_GT_OQ)); }
	}
}

export namespace fy
{
	namespace simd
	{
		template<VectorType T> __forceinline T operator & (const T& left, const T& right) { return v_bitwise_AND(left, right); }
		template<VectorType T> __forceinline T operator | (const T& left, const T& right) { return v_bitwise_OR(left, right); }
		template<VectorType T> __forceinline T operator ^ (const T& left, const T& right) { return v_bitwise_XOR(left, right); }

		template<VectorType T>
		T v_bitwise_AND(const T& left, const T& right)
		{
			if constexpr (sizeof(T) == 32)
			{
				if constexpr (std::is_integral_v<typename T::scalar_t>)
				{
					return T(_mm256_and_si256(left.data, right.data));
				}
				else if constexpr (std::is_same_v<f16, typename T::scalar_t>)
				{
					return T(_mm256_and_si256(left.data, right.data));
				}
				else if constexpr (std::is_same_v<f32, typename T::scalar_t>)
				{
					return T(_mm256_and_ps(left.data, right.data));
				}
				else if constexpr (std::is_same_v<f64, typename T::scalar_t>)
				{
					return T(_mm256_and_pd(left.data, right.data));
				}
				else
				{
					std::unreachable();
				}
			}
			else if constexpr (sizeof(T) == 16)
			{
				if constexpr (std::is_integral_v<typename T::scalar_t>)
				{
					return T(_mm_and_si128(left.data, right.data));
				}
				else if constexpr (std::is_same_v<f16, typename T::scalar_t>)
				{
					return T(_mm_and_si128(left.data, right.data));
				}
				else if constexpr (std::is_same_v<f32, typename T::scalar_t>)
				{
					return T(_mm_and_ps(left.data, right.data));
				}
				else if constexpr (std::is_same_v<f64, typename T::scalar_t>)
				{
					return T(_mm_and_pd(left.data, right.data));
				}
				else
				{
					std::unreachable();
				}
			}
			else
			{
				std::unreachable();
			}
		}


		template<VectorType T>
		T v_bitwise_OR(const T& left, const T& right)
		{
			if constexpr (sizeof(T) == 32)
			{
				if constexpr (std::is_integral_v<typename T::scalar_t>)
				{
					return T(_mm256_or_si256(left.data, right.data));
				}
				else if constexpr (std::is_same_v<f16, typename T::scalar_t>)
				{
					return T(_mm256_or_si256(left.data, right.data));
				}
				else if constexpr (std::is_same_v<f32, typename T::scalar_t>)
				{
					return T(_mm256_or_ps(left.data, right.data));
				}
				else if constexpr (std::is_same_v<f64, typename T::scalar_t>)
				{
					return T(_mm256_or_pd(left.data, right.data));
				}
				else
				{
					std::unreachable();
				}
			}
			else if constexpr (sizeof(T) == 16)
			{
				if constexpr (std::is_integral_v<typename T::scalar_t>)
				{
					return T(_mm_or_si128(left.data, right.data));
				}
				else if constexpr (std::is_same_v<f16, typename T::scalar_t>)
				{
					return T(_mm_or_si128(left.data, right.data));
				}
				else if constexpr (std::is_same_v<f32, typename T::scalar_t>)
				{
					return T(_mm_or_ps(left.data, right.data));
				}
				else if constexpr (std::is_same_v<f64, typename T::scalar_t>)
				{
					return T(_mm_or_pd(left.data, right.data));
				}
				else
				{
					std::unreachable();
				}
			}
			else
			{
				std::unreachable();
			}
		}

		template<VectorType T>
		T v_bitwise_ANDNOT(const T& left, const T& right)
		{
			if constexpr (sizeof(T) == 32)
			{
				if constexpr (std::is_integral_v<typename T::scalar_t>)
				{
					return T(_mm256_andnot_si256(left.data, right.data));
				}
				else if constexpr (std::is_same_v<f16, typename T::scalar_t>)
				{
					return T(_mm256_andnot_si256(left.data, right.data));
				}
				else if constexpr (std::is_same_v<f32, typename T::scalar_t>)
				{
					return T(_mm256_andnot_ps(left.data, right.data));
				}
				else if constexpr (std::is_same_v<f64, typename T::scalar_t>)
				{
					return T(_mm256_andnot_pd(left.data, right.data));
				}
				else
				{
					std::unreachable();
				}
			}
			else if constexpr (sizeof(T) == 16)
			{
				if constexpr (std::is_integral_v<typename T::scalar_t>)
				{
					return T(_mm_andnot_si128(left.data, right.data));
				}
				else if constexpr (std::is_same_v<f32, typename T::scalar_t>)
				{
					return T(_mm_andnot_si128(left.data, right.data));
				}
				else if constexpr (std::is_same_v<f32, typename T::scalar_t>)
				{
					return T(_mm_andnot_ps(left.data, right.data));
				}
				else if constexpr (std::is_same_v<f64, typename T::scalar_t>)
				{
					return T(_mm_andnot_pd(left.data, right.data));
				}
				else
				{
					std::unreachable();
				}
			}
			else
			{
				std::unreachable();
			}
		}

		template<VectorType T>
		T v_bitwise_XOR(const T& left, const T& right)
		{
			if constexpr (sizeof(T) == 32)
			{
				if constexpr (std::is_integral_v<typename T::scalar_t>)
				{
					return T(_mm256_xor_si256(left.data, right.data));
				}
				else if constexpr (std::is_same_v<f16, typename T::scalar_t>)
				{
					return T(_mm256_xor_si256(left.data, right.data));
				}
				else if constexpr (std::is_same_v<f32, typename T::scalar_t>)
				{
					return T(_mm256_xor_ps(left.data, right.data));
				}
				else if constexpr (std::is_same_v<f64, typename T::scalar_t>)
				{
					return T(_mm256_xor_pd(left.data, right.data));
				}
				else
				{
					std::unreachable();
				}
			}
			else if constexpr (sizeof(T) == 16)
			{
				if constexpr (std::is_integral_v<typename T::scalar_t>)
				{
					return T(_mm_xor_si128(left.data, right.data));
				}
				else if constexpr (std::is_same_v<f16, typename T::scalar_t>)
				{
					return T(_mm_xor_si128(left.data, right.data));
				}
				else if constexpr (std::is_same_v<f32, typename T::scalar_t>)
				{
					return T(_mm_xor_ps(left.data, right.data));
				}
				else if constexpr (std::is_same_v<f64, typename T::scalar_t>)
				{
					return T(_mm_xor_pd(left.data, right.data));
				}
				else
				{
					std::unreachable();
				}
			}
			else
			{
				std::unreachable();
			}
		}
	}
}

export namespace fy
{
	namespace simd
	{
		template<VectorType Dst_t, VectorType Src_t> Dst_t v_convert(const Src_t& src)
		{
			static_assert(Src_t::batch_size == Dst_t::batch_size, "Vector type conversion requires the same batch_size");
			return Dst_t(src.data);
		}

		template<> v_int16x16 v_convert<v_int16x16, v_uint8x16>(const v_uint8x16& input)
		{
			__m256i result = _mm256_cvtepu8_epi16(input.data);
			return v_int16x16(result);
		}

		template<> v_uint8x16 v_convert<v_uint8x16, v_int16x16>(const v_int16x16& input)
		{
			__m256i saturated = _mm256_max_epi16(input.data, _mm256_setzero_si256());
			__m256i packed = _mm256_packus_epi16(saturated, saturated);
			__m128i result = _mm256_castsi256_si128(packed);
			return v_uint8x16(result);
		}

		template<> v_uint16x16 v_convert<v_uint16x16, v_uint8x16>(const v_uint8x16& input)
		{
			__m256i result = _mm256_cvtepu8_epi16(input.data);
			return v_uint16x16(result);
		}

		template<> v_uint8x16 v_convert<v_uint8x16, v_uint16x16>(const v_uint16x16& input)
		{
			__m256i packed = _mm256_packus_epi16(input.data, input.data);
			__m128i result = _mm256_castsi256_si128(packed);
			return v_uint8x16(result);
		}

		template<> v_float16x16 v_convert<v_float16x16, v_uint8x16>(const v_uint8x16& input)
		{
			static const __m128i zero_mask = _mm_setzero_si128();
			__m128i low_64 = _mm_unpacklo_epi8(input.data, zero_mask);
			__m128i high_64 = _mm_unpackhi_epi8(input.data, zero_mask);

			__m256i int32_low = _mm256_cvtepu16_epi32(low_64);
			__m256i int32_high = _mm256_cvtepu16_epi32(high_64);

			__m256 float32_low = _mm256_cvtepi32_ps(int32_low);
			__m256 float32_high = _mm256_cvtepi32_ps(int32_high);

			__m128i float16_low = _mm256_cvtps_ph(float32_low, _MM_FROUND_TO_NEAREST_INT);
			__m128i float16_high = _mm256_cvtps_ph(float32_high, _MM_FROUND_TO_NEAREST_INT);

			__m256i result = _mm256_insertf128_si256(_mm256_castsi128_si256(float16_low), float16_high, 1);
			return v_float16x16(result);
		}

		template<> v_uint8x16 v_convert<v_uint8x16, v_float16x16>(const v_float16x16& input)
		{
			__m256 float32_low = _mm256_cvtph_ps(_mm256_extractf128_si256(input.data, 0));
			__m256 float32_high = _mm256_cvtph_ps(_mm256_extractf128_si256(input.data, 1));

			__m256 rounded_low = _mm256_round_ps(float32_low, _MM_FROUND_TO_NEAREST_INT);
			__m256 rounded_high = _mm256_round_ps(float32_high, _MM_FROUND_TO_NEAREST_INT);

			__m256i int32_low = _mm256_cvtps_epi32(rounded_low);
			__m256i int32_high = _mm256_cvtps_epi32(rounded_high);

			__m256i saturated_low = _mm256_packus_epi32(int32_low, int32_high);

			__m128i result = _mm_packus_epi16(
				_mm256_castsi256_si128(saturated_low),
				_mm256_extractf128_si256(saturated_low, 1)
			);

			return v_uint8x16(result);
		}

		template<> v_uint16x16 v_convert<v_uint16x16, v_int8x16>(const v_int8x16& input)
		{
			static const __m256i zero_mask = _mm256_setzero_si256();
			__m256i result = _mm256_max_epi16(_mm256_cvtepi8_epi16(input.data), zero_mask);
			return v_uint16x16(result);
		}

		template<> v_int8x16 v_convert<v_int8x16, v_uint16x16>(const v_uint16x16& input)
		{
			__m256i saturated = _mm256_min_epu16(input.data, _mm256_set1_epi16(127));
			__m128i result = _mm_packs_epi16(_mm256_castsi256_si128(saturated),
				_mm256_extractf128_si256(saturated, 1));
			return v_int8x16(result);
		}

		template<> v_int16x16 v_convert<v_int16x16, v_int8x16>(const v_int8x16& input)
		{
			return v_int16x16(_mm256_cvtepi8_epi16(input.data));
		}

		template<> v_int8x16 v_convert<v_int8x16, v_int16x16>(const v_int16x16& input)
		{
			__m128i result = _mm_packs_epi16(_mm256_castsi256_si128(input.data),
				_mm256_extractf128_si256(input.data, 1));
			return v_int8x16(result);
		}

		template<> v_float16x16 v_convert<v_float16x16, v_int8x16>(const v_int8x16& input)
		{
			static const __m128i zero_mask = _mm_setzero_si128();
			__m128i low_64 = _mm_unpacklo_epi8(input.data, _mm_cmpgt_epi8(zero_mask, input.data));
			__m128i high_64 = _mm_unpackhi_epi8(input.data, _mm_cmpgt_epi8(zero_mask, input.data));

			__m256i int32_low = _mm256_cvtepi16_epi32(low_64);
			__m256i int32_high = _mm256_cvtepi16_epi32(high_64);

			__m256 float32_low = _mm256_cvtepi32_ps(int32_low);
			__m256 float32_high = _mm256_cvtepi32_ps(int32_high);

			__m128i float16_low = _mm256_cvtps_ph(float32_low, _MM_FROUND_TO_NEAREST_INT);
			__m128i float16_high = _mm256_cvtps_ph(float32_high, _MM_FROUND_TO_NEAREST_INT);

			return v_float16x16(_mm256_insertf128_si256(_mm256_castsi128_si256(float16_low), float16_high, 1));
		}

		template<> v_int8x16 v_convert<v_int8x16, v_float16x16>(const v_float16x16& input)
		{
			__m256 float32_low = _mm256_cvtph_ps(_mm256_extractf128_si256(input.data, 0));
			__m256 float32_high = _mm256_cvtph_ps(_mm256_extractf128_si256(input.data, 1));

			__m256 rounded_low = _mm256_round_ps(float32_low, _MM_FROUND_TO_NEAREST_INT);
			__m256 rounded_high = _mm256_round_ps(float32_high, _MM_FROUND_TO_NEAREST_INT);

			__m256i int32_low = _mm256_cvtps_epi32(rounded_low);
			__m256i int32_high = _mm256_cvtps_epi32(rounded_high);

			__m256i int16 = _mm256_packs_epi32(int32_low, int32_high);

			__m128i result = _mm_packs_epi16(_mm256_castsi256_si128(int16),
				_mm256_extractf128_si256(int16, 1));

			return v_int8x16(result);
		}

		template<> v_float32x8 v_convert<v_float32x8, v_uint16x8>(const v_uint16x8& input)
		{
			__m256i int32 = _mm256_cvtepu16_epi32(input.data);
			__m256 result = _mm256_cvtepi32_ps(int32);
			return v_float32x8(result);
		}

		template<> v_uint16x8 v_convert<v_uint16x8, v_float32x8>(const v_float32x8& input)
		{
			__m256 rounded = _mm256_round_ps(input.data, _MM_FROUND_TO_NEAREST_INT);
			__m256i int32 = _mm256_cvtps_epi32(rounded);
			__m256i saturated = _mm256_packus_epi32(int32, int32);
			__m128i result = _mm256_castsi256_si128(saturated);
			return v_uint16x8(result);
		}

		template<> v_int32x8 v_convert<v_int32x8, v_uint16x8>(const v_uint16x8& input)
		{
			return v_int32x8(_mm256_cvtepu16_epi32(input.data));
		}

		template<> v_uint16x8 v_convert<v_uint16x8, v_int32x8>(const v_int32x8& input)
		{
			__m256i saturated = _mm256_packus_epi32(input.data, input.data);
			__m128i result = _mm256_castsi256_si128(saturated);
			return v_uint16x8(result);
		}

		template<> v_uint32x8 v_convert<v_uint32x8, v_uint16x8>(const v_uint16x8& input)
		{
			return v_uint32x8(_mm256_cvtepu16_epi32(input.data));
		}

		template<> v_uint16x8 v_convert<v_uint16x8, v_uint32x8>(const v_uint32x8& input)
		{
			__m256i saturated = _mm256_packus_epi32(input.data, input.data);
			__m128i result = _mm256_castsi256_si128(saturated);
			return v_uint16x8(result);
		}

		template<> v_float32x8 v_convert<v_float32x8, v_int16x8>(const v_int16x8& input)
		{
			__m256i int32 = _mm256_cvtepi16_epi32(input.data);
			__m256 result = _mm256_cvtepi32_ps(int32);
			return v_float32x8(result);
		}

		template<> v_int16x8 v_convert<v_int16x8, v_float32x8>(const v_float32x8& input)
		{
			__m256 rounded = _mm256_round_ps(input.data, _MM_FROUND_TO_NEAREST_INT);
			__m256i int32 = _mm256_cvtps_epi32(rounded);
			__m256i saturated = _mm256_packs_epi32(int32, int32);
			__m128i result = _mm256_castsi256_si128(saturated);
			return v_int16x8(result);
		}

		template<> v_int32x8 v_convert<v_int32x8, v_int16x8>(const v_int16x8& input)
		{
			return v_int32x8(_mm256_cvtepi16_epi32(input.data));
		}

		template<> v_int16x8 v_convert<v_int16x8, v_int32x8>(const v_int32x8& input)
		{
			__m256i saturated = _mm256_packs_epi32(input.data, input.data);
			__m128i result = _mm256_castsi256_si128(saturated);
			return v_int16x8(result);
		}

		template<> v_uint32x8 v_convert<v_uint32x8, v_int16x8>(const v_int16x8& input)
		{
			static const __m256i zero_mask = _mm256_setzero_si256();
			__m256i result = _mm256_max_epi16(_mm256_cvtepi8_epi16(input.data), zero_mask);
			return v_uint32x8(result);
		}

		template<> v_int16x8 v_convert<v_int16x8, v_uint32x8>(const v_uint32x8& input)
		{
			static const __m256i max_mask = _mm256_set1_epi32(std::numeric_limits<i16>::max());
			__m256i limited = _mm256_min_epu32(input.data, max_mask);
			__m256i saturated = _mm256_packs_epi32(limited, limited);
			__m128i result = _mm256_castsi256_si128(saturated);
			return v_int16x8(result);
		}

		template<> v_float32x8 v_convert<v_float32x8, v_float16x8>(const v_float16x8& input)
		{
			return v_float32x8(_mm256_cvtph_ps(input.data));
		}

		template<> v_float16x8 v_convert<v_float16x8, v_float32x8>(const v_float32x8& input)
		{
			return v_float16x8(_mm256_cvtps_ph(input.data, _MM_FROUND_TO_NEAREST_INT));
		}

		template<> v_float32x8 v_convert<v_float32x8, v_bfloat16x8>(const v_bfloat16x8& input)
		{
			__m256i extended = _mm256_cvtepu16_epi32(input.data);
			__m256i shifted = _mm256_slli_epi32(extended, 16);
			return v_float32x8(_mm256_castsi256_ps(shifted));
		}

		template<> v_bfloat16x8 v_convert<v_bfloat16x8, v_float32x8>(const v_float32x8& input)
		{
			constexpr u32 LOW_16_BITS_MASK = 0x0000FFFF;
			constexpr u32 QUIET_NAN_BIT = 0x00000001;

			const __m256i v_exp_mask = _mm256_set1_epi32(v_bfloat16x8::scalar_t::exponent_mask);
			const __m256i v_mant_mask = _mm256_set1_epi32(v_bfloat16x8::scalar_t::mantissa_mask);
			const __m256i v_zero = _mm256_setzero_si256();
			const __m256i v_low16_bit_mask = _mm256_set1_epi32(LOW_16_BITS_MASK);
			const __m256i v_quiet_bit = _mm256_set1_epi32(QUIET_NAN_BIT);

			__m256i v = _mm256_castps_si256(input.data);

			__m256i exponents = _mm256_and_si256(v, v_exp_mask);
			__m256i mantissas = _mm256_and_si256(v, v_mant_mask);

			__m256i is_exp_all_ones = _mm256_cmpeq_epi32(exponents, v_exp_mask);
			__m256i is_mant_nonzero = _mm256_cmpgt_epi32(mantissas, v_zero);
			__m256i is_nan = _mm256_and_si256(is_exp_all_ones, is_mant_nonzero);

			__m256i shifted = _mm256_srli_epi32(v, 16);

			__m256i shifted_low = _mm256_and_si256(shifted, v_low16_bit_mask);
			__m256i is_quiet_nan_needed = _mm256_and_si256(is_nan, _mm256_cmpeq_epi32(shifted_low, v_zero));
			__m256i fix_value = _mm256_and_si256(v_quiet_bit, is_quiet_nan_needed);

			shifted = _mm256_or_si256(shifted, fix_value);

			__m128i lo_half = _mm256_extractf128_si256(shifted, 0);
			__m128i hi_half = _mm256_extractf128_si256(shifted, 1);
			__m128i packed = _mm_packus_epi32(lo_half, hi_half);

			return v_bfloat16x8(packed);
		}

		template<> v_int32x8 v_convert<v_int32x8, v_float16x8>(const v_float16x8& input)
		{
			__m256 float32 = _mm256_cvtph_ps(input.data);
			__m256i result = _mm256_cvtps_epi32(float32);
			return v_int32x8(result);
		}

		template<> v_float16x8 v_convert<v_float16x8, v_int32x8>(const v_int32x8& input)
		{
			__m256 float32 = _mm256_cvtepi32_ps(input.data);
			__m128i result = _mm256_cvtps_ph(float32, _MM_FROUND_TO_NEAREST_INT);
			return v_float16x8(result);
		}

		template<> v_uint32x8 v_convert<v_uint32x8, v_float16x8>(const v_float16x8& input)
		{
			static const __m256i zero_mask = _mm256_setzero_si256();
			__m256 float32 = _mm256_cvtph_ps(input.data);
			__m256 rounded = _mm256_round_ps(float32, _MM_FROUND_TO_NEAREST_INT);
			__m256i int32 = _mm256_cvtps_epi32(rounded);
			__m256i result = _mm256_max_epi32(int32, zero_mask);
			return v_uint32x8(result);
		}

		template<> v_float16x8 v_convert<v_float16x8, v_uint32x8>(const v_uint32x8& input)
		{
			__m256i low_bits = _mm256_and_si256(input.data, _mm256_set1_epi32(0x7FFFFFFF));
			__m256i high_bit = _mm256_srli_epi32(input.data, 31);

			__m256 float_low = _mm256_cvtepi32_ps(low_bits);
			__m256 float_high = _mm256_mul_ps(
				_mm256_cvtepi32_ps(high_bit),
				_mm256_set1_ps(2147483648.0f)
			);

			__m256 float32 = _mm256_add_ps(float_low, float_high);
			__m128i result = _mm256_cvtps_ph(float32, _MM_FROUND_TO_NEAREST_INT);
			return v_float16x8(result);
		}

		template<> v_int64x4 v_convert<v_int64x4, v_int32x4>(const v_int32x4& input)
		{
			return v_int64x4(_mm256_cvtepi32_epi64(input.data));
		}

		template<> v_int32x4 v_convert<v_int32x4, v_int64x4>(const v_int64x4& input)
		{
			return v_int32x4(
				std::clamp(input.data.m256i_i64[0], static_cast<i64>(std::numeric_limits<i32>::lowest()), static_cast<i64>(std::numeric_limits<i32>::max())),
				std::clamp(input.data.m256i_i64[1], static_cast<i64>(std::numeric_limits<i32>::lowest()), static_cast<i64>(std::numeric_limits<i32>::max())),
				std::clamp(input.data.m256i_i64[2], static_cast<i64>(std::numeric_limits<i32>::lowest()), static_cast<i64>(std::numeric_limits<i32>::max())),
				std::clamp(input.data.m256i_i64[3], static_cast<i64>(std::numeric_limits<i32>::lowest()), static_cast<i64>(std::numeric_limits<i32>::max()))
			);
		}

		template<> v_uint64x4 v_convert<v_uint64x4, v_int32x4>(const v_int32x4& input)
		{
			static const __m128i zero_mask = _mm_setzero_si128();
			__m128i positive = _mm_max_epi32(input.data, zero_mask);
			return v_uint64x4(_mm256_cvtepu32_epi64(positive));
		}

		template<> v_int32x4 v_convert<v_int32x4, v_uint64x4>(const v_uint64x4& input)
		{
			return v_int32x4(
				std::clamp(input.data.m256i_u64[0], static_cast<u64>(std::numeric_limits<i32>::lowest()), static_cast<u64>(std::numeric_limits<i32>::max())),
				std::clamp(input.data.m256i_u64[1], static_cast<u64>(std::numeric_limits<i32>::lowest()), static_cast<u64>(std::numeric_limits<i32>::max())),
				std::clamp(input.data.m256i_u64[2], static_cast<u64>(std::numeric_limits<i32>::lowest()), static_cast<u64>(std::numeric_limits<i32>::max())),
				std::clamp(input.data.m256i_u64[3], static_cast<u64>(std::numeric_limits<i32>::lowest()), static_cast<u64>(std::numeric_limits<i32>::max()))
			);
		}

		template<> v_float64x4 v_convert<v_float64x4, v_int32x4>(const v_int32x4& input)
		{
			return v_float64x4(_mm256_cvtepi32_pd(input.data));
		}

		template<> v_int32x4 v_convert<v_int32x4, v_float64x4>(const v_float64x4& input)
		{
			return v_int32x4(_mm256_cvtpd_epi32(input.data));
		}

		template<> v_int64x4 v_convert<v_int64x4, v_uint32x4>(const v_uint32x4& input)
		{
			return v_int64x4(_mm256_cvtepu32_epi64(input.data));
		}

		template<> v_uint32x4 v_convert<v_uint32x4, v_int64x4>(const v_int64x4& input)
		{
			return v_uint32x4(
				static_cast<u32>(std::clamp(input.data.m256i_i64[0], static_cast<i64>(std::numeric_limits<u32>::lowest()), static_cast<i64>(std::numeric_limits<u32>::max()))),
				static_cast<u32>(std::clamp(input.data.m256i_i64[1], static_cast<i64>(std::numeric_limits<u32>::lowest()), static_cast<i64>(std::numeric_limits<u32>::max()))),
				static_cast<u32>(std::clamp(input.data.m256i_i64[2], static_cast<i64>(std::numeric_limits<u32>::lowest()), static_cast<i64>(std::numeric_limits<u32>::max()))),
				static_cast<u32>(std::clamp(input.data.m256i_i64[3], static_cast<i64>(std::numeric_limits<u32>::lowest()), static_cast<i64>(std::numeric_limits<u32>::max())))
			);
		}

		template<> v_uint64x4 v_convert<v_uint64x4, v_uint32x4>(const v_uint32x4& input)
		{
			return v_uint64x4(_mm256_cvtepu32_epi64(input.data));
		}

		template<> v_uint32x4 v_convert<v_uint32x4, v_uint64x4>(const v_uint64x4& input)
		{
			return v_uint32x4(
				static_cast<u32>(std::clamp(input.data.m256i_u64[0], static_cast<u64>(std::numeric_limits<u32>::lowest()), static_cast<u64>(std::numeric_limits<u32>::max()))),
				static_cast<u32>(std::clamp(input.data.m256i_u64[1], static_cast<u64>(std::numeric_limits<u32>::lowest()), static_cast<u64>(std::numeric_limits<u32>::max()))),
				static_cast<u32>(std::clamp(input.data.m256i_u64[2], static_cast<u64>(std::numeric_limits<u32>::lowest()), static_cast<u64>(std::numeric_limits<u32>::max()))),
				static_cast<u32>(std::clamp(input.data.m256i_u64[3], static_cast<u64>(std::numeric_limits<u32>::lowest()), static_cast<u64>(std::numeric_limits<u32>::max())))
			);
		}

		template<> v_float64x4 v_convert<v_float64x4, v_uint32x4>(const v_uint32x4& input)
		{
			__m256i int64 = _mm256_cvtepu32_epi64(input.data);
			return v_float64x4(_mm256_cvtepi64_pd(int64));
		}

		template<> v_uint32x4 v_convert<v_uint32x4, v_float64x4>(const v_float64x4& input)
		{
			static const __m128i zero_mask = _mm_setzero_si128();
			__m128 float32 = _mm256_cvtpd_ps(input.data);
			__m128 rounded = _mm_round_ps(float32, _MM_FROUND_TO_NEAREST_INT);
			__m128i int32 = _mm_cvtps_epi32(rounded);
			__m128i result = _mm_max_epu32(int32, zero_mask);
			return v_uint32x4(result);
		}

		template<> v_int64x4 v_convert<v_int64x4, v_float32x4>(const v_float32x4& input)
		{
			return v_int64x4(
				static_cast<i64>(std::roundf(input.data.m128_f32[0])),
				static_cast<i64>(std::roundf(input.data.m128_f32[1])),
				static_cast<i64>(std::roundf(input.data.m128_f32[2])),
				static_cast<i64>(std::roundf(input.data.m128_f32[3]))
			);
		}

		template<> v_float32x4 v_convert<v_float32x4, v_int64x4>(const v_int64x4& input)
		{
			return v_float32x4(
				static_cast<f32>(input.data.m256i_i64[0]),
				static_cast<f32>(input.data.m256i_i64[1]),
				static_cast<f32>(input.data.m256i_i64[2]),
				static_cast<f32>(input.data.m256i_i64[3])
			);
		}

		template<> v_uint64x4 v_convert<v_uint64x4, v_float32x4>(const v_float32x4& input)
		{
			return v_uint64x4(
				static_cast<u64>(std::max(0.0f, std::roundf(input.data.m128_f32[0]))),
				static_cast<u64>(std::max(0.0f, std::roundf(input.data.m128_f32[1]))),
				static_cast<u64>(std::max(0.0f, std::roundf(input.data.m128_f32[2]))),
				static_cast<u64>(std::max(0.0f, std::roundf(input.data.m128_f32[3])))
			);
		}

		template<> v_float32x4 v_convert<v_float32x4, v_uint64x4>(const v_uint64x4& input)
		{
			return v_float32x4(
				static_cast<float>(input.data.m256i_u64[0]),
				static_cast<float>(input.data.m256i_u64[1]),
				static_cast<float>(input.data.m256i_u64[2]),
				static_cast<float>(input.data.m256i_u64[3])
			);
		}

		template<> v_float64x4 v_convert<v_float64x4, v_float32x4>(const v_float32x4& input)
		{
			__m128d low = _mm_cvtps_pd(input.data);
			__m128d high = _mm_cvtps_pd(_mm_movehl_ps(input.data, input.data));
			return v_float64x4(_mm256_set_m128d(high, low));
		}

		template<> v_float32x4 v_convert<v_float32x4, v_float64x4>(const v_float64x4& input)
		{
			return v_float32x4(_mm256_cvtpd_ps(input.data));
		}

		template<> v_int8x16 v_convert<v_int8x16, v_uint8x16>(const v_uint8x16& input)
		{
			static const __m128i max_i8 = _mm_set1_epi8(std::numeric_limits<v_int8x16::scalar_t>::max());
			__m128i saturated = _mm_min_epu8(input.data, max_i8);
			return v_int8x16(saturated);
		}

		template<> v_uint8x16 v_convert<v_uint8x16, v_int8x16>(const v_int8x16& input)
		{
			static const __m128i zero = _mm_setzero_si128();
			__m128i saturated = _mm_max_epi8(input.data, zero);
			return v_uint8x16(saturated);
		}

		template<> v_int16x8 v_convert<v_int16x8, v_uint16x8>(const v_uint16x8& input)
		{
			static const __m128i max_i8 = _mm_set1_epi16(std::numeric_limits<v_int16x8::scalar_t>::max());
			__m128i saturated = _mm_min_epu16(input.data, max_i8);
			return v_int16x8(saturated);
		}

		template<> v_float16x8 v_convert<v_float16x8, v_uint16x8>(const v_uint16x8& input)
		{
			__m256i uint32 = _mm256_cvtepu16_epi32(input.data);
			__m256 float32 = _mm256_cvtepi32_ps(uint32);
			__m128h result = _mm256_cvtps_ph(float32, _MM_FROUND_TO_NEAREST_INT);
			return v_float16x8(result);
		}

		template<> v_uint16x8 v_convert<v_uint16x8, v_int16x8>(const v_int16x8& input)
		{
			static const __m128i zero = _mm_setzero_si128();
			__m128i saturated = _mm_max_epi16(input.data, zero);
			return v_uint16x8(saturated);
		}

		template<> v_float16x8 v_convert<v_float16x8, v_int16x8>(const v_int16x8& input)
		{
			__m256i int32 = _mm256_cvtepi16_epi32(input.data);
			__m256 float32 = _mm256_cvtepi32_ps(int32);
			__m128h result = _mm256_cvtps_ph(float32, _MM_FROUND_TO_NEAREST_INT);
			return v_float16x8(result);
		}

		template<> v_int32x4 v_convert<v_int32x4, v_uint32x4>(const v_uint32x4& input)
		{
			static const __m128i max_i8 = _mm_set1_epi32(std::numeric_limits<v_int32x4::scalar_t>::max());
			__m128i saturated = _mm_min_epu32(input.data, max_i8);
			return v_int32x4(saturated);
		}

		template<> v_float32x4 v_convert<v_float32x4, v_uint32x4>(const v_uint32x4& input)
		{
			static const __m128i sign_mask = _mm_set1_epi32(0x80000000);
			static const __m128i abs_mask = _mm_set1_epi32(0x7FFFFFFF);
			static const __m128 large_num = _mm_set1_ps(2147483648.0f);

			__m128i sign = _mm_and_si128(input.data, sign_mask);
			__m128i abs_val = _mm_and_si128(input.data, abs_mask);

			__m128 float_abs = _mm_cvtepi32_ps(abs_val);
			__m128 sign_float = _mm_castsi128_ps(sign);
			__m128 result = _mm_or_ps(float_abs, sign_float);

			__m128 mask = _mm_cmpge_ps(result, large_num);
			result = _mm_add_ps(result, _mm_and_ps(mask, large_num));

			return v_float32x4(result);
		}

		template<> v_uint32x4 v_convert<v_uint32x4, v_int32x4>(const v_int32x4& input)
		{
			static const __m128i zero = _mm_setzero_si128();
			__m128i saturated = _mm_max_epi32(input.data, zero);
			return v_uint32x4(saturated);
		}

		template<> v_float32x4 v_convert<v_float32x4, v_int32x4>(const v_int32x4& input)
		{
			return v_float32x4(_mm_cvtepi32_ps(input.data));
		}

		template<> v_uint64x2 v_convert<v_uint64x2, v_int64x2>(const v_int64x2& input)
		{
			return v_uint64x2(
				static_cast<u64>(std::max(input.data.m128i_i64[0], static_cast<i64>(0))),
				static_cast<u64>(std::max(input.data.m128i_i64[1], static_cast<i64>(0)))
			);
		}

		template<> v_float64x2 v_convert<v_float64x2, v_int64x2>(const v_int64x2& input)
		{
			return v_float64x2(
				static_cast<double>(input.data.m128i_i64[0]),
				static_cast<double>(input.data.m128i_i64[1])
			);
		}

		template<> v_int64x2 v_convert<v_int64x2, v_uint64x2>(const v_uint64x2& input)
		{
			return v_int64x2(
				static_cast<i64>(std::min(input.data.m128i_u64[0], static_cast<u64>(std::numeric_limits<i64>::max()))),
				static_cast<i64>(std::min(input.data.m128i_u64[1], static_cast<u64>(std::numeric_limits<i64>::max())))
			);
		}

		template<> v_float64x2 v_convert<v_float64x2, v_uint64x2>(const v_uint64x2& input)
		{
			v_float64x2 result;
			for (int i = 0; i < 2; ++i)
			{
				u64 value = input.data.m128i_u64[i];
				if (value <= static_cast<u64>(std::numeric_limits<int64_t>::max()))
				{
					result.data.m128d_f64[i] = static_cast<f64>(value);
				}
				else
				{
					static constexpr const f64 two_pow_63 = 9223372036854775808.0;
					result.data.m128d_f64[i] = static_cast<f64>(value - (1ULL << 63)) + two_pow_63;
				}
			}
			return result;
		}

		template<> v_uint16x8 v_convert<v_uint16x8, v_float16x8>(const v_float16x8& input)
		{
			static const __m128i zero_mask = _mm_setzero_si128();
			static const __m128i full_mask = _mm_set1_epi32(0x80000000);

			__m128 fp32 = _mm_cvtph_ps(input.data);
			__m128i int32 = _mm_cvtps_epi32(_mm_round_ps(fp32, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));

			__m128i sign_mask = _mm_cmpgt_epi32(zero_mask, int32);
			int32 = _mm_add_epi32(int32, _mm_and_si128(sign_mask, full_mask));

			__m128i result = _mm_packus_epi32(int32, zero_mask);
			return v_uint16x8(result);
		}

		template<> v_int16x8 v_convert<v_int16x8, v_float16x8>(const v_float16x8& input)
		{
			static const __m128i zero_mask = _mm_setzero_si128();

			__m128 fp32 = _mm_cvtph_ps(input.data);
			__m128 rounded = _mm_round_ps(fp32, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
			__m128i int32 = _mm_cvtps_epi32(rounded);
			__m128i result = _mm_packs_epi32(int32, zero_mask);
			return v_int16x8(result);
		}

		template<> v_uint32x4 v_convert<v_uint32x4, v_float32x4>(const v_float32x4& input)
		{
			static const __m128 zero = _mm_setzero_ps();
			static const __m128 offset = _mm_set1_ps(4294967296.0f);

			__m128i positive = _mm_cvttps_epi32(input.data);

			__m128 added = _mm_add_ps(input.data, offset);
			__m128i negative = _mm_cvttps_epi32(added);

			__m128 cmp = _mm_cmplt_ps(input.data, zero);
			__m128i mask = _mm_castps_si128(cmp);

			__m128i result = _mm_or_si128(_mm_andnot_si128(mask, positive), _mm_and_si128(mask, negative));
			return v_uint32x4(result);
		}

		template<> v_int32x4 v_convert<v_int32x4, v_float32x4>(const v_float32x4& input)
		{
			return v_int32x4(_mm_cvtps_epi32(_mm_round_ps(input.data, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC)));
		}

		template<> v_int64x2 v_convert<v_int64x2, v_float64x2>(const v_float64x2& input)
		{
			return v_int64x2(
				static_cast<i64>(std::round(input.data.m128d_f64[0])),
				static_cast<i64>(std::round(input.data.m128d_f64[1]))
			);
		}

		template<> v_uint64x2 v_convert<v_uint64x2, v_float64x2>(const v_float64x2& input)
		{
			return v_uint64x2(
				static_cast<u64>(std::round(input.data.m128d_f64[0])),
				static_cast<u64>(std::round(input.data.m128d_f64[1]))
			);
		}

		template<> v_int8x32 v_convert<v_int8x32, v_uint8x32>(const v_uint8x32& input)
		{
			__m256i max_i8 = _mm256_set1_epi8(std::numeric_limits<v_int8x32::scalar_t>::max());
			__m256i saturated = _mm256_min_epu8(input.data, max_i8);
			return v_int8x32(saturated);
		}

		template<> v_uint8x32 v_convert<v_uint8x32, v_int8x32>(const v_int8x32& input)
		{
			static const __m256i zero = _mm256_setzero_si256();
			__m256i saturated = _mm256_max_epi8(input.data, zero);
			return v_uint8x32(saturated);
		}

		template<> v_int16x16 v_convert<v_int16x16, v_uint16x16>(const v_uint16x16& input)
		{
			static const __m256i max_i16 = _mm256_set1_epi16(std::numeric_limits<i16>::max());
			__m256i saturated = _mm256_min_epu16(input.data, max_i16);
			return v_int16x16(saturated);
		}

		template<> v_float16x16 v_convert<v_float16x16, v_uint16x16>(const v_uint16x16& input)
		{
			__m256i uint32_low = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(input.data));
			__m256i uint32_high = _mm256_cvtepu16_epi32(_mm256_extracti128_si256(input.data, 1));

			__m256 float32_low = _mm256_cvtepi32_ps(uint32_low);
			__m256 float32_high = _mm256_cvtepi32_ps(uint32_high);

			__m128h float16_low = _mm256_cvtps_ph(float32_low, _MM_FROUND_TO_NEAREST_INT);
			__m128h float16_high = _mm256_cvtps_ph(float32_high, _MM_FROUND_TO_NEAREST_INT);

			__m256h result = _mm256_castsi256_ph(_mm256_set_m128i(_mm_castph_si128(float16_high), _mm_castph_si128(float16_low)));
			return v_float16x16(result);
		}

		template<> v_uint16x16 v_convert<v_uint16x16, v_int16x16>(const v_int16x16& input)
		{
			static const __m256i zero = _mm256_setzero_si256();
			__m256i result = _mm256_max_epi16(input.data, zero);
			return v_uint16x16(result);
		}

		template<> v_float16x16 v_convert<v_float16x16, v_int16x16>(const v_int16x16& input)
		{
			__m256 float32_low = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(input.data)));
			__m256 float32_high = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(input.data, 1)));

			__m128h float16_low = _mm256_cvtps_ph(float32_low, _MM_FROUND_TO_NEAREST_INT);
			__m128h float16_high = _mm256_cvtps_ph(float32_high, _MM_FROUND_TO_NEAREST_INT);

			__m256h result = _mm256_castsi256_ph(_mm256_set_m128i(_mm_castph_si128(float16_high), _mm_castph_si128(float16_low)));
			return v_float16x16(result);
		}

		template<> v_int32x8 v_convert<v_int32x8, v_uint32x8>(const v_uint32x8& input)
		{
			static const __m256i max_int32 = _mm256_set1_epi32(std::numeric_limits<i32>::max());
			__m256i saturated = _mm256_min_epu32(input.data, max_int32);
			return v_int32x8(saturated);
		}

		template<> v_float32x8 v_convert<v_float32x8, v_uint32x8>(const v_uint32x8& input)
		{
			static const __m256i sign_mask = _mm256_set1_epi32(0x80000000);
			static const __m256i abs_mask = _mm256_set1_epi32(0x7FFFFFFF);
			static const __m256 large_num = _mm256_set1_ps(2147483648.0f);

			__m256i sign = _mm256_and_si256(input.data, sign_mask);
			__m256i abs_val = _mm256_and_si256(input.data, abs_mask);

			__m256 float_abs = _mm256_cvtepi32_ps(abs_val);
			__m256 sign_float = _mm256_castsi256_ps(sign);
			__m256 result = _mm256_or_ps(float_abs, sign_float);

			__m256 mask = _mm256_cmp_ps(result, large_num, _CMP_GE_OQ);
			return v_float32x8(_mm256_add_ps(result, _mm256_and_ps(mask, large_num)));
		}

		template<> v_uint32x8 v_convert<v_uint32x8, v_int32x8>(const v_int32x8& input)
		{
			static const __m256i zero = _mm256_setzero_si256();
			__m256i saturated = _mm256_max_epi32(input.data, zero);
			return v_uint32x8(saturated);
		}

		template<> v_float32x8 v_convert<v_float32x8, v_int32x8>(const v_int32x8& input)
		{
			return v_float32x8(_mm256_cvtepi32_ps(input.data));
		}

		template<> v_uint64x4 v_convert<v_uint64x4, v_int64x4>(const v_int64x4& input)
		{
			return v_uint64x4(
				static_cast<u64>(std::max(input.data.m256i_i64[0], static_cast<i64>(0))),
				static_cast<u64>(std::max(input.data.m256i_i64[1], static_cast<i64>(0))),
				static_cast<u64>(std::max(input.data.m256i_i64[2], static_cast<i64>(0))),
				static_cast<u64>(std::max(input.data.m256i_i64[3], static_cast<i64>(0)))
			);
		}

		template<> v_float64x4 v_convert<v_float64x4, v_int64x4>(const v_int64x4& input)
		{
			return v_float64x4(
				static_cast<f64>(std::max(input.data.m256i_i64[0], static_cast<i64>(0))),
				static_cast<f64>(std::max(input.data.m256i_i64[1], static_cast<i64>(0))),
				static_cast<f64>(std::max(input.data.m256i_i64[2], static_cast<i64>(0))),
				static_cast<f64>(std::max(input.data.m256i_i64[3], static_cast<i64>(0)))
			);
		}

		template<> v_int64x4 v_convert<v_int64x4, v_uint64x4>(const v_uint64x4& input)
		{
			return v_int64x4(
				static_cast<i64>(std::max(input.data.m256i_u64[0], static_cast<u64>(0))),
				static_cast<i64>(std::max(input.data.m256i_u64[1], static_cast<u64>(0))),
				static_cast<i64>(std::max(input.data.m256i_u64[2], static_cast<u64>(0))),
				static_cast<i64>(std::max(input.data.m256i_u64[3], static_cast<u64>(0)))
			);
		}

		template<> v_float64x4 v_convert<v_float64x4, v_uint64x4>(const v_uint64x4& input)
		{
			return v_float64x4(
				static_cast<f64>(std::max(input.data.m256i_u64[0], static_cast<u64>(0))),
				static_cast<f64>(std::max(input.data.m256i_u64[1], static_cast<u64>(0))),
				static_cast<f64>(std::max(input.data.m256i_u64[2], static_cast<u64>(0))),
				static_cast<f64>(std::max(input.data.m256i_u64[3], static_cast<u64>(0)))
			);
		}

		template<> v_int16x16 v_convert<v_int16x16, v_float16x16>(const v_float16x16& input)
		{
			__m256 fp32_low = _mm256_cvtph_ps(_mm256_extractf128_si256(input.data, 0));
			__m256 fp32_high = _mm256_cvtph_ps(_mm256_extractf128_si256(input.data, 1));

			__m256i int32_low = _mm256_cvtps_epi32(_mm256_round_ps(fp32_low, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
			__m256i int32_high = _mm256_cvtps_epi32(_mm256_round_ps(fp32_high, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));

			__m256i packed = _mm256_packs_epi32(int32_low, int32_high);
			__m256i result = _mm256_permute4x64_epi64(packed, _MM_SHUFFLE(3, 1, 2, 0));
			return v_int16x16(result);
		}

		template<> v_uint16x16 v_convert<v_uint16x16, v_float16x16>(const v_float16x16& input)
		{
			static const __m256i zero_mask = _mm256_setzero_si256();
			static const __m256i full_mask = _mm256_set1_epi32(0x80000000);

			__m256 fp32_low = _mm256_cvtph_ps(_mm256_extractf128_si256(input.data, 0));
			__m256 fp32_high = _mm256_cvtph_ps(_mm256_extractf128_si256(input.data, 1));

			__m256i int32_low = _mm256_cvtps_epi32(_mm256_round_ps(fp32_low, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
			__m256i int32_high = _mm256_cvtps_epi32(_mm256_round_ps(fp32_high, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));

			__m256i sign_mask_low = _mm256_cmpgt_epi32(zero_mask, int32_low);
			__m256i sign_mask_high = _mm256_cmpgt_epi32(zero_mask, int32_high);

			int32_low = _mm256_add_epi32(int32_low, _mm256_and_si256(sign_mask_low, full_mask));
			int32_high = _mm256_add_epi32(int32_high, _mm256_and_si256(sign_mask_high, full_mask));

			__m256i packed = _mm256_packus_epi32(int32_low, int32_high);
			__m256i result = _mm256_permute4x64_epi64(packed, _MM_SHUFFLE(3, 1, 2, 0));
			return v_uint16x16(result);
		}

		template<> v_int32x8 v_convert<v_int32x8, v_float32x8>(const v_float32x8& input)
		{
			return v_int32x8(_mm256_cvtps_epi32(_mm256_round_ps(input.data, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC)));
		}

		template<> v_uint32x8 v_convert<v_uint32x8, v_float32x8>(const v_float32x8& input)
		{
			static const __m256i zero_mask = _mm256_setzero_si256();
			static const __m256i full_mask = _mm256_set1_epi32(0x80000000);

			__m256i signed_int = _mm256_cvtps_epi32(input.data);
			__m256i sign_mask = _mm256_cmpgt_epi32(zero_mask, signed_int);
			__m256i result = _mm256_add_epi32(signed_int, _mm256_and_si256(sign_mask, full_mask));
			return v_uint32x8(result);
		}

		template<> v_int64x4 v_convert<v_int64x4, v_float64x4>(const v_float64x4& input)
		{
			v_int64x4 result;
			for (usize i = 0; i < 4; ++i)
			{
				f64 rounded = std::round(input.data.m256d_f64[i]);
				if (rounded > static_cast<f64>(std::numeric_limits<i64>::max()))
				{
					result.data.m256i_i64[i] = std::numeric_limits<i64>::max();
				}
				else if (rounded < static_cast<f64>(std::numeric_limits<i64>::min()))
				{
					result.data.m256i_i64[i] = std::numeric_limits<i64>::min();
				}
				else
				{
					result.data.m256i_i64[i] = static_cast<i64>(rounded);
				}
			}
			return result;
		}

		template<> v_uint64x4 v_convert<v_uint64x4, v_float64x4>(const v_float64x4& input)
		{
			v_uint64x4 result;
			for (usize i = 0; i < 4; ++i)
			{
				f64 rounded = std::round(input.data.m256d_f64[i]);
				if (rounded > static_cast<f64>(std::numeric_limits<u64>::max()))
				{
					result.data.m256i_u64[i] = std::numeric_limits<u64>::max();
				}
				else if (rounded < 0.0)
				{
					result.data.m256i_u64[i] = 0;
				}
				else
				{
					result.data.m256i_u64[i] = static_cast<u64>(rounded);
				}
			}
			return result;
		}
	}
}

export namespace fy
{
	namespace simd
	{
		template<VectorType Src_t, usize N_Dims> struct load_deinterleave_Invoker
		{
			load_deinterleave_Invoker() noexcept { std::unreachable(); }
			void operator () (const typename Src_t::scalar_t* src_scalar_ptr)
			{
				std::unreachable();
			}
		};

		template<> struct load_deinterleave_Invoker<v_uint8x32, 2>
		{
			v_uint8x32& dst_vec0;
			v_uint8x32& dst_vec1;

			load_deinterleave_Invoker(v_uint8x32& dst0, v_uint8x32& dst1) noexcept
				: dst_vec0(dst0), dst_vec1(dst1)
			{
			}

			void operator () (const u8* src_scalar_ptr) noexcept
			{
				__m256i ab0 = _mm256_load_si256(reinterpret_cast<const __m256i*>(src_scalar_ptr + (v_uint8x32::batch_size * 0)));
				__m256i ab1 = _mm256_load_si256(reinterpret_cast<const __m256i*>(src_scalar_ptr + (v_uint8x32::batch_size * 1)));

				static const __m256i shuffle_mask = _mm256_setr_epi8(
					0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15,
					0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15
				);

				__m256i p0 = _mm256_shuffle_epi8(ab0, shuffle_mask);
				__m256i p1 = _mm256_shuffle_epi8(ab1, shuffle_mask);

				__m256i pl = _mm256_permute2x128_si256(p0, p1, 0 + 2 * 16);
				__m256i ph = _mm256_permute2x128_si256(p0, p1, 1 + 3 * 16);

				dst_vec0.data = _mm256_unpacklo_epi64(pl, ph);
				dst_vec1.data = _mm256_unpackhi_epi64(pl, ph);
			}
		};

		template<> struct load_deinterleave_Invoker<v_uint8x32, 3>
		{
			v_uint8x32& dst_vec0;
			v_uint8x32& dst_vec1;
			v_uint8x32& dst_vec2;

			load_deinterleave_Invoker(v_uint8x32& dst0, v_uint8x32& dst1, v_uint8x32& dst2) noexcept
				: dst_vec0(dst0), dst_vec1(dst1), dst_vec2(dst2)
			{
			}

			void operator () (const u8* src_scalar_ptr) noexcept
			{
				__m256i bgr0 = _mm256_load_si256(reinterpret_cast<const __m256i*>(src_scalar_ptr + (v_uint8x32::batch_size * 0)));
				__m256i bgr1 = _mm256_load_si256(reinterpret_cast<const __m256i*>(src_scalar_ptr + (v_uint8x32::batch_size * 1)));
				__m256i bgr2 = _mm256_load_si256(reinterpret_cast<const __m256i*>(src_scalar_ptr + (v_uint8x32::batch_size * 2)));

				__m256i s02_low = _mm256_permute2x128_si256(bgr0, bgr2, 0 + 2 * 16);
				__m256i s02_high = _mm256_permute2x128_si256(bgr0, bgr2, 1 + 3 * 16);

				static const __m256i m0 = _mm256_setr_epi8(
					0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0,
					0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0
				);

				static const __m256i m1 = _mm256_setr_epi8(
					0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0,
					-1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1
				);

				__m256i b0 = _mm256_blendv_epi8(_mm256_blendv_epi8(s02_low, s02_high, m0), bgr1, m1);
				__m256i g0 = _mm256_blendv_epi8(_mm256_blendv_epi8(s02_high, s02_low, m1), bgr1, m0);
				__m256i r0 = _mm256_blendv_epi8(_mm256_blendv_epi8(bgr1, s02_low, m0), s02_high, m1);

				static const __m256i sh_b = _mm256_setr_epi8(
					0, 3, 6, 9, 12, 15, 2, 5, 8, 11, 14, 1, 4, 7, 10, 13,
					0, 3, 6, 9, 12, 15, 2, 5, 8, 11, 14, 1, 4, 7, 10, 13
				);

				static const __m256i sh_g = _mm256_setr_epi8(
					1, 4, 7, 10, 13, 0, 3, 6, 9, 12, 15, 2, 5, 8, 11, 14,
					1, 4, 7, 10, 13, 0, 3, 6, 9, 12, 15, 2, 5, 8, 11, 14
				);

				static const __m256i sh_r = _mm256_setr_epi8(
					2, 5, 8, 11, 14, 1, 4, 7, 10, 13, 0, 3, 6, 9, 12, 15,
					2, 5, 8, 11, 14, 1, 4, 7, 10, 13, 0, 3, 6, 9, 12, 15
				);

				dst_vec0.data = _mm256_shuffle_epi8(b0, sh_b);
				dst_vec1.data = _mm256_shuffle_epi8(g0, sh_g);
				dst_vec2.data = _mm256_shuffle_epi8(r0, sh_r);
			}
		};

		template<> struct load_deinterleave_Invoker<v_uint8x32, 4>
		{
			v_uint8x32& dst_vec0;
			v_uint8x32& dst_vec1;
			v_uint8x32& dst_vec2;
			v_uint8x32& dst_vec3;

			load_deinterleave_Invoker(v_uint8x32& dst0, v_uint8x32& dst1, v_uint8x32& dst2, v_uint8x32& dst3) noexcept
				: dst_vec0(dst0), dst_vec1(dst1), dst_vec2(dst2), dst_vec3(dst3)
			{
			}

			void operator () (const u8* src_scalar_ptr) noexcept
			{
				__m256i bgr0 = _mm256_load_si256(reinterpret_cast<const __m256i*>(src_scalar_ptr + (v_uint8x32::batch_size * 0)));
				__m256i bgr1 = _mm256_load_si256(reinterpret_cast<const __m256i*>(src_scalar_ptr + (v_uint8x32::batch_size * 1)));
				__m256i bgr2 = _mm256_load_si256(reinterpret_cast<const __m256i*>(src_scalar_ptr + (v_uint8x32::batch_size * 2)));
				__m256i bgr3 = _mm256_load_si256(reinterpret_cast<const __m256i*>(src_scalar_ptr + (v_uint8x32::batch_size * 3)));

				static const __m256i mask = _mm256_setr_epi8(
					0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15,
					0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15
				);

				__m256i p0 = _mm256_shuffle_epi8(bgr0, mask);
				__m256i p1 = _mm256_shuffle_epi8(bgr1, mask);
				__m256i p2 = _mm256_shuffle_epi8(bgr2, mask);
				__m256i p3 = _mm256_shuffle_epi8(bgr3, mask);

				__m256i p01l = _mm256_unpacklo_epi32(p0, p1);
				__m256i p01h = _mm256_unpackhi_epi32(p0, p1);
				__m256i p23l = _mm256_unpacklo_epi32(p2, p3);
				__m256i p23h = _mm256_unpackhi_epi32(p2, p3);

				__m256i pll = _mm256_permute2x128_si256(p01l, p23l, 0 + 2 * 16);
				__m256i plh = _mm256_permute2x128_si256(p01l, p23l, 1 + 3 * 16);
				__m256i phl = _mm256_permute2x128_si256(p01h, p23h, 0 + 2 * 16);
				__m256i phh = _mm256_permute2x128_si256(p01h, p23h, 1 + 3 * 16);

				dst_vec0.data = _mm256_unpacklo_epi32(pll, plh);
				dst_vec1.data = _mm256_unpackhi_epi32(pll, plh);
				dst_vec2.data = _mm256_unpacklo_epi32(phl, phh);
				dst_vec3.data = _mm256_unpackhi_epi32(phl, phh);
			}
		};

		template<> struct load_deinterleave_Invoker<v_uint16x16, 2>
		{
			v_uint16x16& dst_vec0;
			v_uint16x16& dst_vec1;

			load_deinterleave_Invoker(v_uint16x16& dst0, v_uint16x16& dst1) noexcept : dst_vec0(dst0), dst_vec1(dst1) {}

			void operator () (const u16* src_scalar_ptr) noexcept
			{
				__m256i ab0 = _mm256_load_si256(reinterpret_cast<const __m256i*>(src_scalar_ptr + (v_uint16x16::batch_size * 0)));
				__m256i ab1 = _mm256_load_si256(reinterpret_cast<const __m256i*>(src_scalar_ptr + (v_uint16x16::batch_size * 1)));

				static const __m256i sh = _mm256_setr_epi8(
					0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15,
					0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15
				);

				__m256i p0 = _mm256_shuffle_epi8(ab0, sh);
				__m256i p1 = _mm256_shuffle_epi8(ab1, sh);
				__m256i pl = _mm256_permute2x128_si256(p0, p1, 0 + 2 * 16);
				__m256i ph = _mm256_permute2x128_si256(p0, p1, 1 + 3 * 16);

				dst_vec0.data = _mm256_unpacklo_epi64(pl, ph);
				dst_vec1.data = _mm256_unpackhi_epi64(pl, ph);
			}
		};

		template<> struct load_deinterleave_Invoker<v_uint16x16, 3>
		{
			v_uint16x16& dst_vec0;
			v_uint16x16& dst_vec1;
			v_uint16x16& dst_vec2;

			load_deinterleave_Invoker(v_uint16x16& dst0, v_uint16x16& dst1, v_uint16x16& dst2) noexcept : dst_vec0(dst0), dst_vec1(dst1), dst_vec2(dst2) {}

			void operator () (const u16* src_scalar_ptr) noexcept
			{
				__m256i bgr0 = _mm256_load_si256(reinterpret_cast<const __m256i*>(src_scalar_ptr + (v_uint16x16::batch_size * 0)));
				__m256i bgr1 = _mm256_load_si256(reinterpret_cast<const __m256i*>(src_scalar_ptr + (v_uint16x16::batch_size * 1)));
				__m256i bgr2 = _mm256_load_si256(reinterpret_cast<const __m256i*>(src_scalar_ptr + (v_uint16x16::batch_size * 2)));

				__m256i s02_low = _mm256_permute2x128_si256(bgr0, bgr2, 0 + 2 * 16);
				__m256i s02_high = _mm256_permute2x128_si256(bgr0, bgr2, 1 + 3 * 16);

				static const __m256i m0 = _mm256_setr_epi8(
					0, 0, -1, -1, 0, 0, 0, 0, -1, -1, 0, 0, 0, 0, -1, -1,
					0, 0, 0, 0, -1, -1, 0, 0, 0, 0, -1, -1, 0, 0, 0, 0
				);

				static const __m256i m1 = _mm256_setr_epi8(
					0, 0, 0, 0, -1, -1, 0, 0, 0, 0, -1, -1, 0, 0, 0, 0,
					-1, -1, 0, 0, 0, 0, -1, -1, 0, 0, 0, 0, -1, -1, 0, 0
				);

				__m256i b0 = _mm256_blendv_epi8(_mm256_blendv_epi8(s02_low, s02_high, m0), bgr1, m1);
				__m256i g0 = _mm256_blendv_epi8(_mm256_blendv_epi8(bgr1, s02_low, m0), s02_high, m1);
				__m256i r0 = _mm256_blendv_epi8(_mm256_blendv_epi8(s02_high, s02_low, m1), bgr1, m0);

				static const __m256i sh_b = _mm256_setr_epi8(
					0, 1, 6, 7, 12, 13, 2, 3, 8, 9, 14, 15, 4, 5, 10, 11,
					0, 1, 6, 7, 12, 13, 2, 3, 8, 9, 14, 15, 4, 5, 10, 11
				);

				static const __m256i sh_g = _mm256_setr_epi8(
					2, 3, 8, 9, 14, 15, 4, 5, 10, 11, 0, 1, 6, 7, 12, 13,
					2, 3, 8, 9, 14, 15, 4, 5, 10, 11, 0, 1, 6, 7, 12, 13
				);

				static const __m256i sh_r = _mm256_setr_epi8(
					4, 5, 10, 11, 0, 1, 6, 7, 12, 13, 2, 3, 8, 9, 14, 15,
					4, 5, 10, 11, 0, 1, 6, 7, 12, 13, 2, 3, 8, 9, 14, 15
				);

				dst_vec0.data = _mm256_shuffle_epi8(b0, sh_b);
				dst_vec1.data = _mm256_shuffle_epi8(g0, sh_g);
				dst_vec2.data = _mm256_shuffle_epi8(r0, sh_r);
			}
		};

		template<> struct load_deinterleave_Invoker<v_uint16x16, 4>
		{
			v_uint16x16& dst_vec0;
			v_uint16x16& dst_vec1;
			v_uint16x16& dst_vec2;
			v_uint16x16& dst_vec3;

			load_deinterleave_Invoker(v_uint16x16& dst0, v_uint16x16& dst1, v_uint16x16& dst2, v_uint16x16& dst3) noexcept
				: dst_vec0(dst0), dst_vec1(dst1), dst_vec2(dst2), dst_vec3(dst3)
			{
			}

			void operator () (const u16* src_scalar_ptr) noexcept
			{
				__m256i bgr0 = _mm256_load_si256(reinterpret_cast<const __m256i*>(src_scalar_ptr + (v_uint16x16::batch_size * 0)));
				__m256i bgr1 = _mm256_load_si256(reinterpret_cast<const __m256i*>(src_scalar_ptr + (v_uint16x16::batch_size * 1)));
				__m256i bgr2 = _mm256_load_si256(reinterpret_cast<const __m256i*>(src_scalar_ptr + (v_uint16x16::batch_size * 2)));
				__m256i bgr3 = _mm256_load_si256(reinterpret_cast<const __m256i*>(src_scalar_ptr + (v_uint16x16::batch_size * 3)));

				static const __m256i sh = _mm256_setr_epi8(
					0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15,
					0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15
				);
				__m256i p0 = _mm256_shuffle_epi8(bgr0, sh);
				__m256i p1 = _mm256_shuffle_epi8(bgr1, sh);
				__m256i p2 = _mm256_shuffle_epi8(bgr2, sh);
				__m256i p3 = _mm256_shuffle_epi8(bgr3, sh);

				__m256i p01l = _mm256_unpacklo_epi32(p0, p1);
				__m256i p01h = _mm256_unpackhi_epi32(p0, p1);
				__m256i p23l = _mm256_unpacklo_epi32(p2, p3);
				__m256i p23h = _mm256_unpackhi_epi32(p2, p3);

				__m256i pll = _mm256_permute2x128_si256(p01l, p23l, 0 + 2 * 16);
				__m256i plh = _mm256_permute2x128_si256(p01l, p23l, 1 + 3 * 16);
				__m256i phl = _mm256_permute2x128_si256(p01h, p23h, 0 + 2 * 16);
				__m256i phh = _mm256_permute2x128_si256(p01h, p23h, 1 + 3 * 16);

				dst_vec0.data = _mm256_unpacklo_epi32(pll, plh);
				dst_vec1.data = _mm256_unpackhi_epi32(pll, plh);
				dst_vec2.data = _mm256_unpacklo_epi32(phl, phh);
				dst_vec3.data = _mm256_unpackhi_epi32(phl, phh);
			}
		};

		template<> struct load_deinterleave_Invoker<v_uint32x8, 2>
		{
			v_uint32x8& dst_vec0;
			v_uint32x8& dst_vec1;

			load_deinterleave_Invoker(v_uint32x8& dst0, v_uint32x8& dst1) noexcept
				: dst_vec0(dst0), dst_vec1(dst1)
			{
			}

			void operator () (const u32* src_scalar_ptr) noexcept
			{
				__m256i ab0 = _mm256_load_si256(reinterpret_cast<const __m256i*>(src_scalar_ptr + (v_uint32x8::batch_size * 0)));
				__m256i ab1 = _mm256_load_si256(reinterpret_cast<const __m256i*>(src_scalar_ptr + (v_uint32x8::batch_size * 1)));

				static constexpr i32 sh = 0 + 2 * 4 + 1 * 16 + 3 * 64;

				__m256i p0 = _mm256_shuffle_epi32(ab0, sh);
				__m256i p1 = _mm256_shuffle_epi32(ab1, sh);

				__m256i pl = _mm256_permute2x128_si256(p0, p1, 0 + 2 * 16);
				__m256i ph = _mm256_permute2x128_si256(p0, p1, 1 + 3 * 16);

				dst_vec0.data = _mm256_unpacklo_epi64(pl, ph);
				dst_vec1.data = _mm256_unpackhi_epi64(pl, ph);
			}
		};

		template<> struct load_deinterleave_Invoker<v_uint32x8, 3>
		{
			v_uint32x8& dst_vec0;
			v_uint32x8& dst_vec1;
			v_uint32x8& dst_vec2;

			load_deinterleave_Invoker(v_uint32x8& dst0, v_uint32x8& dst1, v_uint32x8& dst2) noexcept : dst_vec0(dst0), dst_vec1(dst1), dst_vec2(dst2) {}
			void operator () (const u32* src_scalar_ptr) noexcept
			{
				__m256i bgr0 = _mm256_load_si256(reinterpret_cast<const __m256i*>(src_scalar_ptr + (v_uint32x8::batch_size * 0)));
				__m256i bgr1 = _mm256_load_si256(reinterpret_cast<const __m256i*>(src_scalar_ptr + (v_uint32x8::batch_size * 1)));
				__m256i bgr2 = _mm256_load_si256(reinterpret_cast<const __m256i*>(src_scalar_ptr + (v_uint32x8::batch_size * 2)));

				__m256i s02_low = _mm256_permute2x128_si256(bgr0, bgr2, 0 + 2 * 16);
				__m256i s02_high = _mm256_permute2x128_si256(bgr0, bgr2, 1 + 3 * 16);

				__m256i b0 = _mm256_blend_epi32(_mm256_blend_epi32(s02_low, s02_high, 0x24), bgr1, 0x92);
				__m256i g0 = _mm256_blend_epi32(_mm256_blend_epi32(s02_high, s02_low, 0x92), bgr1, 0x24);
				__m256i r0 = _mm256_blend_epi32(_mm256_blend_epi32(bgr1, s02_low, 0x24), s02_high, 0x92);

				dst_vec0.data = _mm256_shuffle_epi32(b0, 0x6c);
				dst_vec1.data = _mm256_shuffle_epi32(g0, 0xb1);
				dst_vec2.data = _mm256_shuffle_epi32(r0, 0xc6);
			}
		};

		template<> struct load_deinterleave_Invoker<v_uint32x8, 4>
		{
			v_uint32x8& dst_vec0;
			v_uint32x8& dst_vec1;
			v_uint32x8& dst_vec2;
			v_uint32x8& dst_vec3;

			load_deinterleave_Invoker(v_uint32x8& dst0, v_uint32x8& dst1, v_uint32x8& dst2, v_uint32x8& dst3) noexcept
				: dst_vec0(dst0), dst_vec1(dst1), dst_vec2(dst2), dst_vec3(dst3)
			{
			}

			void operator () (const u32* src_scalar_ptr) noexcept
			{
				__m256i p0 = _mm256_load_si256(reinterpret_cast<const __m256i*>(src_scalar_ptr + (v_uint32x8::batch_size * 0)));
				__m256i p1 = _mm256_load_si256(reinterpret_cast<const __m256i*>(src_scalar_ptr + (v_uint32x8::batch_size * 1)));
				__m256i p2 = _mm256_load_si256(reinterpret_cast<const __m256i*>(src_scalar_ptr + (v_uint32x8::batch_size * 2)));
				__m256i p3 = _mm256_load_si256(reinterpret_cast<const __m256i*>(src_scalar_ptr + (v_uint32x8::batch_size * 3)));

				__m256i p01l = _mm256_unpacklo_epi32(p0, p1);
				__m256i p01h = _mm256_unpackhi_epi32(p0, p1);
				__m256i p23l = _mm256_unpacklo_epi32(p2, p3);
				__m256i p23h = _mm256_unpackhi_epi32(p2, p3);

				__m256i pll = _mm256_permute2x128_si256(p01l, p23l, 0 + 2 * 16);
				__m256i plh = _mm256_permute2x128_si256(p01l, p23l, 1 + 3 * 16);
				__m256i phl = _mm256_permute2x128_si256(p01h, p23h, 0 + 2 * 16);
				__m256i phh = _mm256_permute2x128_si256(p01h, p23h, 1 + 3 * 16);

				dst_vec0.data = _mm256_unpacklo_epi32(pll, plh);
				dst_vec1.data = _mm256_unpackhi_epi32(pll, plh);
				dst_vec2.data = _mm256_unpacklo_epi32(phl, phh);
				dst_vec3.data = _mm256_unpackhi_epi32(phl, phh);
			}
		};

		template<> struct load_deinterleave_Invoker<v_uint64x4, 2>
		{
			v_uint64x4& dst_vec0;
			v_uint64x4& dst_vec1;

			load_deinterleave_Invoker(v_uint64x4& dst0, v_uint64x4& dst1) noexcept : dst_vec0(dst0), dst_vec1(dst1) {}

			void operator () (const u64* src_scalar_ptr) noexcept
			{
				__m256i ab0 = _mm256_load_si256(reinterpret_cast<const __m256i*>(src_scalar_ptr + (v_uint64x4::batch_size * 0)));
				__m256i ab1 = _mm256_load_si256(reinterpret_cast<const __m256i*>(src_scalar_ptr + (v_uint64x4::batch_size * 1)));

				__m256i pl = _mm256_permute2x128_si256(ab0, ab1, 0 + 2 * 16);
				__m256i ph = _mm256_permute2x128_si256(ab0, ab1, 1 + 3 * 16);

				dst_vec0.data = _mm256_unpacklo_epi64(pl, ph);
				dst_vec1.data = _mm256_unpackhi_epi64(pl, ph);
			}
		};

		template<> struct load_deinterleave_Invoker<v_uint64x4, 3>
		{
			v_uint64x4& dst_vec0;
			v_uint64x4& dst_vec1;
			v_uint64x4& dst_vec2;

			load_deinterleave_Invoker(v_uint64x4& dst0, v_uint64x4& dst1, v_uint64x4& dst2) noexcept : dst_vec0(dst0), dst_vec1(dst1), dst_vec2(dst2) {}

			void operator () (const u64* src_scalar_ptr) noexcept
			{
				__m256i bgr0 = _mm256_load_si256(reinterpret_cast<const __m256i*>(src_scalar_ptr + (v_uint64x4::batch_size * 0)));
				__m256i bgr1 = _mm256_load_si256(reinterpret_cast<const __m256i*>(src_scalar_ptr + (v_uint64x4::batch_size * 1)));
				__m256i bgr2 = _mm256_load_si256(reinterpret_cast<const __m256i*>(src_scalar_ptr + (v_uint64x4::batch_size * 2)));

				__m256i s01 = _mm256_blend_epi32(bgr0, bgr1, 0xf0);
				__m256i s12 = _mm256_blend_epi32(bgr1, bgr2, 0xf0);
				__m256i s20r = _mm256_permute4x64_epi64(_mm256_blend_epi32(bgr2, bgr0, 0xf0), 0x1b);

				dst_vec0.data = _mm256_unpacklo_epi64(s01, s20r);
				dst_vec1.data = _mm256_alignr_epi8(s12, s01, 8);
				dst_vec2.data = _mm256_unpackhi_epi64(s20r, s12);
			}
		};

		template<> struct load_deinterleave_Invoker<v_uint64x4, 4>
		{
			v_uint64x4& dst_vec0;
			v_uint64x4& dst_vec1;
			v_uint64x4& dst_vec2;
			v_uint64x4& dst_vec3;

			load_deinterleave_Invoker(v_uint64x4& dst0, v_uint64x4& dst1, v_uint64x4& dst2, v_uint64x4& dst3) noexcept
				: dst_vec0(dst0), dst_vec1(dst1), dst_vec2(dst2), dst_vec3(dst3)
			{
			}

			void operator () (const u64* src_scalar_ptr) noexcept
			{
				__m256i bgra0 = _mm256_load_si256(reinterpret_cast<const __m256i*>(src_scalar_ptr + (v_uint64x4::batch_size * 0)));
				__m256i bgra1 = _mm256_load_si256(reinterpret_cast<const __m256i*>(src_scalar_ptr + (v_uint64x4::batch_size * 1)));
				__m256i bgra2 = _mm256_load_si256(reinterpret_cast<const __m256i*>(src_scalar_ptr + (v_uint64x4::batch_size * 2)));
				__m256i bgra3 = _mm256_load_si256(reinterpret_cast<const __m256i*>(src_scalar_ptr + (v_uint64x4::batch_size * 3)));

				__m256i l02 = _mm256_permute2x128_si256(bgra0, bgra2, 0 + 2 * 16);
				__m256i h02 = _mm256_permute2x128_si256(bgra0, bgra2, 1 + 3 * 16);
				__m256i l13 = _mm256_permute2x128_si256(bgra1, bgra3, 0 + 2 * 16);
				__m256i h13 = _mm256_permute2x128_si256(bgra1, bgra3, 1 + 3 * 16);

				dst_vec0.data = _mm256_unpacklo_epi64(l02, l13);
				dst_vec1.data = _mm256_unpackhi_epi64(l02, l13);
				dst_vec2.data = _mm256_unpacklo_epi64(h02, h13);
				dst_vec3.data = _mm256_unpackhi_epi64(h02, h13);
			}
		};

		template<> struct load_deinterleave_Invoker<v_int8x32, 2> : public load_deinterleave_Invoker<v_uint8x32, 2>
		{
			load_deinterleave_Invoker(v_int8x32& dst0, v_int8x32& dst1) noexcept
				: load_deinterleave_Invoker<v_uint8x32, 2>(reinterpret_cast<v_uint8x32&>(dst0), reinterpret_cast<v_uint8x32&>(dst1)) {}

			void operator () (const i8* src_scalar_ptr) noexcept
			{
				load_deinterleave_Invoker<v_uint8x32, 2>::operator()(reinterpret_cast<const u8*>(src_scalar_ptr));
			}
		};

		template<> struct load_deinterleave_Invoker<v_int8x32, 3> : public load_deinterleave_Invoker<v_uint8x32, 3>
		{
			load_deinterleave_Invoker(v_int8x32& dst0, v_int8x32& dst1, v_int8x32& dst2) noexcept
				: load_deinterleave_Invoker<v_uint8x32, 3>(reinterpret_cast<v_uint8x32&>(dst0), reinterpret_cast<v_uint8x32&>(dst1), reinterpret_cast<v_uint8x32&>(dst2)) {}

			void operator () (const i8* src_scalar_ptr) noexcept
			{
				load_deinterleave_Invoker<v_uint8x32, 3>::operator()(reinterpret_cast<const u8*>(src_scalar_ptr));
			}
		};

		template<> struct load_deinterleave_Invoker<v_int8x32, 4> : public load_deinterleave_Invoker<v_uint8x32, 4>
		{
			load_deinterleave_Invoker(v_int8x32& dst0, v_int8x32& dst1, v_int8x32& dst2, v_int8x32& dst3) noexcept
				: load_deinterleave_Invoker<v_uint8x32, 4>(
					reinterpret_cast<v_uint8x32&>(dst0), reinterpret_cast<v_uint8x32&>(dst1),
					reinterpret_cast<v_uint8x32&>(dst2), reinterpret_cast<v_uint8x32&>(dst3))
			{
			}

			void operator () (const i8* src_scalar_ptr) noexcept
			{
				load_deinterleave_Invoker<v_uint8x32, 4>::operator()(reinterpret_cast<const u8*>(src_scalar_ptr));
			}
		};

		template<> struct load_deinterleave_Invoker<v_int16x16, 2> : public load_deinterleave_Invoker<v_uint16x16, 2>
		{
			load_deinterleave_Invoker(v_int16x16& dst0, v_int16x16& dst1) noexcept
				: load_deinterleave_Invoker<v_uint16x16, 2>(reinterpret_cast<v_uint16x16&>(dst0), reinterpret_cast<v_uint16x16&>(dst1)) {}

			void operator () (const i16* src_scalar_ptr) noexcept
			{
				load_deinterleave_Invoker<v_uint16x16, 2>::operator()(reinterpret_cast<const u16*>(src_scalar_ptr));
			}
		};

		template<> struct load_deinterleave_Invoker<v_int16x16, 3> : public load_deinterleave_Invoker<v_uint16x16, 3>
		{
			load_deinterleave_Invoker(v_int16x16& dst0, v_int16x16& dst1, v_int16x16& dst2) noexcept
				: load_deinterleave_Invoker<v_uint16x16, 3>(reinterpret_cast<v_uint16x16&>(dst0), reinterpret_cast<v_uint16x16&>(dst1), reinterpret_cast<v_uint16x16&>(dst2)) {}

			void operator () (const i16* src_scalar_ptr) noexcept
			{
				load_deinterleave_Invoker<v_uint16x16, 3>::operator()(reinterpret_cast<const u16*>(src_scalar_ptr));
			}
		};

		template<> struct load_deinterleave_Invoker<v_int16x16, 4> : public load_deinterleave_Invoker<v_uint16x16, 4>
		{
			load_deinterleave_Invoker(v_int16x16& dst0, v_int16x16& dst1, v_int16x16& dst2, v_int16x16& dst3) noexcept
				: load_deinterleave_Invoker<v_uint16x16, 4>(
					reinterpret_cast<v_uint16x16&>(dst0), reinterpret_cast<v_uint16x16&>(dst1),
					reinterpret_cast<v_uint16x16&>(dst2), reinterpret_cast<v_uint16x16&>(dst3))
			{
			}

			void operator () (const i16* src_scalar_ptr) noexcept
			{
				load_deinterleave_Invoker<v_uint16x16, 4>::operator()(reinterpret_cast<const u16*>(src_scalar_ptr));
			}
		};

		template<> struct load_deinterleave_Invoker<v_int32x8, 2> : public load_deinterleave_Invoker<v_uint32x8, 2>
		{
			load_deinterleave_Invoker(v_int32x8& dst0, v_int32x8& dst1) noexcept
				: load_deinterleave_Invoker<v_uint32x8, 2>(reinterpret_cast<v_uint32x8&>(dst0), reinterpret_cast<v_uint32x8&>(dst1)) {}

			void operator () (const i32* src_scalar_ptr) noexcept
			{
				load_deinterleave_Invoker<v_uint32x8, 2>::operator()(reinterpret_cast<const u32*>(src_scalar_ptr));
			}
		};

		template<> struct load_deinterleave_Invoker<v_int32x8, 3> : public load_deinterleave_Invoker<v_uint32x8, 3>
		{
			load_deinterleave_Invoker(v_int32x8& dst0, v_int32x8& dst1, v_int32x8& dst2) noexcept
				: load_deinterleave_Invoker<v_uint32x8, 3>(reinterpret_cast<v_uint32x8&>(dst0), reinterpret_cast<v_uint32x8&>(dst1), reinterpret_cast<v_uint32x8&>(dst2)) {}

			void operator () (const i32* src_scalar_ptr) noexcept
			{
				load_deinterleave_Invoker<v_uint32x8, 3>::operator()(reinterpret_cast<const u32*>(src_scalar_ptr));
			}
		};

		template<> struct load_deinterleave_Invoker<v_int32x8, 4> : public load_deinterleave_Invoker<v_uint32x8, 4>
		{
			load_deinterleave_Invoker(v_int32x8& dst0, v_int32x8& dst1, v_int32x8& dst2, v_int32x8& dst3) noexcept
				: load_deinterleave_Invoker<v_uint32x8, 4>(
					reinterpret_cast<v_uint32x8&>(dst0), reinterpret_cast<v_uint32x8&>(dst1),
					reinterpret_cast<v_uint32x8&>(dst2), reinterpret_cast<v_uint32x8&>(dst3))
			{
			}

			void operator () (const i32* src_scalar_ptr) noexcept
			{
				load_deinterleave_Invoker<v_uint32x8, 4>::operator()(reinterpret_cast<const u32*>(src_scalar_ptr));
			}
		};

		template<> struct load_deinterleave_Invoker<v_int64x4, 2> : public load_deinterleave_Invoker<v_uint64x4, 2>
		{
			load_deinterleave_Invoker(v_int64x4& dst0, v_int64x4& dst1) noexcept
				: load_deinterleave_Invoker<v_uint64x4, 2>(reinterpret_cast<v_uint64x4&>(dst0), reinterpret_cast<v_uint64x4&>(dst1)) {}

			void operator () (const i64* src_scalar_ptr) noexcept
			{
				load_deinterleave_Invoker<v_uint64x4, 2>::operator()(reinterpret_cast<const u64*>(src_scalar_ptr));
			}
		};

		template<> struct load_deinterleave_Invoker<v_int64x4, 3> : public load_deinterleave_Invoker<v_uint64x4, 3>
		{
			load_deinterleave_Invoker(v_int64x4& dst0, v_int64x4& dst1, v_int64x4& dst2) noexcept
				: load_deinterleave_Invoker<v_uint64x4, 3>(
					reinterpret_cast<v_uint64x4&>(dst0), reinterpret_cast<v_uint64x4&>(dst1), reinterpret_cast<v_uint64x4&>(dst2))
			{
			}

			void operator () (const i64* src_scalar_ptr) noexcept
			{
				load_deinterleave_Invoker<v_uint64x4, 3>::operator()(reinterpret_cast<const u64*>(src_scalar_ptr));
			}
		};

		template<> struct load_deinterleave_Invoker<v_int64x4, 4> : public load_deinterleave_Invoker<v_uint64x4, 4>
		{
			load_deinterleave_Invoker(v_int64x4& dst0, v_int64x4& dst1, v_int64x4& dst2, v_int64x4& dst3) noexcept
				: load_deinterleave_Invoker<v_uint64x4, 4>(
					reinterpret_cast<v_uint64x4&>(dst0), reinterpret_cast<v_uint64x4&>(dst1),
					reinterpret_cast<v_uint64x4&>(dst2), reinterpret_cast<v_uint64x4&>(dst3))
			{
			}

			void operator () (const i64* src_scalar_ptr) noexcept
			{
				load_deinterleave_Invoker<v_uint64x4, 4>::operator()(reinterpret_cast<const u64*>(src_scalar_ptr));
			}
		};

		template<> struct load_deinterleave_Invoker<v_float32x8, 2> : public load_deinterleave_Invoker<v_uint32x8, 2>
		{
			load_deinterleave_Invoker(v_float32x8& dst0, v_float32x8& dst1) noexcept
				: load_deinterleave_Invoker<v_uint32x8, 2>(reinterpret_cast<v_uint32x8&>(dst0), reinterpret_cast<v_uint32x8&>(dst1)) {}

			void operator () (const f32* src_scalar_ptr) noexcept
			{
				load_deinterleave_Invoker<v_uint32x8, 2>::operator()(reinterpret_cast<const u32*>(src_scalar_ptr));
			}
		};

		template<> struct load_deinterleave_Invoker<v_float32x8, 3> : public load_deinterleave_Invoker<v_uint32x8, 3>
		{
			load_deinterleave_Invoker(v_float32x8& dst0, v_float32x8& dst1, v_float32x8& dst2) noexcept
				: load_deinterleave_Invoker<v_uint32x8, 3>(
					reinterpret_cast<v_uint32x8&>(dst0), reinterpret_cast<v_uint32x8&>(dst1), reinterpret_cast<v_uint32x8&>(dst2))
			{
			}

			void operator () (const f32* src_scalar_ptr) noexcept
			{
				load_deinterleave_Invoker<v_uint32x8, 3>::operator()(reinterpret_cast<const u32*>(src_scalar_ptr));
			}
		};

		template<> struct load_deinterleave_Invoker<v_float32x8, 4> : public load_deinterleave_Invoker<v_uint32x8, 4>
		{
			load_deinterleave_Invoker(v_float32x8& dst0, v_float32x8& dst1, v_float32x8& dst2, v_float32x8& dst3) noexcept
				: load_deinterleave_Invoker<v_uint32x8, 4>(
					reinterpret_cast<v_uint32x8&>(dst0), reinterpret_cast<v_uint32x8&>(dst1),
					reinterpret_cast<v_uint32x8&>(dst2), reinterpret_cast<v_uint32x8&>(dst3))
			{
			}

			void operator () (const f32* src_scalar_ptr) noexcept
			{
				load_deinterleave_Invoker<v_uint32x8, 4>::operator()(reinterpret_cast<const u32*>(src_scalar_ptr));
			}
		};

		template<> struct load_deinterleave_Invoker<v_float64x4, 2> : public load_deinterleave_Invoker<v_uint64x4, 2>
		{
			load_deinterleave_Invoker(v_float64x4& dst0, v_float64x4& dst1) noexcept
				: load_deinterleave_Invoker<v_uint64x4, 2>(reinterpret_cast<v_uint64x4&>(dst0), reinterpret_cast<v_uint64x4&>(dst1)) {}

			void operator () (const f64* src_scalar_ptr) noexcept
			{
				load_deinterleave_Invoker<v_uint64x4, 2>::operator()(reinterpret_cast<const u64*>(src_scalar_ptr));
			}
		};

		template<> struct load_deinterleave_Invoker<v_float64x4, 3> : public load_deinterleave_Invoker<v_uint64x4, 3>
		{
			load_deinterleave_Invoker(v_float64x4& dst0, v_float64x4& dst1, v_float64x4& dst2) noexcept
				: load_deinterleave_Invoker<v_uint64x4, 3>(reinterpret_cast<v_uint64x4&>(dst0), reinterpret_cast<v_uint64x4&>(dst1), reinterpret_cast<v_uint64x4&>(dst2)) {}

			void operator () (const f64* src_scalar_ptr) noexcept
			{
				load_deinterleave_Invoker<v_uint64x4, 3>::operator()(reinterpret_cast<const u64*>(src_scalar_ptr));
			}
		};

		template<> struct load_deinterleave_Invoker<v_float64x4, 4> : public load_deinterleave_Invoker<v_uint64x4, 4>
		{
			load_deinterleave_Invoker(v_float64x4& dst0, v_float64x4& dst1, v_float64x4& dst2, v_float64x4& dst3) noexcept
				: load_deinterleave_Invoker<v_uint64x4, 4>(
					reinterpret_cast<v_uint64x4&>(dst0), reinterpret_cast<v_uint64x4&>(dst1),
					reinterpret_cast<v_uint64x4&>(dst2), reinterpret_cast<v_uint64x4&>(dst3))
			{
			}

			void operator () (const f64* src_scalar_ptr) noexcept
			{
				load_deinterleave_Invoker<v_uint64x4, 4>::operator()(reinterpret_cast<const u64*>(src_scalar_ptr));
			}
		};

		template<> struct load_deinterleave_Invoker<v_float16x16, 2> : public load_deinterleave_Invoker<v_uint16x16, 2>
		{
			load_deinterleave_Invoker(v_float16x16& dst0, v_float16x16& dst1) noexcept
				: load_deinterleave_Invoker<v_uint16x16, 2>(reinterpret_cast<v_uint16x16&>(dst0), reinterpret_cast<v_uint16x16&>(dst1)) {}

			void operator () (const f16* src_scalar_ptr) noexcept
			{
				load_deinterleave_Invoker<v_uint16x16, 2>::operator()(reinterpret_cast<const u16*>(src_scalar_ptr));
			}
		};

		template<> struct load_deinterleave_Invoker<v_float16x16, 3> : public load_deinterleave_Invoker<v_uint16x16, 3>
		{
			load_deinterleave_Invoker(v_float16x16& dst0, v_float16x16& dst1, v_float16x16& dst2) noexcept
				: load_deinterleave_Invoker<v_uint16x16, 3>(
					reinterpret_cast<v_uint16x16&>(dst0), reinterpret_cast<v_uint16x16&>(dst1), reinterpret_cast<v_uint16x16&>(dst2))
			{
			}

			void operator () (const f16* src_scalar_ptr) noexcept
			{
				load_deinterleave_Invoker<v_uint16x16, 3>::operator()(reinterpret_cast<const u16*>(src_scalar_ptr));
			}
		};

		template<> struct load_deinterleave_Invoker<v_float16x16, 4> : public load_deinterleave_Invoker<v_uint16x16, 4>
		{
			load_deinterleave_Invoker(v_float16x16& dst0, v_float16x16& dst1, v_float16x16& dst2, v_float16x16& dst3) noexcept
				: load_deinterleave_Invoker<v_uint16x16, 4>(
					reinterpret_cast<v_uint16x16&>(dst0), reinterpret_cast<v_uint16x16&>(dst1),
					reinterpret_cast<v_uint16x16&>(dst2), reinterpret_cast<v_uint16x16&>(dst3))
			{
			}

			void operator () (const f16* src_scalar_ptr) noexcept
			{
				load_deinterleave_Invoker<v_uint16x16, 4>::operator()(reinterpret_cast<const u16*>(src_scalar_ptr));
			}
		};

		template<BasicArithmetic Scalar_t, VectorType ... Vec>
		void v_load_deinterleave(const Scalar_t* src_scalar_ptr, Vec& ... vsrc)
		{
			static_assert((... && std::is_same_v<Scalar_t, typename Vec::scalar_t>), "All vector scalar types must match the input scalar type");
			static_assert(sizeof...(Vec) > 0, "At least one vector must be provided");

			using vector_t = std::tuple_element_t<0, std::tuple<Vec...>>;

			static_assert(std::is_same_v<typename vector_t::scalar_t, Scalar_t>, "vector scalar types must be same");

			if constexpr (sizeof...(Vec) == 1 || sizeof...(Vec) == 3 || sizeof...(Vec) == 4)
			{
				load_deinterleave_Invoker<vector_t, sizeof...(Vec)> invoker(vsrc...);
				invoker(reinterpret_cast<const typename vector_t::scalar_t*>(src_scalar_ptr));
			}
			else
			{
				for (usize i = 0; i < vector_t::batch_size; ++i)
				{
					usize j = 0;
					(..., (vsrc[i] = src_scalar_ptr[i * (sizeof...(Vec)) + (j++)], void()));
				}
			}
		}
	}
}

export namespace fy
{
	namespace simd
	{
		template<BasicArithmetic Scalar_t, usize N> struct v_store_interleave_Invoker
		{
			using vector_t = AVX_t<Scalar_t>;
			Scalar_t* dst;
			alignas(32) Scalar_t temp_array[N][vector_t::batch_size];

			v_store_interleave_Invoker(Scalar_t* dst_) noexcept : dst(dst_) {}

			template<VectorType... Vectors> void operator () (const Vectors&... vectors) noexcept
			{
				//static_assert((!is_image_like_v<sizeof...(Vectors)> && sizeof...(Vectors) != 2) && sizeof...(Vectors) > 1);
				static_assert(sizeof...(Vectors) == N, "Number of vectors must match N");
				static_assert((std::is_same_v<Vectors, vector_t> && ...), "All vectors must be of the same type");

				usize i = 0;
				(vectors.download(temp_array[i++]), ...);

				for (usize j = 0; j < vector_t::batch_size; ++j)
				{
					for (usize k = 0; k < N; ++k)
					{
						dst[j * N + k] = temp_array[k][j];
					}
				}
			}
		};

		template<> struct v_store_interleave_Invoker<u8, 2>
		{
			using vector_t = v_uint8x32;
			u8* dst;

			v_store_interleave_Invoker(u8* dst_) noexcept : dst(dst_) {}
			void operator () (const vector_t& v0, const vector_t& v1) noexcept
			{
				__m256i xy_l = _mm256_unpacklo_epi8(v0.data, v1.data);
				__m256i xy_h = _mm256_unpackhi_epi8(v0.data, v1.data);

				__m256i xy0 = _mm256_permute2x128_si256(xy_l, xy_h, 0 + 2 * 16);
				__m256i xy1 = _mm256_permute2x128_si256(xy_l, xy_h, 1 + 3 * 16);

				_mm256_store_si256(reinterpret_cast<__m256i*>((dst + (vector_t::batch_size * 0))), xy0);
				_mm256_store_si256(reinterpret_cast<__m256i*>((dst + (vector_t::batch_size * 1))), xy1);
			}
		};

		template<> struct v_store_interleave_Invoker<u8, 3>
		{
			using vector_t = v_uint8x32;
			u8* dst;

			v_store_interleave_Invoker(u8* dst_) noexcept : dst(dst_) {}
			void operator () (const vector_t& v0, const vector_t& v1, const vector_t& v2) noexcept
			{
				static const __m256i sh_b = _mm256_setr_epi8(
					0, 11, 6, 1, 12, 7, 2, 13, 8, 3, 14, 9, 4, 15, 10, 5,
					0, 11, 6, 1, 12, 7, 2, 13, 8, 3, 14, 9, 4, 15, 10, 5);

				static const __m256i sh_g = _mm256_setr_epi8(
					5, 0, 11, 6, 1, 12, 7, 2, 13, 8, 3, 14, 9, 4, 15, 10,
					5, 0, 11, 6, 1, 12, 7, 2, 13, 8, 3, 14, 9, 4, 15, 10);

				static const __m256i sh_r = _mm256_setr_epi8(
					10, 5, 0, 11, 6, 1, 12, 7, 2, 13, 8, 3, 14, 9, 4, 15,
					10, 5, 0, 11, 6, 1, 12, 7, 2, 13, 8, 3, 14, 9, 4, 15);

				static const __m256i m0 = _mm256_setr_epi8(
					0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0,
					0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0);

				static const __m256i m1 = _mm256_setr_epi8(
					0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0,
					0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0);

				__m256i b0 = _mm256_shuffle_epi8(v0.data, sh_b);
				__m256i g0 = _mm256_shuffle_epi8(v1.data, sh_g);
				__m256i r0 = _mm256_shuffle_epi8(v2.data, sh_r);

				__m256i p0 = _mm256_blendv_epi8(_mm256_blendv_epi8(b0, g0, m0), r0, m1);
				__m256i p1 = _mm256_blendv_epi8(_mm256_blendv_epi8(g0, r0, m0), b0, m1);
				__m256i p2 = _mm256_blendv_epi8(_mm256_blendv_epi8(r0, b0, m0), g0, m1);

				__m256i bgr0 = _mm256_permute2x128_si256(p0, p1, 0 + 2 * 16);
				__m256i bgr1 = _mm256_permute2x128_si256(p2, p0, 0 + 3 * 16);
				__m256i bgr2 = _mm256_permute2x128_si256(p1, p2, 1 + 3 * 16);

				_mm256_store_si256(reinterpret_cast<__m256i*>((dst + (vector_t::batch_size * 0))), bgr0);
				_mm256_store_si256(reinterpret_cast<__m256i*>((dst + (vector_t::batch_size * 1))), bgr1);
				_mm256_store_si256(reinterpret_cast<__m256i*>((dst + (vector_t::batch_size * 2))), bgr2);
			}
		};

		template<> struct v_store_interleave_Invoker<u8, 4>
		{
			using vector_t = v_uint8x32;
			u8* dst;

			v_store_interleave_Invoker(u8* dst_) noexcept : dst(dst_) {}
			void operator () (const vector_t& v0, const vector_t& v1, const vector_t& v2, const vector_t& v3) noexcept
			{
				__m256i bg0 = _mm256_unpacklo_epi8(v0.data, v1.data);
				__m256i bg1 = _mm256_unpackhi_epi8(v0.data, v1.data);

				__m256i ra0 = _mm256_unpacklo_epi8(v2.data, v3.data);
				__m256i ra1 = _mm256_unpackhi_epi8(v2.data, v3.data);

				__m256i bgra0_ = _mm256_unpacklo_epi16(bg0, ra0);
				__m256i bgra1_ = _mm256_unpackhi_epi16(bg0, ra0);
				__m256i bgra2_ = _mm256_unpacklo_epi16(bg1, ra1);
				__m256i bgra3_ = _mm256_unpackhi_epi16(bg1, ra1);

				__m256i bgra0 = _mm256_permute2x128_si256(bgra0_, bgra1_, 0 + 2 * 16);
				__m256i bgra2 = _mm256_permute2x128_si256(bgra0_, bgra1_, 1 + 3 * 16);
				__m256i bgra1 = _mm256_permute2x128_si256(bgra2_, bgra3_, 0 + 2 * 16);
				__m256i bgra3 = _mm256_permute2x128_si256(bgra2_, bgra3_, 1 + 3 * 16);

				_mm256_store_si256(reinterpret_cast<__m256i*>((dst + (vector_t::batch_size * 0))), bgra0);
				_mm256_store_si256(reinterpret_cast<__m256i*>((dst + (vector_t::batch_size * 1))), bgra1);
				_mm256_store_si256(reinterpret_cast<__m256i*>((dst + (vector_t::batch_size * 2))), bgra2);
				_mm256_store_si256(reinterpret_cast<__m256i*>((dst + (vector_t::batch_size * 3))), bgra3);
			}
		};

		template<> struct v_store_interleave_Invoker<u16, 2>
		{
			using vector_t = v_uint16x16;
			u16* dst;

			v_store_interleave_Invoker(u16* dst_) noexcept : dst(dst_) {}
			void operator () (const vector_t& v0, const vector_t& v1) noexcept
			{
				__m256i xy_l = _mm256_unpacklo_epi16(v0.data, v1.data);
				__m256i xy_h = _mm256_unpackhi_epi16(v0.data, v1.data);

				__m256i xy0 = _mm256_permute2x128_si256(xy_l, xy_h, 0 + 2 * 16);
				__m256i xy1 = _mm256_permute2x128_si256(xy_l, xy_h, 1 + 3 * 16);

				_mm256_store_si256(reinterpret_cast<__m256i*>((dst + (vector_t::batch_size * 0))), xy0);
				_mm256_store_si256(reinterpret_cast<__m256i*>((dst + (vector_t::batch_size * 1))), xy1);
			}
		};

		template<> struct v_store_interleave_Invoker<u16, 3>
		{
			using vector_t = v_uint16x16;
			u16* dst;

			v_store_interleave_Invoker(u16* dst_) noexcept : dst(dst_) {}
			void operator () (const vector_t& v0, const vector_t& v1, const vector_t& v2) noexcept
			{
				static const __m256i sh_b = _mm256_setr_epi8(
					0, 1, 6, 7, 12, 13, 2, 3, 8, 9, 14, 15, 4, 5, 10, 11,
					0, 1, 6, 7, 12, 13, 2, 3, 8, 9, 14, 15, 4, 5, 10, 11);

				static const __m256i sh_g = _mm256_setr_epi8(
					10, 11, 0, 1, 6, 7, 12, 13, 2, 3, 8, 9, 14, 15, 4, 5,
					10, 11, 0, 1, 6, 7, 12, 13, 2, 3, 8, 9, 14, 15, 4, 5);

				static const __m256i sh_r = _mm256_setr_epi8(
					4, 5, 10, 11, 0, 1, 6, 7, 12, 13, 2, 3, 8, 9, 14, 15,
					4, 5, 10, 11, 0, 1, 6, 7, 12, 13, 2, 3, 8, 9, 14, 15);

				static const __m256i m0 = _mm256_setr_epi8(
					0, 0, -1, -1, 0, 0, 0, 0, -1, -1, 0, 0, 0, 0, -1, -1,
					0, 0, 0, 0, -1, -1, 0, 0, 0, 0, -1, -1, 0, 0, 0, 0);

				static const __m256i m1 = _mm256_setr_epi8(
					0, 0, 0, 0, -1, -1, 0, 0, 0, 0, -1, -1, 0, 0, 0, 0,
					-1, -1, 0, 0, 0, 0, -1, -1, 0, 0, 0, 0, -1, -1, 0, 0);

				__m256i b0 = _mm256_shuffle_epi8(v0.data, sh_b);
				__m256i g0 = _mm256_shuffle_epi8(v1.data, sh_g);
				__m256i r0 = _mm256_shuffle_epi8(v2.data, sh_r);

				__m256i p0 = _mm256_blendv_epi8(_mm256_blendv_epi8(b0, g0, m0), r0, m1);
				__m256i p1 = _mm256_blendv_epi8(_mm256_blendv_epi8(g0, r0, m0), b0, m1);
				__m256i p2 = _mm256_blendv_epi8(_mm256_blendv_epi8(r0, b0, m0), g0, m1);

				__m256i bgr0 = _mm256_permute2x128_si256(p0, p2, 0 + 2 * 16);
				__m256i bgr2 = _mm256_permute2x128_si256(p0, p2, 1 + 3 * 16);

				_mm256_store_si256(reinterpret_cast<__m256i*>((dst + (vector_t::batch_size * 0))), bgr0);
				_mm256_store_si256(reinterpret_cast<__m256i*>((dst + (vector_t::batch_size * 1))), p1);
				_mm256_store_si256(reinterpret_cast<__m256i*>((dst + (vector_t::batch_size * 2))), bgr2);
			}
		};

		template<> struct v_store_interleave_Invoker<u16, 4>
		{
			using vector_t = v_uint16x16;
			u16* dst;

			v_store_interleave_Invoker(u16* dst_) noexcept : dst(dst_) {}
			void operator () (const vector_t& v0, const vector_t& v1, const vector_t& v2, const vector_t& v3) noexcept
			{
				__m256i bg0 = _mm256_unpacklo_epi16(v0.data, v1.data);
				__m256i bg1 = _mm256_unpackhi_epi16(v0.data, v1.data);

				__m256i ra0 = _mm256_unpacklo_epi16(v2.data, v3.data);
				__m256i ra1 = _mm256_unpackhi_epi16(v2.data, v3.data);

				__m256i bgra0_ = _mm256_unpacklo_epi32(bg0, ra0);
				__m256i bgra1_ = _mm256_unpackhi_epi32(bg0, ra0);
				__m256i bgra2_ = _mm256_unpacklo_epi32(bg1, ra1);
				__m256i bgra3_ = _mm256_unpackhi_epi32(bg1, ra1);

				__m256i bgra0 = _mm256_permute2x128_si256(bgra0_, bgra1_, 0 + 2 * 16);
				__m256i bgra2 = _mm256_permute2x128_si256(bgra0_, bgra1_, 1 + 3 * 16);
				__m256i bgra1 = _mm256_permute2x128_si256(bgra2_, bgra3_, 0 + 2 * 16);
				__m256i bgra3 = _mm256_permute2x128_si256(bgra2_, bgra3_, 1 + 3 * 16);

				_mm256_store_si256(reinterpret_cast<__m256i*>((dst + (vector_t::batch_size * 0))), bgra0);
				_mm256_store_si256(reinterpret_cast<__m256i*>((dst + (vector_t::batch_size * 1))), bgra1);
				_mm256_store_si256(reinterpret_cast<__m256i*>((dst + (vector_t::batch_size * 2))), bgra2);
				_mm256_store_si256(reinterpret_cast<__m256i*>((dst + (vector_t::batch_size * 3))), bgra3);
			}
		};

		template<> struct v_store_interleave_Invoker<u32, 2>
		{
			using vector_t = v_uint32x8;
			u32* dst;

			v_store_interleave_Invoker(u32* dst_) noexcept : dst(dst_) {}
			void operator () (const vector_t& v0, const vector_t& v1) noexcept
			{
				__m256i xy_l = _mm256_unpacklo_epi32(v0.data, v1.data);
				__m256i xy_h = _mm256_unpackhi_epi32(v0.data, v1.data);

				__m256i xy0 = _mm256_permute2x128_si256(xy_l, xy_h, 0 + 2 * 16);
				__m256i xy1 = _mm256_permute2x128_si256(xy_l, xy_h, 1 + 3 * 16);

				_mm256_store_si256(reinterpret_cast<__m256i*>((dst + (vector_t::batch_size * 0))), xy0);
				_mm256_store_si256(reinterpret_cast<__m256i*>((dst + (vector_t::batch_size * 1))), xy1);
			}
		};

		template<> struct v_store_interleave_Invoker<u32, 3>
		{
			using vector_t = v_uint32x8;
			u32* dst;

			v_store_interleave_Invoker(u32* dst_) noexcept : dst(dst_) {}
			void operator () (const vector_t& v0, const vector_t& v1, const vector_t& v2) noexcept
			{
				__m256i b0 = _mm256_shuffle_epi32(v0.data, 0x6c);
				__m256i g0 = _mm256_shuffle_epi32(v1.data, 0xb1);
				__m256i r0 = _mm256_shuffle_epi32(v2.data, 0xc6);

				__m256i p0 = _mm256_blend_epi32(_mm256_blend_epi32(b0, g0, 0x92), r0, 0x24);
				__m256i p1 = _mm256_blend_epi32(_mm256_blend_epi32(g0, r0, 0x92), b0, 0x24);
				__m256i p2 = _mm256_blend_epi32(_mm256_blend_epi32(r0, b0, 0x92), g0, 0x24);

				__m256i bgr0 = _mm256_permute2x128_si256(p0, p1, 0 + 2 * 16);
				__m256i bgr2 = _mm256_permute2x128_si256(p0, p1, 1 + 3 * 16);

				_mm256_store_si256(reinterpret_cast<__m256i*>((dst + (vector_t::batch_size * 0))), bgr0);
				_mm256_store_si256(reinterpret_cast<__m256i*>((dst + (vector_t::batch_size * 1))), p2);
				_mm256_store_si256(reinterpret_cast<__m256i*>((dst + (vector_t::batch_size * 2))), bgr2);
			}
		};

		template<> struct v_store_interleave_Invoker<u32, 4>
		{
			using vector_t = v_uint32x8;
			u32* dst;

			v_store_interleave_Invoker(u32* dst_) noexcept : dst(dst_) {}
			void operator () (const vector_t& v0, const vector_t& v1, const vector_t& v2, const vector_t& v3) noexcept
			{
				__m256i bg0 = _mm256_unpacklo_epi32(v0.data, v1.data);
				__m256i bg1 = _mm256_unpackhi_epi32(v0.data, v1.data);

				__m256i ra0 = _mm256_unpacklo_epi32(v2.data, v3.data);
				__m256i ra1 = _mm256_unpackhi_epi32(v2.data, v3.data);

				__m256i bgra0_ = _mm256_unpacklo_epi64(bg0, ra0);
				__m256i bgra1_ = _mm256_unpackhi_epi64(bg0, ra0);
				__m256i bgra2_ = _mm256_unpacklo_epi64(bg1, ra1);
				__m256i bgra3_ = _mm256_unpackhi_epi64(bg1, ra1);

				__m256i bgra0 = _mm256_permute2x128_si256(bgra0_, bgra1_, 0 + 2 * 16);
				__m256i bgra2 = _mm256_permute2x128_si256(bgra0_, bgra1_, 1 + 3 * 16);
				__m256i bgra1 = _mm256_permute2x128_si256(bgra2_, bgra3_, 0 + 2 * 16);
				__m256i bgra3 = _mm256_permute2x128_si256(bgra2_, bgra3_, 1 + 3 * 16);

				_mm256_store_si256(reinterpret_cast<__m256i*>((dst + (vector_t::batch_size * 0))), bgra0);
				_mm256_store_si256(reinterpret_cast<__m256i*>((dst + (vector_t::batch_size * 1))), bgra1);
				_mm256_store_si256(reinterpret_cast<__m256i*>((dst + (vector_t::batch_size * 2))), bgra2);
				_mm256_store_si256(reinterpret_cast<__m256i*>((dst + (vector_t::batch_size * 3))), bgra3);
			}
		};

		template<> struct v_store_interleave_Invoker<u64, 2>
		{
			using vector_t = v_uint64x4;
			u64* dst;

			v_store_interleave_Invoker(u64* dst_) noexcept : dst(dst_) {}
			void operator () (const vector_t& v0, const vector_t& v1) noexcept
			{
				__m256i xy_l = _mm256_unpacklo_epi64(v0.data, v1.data);
				__m256i xy_h = _mm256_unpackhi_epi64(v0.data, v1.data);

				__m256i xy0 = _mm256_permute2x128_si256(xy_l, xy_h, 0 + 2 * 16);
				__m256i xy1 = _mm256_permute2x128_si256(xy_l, xy_h, 1 + 3 * 16);

				_mm256_store_si256(reinterpret_cast<__m256i*>((dst + (vector_t::batch_size * 0))), xy0);
				_mm256_store_si256(reinterpret_cast<__m256i*>((dst + (vector_t::batch_size * 1))), xy1);
			}
		};

		template<> struct v_store_interleave_Invoker<u64, 3>
		{
			using vector_t = v_uint64x4;
			u64* dst;

			v_store_interleave_Invoker(u64* dst_) noexcept : dst(dst_) {}
			void operator () (const vector_t& v0, const vector_t& v1, const vector_t& v2) noexcept
			{
				__m256i s01 = _mm256_unpacklo_epi64(v0.data, v1.data);
				__m256i s12 = _mm256_unpackhi_epi64(v1.data, v2.data);
				__m256i s20 = _mm256_blend_epi32(v2.data, v0.data, 0xcc);

				__m256i bgr0 = _mm256_permute2x128_si256(s01, s20, 0 + 2 * 16);
				__m256i bgr1 = _mm256_blend_epi32(s01, s12, 0x0f);
				__m256i bgr2 = _mm256_permute2x128_si256(s20, s12, 1 + 3 * 16);

				_mm256_store_si256(reinterpret_cast<__m256i*>((dst + (vector_t::batch_size * 0))), bgr0);
				_mm256_store_si256(reinterpret_cast<__m256i*>((dst + (vector_t::batch_size * 1))), bgr1);
				_mm256_store_si256(reinterpret_cast<__m256i*>((dst + (vector_t::batch_size * 2))), bgr2);
			}
		};

		template<> struct v_store_interleave_Invoker<u64, 4>
		{
			using vector_t = v_uint64x4;
			u64* dst;

			v_store_interleave_Invoker(u64* dst_) noexcept : dst(dst_) {}
			void operator () (const vector_t& v0, const vector_t& v1, const vector_t& v2, const vector_t& v3) noexcept
			{
				__m256i bg0 = _mm256_unpacklo_epi64(v0.data, v1.data);
				__m256i bg1 = _mm256_unpackhi_epi64(v0.data, v1.data);

				__m256i ra0 = _mm256_unpacklo_epi64(v2.data, v3.data);
				__m256i ra1 = _mm256_unpackhi_epi64(v2.data, v3.data);

				__m256i bgra0 = _mm256_permute2x128_si256(bg0, ra0, 0 + 2 * 16);
				__m256i bgra1 = _mm256_permute2x128_si256(bg1, ra1, 0 + 2 * 16);
				__m256i bgra2 = _mm256_permute2x128_si256(bg0, ra0, 1 + 3 * 16);
				__m256i bgra3 = _mm256_permute2x128_si256(bg1, ra1, 1 + 3 * 16);

				_mm256_store_si256(reinterpret_cast<__m256i*>((dst + (vector_t::batch_size * 0))), bgra0);
				_mm256_store_si256(reinterpret_cast<__m256i*>((dst + (vector_t::batch_size * 1))), bgra1);
				_mm256_store_si256(reinterpret_cast<__m256i*>((dst + (vector_t::batch_size * 2))), bgra2);
				_mm256_store_si256(reinterpret_cast<__m256i*>((dst + (vector_t::batch_size * 3))), bgra3);
			}
		};


		template<> struct v_store_interleave_Invoker<i8, 2> final : public v_store_interleave_Invoker<u8, 2>
		{
			using vector_t = v_int8x32;
			using base_vector_t = v_uint8x32;

			v_store_interleave_Invoker(i8* dst_) noexcept : v_store_interleave_Invoker<u8, 2>(reinterpret_cast<u8*>(dst_)) {}

			void operator () (const vector_t& v0, const vector_t& v1) noexcept
			{
				v_store_interleave_Invoker<u8, 2>::operator()(v_reinterpret_convert<base_vector_t>(v0), v_reinterpret_convert<base_vector_t>(v1));
			}
		};

		template<> struct v_store_interleave_Invoker<i8, 3> final : public v_store_interleave_Invoker<u8, 3>
		{
			using vector_t = v_int8x32;
			using base_vector_t = v_uint8x32;

			v_store_interleave_Invoker(i8* dst_) noexcept : v_store_interleave_Invoker<u8, 3>(reinterpret_cast<u8*>(dst_)) {}

			void operator () (const vector_t& v0, const vector_t& v1, const vector_t& v2) noexcept
			{
				v_store_interleave_Invoker<u8, 3>::operator()(v_reinterpret_convert<base_vector_t>(v0), v_reinterpret_convert<base_vector_t>(v1), v_reinterpret_convert<base_vector_t>(v2));
			}
		};

		template<> struct v_store_interleave_Invoker<i8, 4> final : public v_store_interleave_Invoker<u8, 4>
		{
			using vector_t = v_int8x32;
			using base_vector_t = v_uint8x32;

			v_store_interleave_Invoker(i8* dst_) noexcept : v_store_interleave_Invoker<u8, 4>(reinterpret_cast<u8*>(dst_)) {}

			void operator () (const vector_t& v0, const vector_t& v1, const vector_t& v2, const vector_t& v3) noexcept
			{
				v_store_interleave_Invoker<u8, 4>::operator()(
					v_reinterpret_convert<base_vector_t>(v0), v_reinterpret_convert<base_vector_t>(v1),
					v_reinterpret_convert<base_vector_t>(v2), v_reinterpret_convert<base_vector_t>(v3));
			}
		};

		template<> struct v_store_interleave_Invoker<i16, 2> final : public v_store_interleave_Invoker<u16, 2>
		{
			using vector_t = v_int16x16;
			using base_vector_t = v_uint16x16;

			v_store_interleave_Invoker(i16* dst_) noexcept : v_store_interleave_Invoker<u16, 2>(reinterpret_cast<u16*>(dst_)) {}

			void operator () (const vector_t& v0, const vector_t& v1) noexcept
			{
				v_store_interleave_Invoker<u16, 2>::operator()(v_reinterpret_convert<base_vector_t>(v0), v_reinterpret_convert<base_vector_t>(v1));
			}
		};

		template<> struct v_store_interleave_Invoker<i16, 3> final : public v_store_interleave_Invoker<u16, 3>
		{
			using vector_t = v_int16x16;
			using base_vector_t = v_uint16x16;

			v_store_interleave_Invoker(i16* dst_) noexcept : v_store_interleave_Invoker<u16, 3>(reinterpret_cast<u16*>(dst_)) {}

			void operator () (const vector_t& v0, const vector_t& v1, const vector_t& v2) noexcept
			{
				v_store_interleave_Invoker<u16, 3>::operator()(v_reinterpret_convert<base_vector_t>(v0), v_reinterpret_convert<base_vector_t>(v1), v_reinterpret_convert<base_vector_t>(v2));
			}
		};

		template<> struct v_store_interleave_Invoker<i16, 4> final : public v_store_interleave_Invoker<u16, 4>
		{
			using vector_t = v_int16x16;
			using base_vector_t = v_uint16x16;

			v_store_interleave_Invoker(i16* dst_) noexcept : v_store_interleave_Invoker<u16, 4>(reinterpret_cast<u16*>(dst_)) {}

			void operator () (const vector_t& v0, const vector_t& v1, const vector_t& v2, const vector_t& v3) noexcept
			{
				v_store_interleave_Invoker<u16, 4>::operator()(
					v_reinterpret_convert<base_vector_t>(v0), v_reinterpret_convert<base_vector_t>(v1),
					v_reinterpret_convert<base_vector_t>(v2), v_reinterpret_convert<base_vector_t>(v3));
			}
		};

		template<> struct v_store_interleave_Invoker<i32, 2> final : public v_store_interleave_Invoker<u32, 2>
		{
			using vector_t = v_int32x8;
			using base_vector_t = v_uint32x8;

			v_store_interleave_Invoker(i32* dst_) noexcept : v_store_interleave_Invoker<u32, 2>(reinterpret_cast<u32*>(dst_)) {}

			void operator () (const vector_t& v0, const vector_t& v1) noexcept
			{
				v_store_interleave_Invoker<u32, 2>::operator()(v_reinterpret_convert<base_vector_t>(v0), v_reinterpret_convert<base_vector_t>(v1));
			}
		};

		template<> struct v_store_interleave_Invoker<i32, 3> final : public v_store_interleave_Invoker<u32, 3>
		{
			using vector_t = v_int32x8;
			using base_vector_t = v_uint32x8;

			v_store_interleave_Invoker(i32* dst_) noexcept : v_store_interleave_Invoker<u32, 3>(reinterpret_cast<u32*>(dst_)) {}

			void operator () (const vector_t& v0, const vector_t& v1, const vector_t& v2) noexcept
			{
				v_store_interleave_Invoker<u32, 3>::operator()(v_reinterpret_convert<base_vector_t>(v0), v_reinterpret_convert<base_vector_t>(v1), v_reinterpret_convert<base_vector_t>(v2));
			}
		};

		template<> struct v_store_interleave_Invoker<i32, 4> final : public v_store_interleave_Invoker<u32, 4>
		{
			using vector_t = v_int32x8;
			using base_vector_t = v_uint32x8;

			v_store_interleave_Invoker(i32* dst_) noexcept : v_store_interleave_Invoker<u32, 4>(reinterpret_cast<u32*>(dst_)) {}

			void operator () (const vector_t& v0, const vector_t& v1, const vector_t& v2, const vector_t& v3) noexcept
			{
				v_store_interleave_Invoker<u32, 4>::operator()(
					v_reinterpret_convert<base_vector_t>(v0), v_reinterpret_convert<base_vector_t>(v1),
					v_reinterpret_convert<base_vector_t>(v2), v_reinterpret_convert<base_vector_t>(v3));
			}
		};

		template<> struct v_store_interleave_Invoker<i64, 2> final : public v_store_interleave_Invoker<u64, 2>
		{
			using vector_t = v_int64x4;
			using base_vector_t = v_uint64x4;

			v_store_interleave_Invoker(i64* dst_) noexcept : v_store_interleave_Invoker<u64, 2>(reinterpret_cast<u64*>(dst_)) {}

			void operator () (const vector_t& v0, const vector_t& v1) noexcept
			{
				v_store_interleave_Invoker<u64, 2>::operator()(v_reinterpret_convert<base_vector_t>(v0), v_reinterpret_convert<base_vector_t>(v1));
			}
		};

		template<> struct v_store_interleave_Invoker<i64, 3> final : public v_store_interleave_Invoker<u64, 3>
		{
			using vector_t = v_int64x4;
			using base_vector_t = v_uint64x4;

			v_store_interleave_Invoker(i64* dst_) noexcept : v_store_interleave_Invoker<u64, 3>(reinterpret_cast<u64*>(dst_)) {}

			void operator () (const vector_t& v0, const vector_t& v1, const vector_t& v2) noexcept
			{
				v_store_interleave_Invoker<u64, 3>::operator()(v_reinterpret_convert<base_vector_t>(v0), v_reinterpret_convert<base_vector_t>(v1), v_reinterpret_convert<base_vector_t>(v2));
			}
		};

		template<> struct v_store_interleave_Invoker<i64, 4> final : public v_store_interleave_Invoker<u64, 4>
		{
			using vector_t = v_int64x4;
			using base_vector_t = v_uint64x4;

			v_store_interleave_Invoker(i64* dst_) noexcept : v_store_interleave_Invoker<u64, 4>(reinterpret_cast<u64*>(dst_)) {}

			void operator () (const vector_t& v0, const vector_t& v1, const vector_t& v2, const vector_t& v3) noexcept
			{
				v_store_interleave_Invoker<u64, 4>::operator()(
					v_reinterpret_convert<base_vector_t>(v0), v_reinterpret_convert<base_vector_t>(v1),
					v_reinterpret_convert<base_vector_t>(v2), v_reinterpret_convert<base_vector_t>(v3));
			}
		};


		template<> struct v_store_interleave_Invoker<f16, 2> final : public v_store_interleave_Invoker<u16, 2>
		{
			using vector_t = v_float16x16;
			using base_vector_t = v_uint16x16;

			v_store_interleave_Invoker(f16* dst_) noexcept : v_store_interleave_Invoker<u16, 2>(reinterpret_cast<u16*>(dst_)) {}

			void operator () (const vector_t& v0, const vector_t& v1) noexcept
			{
				v_store_interleave_Invoker<u16, 2>::operator()(v_reinterpret_convert<base_vector_t>(v0), v_reinterpret_convert<base_vector_t>(v1));
			}
		};

		template<> struct v_store_interleave_Invoker<f16, 3> final : public v_store_interleave_Invoker<u16, 3>
		{
			using vector_t = v_float16x16;
			using base_vector_t = v_uint16x16;

			v_store_interleave_Invoker(f16* dst_) noexcept : v_store_interleave_Invoker<u16, 3>(reinterpret_cast<u16*>(dst_)) {}

			void operator () (const vector_t& v0, const vector_t& v1, const vector_t& v2) noexcept
			{
				v_store_interleave_Invoker<u16, 3>::operator()(v_reinterpret_convert<base_vector_t>(v0), v_reinterpret_convert<base_vector_t>(v1), v_reinterpret_convert<base_vector_t>(v2));
			}
		};

		template<> struct v_store_interleave_Invoker<f16, 4> final : public v_store_interleave_Invoker<u16, 4>
		{
			using vector_t = v_float16x16;
			using base_vector_t = v_uint16x16;

			v_store_interleave_Invoker(f16* dst_) noexcept : v_store_interleave_Invoker<u16, 4>(reinterpret_cast<u16*>(dst_)) {}

			void operator () (const vector_t& v0, const vector_t& v1, const vector_t& v2, const vector_t& v3) noexcept
			{
				v_store_interleave_Invoker<u16, 4>::operator()(
					v_reinterpret_convert<base_vector_t>(v0), v_reinterpret_convert<base_vector_t>(v1),
					v_reinterpret_convert<base_vector_t>(v2), v_reinterpret_convert<base_vector_t>(v3));
			}
		};

		template<> struct v_store_interleave_Invoker<f32, 2> final : public v_store_interleave_Invoker<u32, 2>
		{
			using vector_t = v_float32x8;
			using base_vector_t = v_uint32x8;

			v_store_interleave_Invoker(f32* dst_) noexcept : v_store_interleave_Invoker<u32, 2>(reinterpret_cast<u32*>(dst_)) {}

			void operator () (const vector_t& v0, const vector_t& v1) noexcept
			{
				v_store_interleave_Invoker<u32, 2>::operator()(v_reinterpret_convert<base_vector_t>(v0), v_reinterpret_convert<base_vector_t>(v1));
			}
		};

		template<> struct v_store_interleave_Invoker<f32, 3> final : public v_store_interleave_Invoker<u32, 3>
		{
			using vector_t = v_float32x8;
			using base_vector_t = v_uint32x8;

			v_store_interleave_Invoker(f32* dst_) noexcept : v_store_interleave_Invoker<u32, 3>(reinterpret_cast<u32*>(dst_)) {}

			void operator () (const vector_t& v0, const vector_t& v1, const vector_t& v2) noexcept
			{
				v_store_interleave_Invoker<u32, 3>::operator()(v_reinterpret_convert<base_vector_t>(v0), v_reinterpret_convert<base_vector_t>(v1), v_reinterpret_convert<base_vector_t>(v2));
			}
		};

		template<> struct v_store_interleave_Invoker<f32, 4> final : public v_store_interleave_Invoker<u32, 4>
		{
			using vector_t = v_float32x8;
			using base_vector_t = v_uint32x8;

			v_store_interleave_Invoker(f32* dst_) noexcept : v_store_interleave_Invoker<u32, 4>(reinterpret_cast<u32*>(dst_)) {}

			void operator () (const vector_t& v0, const vector_t& v1, const vector_t& v2, const vector_t& v3) noexcept
			{
				v_store_interleave_Invoker<u32, 4>::operator()(
					v_reinterpret_convert<base_vector_t>(v0), v_reinterpret_convert<base_vector_t>(v1),
					v_reinterpret_convert<base_vector_t>(v2), v_reinterpret_convert<base_vector_t>(v3));
			}
		};

		template<> struct v_store_interleave_Invoker<f64, 2> final : public v_store_interleave_Invoker<u64, 2>
		{
			using vector_t = v_float64x4;
			using base_vector_t = v_uint64x4;

			v_store_interleave_Invoker(f64* dst_) noexcept : v_store_interleave_Invoker<u64, 2>(reinterpret_cast<u64*>(dst_)) {}

			void operator () (const vector_t& v0, const vector_t& v1) noexcept
			{
				v_store_interleave_Invoker<u64, 2>::operator()(v_reinterpret_convert<base_vector_t>(v0), v_reinterpret_convert<base_vector_t>(v1));
			}
		};

		template<> struct v_store_interleave_Invoker<f64, 3> final : public v_store_interleave_Invoker<u64, 3>
		{
			using vector_t = v_float64x4;
			using base_vector_t = v_uint64x4;

			v_store_interleave_Invoker(f64* dst_) noexcept : v_store_interleave_Invoker<u64, 3>(reinterpret_cast<u64*>(dst_)) {}

			void operator () (const vector_t& v0, const vector_t& v1, const vector_t& v2) noexcept
			{
				v_store_interleave_Invoker<u64, 3>::operator()(v_reinterpret_convert<base_vector_t>(v0), v_reinterpret_convert<base_vector_t>(v1), v_reinterpret_convert<base_vector_t>(v2));
			}
		};

		template<> struct v_store_interleave_Invoker<f64, 4> final : public v_store_interleave_Invoker<u64, 4>
		{
			using vector_t = v_float64x4;
			using base_vector_t = v_uint64x4;

			v_store_interleave_Invoker(f64* dst_) noexcept : v_store_interleave_Invoker<u64, 4>(reinterpret_cast<u64*>(dst_)) {}

			void operator () (const vector_t& v0, const vector_t& v1, const vector_t& v2, const vector_t& v3) noexcept
			{
				v_store_interleave_Invoker<u64, 4>::operator()(
					v_reinterpret_convert<base_vector_t>(v0), v_reinterpret_convert<base_vector_t>(v1),
					v_reinterpret_convert<base_vector_t>(v2), v_reinterpret_convert<base_vector_t>(v3));
			}
		};

		template<BasicArithmetic Scalar_t, VectorType ... Vec>
		void v_store_interleave(Scalar_t* dst_ptr, const Vec& ... vsrc)
		{
			static_assert((... && std::is_same_v<Scalar_t, typename Vec::scalar_t>), "All vector scalar types must match the input scalar type");
			static_assert(sizeof...(Vec) >= 2, "At least two vector must be provided");
			static_assert(std::is_same_v<typename std::tuple_element_t<0, std::tuple<Vec...>>::scalar_t, Scalar_t>, "vector scalar types must be same");
			static_assert(std::tuple_element_t<0, std::tuple<Vec...>>::batch_size >= sizeof...(Vec), "Maximun dims can not gt than num of input vector");

			v_store_interleave_Invoker<Scalar_t, sizeof...(Vec)> invoker(dst_ptr);
			invoker(vsrc ...);
		}
	}
}


export namespace fy
{
	namespace simd
	{
		template<VectorType Src_t, VectorType Dst_t> Dst_t v_pack(const Src_t& src1, const Src_t& src2)
		{
			static_assert(Dst_t::batch_size == Src_t::batch_size * 2);
			static_assert(std::is_integral_v<typename Src_t::scalar_t> && std::is_integral_v<typename Dst_t::scalar_t>);
			return Dst_t(_mm256_inserti128_si256(_mm256_castsi128_si256(src1.data), src2.data, 1));
		}

		template<> v_float16x16 v_pack(const v_float16x8& src1, const v_float16x8& src2)
		{
			return v_float16x16(_mm256_inserti128_si256(_mm256_castsi128_si256(src1.data), src2.data, 1));
		}

		template<> v_float32x8 v_pack(const v_float32x4& src1, const v_float32x4& src2)
		{
			return v_float32x8(_mm256_insertf128_ps(_mm256_castps128_ps256(src1.data), src2.data, 1));
		}

		template<> v_float64x4 v_pack(const v_float64x2& src1, const v_float64x2& src2)
		{
			return v_float64x4(_mm256_insertf128_pd(_mm256_castpd128_pd256(src1.data), src2.data, 1));
		}
	}
}

export namespace fy
{
	namespace simd
	{
		template<VectorType Src_t, VectorType Dst_t> void v_unpack(const Src_t& src, Dst_t& dst1, Dst_t& dst2)
		{
			static_assert(std::is_integral_v<typename Src_t::scalar_t> && std::is_integral_v<typename Dst_t::scalar_t>);
			static_assert(sizeof(Src_t) == sizeof(Dst_t) * 2);

			dst1.data = _mm256_extracti128_si256(src.data, 0);
			dst2.data = _mm256_extracti128_si256(src.data, 1);
		}

		template<> void v_unpack(const v_float16x16& src, v_float16x8& dst1, v_float16x8& dst2)
		{
			dst1.data = _mm256_extracti128_si256(src.data, 0);
			dst2.data = _mm256_extracti128_si256(src.data, 1);
		}
		template<> void v_unpack(const v_float32x8& src, v_float32x4& dst1, v_float32x4& dst2)
		{
			dst1.data = _mm256_extractf128_ps(src.data, 0);
			dst2.data = _mm256_extractf128_ps(src.data, 1);
		}
		template<> void v_unpack(const v_float64x4& src, v_float64x2& dst1, v_float64x2& dst2)
		{
			dst1.data = _mm256_extractf128_pd(src.data, 0);
			dst2.data = _mm256_extractf128_pd(src.data, 1);
		}
	}
}

export namespace fy
{
	namespace simd
	{
		template<VectorType Src_t, VectorType Dst_t> void v_expand(const Src_t& src, Dst_t& dst1, Dst_t& dst2) { std::unreachable(); }

		template<> void v_expand(const v_uint8x16& a, v_uint16x8& dst1, v_uint16x8& dst2)
		{
			__m256i expanded = _mm256_cvtepu8_epi16(a.data);
			dst1.data = _mm_load_si128(reinterpret_cast<const __m128i*>(&expanded));
			dst2.data = _mm_load_si128(reinterpret_cast<const __m128i*>(&expanded) + 1);
		}

		template<> void v_expand(const v_uint16x8& a, v_uint32x4& dst1, v_uint32x4& dst2)
		{
			__m256i expanded = _mm256_cvtepu16_epi32(a.data);
			dst1.data = _mm_load_si128(reinterpret_cast<const __m128i*>(&expanded));
			dst2.data = _mm_load_si128(reinterpret_cast<const __m128i*>(&expanded) + 1);
		}

		template<> void v_expand(const v_uint32x4& a, v_uint64x2& dst1, v_uint64x2& dst2)
		{
			__m256i expanded = _mm256_cvtepu32_epi64(a.data);
			dst1.data = _mm_load_si128(reinterpret_cast<const __m128i*>(&expanded));
			dst2.data = _mm_load_si128(reinterpret_cast<const __m128i*>(&expanded) + 1);
		}

		template<> void v_expand(const v_int8x16& a, v_int16x8& dst1, v_int16x8& dst2)
		{
			__m256i expanded = _mm256_cvtepi8_epi16(a.data);
			dst1.data = _mm_load_si128(reinterpret_cast<const __m128i*>(&expanded));
			dst2.data = _mm_load_si128(reinterpret_cast<const __m128i*>(&expanded) + 1);
		}

		template<> void v_expand(const v_int16x8& a, v_int32x4& dst1, v_int32x4& dst2)
		{
			__m256i expanded = _mm256_cvtepi16_epi32(a.data);
			dst1.data = _mm_load_si128(reinterpret_cast<const __m128i*>(&expanded));
			dst2.data = _mm_load_si128(reinterpret_cast<const __m128i*>(&expanded) + 1);
		}

		template<> void v_expand(const v_int32x4& a, v_int64x2& dst1, v_int64x2& dst2)
		{
			__m256i expanded = _mm256_cvtepi32_epi64(a.data);
			dst1.data = _mm_load_si128(reinterpret_cast<const __m128i*>(&expanded));
			dst2.data = _mm_load_si128(reinterpret_cast<const __m128i*>(&expanded) + 1);
		}

		template<> void v_expand(const v_float16x8& a, v_float32x4& dst1, v_float32x4& dst2)
		{
			__m256 expanded = _mm256_cvtph_ps(a.data);
			dst1.data = _mm256_extractf128_ps(expanded, 0);
			dst2.data = _mm256_extractf128_ps(expanded, 1);
		}

		template<> void v_expand(const v_float32x4& a, v_float64x2& dst1, v_float64x2& dst2)
		{
			dst1.data = _mm_cvtps_pd(a.data);
			dst2.data = _mm_cvtps_pd(_mm_movehl_ps(a.data, a.data));
		}

		template<> void v_expand(const v_uint8x32& a, v_uint16x16& dst1, v_uint16x16& dst2)
		{
			dst1.data = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(a.data, 0));
			dst2.data = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(a.data, 1));
		}

		template<> void v_expand(const v_uint16x16& a, v_uint32x8& dst1, v_uint32x8& dst2)
		{
			dst1.data = _mm256_cvtepu16_epi32(_mm256_extracti128_si256(a.data, 0));
			dst2.data = _mm256_cvtepu16_epi32(_mm256_extracti128_si256(a.data, 1));
		}

		template<> void v_expand(const v_uint32x8& a, v_uint64x4& dst1, v_uint64x4& dst2)
		{
			dst1.data = _mm256_cvtepu32_epi64(_mm256_extracti128_si256(a.data, 0));
			dst2.data = _mm256_cvtepu32_epi64(_mm256_extracti128_si256(a.data, 1));
		}

		template<> void v_expand(const v_int8x32& a, v_int16x16& dst1, v_int16x16& dst2)
		{
			dst1.data = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(a.data, 0));
			dst2.data = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(a.data, 1));
		}

		template<> void v_expand(const v_int16x16& a, v_int32x8& dst1, v_int32x8& dst2)
		{
			dst1.data = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(a.data, 0));
			dst2.data = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(a.data, 1));
		}

		template<> void v_expand(const v_int32x8& a, v_int64x4& dst1, v_int64x4& dst2)
		{
			dst1.data = _mm256_cvtepi32_epi64(_mm256_extracti128_si256(a.data, 0));
			dst2.data = _mm256_cvtepi32_epi64(_mm256_extracti128_si256(a.data, 1));
		}

		template<> void v_expand(const v_float16x16& a, v_float32x8& dst1, v_float32x8& dst2)
		{
			dst1.data = _mm256_cvtph_ps(_mm256_extractf128_si256(a.data, 0));
			dst2.data = _mm256_cvtph_ps(_mm256_extractf128_si256(a.data, 1));
		}

		template<> void v_expand(const v_float32x8& a, v_float64x4& dst1, v_float64x4& dst2)
		{
			dst1.data = _mm256_cvtps_pd(_mm256_extractf128_ps(a.data, 0));
			dst2.data = _mm256_cvtps_pd(_mm256_extractf128_ps(a.data, 1));
		}
	}
}


export namespace fy
{
	namespace simd
	{
		template<Floating_VectorType T>
		T v_close(const T& a, const T& b, const T& vtolerance)
		{
			std::unreachable();
		}

		template<> v_float16x8 v_close(const v_float16x8& a, const v_float16x8& b, const v_float16x8& vtolerance)
		{
			v_float32x8 temp_a = v_convert<v_float32x8>(a);
			v_float32x8 temp_b = v_convert<v_float32x8>(b);
			v_float32x8 temp_vtolerance = v_convert<v_float32x8>(vtolerance);

			static __m256 mask = _mm256_set1_ps(-0.0f);
			__m256 diff = _mm256_sub_ps(temp_a.data, temp_b.data);
			__m256 abs_diff = _mm256_andnot_ps(mask, diff);
			__m256 cmp_result = _mm256_cmp_ps(abs_diff, temp_vtolerance.data, _CMP_LE_OQ);

			return v_convert<v_float16x8>(v_float32x8(cmp_result));
		}

		template<> v_float32x4 v_close(const v_float32x4& a, const v_float32x4& b, const v_float32x4& vtolerance)
		{
			static __m128 mask = _mm_set1_ps(-0.0f);
			__m128 diff = _mm_sub_ps(a.data, b.data);
			__m128 abs_diff = _mm_andnot_ps(mask, diff);
			__m128 cmp_result = _mm_cmp_ps(abs_diff, vtolerance.data, _CMP_LE_OQ);
			return v_float32x4(cmp_result);
		}

		template<> v_float64x2 v_close(const v_float64x2& a, const v_float64x2& b, const v_float64x2& vtolerance)
		{
			static __m128d mask = _mm_set1_pd(-0.0f);
			__m128d diff = _mm_sub_pd(a.data, b.data);
			__m128d abs_diff = _mm_andnot_pd(mask, diff);
			__m128d cmp_result = _mm_cmp_pd(abs_diff, vtolerance.data, _CMP_LE_OQ);
			return v_float64x2(cmp_result);
		}

		template<> v_float16x16 v_close(const v_float16x16& a, const v_float16x16& b, const v_float16x16& vtolerance)
		{
			v_float32x8 temp_a[2];
			v_float32x8 temp_b[2];

			v_float16x8 res[2];

			v_expand(a, temp_a[0], temp_a[1]);
			v_expand(b, temp_b[0], temp_b[1]);

			v_float32x8 temp_vtolerance(vtolerance[0]);

			for (usize i = 0; i < 2; ++i)
			{
				static __m256 mask = _mm256_set1_ps(-0.0f);
				__m256 diff = _mm256_sub_ps(temp_a[i].data, temp_b[i].data);
				__m256 abs_diff = _mm256_andnot_ps(mask, diff);
				__m256 cmp_result = _mm256_cmp_ps(abs_diff, temp_vtolerance.data, _CMP_LE_OQ);

				res[i] = v_convert<v_float16x8>(v_float32x8(cmp_result));
			}

			return v_pack<v_float16x8, v_float16x16>(res[0], res[1]);
		}


		template<> v_float32x8 v_close(const v_float32x8& a, const v_float32x8& b, const v_float32x8& vtolerance)
		{
			static __m256 mask = _mm256_set1_ps(-0.0f);
			__m256 diff = _mm256_sub_ps(a.data, b.data);
			__m256 abs_diff = _mm256_andnot_ps(mask, diff);
			__m256 cmp_result = _mm256_cmp_ps(abs_diff, vtolerance.data, _CMP_LE_OQ);
			return v_float32x8(cmp_result);
		}

		template<> v_float64x4 v_close(const v_float64x4& a, const v_float64x4& b, const v_float64x4& vtolerance)
		{
			static __m256d mask = _mm256_set1_pd(-0.0f);
			__m256d diff = _mm256_sub_pd(a.data, b.data);
			__m256d abs_diff = _mm256_andnot_pd(mask, diff);
			__m256d cmp_result = _mm256_cmp_pd(abs_diff, vtolerance.data, _CMP_LE_OQ);
			return v_float64x4(cmp_result);
		}
	}
}

export namespace fy
{
	namespace simd
	{
		template<VectorType Src_t> Src_t v_min_replace(const Src_t&, const Src_t&) { std::unreachable(); }

		template<> v_uint8x16 v_min_replace<v_uint8x16>(const v_uint8x16& left, const v_uint8x16& right) { return v_uint8x16(_mm_min_epu8(left.data, right.data)); }
		template<> v_uint16x8 v_min_replace<v_uint16x8>(const v_uint16x8& left, const v_uint16x8& right) { return v_uint16x8(_mm_min_epu16(left.data, right.data)); }
		template<> v_uint32x4 v_min_replace<v_uint32x4>(const v_uint32x4& left, const v_uint32x4& right) { return v_uint32x4(_mm_min_epu32(left.data, right.data)); }

		template<> v_int8x16 v_min_replace<v_int8x16>(const v_int8x16& left, const v_int8x16& right) { return v_int8x16(_mm_min_epi8(left.data, right.data)); }
		template<> v_int16x8 v_min_replace<v_int16x8>(const v_int16x8& left, const v_int16x8& right) { return v_int16x8(_mm_min_epi16(left.data, right.data)); }
		template<> v_int32x4 v_min_replace<v_int32x4>(const v_int32x4& left, const v_int32x4& right) { return v_int32x4(_mm_min_epi32(left.data, right.data)); }

		template<> v_uint8x32 v_min_replace<v_uint8x32>(const v_uint8x32& left, const v_uint8x32& right) { return v_uint8x32(_mm256_min_epu8(left.data, right.data)); }
		template<> v_uint16x16 v_min_replace<v_uint16x16>(const v_uint16x16& left, const v_uint16x16& right) { return v_uint16x16(_mm256_min_epu16(left.data, right.data)); }
		template<> v_uint32x8 v_min_replace<v_uint32x8>(const v_uint32x8& left, const v_uint32x8& right) { return v_uint32x8(_mm256_min_epu32(left.data, right.data)); }

		template<> v_int8x32 v_min_replace<v_int8x32>(const v_int8x32& left, const v_int8x32& right) { return v_int8x32(_mm256_min_epi8(left.data, right.data)); }
		template<> v_int16x16 v_min_replace<v_int16x16>(const v_int16x16& left, const v_int16x16& right) { return v_int16x16(_mm256_min_epi16(left.data, right.data)); }
		template<> v_int32x8 v_min_replace<v_int32x8>(const v_int32x8& left, const v_int32x8& right) { return v_int32x8(_mm256_min_epi32(left.data, right.data)); }

		template<> v_float32x4 v_min_replace<v_float32x4>(const v_float32x4& left, const v_float32x4& right) { return v_float32x4(_mm_min_ps(left.data, right.data)); }
		template<> v_float32x8 v_min_replace<v_float32x8>(const v_float32x8& left, const v_float32x8& right) { return v_float32x8(_mm256_min_ps(left.data, right.data)); }

		template<> v_float64x2 v_min_replace<v_float64x2>(const v_float64x2& left, const v_float64x2& right) { return v_float64x2(_mm_min_pd(left.data, right.data)); }
		template<> v_float64x4 v_min_replace<v_float64x4>(const v_float64x4& left, const v_float64x4& right) { return v_float64x4(_mm256_min_pd(left.data, right.data)); }

		template<> v_int64x2 v_min_replace<v_int64x2>(const v_int64x2& left, const v_int64x2& right)
		{
			return v_int64x2(
				std::min(left.data.m128i_i64[0], right.data.m128i_i64[0]),
				std::min(left.data.m128i_i64[1], right.data.m128i_i64[1])
			);
		}

		template<> v_int64x4 v_min_replace<v_int64x4>(const v_int64x4& left, const v_int64x4& right)
		{
			return v_int64x4(
				std::min(left.data.m256i_i64[0], right.data.m256i_i64[0]),
				std::min(left.data.m256i_i64[1], right.data.m256i_i64[1]),
				std::min(left.data.m256i_i64[2], right.data.m256i_i64[2]),
				std::min(left.data.m256i_i64[3], right.data.m256i_i64[3])
			);
		}

		template<> v_uint64x2 v_min_replace<v_uint64x2>(const v_uint64x2& left, const v_uint64x2& right)
		{
			return v_uint64x2(
				std::min(left.data.m128i_u64[0], right.data.m128i_u64[0]),
				std::min(left.data.m128i_u64[1], right.data.m128i_u64[1])
			);
		}

		template<> v_uint64x4 v_min_replace<v_uint64x4>(const v_uint64x4& left, const v_uint64x4& right)
		{
			return v_uint64x4(
				std::min(left.data.m256i_u64[0], right.data.m256i_u64[0]),
				std::min(left.data.m256i_u64[1], right.data.m256i_u64[1]),
				std::min(left.data.m256i_u64[2], right.data.m256i_u64[2]),
				std::min(left.data.m256i_u64[3], right.data.m256i_u64[3])
			);
		}

		template<> v_float16x8 v_min_replace<v_float16x8>(const v_float16x8& left, const v_float16x8& right)
		{
			v_float32x8 vsrc_left = v_convert<v_float32x8>(left);
			v_float32x8 vsrc_right = v_convert<v_float32x8>(right);
			v_float32x8 vres = v_min_replace<v_float32x8>(vsrc_left, vsrc_right);
			return v_convert<v_float16x8>(vres);
		}

		template<> v_float16x16 v_min_replace<v_float16x16>(const v_float16x16& left, const v_float16x16& right)
		{
			v_float32x8 left_0, left_1, right_0, right_1;

			left_0.data = _mm256_cvtph_ps(_mm256_extractf128_si256(left.data, 0));
			left_1.data = _mm256_cvtph_ps(_mm256_extractf128_si256(left.data, 1));

			right_0.data = _mm256_cvtph_ps(_mm256_extractf128_si256(right.data, 0));
			right_1.data = _mm256_cvtph_ps(_mm256_extractf128_si256(right.data, 1));

			v_float32x8 vres_0 = v_min_replace<v_float32x8>(left_0, right_0);
			v_float32x8 vres_1 = v_min_replace<v_float32x8>(left_1, right_1);

			v_float16x8 vsrc_left = v_convert<v_float16x8>(vres_0);
			v_float16x8 vsrc_right = v_convert<v_float16x8>(vres_1);

			return v_float16x16(_mm256_inserti128_si256(_mm256_castsi128_si256(vsrc_left.data), vsrc_right.data, 1));
		}

	}
}

export namespace fy
{
	namespace simd
	{
		template<VectorType Src_t> Src_t v_max_replace(const Src_t&, const Src_t&) { std::unreachable(); }

		template<> v_uint8x16 v_max_replace<v_uint8x16>(const v_uint8x16& left, const v_uint8x16& right) { return v_uint8x16(_mm_max_epu8(left.data, right.data)); }
		template<> v_uint16x8 v_max_replace<v_uint16x8>(const v_uint16x8& left, const v_uint16x8& right) { return v_uint16x8(_mm_max_epu16(left.data, right.data)); }
		template<> v_uint32x4 v_max_replace<v_uint32x4>(const v_uint32x4& left, const v_uint32x4& right) { return v_uint32x4(_mm_max_epu32(left.data, right.data)); }

		template<> v_int8x16 v_max_replace<v_int8x16>(const v_int8x16& left, const v_int8x16& right) { return v_int8x16(_mm_max_epi8(left.data, right.data)); }
		template<> v_int16x8 v_max_replace<v_int16x8>(const v_int16x8& left, const v_int16x8& right) { return v_int16x8(_mm_max_epi16(left.data, right.data)); }
		template<> v_int32x4 v_max_replace<v_int32x4>(const v_int32x4& left, const v_int32x4& right) { return v_int32x4(_mm_max_epi32(left.data, right.data)); }

		template<> v_uint8x32 v_max_replace<v_uint8x32>(const v_uint8x32& left, const v_uint8x32& right) { return v_uint8x32(_mm256_max_epu8(left.data, right.data)); }
		template<> v_uint16x16 v_max_replace<v_uint16x16>(const v_uint16x16& left, const v_uint16x16& right) { return v_uint16x16(_mm256_max_epu16(left.data, right.data)); }
		template<> v_uint32x8 v_max_replace<v_uint32x8>(const v_uint32x8& left, const v_uint32x8& right) { return v_uint32x8(_mm256_max_epu32(left.data, right.data)); }

		template<> v_int8x32 v_max_replace<v_int8x32>(const v_int8x32& left, const v_int8x32& right) { return v_int8x32(_mm256_max_epi8(left.data, right.data)); }
		template<> v_int16x16 v_max_replace<v_int16x16>(const v_int16x16& left, const v_int16x16& right) { return v_int16x16(_mm256_max_epi16(left.data, right.data)); }
		template<> v_int32x8 v_max_replace<v_int32x8>(const v_int32x8& left, const v_int32x8& right) { return v_int32x8(_mm256_max_epi32(left.data, right.data)); }

		template<> v_float32x4 v_max_replace<v_float32x4>(const v_float32x4& left, const v_float32x4& right) { return v_float32x4(_mm_max_ps(left.data, right.data)); }
		template<> v_float32x8 v_max_replace<v_float32x8>(const v_float32x8& left, const v_float32x8& right) { return v_float32x8(_mm256_max_ps(left.data, right.data)); }

		template<> v_float64x2 v_max_replace<v_float64x2>(const v_float64x2& left, const v_float64x2& right) { return v_float64x2(_mm_max_pd(left.data, right.data)); }
		template<> v_float64x4 v_max_replace<v_float64x4>(const v_float64x4& left, const v_float64x4& right) { return v_float64x4(_mm256_max_pd(left.data, right.data)); }

		template<> v_int64x2 v_max_replace<v_int64x2>(const v_int64x2& left, const v_int64x2& right)
		{
			return v_int64x2(
				std::max(left.data.m128i_i64[0], right.data.m128i_i64[0]),
				std::max(left.data.m128i_i64[1], right.data.m128i_i64[1])
			);
		}

		template<> v_int64x4 v_max_replace<v_int64x4>(const v_int64x4& left, const v_int64x4& right)
		{
			return v_int64x4(
				std::max(left.data.m256i_i64[0], right.data.m256i_i64[0]),
				std::max(left.data.m256i_i64[1], right.data.m256i_i64[1]),
				std::max(left.data.m256i_i64[2], right.data.m256i_i64[2]),
				std::max(left.data.m256i_i64[3], right.data.m256i_i64[3])
			);
		}

		template<> v_uint64x2 v_max_replace<v_uint64x2>(const v_uint64x2& left, const v_uint64x2& right)
		{
			return v_uint64x2(
				std::max(left.data.m128i_u64[0], right.data.m128i_u64[0]),
				std::max(left.data.m128i_u64[1], right.data.m128i_u64[1])
			);
		}

		template<> v_uint64x4 v_max_replace<v_uint64x4>(const v_uint64x4& left, const v_uint64x4& right)
		{
			return v_uint64x4(
				std::max(left.data.m256i_u64[0], right.data.m256i_u64[0]),
				std::max(left.data.m256i_u64[1], right.data.m256i_u64[1]),
				std::max(left.data.m256i_u64[2], right.data.m256i_u64[2]),
				std::max(left.data.m256i_u64[3], right.data.m256i_u64[3])
			);
		}

		template<> v_float16x8 v_max_replace<v_float16x8>(const v_float16x8& left, const v_float16x8& right)
		{
			v_float32x8 vsrc_left = v_convert<v_float32x8>(left);
			v_float32x8 vsrc_right = v_convert<v_float32x8>(right);
			v_float32x8 vres = v_max_replace<v_float32x8>(vsrc_left, vsrc_right);
			return v_convert<v_float16x8>(vres);
		}

		template<> v_float16x16 v_max_replace<v_float16x16>(const v_float16x16& left, const v_float16x16& right)
		{
			v_float32x8 left_0, left_1, right_0, right_1;

			left_0.data = _mm256_cvtph_ps(_mm256_extractf128_si256(left.data, 0));
			left_1.data = _mm256_cvtph_ps(_mm256_extractf128_si256(left.data, 1));

			right_0.data = _mm256_cvtph_ps(_mm256_extractf128_si256(right.data, 0));
			right_1.data = _mm256_cvtph_ps(_mm256_extractf128_si256(right.data, 1));

			v_float32x8 vres_0 = v_max_replace<v_float32x8>(left_0, right_0);
			v_float32x8 vres_1 = v_max_replace<v_float32x8>(left_1, right_1);

			v_float16x8 vsrc_left = v_convert<v_float16x8>(vres_0);
			v_float16x8 vsrc_right = v_convert<v_float16x8>(vres_1);

			return v_float16x16(_mm256_inserti128_si256(_mm256_castsi128_si256(vsrc_left.data), vsrc_right.data, 1));
		}
	}
}

export namespace fy
{
	namespace simd
	{
		template<VectorType Src_t> Src_t::scalar_t v_min_reduce(const Src_t& a) { std::unreachable(); }

		template<> v_uint8x16::scalar_t v_min_reduce<v_uint8x16>(const v_uint8x16& a)
		{
			__m128i val = a.data;
			val = _mm_min_epu8(val, _mm_srli_si128(val, 8));
			val = _mm_min_epu8(val, _mm_srli_si128(val, 4));
			val = _mm_min_epu8(val, _mm_srli_si128(val, 2));
			val = _mm_min_epu8(val, _mm_srli_si128(val, 1));
			return static_cast<typename v_uint8x16::scalar_t>(_mm_cvtsi128_si32(val));
		}

		template<> v_uint16x8::scalar_t v_min_reduce<v_uint16x8>(const v_uint16x8& a)
		{
			__m128i val = a.data;
			val = _mm_min_epu16(val, _mm_srli_si128(val, 8));
			val = _mm_min_epu16(val, _mm_srli_si128(val, 4));
			val = _mm_min_epu16(val, _mm_srli_si128(val, 2));
			return static_cast<typename v_uint16x8::scalar_t>(_mm_cvtsi128_si32(val));
		}

		template<> v_uint32x4::scalar_t v_min_reduce<v_uint32x4>(const v_uint32x4& a)
		{
			__m128i val = a.data;
			val = _mm_min_epu32(val, _mm_srli_si128(val, 8));
			val = _mm_min_epu32(val, _mm_srli_si128(val, 4));
			return static_cast<typename v_uint32x4::scalar_t>(_mm_cvtsi128_si32(val));
		}

		template<> v_uint64x2::scalar_t v_min_reduce<v_uint64x2>(const v_uint64x2& a)
		{
			return std::min(a.data.m128i_u64[0], a.data.m128i_u64[1]);
		}

		template<> v_int8x16::scalar_t v_min_reduce<v_int8x16>(const v_int8x16& a)
		{
			__m128i val = a.data;
			val = _mm_min_epi8(val, _mm_srli_si128(val, 8));
			val = _mm_min_epi8(val, _mm_srli_si128(val, 4));
			val = _mm_min_epi8(val, _mm_srli_si128(val, 2));
			val = _mm_min_epi8(val, _mm_srli_si128(val, 1));
			return static_cast<typename v_int8x16::scalar_t>(_mm_extract_epi8(val, 0));
		}

		template<> v_int16x8::scalar_t v_min_reduce<v_int16x8>(const v_int16x8& a)
		{
			__m128i val = a.data;
			val = _mm_min_epi16(val, _mm_srli_si128(val, 8));
			val = _mm_min_epi16(val, _mm_srli_si128(val, 4));
			val = _mm_min_epi16(val, _mm_srli_si128(val, 2));
			return static_cast<typename v_int16x8::scalar_t>(_mm_extract_epi16(val, 0));
		}

		template<> v_int32x4::scalar_t v_min_reduce<v_int32x4>(const v_int32x4& a)
		{
			__m128i val = a.data;
			val = _mm_min_epi32(val, _mm_srli_si128(val, 8));
			val = _mm_min_epi32(val, _mm_srli_si128(val, 4));
			return static_cast<typename v_int32x4::scalar_t>(_mm_cvtsi128_si32(val));
		}

		template<> v_int64x2::scalar_t v_min_reduce<v_int64x2>(const v_int64x2& a)
		{
			return std::min(a.data.m128i_i64[0], a.data.m128i_i64[1]);
		}

		template<> v_float32x4::scalar_t v_min_reduce<v_float32x4>(const v_float32x4& a)
		{
			__m128 v0 = a.data;
			v0 = _mm_min_ps(v0, _mm_movehl_ps(v0, v0));
			v0 = _mm_min_ss(v0, _mm_shuffle_ps(v0, v0, 1));
			return _mm_cvtss_f32(v0);
		}

		template<> v_float16x8::scalar_t v_min_reduce<v_float16x8>(const v_float16x8& a)
		{
			__m256 raw = _mm256_cvtph_ps(a.data);

			__m128 v0 = _mm256_castps256_ps128(raw);
			__m128 v1 = _mm256_extractf128_ps(raw, 1);
			v0 = _mm_min_ps(v0, v1);
			v0 = _mm_min_ps(v0, _mm_permute_ps(v0, (((0) << 6) | ((0) << 4) | ((3) << 2) | ((2)))));
			v0 = _mm_min_ps(v0, _mm_permute_ps(v0, (((0) << 6) | ((0) << 4) | ((0) << 2) | ((1)))));
			return f16(static_cast<typename v_float32x8::scalar_t>(_mm_cvtss_f32(v0)));
		}

		template<> v_float64x2::scalar_t v_min_reduce<v_float64x2>(const v_float64x2& a)
		{
			return std::min(a.data.m128d_f64[0], a.data.m128d_f64[1]);
		}

		template<> v_uint8x32::scalar_t v_min_reduce<v_uint8x32>(const v_uint8x32& a)
		{
			__m128i val = _mm_min_epu8(_mm256_castsi256_si128(a.data), _mm256_extracti128_si256(a.data, 1));
			val = _mm_min_epu8(val, _mm_srli_si128(val, 8));
			val = _mm_min_epu8(val, _mm_srli_si128(val, 4));
			val = _mm_min_epu8(val, _mm_srli_si128(val, 2));
			val = _mm_min_epu8(val, _mm_srli_si128(val, 1));
			return static_cast<typename v_uint8x32::scalar_t>(_mm_cvtsi128_si32(val));
		}

		template<> v_uint16x16::scalar_t v_min_reduce(const v_uint16x16& a)
		{
			__m128i v0 = _mm256_castsi256_si128(a.data);
			v0 = _mm_min_epu16(v0, _mm256_extracti128_si256(a.data, 1));
			v0 = _mm_min_epu16(v0, _mm_srli_si128(v0, 8));
			v0 = _mm_min_epu16(v0, _mm_srli_si128(v0, 4));
			v0 = _mm_min_epu16(v0, _mm_srli_si128(v0, 2));
			return static_cast<typename v_uint16x16::scalar_t>(_mm_cvtsi128_si32(v0));
		}

		template<> v_uint32x8::scalar_t v_min_reduce(const v_uint32x8& a)
		{
			__m128i v0 = _mm256_castsi256_si128(a.data);
			__m128i v1 = _mm256_extracti128_si256(a.data, 1);
			v0 = _mm_min_epu32(v0, v1);
			v0 = _mm_min_epu32(v0, _mm_srli_si128(v0, 8));
			v0 = _mm_min_epu32(v0, _mm_srli_si128(v0, 4));
			return static_cast<typename v_uint32x8::scalar_t>(_mm_cvtsi128_si32(v0));
		}

		template<> v_uint64x4::scalar_t v_min_reduce(const v_uint64x4& a)
		{
			__m128i v0 = _mm256_castsi256_si128(a.data);
			__m128i v1 = _mm256_extracti128_si256(a.data, 1);

			u64 min1 = std::min(v0.m128i_u64[0], v0.m128i_u64[1]);
			u64 min2 = std::min(v1.m128i_u64[0], v1.m128i_u64[1]);

			return std::min(min1, min2);
		}

		template<> v_int8x32::scalar_t v_min_reduce(const v_int8x32& a)
		{
			__m128i val = _mm_min_epi8(_mm256_castsi256_si128(a.data), _mm256_extracti128_si256(a.data, 1));
			val = _mm_min_epi8(val, _mm_srli_si128(val, 8));
			val = _mm_min_epi8(val, _mm_srli_si128(val, 4));
			val = _mm_min_epi8(val, _mm_srli_si128(val, 2));
			val = _mm_min_epi8(val, _mm_srli_si128(val, 1));
			return static_cast<typename v_int8x32::scalar_t>(_mm_cvtsi128_si32(val));
		}

		template<> v_int16x16::scalar_t v_min_reduce(const v_int16x16& a)
		{
			__m128i v0 = _mm256_castsi256_si128(a.data);
			__m128i v1 = _mm256_extracti128_si256(a.data, 1);
			v0 = _mm_min_epi16(v0, v1);
			v0 = _mm_min_epi16(v0, _mm_srli_si128(v0, 8));
			v0 = _mm_min_epi16(v0, _mm_srli_si128(v0, 4));
			v0 = _mm_min_epi16(v0, _mm_srli_si128(v0, 2));
			return static_cast<typename v_int16x16::scalar_t>(_mm_cvtsi128_si32(v0));
		}

		template<> v_int32x8::scalar_t v_min_reduce(const v_int32x8& a)
		{
			__m128i v0 = _mm256_castsi256_si128(a.data);
			__m128i v1 = _mm256_extracti128_si256(a.data, 1);
			v0 = _mm_min_epi32(v0, v1);
			v0 = _mm_min_epi32(v0, _mm_srli_si128(v0, 8));
			v0 = _mm_min_epi32(v0, _mm_srli_si128(v0, 4));
			return static_cast<typename v_int32x8::scalar_t>(_mm_cvtsi128_si32(v0));
		}

		template<> v_int64x4::scalar_t v_min_reduce(const v_int64x4& a)
		{
			__m128i v0 = _mm256_castsi256_si128(a.data);
			__m128i v1 = _mm256_extracti128_si256(a.data, 1);

			i64 min1 = std::min(v0.m128i_i64[0], v0.m128i_i64[1]);
			i64 min2 = std::min(v1.m128i_i64[0], v1.m128i_i64[1]);

			return std::min(min1, min2);
		}

		template<> v_float32x8::scalar_t v_min_reduce(const v_float32x8& a)
		{
			__m128 v0 = _mm256_castps256_ps128(a.data);
			__m128 v1 = _mm256_extractf128_ps(a.data, 1);
			v0 = _mm_min_ps(v0, v1);
			v0 = _mm_min_ps(v0, _mm_permute_ps(v0, (((0) << 6) | ((0) << 4) | ((3) << 2) | ((2)))));
			v0 = _mm_min_ps(v0, _mm_permute_ps(v0, (((0) << 6) | ((0) << 4) | ((0) << 2) | ((1)))));
			return static_cast<typename v_float32x8::scalar_t>(_mm_cvtss_f32(v0));
		}

		template<> v_float64x4::scalar_t v_min_reduce(const v_float64x4& a)
		{
			__m256d v0 = a.data;
			__m128d v1 = _mm256_extractf128_pd(v0, 1);
			__m128d v2 = _mm256_castpd256_pd128(v0);
			v2 = _mm_min_pd(v2, v1);
			v2 = _mm_min_pd(v2, _mm_permute_pd(v2, 1));
			return static_cast<typename v_float64x4::scalar_t>(_mm_cvtsd_f64(v2));
		}

		template<> v_float16x16::scalar_t v_min_reduce(const v_float16x16& a)
		{
			v_float32x8 f32_dst_0, f32_dst_1;
			f32_dst_0.data = _mm256_cvtph_ps(_mm256_extractf128_si256(a.data, 0));
			f32_dst_1.data = _mm256_cvtph_ps(_mm256_extractf128_si256(a.data, 1));

			v_float32x8::scalar_t res_0 = v_min_reduce<v_float32x8>(f32_dst_0);
			v_float32x8::scalar_t res_1 = v_min_reduce<v_float32x8>(f32_dst_1);
			return f16(std::min(res_0, res_1));
		}
	}
}
//
//export namespace fy
//{
//	namespace simd
//	{
//		template<VectorType Src_t> 
//		auto v_max_reduce(const Src_t& a) { std::unreachable(); }
//
//		template<>
//		auto v_max_reduce(const v_uint8x32& a)
//		{
//			__m128i val = _mm_max_epu8((_mm256_castsi256_si128(a.data)), (_mm256_extracti128_si256((a.data), 1))); 
//			val = _mm_max_epu8(val, _mm_srli_si128(val, 8)); 
//			val = _mm_max_epu8(val, _mm_srli_si128(val, 4)); 
//			val = _mm_max_epu8(val, _mm_srli_si128(val, 2)); 
//			val = _mm_max_epu8(val, _mm_srli_si128(val, 1)); 
//			return _mm_cvtsi128_si32(val);
//		}
//
//		template<>
//		auto v_max_reduce(const v_int8x32& a)
//		{
//			__m128i val = _mm_max_epi8((_mm256_castsi256_si128(a.data)), (_mm256_extracti128_si256((a.data), 1))); 
//			val = _mm_max_epi8(val, _mm_srli_si128(val, 8)); 
//			val = _mm_max_epi8(val, _mm_srli_si128(val, 4)); 
//			val = _mm_max_epi8(val, _mm_srli_si128(val, 2)); 
//			val = _mm_max_epi8(val, _mm_srli_si128(val, 1)); 
//			return _mm_cvtsi128_si32(val);
//		}
//
//		template<>
//		auto v_max_reduce(const v_uint16x16& a)
//		{
//			__m128i v0 = (_mm256_castsi256_si128(a.data)); 
//			__m128i v1 = (_mm256_extracti128_si256((a.data), 1));
//			v0 = _mm_max_epu16(v0, v1); 
//			v0 = _mm_max_epu16(v0, _mm_srli_si128(v0, 8)); 
//			v0 = _mm_max_epu16(v0, _mm_srli_si128(v0, 4)); 
//			v0 = _mm_max_epu16(v0, _mm_srli_si128(v0, 2)); 
//			return _mm_cvtsi128_si32(v0);
//		}
//
//		template<>
//		auto v_max_reduce(const v_int16x16& a)
//		{
//			__m128i v0 = (_mm256_castsi256_si128(a.data)); 
//			__m128i v1 = (_mm256_extracti128_si256((a.data), 1));
//			v0 = _mm_max_epi16(v0, v1); 
//			v0 = _mm_max_epi16(v0, _mm_srli_si128(v0, 8));
//			v0 = _mm_max_epi16(v0, _mm_srli_si128(v0, 4));
//			v0 = _mm_max_epi16(v0, _mm_srli_si128(v0, 2)); 
//			return _mm_cvtsi128_si32(v0);
//		}
//
//		template<>
//		auto v_max_reduce(const v_uint32x8& a)
//		{
//			__m128i v0 = (_mm256_castsi256_si128(a.data)); 
//			__m128i v1 = (_mm256_extracti128_si256((a.data), 1));
//			v0 = _mm_max_epu32(v0, v1); 
//			v0 = _mm_max_epu32(v0, _mm_srli_si128(v0, 8)); 
//			v0 = _mm_max_epu32(v0, _mm_srli_si128(v0, 4)); 
//			return _mm_cvtsi128_si32(v0);
//		}
//
//		template<>
//		auto v_max_reduce(const v_int32x8& a)
//		{
//			__m128i v0 = (_mm256_castsi256_si128(a.data)); 
//			__m128i v1 = (_mm256_extracti128_si256((a.data), 1));
//			v0 = _mm_max_epi32(v0, v1); 
//			v0 = _mm_max_epi32(v0, _mm_srli_si128(v0, 8)); 
//			v0 = _mm_max_epi32(v0, _mm_srli_si128(v0, 4)); 
//			return _mm_cvtsi128_si32(v0);
//		}
//
//		template<>
//		auto v_max_reduce(const v_uint64x4& a)
//		{
//			u64 max_val = a.data.m256i_u64[0];
//			for (i32 i = 1; i < 4; i++)
//			{
//				if (a.data.m256i_u64[i] > max_val)
//					max_val = a.data.m256i_u64[i];
//			}
//			return max_val;
//		}
//
//		template<>
//		auto v_max_reduce(const v_int64x4& a)
//		{
//			i64 max_val = a.data.m256i_i64[0];
//			for (i32 i = 1; i < 4; i++)
//			{
//				if (a.data.m256i_i64[i] > max_val)
//					max_val = a.data.m256i_i64[i];
//			}
//			return max_val;
//		}
//
//		template<>
//		auto v_max_reduce(const v_float32x8& a)
//		{
//			__m128 v0 = (_mm256_castps256_ps128(a.data));
//			__m128 v1 = (_mm256_extractf128_ps((a.data), 1)); 
//			v0 = _mm_max_ps(v0, v1); 
//			v0 = _mm_max_ps(v0, _mm_permute_ps(v0, (((0) << 6) | ((0) << 4) | ((3) << 2) | ((2))))); 
//			v0 = _mm_max_ps(v0, _mm_permute_ps(v0, (((0) << 6) | ((0) << 4) | ((0) << 2) | ((1)))));
//			return _mm_cvtss_f32(v0);
//		}
//
//		template<>
//		auto v_max_reduce(const v_float64x4& a)
//		{
//			f64 max_val = a.data.m256d_f64[0];
//			for (i32 i = 1; i < 4; i++)
//			{
//				if (a.data.m256d_f64[i] > max_val)
//					max_val = a.data.m256d_f64[i];
//			}
//			return max_val;
//		}
//	}
//}
//

export namespace fy
{
	namespace simd
	{
		template<VectorType Vec_t> struct v_abs_Invoker
		{
			v_abs_Invoker() {}
			Vec_t operator () (const Vec_t& src) const noexcept
			{
				if constexpr (std::is_integral_v<Vec_t::scalar_t>)
				{
					if constexpr (std::is_unsigned_v<Vec_t::scalar_t>)
					{
						return Vec_t(src.data);
					}
					else
					{
						if constexpr (sizeof(Vec_t::scalar_t) == sizeof(i8))
						{
							if constexpr (sizeof(Vec_t) == sizeof(__m128))
							{
								return Vec_t(_mm_abs_epi8(src.data));
							}
							else if constexpr (sizeof(Vec_t) == sizeof(__m256))
							{
								return Vec_t(_mm256_abs_epi8(src.data));
							}
							else
							{
								std::unreachable();
							}
						}
						else if constexpr (sizeof(Vec_t::scalar_t) == sizeof(i16))
						{
							if constexpr (sizeof(Vec_t) == sizeof(__m128))
							{
								return Vec_t(_mm_abs_epi16(src.data));
							}
							else if constexpr (sizeof(Vec_t) == sizeof(__m256))
							{
								return Vec_t(_mm256_abs_epi16(src.data));
							}
							else
							{
								std::unreachable();
							}
						}
						else if constexpr (sizeof(Vec_t::scalar_t) == sizeof(i32))
						{
							if constexpr (sizeof(Vec_t) == sizeof(__m128))
							{
								return Vec_t(_mm_abs_epi32(src.data));
							}
							else if constexpr (sizeof(Vec_t) == sizeof(__m256))
							{
								return Vec_t(_mm256_abs_epi32(src.data));
							}
							else
							{
								std::unreachable();
							}
						}
						else if constexpr (sizeof(Vec_t::scalar_t) == sizeof(i64))
						{
							if constexpr (sizeof(Vec_t) == sizeof(__m128i))
							{
								__m128i sign_mask = _mm_cmpgt_epi64(_mm_setzero_si128(), src.data);
								__m128i abs_val = _mm_add_epi64(_mm_xor_si128(src.data, sign_mask), _mm_srli_epi64(sign_mask, 63));
								return Vec_t(abs_val);
							}
							else if constexpr (sizeof(Vec_t) == sizeof(__m256i))
							{
								__m256i sign_mask = _mm256_cmpgt_epi64(_mm256_setzero_si256(), src.data);
								__m256i abs_val = _mm256_add_epi64(_mm256_xor_si256(src.data, sign_mask), _mm256_srli_epi64(sign_mask, 63));
								return Vec_t(abs_val);
							}
							else
							{
								std::unreachable();
							}
						}
						else
						{
							std::unreachable();
						}
					}
				}
				else if constexpr (std::is_floating_point_v<Vec_t::scalar_t> || std::is_same_v<typename Vec_t::scalar_t, f16>)
				{
					if constexpr (std::is_same_v<typename Vec_t::scalar_t, f16>)
					{
						if constexpr (std::is_same_v<Vec_t, v_float16x8>)
						{
							return Vec_t(_mm_abs_ph(src.data));
						}
						else if constexpr (std::is_same_v<Vec_t, v_float16x16>)
						{
							return Vec_t(_mm256_abs_ph(src.data));
						}
						else
						{
							std::unreachable();
						}
					}
					else if constexpr (std::is_same_v<typename Vec_t::scalar_t, f32>)
					{
						if constexpr (sizeof(Vec_t) == sizeof(__m128))
						{
							static const __m128 abs_mask = _mm_castsi128_ps(_mm_set1_epi32(0x7FFFFFFF));
							return Vec_t(_mm_and_ps(src.data, abs_mask));
						}
						else if constexpr (sizeof(Vec_t) == sizeof(__m256))
						{
							static const __m256 abs_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
							return Vec_t(_mm256_and_ps(src.data, abs_mask));
						}
						else
						{
							std::unreachable();
						}
					}
					else if constexpr (std::is_same_v<typename Vec_t::scalar_t, f64>)
					{
						if constexpr (sizeof(Vec_t) == sizeof(__m128d))
						{
							static const __m128d abs_mask = _mm_castsi128_pd(_mm_set1_epi64x(0x7FFFFFFFFFFFFFFFLL));
							return Vec_t(_mm_and_pd(src.data, abs_mask));
						}
						else if constexpr (sizeof(Vec_t) == sizeof(__m256d))
						{
							static const __m256d abs_mask = _mm256_castsi256_pd(_mm256_set1_epi64x(0x7FFFFFFFFFFFFFFFLL));
							return Vec_t(_mm256_and_pd(src.data, abs_mask));
						}
						else
						{
							std::unreachable();
						}
					}
					else
					{
						std::unreachable();
					}
				}
				else
				{
					std::unreachable();
				}
			}
		};

		template<VectorType Vec_t>
		Vec_t v_abs(const Vec_t& src)
		{
			return v_abs_Invoker<Vec_t>()(src);
		}
	}
}

export namespace fy
{
	namespace simd
	{
		template<VectorType Vec_t>
		Vec_t v_avg(const Vec_t& src0, const Vec_t& src1)
		{
			return (src0 + src1 + 1) / 2;
		}

		template<> v_uint8x32 v_avg(const v_uint8x32& src0, const v_uint8x32& src1)
		{
			return v_uint8x32(_mm256_avg_epu8(src0.data, src1.data));
		}

		template<> v_uint16x16 v_avg(const v_uint16x16& src0, const v_uint16x16& src1)
		{
			return v_uint16x16(_mm256_avg_epu16(src0.data, src1.data));
		}


	}
}


export namespace fy
{
	namespace simd
	{
		template<Floating_VectorType Vec_t>
		Vec_t v_exp(const Vec_t& src)
		{
			std::unreachable();
		}

		template<> v_float16x8 v_exp(const v_float16x8& src)
		{
			__m256 vsrc = _mm256_cvtph_ps(src.data);
			vsrc = _mm256_exp_ps(vsrc);
			return v_float16x8(_mm256_cvtps_ph(vsrc, 0x00));
		}

		template<> v_float32x4 v_exp(const v_float32x4& src)
		{
			return v_float32x4(_mm_exp_ps(src.data));
		}

		template<> v_float64x2 v_exp(const v_float64x2& src)
		{
			return v_float64x2(_mm_exp_pd(src.data));
		}

		template<> v_float16x16 v_exp(const v_float16x16& src)
		{
			__m256 res_lo = _mm256_exp_ps(_mm256_cvtph_ps(_mm256_extractf128_si256(src.data, 0)));
			__m256 res_hi = _mm256_exp_ps(_mm256_cvtph_ps(_mm256_extractf128_si256(src.data, 1)));

			__m128i low = _mm256_cvtps_ph(res_lo, _MM_FROUND_TO_NEAREST_INT);
			__m128i high = _mm256_cvtps_ph(res_hi, _MM_FROUND_TO_NEAREST_INT);

			return v_float16x16(_mm256_insertf128_si256(_mm256_castsi128_si256(low), high, 1));
		}

		template<> v_float32x8 v_exp(const v_float32x8& src)
		{
			return v_float32x8(_mm256_exp_ps(src.data));
		}

		template<> v_float64x4 v_exp(const v_float64x4& src)
		{
			return v_float64x4(_mm256_exp_pd(src.data));
		}
	}
}

export namespace fy
{
	namespace simd
	{
		template<Floating_VectorType Vec_t>
		Vec_t v_exp2(const Vec_t& src)
		{
			std::unreachable();
		}

		template<> v_float16x8 v_exp2(const v_float16x8& src)
		{
			__m256 vsrc = _mm256_cvtph_ps(src.data);
			vsrc = _mm256_exp2_ps(vsrc);
			return v_float16x8(_mm256_cvtps_ph(vsrc, 0x00));
		}

		template<> v_float32x4 v_exp2(const v_float32x4& src)
		{
			return v_float32x4(_mm_exp2_ps(src.data));
		}

		template<> v_float64x2 v_exp2(const v_float64x2& src)
		{
			return v_float64x2(_mm_exp2_pd(src.data));
		}

		template<> v_float16x16 v_exp2(const v_float16x16& src)
		{
			__m256 res_lo = _mm256_exp2_ps(_mm256_cvtph_ps(_mm256_extractf128_si256(src.data, 0)));
			__m256 res_hi = _mm256_exp2_ps(_mm256_cvtph_ps(_mm256_extractf128_si256(src.data, 1)));

			__m128i low = _mm256_cvtps_ph(res_lo, _MM_FROUND_TO_NEAREST_INT);
			__m128i high = _mm256_cvtps_ph(res_hi, _MM_FROUND_TO_NEAREST_INT);

			return v_float16x16(_mm256_insertf128_si256(_mm256_castsi128_si256(low), high, 1));
		}

		template<> v_float32x8 v_exp2(const v_float32x8& src)
		{
			return v_float32x8(_mm256_exp2_ps(src.data));
		}

		template<> v_float64x4 v_exp2(const v_float64x4& src)
		{
			return v_float64x4(_mm256_exp2_pd(src.data));
		}
	}
}

export namespace fy
{
	namespace simd
	{
		template<Floating_VectorType Vec_t>
		Vec_t v_exp10(const Vec_t& src)
		{
			std::unreachable();
		}

		template<> v_float16x8 v_exp10(const v_float16x8& src)
		{
			__m256 vsrc = _mm256_cvtph_ps(src.data);
			vsrc = _mm256_exp10_ps(vsrc);
			return v_float16x8(_mm256_cvtps_ph(vsrc, 0x00));
		}

		template<> v_float32x4 v_exp10(const v_float32x4& src)
		{
			return v_float32x4(_mm_exp10_ps(src.data));
		}

		template<> v_float64x2 v_exp10(const v_float64x2& src)
		{
			return v_float64x2(_mm_exp10_pd(src.data));
		}

		template<> v_float16x16 v_exp10(const v_float16x16& src)
		{
			__m256 res_lo = _mm256_exp10_ps(_mm256_cvtph_ps(_mm256_extractf128_si256(src.data, 0)));
			__m256 res_hi = _mm256_exp10_ps(_mm256_cvtph_ps(_mm256_extractf128_si256(src.data, 1)));

			__m128i low = _mm256_cvtps_ph(res_lo, _MM_FROUND_TO_NEAREST_INT);
			__m128i high = _mm256_cvtps_ph(res_hi, _MM_FROUND_TO_NEAREST_INT);

			return v_float16x16(_mm256_insertf128_si256(_mm256_castsi128_si256(low), high, 1));
		}

		template<> v_float32x8 v_exp10(const v_float32x8& src)
		{
			return v_float32x8(_mm256_exp10_ps(src.data));
		}

		template<> v_float64x4 v_exp10(const v_float64x4& src)
		{
			return v_float64x4(_mm256_exp10_pd(src.data));
		}
	}
}

export namespace fy
{
	namespace simd
	{
		template<Floating_VectorType Vec_t>
		Vec_t v_log(const Vec_t& src)
		{
			std::unreachable();
		}

		template<> v_float16x8 v_log(const v_float16x8& src)
		{
			__m256 vsrc = _mm256_cvtph_ps(src.data);
			vsrc = _mm256_log_ps(vsrc);
			return v_float16x8(_mm256_cvtps_ph(vsrc, 0x00));
		}

		template<> v_float32x4 v_log(const v_float32x4& src)
		{
			return v_float32x4(_mm_log_ps(src.data));
		}

		template<> v_float64x2 v_log(const v_float64x2& src)
		{
			return v_float64x2(_mm_log_pd(src.data));
		}

		template<> v_float16x16 v_log(const v_float16x16& src)
		{
			__m256 res_lo = _mm256_log_ps(_mm256_cvtph_ps(_mm256_extractf128_si256(src.data, 0)));
			__m256 res_hi = _mm256_log_ps(_mm256_cvtph_ps(_mm256_extractf128_si256(src.data, 1)));

			__m128i low = _mm256_cvtps_ph(res_lo, _MM_FROUND_TO_NEAREST_INT);
			__m128i high = _mm256_cvtps_ph(res_hi, _MM_FROUND_TO_NEAREST_INT);

			return v_float16x16(_mm256_insertf128_si256(_mm256_castsi128_si256(low), high, 1));
		}

		template<> v_float32x8 v_log(const v_float32x8& src)
		{
			return v_float32x8(_mm256_log_ps(src.data));
		}

		template<> v_float64x4 v_log(const v_float64x4& src)
		{
			return v_float64x4(_mm256_log_pd(src.data));
		}
	}
}

export namespace fy
{
	namespace simd
	{
		template<Floating_VectorType Vec_t>
		Vec_t v_log2(const Vec_t& src)
		{
			std::unreachable();
		}

		template<> v_float16x8 v_log2(const v_float16x8& src)
		{
			__m256 vsrc = _mm256_cvtph_ps(src.data);
			vsrc = _mm256_log2_ps(vsrc);
			return v_float16x8(_mm256_cvtps_ph(vsrc, 0x00));
		}

		template<> v_float32x4 v_log2(const v_float32x4& src)
		{
			return v_float32x4(_mm_log2_ps(src.data));
		}

		template<> v_float64x2 v_log2(const v_float64x2& src)
		{
			return v_float64x2(_mm_log2_pd(src.data));
		}

		template<> v_float16x16 v_log2(const v_float16x16& src)
		{
			__m256 res_lo = _mm256_log2_ps(_mm256_cvtph_ps(_mm256_extractf128_si256(src.data, 0)));
			__m256 res_hi = _mm256_log2_ps(_mm256_cvtph_ps(_mm256_extractf128_si256(src.data, 1)));

			__m128i low = _mm256_cvtps_ph(res_lo, _MM_FROUND_TO_NEAREST_INT);
			__m128i high = _mm256_cvtps_ph(res_hi, _MM_FROUND_TO_NEAREST_INT);

			return v_float16x16(_mm256_insertf128_si256(_mm256_castsi128_si256(low), high, 1));
		}

		template<> v_float32x8 v_log2(const v_float32x8& src)
		{
			return v_float32x8(_mm256_log2_ps(src.data));
		}

		template<> v_float64x4 v_log2(const v_float64x4& src)
		{
			return v_float64x4(_mm256_log2_pd(src.data));
		}
	}
}

export namespace fy
{
	namespace simd
	{
		template<Floating_VectorType Vec_t>
		Vec_t v_log10(const Vec_t& src)
		{
			std::unreachable();
		}

		template<> v_float16x8 v_log10(const v_float16x8& src)
		{
			__m256 vsrc = _mm256_cvtph_ps(src.data);
			vsrc = _mm256_log10_ps(vsrc);
			return v_float16x8(_mm256_cvtps_ph(vsrc, 0x00));
		}

		template<> v_float32x4 v_log10(const v_float32x4& src)
		{
			return v_float32x4(_mm_log10_ps(src.data));
		}

		template<> v_float64x2 v_log10(const v_float64x2& src)
		{
			return v_float64x2(_mm_log10_pd(src.data));
		}

		template<> v_float16x16 v_log10(const v_float16x16& src)
		{
			__m256 res_lo = _mm256_log10_ps(_mm256_cvtph_ps(_mm256_extractf128_si256(src.data, 0)));
			__m256 res_hi = _mm256_log10_ps(_mm256_cvtph_ps(_mm256_extractf128_si256(src.data, 1)));

			__m128i low = _mm256_cvtps_ph(res_lo, _MM_FROUND_TO_NEAREST_INT);
			__m128i high = _mm256_cvtps_ph(res_hi, _MM_FROUND_TO_NEAREST_INT);

			return v_float16x16(_mm256_insertf128_si256(_mm256_castsi128_si256(low), high, 1));
		}

		template<> v_float32x8 v_log10(const v_float32x8& src)
		{
			return v_float32x8(_mm256_log10_ps(src.data));
		}

		template<> v_float64x4 v_log10(const v_float64x4& src)
		{
			return v_float64x4(_mm256_log10_pd(src.data));
		}
	}
}

export namespace fy
{
	namespace simd
	{
		template<Floating_VectorType Vec_t> Vec_t v_sqrt(const Vec_t& src) { std::unreachable(); }

		template<> v_float32x4 v_sqrt(const v_float32x4& src) { return v_float32x4(_mm_sqrt_ps(src.data)); }
		template<> v_float64x2 v_sqrt(const v_float64x2& src) { return v_float64x2(_mm_sqrt_pd(src.data)); }
		template<> v_float32x8 v_sqrt(const v_float32x8& src) { return v_float32x8(_mm256_sqrt_ps(src.data)); }
		template<> v_float64x4 v_sqrt(const v_float64x4& src) { return v_float64x4(_mm256_sqrt_pd(src.data)); }

		template<> v_float16x8 v_sqrt(const v_float16x8& src)
		{
			__m256 raw_float32 = _mm256_cvtph_ps(src.data);
			__m256 res_float32 = _mm256_sqrt_ps(raw_float32);
			__m128h vres = _mm256_cvtps_ph(res_float32, _MM_FROUND_TO_NEAREST_INT);
			return v_float16x8(vres);
		}

		template<> v_float16x16 v_sqrt(const v_float16x16& src)
		{
			__m256 res_lo = _mm256_sqrt_ps(_mm256_cvtph_ps(_mm256_extractf128_si256(src.data, 0)));
			__m256 res_hi = _mm256_sqrt_ps(_mm256_cvtph_ps(_mm256_extractf128_si256(src.data, 1)));

			__m128i low = _mm256_cvtps_ph(res_lo, _MM_FROUND_TO_NEAREST_INT);
			__m128i high = _mm256_cvtps_ph(res_hi, _MM_FROUND_TO_NEAREST_INT);

			return v_float16x16(_mm256_insertf128_si256(_mm256_castsi128_si256(low), high, 1));
		}
	}
}

export namespace fy
{
	namespace simd
	{
		template<Floating_VectorType Vec_t> Vec_t v_rsqrt(const Vec_t& src) { std::unreachable(); }

		template<> v_float32x4 v_rsqrt(const v_float32x4& src) { return v_float32x4(_mm_rsqrt_ps(src.data)); }
		template<> v_float32x8 v_rsqrt(const v_float32x8& src) { return v_float32x8(_mm256_rsqrt_ps(src.data)); }

		template<> v_float16x8 v_rsqrt(const v_float16x8& src)
		{
			__m256 raw_float32 = _mm256_cvtph_ps(src.data);
			__m256 res_float32 = _mm256_rsqrt_ps(raw_float32);
			__m128h vres = _mm256_cvtps_ph(res_float32, _MM_FROUND_TO_NEAREST_INT);
			return v_float16x8(vres);
		}

		template<> v_float16x16 v_rsqrt(const v_float16x16& src)
		{
			__m256 res_lo = _mm256_rsqrt_ps(_mm256_cvtph_ps(_mm256_extractf128_si256(src.data, 0)));
			__m256 res_hi = _mm256_rsqrt_ps(_mm256_cvtph_ps(_mm256_extractf128_si256(src.data, 1)));

			__m128i low = _mm256_cvtps_ph(res_lo, _MM_FROUND_TO_NEAREST_INT);
			__m128i high = _mm256_cvtps_ph(res_hi, _MM_FROUND_TO_NEAREST_INT);

			return v_float16x16(_mm256_insertf128_si256(_mm256_castsi128_si256(low), high, 1));
		}

		template<> v_float64x2 v_rsqrt(const v_float64x2& src)
		{
			constexpr usize N = 2;
			const __m128i MAGIC = _mm_set1_epi64x(0x5FE6EB50C7B537AA);
			const __m128d HALF = _mm_set1_pd(0.5);
			const __m128d ONE_HALF = _mm_set1_pd(1.5);

			__m128i i = _mm_castpd_si128(src.data);
			i = _mm_sub_epi64(MAGIC, _mm_srli_epi64(i, 1));
			__m128d y = _mm_castsi128_pd(i);

			for (usize iter = 0; iter < N; ++iter)
			{
				__m128d y2 = _mm_mul_pd(y, y);
				__m128d half_src = _mm_mul_pd(src.data, HALF);
				__m128d term = _mm_fnmadd_pd(half_src, y2, ONE_HALF);

				y = _mm_mul_pd(y, term);
			}

			return v_float64x2(y);
		}

		template<> v_float64x4 v_rsqrt(const v_float64x4& src)
		{
			constexpr usize N = 2;
			const __m256i MAGIC = _mm256_set1_epi64x(0x5FE6EB50C7B537AA);
			const __m256d HALF = _mm256_set1_pd(0.5);
			const __m256d ONE_HALF = _mm256_set1_pd(1.5);

			__m256i i = _mm256_castpd_si256(src.data);
			i = _mm256_sub_epi64(MAGIC, _mm256_srli_epi64(i, 1));
			__m256d y = _mm256_castsi256_pd(i);

			for (usize iter = 0; iter < N; ++iter)
			{
				__m256d y2 = _mm256_mul_pd(y, y);
				__m256d half_src = _mm256_mul_pd(src.data, HALF);
				__m256d term = _mm256_fnmadd_pd(half_src, y2, ONE_HALF);

				y = _mm256_mul_pd(y, term);
			}

			return v_float64x4(y);
		}
	}
}

export namespace fy
{
	namespace simd
	{
		template<Floating_VectorType Vec_t> Vec_t v_rcp(const Vec_t& src) { std::unreachable(); }

		template<> v_float32x4 v_rcp(const v_float32x4& src) { return v_float32x4(_mm_rcp_ps(src.data)); }
		template<> v_float32x8 v_rcp(const v_float32x8& src) { return v_float32x8(_mm256_rcp_ps(src.data)); }
		
		template<> v_float16x8 v_rcp(const v_float16x8& src)
		{
			__m256 raw_float32 = _mm256_cvtph_ps(src.data);
			__m256 res_float32 = _mm256_rcp_ps(raw_float32);
			__m128h vres = _mm256_cvtps_ph(res_float32, _MM_FROUND_TO_NEAREST_INT);
			return v_float16x8(vres);
		}

		template<> v_float16x16 v_rcp(const v_float16x16& src)
		{
			__m256 res_lo = _mm256_rcp_ps(_mm256_cvtph_ps(_mm256_extractf128_si256(src.data, 0)));
			__m256 res_hi = _mm256_rcp_ps(_mm256_cvtph_ps(_mm256_extractf128_si256(src.data, 1)));

			__m128i low = _mm256_cvtps_ph(res_lo, _MM_FROUND_TO_NEAREST_INT);
			__m128i high = _mm256_cvtps_ph(res_hi, _MM_FROUND_TO_NEAREST_INT);

			return v_float16x16(_mm256_insertf128_si256(_mm256_castsi128_si256(low), high, 1));
		}

		template<> v_float64x2 v_rcp(const v_float64x2& src)
		{
			return v_div(v_float64x2(1), src);
		}

		template<> v_float64x4 v_rcp(const v_float64x4& src)
		{
			return v_div(v_float64x4(1), src);
		}
	}
}

export namespace fy
{
	namespace simd
	{
		template<VectorType Src_t> struct v_as_floating_Invoker
		{
			v_as_floating_Invoker() { std::unreachable(); }
			void operator () () noexcept { std::unreachable(); }
		};

		template<> struct v_as_floating_Invoker<v_uint8x32>
		{
			alignas(32) v_uint16x16 temp_u16[2];
			alignas(32) v_uint32x8 temp_u32[4];

			v_as_floating_Invoker() {}
			void operator () (const v_uint8x32& src, v_float32x8& dst_0, v_float32x8& dst_1, v_float32x8& dst_2, v_float32x8& dst_3) noexcept
			{
				v_expand(src, temp_u16[0], temp_u16[1]);
				v_expand(temp_u16[0], temp_u32[0], temp_u32[1]);
				v_expand(temp_u16[1], temp_u32[2], temp_u32[3]);

				dst_0 = v_convert<v_float32x8>(v_reinterpret_convert<v_int32x8>(temp_u32[0]));
				dst_1 = v_convert<v_float32x8>(v_reinterpret_convert<v_int32x8>(temp_u32[1]));
				dst_2 = v_convert<v_float32x8>(v_reinterpret_convert<v_int32x8>(temp_u32[2]));
				dst_3 = v_convert<v_float32x8>(v_reinterpret_convert<v_int32x8>(temp_u32[3]));
			}
		};

		template<> struct v_as_floating_Invoker<v_uint16x16>
		{
			alignas(32) v_uint32x8 temp_u32[2];

			v_as_floating_Invoker() {}
			void operator () (const v_uint16x16& src, v_float32x8& dst_0, v_float32x8& dst_1) noexcept
			{
				v_expand(src, temp_u32[0], temp_u32[1]);

				dst_0 = v_convert<v_float32x8>(temp_u32[0]);
				dst_1 = v_convert<v_float32x8>(temp_u32[1]);
			}
		};

		template<> struct v_as_floating_Invoker<v_uint32x8>
		{
			v_as_floating_Invoker() {}
			void operator () (const v_uint32x8& src, v_float32x8& dst_0) noexcept
			{
				dst_0 = v_convert<v_float32x8>(src);
			}
		};

		template<> struct v_as_floating_Invoker<v_uint64x4>
		{
			v_as_floating_Invoker() {}
			void operator () (const v_uint64x4& src, v_float64x4& dst_0) noexcept
			{
				dst_0 = v_convert<v_float64x4>(src);
			}
		};

		template<> struct v_as_floating_Invoker<v_int8x32>
		{
			alignas(32) v_int16x16 temp_i16[2];
			alignas(32) v_int32x8 temp_i32[4];

			v_as_floating_Invoker() {}
			void operator () (const v_int8x32& src, v_float32x8& dst_0, v_float32x8& dst_1, v_float32x8& dst_2, v_float32x8& dst_3) noexcept
			{
				v_expand(src, temp_i16[0], temp_i16[1]);
				v_expand(temp_i16[0], temp_i32[0], temp_i32[1]);
				v_expand(temp_i16[1], temp_i32[2], temp_i32[3]);

				dst_0 = v_convert<v_float32x8>(temp_i32[0]);
				dst_1 = v_convert<v_float32x8>(temp_i32[1]);
				dst_2 = v_convert<v_float32x8>(temp_i32[2]);
				dst_3 = v_convert<v_float32x8>(temp_i32[3]);
			}
		};

		template<> struct v_as_floating_Invoker<v_int16x16>
		{
			alignas(32) v_int32x8 temp_i32[2];

			v_as_floating_Invoker() {}
			void operator () (const v_int16x16& src, v_float32x8& dst_0, v_float32x8& dst_1) noexcept
			{
				v_expand(src, temp_i32[0], temp_i32[1]);

				dst_0 = v_convert<v_float32x8>(temp_i32[0]);
				dst_1 = v_convert<v_float32x8>(temp_i32[1]);
			}
		};


		template<> struct v_as_floating_Invoker<v_int32x8>
		{
			v_as_floating_Invoker() {}
			void operator () (const v_int32x8& src, v_float32x8& dst_0) noexcept
			{
				dst_0 = v_convert<v_float32x8>(src);
			}
		};

		template<> struct v_as_floating_Invoker<v_int64x4>
		{
			v_as_floating_Invoker() {}
			void operator () (const v_int64x4& src, v_float64x4& dst_0) noexcept
			{
				dst_0 = v_convert<v_float64x4>(src);
			}
		};

		template<VectorType Src_t, Floating_VectorType... Dst_t>
		void v_as_floating(const Src_t& src, Dst_t&... rest)
		{
			using First_t = std::tuple_element_t<0, std::tuple<Dst_t...>>;
			static_assert((... && std::is_same_v<First_t, Dst_t>), "All vector scalar types must match the input scalar type");

			constexpr size_t total_dst_elements = (... + Dst_t::batch_size);
			static_assert(Src_t::batch_size == total_dst_elements, "Sum of all floating batch_size must eq to src");

			v_as_floating_Invoker<Src_t> invoker;
			invoker(src, rest...);
		}
	}
}

export namespace fy
{
	namespace simd
	{
		template<VectorType Dst_t, VectorType Src_t> struct v_merge_Invoker
		{
			static_assert(false, "Not supported yet");
			template<typename ... Args> v_merge_Invoker(Args&& ...) {}
			void operator () () const { std::unreachable(); }
		};

		template<> struct v_merge_Invoker<v_uint8x32, v_uint16x16>
		{
			const v_uint16x16& src_0;
			const v_uint16x16& src_1;

			v_merge_Invoker(const v_uint16x16& src_0_, const v_uint16x16& src_1_) noexcept : src_0(src_0_), src_1(src_1_) {}
			v_uint8x32 operator () () const noexcept
			{
				__m256i result = _mm256_min_epu16(src_0.data, _mm256_set1_epi16(255));
				result = _mm256_packus_epi16(result, src_1.data);

				result = _mm256_permute4x64_epi64(result, _MM_SHUFFLE(3, 1, 2, 0));
				return v_uint8x32(result);
			}
		};

		template<> struct v_merge_Invoker<v_uint8x32, v_uint32x8>
		{
			const v_uint32x8& src_0;
			const v_uint32x8& src_1;
			const v_uint32x8& src_2;
			const v_uint32x8& src_3;

			v_merge_Invoker(const v_uint32x8& src_0_, const v_uint32x8& src_1_, const v_uint32x8& src_2_, const v_uint32x8& src_3_) noexcept
				: src_0(src_0_), src_1(src_1_), src_2(src_2_), src_3(src_3_)
			{
			}

			v_uint8x32 operator () () const noexcept
			{
				static const __m256i mask = _mm256_set_epi32(7, 3, 6, 2, 5, 1, 4, 0);
				__m256i result0 = _mm256_packus_epi32(src_0.data, src_1.data);
				__m256i result1 = _mm256_packus_epi32(src_2.data, src_3.data);
				__m256i result = _mm256_packus_epi16(result0, result1);
				result = _mm256_permutevar8x32_epi32(result, mask);
				return v_uint8x32(result);
			}
		};

		template<> struct v_merge_Invoker<v_uint8x32, v_uint64x4>
		{
			const v_uint64x4& src_0;
			const v_uint64x4& src_1;
			const v_uint64x4& src_2;
			const v_uint64x4& src_3;
			const v_uint64x4& src_4;
			const v_uint64x4& src_5;
			const v_uint64x4& src_6;
			const v_uint64x4& src_7;

			v_merge_Invoker(const v_uint64x4& v0, const v_uint64x4& v1, const v_uint64x4& v2, const v_uint64x4& v3,
				const v_uint64x4& v4, const v_uint64x4& v5, const v_uint64x4& v6, const v_uint64x4& v7) noexcept
				: src_0(v0), src_1(v1), src_2(v2), src_3(v3), src_4(v4), src_5(v5), src_6(v6), src_7(v7)
			{
			}

			v_uint8x32 operator () () const noexcept
			{
				static const __m256i mask = _mm256_set_epi32(7, 6, 3, 2, 5, 4, 1, 0);

				__m256i pack32_0 = _mm256_permute4x64_epi64(_mm256_packus_epi32(src_0.data, src_1.data), 0xD8);
				__m256i pack32_1 = _mm256_permute4x64_epi64(_mm256_packus_epi32(src_2.data, src_3.data), 0xD8);
				__m256i pack32_2 = _mm256_permute4x64_epi64(_mm256_packus_epi32(src_4.data, src_5.data), 0xD8);
				__m256i pack32_3 = _mm256_permute4x64_epi64(_mm256_packus_epi32(src_6.data, src_7.data), 0xD8);

				__m256i pack16_0 = _mm256_permute4x64_epi64(_mm256_packus_epi32(pack32_0, pack32_1), 0xD8);
				__m256i pack16_1 = _mm256_permute4x64_epi64(_mm256_packus_epi32(pack32_2, pack32_3), 0xD8);

				__m256i result_ = _mm256_packus_epi16(pack16_0, pack16_1);

				__m256i result = _mm256_permutevar8x32_epi32(result_, mask);
				return v_uint8x32(result);
			}
		};

		template<> struct v_merge_Invoker<v_uint8x32, v_float16x16>
		{
			const v_float16x16& src_0;
			const v_float16x16& src_1;

			v_merge_Invoker(const v_float16x16& src_0_, const v_float16x16& src_1_) noexcept : src_0(src_0_), src_1(src_1_) {}

			v_uint8x32 operator () () const noexcept
			{
				static const __m256i perm_mask = _mm256_set_epi32(7, 3, 6, 2, 5, 1, 4, 0);

				__m256 float32_0 = _mm256_cvtph_ps(_mm256_castsi256_si128(src_0.data));
				__m256 float32_1 = _mm256_cvtph_ps(_mm256_extracti128_si256(src_0.data, 1));
				__m256 float32_2 = _mm256_cvtph_ps(_mm256_castsi256_si128(src_1.data));
				__m256 float32_3 = _mm256_cvtph_ps(_mm256_extracti128_si256(src_1.data, 1));

				__m256i int32_0 = _mm256_cvtps_epi32(_mm256_round_ps(float32_0, _MM_FROUND_TO_NEAREST_INT));
				__m256i int32_1 = _mm256_cvtps_epi32(_mm256_round_ps(float32_1, _MM_FROUND_TO_NEAREST_INT));
				__m256i int32_2 = _mm256_cvtps_epi32(_mm256_round_ps(float32_2, _MM_FROUND_TO_NEAREST_INT));
				__m256i int32_3 = _mm256_cvtps_epi32(_mm256_round_ps(float32_3, _MM_FROUND_TO_NEAREST_INT));

				__m256i int16_0 = _mm256_packs_epi32(int32_0, int32_1);
				__m256i int16_1 = _mm256_packs_epi32(int32_2, int32_3);

				__m256i result = _mm256_packus_epi16(int16_0, int16_1);

				__m256i uint8_result = _mm256_permutevar8x32_epi32(result, perm_mask);
				return v_uint8x32(uint8_result);
			}
		};

		template<> struct v_merge_Invoker<v_uint8x32, v_float32x8>
		{
			const v_float32x8& src_0;
			const v_float32x8& src_1;
			const v_float32x8& src_2;
			const v_float32x8& src_3;

			v_merge_Invoker(const v_float32x8& src_0_, const v_float32x8& src_1_, const v_float32x8& src_2_, const v_float32x8& src_3_) noexcept
				: src_0(src_0_), src_1(src_1_), src_2(src_2_), src_3(src_3_)
			{
			}

			v_uint8x32 operator () () const noexcept
			{
				static const __m256i perm_mask = _mm256_set_epi32(7, 3, 6, 2, 5, 1, 4, 0);

				__m256i int32_0 = _mm256_cvtps_epi32(_mm256_round_ps(src_0.data, _MM_FROUND_TO_NEAREST_INT));
				__m256i int32_1 = _mm256_cvtps_epi32(_mm256_round_ps(src_1.data, _MM_FROUND_TO_NEAREST_INT));
				__m256i int32_2 = _mm256_cvtps_epi32(_mm256_round_ps(src_2.data, _MM_FROUND_TO_NEAREST_INT));
				__m256i int32_3 = _mm256_cvtps_epi32(_mm256_round_ps(src_3.data, _MM_FROUND_TO_NEAREST_INT));

				__m256i int16_0 = _mm256_packs_epi32(int32_0, int32_1);
				__m256i int16_1 = _mm256_packs_epi32(int32_2, int32_3);

				__m256i uint8_result = _mm256_packus_epi16(int16_0, int16_1);

				uint8_result = _mm256_permutevar8x32_epi32(uint8_result, perm_mask);
				return v_uint8x32(uint8_result);
			}
		};

		template<> struct v_merge_Invoker<v_uint8x32, v_float64x4>
		{
			const v_float64x4& src_0;
			const v_float64x4& src_1;
			const v_float64x4& src_2;
			const v_float64x4& src_3;
			const v_float64x4& src_4;
			const v_float64x4& src_5;
			const v_float64x4& src_6;
			const v_float64x4& src_7;

			v_merge_Invoker(
				const v_float64x4& src_0_, const v_float64x4& src_1_, const v_float64x4& src_2_, const v_float64x4& src_3_,
				const v_float64x4& src_4_, const v_float64x4& src_5_, const v_float64x4& src_6_, const v_float64x4& src_7_) noexcept
				: src_0(src_0_), src_1(src_1_), src_2(src_2_), src_3(src_3_), src_4(src_4_), src_5(src_5_), src_6(src_6_), src_7(src_7_)
			{
			}

			v_uint8x32 operator () () const noexcept
			{
				static const __m256i perm_mask = _mm256_set_epi32(7, 3, 6, 2, 5, 1, 4, 0);

				__m128i int32_0 = _mm256_cvtpd_epi32(_mm256_round_pd(src_0.data, _MM_FROUND_TO_NEAREST_INT));
				__m128i int32_1 = _mm256_cvtpd_epi32(_mm256_round_pd(src_1.data, _MM_FROUND_TO_NEAREST_INT));
				__m128i int32_2 = _mm256_cvtpd_epi32(_mm256_round_pd(src_2.data, _MM_FROUND_TO_NEAREST_INT));
				__m128i int32_3 = _mm256_cvtpd_epi32(_mm256_round_pd(src_3.data, _MM_FROUND_TO_NEAREST_INT));
				__m128i int32_4 = _mm256_cvtpd_epi32(_mm256_round_pd(src_4.data, _MM_FROUND_TO_NEAREST_INT));
				__m128i int32_5 = _mm256_cvtpd_epi32(_mm256_round_pd(src_5.data, _MM_FROUND_TO_NEAREST_INT));
				__m128i int32_6 = _mm256_cvtpd_epi32(_mm256_round_pd(src_6.data, _MM_FROUND_TO_NEAREST_INT));
				__m128i int32_7 = _mm256_cvtpd_epi32(_mm256_round_pd(src_7.data, _MM_FROUND_TO_NEAREST_INT));

				__m256i int32_01 = _mm256_insertf128_si256(_mm256_castsi128_si256(int32_0), int32_1, 1);
				__m256i int32_23 = _mm256_insertf128_si256(_mm256_castsi128_si256(int32_2), int32_3, 1);
				__m256i int32_45 = _mm256_insertf128_si256(_mm256_castsi128_si256(int32_4), int32_5, 1);
				__m256i int32_67 = _mm256_insertf128_si256(_mm256_castsi128_si256(int32_6), int32_7, 1);

				__m256i int16_0123 = _mm256_packs_epi32(int32_01, int32_23);
				__m256i int16_4567 = _mm256_packs_epi32(int32_45, int32_67);

				__m256i uint8_result = _mm256_packus_epi16(int16_0123, int16_4567);

				uint8_result = _mm256_permutevar8x32_epi32(uint8_result, perm_mask);
				return v_uint8x32(uint8_result);
			}
		};

		template<> struct v_merge_Invoker<v_uint8x32, v_uint8x16>
		{
			const v_uint8x16& src_0;
			const v_uint8x16& src_1;

			v_merge_Invoker(const v_uint8x16& src_0_, const v_uint8x16& src_1_) noexcept : src_0(src_0_), src_1(src_1_) {}
			v_uint8x32 operator () () const noexcept
			{
				return v_uint8x32(_mm256_inserti128_si256(_mm256_castsi128_si256(src_0.data), src_1.data, 1));
			}
		};

		template<> struct v_merge_Invoker<v_uint8x32, v_uint16x8>
		{
			const v_uint16x8& src_0;
			const v_uint16x8& src_1;
			const v_uint16x8& src_2;
			const v_uint16x8& src_3;

			v_merge_Invoker(const v_uint16x8& src_0_, const v_uint16x8& src_1_, const v_uint16x8& src_2_, const v_uint16x8& src_3_)
				noexcept : src_0(src_0_), src_1(src_1_), src_2(src_2_), src_3(src_3_)
			{
			}
			v_uint8x32 operator () () const noexcept
			{
				static const __m256i perm_mask = _mm256_set_epi32(7, 6, 3, 2, 5, 4, 1, 0);

				__m256i pack16_01 = _mm256_inserti128_si256(_mm256_castsi128_si256(src_0.data), src_1.data, 1);
				__m256i pack16_23 = _mm256_inserti128_si256(_mm256_castsi128_si256(src_2.data), src_3.data, 1);

				__m256i result_ = _mm256_packus_epi16(pack16_01, pack16_23);
				__m256i result = _mm256_permutevar8x32_epi32(result_, perm_mask);
				return v_uint8x32(result);
			}
		};

		template<> struct v_merge_Invoker<v_uint8x32, v_uint32x4>
		{
			const v_uint32x4& src_0;
			const v_uint32x4& src_1;
			const v_uint32x4& src_2;
			const v_uint32x4& src_3;
			const v_uint32x4& src_4;
			const v_uint32x4& src_5;
			const v_uint32x4& src_6;
			const v_uint32x4& src_7;

			v_merge_Invoker(const v_uint32x4& src_0_, const v_uint32x4& src_1_, const v_uint32x4& src_2_, const v_uint32x4& src_3_,
				const v_uint32x4& src_4_, const v_uint32x4& src_5_, const v_uint32x4& src_6_, const v_uint32x4& src_7_)
				noexcept : src_0(src_0_), src_1(src_1_), src_2(src_2_), src_3(src_3_), src_4(src_4_), src_5(src_5_), src_6(src_6_), src_7(src_7_)
			{
			}

			v_uint8x32 operator () () const noexcept
			{
				static const __m256i perm_mask = _mm256_set_epi32(7, 3, 6, 2, 5, 1, 4, 0);

				__m256i pack32_01 = _mm256_inserti128_si256(_mm256_castsi128_si256(src_0.data), src_1.data, 1);
				__m256i pack32_23 = _mm256_inserti128_si256(_mm256_castsi128_si256(src_2.data), src_3.data, 1);
				__m256i pack32_45 = _mm256_inserti128_si256(_mm256_castsi128_si256(src_4.data), src_5.data, 1);
				__m256i pack32_67 = _mm256_inserti128_si256(_mm256_castsi128_si256(src_6.data), src_7.data, 1);

				__m256i pack16_0123 = _mm256_packus_epi32(pack32_01, pack32_23);
				__m256i pack16_4567 = _mm256_packus_epi32(pack32_45, pack32_67);

				__m256i result = _mm256_packus_epi16(pack16_0123, pack16_4567);

				result = _mm256_permutevar8x32_epi32(result, perm_mask);
				return v_uint8x32(result);
			}
		};

		template<> struct v_merge_Invoker<v_uint8x32, v_uint64x2>
		{
			const v_uint64x2& src_0;
			const v_uint64x2& src_1;
			const v_uint64x2& src_2;
			const v_uint64x2& src_3;
			const v_uint64x2& src_4;
			const v_uint64x2& src_5;
			const v_uint64x2& src_6;
			const v_uint64x2& src_7;
			const v_uint64x2& src_8;
			const v_uint64x2& src_9;
			const v_uint64x2& src_10;
			const v_uint64x2& src_11;
			const v_uint64x2& src_12;
			const v_uint64x2& src_13;
			const v_uint64x2& src_14;
			const v_uint64x2& src_15;

			v_merge_Invoker(
				const v_uint64x2& src_0_, const v_uint64x2& src_1_, const v_uint64x2& src_2_, const v_uint64x2& src_3_,
				const v_uint64x2& src_4_, const v_uint64x2& src_5_, const v_uint64x2& src_6_, const v_uint64x2& src_7_,
				const v_uint64x2& src_8_, const v_uint64x2& src_9_, const v_uint64x2& src_10_, const v_uint64x2& src_11_,
				const v_uint64x2& src_12_, const v_uint64x2& src_13_, const v_uint64x2& src_14_, const v_uint64x2& src_15_) noexcept
				: src_0(src_0_), src_1(src_1_), src_2(src_2_), src_3(src_3_), src_4(src_4_), src_5(src_5_), src_6(src_6_), src_7(src_7_),
				src_8(src_8_), src_9(src_9_), src_10(src_10_), src_11(src_11_), src_12(src_12_), src_13(src_13_), src_14(src_14_), src_15(src_15_)
			{
			}

			v_uint8x32 operator () () const noexcept
			{
				static const __m256i perm_mask = _mm256_set_epi32(7, 6, 3, 2, 5, 4, 1, 0);

				__m128i pack32_0 = _mm_packus_epi32(src_0.data, src_1.data);
				__m128i pack32_1 = _mm_packus_epi32(src_2.data, src_3.data);
				__m128i pack32_2 = _mm_packus_epi32(src_4.data, src_5.data);
				__m128i pack32_3 = _mm_packus_epi32(src_6.data, src_7.data);
				__m128i pack32_4 = _mm_packus_epi32(src_8.data, src_9.data);
				__m128i pack32_5 = _mm_packus_epi32(src_10.data, src_11.data);
				__m128i pack32_6 = _mm_packus_epi32(src_12.data, src_13.data);
				__m128i pack32_7 = _mm_packus_epi32(src_14.data, src_15.data);

				__m128i pack16_0 = _mm_packus_epi16(pack32_0, pack32_1);
				__m128i pack16_1 = _mm_packus_epi16(pack32_2, pack32_3);
				__m128i pack16_2 = _mm_packus_epi16(pack32_4, pack32_5);
				__m128i pack16_3 = _mm_packus_epi16(pack32_6, pack32_7);

				__m256i pack16_01 = _mm256_inserti128_si256(_mm256_castsi128_si256(pack16_0), pack16_1, 1);
				__m256i pack16_23 = _mm256_inserti128_si256(_mm256_castsi128_si256(pack16_2), pack16_3, 1);

				__m256i result = _mm256_packus_epi16(pack16_01, pack16_23);

				result = _mm256_permutevar8x32_epi32(result, perm_mask);
				return v_uint8x32(result);
			}
		};

		template<> struct v_merge_Invoker<v_uint8x32, v_int16x16>
		{
			const v_int16x16& src_0;
			const v_int16x16& src_1;

			v_merge_Invoker(const v_int16x16& src_0_, const v_int16x16& src_1_) noexcept : src_0(src_0_), src_1(src_1_) {}
			v_uint8x32 operator () () const noexcept
			{
				static const __m256i zero = _mm256_setzero_si256();
				__m256i src_0_pos = _mm256_max_epi16(src_0.data, zero);
				__m256i src_1_pos = _mm256_max_epi16(src_1.data, zero);

				__m256i result = _mm256_packus_epi16(src_0_pos, src_1_pos);

				result = _mm256_permute4x64_epi64(result, _MM_SHUFFLE(3, 1, 2, 0));
				return v_uint8x32(result);
			}
		};

		template<> struct v_merge_Invoker<v_uint8x32, v_int32x8>
		{
			const v_int32x8& src_0;
			const v_int32x8& src_1;
			const v_int32x8& src_2;
			const v_int32x8& src_3;

			v_merge_Invoker(const v_int32x8& src_0_, const v_int32x8& src_1_, const v_int32x8& src_2_, const v_int32x8& src_3_) noexcept
				: src_0(src_0_), src_1(src_1_), src_2(src_2_), src_3(src_3_)
			{
			}

			v_uint8x32 operator () () const noexcept
			{
				static const __m256i mask = _mm256_set_epi32(7, 3, 6, 2, 5, 1, 4, 0);
				static const __m256i zero = _mm256_setzero_si256();

				__m256i src_0_pos = _mm256_max_epi32(src_0.data, zero);
				__m256i src_1_pos = _mm256_max_epi32(src_1.data, zero);
				__m256i src_2_pos = _mm256_max_epi32(src_2.data, zero);
				__m256i src_3_pos = _mm256_max_epi32(src_3.data, zero);

				__m256i result0 = _mm256_packus_epi32(src_0_pos, src_1_pos);
				__m256i result1 = _mm256_packus_epi32(src_2_pos, src_3_pos);

				__m256i result = _mm256_packus_epi16(result0, result1);

				result = _mm256_permutevar8x32_epi32(result, mask);
				return v_uint8x32(result);
			}
		};

		template<> struct v_merge_Invoker<v_uint8x32, v_int64x4>
		{
			const v_int64x4& src_0;
			const v_int64x4& src_1;
			const v_int64x4& src_2;
			const v_int64x4& src_3;
			const v_int64x4& src_4;
			const v_int64x4& src_5;
			const v_int64x4& src_6;
			const v_int64x4& src_7;

			v_merge_Invoker(const v_int64x4& v0, const v_int64x4& v1, const v_int64x4& v2, const v_int64x4& v3,
				const v_int64x4& v4, const v_int64x4& v5, const v_int64x4& v6, const v_int64x4& v7) noexcept
				: src_0(v0), src_1(v1), src_2(v2), src_3(v3), src_4(v4), src_5(v5), src_6(v6), src_7(v7)
			{
			}

			v_uint8x32 operator () () const noexcept
			{
				static auto clamp_and_truncate = [](__m256i x) -> __m128i {
					static const __m256i zero = _mm256_setzero_si256();
					static const __m256i F = _mm256_set1_epi32(255);

					__m256i pos = _mm256_max_epi32(x, zero);
					__m256i clamped = _mm256_min_epu32(pos, F);
					__m128i low = _mm256_castsi256_si128(clamped);
					__m128i high = _mm256_extracti128_si256(clamped, 1);
					return _mm_packus_epi32(low, high);
				};

				__m128i pack16_0 = clamp_and_truncate(src_0.data);
				__m128i pack16_1 = clamp_and_truncate(src_1.data);
				__m128i pack16_2 = clamp_and_truncate(src_2.data);
				__m128i pack16_3 = clamp_and_truncate(src_3.data);
				__m128i pack16_4 = clamp_and_truncate(src_4.data);
				__m128i pack16_5 = clamp_and_truncate(src_5.data);
				__m128i pack16_6 = clamp_and_truncate(src_6.data);
				__m128i pack16_7 = clamp_and_truncate(src_7.data);

				__m256i pack8_0 = _mm256_setr_m128i(_mm_packus_epi16(pack16_0, pack16_1), _mm_packus_epi16(pack16_2, pack16_3));
				__m256i pack8_1 = _mm256_setr_m128i(_mm_packus_epi16(pack16_4, pack16_5), _mm_packus_epi16(pack16_6, pack16_7));

				__m256i result = _mm256_permute4x64_epi64(_mm256_packus_epi16(pack8_0, pack8_1), 0xD8);
				return v_uint8x32(result);
			}
		};

		template<> struct v_merge_Invoker<v_uint8x32, v_float16x8>
		{
			const v_float16x8& src_0;
			const v_float16x8& src_1;
			const v_float16x8& src_2;
			const v_float16x8& src_3;

			v_merge_Invoker(const v_float16x8& src_0_, const v_float16x8& src_1_, const v_float16x8& src_2_, const v_float16x8& src_3_)
				noexcept : src_0(src_0_), src_1(src_1_), src_2(src_2_), src_3(src_3_)
			{
			}
			v_uint8x32 operator () () const noexcept
			{
				static const __m256i perm_mask = _mm256_set_epi32(7, 3, 6, 2, 5, 1, 4, 0);

				__m256 float32_0 = _mm256_cvtph_ps(src_0.data);
				__m256 float32_1 = _mm256_cvtph_ps(src_1.data);
				__m256 float32_2 = _mm256_cvtph_ps(src_2.data);
				__m256 float32_3 = _mm256_cvtph_ps(src_3.data);

				__m256i int32_0 = _mm256_cvtps_epi32(_mm256_round_ps(float32_0, _MM_FROUND_TO_NEAREST_INT));
				__m256i int32_1 = _mm256_cvtps_epi32(_mm256_round_ps(float32_1, _MM_FROUND_TO_NEAREST_INT));
				__m256i int32_2 = _mm256_cvtps_epi32(_mm256_round_ps(float32_2, _MM_FROUND_TO_NEAREST_INT));
				__m256i int32_3 = _mm256_cvtps_epi32(_mm256_round_ps(float32_3, _MM_FROUND_TO_NEAREST_INT));

				__m256i int16_0 = _mm256_packs_epi32(int32_0, int32_1);
				__m256i int16_1 = _mm256_packs_epi32(int32_2, int32_3);

				__m256i uint8_result = _mm256_packus_epi16(int16_0, int16_1);
				uint8_result = _mm256_permutevar8x32_epi32(uint8_result, perm_mask);
				return v_uint8x32(uint8_result);
			}
		};

		template<> struct v_merge_Invoker<v_uint8x32, v_float32x4>
		{
			const v_float32x4& src_0;
			const v_float32x4& src_1;
			const v_float32x4& src_2;
			const v_float32x4& src_3;
			const v_float32x4& src_4;
			const v_float32x4& src_5;
			const v_float32x4& src_6;
			const v_float32x4& src_7;

			v_merge_Invoker(
				const v_float32x4& src_0_, const v_float32x4& src_1_, const v_float32x4& src_2_, const v_float32x4& src_3_,
				const v_float32x4& src_4_, const v_float32x4& src_5_, const v_float32x4& src_6_, const v_float32x4& src_7_)
				noexcept : src_0(src_0_), src_1(src_1_), src_2(src_2_), src_3(src_3_), src_4(src_4_), src_5(src_5_), src_6(src_6_), src_7(src_7_)
			{
			}

			v_uint8x32 operator () () const noexcept
			{
				__m128i int32_0 = _mm_cvtps_epi32(_mm_round_ps(src_0.data, _MM_FROUND_TO_NEAREST_INT));
				__m128i int32_1 = _mm_cvtps_epi32(_mm_round_ps(src_1.data, _MM_FROUND_TO_NEAREST_INT));
				__m128i int32_2 = _mm_cvtps_epi32(_mm_round_ps(src_2.data, _MM_FROUND_TO_NEAREST_INT));
				__m128i int32_3 = _mm_cvtps_epi32(_mm_round_ps(src_3.data, _MM_FROUND_TO_NEAREST_INT));
				__m128i int32_4 = _mm_cvtps_epi32(_mm_round_ps(src_4.data, _MM_FROUND_TO_NEAREST_INT));
				__m128i int32_5 = _mm_cvtps_epi32(_mm_round_ps(src_5.data, _MM_FROUND_TO_NEAREST_INT));
				__m128i int32_6 = _mm_cvtps_epi32(_mm_round_ps(src_6.data, _MM_FROUND_TO_NEAREST_INT));
				__m128i int32_7 = _mm_cvtps_epi32(_mm_round_ps(src_7.data, _MM_FROUND_TO_NEAREST_INT));

				__m128i int16_0 = _mm_packs_epi32(int32_0, int32_1);
				__m128i int16_1 = _mm_packs_epi32(int32_2, int32_3);
				__m128i int16_2 = _mm_packs_epi32(int32_4, int32_5);
				__m128i int16_3 = _mm_packs_epi32(int32_6, int32_7);

				__m128i uint8_0 = _mm_packus_epi16(int16_0, int16_1);
				__m128i uint8_1 = _mm_packus_epi16(int16_2, int16_3);

				__m256i uint8_result = _mm256_insertf128_si256(_mm256_castsi128_si256(uint8_0), uint8_1, 1);
				return v_uint8x32(uint8_result);
			}
		};

		template<> struct v_merge_Invoker<v_uint8x32, v_float64x2>
		{
			const v_float64x2& src_0;
			const v_float64x2& src_1;
			const v_float64x2& src_2;
			const v_float64x2& src_3;
			const v_float64x2& src_4;
			const v_float64x2& src_5;
			const v_float64x2& src_6;
			const v_float64x2& src_7;
			const v_float64x2& src_8;
			const v_float64x2& src_9;
			const v_float64x2& src_10;
			const v_float64x2& src_11;
			const v_float64x2& src_12;
			const v_float64x2& src_13;
			const v_float64x2& src_14;
			const v_float64x2& src_15;

			v_merge_Invoker(
				const v_float64x2& src_0_, const v_float64x2& src_1_, const v_float64x2& src_2_, const v_float64x2& src_3_,
				const v_float64x2& src_4_, const v_float64x2& src_5_, const v_float64x2& src_6_, const v_float64x2& src_7_,
				const v_float64x2& src_8_, const v_float64x2& src_9_, const v_float64x2& src_10_, const v_float64x2& src_11_,
				const v_float64x2& src_12_, const v_float64x2& src_13_, const v_float64x2& src_14_, const v_float64x2& src_15_)
				noexcept : src_0(src_0_), src_1(src_1_), src_2(src_2_), src_3(src_3_),
				src_4(src_4_), src_5(src_5_), src_6(src_6_), src_7(src_7_),
				src_8(src_8_), src_9(src_9_), src_10(src_10_), src_11(src_11_),
				src_12(src_12_), src_13(src_13_), src_14(src_14_), src_15(src_15_)
			{
			}

			v_uint8x32 operator () () const noexcept
			{
				__m128i int32_0 = _mm_cvtpd_epi32(_mm_round_pd(src_0.data, _MM_FROUND_TO_NEAREST_INT));
				__m128i int32_1 = _mm_cvtpd_epi32(_mm_round_pd(src_1.data, _MM_FROUND_TO_NEAREST_INT));
				__m128i int32_2 = _mm_cvtpd_epi32(_mm_round_pd(src_2.data, _MM_FROUND_TO_NEAREST_INT));
				__m128i int32_3 = _mm_cvtpd_epi32(_mm_round_pd(src_3.data, _MM_FROUND_TO_NEAREST_INT));
				__m128i int32_4 = _mm_cvtpd_epi32(_mm_round_pd(src_4.data, _MM_FROUND_TO_NEAREST_INT));
				__m128i int32_5 = _mm_cvtpd_epi32(_mm_round_pd(src_5.data, _MM_FROUND_TO_NEAREST_INT));
				__m128i int32_6 = _mm_cvtpd_epi32(_mm_round_pd(src_6.data, _MM_FROUND_TO_NEAREST_INT));
				__m128i int32_7 = _mm_cvtpd_epi32(_mm_round_pd(src_7.data, _MM_FROUND_TO_NEAREST_INT));
				__m128i int32_8 = _mm_cvtpd_epi32(_mm_round_pd(src_8.data, _MM_FROUND_TO_NEAREST_INT));
				__m128i int32_9 = _mm_cvtpd_epi32(_mm_round_pd(src_9.data, _MM_FROUND_TO_NEAREST_INT));
				__m128i int32_10 = _mm_cvtpd_epi32(_mm_round_pd(src_10.data, _MM_FROUND_TO_NEAREST_INT));
				__m128i int32_11 = _mm_cvtpd_epi32(_mm_round_pd(src_11.data, _MM_FROUND_TO_NEAREST_INT));
				__m128i int32_12 = _mm_cvtpd_epi32(_mm_round_pd(src_12.data, _MM_FROUND_TO_NEAREST_INT));
				__m128i int32_13 = _mm_cvtpd_epi32(_mm_round_pd(src_13.data, _MM_FROUND_TO_NEAREST_INT));
				__m128i int32_14 = _mm_cvtpd_epi32(_mm_round_pd(src_14.data, _MM_FROUND_TO_NEAREST_INT));
				__m128i int32_15 = _mm_cvtpd_epi32(_mm_round_pd(src_15.data, _MM_FROUND_TO_NEAREST_INT));

				__m128i int16_0 = _mm_packs_epi32(_mm_unpacklo_epi64(int32_0, int32_1), _mm_unpacklo_epi64(int32_2, int32_3));
				__m128i int16_1 = _mm_packs_epi32(_mm_unpacklo_epi64(int32_4, int32_5), _mm_unpacklo_epi64(int32_6, int32_7));
				__m128i int16_2 = _mm_packs_epi32(_mm_unpacklo_epi64(int32_8, int32_9), _mm_unpacklo_epi64(int32_10, int32_11));
				__m128i int16_3 = _mm_packs_epi32(_mm_unpacklo_epi64(int32_12, int32_13), _mm_unpacklo_epi64(int32_14, int32_15));

				__m128i uint8_0 = _mm_packus_epi16(int16_0, int16_1);
				__m128i uint8_1 = _mm_packus_epi16(int16_2, int16_3);

				__m256i uint8_result = _mm256_insertf128_si256(_mm256_castsi128_si256(uint8_0), uint8_1, 1);
				return v_uint8x32(uint8_result);
			}
		};

		template<> struct v_merge_Invoker<v_int8x32, v_uint16x16>
		{
			const v_uint16x16& src_0;
			const v_uint16x16& src_1;

			v_merge_Invoker(const v_uint16x16& src_0_, const v_uint16x16& src_1_) noexcept : src_0(src_0_), src_1(src_1_) {}
			v_int8x32 operator () () const noexcept
			{
				__m256i result = _mm256_packs_epi16(src_0.data, src_1.data);
				result = _mm256_permute4x64_epi64(result, _MM_SHUFFLE(3, 1, 2, 0));
				return v_int8x32(result);
			}
		};

		template<> struct v_merge_Invoker<v_int8x32, v_uint32x8>
		{
			const v_uint32x8& src_0;
			const v_uint32x8& src_1;
			const v_uint32x8& src_2;
			const v_uint32x8& src_3;

			v_merge_Invoker(const v_uint32x8& src_0_, const v_uint32x8& src_1_, const v_uint32x8& src_2_, const v_uint32x8& src_3_) noexcept
				: src_0(src_0_), src_1(src_1_), src_2(src_2_), src_3(src_3_)
			{
			}

			v_int8x32 operator () () const noexcept
			{
				__m256i pack1 = _mm256_packs_epi32(src_0.data, src_1.data);
				__m256i pack2 = _mm256_packs_epi32(src_2.data, src_3.data);

				pack1 = _mm256_permute4x64_epi64(pack1, _MM_SHUFFLE(3, 1, 2, 0));
				pack2 = _mm256_permute4x64_epi64(pack2, _MM_SHUFFLE(3, 1, 2, 0));

				__m256i result = _mm256_packs_epi16(pack1, pack2);

				result = _mm256_permute4x64_epi64(result, _MM_SHUFFLE(3, 1, 2, 0));

				return v_int8x32(result);
			}
		};

		template<> struct v_merge_Invoker<v_int8x32, v_uint64x4>
		{
			const v_uint64x4& src_0;
			const v_uint64x4& src_1;
			const v_uint64x4& src_2;
			const v_uint64x4& src_3;
			const v_uint64x4& src_4;
			const v_uint64x4& src_5;
			const v_uint64x4& src_6;
			const v_uint64x4& src_7;

			v_merge_Invoker(const v_uint64x4& v0, const v_uint64x4& v1, const v_uint64x4& v2, const v_uint64x4& v3,
				const v_uint64x4& v4, const v_uint64x4& v5, const v_uint64x4& v6, const v_uint64x4& v7) noexcept
				: src_0(v0), src_1(v1), src_2(v2), src_3(v3), src_4(v4), src_5(v5), src_6(v6), src_7(v7)
			{
			}

			v_int8x32 operator () () const noexcept
			{
				static const __m256i mask = _mm256_set_epi32(7, 6, 3, 2, 5, 4, 1, 0);

				__m256i pack32_0 = _mm256_permute4x64_epi64(_mm256_packs_epi32(src_0.data, src_1.data), 0xD8);
				__m256i pack32_1 = _mm256_permute4x64_epi64(_mm256_packs_epi32(src_2.data, src_3.data), 0xD8);
				__m256i pack32_2 = _mm256_permute4x64_epi64(_mm256_packs_epi32(src_4.data, src_5.data), 0xD8);
				__m256i pack32_3 = _mm256_permute4x64_epi64(_mm256_packs_epi32(src_6.data, src_7.data), 0xD8);

				__m256i pack16_0 = _mm256_permute4x64_epi64(_mm256_packs_epi32(pack32_0, pack32_1), 0xD8);
				__m256i pack16_1 = _mm256_permute4x64_epi64(_mm256_packs_epi32(pack32_2, pack32_3), 0xD8);

				__m256i result_ = _mm256_packs_epi16(pack16_0, pack16_1);

				__m256i result = _mm256_permutevar8x32_epi32(result_, mask);
				return v_int8x32(result);
			}
		};

		template<> struct v_merge_Invoker<v_int8x32, v_float16x16>
		{
			const v_float16x16& src_0;
			const v_float16x16& src_1;

			v_merge_Invoker(const v_float16x16& src_0_, const v_float16x16& src_1_) noexcept : src_0(src_0_), src_1(src_1_) {}

			v_int8x32 operator () () const noexcept
			{
				static const __m256i perm_mask = _mm256_set_epi32(7, 3, 6, 2, 5, 1, 4, 0);

				static const __m256 min_val = _mm256_set1_ps(-128.0f);
				static const __m256 max_val = _mm256_set1_ps(127.0f);

				__m256 float32_0 = _mm256_cvtph_ps(_mm256_castsi256_si128(src_0.data));
				__m256 float32_1 = _mm256_cvtph_ps(_mm256_extracti128_si256(src_0.data, 1));
				__m256 float32_2 = _mm256_cvtph_ps(_mm256_castsi256_si128(src_1.data));
				__m256 float32_3 = _mm256_cvtph_ps(_mm256_extracti128_si256(src_1.data, 1));

				float32_0 = _mm256_min_ps(_mm256_max_ps(float32_0, min_val), max_val);
				float32_1 = _mm256_min_ps(_mm256_max_ps(float32_1, min_val), max_val);
				float32_2 = _mm256_min_ps(_mm256_max_ps(float32_2, min_val), max_val);
				float32_3 = _mm256_min_ps(_mm256_max_ps(float32_3, min_val), max_val);

				__m256i int32_0 = _mm256_cvtps_epi32(_mm256_round_ps(float32_0, _MM_FROUND_TO_NEAREST_INT));
				__m256i int32_1 = _mm256_cvtps_epi32(_mm256_round_ps(float32_1, _MM_FROUND_TO_NEAREST_INT));
				__m256i int32_2 = _mm256_cvtps_epi32(_mm256_round_ps(float32_2, _MM_FROUND_TO_NEAREST_INT));
				__m256i int32_3 = _mm256_cvtps_epi32(_mm256_round_ps(float32_3, _MM_FROUND_TO_NEAREST_INT));

				__m256i int16_0 = _mm256_packs_epi32(int32_0, int32_1);
				__m256i int16_1 = _mm256_packs_epi32(int32_2, int32_3);

				__m256i result = _mm256_packs_epi16(int16_0, int16_1);

				__m256i int8_result = _mm256_permutevar8x32_epi32(result, perm_mask);
				return v_int8x32(int8_result);
			}
		};

		template<> struct v_merge_Invoker<v_int8x32, v_float32x8>
		{
			const v_float32x8& src_0;
			const v_float32x8& src_1;
			const v_float32x8& src_2;
			const v_float32x8& src_3;

			v_merge_Invoker(const v_float32x8& src_0_, const v_float32x8& src_1_, const v_float32x8& src_2_, const v_float32x8& src_3_) noexcept
				: src_0(src_0_), src_1(src_1_), src_2(src_2_), src_3(src_3_)
			{
			}

			v_int8x32 operator () () const noexcept
			{
				static const __m256i perm_mask = _mm256_set_epi32(7, 3, 6, 2, 5, 1, 4, 0);

				static const __m256 min_val = _mm256_set1_ps(-128.0f);
				static const __m256 max_val = _mm256_set1_ps(127.0f);

				__m256 clamped_0 = _mm256_min_ps(_mm256_max_ps(src_0.data, min_val), max_val);
				__m256 clamped_1 = _mm256_min_ps(_mm256_max_ps(src_1.data, min_val), max_val);
				__m256 clamped_2 = _mm256_min_ps(_mm256_max_ps(src_2.data, min_val), max_val);
				__m256 clamped_3 = _mm256_min_ps(_mm256_max_ps(src_3.data, min_val), max_val);

				__m256i int32_0 = _mm256_cvtps_epi32(_mm256_round_ps(clamped_0, _MM_FROUND_TO_NEAREST_INT));
				__m256i int32_1 = _mm256_cvtps_epi32(_mm256_round_ps(clamped_1, _MM_FROUND_TO_NEAREST_INT));
				__m256i int32_2 = _mm256_cvtps_epi32(_mm256_round_ps(clamped_2, _MM_FROUND_TO_NEAREST_INT));
				__m256i int32_3 = _mm256_cvtps_epi32(_mm256_round_ps(clamped_3, _MM_FROUND_TO_NEAREST_INT));

				__m256i int16_0 = _mm256_packs_epi32(int32_0, int32_1);
				__m256i int16_1 = _mm256_packs_epi32(int32_2, int32_3);

				__m256i int8_result = _mm256_packs_epi16(int16_0, int16_1);

				int8_result = _mm256_permutevar8x32_epi32(int8_result, perm_mask);
				return v_int8x32(int8_result);
			}
		};

		template<> struct v_merge_Invoker<v_int8x32, v_float64x4>
		{
			const v_float64x4& src_0;
			const v_float64x4& src_1;
			const v_float64x4& src_2;
			const v_float64x4& src_3;
			const v_float64x4& src_4;
			const v_float64x4& src_5;
			const v_float64x4& src_6;
			const v_float64x4& src_7;

			v_merge_Invoker(
				const v_float64x4& src_0_, const v_float64x4& src_1_, const v_float64x4& src_2_, const v_float64x4& src_3_,
				const v_float64x4& src_4_, const v_float64x4& src_5_, const v_float64x4& src_6_, const v_float64x4& src_7_) noexcept
				: src_0(src_0_), src_1(src_1_), src_2(src_2_), src_3(src_3_), src_4(src_4_), src_5(src_5_), src_6(src_6_), src_7(src_7_)
			{
			}

			v_int8x32 operator () () const noexcept
			{
				static const __m256i perm_mask = _mm256_set_epi32(7, 3, 6, 2, 5, 1, 4, 0);
				static const __m256d min_val = _mm256_set1_pd(-128.0);
				static const __m256d max_val = _mm256_set1_pd(127.0);

				__m256d clamped_0 = _mm256_min_pd(_mm256_max_pd(src_0.data, min_val), max_val);
				__m256d clamped_1 = _mm256_min_pd(_mm256_max_pd(src_1.data, min_val), max_val);
				__m256d clamped_2 = _mm256_min_pd(_mm256_max_pd(src_2.data, min_val), max_val);
				__m256d clamped_3 = _mm256_min_pd(_mm256_max_pd(src_3.data, min_val), max_val);
				__m256d clamped_4 = _mm256_min_pd(_mm256_max_pd(src_4.data, min_val), max_val);
				__m256d clamped_5 = _mm256_min_pd(_mm256_max_pd(src_5.data, min_val), max_val);
				__m256d clamped_6 = _mm256_min_pd(_mm256_max_pd(src_6.data, min_val), max_val);
				__m256d clamped_7 = _mm256_min_pd(_mm256_max_pd(src_7.data, min_val), max_val);

				__m128i int32_0 = _mm256_cvtpd_epi32(_mm256_round_pd(clamped_0, _MM_FROUND_TO_NEAREST_INT));
				__m128i int32_1 = _mm256_cvtpd_epi32(_mm256_round_pd(clamped_1, _MM_FROUND_TO_NEAREST_INT));
				__m128i int32_2 = _mm256_cvtpd_epi32(_mm256_round_pd(clamped_2, _MM_FROUND_TO_NEAREST_INT));
				__m128i int32_3 = _mm256_cvtpd_epi32(_mm256_round_pd(clamped_3, _MM_FROUND_TO_NEAREST_INT));
				__m128i int32_4 = _mm256_cvtpd_epi32(_mm256_round_pd(clamped_4, _MM_FROUND_TO_NEAREST_INT));
				__m128i int32_5 = _mm256_cvtpd_epi32(_mm256_round_pd(clamped_5, _MM_FROUND_TO_NEAREST_INT));
				__m128i int32_6 = _mm256_cvtpd_epi32(_mm256_round_pd(clamped_6, _MM_FROUND_TO_NEAREST_INT));
				__m128i int32_7 = _mm256_cvtpd_epi32(_mm256_round_pd(clamped_7, _MM_FROUND_TO_NEAREST_INT));

				__m256i int32_01 = _mm256_insertf128_si256(_mm256_castsi128_si256(int32_0), int32_1, 1);
				__m256i int32_23 = _mm256_insertf128_si256(_mm256_castsi128_si256(int32_2), int32_3, 1);
				__m256i int32_45 = _mm256_insertf128_si256(_mm256_castsi128_si256(int32_4), int32_5, 1);
				__m256i int32_67 = _mm256_insertf128_si256(_mm256_castsi128_si256(int32_6), int32_7, 1);

				__m256i int16_0123 = _mm256_packs_epi32(int32_01, int32_23);
				__m256i int16_4567 = _mm256_packs_epi32(int32_45, int32_67);

				__m256i int8_result = _mm256_packs_epi16(int16_0123, int16_4567);

				int8_result = _mm256_permutevar8x32_epi32(int8_result, perm_mask);
				return v_int8x32(int8_result);
			}
		};

		template<> struct v_merge_Invoker<v_int8x32, v_uint8x16>
		{
			const v_uint8x16& src_0;
			const v_uint8x16& src_1;

			v_merge_Invoker(const v_uint8x16& src_0_, const v_uint8x16& src_1_) noexcept : src_0(src_0_), src_1(src_1_) {}
			v_int8x32 operator () () const noexcept
			{
				static const __m256i v_min = _mm256_set1_epi8(-128);
				static const __m256i v_max = _mm256_set1_epi8(127);

				__m256i combined = _mm256_inserti128_si256(_mm256_castsi128_si256(src_0.data), src_1.data, 1);
				__m256i mask = _mm256_cmpeq_epi8(_mm256_min_epu8(combined, v_min), v_min);
				__m256i saturated = _mm256_blendv_epi8(combined, v_max, mask);

				return v_int8x32(saturated);
			}
		};

		template<> struct v_merge_Invoker<v_int8x32, v_uint16x8>
		{
			const v_uint16x8& src_0;
			const v_uint16x8& src_1;
			const v_uint16x8& src_2;
			const v_uint16x8& src_3;

			v_merge_Invoker(const v_uint16x8& src_0_, const v_uint16x8& src_1_, const v_uint16x8& src_2_, const v_uint16x8& src_3_)
				noexcept : src_0(src_0_), src_1(src_1_), src_2(src_2_), src_3(src_3_)
			{
			}

			v_int8x32 operator () () const noexcept
			{
				static const __m256i perm_mask = _mm256_set_epi32(7, 6, 3, 2, 5, 4, 1, 0);
				static const __m256i v_min = _mm256_set1_epi8(-128);
				static const __m256i v_max = _mm256_set1_epi8(127);

				__m256i pack16_01 = _mm256_inserti128_si256(_mm256_castsi128_si256(src_0.data), src_1.data, 1);
				__m256i pack16_23 = _mm256_inserti128_si256(_mm256_castsi128_si256(src_2.data), src_3.data, 1);

				__m256i result_u8 = _mm256_packus_epi16(pack16_01, pack16_23);

				result_u8 = _mm256_permutevar8x32_epi32(result_u8, perm_mask);

				__m256i mask = _mm256_cmpeq_epi8(_mm256_min_epu8(result_u8, v_min), v_min);
				__m256i result_i8 = _mm256_blendv_epi8(result_u8, v_max, mask);

				return v_int8x32(result_i8);
			}
		};

		template<> struct v_merge_Invoker<v_int8x32, v_uint32x4>
		{
			const v_uint32x4& src_0;
			const v_uint32x4& src_1;
			const v_uint32x4& src_2;
			const v_uint32x4& src_3;
			const v_uint32x4& src_4;
			const v_uint32x4& src_5;
			const v_uint32x4& src_6;
			const v_uint32x4& src_7;

			v_merge_Invoker(
				const v_uint32x4& src_0_, const v_uint32x4& src_1_, const v_uint32x4& src_2_, const v_uint32x4& src_3_,
				const v_uint32x4& src_4_, const v_uint32x4& src_5_, const v_uint32x4& src_6_, const v_uint32x4& src_7_)
				noexcept : src_0(src_0_), src_1(src_1_), src_2(src_2_), src_3(src_3_), src_4(src_4_), src_5(src_5_), src_6(src_6_), src_7(src_7_)
			{
			}

			v_int8x32 operator () () const noexcept
			{
				static const __m256i perm_mask = _mm256_set_epi32(7, 3, 6, 2, 5, 1, 4, 0);
				static const __m256i v_min = _mm256_set1_epi8(-128);
				static const __m256i v_max = _mm256_set1_epi8(127);

				__m256i pack32_01 = _mm256_inserti128_si256(_mm256_castsi128_si256(src_0.data), src_1.data, 1);
				__m256i pack32_23 = _mm256_inserti128_si256(_mm256_castsi128_si256(src_2.data), src_3.data, 1);
				__m256i pack32_45 = _mm256_inserti128_si256(_mm256_castsi128_si256(src_4.data), src_5.data, 1);
				__m256i pack32_67 = _mm256_inserti128_si256(_mm256_castsi128_si256(src_6.data), src_7.data, 1);

				__m256i pack16_0123 = _mm256_packus_epi32(pack32_01, pack32_23);
				__m256i pack16_4567 = _mm256_packus_epi32(pack32_45, pack32_67);

				__m256i pack8_u = _mm256_packus_epi16(pack16_0123, pack16_4567);

				pack8_u = _mm256_permutevar8x32_epi32(pack8_u, perm_mask);

				__m256i mask = _mm256_cmpeq_epi8(_mm256_min_epu8(pack8_u, v_min), v_min);
				__m256i result = _mm256_blendv_epi8(pack8_u, v_max, mask);

				return v_int8x32(result);
			}
		};

		template<> struct v_merge_Invoker<v_int8x32, v_uint64x2>
		{
			const v_uint64x2& src_0;
			const v_uint64x2& src_1;
			const v_uint64x2& src_2;
			const v_uint64x2& src_3;
			const v_uint64x2& src_4;
			const v_uint64x2& src_5;
			const v_uint64x2& src_6;
			const v_uint64x2& src_7;
			const v_uint64x2& src_8;
			const v_uint64x2& src_9;
			const v_uint64x2& src_10;
			const v_uint64x2& src_11;
			const v_uint64x2& src_12;
			const v_uint64x2& src_13;
			const v_uint64x2& src_14;
			const v_uint64x2& src_15;

			v_merge_Invoker(
				const v_uint64x2& src_0_, const v_uint64x2& src_1_, const v_uint64x2& src_2_, const v_uint64x2& src_3_,
				const v_uint64x2& src_4_, const v_uint64x2& src_5_, const v_uint64x2& src_6_, const v_uint64x2& src_7_,
				const v_uint64x2& src_8_, const v_uint64x2& src_9_, const v_uint64x2& src_10_, const v_uint64x2& src_11_,
				const v_uint64x2& src_12_, const v_uint64x2& src_13_, const v_uint64x2& src_14_, const v_uint64x2& src_15_) noexcept
				: src_0(src_0_), src_1(src_1_), src_2(src_2_), src_3(src_3_), src_4(src_4_), src_5(src_5_), src_6(src_6_), src_7(src_7_),
				src_8(src_8_), src_9(src_9_), src_10(src_10_), src_11(src_11_), src_12(src_12_), src_13(src_13_), src_14(src_14_), src_15(src_15_)
			{
			}

			v_int8x32 operator () () const noexcept
			{
				static const __m256i perm_mask = _mm256_set_epi32(7, 6, 3, 2, 5, 4, 1, 0);
				static const __m256i v_min = _mm256_set1_epi8(-128);
				static const __m256i v_max = _mm256_set1_epi8(127);

				__m128i pack32_0 = _mm_packus_epi32(src_0.data, src_1.data);
				__m128i pack32_1 = _mm_packus_epi32(src_2.data, src_3.data);
				__m128i pack32_2 = _mm_packus_epi32(src_4.data, src_5.data);
				__m128i pack32_3 = _mm_packus_epi32(src_6.data, src_7.data);
				__m128i pack32_4 = _mm_packus_epi32(src_8.data, src_9.data);
				__m128i pack32_5 = _mm_packus_epi32(src_10.data, src_11.data);
				__m128i pack32_6 = _mm_packus_epi32(src_12.data, src_13.data);
				__m128i pack32_7 = _mm_packus_epi32(src_14.data, src_15.data);

				__m128i pack16_0 = _mm_packus_epi16(pack32_0, pack32_1);
				__m128i pack16_1 = _mm_packus_epi16(pack32_2, pack32_3);
				__m128i pack16_2 = _mm_packus_epi16(pack32_4, pack32_5);
				__m128i pack16_3 = _mm_packus_epi16(pack32_6, pack32_7);

				__m256i pack16_01 = _mm256_inserti128_si256(_mm256_castsi128_si256(pack16_0), pack16_1, 1);
				__m256i pack16_23 = _mm256_inserti128_si256(_mm256_castsi128_si256(pack16_2), pack16_3, 1);

				__m256i pack8_u = _mm256_packus_epi16(pack16_01, pack16_23);

				pack8_u = _mm256_permutevar8x32_epi32(pack8_u, perm_mask);

				__m256i mask = _mm256_cmpeq_epi8(_mm256_min_epu8(pack8_u, v_min), v_min);
				__m256i result = _mm256_blendv_epi8(pack8_u, v_max, mask);

				return v_int8x32(result);
			}
		};

		template<> struct v_merge_Invoker<v_int8x32, v_int16x16>
		{
			const v_int16x16& src_0;
			const v_int16x16& src_1;

			v_merge_Invoker(const v_int16x16& src_0_, const v_int16x16& src_1_) noexcept : src_0(src_0_), src_1(src_1_) {}
			v_int8x32 operator () () const noexcept
			{
				__m256i result = _mm256_packs_epi16(src_0.data, src_1.data);

				result = _mm256_permute4x64_epi64(result, _MM_SHUFFLE(3, 1, 2, 0));

				return v_int8x32(result);
			}
		};

		template<> struct v_merge_Invoker<v_int8x32, v_int32x8>
		{
			const v_int32x8& src_0;
			const v_int32x8& src_1;
			const v_int32x8& src_2;
			const v_int32x8& src_3;

			v_merge_Invoker(const v_int32x8& src_0_, const v_int32x8& src_1_, const v_int32x8& src_2_, const v_int32x8& src_3_) noexcept
				: src_0(src_0_), src_1(src_1_), src_2(src_2_), src_3(src_3_)
			{
			}

			v_int8x32 operator () () const noexcept
			{
				static const __m256i mask = _mm256_set_epi32(7, 3, 6, 2, 5, 1, 4, 0);

				__m256i saturated_0 = _mm256_packs_epi32(src_0.data, src_1.data);
				__m256i saturated_1 = _mm256_packs_epi32(src_2.data, src_3.data);

				__m256i result = _mm256_packs_epi16(saturated_0, saturated_1);

				result = _mm256_permutevar8x32_epi32(result, mask);

				return v_int8x32(result);
			}
		};


		template<> struct v_merge_Invoker<v_int8x32, v_int64x4>
		{
			const v_int64x4& src_0;
			const v_int64x4& src_1;
			const v_int64x4& src_2;
			const v_int64x4& src_3;
			const v_int64x4& src_4;
			const v_int64x4& src_5;
			const v_int64x4& src_6;
			const v_int64x4& src_7;

			v_merge_Invoker(const v_int64x4& v0, const v_int64x4& v1, const v_int64x4& v2, const v_int64x4& v3,
				const v_int64x4& v4, const v_int64x4& v5, const v_int64x4& v6, const v_int64x4& v7) noexcept
				: src_0(v0), src_1(v1), src_2(v2), src_3(v3), src_4(v4), src_5(v5), src_6(v6), src_7(v7)
			{
			}

			v_int8x32 operator () () const noexcept
			{
				static const __m256i min_val = _mm256_set1_epi32(-128);
				static const __m256i max_val = _mm256_set1_epi32(127);

				__m256i clamped_0 = _mm256_max_epi32(_mm256_min_epi32(src_0.data, max_val), min_val);
				__m256i clamped_1 = _mm256_max_epi32(_mm256_min_epi32(src_1.data, max_val), min_val);
				__m256i clamped_2 = _mm256_max_epi32(_mm256_min_epi32(src_2.data, max_val), min_val);
				__m256i clamped_3 = _mm256_max_epi32(_mm256_min_epi32(src_3.data, max_val), min_val);
				__m256i clamped_4 = _mm256_max_epi32(_mm256_min_epi32(src_4.data, max_val), min_val);
				__m256i clamped_5 = _mm256_max_epi32(_mm256_min_epi32(src_5.data, max_val), min_val);
				__m256i clamped_6 = _mm256_max_epi32(_mm256_min_epi32(src_6.data, max_val), min_val);
				__m256i clamped_7 = _mm256_max_epi32(_mm256_min_epi32(src_7.data, max_val), min_val);

				__m128i pack16_0 = _mm_packs_epi32(_mm256_castsi256_si128(clamped_0), _mm256_extracti128_si256(clamped_0, 1));
				__m128i pack16_1 = _mm_packs_epi32(_mm256_castsi256_si128(clamped_1), _mm256_extracti128_si256(clamped_1, 1));
				__m128i pack16_2 = _mm_packs_epi32(_mm256_castsi256_si128(clamped_2), _mm256_extracti128_si256(clamped_2, 1));
				__m128i pack16_3 = _mm_packs_epi32(_mm256_castsi256_si128(clamped_3), _mm256_extracti128_si256(clamped_3, 1));
				__m128i pack16_4 = _mm_packs_epi32(_mm256_castsi256_si128(clamped_4), _mm256_extracti128_si256(clamped_4, 1));
				__m128i pack16_5 = _mm_packs_epi32(_mm256_castsi256_si128(clamped_5), _mm256_extracti128_si256(clamped_5, 1));
				__m128i pack16_6 = _mm_packs_epi32(_mm256_castsi256_si128(clamped_6), _mm256_extracti128_si256(clamped_6, 1));
				__m128i pack16_7 = _mm_packs_epi32(_mm256_castsi256_si128(clamped_7), _mm256_extracti128_si256(clamped_7, 1));

				__m256i pack8_0 = _mm256_setr_m128i(_mm_packs_epi16(pack16_0, pack16_1), _mm_packs_epi16(pack16_2, pack16_3));
				__m256i pack8_1 = _mm256_setr_m128i(_mm_packs_epi16(pack16_4, pack16_5), _mm_packs_epi16(pack16_6, pack16_7));

				__m256i result = _mm256_permute4x64_epi64(_mm256_packs_epi16(pack8_0, pack8_1), 0xD8);
				return v_int8x32(result);
			}
		};

		template<> struct v_merge_Invoker<v_int8x32, v_float16x8>
		{
			const v_float16x8& src_0;
			const v_float16x8& src_1;
			const v_float16x8& src_2;
			const v_float16x8& src_3;

			v_merge_Invoker(const v_float16x8& src_0_, const v_float16x8& src_1_, const v_float16x8& src_2_, const v_float16x8& src_3_)
				noexcept : src_0(src_0_), src_1(src_1_), src_2(src_2_), src_3(src_3_)
			{
			}
			v_int8x32 operator () () const noexcept
			{
				static const __m256i perm_mask = _mm256_set_epi32(7, 3, 6, 2, 5, 1, 4, 0);
				static const __m256 min_val = _mm256_set1_ps(-128.0f);
				static const __m256 max_val = _mm256_set1_ps(127.0f);

				__m256 float32_0 = _mm256_min_ps(_mm256_max_ps(_mm256_cvtph_ps(src_0.data), min_val), max_val);
				__m256 float32_1 = _mm256_min_ps(_mm256_max_ps(_mm256_cvtph_ps(src_1.data), min_val), max_val);
				__m256 float32_2 = _mm256_min_ps(_mm256_max_ps(_mm256_cvtph_ps(src_2.data), min_val), max_val);
				__m256 float32_3 = _mm256_min_ps(_mm256_max_ps(_mm256_cvtph_ps(src_3.data), min_val), max_val);

				__m256i int32_0 = _mm256_cvtps_epi32(_mm256_round_ps(float32_0, _MM_FROUND_TO_NEAREST_INT));
				__m256i int32_1 = _mm256_cvtps_epi32(_mm256_round_ps(float32_1, _MM_FROUND_TO_NEAREST_INT));
				__m256i int32_2 = _mm256_cvtps_epi32(_mm256_round_ps(float32_2, _MM_FROUND_TO_NEAREST_INT));
				__m256i int32_3 = _mm256_cvtps_epi32(_mm256_round_ps(float32_3, _MM_FROUND_TO_NEAREST_INT));

				__m256i int16_0 = _mm256_packs_epi32(int32_0, int32_1);
				__m256i int16_1 = _mm256_packs_epi32(int32_2, int32_3);

				__m256i int8_result = _mm256_packs_epi16(int16_0, int16_1);
				int8_result = _mm256_permutevar8x32_epi32(int8_result, perm_mask);
				return v_int8x32(int8_result);
			}
		};


		template<> struct v_merge_Invoker<v_int8x32, v_float32x4>
		{
			const v_float32x4& src_0;
			const v_float32x4& src_1;
			const v_float32x4& src_2;
			const v_float32x4& src_3;
			const v_float32x4& src_4;
			const v_float32x4& src_5;
			const v_float32x4& src_6;
			const v_float32x4& src_7;

			v_merge_Invoker(
				const v_float32x4& src_0_, const v_float32x4& src_1_, const v_float32x4& src_2_, const v_float32x4& src_3_,
				const v_float32x4& src_4_, const v_float32x4& src_5_, const v_float32x4& src_6_, const v_float32x4& src_7_)
				noexcept : src_0(src_0_), src_1(src_1_), src_2(src_2_), src_3(src_3_), src_4(src_4_), src_5(src_5_), src_6(src_6_), src_7(src_7_)
			{
			}

			v_int8x32 operator () () const noexcept
			{
				static const __m128 min_val = _mm_set1_ps(-128.0f);
				static const  __m128 max_val = _mm_set1_ps(127.0f);

				__m128i int32_0 = _mm_cvtps_epi32(_mm_round_ps(_mm_min_ps(_mm_max_ps(src_0.data, min_val), max_val), _MM_FROUND_TO_NEAREST_INT));
				__m128i int32_1 = _mm_cvtps_epi32(_mm_round_ps(_mm_min_ps(_mm_max_ps(src_1.data, min_val), max_val), _MM_FROUND_TO_NEAREST_INT));
				__m128i int32_2 = _mm_cvtps_epi32(_mm_round_ps(_mm_min_ps(_mm_max_ps(src_2.data, min_val), max_val), _MM_FROUND_TO_NEAREST_INT));
				__m128i int32_3 = _mm_cvtps_epi32(_mm_round_ps(_mm_min_ps(_mm_max_ps(src_3.data, min_val), max_val), _MM_FROUND_TO_NEAREST_INT));
				__m128i int32_4 = _mm_cvtps_epi32(_mm_round_ps(_mm_min_ps(_mm_max_ps(src_4.data, min_val), max_val), _MM_FROUND_TO_NEAREST_INT));
				__m128i int32_5 = _mm_cvtps_epi32(_mm_round_ps(_mm_min_ps(_mm_max_ps(src_5.data, min_val), max_val), _MM_FROUND_TO_NEAREST_INT));
				__m128i int32_6 = _mm_cvtps_epi32(_mm_round_ps(_mm_min_ps(_mm_max_ps(src_6.data, min_val), max_val), _MM_FROUND_TO_NEAREST_INT));
				__m128i int32_7 = _mm_cvtps_epi32(_mm_round_ps(_mm_min_ps(_mm_max_ps(src_7.data, min_val), max_val), _MM_FROUND_TO_NEAREST_INT));

				__m128i int16_0 = _mm_packs_epi32(int32_0, int32_1);
				__m128i int16_1 = _mm_packs_epi32(int32_2, int32_3);
				__m128i int16_2 = _mm_packs_epi32(int32_4, int32_5);
				__m128i int16_3 = _mm_packs_epi32(int32_6, int32_7);

				__m128i int8_0 = _mm_packs_epi16(int16_0, int16_1);
				__m128i int8_1 = _mm_packs_epi16(int16_2, int16_3);

				__m256i int8_result = _mm256_insertf128_si256(_mm256_castsi128_si256(int8_0), int8_1, 1);
				return v_int8x32(int8_result);
			}
		};

		template<> struct v_merge_Invoker<v_int8x32, v_float64x2>
		{
			const v_float64x2& src_0;
			const v_float64x2& src_1;
			const v_float64x2& src_2;
			const v_float64x2& src_3;
			const v_float64x2& src_4;
			const v_float64x2& src_5;
			const v_float64x2& src_6;
			const v_float64x2& src_7;
			const v_float64x2& src_8;
			const v_float64x2& src_9;
			const v_float64x2& src_10;
			const v_float64x2& src_11;
			const v_float64x2& src_12;
			const v_float64x2& src_13;
			const v_float64x2& src_14;
			const v_float64x2& src_15;

			v_merge_Invoker(
				const v_float64x2& src_0_, const v_float64x2& src_1_, const v_float64x2& src_2_, const v_float64x2& src_3_,
				const v_float64x2& src_4_, const v_float64x2& src_5_, const v_float64x2& src_6_, const v_float64x2& src_7_,
				const v_float64x2& src_8_, const v_float64x2& src_9_, const v_float64x2& src_10_, const v_float64x2& src_11_,
				const v_float64x2& src_12_, const v_float64x2& src_13_, const v_float64x2& src_14_, const v_float64x2& src_15_)
				noexcept
				: src_0(src_0_), src_1(src_1_), src_2(src_2_), src_3(src_3_)
				, src_4(src_4_), src_5(src_5_), src_6(src_6_), src_7(src_7_)
				, src_8(src_8_), src_9(src_9_), src_10(src_10_), src_11(src_11_)
				, src_12(src_12_), src_13(src_13_), src_14(src_14_), src_15(src_15_)
			{
			}

			v_int8x32 operator () () const noexcept
			{
				static const __m128d min_val = _mm_set1_pd(-128.0);
				static const __m128d max_val = _mm_set1_pd(127.0);

				__m128i int32_0 = _mm_cvtpd_epi32(_mm_round_pd(_mm_min_pd(_mm_max_pd(src_0.data, min_val), max_val), _MM_FROUND_TO_NEAREST_INT));
				__m128i int32_1 = _mm_cvtpd_epi32(_mm_round_pd(_mm_min_pd(_mm_max_pd(src_1.data, min_val), max_val), _MM_FROUND_TO_NEAREST_INT));
				__m128i int32_2 = _mm_cvtpd_epi32(_mm_round_pd(_mm_min_pd(_mm_max_pd(src_2.data, min_val), max_val), _MM_FROUND_TO_NEAREST_INT));
				__m128i int32_3 = _mm_cvtpd_epi32(_mm_round_pd(_mm_min_pd(_mm_max_pd(src_3.data, min_val), max_val), _MM_FROUND_TO_NEAREST_INT));
				__m128i int32_4 = _mm_cvtpd_epi32(_mm_round_pd(_mm_min_pd(_mm_max_pd(src_4.data, min_val), max_val), _MM_FROUND_TO_NEAREST_INT));
				__m128i int32_5 = _mm_cvtpd_epi32(_mm_round_pd(_mm_min_pd(_mm_max_pd(src_5.data, min_val), max_val), _MM_FROUND_TO_NEAREST_INT));
				__m128i int32_6 = _mm_cvtpd_epi32(_mm_round_pd(_mm_min_pd(_mm_max_pd(src_6.data, min_val), max_val), _MM_FROUND_TO_NEAREST_INT));
				__m128i int32_7 = _mm_cvtpd_epi32(_mm_round_pd(_mm_min_pd(_mm_max_pd(src_7.data, min_val), max_val), _MM_FROUND_TO_NEAREST_INT));
				__m128i int32_8 = _mm_cvtpd_epi32(_mm_round_pd(_mm_min_pd(_mm_max_pd(src_8.data, min_val), max_val), _MM_FROUND_TO_NEAREST_INT));
				__m128i int32_9 = _mm_cvtpd_epi32(_mm_round_pd(_mm_min_pd(_mm_max_pd(src_9.data, min_val), max_val), _MM_FROUND_TO_NEAREST_INT));
				__m128i int32_10 = _mm_cvtpd_epi32(_mm_round_pd(_mm_min_pd(_mm_max_pd(src_10.data, min_val), max_val), _MM_FROUND_TO_NEAREST_INT));
				__m128i int32_11 = _mm_cvtpd_epi32(_mm_round_pd(_mm_min_pd(_mm_max_pd(src_11.data, min_val), max_val), _MM_FROUND_TO_NEAREST_INT));
				__m128i int32_12 = _mm_cvtpd_epi32(_mm_round_pd(_mm_min_pd(_mm_max_pd(src_12.data, min_val), max_val), _MM_FROUND_TO_NEAREST_INT));
				__m128i int32_13 = _mm_cvtpd_epi32(_mm_round_pd(_mm_min_pd(_mm_max_pd(src_13.data, min_val), max_val), _MM_FROUND_TO_NEAREST_INT));
				__m128i int32_14 = _mm_cvtpd_epi32(_mm_round_pd(_mm_min_pd(_mm_max_pd(src_14.data, min_val), max_val), _MM_FROUND_TO_NEAREST_INT));
				__m128i int32_15 = _mm_cvtpd_epi32(_mm_round_pd(_mm_min_pd(_mm_max_pd(src_15.data, min_val), max_val), _MM_FROUND_TO_NEAREST_INT));

				__m128i int16_0 = _mm_packs_epi32(_mm_unpacklo_epi64(int32_0, int32_1), _mm_unpacklo_epi64(int32_2, int32_3));
				__m128i int16_1 = _mm_packs_epi32(_mm_unpacklo_epi64(int32_4, int32_5), _mm_unpacklo_epi64(int32_6, int32_7));
				__m128i int16_2 = _mm_packs_epi32(_mm_unpacklo_epi64(int32_8, int32_9), _mm_unpacklo_epi64(int32_10, int32_11));
				__m128i int16_3 = _mm_packs_epi32(_mm_unpacklo_epi64(int32_12, int32_13), _mm_unpacklo_epi64(int32_14, int32_15));

				__m128i int8_0 = _mm_packs_epi16(int16_0, int16_1);
				__m128i int8_1 = _mm_packs_epi16(int16_2, int16_3);

				__m256i int8_result = _mm256_insertf128_si256(_mm256_castsi128_si256(int8_0), int8_1, 1);
				return v_int8x32(int8_result);
			}
		};

		template<> struct v_merge_Invoker<v_uint16x16, v_uint32x8>
		{
			const v_uint32x8& src_0;
			const v_uint32x8& src_1;

			v_merge_Invoker(const v_uint32x8& src_0_, const v_uint32x8& src_1_) noexcept : src_0(src_0_), src_1(src_1_) {}
			v_uint16x16 operator () () const noexcept
			{
				__m256i result = _mm256_packus_epi32(src_0.data, src_1.data);
				result = _mm256_permute4x64_epi64(result, _MM_SHUFFLE(3, 1, 2, 0));
				return v_uint16x16(result);
			}
		};

		template<> struct v_merge_Invoker<v_uint16x16, v_uint64x4>
		{
			const v_uint64x4& src_0;
			const v_uint64x4& src_1;
			const v_uint64x4& src_2;
			const v_uint64x4& src_3;

			v_merge_Invoker(const v_uint64x4& src_0_, const v_uint64x4& src_1_, const v_uint64x4& src_2_, const v_uint64x4& src_3_)
				noexcept : src_0(src_0_), src_1(src_1_), src_2(src_2_), src_3(src_3_)
			{
			}
			v_uint16x16 operator () () const noexcept
			{
				__m256i pack32_0 = _mm256_permute4x64_epi64(_mm256_packus_epi32(src_0.data, src_1.data), _MM_SHUFFLE(3, 1, 2, 0));
				__m256i pack32_1 = _mm256_permute4x64_epi64(_mm256_packus_epi32(src_2.data, src_3.data), _MM_SHUFFLE(3, 1, 2, 0));

				__m256i result = _mm256_packus_epi32(pack32_0, pack32_1);

				result = _mm256_permute4x64_epi64(result, _MM_SHUFFLE(3, 1, 2, 0));
				return v_uint16x16(result);
			}
		};

		template<> struct v_merge_Invoker<v_uint16x16, v_float32x8>
		{
			const v_float32x8& src_0;
			const v_float32x8& src_1;

			v_merge_Invoker(const v_float32x8& src_0_, const v_float32x8& src_1_) noexcept : src_0(src_0_), src_1(src_1_) {}

			v_uint16x16 operator () () const noexcept
			{
				static const __m256i zero = _mm256_setzero_si256();

				__m256i int32_0 = _mm256_cvtps_epi32(src_0.data);
				__m256i int32_1 = _mm256_cvtps_epi32(src_1.data);

				int32_0 = _mm256_max_epi32(int32_0, zero);
				int32_1 = _mm256_max_epi32(int32_1, zero);

				__m256i result = _mm256_packus_epi32(int32_0, int32_1);

				result = _mm256_permute4x64_epi64(result, _MM_SHUFFLE(3, 1, 2, 0));
				return v_uint16x16(result);
			}
		};

		template<> struct v_merge_Invoker<v_uint16x16, v_float64x4>
		{
			const v_float64x4& src_0;
			const v_float64x4& src_1;
			const v_float64x4& src_2;
			const v_float64x4& src_3;

			v_merge_Invoker(const v_float64x4& src_0_, const v_float64x4& src_1_,
				const v_float64x4& src_2_, const v_float64x4& src_3_) noexcept
				: src_0(src_0_), src_1(src_1_), src_2(src_2_), src_3(src_3_)
			{
			}

			v_uint16x16 operator () () const noexcept
			{
				static const __m256i zero = _mm256_setzero_si256();

				__m128i int32_0 = _mm256_cvtpd_epi32(src_0.data);
				__m128i int32_1 = _mm256_cvtpd_epi32(src_1.data);
				__m128i int32_2 = _mm256_cvtpd_epi32(src_2.data);
				__m128i int32_3 = _mm256_cvtpd_epi32(src_3.data);

				__m256i combined_0 = _mm256_inserti128_si256(_mm256_castsi128_si256(int32_0), int32_1, 1);
				__m256i combined_1 = _mm256_inserti128_si256(_mm256_castsi128_si256(int32_2), int32_3, 1);

				combined_0 = _mm256_max_epi32(combined_0, zero);
				combined_1 = _mm256_max_epi32(combined_1, zero);

				__m256i result = _mm256_packus_epi32(combined_0, combined_1);

				result = _mm256_permute4x64_epi64(result, _MM_SHUFFLE(3, 1, 2, 0));

				return v_uint16x16(result);
			}
		};

		template<> struct v_merge_Invoker<v_uint16x16, v_int32x8>
		{
			const v_int32x8& src_0;
			const v_int32x8& src_1;

			v_merge_Invoker(const v_int32x8& src_0_, const v_int32x8& src_1_) noexcept : src_0(src_0_), src_1(src_1_) {}

			v_uint16x16 operator () () const noexcept
			{
				static const __m256i zero = _mm256_setzero_si256();
				__m256i mask_0 = _mm256_cmpgt_epi32(zero, src_0.data);
				__m256i mask_1 = _mm256_cmpgt_epi32(zero, src_1.data);

				__m256i pos_0 = _mm256_andnot_si256(mask_0, src_0.data);
				__m256i pos_1 = _mm256_andnot_si256(mask_1, src_1.data);

				__m256i result = _mm256_packus_epi32(pos_0, pos_1);

				result = _mm256_permute4x64_epi64(result, _MM_SHUFFLE(3, 1, 2, 0));
				return v_uint16x16(result);
			}
		};

		template<> struct v_merge_Invoker<v_uint16x16, v_int64x4>
		{
			const v_int64x4& src_0;
			const v_int64x4& src_1;
			const v_int64x4& src_2;
			const v_int64x4& src_3;

			v_merge_Invoker(const v_int64x4& src_0_, const v_int64x4& src_1_, const v_int64x4& src_2_, const v_int64x4& src_3_)
				noexcept : src_0(src_0_), src_1(src_1_), src_2(src_2_), src_3(src_3_)
			{
			}

			v_uint16x16 operator () () const noexcept
			{
				static const __m256i zero = _mm256_setzero_si256();
				__m256i mask_0 = _mm256_cmpgt_epi64(zero, src_0.data);
				__m256i mask_1 = _mm256_cmpgt_epi64(zero, src_1.data);
				__m256i mask_2 = _mm256_cmpgt_epi64(zero, src_2.data);
				__m256i mask_3 = _mm256_cmpgt_epi64(zero, src_3.data);

				__m256i pos_0 = _mm256_andnot_si256(mask_0, src_0.data);
				__m256i pos_1 = _mm256_andnot_si256(mask_1, src_1.data);
				__m256i pos_2 = _mm256_andnot_si256(mask_2, src_2.data);
				__m256i pos_3 = _mm256_andnot_si256(mask_3, src_3.data);

				__m256i pack32_0 = _mm256_permute4x64_epi64(_mm256_packus_epi32(pos_0, pos_1), _MM_SHUFFLE(3, 1, 2, 0));
				__m256i pack32_1 = _mm256_permute4x64_epi64(_mm256_packus_epi32(pos_2, pos_3), _MM_SHUFFLE(3, 1, 2, 0));

				__m256i result = _mm256_packus_epi32(pack32_0, pack32_1);

				result = _mm256_permute4x64_epi64(result, _MM_SHUFFLE(3, 1, 2, 0));
				return v_uint16x16(result);
			}
		};

		template<> struct v_merge_Invoker<v_uint16x16, v_uint16x8>
		{
			const v_uint16x8& src_0;
			const v_uint16x8& src_1;

			v_merge_Invoker(const v_uint16x8& src_0_, const v_uint16x8& src_1_) noexcept : src_0(src_0_), src_1(src_1_) {}

			v_uint16x16 operator () () const noexcept
			{
				__m256i result = _mm256_setzero_si256();
				result = _mm256_insertf128_si256(result, src_0.data, 0);
				result = _mm256_insertf128_si256(result, src_1.data, 1);
				return v_uint16x16(result);
			}
		};

		template<> struct v_merge_Invoker<v_uint16x16, v_int16x8>
		{
			const v_int16x8& src_0;
			const v_int16x8& src_1;

			v_merge_Invoker(const v_int16x8& src_0_, const v_int16x8& src_1_) noexcept : src_0(src_0_), src_1(src_1_) {}

			v_uint16x16 operator () () const noexcept
			{
				static const __m128i zero = _mm_setzero_si128();

				__m128i clipped_0 = _mm_max_epi16(src_0.data, zero);
				__m128i clipped_1 = _mm_max_epi16(src_1.data, zero);

				__m256i result = _mm256_setzero_si256();
				result = _mm256_insertf128_si256(result, clipped_0, 0);
				result = _mm256_insertf128_si256(result, clipped_1, 1);

				return v_uint16x16(result);
			}
		};

		template<> struct v_merge_Invoker<v_uint16x16, v_uint32x4>
		{
			const v_uint32x4& src_0;
			const v_uint32x4& src_1;
			const v_uint32x4& src_2;
			const v_uint32x4& src_3;

			v_merge_Invoker(const v_uint32x4& src_0_, const v_uint32x4& src_1_, const v_uint32x4& src_2_, const v_uint32x4& src_3_)
				noexcept : src_0(src_0_), src_1(src_1_), src_2(src_2_), src_3(src_3_)
			{
			}

			v_uint16x16 operator () () const noexcept
			{
				__m256i combined_0 = _mm256_setr_m128i(src_0.data, src_1.data);
				__m256i combined_1 = _mm256_setr_m128i(src_2.data, src_3.data);

				__m256i result = _mm256_packus_epi32(combined_0, combined_1);

				result = _mm256_permute4x64_epi64(result, _MM_SHUFFLE(3, 1, 2, 0));
				return v_uint16x16(result);
			}
		};

		template<> struct v_merge_Invoker<v_uint16x16, v_int32x4>
		{
			const v_int32x4& src_0;
			const v_int32x4& src_1;
			const v_int32x4& src_2;
			const v_int32x4& src_3;

			v_merge_Invoker(const v_int32x4& src_0_, const v_int32x4& src_1_, const v_int32x4& src_2_, const v_int32x4& src_3_)
				noexcept : src_0(src_0_), src_1(src_1_), src_2(src_2_), src_3(src_3_)
			{
			}

			v_uint16x16 operator () () const noexcept
			{
				static const __m128i zero = _mm_setzero_si128();

				__m128i clipped_0 = _mm_max_epi32(src_0.data, zero);
				__m128i clipped_1 = _mm_max_epi32(src_1.data, zero);
				__m128i clipped_2 = _mm_max_epi32(src_2.data, zero);
				__m128i clipped_3 = _mm_max_epi32(src_3.data, zero);

				__m256i combined_0 = _mm256_setr_m128i(clipped_0, clipped_1);
				__m256i combined_1 = _mm256_setr_m128i(clipped_2, clipped_3);

				__m256i result = _mm256_packus_epi32(combined_0, combined_1);

				result = _mm256_permute4x64_epi64(result, _MM_SHUFFLE(3, 1, 2, 0));

				return v_uint16x16(result);
			}
		};

		template<> struct v_merge_Invoker<v_uint16x16, v_uint64x2>
		{
			const v_uint64x2& src_0;
			const v_uint64x2& src_1;
			const v_uint64x2& src_2;
			const v_uint64x2& src_3;
			const v_uint64x2& src_4;
			const v_uint64x2& src_5;
			const v_uint64x2& src_6;
			const v_uint64x2& src_7;

			v_merge_Invoker(const v_uint64x2& src_0_, const v_uint64x2& src_1_,
				const v_uint64x2& src_2_, const v_uint64x2& src_3_,
				const v_uint64x2& src_4_, const v_uint64x2& src_5_,
				const v_uint64x2& src_6_, const v_uint64x2& src_7_)
				noexcept : src_0(src_0_), src_1(src_1_), src_2(src_2_), src_3(src_3_),
				src_4(src_4_), src_5(src_5_), src_6(src_6_), src_7(src_7_)
			{
			}

			v_uint16x16 operator () () const noexcept
			{
				__m128i pack32_0 = _mm_packus_epi32(src_0.data, src_1.data);
				__m128i pack32_1 = _mm_packus_epi32(src_2.data, src_3.data);
				__m128i pack32_2 = _mm_packus_epi32(src_4.data, src_5.data);
				__m128i pack32_3 = _mm_packus_epi32(src_6.data, src_7.data);

				__m256i combined_0 = _mm256_setr_m128i(pack32_0, pack32_1);
				__m256i combined_1 = _mm256_setr_m128i(pack32_2, pack32_3);

				__m256i result = _mm256_packus_epi32(combined_0, combined_1);

				result = _mm256_permute4x64_epi64(result, _MM_SHUFFLE(3, 1, 2, 0));
				return v_uint16x16(result);
			}
		};

		template<> struct v_merge_Invoker<v_uint16x16, v_int64x2>
		{
			const v_int64x2& src_0;
			const v_int64x2& src_1;
			const v_int64x2& src_2;
			const v_int64x2& src_3;
			const v_int64x2& src_4;
			const v_int64x2& src_5;
			const v_int64x2& src_6;
			const v_int64x2& src_7;

			v_merge_Invoker(
				const v_int64x2& src_0_, const v_int64x2& src_1_,
				const v_int64x2& src_2_, const v_int64x2& src_3_,
				const v_int64x2& src_4_, const v_int64x2& src_5_,
				const v_int64x2& src_6_, const v_int64x2& src_7_)
				noexcept : src_0(src_0_), src_1(src_1_), src_2(src_2_), src_3(src_3_),
				src_4(src_4_), src_5(src_5_), src_6(src_6_), src_7(src_7_)
			{
			}

			v_uint16x16 operator () () const noexcept
			{
				return v_uint16x16(
					saturate_cast<u16>(src_0.data.m128i_i64[0]),
					saturate_cast<u16>(src_0.data.m128i_i64[1]),
					saturate_cast<u16>(src_1.data.m128i_i64[0]),
					saturate_cast<u16>(src_1.data.m128i_i64[1]),
					saturate_cast<u16>(src_2.data.m128i_i64[0]),
					saturate_cast<u16>(src_2.data.m128i_i64[1]),
					saturate_cast<u16>(src_3.data.m128i_i64[0]),
					saturate_cast<u16>(src_3.data.m128i_i64[1]),
					saturate_cast<u16>(src_4.data.m128i_i64[0]),
					saturate_cast<u16>(src_4.data.m128i_i64[1]),
					saturate_cast<u16>(src_5.data.m128i_i64[0]),
					saturate_cast<u16>(src_5.data.m128i_i64[1]),
					saturate_cast<u16>(src_6.data.m128i_i64[0]),
					saturate_cast<u16>(src_6.data.m128i_i64[1]),
					saturate_cast<u16>(src_7.data.m128i_i64[0]),
					saturate_cast<u16>(src_7.data.m128i_i64[1])
				);
			}
		};

		template<> struct v_merge_Invoker<v_uint16x16, v_float16x8>
		{
			const v_float16x8& src_0;
			const v_float16x8& src_1;

			v_merge_Invoker(const v_float16x8& src_0_, const v_float16x8& src_1_) noexcept : src_0(src_0_), src_1(src_1_) {}

			v_uint16x16 operator () () const noexcept
			{
				static const __m256i zero = _mm256_setzero_si256();

				__m256 float32_0 = _mm256_cvtph_ps(src_0.data);
				__m256 float32_1 = _mm256_cvtph_ps(src_1.data);

				__m256i int32_0 = _mm256_cvtps_epi32(float32_0);
				__m256i int32_1 = _mm256_cvtps_epi32(float32_1);

				int32_0 = _mm256_max_epi32(int32_0, zero);
				int32_1 = _mm256_max_epi32(int32_1, zero);

				__m256i result = _mm256_packus_epi32(int32_0, int32_1);

				result = _mm256_permute4x64_epi64(result, _MM_SHUFFLE(3, 1, 2, 0));

				return v_uint16x16(result);
			}
		};

		template<> struct v_merge_Invoker<v_uint16x16, v_float32x4>
		{
			const v_float32x4& src_0;
			const v_float32x4& src_1;
			const v_float32x4& src_2;
			const v_float32x4& src_3;

			v_merge_Invoker(const v_float32x4& src_0_, const v_float32x4& src_1_, const v_float32x4& src_2_, const v_float32x4& src_3_)
				noexcept : src_0(src_0_), src_1(src_1_), src_2(src_2_), src_3(src_3_)
			{
			}

			v_uint16x16 operator () () const noexcept
			{
				static const __m256i zero = _mm256_setzero_si256();

				__m256 combined_0 = _mm256_insertf128_ps(_mm256_castps128_ps256(src_0.data), src_1.data, 1);
				__m256 combined_1 = _mm256_insertf128_ps(_mm256_castps128_ps256(src_2.data), src_3.data, 1);

				__m256i int32_0 = _mm256_cvtps_epi32(combined_0);
				__m256i int32_1 = _mm256_cvtps_epi32(combined_1);

				int32_0 = _mm256_max_epi32(int32_0, zero);
				int32_1 = _mm256_max_epi32(int32_1, zero);

				__m256i result = _mm256_packus_epi32(int32_0, int32_1);

				result = _mm256_permute4x64_epi64(result, _MM_SHUFFLE(3, 1, 2, 0));
				return v_uint16x16(result);
			}
		};

		template<> struct v_merge_Invoker<v_uint16x16, v_float64x2>
		{
			const v_float64x2& src_0;
			const v_float64x2& src_1;
			const v_float64x2& src_2;
			const v_float64x2& src_3;
			const v_float64x2& src_4;
			const v_float64x2& src_5;
			const v_float64x2& src_6;
			const v_float64x2& src_7;

			v_merge_Invoker(const v_float64x2& src_0_, const v_float64x2& src_1_,
				const v_float64x2& src_2_, const v_float64x2& src_3_,
				const v_float64x2& src_4_, const v_float64x2& src_5_,
				const v_float64x2& src_6_, const v_float64x2& src_7_)
				noexcept : src_0(src_0_), src_1(src_1_), src_2(src_2_), src_3(src_3_),
				src_4(src_4_), src_5(src_5_), src_6(src_6_), src_7(src_7_)
			{
			}

			v_uint16x16 operator () () const noexcept
			{
				return v_uint16x16(
					saturate_cast<u16>(src_0.data.m128d_f64[0]), saturate_cast<u16>(src_0.data.m128d_f64[1]),
					saturate_cast<u16>(src_1.data.m128d_f64[0]), saturate_cast<u16>(src_1.data.m128d_f64[1]),
					saturate_cast<u16>(src_2.data.m128d_f64[0]), saturate_cast<u16>(src_2.data.m128d_f64[1]),
					saturate_cast<u16>(src_3.data.m128d_f64[0]), saturate_cast<u16>(src_3.data.m128d_f64[1]),
					saturate_cast<u16>(src_4.data.m128d_f64[0]), saturate_cast<u16>(src_4.data.m128d_f64[1]),
					saturate_cast<u16>(src_5.data.m128d_f64[0]), saturate_cast<u16>(src_5.data.m128d_f64[1]),
					saturate_cast<u16>(src_6.data.m128d_f64[0]), saturate_cast<u16>(src_6.data.m128d_f64[1]),
					saturate_cast<u16>(src_7.data.m128d_f64[0]), saturate_cast<u16>(src_7.data.m128d_f64[1])
				);
			}
		};




		template<> struct v_merge_Invoker<v_int16x16, v_uint32x8>
		{
			const v_uint32x8& src_0;
			const v_uint32x8& src_1;

			v_merge_Invoker(const v_uint32x8& src_0_, const v_uint32x8& src_1_) noexcept : src_0(src_0_), src_1(src_1_) {}
			v_int16x16 operator () () const noexcept
			{
				__m256i result = _mm256_packs_epi32(src_0.data, src_1.data);
				result = _mm256_permute4x64_epi64(result, _MM_SHUFFLE(3, 1, 2, 0));
				return v_int16x16(result);
			}
		};

		template<> struct v_merge_Invoker<v_int16x16, v_uint64x4>
		{
			const v_uint64x4& src_0;
			const v_uint64x4& src_1;
			const v_uint64x4& src_2;
			const v_uint64x4& src_3;

			v_merge_Invoker(const v_uint64x4& src_0_, const v_uint64x4& src_1_, const v_uint64x4& src_2_, const v_uint64x4& src_3_)
				noexcept : src_0(src_0_), src_1(src_1_), src_2(src_2_), src_3(src_3_)
			{
			}

			v_int16x16 operator () () const noexcept
			{
				__m256i pack32_0 = _mm256_permute4x64_epi64(_mm256_packus_epi32(src_0.data, src_1.data), _MM_SHUFFLE(3, 1, 2, 0));
				__m256i pack32_1 = _mm256_permute4x64_epi64(_mm256_packus_epi32(src_2.data, src_3.data), _MM_SHUFFLE(3, 1, 2, 0));

				__m256i result = _mm256_packs_epi32(pack32_0, pack32_1);
				result = _mm256_permute4x64_epi64(result, _MM_SHUFFLE(3, 1, 2, 0));
				return v_int16x16(result);
			}
		};

		template<> struct v_merge_Invoker<v_int16x16, v_float32x8>
		{
			const v_float32x8& src_0;
			const v_float32x8& src_1;

			v_merge_Invoker(const v_float32x8& src_0_, const v_float32x8& src_1_) noexcept : src_0(src_0_), src_1(src_1_) {}

			v_int16x16 operator () () const noexcept
			{
				__m256i int32_0 = _mm256_cvtps_epi32(src_0.data);
				__m256i int32_1 = _mm256_cvtps_epi32(src_1.data);

				__m256i result = _mm256_packs_epi32(int32_0, int32_1);

				result = _mm256_permute4x64_epi64(result, _MM_SHUFFLE(3, 1, 2, 0));
				return v_int16x16(result);
			}
		};

		template<> struct v_merge_Invoker<v_int16x16, v_float64x4>
		{
			const v_float64x4& src_0;
			const v_float64x4& src_1;
			const v_float64x4& src_2;
			const v_float64x4& src_3;

			v_merge_Invoker(const v_float64x4& src_0_, const v_float64x4& src_1_,
				const v_float64x4& src_2_, const v_float64x4& src_3_) noexcept
				: src_0(src_0_), src_1(src_1_), src_2(src_2_), src_3(src_3_)
			{
			}

			v_int16x16 operator () () const noexcept
			{
				__m128i int32_0 = _mm256_cvtpd_epi32(src_0.data);
				__m128i int32_1 = _mm256_cvtpd_epi32(src_1.data);
				__m128i int32_2 = _mm256_cvtpd_epi32(src_2.data);
				__m128i int32_3 = _mm256_cvtpd_epi32(src_3.data);

				__m256i combined_0 = _mm256_inserti128_si256(_mm256_castsi128_si256(int32_0), int32_1, 1);
				__m256i combined_1 = _mm256_inserti128_si256(_mm256_castsi128_si256(int32_2), int32_3, 1);

				__m256i result = _mm256_packs_epi32(combined_0, combined_1);

				result = _mm256_permute4x64_epi64(result, _MM_SHUFFLE(3, 1, 2, 0));

				return v_int16x16(result);
			}
		};

		template<> struct v_merge_Invoker<v_int16x16, v_int32x8>
		{
			const v_int32x8& src_0;
			const v_int32x8& src_1;

			v_merge_Invoker(const v_int32x8& src_0_, const v_int32x8& src_1_) noexcept : src_0(src_0_), src_1(src_1_) {}

			v_int16x16 operator () () const noexcept
			{
				__m256i result = _mm256_packs_epi32(src_0.data, src_1.data);

				result = _mm256_permute4x64_epi64(result, _MM_SHUFFLE(3, 1, 2, 0));
				return v_int16x16(result);
			}
		};

		template<> struct v_merge_Invoker<v_int16x16, v_int64x4>
		{
			const v_int64x4& src_0;
			const v_int64x4& src_1;
			const v_int64x4& src_2;
			const v_int64x4& src_3;

			v_merge_Invoker(const v_int64x4& src_0_, const v_int64x4& src_1_, const v_int64x4& src_2_, const v_int64x4& src_3_)
				noexcept : src_0(src_0_), src_1(src_1_), src_2(src_2_), src_3(src_3_)
			{
			}

			v_int16x16 operator () () const noexcept
			{
				__m256i pack32_0 = _mm256_packs_epi32(src_0.data, src_1.data);
				__m256i pack32_1 = _mm256_packs_epi32(src_2.data, src_3.data);

				__m256i shuffuled_pack32_0 = _mm256_permute4x64_epi64(pack32_0, _MM_SHUFFLE(3, 1, 2, 0));
				__m256i shuffuled_pack32_1 = _mm256_permute4x64_epi64(pack32_1, _MM_SHUFFLE(3, 1, 2, 0));

				__m256i result = _mm256_packs_epi32(shuffuled_pack32_0, shuffuled_pack32_1);

				result = _mm256_permute4x64_epi64(result, _MM_SHUFFLE(3, 1, 2, 0));
				return v_int16x16(result);
			}
		};

		template<> struct v_merge_Invoker<v_int16x16, v_uint16x8>
		{
			const v_uint16x8& src_0;
			const v_uint16x8& src_1;

			v_merge_Invoker(const v_uint16x8& src_0_, const v_uint16x8& src_1_) noexcept : src_0(src_0_), src_1(src_1_) {}

			v_int16x16 operator () () const noexcept
			{
				__m256i result = _mm256_setr_m128i(src_0.data, src_1.data);
				return v_int16x16(result);
			}
		};

		template<> struct v_merge_Invoker<v_int16x16, v_int16x8>
		{
			const v_int16x8& src_0;
			const v_int16x8& src_1;

			v_merge_Invoker(const v_int16x8& src_0_, const v_int16x8& src_1_) noexcept : src_0(src_0_), src_1(src_1_) {}

			v_int16x16 operator () () const noexcept
			{
				__m256i result = _mm256_setr_m128i(src_0.data, src_1.data);
				return v_int16x16(result);
			}
		};

		template<> struct v_merge_Invoker<v_int16x16, v_uint32x4>
		{
			const v_uint32x4& src_0;
			const v_uint32x4& src_1;
			const v_uint32x4& src_2;
			const v_uint32x4& src_3;

			v_merge_Invoker(const v_uint32x4& src_0_, const v_uint32x4& src_1_, const v_uint32x4& src_2_, const v_uint32x4& src_3_)
				noexcept : src_0(src_0_), src_1(src_1_), src_2(src_2_), src_3(src_3_)
			{
			}

			v_int16x16 operator () () const noexcept
			{
				__m256i combined_0 = _mm256_setr_m128i(src_0.data, src_1.data);
				__m256i combined_1 = _mm256_setr_m128i(src_2.data, src_3.data);

				__m256i result = _mm256_packs_epi32(combined_0, combined_1);

				result = _mm256_permute4x64_epi64(result, _MM_SHUFFLE(3, 1, 2, 0));
				return v_int16x16(result);
			}
		};

		template<> struct v_merge_Invoker<v_int16x16, v_int32x4>
		{
			const v_int32x4& src_0;
			const v_int32x4& src_1;
			const v_int32x4& src_2;
			const v_int32x4& src_3;

			v_merge_Invoker(const v_int32x4& src_0_, const v_int32x4& src_1_, const v_int32x4& src_2_, const v_int32x4& src_3_)
				noexcept : src_0(src_0_), src_1(src_1_), src_2(src_2_), src_3(src_3_)
			{
			}

			v_int16x16 operator () () const noexcept
			{
				__m256i combined_0 = _mm256_setr_m128i(src_0.data, src_1.data);
				__m256i combined_1 = _mm256_setr_m128i(src_2.data, src_3.data);

				__m256i result = _mm256_packs_epi32(combined_0, combined_1);

				result = _mm256_permute4x64_epi64(result, _MM_SHUFFLE(3, 1, 2, 0));

				return v_int16x16(result);
			}
		};

		template<> struct v_merge_Invoker<v_int16x16, v_uint64x2>
		{
			const v_uint64x2& src_0;
			const v_uint64x2& src_1;
			const v_uint64x2& src_2;
			const v_uint64x2& src_3;
			const v_uint64x2& src_4;
			const v_uint64x2& src_5;
			const v_uint64x2& src_6;
			const v_uint64x2& src_7;

			v_merge_Invoker(const v_uint64x2& src_0_, const v_uint64x2& src_1_,
				const v_uint64x2& src_2_, const v_uint64x2& src_3_,
				const v_uint64x2& src_4_, const v_uint64x2& src_5_,
				const v_uint64x2& src_6_, const v_uint64x2& src_7_)
				noexcept : src_0(src_0_), src_1(src_1_), src_2(src_2_), src_3(src_3_),
				src_4(src_4_), src_5(src_5_), src_6(src_6_), src_7(src_7_)
			{
			}

			v_int16x16 operator () () const noexcept
			{
				__m128i pack32_0 = _mm_packs_epi32(src_0.data, src_1.data);
				__m128i pack32_1 = _mm_packs_epi32(src_2.data, src_3.data);
				__m128i pack32_2 = _mm_packs_epi32(src_4.data, src_5.data);
				__m128i pack32_3 = _mm_packs_epi32(src_6.data, src_7.data);

				__m256i combined_0 = _mm256_setr_m128i(pack32_0, pack32_1);
				__m256i combined_1 = _mm256_setr_m128i(pack32_2, pack32_3);

				__m256i result = _mm256_packs_epi32(combined_0, combined_1);

				result = _mm256_permute4x64_epi64(result, _MM_SHUFFLE(3, 1, 2, 0));
				return v_int16x16(result);
			}
		};

		template<> struct v_merge_Invoker<v_int16x16, v_int64x2>
		{
			const v_int64x2& src_0;
			const v_int64x2& src_1;
			const v_int64x2& src_2;
			const v_int64x2& src_3;
			const v_int64x2& src_4;
			const v_int64x2& src_5;
			const v_int64x2& src_6;
			const v_int64x2& src_7;

			v_merge_Invoker(
				const v_int64x2& src_0_, const v_int64x2& src_1_,
				const v_int64x2& src_2_, const v_int64x2& src_3_,
				const v_int64x2& src_4_, const v_int64x2& src_5_,
				const v_int64x2& src_6_, const v_int64x2& src_7_)
				noexcept : src_0(src_0_), src_1(src_1_), src_2(src_2_), src_3(src_3_),
				src_4(src_4_), src_5(src_5_), src_6(src_6_), src_7(src_7_)
			{
			}

			v_int16x16 operator () () const noexcept
			{
				return v_int16x16(
					saturate_cast<i16>(src_0.data.m128i_i64[0]),
					saturate_cast<i16>(src_0.data.m128i_i64[1]),
					saturate_cast<i16>(src_1.data.m128i_i64[0]),
					saturate_cast<i16>(src_1.data.m128i_i64[1]),
					saturate_cast<i16>(src_2.data.m128i_i64[0]),
					saturate_cast<i16>(src_2.data.m128i_i64[1]),
					saturate_cast<i16>(src_3.data.m128i_i64[0]),
					saturate_cast<i16>(src_3.data.m128i_i64[1]),
					saturate_cast<i16>(src_4.data.m128i_i64[0]),
					saturate_cast<i16>(src_4.data.m128i_i64[1]),
					saturate_cast<i16>(src_5.data.m128i_i64[0]),
					saturate_cast<i16>(src_5.data.m128i_i64[1]),
					saturate_cast<i16>(src_6.data.m128i_i64[0]),
					saturate_cast<i16>(src_6.data.m128i_i64[1]),
					saturate_cast<i16>(src_7.data.m128i_i64[0]),
					saturate_cast<i16>(src_7.data.m128i_i64[1])
				);
			}
		};

		template<> struct v_merge_Invoker<v_int16x16, v_float16x8>
		{
			const v_float16x8& src_0;
			const v_float16x8& src_1;

			v_merge_Invoker(const v_float16x8& src_0_, const v_float16x8& src_1_) noexcept : src_0(src_0_), src_1(src_1_) {}

			v_int16x16 operator () () const noexcept
			{
				__m256 float32_0 = _mm256_cvtph_ps(src_0.data);
				__m256 float32_1 = _mm256_cvtph_ps(src_1.data);

				__m256i int32_0 = _mm256_cvtps_epi32(float32_0);
				__m256i int32_1 = _mm256_cvtps_epi32(float32_1);

				__m256i result = _mm256_packs_epi32(int32_0, int32_1);

				result = _mm256_permute4x64_epi64(result, _MM_SHUFFLE(3, 1, 2, 0));

				return v_int16x16(result);
			}
		};

		template<> struct v_merge_Invoker<v_int16x16, v_float32x4>
		{
			const v_float32x4& src_0;
			const v_float32x4& src_1;
			const v_float32x4& src_2;
			const v_float32x4& src_3;

			v_merge_Invoker(const v_float32x4& src_0_, const v_float32x4& src_1_, const v_float32x4& src_2_, const v_float32x4& src_3_)
				noexcept : src_0(src_0_), src_1(src_1_), src_2(src_2_), src_3(src_3_)
			{
			}

			v_int16x16 operator () () const noexcept
			{
				__m128i int32_0 = _mm_cvtps_epi32(src_0.data);
				__m128i int32_1 = _mm_cvtps_epi32(src_1.data);
				__m128i int32_2 = _mm_cvtps_epi32(src_2.data);
				__m128i int32_3 = _mm_cvtps_epi32(src_3.data);

				__m256i combined_0 = _mm256_inserti128_si256(_mm256_castsi128_si256(int32_0), int32_1, 1);
				__m256i combined_1 = _mm256_inserti128_si256(_mm256_castsi128_si256(int32_2), int32_3, 1);

				__m256i result = _mm256_packs_epi32(combined_0, combined_1);

				result = _mm256_permute4x64_epi64(result, _MM_SHUFFLE(3, 1, 2, 0));

				return v_int16x16(result);
			}
		};

		template<> struct v_merge_Invoker<v_int16x16, v_float64x2>
		{
			const v_float64x2& src_0;
			const v_float64x2& src_1;
			const v_float64x2& src_2;
			const v_float64x2& src_3;
			const v_float64x2& src_4;
			const v_float64x2& src_5;
			const v_float64x2& src_6;
			const v_float64x2& src_7;

			v_merge_Invoker(const v_float64x2& src_0_, const v_float64x2& src_1_,
				const v_float64x2& src_2_, const v_float64x2& src_3_,
				const v_float64x2& src_4_, const v_float64x2& src_5_,
				const v_float64x2& src_6_, const v_float64x2& src_7_)
				noexcept : src_0(src_0_), src_1(src_1_), src_2(src_2_), src_3(src_3_),
				src_4(src_4_), src_5(src_5_), src_6(src_6_), src_7(src_7_)
			{
			}

			v_int16x16 operator () () const noexcept
			{
				return v_int16x16(
					saturate_cast<i16>(src_0.data.m128d_f64[0]), saturate_cast<i16>(src_0.data.m128d_f64[1]),
					saturate_cast<i16>(src_1.data.m128d_f64[0]), saturate_cast<i16>(src_1.data.m128d_f64[1]),
					saturate_cast<i16>(src_2.data.m128d_f64[0]), saturate_cast<i16>(src_2.data.m128d_f64[1]),
					saturate_cast<i16>(src_3.data.m128d_f64[0]), saturate_cast<i16>(src_3.data.m128d_f64[1]),
					saturate_cast<i16>(src_4.data.m128d_f64[0]), saturate_cast<i16>(src_4.data.m128d_f64[1]),
					saturate_cast<i16>(src_5.data.m128d_f64[0]), saturate_cast<i16>(src_5.data.m128d_f64[1]),
					saturate_cast<i16>(src_6.data.m128d_f64[0]), saturate_cast<i16>(src_6.data.m128d_f64[1]),
					saturate_cast<i16>(src_7.data.m128d_f64[0]), saturate_cast<i16>(src_7.data.m128d_f64[1])
				);
			}
		};

		template<> struct v_merge_Invoker<v_uint32x8, v_uint64x4>
		{
			const v_uint64x4& src_0;
			const v_uint64x4& src_1;
			v_merge_Invoker(const v_uint64x4& src_0_, const v_uint64x4& src_1_) noexcept : src_0(src_0_), src_1(src_1_) {}

			v_uint32x8 operator () () const noexcept
			{
				return v_uint32x8(
					saturate_cast<u32>(src_0.data.m256i_u64[0]), saturate_cast<u32>(src_0.data.m256i_u64[1]),
					saturate_cast<u32>(src_0.data.m256i_u64[2]), saturate_cast<u32>(src_0.data.m256i_u64[3]),
					saturate_cast<u32>(src_1.data.m256i_u64[0]), saturate_cast<u32>(src_1.data.m256i_u64[1]),
					saturate_cast<u32>(src_1.data.m256i_u64[2]), saturate_cast<u32>(src_1.data.m256i_u64[3])
				);
			}
		};

		template<> struct v_merge_Invoker<v_uint32x8, v_int64x4>
		{
			const v_int64x4& src_0;
			const v_int64x4& src_1;
			v_merge_Invoker(const v_int64x4& src_0_, const v_int64x4& src_1_) noexcept : src_0(src_0_), src_1(src_1_) {}

			v_uint32x8 operator () () const noexcept
			{
				__m128i low_0 = _mm256_extracti128_si256(src_0.data, 0);
				__m128i high_0 = _mm256_extracti128_si256(src_0.data, 1);
				__m128i result_0 = _mm_packus_epi32(low_0, high_0);

				__m128i low_1 = _mm256_extracti128_si256(src_1.data, 0);
				__m128i high_1 = _mm256_extracti128_si256(src_1.data, 1);
				__m128i result_1 = _mm_packus_epi32(low_1, high_1);

				__m256i result = _mm256_insertf128_si256(_mm256_castsi128_si256(result_0), result_1, 1);

				return v_uint32x8(result);
			}
		};


		template<> struct v_merge_Invoker<v_uint32x8, v_float64x4>
		{
			const v_float64x4& src_0;
			const v_float64x4& src_1;
			v_merge_Invoker(const v_float64x4& src_0_, const v_float64x4& src_1_) noexcept : src_0(src_0_), src_1(src_1_) {}

			v_uint32x8 operator () () const noexcept
			{
				static const __m128 zero = _mm_setzero_ps();

				__m128 float_0 = _mm256_cvtpd_ps(src_0.data);
				__m128 float_1 = _mm256_cvtpd_ps(src_1.data);

				float_0 = _mm_max_ps(float_0, zero);
				float_1 = _mm_max_ps(float_1, zero);

				__m128i int32_0 = _mm_cvtps_epi32(float_0);
				__m128i int32_1 = _mm_cvtps_epi32(float_1);

				__m256i result = _mm256_insertf128_si256(_mm256_castsi128_si256(int32_0), int32_1, 1);

				return v_uint32x8(result);
			}
		};

		template<> struct v_merge_Invoker<v_int32x8, v_uint64x4>
		{
			const v_uint64x4& src_0;
			const v_uint64x4& src_1;
			v_merge_Invoker(const v_uint64x4& src_0_, const v_uint64x4& src_1_) noexcept : src_0(src_0_), src_1(src_1_) {}

			v_int32x8 operator () () const noexcept
			{
				return v_int32x8(
					saturate_cast<i32>(src_0.data.m256i_u64[0]), saturate_cast<i32>(src_0.data.m256i_u64[1]),
					saturate_cast<i32>(src_0.data.m256i_u64[2]), saturate_cast<i32>(src_0.data.m256i_u64[3]),
					saturate_cast<i32>(src_1.data.m256i_u64[0]), saturate_cast<i32>(src_1.data.m256i_u64[1]),
					saturate_cast<i32>(src_1.data.m256i_u64[2]), saturate_cast<i32>(src_1.data.m256i_u64[3])
				);
			}
		};

		template<> struct v_merge_Invoker<v_int32x8, v_int64x4>
		{
			const v_int64x4& src_0;
			const v_int64x4& src_1;
			v_merge_Invoker(const v_int64x4& src_0_, const v_int64x4& src_1_) noexcept : src_0(src_0_), src_1(src_1_) {}

			v_int32x8 operator () () const noexcept
			{
				return v_int32x8(
					saturate_cast<i32>(src_0.data.m256i_i64[0]), saturate_cast<i32>(src_0.data.m256i_i64[1]),
					saturate_cast<i32>(src_0.data.m256i_i64[2]), saturate_cast<i32>(src_0.data.m256i_i64[3]),
					saturate_cast<i32>(src_1.data.m256i_i64[0]), saturate_cast<i32>(src_1.data.m256i_i64[1]),
					saturate_cast<i32>(src_1.data.m256i_i64[2]), saturate_cast<i32>(src_1.data.m256i_i64[3])
				);
			}
		};

		template<> struct v_merge_Invoker<v_int32x8, v_float64x4>
		{
			const v_float64x4& src_0;
			const v_float64x4& src_1;
			v_merge_Invoker(const v_float64x4& src_0_, const v_float64x4& src_1_) noexcept : src_0(src_0_), src_1(src_1_) {}

			v_int32x8 operator () () const noexcept
			{
				__m128 float_0 = _mm256_cvtpd_ps(src_0.data);
				__m128 float_1 = _mm256_cvtpd_ps(src_1.data);

				__m128i int32_0 = _mm_cvtps_epi32(float_0);
				__m128i int32_1 = _mm_cvtps_epi32(float_1);

				__m256i result = _mm256_insertf128_si256(_mm256_castsi128_si256(int32_0), int32_1, 1);

				return v_int32x8(result);
			}
		};

		template<VectorType Dst_t, VectorType ... Src_t> Dst_t v_merge_as(const Src_t& ... src)
		{
			static_assert(sizeof...(Src_t) > 1, "The vectors used for merging must not be less than 2");
			static_assert((std::is_same_v<Src_t, typename std::tuple_element_t<0, std::tuple<Src_t...>>> && ...), "All source vector types must be the same");
			static_assert((Src_t::batch_size + ...) == Dst_t::batch_size, "Sum of source batch sizes must equal destination batch size");
			v_merge_Invoker<Dst_t, std::tuple_element_t<0, std::tuple<Src_t...>>> invoker(src ...);
			return invoker();
		}
	}
}


export namespace fy
{
	namespace simd
	{
		template<VectorType Src_t> Src_t v_reverse(const Src_t& a) { std::unreachable(); }


		template<> v_uint8x32 v_reverse(const v_uint8x32& a)
		{
			static const __m256i perm = _mm256_setr_epi8(
				15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0,
				15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0
			);

			__m256i vec = _mm256_shuffle_epi8(a.data, perm);
			__m256i res = _mm256_permute2x128_si256(vec, vec, 1);

			return v_uint8x32(res);
		}

		inline v_uint16x16 v_reverse(const v_uint16x16& a)
		{
			static const __m256i perm = _mm256_setr_epi8(
				14, 15, 12, 13, 10, 11, 8, 9, 6, 7, 4, 5, 2, 3, 0, 1,
				14, 15, 12, 13, 10, 11, 8, 9, 6, 7, 4, 5, 2, 3, 0, 1
			);

			__m256i vec = _mm256_shuffle_epi8(a.data, perm);
			__m256i res = _mm256_permute2x128_si256(vec, vec, 1);

			return v_uint16x16(res);
		}

		template<> v_uint32x8 v_reverse(const v_uint32x8& a)
		{
			static const __m256i perm = _mm256_setr_epi32(7, 6, 5, 4, 3, 2, 1, 0);
			__m256i res = _mm256_permutevar8x32_epi32(a.data, perm);

			return v_uint32x8(res);
		}

		template<> v_uint64x4 v_reverse(const v_uint64x4& a)
		{
			return v_uint64x4(_mm256_permute4x64_epi64(a.data, _MM_SHUFFLE(0, 1, 2, 3)));
		}

		template<> v_int8x32 v_reverse(const v_int8x32& a)
		{
			return v_int8x32(v_reverse(v_reinterpret_convert<v_uint8x32>(a)));
		}

		template<> v_int16x16 v_reverse(const v_int16x16& a)
		{
			return v_int16x16(v_reverse(v_reinterpret_convert<v_uint16x16>(a)));
		}

		template<> v_int32x8 v_reverse(const v_int32x8& a)
		{
			return v_int32x8(v_reverse(v_reinterpret_convert<v_uint32x8>(a)));
		}

		template<> v_int64x4 v_reverse(const v_int64x4& a)
		{
			return v_int64x4(v_reverse(v_reinterpret_convert<v_uint64x4>(a)));
		}

		inline v_float16x16 v_reverse(const v_float16x16& a)
		{
			static const __m256i perm = _mm256_setr_epi8(
				14, 15, 12, 13, 10, 11, 8, 9, 6, 7, 4, 5, 2, 3, 0, 1,
				14, 15, 12, 13, 10, 11, 8, 9, 6, 7, 4, 5, 2, 3, 0, 1
			);

			__m256i vec = _mm256_shuffle_epi8(a.data, perm);
			__m256i res = _mm256_permute2x128_si256(vec, vec, 1);

			return v_float16x16(res);
		}

		template<> v_float32x8 v_reverse(const v_float32x8& a)
		{
			static const __m256i perm = _mm256_setr_epi32(7, 6, 5, 4, 3, 2, 1, 0);

			__m256i raw = _mm256_castps_si256(a.data);
			__m256i res = _mm256_permutevar8x32_epi32(raw, perm);
			__m256 vres_f = _mm256_castsi256_ps(res);

			return v_float32x8(vres_f);
		}

		template<> v_float64x4 v_reverse(const v_float64x4& a)
		{
			__m256i raw = _mm256_castpd_si256(a.data);
			__m256i res = _mm256_permute4x64_epi64(raw, _MM_SHUFFLE(0, 1, 2, 3));
			__m256d vres_f = _mm256_castsi256_pd(res);
			return v_float64x4(vres_f);
		}
	}
}


export namespace fy
{
	namespace simd
	{
		template<VectorType Vec_t> bool v_any_match(const Vec_t&) { std::unreachable(); }
		template<VectorType Vec_t> bool v_all_match(const Vec_t&) { std::unreachable(); }

		template<> bool v_any_match(const v_uint8x32& mask) { return _mm256_movemask_epi8(mask.data) != 0; }
		template<> bool v_all_match(const v_uint8x32& mask) { return _mm256_movemask_epi8(mask.data) == -1; }
	}
}

export namespace fy
{
	namespace simd
	{
		template<VectorType T>
		auto v_reduce_sum(const T&)
		{
			std::unreachable();
		}

		u32 v_reduce_sum(const v_uint8x32& a)
		{
			__m256i half = _mm256_sad_epu8(a.data, _mm256_setzero_si256());
			__m128i quarter = _mm_add_epi32(_mm256_castsi256_si128(half), _mm256_extracti128_si256(half, 1));
			return static_cast<u32>(_mm_cvtsi128_si32(_mm_add_epi32(quarter, _mm_unpackhi_epi64(quarter, quarter))));
		}

		i32 v_reduce_sum(const v_int8x32& a)
		{
			__m256i half = _mm256_sad_epu8(_mm256_xor_si256(a.data, _mm256_set1_epi8(static_cast<i8>(-128))), _mm256_setzero_si256());
			__m128i quarter = _mm_add_epi32(_mm256_castsi256_si128(half), _mm256_extracti128_si256(half, 1));
			return static_cast<i32>(_mm_cvtsi128_si32(_mm_add_epi32(quarter, _mm_unpackhi_epi64(quarter, quarter)))) - 4096;
		}

		i32 v_reduce_sum(const v_int32x8& a)
		{
			__m256i s0 = _mm256_hadd_epi32(a.data, a.data);
			s0 = _mm256_hadd_epi32(s0, s0);

			__m128i s1 = _mm256_extracti128_si256(s0, 1);
			s1 = _mm_add_epi32(_mm256_castsi256_si128(s0), s1);

			return static_cast<i32>(_mm_cvtsi128_si32(s1));
		}

		u32 v_reduce_sum(const v_uint32x8& a)
		{
			return v_reduce_sum(v_int32x8(a.data));
		}

		i32 v_reduce_sum(const v_int16x16& a)
		{
			v_int32x8 res_0(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(a.data)));
			v_int32x8 res_1(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(a.data, 1)));
			return v_reduce_sum(v_add(res_0, res_1));
		}

		u32 v_reduce_sum(const v_uint16x16& a)
		{
			v_uint32x8 res_0(_mm256_cvtepu16_epi32(_mm256_castsi256_si128(a.data)));
			v_uint32x8 res_1(_mm256_cvtepu16_epi32(_mm256_extracti128_si256(a.data, 1)));
			return v_reduce_sum(v_add(res_0, res_1));
		}

		u64 v_reduce_sum(const v_uint64x4& a)
		{
			alignas(32) u64 idx[2];
			_mm_store_si128(reinterpret_cast<__m128i*>(idx), _mm_add_epi64(_mm256_castsi256_si128(a.data), _mm256_extracti128_si256(a.data, 1)));
			return idx[0] + idx[1];
		}

		i64 v_reduce_sum(const v_int64x4& a)
		{
			alignas(32) i64 idx[2];
			_mm_store_si128(reinterpret_cast<__m128i*>(idx), _mm_add_epi64(_mm256_castsi256_si128(a.data), _mm256_extracti128_si256(a.data, 1)));
			return idx[0] + idx[1];
		}

		f32 v_reduce_sum(const v_float32x8& a)
		{
			__m256 s0 = _mm256_hadd_ps(a.data, a.data);
			s0 = _mm256_hadd_ps(s0, s0);

			__m128 s1 = _mm256_extractf128_ps(s0, 1);
			s1 = _mm_add_ps(_mm256_castps256_ps128(s0), s1);

			return static_cast<f32>(_mm_cvtss_f32(s1));
		}

		f64 v_reduce_sum(const v_float64x4& a)
		{
			__m256d s0 = _mm256_hadd_pd(a.data, a.data);
			return static_cast<f64>(_mm_cvtsd_f64(_mm_add_pd(_mm256_castpd256_pd128(s0), _mm256_extractf128_pd(s0, 1))));
		}

	}
}