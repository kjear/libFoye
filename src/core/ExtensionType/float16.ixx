export module foye.extensionType.float16;

import std;
import foye.alias;

namespace fy
{
	/*union Bits64
	{
		i64 i; u64 u; f64 d;
		constexpr Bits64() noexcept : i(0) {}
		constexpr explicit Bits64(i64 val) noexcept : i(val) {}
		constexpr explicit Bits64(u64 val) noexcept : u(val) {}
		constexpr explicit Bits64(f64 val) noexcept : d(val) {}
	};*/

	/*union Bits32
	{
		i32 i; u32 u; f32 f;
		constexpr Bits32() noexcept : i(0) {}
		constexpr explicit Bits32(i32 val) noexcept : i(val) {}
		constexpr explicit Bits32(u32 val) noexcept : u(val) {}
		constexpr explicit Bits32(f32 val) noexcept : f(val) {}
	};*/
}

export namespace fy
{
	struct f16 final
	{
		constexpr f16() noexcept : w(0) {}

		template<typename T> requires std::integral<T>
		constexpr explicit f16(T x) noexcept
		{
			if constexpr (sizeof(T) <= sizeof(f32))
			{
				*this = f16(static_cast<f32>(x));
				return;
			}
			else if constexpr (sizeof(T) <= sizeof(f64))
			{
				*this = f16(static_cast<f64>(x));
				return;
			}
		}

		constexpr explicit f16(f32 x) noexcept
		{
			u32 in = std::bit_cast<u32>(x);
			u32 sign = in & 0x80000000;
			in ^= sign;

			if (in >= 0x47800000)
			{
				if (in > 0x7f800000)
				{
					w = static_cast<u16>(0x7e00);
				}
				else
				{
					w = static_cast<u16>(0x7c00);
				}
			}
			else
			{
				if (in < 0x38800000)
				{
					f32 temp = std::bit_cast<f32>(in) + 0.5f;
					in = std::bit_cast<u32>(temp);
					w = static_cast<u16>(in - 0x3f000000);
				}
				else
				{
					u32 t = in + 0xc8000fff;
					w = static_cast<u16>((t + ((in >> 13) & 1)) >> 13);
				}
			}

			w = static_cast<u16>(w | (sign >> 16));
		}

		constexpr explicit f16(f64 x) noexcept
		{
			union
			{
				i64 i; u64 u; f64 d;
			} in;

			in.d = x;

			u64 sign = in.u & 0x8000000000000000ULL;
			in.u ^= sign;

			if (in.u >= 0x7FF0000000000000ULL)
			{
				if (in.u > 0x7FF0000000000000ULL)
				{
					w = static_cast<u16>(0x7E00);
				}
				else
				{
					w = static_cast<u16>(0x7C00);
				}
			}
			else if (in.u >= 0x47EFFFFFFFFFFFFFULL)
			{
				w = static_cast<u16>(0x7C00);
			}
			else if (in.u < 0x3F00000000000000ULL)
			{
				u64 m = (in.u & 0xFFFFFFFFFFFFFULL) | 0x10000000000000ULL;
				i32 e = static_cast<i32>((in.u >> 52) & 0x7FF) - 1008;

				while (e < -14)
				{
					m >>= 1;
					++e;
				}

				w = static_cast<u16>((m >> (52 - 10 + 14 + e)) & 0x3FF);
			}
			else
			{
				u64 m = in.u & 0xFFFFFFFFFFFFFULL;
				i32 e = static_cast<i32>(((in.u) >> 52) & 0x7FF) - 1023 + 15;

				w = static_cast<u16>((e << 10) | (m >> 42));
			}

			w |= static_cast<u16>(sign >> 48);
		}

		constexpr operator f32() const noexcept
		{
			union
			{
				i32 i; u32 u; f32 f;
			} out;

			u32 t = ((w & 0x7fff) << 13) + 0x38000000;
			u32 sign = (w & 0x8000) << 16;
			u32 e = w & 0x7c00;

			out.u = t + (1 << 23);
			if (e >= 0x7c00)
			{
				out.u = t + 0x38000000;
			}
			else if (e == 0)
			{
				out.f -= 6.103515625e-05f;
				out.u = out.u;
			}
			else
			{
				out.u = t;
			}
			out.u |= sign;
			return out.f;
		}

		constexpr void operator ++ () noexcept { *this += f16(1); }
		constexpr void operator -- () noexcept { *this -= f16(1); }

		constexpr f16 operator-() const noexcept
		{
			if (isNaN())
			{
				return *this;
			}

			if (isZero())
			{
				return f16::negativeZero();
			}

			f16 result;
			result.w = w ^ 0x8000;
			return result;
		}

		constexpr bool isNaN() const noexcept
		{
			return (w & 0x7C00) == 0x7C00 && (w & 0x03FF) != 0;
		}

		constexpr bool isZero() const noexcept
		{
			return (w & 0x7FFF) == 0;
		}

		constexpr f16 operator + (const f16& other) const { return f16::round(static_cast<f32>(*this) + static_cast<f32>(other)); }
		constexpr f16 operator - (const f16& other) const { return f16::round(static_cast<f32>(*this) - static_cast<f32>(other)); }
		constexpr f16 operator * (const f16& other) const { return f16::round(static_cast<f32>(*this) * static_cast<f32>(other)); }
		constexpr f16 operator / (const f16& other) const { return f16::round(static_cast<f32>(*this) / static_cast<f32>(other)); }

		constexpr f16& operator += (const f16& other) { *this = *this + other; return *this; }
		constexpr f16& operator -= (const f16& other) { *this = *this - other; return *this; }
		constexpr f16& operator *= (const f16& other) { *this = *this * other; return *this; }
		constexpr f16& operator /= (const f16& other) { *this = *this / other; return *this; }

		constexpr bool operator == (const f16& other) const { return w == other.w; }
		constexpr bool operator != (const f16& other) const { return w != other.w; }

		constexpr bool operator < (const f16& other) const { return static_cast<f32>(*this) < static_cast<f32>(other); }
		constexpr bool operator > (const f16& other) const { return static_cast<f32>(*this) > static_cast<f32>(other); }
		constexpr bool operator <= (const f16& other) const { return static_cast<f32>(*this) <= static_cast<f32>(other); }
		constexpr bool operator >= (const f16& other) const { return static_cast<f32>(*this) >= static_cast<f32>(other); }

		constexpr f16 abs() const
		{
			f16 result;
			result.w = w & 0x7FFF;
			return result;
		}

		static constexpr f16 round(f64 x)
		{
			return round(static_cast<f32>(x));
		}

		static constexpr f16 round(f32 x)
		{
			union Bits32
			{
				i32 i; u32 u; f32 f;
			} in;

			in.f = x;
			u32 sign = in.u & 0x80000000;
			in.u ^= sign;

			f16 result{};

			if (in.u >= 0x47800000)
			{
				if (in.u > 0x7f800000)
				{
					result.w = 0x7e00;
				}
				else
				{
					result.w = 0x7c00;
				}
			}
			else
			{
				if (in.u < 0x38800000)
				{
					f32 rounded = std::round(x * 0x1p+24f) * 0x1p-24f;
					return f16(rounded);
				}
				else
				{
					u32 r = ((in.u >> 16) + 1) & 1;
					result.w = static_cast<u16>((in.u >> 13) + r);
				}
			}

			result.w |= static_cast<u16>(sign >> 16);
			return result;
		}

		static constexpr f16 negativeZero()
		{
			f16 result;
			result.w = 0x8000;
			return result;
		}

		static constexpr f16 abs(f16 value)
		{
			f16 result;
			result.w = value.w & 0x7FFF;
			return result;
		}

		static constexpr f16 hfloatFromBits(u16 w)
		{
			union
			{
				i32 i; u32 u; f32 f;
			} out;

			u32 t = ((w & 0x7fff) << 13) + 0x38000000;
			u32 sign = (w & 0x8000) << 16;
			u32 e = w & 0x7c00;

			out.u = t + (1 << 23);
			if (e >= 0x7c00)
			{
				out.u = t + 0x38000000;
			}
			else if (e == 0)
			{
				out.f -= 6.103515625e-05f;
				out.u = out.u;
			}
			else
			{
				out.u = t;
			}
			out.u |= sign;
			f16 res(out.f);
			return res;
		}

		u16 w;
	};


	constexpr f16 operator "" _fp16(long double c) { return f16(static_cast<f64>(c)); }
	constexpr f16 operator "" _fp16(unsigned long long c) { return f16(c); }


	using float16 = f16;
}