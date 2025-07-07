export module foye.extensionType.bfloat16;

import std;
import foye.alias;

export namespace fy
{
	struct bfloat16
	{
		constexpr bfloat16() noexcept : bits_(0) { }

		constexpr explicit bfloat16(f32 from_f32) noexcept
		{
            union
            {
                f32 f;
                u32 u;
            } converter{ from_f32 };
            u32 u32_val = converter.u;

            if (((u32_val & exponent_mask) == exponent_mask) && (u32_val & mantissa_mask))
            {
                bits_ = static_cast<u16>(u32_val >> 16);
                if ((bits_ & 0x007F) == 0)
                {
                    bits_ |= 0x0001;
                }
            }
            else
            {
                bits_ = static_cast<u16>(u32_val >> 16);
            }
		}

        constexpr operator f32() const noexcept
        {
            union
            {
                u32 u;
                f32 f;
            } converter{};

            converter.u = static_cast<u32>(bits_) << 16;
            return converter.f;
        }

        constexpr bfloat16 operator-() const noexcept
        {
            bfloat16 result;
            result.bits_ = bits_ ^ 0x8000;
            return result;
        }

        static constexpr bfloat16 bfloatFromBits(u16 w)
        {
            u32 u32_val = static_cast<u32>(w) << 16;

            if (((u32_val & exponent_mask) == exponent_mask) && (u32_val & mantissa_mask))
            {
                u32_val |= 0x00400000;
                u32_val &= ~0x00200000;
            }
            else if ((u32_val & exponent_mask) == exponent_mask)
            {
                u32_val &= ~mantissa_mask;
            }

            bfloat16 result;
            result.bits_ = static_cast<u16>(u32_val >> 16);
            return result;
        }

        constexpr bool isNaN() const noexcept
        {
            return ((bits_ & exponent_mask) == exponent_mask) &&
                ((bits_ & mantissa_mask) != 0);
        }

        constexpr bool isZero() const noexcept
        {
            return (bits_ & non_sign_mask) == 0;
        }

        static constexpr bfloat16 min() noexcept
        {
            return bfloatFromBits(0x0080);
        }

        static constexpr bfloat16 max() noexcept
        {
            return bfloatFromBits(0x7F7F);
        }

        static constexpr bfloat16 lowest() noexcept
        {
            return bfloatFromBits(0xFF7F);
        }

        static constexpr bfloat16 negativeZero() noexcept
        {
            return bfloatFromBits(0x8000);
        }

        static constexpr bfloat16 epsilon() noexcept
        {
            return bfloatFromBits(0x3C00);
        }

        static constexpr bfloat16 abs(bfloat16 value) noexcept
        {
            return bfloatFromBits(value.bits_ & 0x7FFF);
        }

        constexpr bfloat16 abs() const
        {
            return bfloat16::abs(*this);
        }

        static constexpr bfloat16 round(f64 x)
        {
            if (x != x) { return bfloatFromBits(0x7FC0); }
            if (x > static_cast<f64>(max().operator f32())) return max();
            if (x < static_cast<f64>(lowest().operator f32())) return lowest();
            if (x == std::numeric_limits<f64>::infinity()) return bfloatFromBits(0x7F80);
            if (x == -std::numeric_limits<f64>::infinity()) return bfloatFromBits(0xFF80);

            f64 int_part;
            f64 frac_part = std::modf(x, &int_part);

            if (x >= 0)
            {
                if (frac_part > 0.5)
                {
                    int_part += 1.0;
                }
                else if (frac_part == 0.5)
                {
                    if (std::fmod(int_part, 2.0) != 0.0)
                    {
                        int_part += 1.0;
                    }
                }
            }
            else
            {
                if (frac_part < -0.5)
                {
                    int_part -= 1.0;
                }
                else if (frac_part == -0.5)
                {
                    if (std::fmod(int_part, 2.0) != 0.0)
                    {
                        int_part -= 1.0;
                    }
                }
            }

            return bfloat16(static_cast<f32>(int_part));
        }

        static constexpr bfloat16 round(f32 x)
        {
            if (x != x) return bfloatFromBits(0x7FC0);
            if (x > max().operator f32()) return max();
            if (x < lowest().operator f32()) return lowest();
            if (x == std::numeric_limits<f32>::infinity()) return bfloatFromBits(0x7F80);
            if (x == -std::numeric_limits<f32>::infinity()) return bfloatFromBits(0xFF80);

            f32 int_part;
            f32 frac_part = std::modf(x, &int_part);

            if (x >= 0)
            {
                if (frac_part > 0.5f)
                {
                    int_part += 1.0f;
                }
                else if (frac_part == 0.5f)
                {
                    if (std::fmod(int_part, 2.0f) != 0.0f)
                    {
                        int_part += 1.0f;
                    }
                }
            }
            else
            {
                if (frac_part < -0.5f)
                {
                    int_part -= 1.0f;
                }
                else if (frac_part == -0.5f)
                {
                    if (std::fmod(int_part, 2.0f) != 0.0f)
                    {
                        int_part -= 1.0f;
                    }
                }
            }

            return bfloat16(int_part);
        }

        constexpr bfloat16 operator + (bfloat16 other) const { return bfloat16::round(static_cast<f32>(*this) + static_cast<f32>(other)); }
        constexpr bfloat16 operator - (bfloat16 other) const { return bfloat16::round(static_cast<f32>(*this) - static_cast<f32>(other)); }
        constexpr bfloat16 operator * (bfloat16 other) const { return bfloat16::round(static_cast<f32>(*this) * static_cast<f32>(other)); }
        constexpr bfloat16 operator / (bfloat16 other) const { return bfloat16::round(static_cast<f32>(*this) / static_cast<f32>(other)); }

        constexpr bfloat16& operator += (bfloat16 other) { *this = *this + other; return *this; }
        constexpr bfloat16& operator -= (bfloat16 other) { *this = *this - other; return *this; }
        constexpr bfloat16& operator *= (bfloat16 other) { *this = *this * other; return *this; }
        constexpr bfloat16& operator /= (bfloat16 other) { *this = *this / other; return *this; }

        constexpr bool operator == (bfloat16 other) const { return bits_ == other.bits_; }
        constexpr bool operator != (bfloat16 other) const { return bits_ != other.bits_; }

        constexpr bool operator < (bfloat16 other) const { return static_cast<f32>(*this) < static_cast<f32>(other); }
        constexpr bool operator > (bfloat16 other) const { return static_cast<f32>(*this) > static_cast<f32>(other); }
        constexpr bool operator <= (bfloat16 other) const { return static_cast<f32>(*this) <= static_cast<f32>(other); }
        constexpr bool operator >= (bfloat16 other) const { return static_cast<f32>(*this) >= static_cast<f32>(other); }

        static constexpr u32 exponent_mask = 0x7F800000;
        static constexpr u32 mantissa_mask = 0x007FFFFF;
        static constexpr u16 non_sign_mask = 0x7FFF;
		u16 bits_;
	};


    constexpr bfloat16 operator "" _bf16(long double c) { return bfloat16(static_cast<f32>(c)); }
    constexpr bfloat16 operator "" _bf16(unsigned long long c) { return bfloat16(static_cast<f32>(static_cast<f64>(c))); }

    using bf16 = bfloat16;
}