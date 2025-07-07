export module foye.extensionType.float8;

import std;
import foye.alias;
import foye.math_utility;

export namespace fy
{
    /*
        E4M3 (4 位指数和 3 位尾数)
        E5M2 (5 位指数和 2 位尾数)

        E5M2 遵循 IEEE 754 的特殊值表示规范
        E4M3 通过不表示无穷大且仅使用一个尾数位模式表示 NaN 来扩展动态范围
    */

    struct float8_E5M2
    {
        constexpr float8_E5M2() noexcept : bits_(0) {}

        constexpr float8_E5M2(f32 from_float) noexcept
        {
            if (fy::isnan(from_float))
            {
                // 设置NaN：符号位 + 全1指数 + 非零尾数
                bits_ = (std::signbit(from_float) << 7) | 0x7C;
                return;
            }

            u32 f32_bits = std::bit_cast<u32>(from_float);
            u32 sign = (f32_bits >> 31) & 1;
            u32 e32 = (f32_bits >> 23) & 0xFF;
            u32 m32 = f32_bits & 0x7FFFFF;

            // 处理无穷大
            if (e32 == 0xFF)
            {
                bits_ = (sign << 7) | 0x78; // 0x78 = 0b1111000
                return;
            }

            i32 exp32;
            u32 mantissa32;

            if (e32 == 0)
            {
                exp32 = -126;
                mantissa32 = m32; // 无隐含位
            }
            else
            {
                exp32 = e32 - 127;
                mantissa32 = (1 << 23) | m32; // 包含隐含位
            }

            f32 abs_val = std::abs(from_float);

            // 处理极小值（非正规数）
            if (exp32 < -14)
            {
                // 若值小于最小非正规数，返回0
                if (abs_val < (1.0f / 16384.0f))
                {
                    bits_ = sign << 7;
                    return;
                }

                // 缩放至非正规数范围内
                f32 scaled = abs_val * 16384.0f; // 2^14
                u32 fraction = static_cast<u32>(scaled + 0.5f);
                fraction = std::clamp(fraction, 1u, 3u); // 尾数范围1-3
                bits_ = (sign << 7) | fraction; // 指数=0
                return;
            }

            i32 e5 = exp32 + 15; // E5M2偏移15

            // 处理溢出
            if (exp32 > 15)
            {
                bits_ = (sign << 7) | 0x78; // 无穷大
                return;
            }

            // 提取并舍入尾数（保留高2位）
            u32 base = (mantissa32 >> 21) & 0x7; // 高3位
            u32 rest = mantissa32 & 0x1FFFFF;    // 低21位

            // 标准四舍五入（向偶数舍入）
            bool round_up = false;
            if (rest > 0x100000)
            { // >0.5 in LSB
                round_up = true;
            }
            else if (rest == 0x100000)
            { // 中间值
                round_up = (base & 0x1) != 0; // 向偶数舍入
            }

            if (round_up)
            {
                base++;
                if (base > 7)
                {   // 若进位后溢出
                    base = 0;
                    e5++;
                }
            }

            // 再次检查指数溢出
            if (e5 > 30 || exp32 < -14)
            {
                bits_ = (sign << 7) | 0x78;
                return;
            }

            bits_ = (sign << 7) | (e5 << 2) | (base & 0x3); // 仅保留2位尾数
        }

        constexpr operator f32() const noexcept
        {
            u8 sign = bits_ >> 7;
            u8 exp = (bits_ >> 2) & 0x1F;
            u8 mantissa = bits_ & 0x03;

            // 处理特殊值
            if (exp == 0x1F)
            {
                if (mantissa != 0)
                {
                    return std::bit_cast<f32>(0x7FC00000 | (sign << 31)); // NaN
                }
                return std::bit_cast<f32>((sign << 31) | 0x7F800000); // Inf
            }

            // 处理非正规数
            if (exp == 0)
            {
                if (mantissa == 0)
                {
                    return std::bit_cast<f32>(sign << 31); // ±0
                }
                // 值 = mantissa * 2^-14
                u32 f32_val = (sign << 31) | (112 << 23) | (mantissa << 21);
                return std::bit_cast<f32>(f32_val);
            }

            // 正规数：计算真实尾数
            u32 true_mantissa = (0x04 | (mantissa >> 1)) << 21; // 1.xx
            i32 exp32 = static_cast<i32>(exp) - 15 + 127;
            u32 f32_val = (sign << 31) | (exp32 << 23) | true_mantissa;

            return std::bit_cast<f32>(f32_val);
        }

        u8 bits_;
    };
}

#if 0
export namespace fy
{
	/*
		E4M3 (4 位指数和 3 位尾数)
		E5M2 (5 位指数和 2 位尾数)

		E5M2 遵循 IEEE 754 的特殊值表示规范
		E4M3 通过不表示无穷大且仅使用一个尾数位模式表示 NaN 来扩展动态范围
	*/

    enum class fp8_encode_type { E5M2, E4M3 };
    template<fp8_encode_type encode_t> struct float8__;

    using float8_E5M2 = float8__<fp8_encode_type::E5M2>;
    using float8_E4M3 = float8__<fp8_encode_type::E4M3>;



    template<fp8_encode_type encode_t> struct float8__
    {
        constexpr float8__() noexcept : bits_(0) {}
        explicit constexpr float8__(f32 from_float) noexcept;
        constexpr operator f32() const noexcept;

        //constexpr float8__ operator-() const noexcept;
        //constexpr float8__ abs() const noexcept;

        //constexpr bool isNaN() const noexcept;
        //constexpr bool isZero() const noexcept;

        //static constexpr float8__ negativeZero() noexcept;


        //static constexpr float8__ abs(float8__ value) noexcept;
        //static constexpr float8__ round(f32 x) noexcept;
        //static constexpr float8__ round(f64 x) noexcept;

        u8 bits_;
    };
}

namespace fy
{
    constexpr float8_E5M2::float8__(f32 from_float) noexcept
    {
        if (fy::isnan(from_float))
        {
            bits_ = (std::signbit(from_float) << 7) | (0x1F << 2) | 0x03;
            return;
        }

        u32 f32_bits = std::bit_cast<u32>(from_float);
        u32 sign = (f32_bits >> 31) & 1;
        u32 e32 = (f32_bits >> 23) & 0xFF;
        u32 m32 = f32_bits & 0x7FFFFF;

        if (e32 == 0xFF)
        {
            bits_ = (sign << 7) | (0x1F << 2) | 0x00;
            return;
        }

        i32 exp32;
        u32 mantissa32;

        if (e32 == 0)
        {
            exp32 = -126;
            mantissa32 = m32;
        }
        else
        {
            exp32 = e32 - 127;
            mantissa32 = (1 << 23) | m32;
        }

        f32 abs_val = std::abs(from_float);

        if (exp32 < -16)
        {
            bits_ = sign << 7;
            return;
        }

        if (exp32 < -14)
        {
            f32 scaled = abs_val * 65536.0f;
            u32 fraction = static_cast<u32>(scaled + 0.5f);

            fraction = std::clamp(fraction, 1u, 3u);
            bits_ = (sign << 7) | (0x00 << 2) | fraction;
            return;
        }

        i32 e5 = exp32 + 15;
        if (e5 > 30)
        {
            bits_ = (sign << 7) | (0x1F << 2) | 0x00;
            return;
        }

        u32 base = (mantissa32 >> 21) & 0x7;
        u32 rest = mantissa32 & 0x1FFFFF;
        u32 round_bit = (rest > 0x100000) ? 1 : 0;
        u32 total = base + round_bit;

        if (total >= 4)
        {
            e5++;
            if (e5 > 30)
            {
                bits_ = (sign << 7) | (0x1F << 2) | 0x00;
                return;
            }
            bits_ = (sign << 7) | (e5 << 2) | 0x00;
        }
        else
        {
            bits_ = (sign << 7) | (e5 << 2) | total;
        }
    }

    constexpr float8_E4M3::float8__(f32 from_float) noexcept
    {
        if (fy::isnan(from_float))
        {
            bits_ = (std::signbit(from_float) << 7) | 0x7F;
            return;
        }

        u32 f32_bits = std::bit_cast<u32>(from_float);
        u32 sign = (f32_bits >> 31) & 1;
        u32 e32 = (f32_bits >> 23) & 0xFF;
        u32 m32 = f32_bits & 0x7FFFFF;

        if (e32 == 0xFF)
        {
            bits_ = (sign << 7) | (0x0E << 3) | 0x07;
            return;
        }

        f32 abs_val = std::abs(from_float);
        if (abs_val < 0x1p-9f)
        {
            bits_ = sign << 7;
            return;
        }

        if (abs_val < 0x1p-6f)
        {
            u32 fraction = static_cast<u32>(abs_val * 512.0f + 0.5f);
            if (fraction >= 8)
            {
                bits_ = (sign << 7) | (0x01 << 3) | 0x00;
            }
            else
            {
                fraction = std::max(1u, fraction);
                bits_ = (sign << 7) | (0x00 << 3) | fraction;
            }
            return;
        }

        i32 exp32 = e32 ? e32 - 127 : -126;
        i32 e4 = exp32 + 7;
        if (e4 > 14)
        {
            bits_ = (sign << 7) | (0x0E << 3) | 0x07;
            return;
        }
        else if (e4 < 1)
        {
            e4 = 1;
        }

        u32 mantissa32 = e32 ? (1 << 23) | m32 : m32;
        u32 base = (mantissa32 >> 20) & 0x7;
        u32 rest = mantissa32 & 0xFFFFF;
        u32 round_bit = (rest > 0x80000) ? 1 : 0;
        u32 total = base + round_bit;

        if (total >= 8)
        {
            e4++;
            if (e4 > 14)
            {
                bits_ = (sign << 7) | (0x0E << 3) | 0x07;
                return;
            }
            bits_ = (sign << 7) | (e4 << 3) | 0;
        }
        else
        {
            bits_ = (sign << 7) | (e4 << 3) | total;
        }
    }



    constexpr float8__<fp8_encode_type::E5M2>::operator f32() const noexcept
    {
        const u8 sign = bits_ >> 7;
        const u8 exp = (bits_ >> 2) & 0x1F;
        const u8 mantissa = bits_ & 0x03;

        if (exp == 0x1F)
        {
            if (mantissa != 0)
            {
                return std::bit_cast<f32>(0x7FC00000 | (sign << 31));
            }
            return std::bit_cast<f32>((sign << 31) | 0x7F800000);
        }

        if (exp == 0)
        {
            if (mantissa == 0)
            {
                return std::bit_cast<f32>(sign << 31); // ±0
            }
            const u32 f32_val =
                (sign << 31) |
                ((127 - 14) << 23) |
                ((static_cast<u32>(mantissa) | 0x4) << 21);
            return std::bit_cast<f32>(f32_val);
        }

        u32 mantissa32 = (0x80 | (mantissa << 6)) << 15;
        i32 exp32 = static_cast<i32>(exp) - 15 + 127;

        if (exp32 < 0)
        {
            mantissa32 >>= (-exp32 + 1);
            exp32 = 0;
        }

        const u32 f32_val =
            (sign << 31) |
            (static_cast<u32>(exp32) << 23) |
            (mantissa32 & 0x7FFFFF);

        return std::bit_cast<f32>(f32_val);
    }

    constexpr float8__<fp8_encode_type::E4M3>::operator f32() const noexcept
    {
        const u8 sign = bits_ >> 7;
        const u8 exp = (bits_ >> 3) & 0x0F;
        const u8 mantissa = bits_ & 0x07;

        if ((bits_ & 0x7F) == 0x7F)
        {
            return std::bit_cast<f32>(0x7FC00000 | (sign << 31));
        }

        if (exp == 0)
        {
            if (mantissa == 0)
            {
                return std::bit_cast<f32>(sign << 31);
            }
            return std::bit_cast<f32>(
                (sign << 31) |
                ((127 - 9) << 23) |
                (static_cast<u32>(mantissa) << 20)
            );
        }

        const u32 mantissa32 = (0x80 | (mantissa << 5)) << 15;
        i32 exp32 = static_cast<i32>(exp) - 7 + 127;

        if (exp32 < 0)
        {
            u32 shifted_mantissa = mantissa32 >> (-exp32 + 1);
            return std::bit_cast<f32>(
                (sign << 31) |
                shifted_mantissa
            );
        }

        return std::bit_cast<f32>(
            (sign << 31) |
            (static_cast<u32>(exp32) << 23) |
            (mantissa32 & 0x7FFFFF)
        );
    }

}

#endif