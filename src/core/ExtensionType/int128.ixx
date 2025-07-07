module;
#include <Windows.h>
#undef min
#undef max
#include <intrin.h>

export module foye.extensionType.int128;

import std;
import foye.alias;

namespace fy
{
    constexpr u8 add_carry64(u8 carry, u64 lhs, u64 rhs, u64& res) noexcept
    {
        if consteval
        {
            const u64 sum = lhs + rhs + carry;
            res = sum;
            return carry ? sum <= lhs : sum < lhs;
        }
        else
        {
            return _addcarry_u64(carry, lhs, rhs, &res);
        }
    }

    constexpr u8 sub_borrow64(u8 carry, u64 lhs, u64 rhs, u64& res) noexcept
    {
        if consteval
        {
            const u64 difference = lhs - rhs - carry;
            res = difference;
            return carry ? difference >= lhs : difference > lhs;
        }
        else
        {
            return _subborrow_u64(carry, lhs, rhs, &res);
        }
    }

    template<usize m, usize n>
    constexpr void knuth_431M(const u32(&first)[m], const u32(&second)[n], u32(&result)[n + m]) noexcept
    {
        for (u32& elem : result)
        {
            elem = 0;
        }

        for (i32 second_idx = 0; second_idx < static_cast<i32>(n); ++second_idx)
        {
            u64 accumulator = 0;

            for (i32 first_idx = 0; first_idx < static_cast<i32>(m); ++first_idx)
            {
                accumulator += static_cast<u64>(first[first_idx]) * second[second_idx] + result[first_idx + second_idx];

                result[first_idx + second_idx] = static_cast<u32>(accumulator);

                accumulator >>= 32;
            }

            result[second_idx + m] = static_cast<u32>(accumulator);
        }
    }

    constexpr u64 mulu128(const u64 multiplicand, const u64 multiplier, u64& high_result) noexcept
    {
        if consteval
        {
            const u32 multiplicand_parts[2] = { static_cast<u32>(multiplicand), static_cast<u32>(multiplicand >> 32) };
            const u32 multiplier_parts[2] = { static_cast<u32>(multiplier), static_cast<u32>(multiplier >> 32) };

            u32 product_parts[4];
            knuth_431M(multiplicand_parts, multiplier_parts, product_parts);

            high_result = (static_cast<u64>(product_parts[3]) << 32) | product_parts[2];
            return (static_cast<u64>(product_parts[1]) << 32) | product_parts[0];
        }
        else
        {
            return _umul128(multiplicand, multiplier, &high_result);
        }
    }

    constexpr void knuth_431D(u32* const dividend, const usize dividend_size,
        const u32* const divisor, const usize divisor_size,
        u32* const quotient) noexcept
    {
        const i32 divisor_len = static_cast<i32>(divisor_size);
        const i32 quotient_len = static_cast<i32>(dividend_size - divisor_size - 1);

        for (i32 quotient_idx = quotient_len; quotient_idx >= 0; --quotient_idx)
        {
            const u64 two_digits = (static_cast<u64>(dividend[quotient_idx + divisor_len]) << 32)
                | static_cast<u64>(dividend[quotient_idx + divisor_len - 1]);

            u64 trial_quotient = two_digits / static_cast<u64>(divisor[divisor_len - 1]);
            u64 trial_remainder = two_digits % static_cast<u64>(divisor[divisor_len - 1]);

            while (trial_quotient > u64{ 0xFFFFFFFF }
                || (trial_quotient * static_cast<u64>(divisor[divisor_len - 2]) 
                > ((trial_remainder << 32) | static_cast<u64>(dividend[quotient_idx + divisor_len - 2]))))
            {
                --trial_quotient;
                trial_remainder += divisor[divisor_len - 1];
                if (trial_remainder > u64{ 0xFFFFFFFF })
                {
                    break;
                }
            }

            i64 borrow = 0;
            for (i32 i = 0; i < divisor_len; ++i)
            {
                const u64 product = static_cast<u32>(trial_quotient) * static_cast<u64>(divisor[i]);
                i64 temp = dividend[i + quotient_idx] - borrow - static_cast<u32>(product);

                dividend[i + quotient_idx] = static_cast<u32>(temp);
                borrow = static_cast<i64>(product >> 32) - (temp >> 32);
            }

            i64 temp = dividend[quotient_idx + divisor_len] - borrow;
            dividend[quotient_idx + divisor_len] = static_cast<u32>(temp);

            quotient[quotient_idx] = static_cast<u32>(trial_quotient);

            if (temp < 0)
            {
                --quotient[quotient_idx];
                borrow = 0;
                for (i32 i = 0; i < divisor_len; ++i)
                {
                    i64 temp_value = dividend[i + quotient_idx] + borrow + divisor[i];
                    dividend[i + quotient_idx] = static_cast<u32>(temp_value);
                    borrow = temp_value >> 32;
                }
                dividend[quotient_idx + divisor_len] += static_cast<i32>(borrow);
            }
        }
    }

    constexpr u64 divu128(
        u64 high_dividend,
        u64 low_dividend,
        u64 divisor,
        u64& remainder) noexcept
    {
        if consteval
        {
            const i32 leading_zeros = std::countl_zero(static_cast<u32>(divisor >> 32));
            if (leading_zeros >= 32)
            {
                u64 remainder_temp = (high_dividend << 32) | (low_dividend >> 32);
                u64 quotient_temp = remainder_temp / static_cast<u32>(divisor);

                remainder_temp = ((remainder_temp % static_cast<u32>(divisor)) << 32)
                    | static_cast<u32>(low_dividend);

                quotient_temp = (quotient_temp << 32) | (remainder_temp / static_cast<u32>(divisor));
                remainder = remainder_temp % static_cast<u32>(divisor);

                return quotient_temp;
            }

            u32 extended_dividend[5] = {
                static_cast<u32>(low_dividend << leading_zeros),
                static_cast<u32>((low_dividend >> (32 - leading_zeros)) | (high_dividend << leading_zeros)),
                static_cast<u32>(high_dividend >> (32 - leading_zeros)),
                static_cast<u32>(high_dividend >> (64 - leading_zeros)),
                0
            };

            u32 extended_divisor[2] = {
                static_cast<u32>(divisor << leading_zeros),
                static_cast<u32>(divisor >> (32 - leading_zeros))
            };

            u32 quotient_parts[3];

            knuth_431D(extended_dividend, 5, extended_divisor, 2, quotient_parts);

            remainder = (static_cast<u64>(extended_dividend[1]) << (32 - leading_zeros))
                | (extended_dividend[0] >> leading_zeros);

            return (static_cast<u64>(quotient_parts[1]) << 32) | quotient_parts[0];
        }
        else
        {
            return _udiv128(high_dividend, low_dividend, divisor, &remainder);
        }
    }
}

export namespace fy
{
    struct basic_128bit
    {
        u64 bits[2];

        constexpr basic_128bit() noexcept : bits{} {}

        template<std::integral T>
        constexpr basic_128bit(const T val) noexcept : bits{ static_cast<u64>(val) }
        {
            if constexpr (std::signed_integral<T>)
            {
                if (val < 0)
                {
                    bits[1] = ~0ull;
                }
            }
        }

        constexpr explicit basic_128bit(const u64 low, const u64 high) noexcept
            : bits{ low, high } {  }

        template<std::integral T>
        constexpr explicit operator T() const noexcept
        {
            return static_cast<T>(bits[0]);
        }

        friend constexpr bool operator == (const basic_128bit& left, const basic_128bit& right) noexcept
        {
            return left.bits[0] == right.bits[0] && left.bits[1] == right.bits[1];
        }

        friend constexpr bool operator < (const basic_128bit& lhs, const basic_128bit& rhs) noexcept
        {
            if (lhs.bits[1] < rhs.bits[1])
            {
                return true;
            }

            if (lhs.bits[1] > rhs.bits[1])
            {
                return false;
            }
            return lhs.bits[0] < rhs.bits[0];
        }

        friend constexpr bool operator > (const basic_128bit& lhs, const basic_128bit& rhs) noexcept
        {
            return rhs < lhs;
        }

        friend constexpr bool operator <= (const basic_128bit& lhs, const basic_128bit& rhs) noexcept
        {
            return !(rhs < lhs);
        }

        friend constexpr bool operator >= (const basic_128bit& lhs, const basic_128bit& rhs) noexcept
        {
            return !(lhs < rhs);
        }

        template<std::integral T>
        friend constexpr T operator << (const T lhs, const basic_128bit& rhs) noexcept
        {
            return lhs << rhs.bits[0];
        }

        template<std::integral T>
        friend constexpr T operator >> (const T lhs, const basic_128bit& rhs) noexcept
        {
            return lhs >> rhs.bits[0];
        }

        template<std::integral T>
        constexpr basic_128bit& operator <<= (const T count) noexcept
        {
            lhs_shift(static_cast<u8>(count));
            return *this;
        }

        template<std::integral T>
        friend constexpr T& operator <<= (T& lhs, const basic_128bit& rhs) noexcept
        {
            lhs <<= rhs.bits[0];
            return lhs;
        }

        template<std::integral T>
        constexpr basic_128bit& operator >>= (const T count) noexcept
        {
            unsignedrhs_shift(static_cast<u8>(count));
            return *this;
        }

        template<std::integral T>
        friend constexpr T& operator >>= (T& lhs, const basic_128bit& rhs) noexcept
        {
            lhs >>= rhs.bits[0];
            return lhs;
        }

        constexpr basic_128bit& operator++() noexcept
        {
            if (++bits[0] == 0)
            {
                ++bits[1];
            }
            return *this;
        }

        constexpr basic_128bit operator++(int) noexcept
        {
            basic_128bit tmp = *this;
            ++*this;
            return tmp;
        }

        constexpr basic_128bit& operator--() noexcept
        {
            if (bits[0]-- == 0)
            {
                --bits[1];
            }
            return *this;
        }

        constexpr basic_128bit operator--(int) noexcept
        {
            basic_128bit tmp = *this;
            --*this;
            return tmp;
        }

        static constexpr basic_128bit multiply(const basic_128bit& lhs, const basic_128bit& rhs) noexcept
        {
            basic_128bit result;
            result.bits[0] = mulu128(lhs.bits[0], rhs.bits[0], result.bits[1]);
            result.bits[1] += lhs.bits[0] * rhs.bits[1];
            result.bits[1] += lhs.bits[1] * rhs.bits[0];
            return result;
        }

        static constexpr basic_128bit divide(const basic_128bit& number, const u64 dens) noexcept
        {
            basic_128bit result;
            result.bits[1] = number.bits[1] / dens;
            u64 _Rem = number.bits[1] % dens;
            result.bits[0] = divu128(_Rem, number.bits[0], dens, _Rem);
            return result;
        }
        
        static constexpr basic_128bit divide(basic_128bit number, basic_128bit dens) noexcept
        {
            if (dens.bits[1] >= number.bits[1])
            {
                if (dens.bits[1] > number.bits[1])
                {
                    return 0;
                }

                return number.bits[1] == 0 ? number.bits[0] / dens.bits[0] : number.bits[0] >= dens.bits[0];
            }

            if (dens.bits[1] == 0)
            {
                return divide(number, dens.bits[0]);
            }

            const i32 d = std::countl_zero(dens.bits[1]);
            dens <<= d;

            u64 high_digit = d == 0 ? 0 : number.bits[1] >> (64 - d);
            number <<= d;

            basic_128bit qhat;
            qhat.bits[1] = high_digit >= dens.bits[1];

            u64 rhat;
            qhat.bits[0] = divu128(high_digit >= dens.bits[1] ? high_digit - dens.bits[1] : high_digit,
                number.bits[1], dens.bits[1], rhat);

            while (true)
            {
                if (qhat.bits[1] > 0)
                {
                    --qhat;
                }
                else
                {
                    basic_128bit prod;
                    prod.bits[0] = mulu128(qhat.bits[0], dens.bits[0], prod.bits[1]);
                    if (prod <= basic_128bit{ number.bits[0], rhat })
                    {
                        break;
                    }
                    --qhat.bits[0];
                }

                const u64 sum = rhat + dens.bits[1];
                if (rhat > sum)
                {
                    break;
                }
                rhat = sum;
            }

            u64 prod0_hi;
            u64 prod_lo = mulu128(qhat.bits[0], dens.bits[0], prod0_hi);
            u8 borrow = sub_borrow64(0, number.bits[0], prod_lo, number.bits[0]);
            u64 _Prod1_hi;
            prod_lo = mulu128(qhat.bits[0], dens.bits[1], _Prod1_hi);
            _Prod1_hi += add_carry64(0, prod_lo, prod0_hi, prod_lo);
            borrow = sub_borrow64(borrow, number.bits[1], prod_lo, number.bits[1]);
            borrow = sub_borrow64(borrow, high_digit, _Prod1_hi, high_digit);
            if (borrow)
            {
                --qhat.bits[0];
            }
            return qhat;
        }

        static constexpr basic_128bit modulo(const basic_128bit& number, const u64 dens) noexcept
        {
            u64 rem;
            divu128(number.bits[1] % dens, number.bits[0], dens, rem);
            return rem;
        }

        static constexpr basic_128bit modulo(basic_128bit number, basic_128bit dens) noexcept
        {
            if (dens.bits[1] >= number.bits[1])
            {
                if (dens.bits[1] > number.bits[1])
                {
                    return number;
                }

                if (dens.bits[0] <= number.bits[0])
                {
                    return number.bits[1] == 0 ? number.bits[0] % dens.bits[0] : number.bits[0] - dens.bits[0];
                }

                return number;
            }

            if (dens.bits[1] == 0)
            {
                return modulo(number, dens.bits[0]);
            }
            
            const i32 d = std::countl_zero(dens.bits[1]);
            dens <<= d;
            u64 high_digit = d == 0 ? 0 : number.bits[1] >> (64 - d);
            number <<= d;

            u64 qhat_high = high_digit >= dens.bits[1];
            u64 rhat;
            u64 qhat = divu128(high_digit >= dens.bits[1] ? high_digit - dens.bits[1] : high_digit,
                number.bits[1], dens.bits[1], rhat);

            while (true)
            {
                if (qhat_high > 0)
                {
                    if (qhat-- == 0)
                    {
                        --qhat_high;
                    }
                }
                else
                {
                    basic_128bit _Prod;
                    _Prod.bits[0] = mulu128(qhat, dens.bits[0], _Prod.bits[1]);
                    if (_Prod <= basic_128bit{ number.bits[0], rhat })
                    {
                        break;
                    }
                    --qhat;
                }

                const u64 _Sum = rhat + dens.bits[1];
                if (rhat > _Sum)
                {
                    break;
                }
                rhat = _Sum;
            }

            u64 prod0_hi;
            u64 prod_lo = mulu128(qhat, dens.bits[0], prod0_hi);
            u8 borrow = sub_borrow64(0, number.bits[0], prod_lo, number.bits[0]);
            u64 prod1_hi;
            prod_lo = mulu128(qhat, dens.bits[1], prod1_hi);
            prod1_hi += add_carry64(0, prod_lo, prod0_hi, prod_lo);
            borrow = sub_borrow64(borrow, number.bits[1], prod_lo, number.bits[1]);
            borrow = sub_borrow64(borrow, high_digit, prod1_hi, high_digit);
            if (borrow)
            {
                u8 _Carry = add_carry64(0, number.bits[0], dens.bits[0], number.bits[0]);
                add_carry64(_Carry, number.bits[1], dens.bits[1], number.bits[1]);
            }
            number >>= d;
            return number;
        }

        template<std::integral T>
        friend constexpr T& operator &= (T& lhs, const basic_128bit& rhs) noexcept
        {
            lhs &= rhs.bits[0];
            return lhs;
        }

        template<std::integral T>
        friend constexpr T& operator ^= (T& lhs, const basic_128bit& rhs) noexcept
        {
            lhs ^= rhs.bits[0];
            return lhs;
        }

        template<std::integral T>
        friend constexpr T& operator |= (T& lhs, const basic_128bit& rhs) noexcept
        {
            lhs |= rhs.bits[0];
            return lhs;
        }

        constexpr explicit operator bool() const noexcept
        {
            return (bits[0] | bits[1]) != 0;
        }

        constexpr void lhs_shift(const u8 count) noexcept
        {
            if (count == 0)
            {
                return;
            }

            if (count >= 64)
            {
                bits[1] = bits[0] << (count % 64);
                bits[0] = 0;
                return;
            }

            if consteval
            {
                bits[1] = (bits[1] << count) | (bits[0] >> (64 - count));
            }
            else
            {
                bits[1] = __shiftleft128(bits[0], bits[1], count);
            }

            bits[0] <<= count;
        }

        constexpr void unsignedrhs_shift(const u8 count) noexcept
        {
            if (count == 0)
            {
                return;
            }

            if (count >= 64)
            {
                bits[0] = bits[1] >> (count % 64);
                bits[1] = 0;
                return;
            }

            if consteval
            {
                bits[0] = (bits[0] >> count) | (bits[1] << (64 - count));
            }
            else
            {
                bits[0] = __shiftright128(bits[0], bits[1], count);
            }

            bits[1] >>= count;
        }
    };
}

export namespace fy
{
    struct int128;
    struct uint128;

    using u128 = uint128;
    using i128 = int128;

    struct uint128 : basic_128bit
    {
        using signed_type = int128;
        using unsigned_type = uint128;

        using basic_128bit::basic_128bit;

        constexpr explicit uint128(const basic_128bit& otherhs) noexcept : basic_128bit{ otherhs } { }

        constexpr uint128& operator = (const basic_128bit& otherhs) noexcept
        {
            basic_128bit::operator=(otherhs);
            return *this;
        }

        friend constexpr std::strong_ordering operator <=> (const uint128& lhs, const uint128& rhs) noexcept
        {
            std::strong_ordering ord = lhs.bits[1] <=> rhs.bits[1];
            if (ord == std::strong_ordering::equal)
            {
                ord = lhs.bits[0] <=> rhs.bits[0];
            }
            return ord;
        }

        friend constexpr uint128 operator << (const uint128& lhs, const basic_128bit& rhs) noexcept
        {
            uint128 tmp{ lhs };
            tmp.lhs_shift(static_cast<u8>(rhs.bits[0]));
            return tmp;
        }

        template<std::integral T>
        constexpr uint128& operator <<= (const T count) noexcept
        {
            lhs_shift(static_cast<u8>(count));
            return *this;
        }

        constexpr uint128& operator <<= (const basic_128bit& count) noexcept
        {
            lhs_shift(static_cast<u8>(count.bits[0]));
            return *this;
        }

        friend constexpr uint128 operator >> (const uint128& lhs, const basic_128bit& rhs) noexcept
        {
            uint128 tmp{ lhs };
            tmp.unsignedrhs_shift(static_cast<u8>(rhs.bits[0]));
            return tmp;
        }

        template<std::integral T>
        constexpr uint128& operator >>= (const T count) noexcept
        {
            unsignedrhs_shift(static_cast<u8>(count));
            return *this;
        }

        constexpr uint128& operator >>= (const basic_128bit& count) noexcept
        {
            unsignedrhs_shift(static_cast<u8>(count.bits[0]));
            return *this;
        }

        constexpr uint128& operator ++ () noexcept
        {
            if (++bits[0] == 0)
            {
                ++bits[1];
            }
            return *this;
        }

        constexpr uint128 operator ++ (int) noexcept
        {
            uint128 tmp = *this;
            ++*this;
            return tmp;
        }

        constexpr uint128& operator -- () noexcept
        {
            if (bits[0]-- == 0)
            {
                --bits[1];
            }
            return *this;
        }

        constexpr uint128 operator -- (int) noexcept
        {
            uint128 tmp = *this;
            --*this;
            return tmp;
        }

        constexpr uint128 operator + () const noexcept
        {
            return *this;
        }

        constexpr uint128 operator - () const noexcept
        {
            return uint128{} - *this;
        }

        constexpr uint128 operator ~ () const noexcept
        {
            return uint128{ ~bits[0], ~bits[1] };
        }

        friend constexpr uint128 operator + (const basic_128bit& lhs, const basic_128bit& rhs) noexcept
        {
            uint128 result;
            const u8 carry = add_carry64(0, lhs.bits[0], rhs.bits[0], result.bits[0]);
            add_carry64(carry, lhs.bits[1], rhs.bits[1], result.bits[1]);
            return result;
        }

        constexpr uint128& operator += (const basic_128bit& otherhs) noexcept
        {
            const u8 carry = add_carry64(0, bits[0], otherhs.bits[0], bits[0]);
            add_carry64(carry, bits[1], otherhs.bits[1], bits[1]);
            return *this;
        }

        template<std::integral T>
        friend constexpr T& operator += (T& lhs, const uint128& rhs) noexcept
        {
            lhs += rhs.bits[0];
            return lhs;
        }

        friend constexpr uint128 operator - (const basic_128bit& lhs, const basic_128bit& rhs) noexcept
        {
            uint128 result;
            const u8 borrow = sub_borrow64(0, lhs.bits[0], rhs.bits[0], result.bits[0]);
            sub_borrow64(borrow, lhs.bits[1], rhs.bits[1], result.bits[1]);
            return result;
        }

        constexpr uint128& operator -= (const basic_128bit& otherhs) noexcept
        {
            const u8 borrow = sub_borrow64(0, bits[0], otherhs.bits[0], bits[0]);
            sub_borrow64(borrow, bits[1], otherhs.bits[1], bits[1]);
            return *this;
        }

        template<std::integral T>
        friend constexpr T& operator -= (T& lhs, const uint128& rhs) noexcept
        {
            lhs -= rhs.bits[0];
            return lhs;
        }

        friend constexpr uint128 operator * (const basic_128bit& lhs, const basic_128bit& rhs) noexcept
        {
            return uint128{ basic_128bit::multiply(lhs, rhs) };
        }

        constexpr uint128& operator *= (const basic_128bit& otherhs) noexcept
        {
            *this = *this * otherhs;
            return *this;
        }

        template<std::integral T>
        friend constexpr T& operator *= (T& lhs, const uint128& rhs) noexcept
        {
            lhs *= rhs.bits[0];
            return lhs;
        }

        friend constexpr uint128 operator / (const basic_128bit& number, const basic_128bit& dens) noexcept
        {
            return uint128{ basic_128bit::divide(number, dens) };
        }

        template<std::integral T>
        friend constexpr uint128 operator / (const uint128& number, const T dens) noexcept
        {
            return uint128{ basic_128bit::divide(number, static_cast<u64>(dens)) };
        }

        template<std::integral T>
        constexpr uint128& operator /= (const T otherhs) noexcept
        {
            *this = uint128{ basic_128bit::divide(*this, static_cast<u64>(otherhs)) };
            return *this;
        }

        constexpr uint128& operator /= (const basic_128bit& otherhs) noexcept
        {
            *this = uint128{ basic_128bit::divide(*this, otherhs) };
            return *this;
        }

        template<std::integral T>
        friend constexpr T& operator /= (T& lhs, const uint128& rhs) noexcept
        {
            if (rhs.bits[1] != 0)
            {
                lhs = 0;
            }
            else
            {
                lhs /= rhs.bits[0];
            }
            return lhs;
        }

        template<std::integral T>
        friend constexpr uint128 operator % (const basic_128bit& number, const T dens) noexcept
        {
            return uint128{ basic_128bit::modulo(number, static_cast<u64>(dens)) };
        }

        friend constexpr uint128 operator % (const basic_128bit& number, const basic_128bit& dens) noexcept
        {
            return uint128{ basic_128bit::modulo(number, dens) };
        }

        template<std::integral T>
        constexpr uint128& operator %= (const T dens) noexcept
        {
            *this = *this % dens;
            return *this;
        }

        constexpr uint128& operator %= (const basic_128bit& dens) noexcept
        {
            *this = *this % dens;
            return *this;
        }

        template<std::integral T>
        friend constexpr T& operator %= (T& lhs, const uint128& rhs) noexcept
        {
            if (rhs.bits[1] == 0)
            {
                lhs %= rhs.bits[0];
            }
            return lhs;
        }

        friend constexpr uint128 operator & (const basic_128bit& lhs, const basic_128bit& rhs) noexcept
        {
            return uint128{ lhs.bits[0] & rhs.bits[0], lhs.bits[1] & rhs.bits[1] };
        }

        constexpr uint128& operator &= (const basic_128bit& otherhs) noexcept
        {
            bits[0] &= otherhs.bits[0];
            bits[1] &= otherhs.bits[1];
            return *this;
        }

        friend constexpr uint128 operator ^ (const basic_128bit& lhs, const basic_128bit& rhs) noexcept
        {
            return uint128{ lhs.bits[0] ^ rhs.bits[0], lhs.bits[1] ^ rhs.bits[1] };
        }

        constexpr uint128& operator ^= (const basic_128bit& otherhs) noexcept
        {
            bits[0] ^= otherhs.bits[0];
            bits[1] ^= otherhs.bits[1];
            return *this;
        }

        friend constexpr uint128 operator | (const basic_128bit& lhs, const basic_128bit& rhs) noexcept
        {
            return uint128{ lhs.bits[0] | rhs.bits[0], lhs.bits[1] | rhs.bits[1] };
        }

        constexpr uint128& operator |= (const basic_128bit& otherhs) noexcept
        {
            bits[0] |= otherhs.bits[0];
            bits[1] |= otherhs.bits[1];
            return *this;
        }
    };
}


export namespace fy
{
    struct int128 : basic_128bit
    {
        using signed_type = int128;
        using unsigned_type = uint128;

        using basic_128bit::basic_128bit;

        constexpr explicit int128(const basic_128bit& otherhs) noexcept : basic_128bit{ otherhs } {}

        constexpr int128& operator=(const basic_128bit& otherhs) noexcept
        {
            basic_128bit::operator=(otherhs);
            return *this;
        }

        friend constexpr std::strong_ordering operator <=> (const int128& lhs, const int128& rhs) noexcept
        {
            std::strong_ordering ord = static_cast<i64>(lhs.bits[1]) <=> static_cast<i64>(rhs.bits[1]);
            if (ord == std::strong_ordering::equal)
            {
                ord = lhs.bits[0] <=> rhs.bits[0];
            }
            return ord;
        }

        friend constexpr int128 operator << (const int128& lhs, const basic_128bit& rhs) noexcept
        {
            int128 tmp{ lhs };
            tmp.lhs_shift(static_cast<u8>(rhs.bits[0]));
            return tmp;
        }

        template<std::integral T>
        constexpr int128& operator <<= (const T count) noexcept
        {
            lhs_shift(static_cast<u8>(count));
            return *this;
        }
        constexpr int128& operator <<= (const basic_128bit& count) noexcept
        {
            lhs_shift(static_cast<u8>(count.bits[0]));
            return *this;
        }

        constexpr void signedrhs_shift(const u8 count) noexcept
        {
            if (count == 0)
            {
                return;
            }

            if (count >= 64)
            {
                bits[0] = static_cast<u64>(static_cast<i64>(bits[1]) >> (count % 64));
                bits[1] = (bits[1] & (1ull << 63)) == 0 ? 0 : ~0ull;
                return;
            }

            if consteval
            {
                bits[0] = (bits[0] >> count) | (bits[1] << (64 - count));
            }
            else
            {
                bits[0] = __shiftright128(bits[0], bits[1], count);
            }

            bits[1] = static_cast<u64>(static_cast<i64>(bits[1]) >> count);
        }

        friend constexpr int128 operator >> (const int128& lhs, const basic_128bit& rhs) noexcept
        {
            int128 tmp{ lhs };
            tmp.signedrhs_shift(static_cast<u8>(rhs.bits[0]));
            return tmp;
        }

        template<std::integral T>
        constexpr int128& operator >>= (const T count) noexcept
        {
            signedrhs_shift(static_cast<u8>(count));
            return *this;
        }

        constexpr int128& operator >>= (const basic_128bit& count) noexcept
        {
            signedrhs_shift(static_cast<u8>(count.bits[0]));
            return *this;
        }

        constexpr int128& operator ++ () noexcept
        {
            if (++bits[0] == 0)
            {
                ++bits[1];
            }
            return *this;
        }

        constexpr int128 operator ++ (int) noexcept
        {
            int128 tmp = *this;
            ++*this;
            return tmp;
        }

        constexpr int128& operator -- () noexcept
        {
            if (bits[0]-- == 0)
            {
                --bits[1];
            }
            return *this;
        }

        constexpr int128 operator -- (int) noexcept
        {
            int128 tmp = *this;
            --*this;
            return tmp;
        }

        constexpr int128 operator + () const noexcept
        {
            return *this;
        }

        constexpr int128 operator - () const noexcept
        {
            return int128{} - *this;
        }

        constexpr int128 operator ~ () const noexcept
        {
            return int128{ ~bits[0], ~bits[1] };
        }

        friend constexpr int128 operator + (const int128& lhs, const int128& rhs) noexcept
        {
            int128 result;
            const u8 carry = add_carry64(0, lhs.bits[0], rhs.bits[0], result.bits[0]);
            add_carry64(carry, lhs.bits[1], rhs.bits[1], result.bits[1]);
            return result;
        }

        constexpr int128& operator += (const basic_128bit& otherhs) noexcept
        {
            const u8 carry = add_carry64(0, bits[0], otherhs.bits[0], bits[0]);
            add_carry64(carry, bits[1], otherhs.bits[1], bits[1]);
            return *this;
        }

        template<std::integral T>
        friend constexpr T& operator += (T& lhs, const int128& rhs) noexcept
        {
            lhs = static_cast<T>(int128{ lhs } + rhs);
            return lhs;
        }

        friend constexpr int128 operator - (const int128& lhs, const int128& rhs) noexcept
        {
            int128 result;
            const u8 borrow = sub_borrow64(0, lhs.bits[0], rhs.bits[0], result.bits[0]);
            sub_borrow64(borrow, lhs.bits[1], rhs.bits[1], result.bits[1]);
            return result;
        }

        constexpr int128& operator -= (const basic_128bit& otherhs) noexcept
        {
            const u8 borrow = sub_borrow64(0, bits[0], otherhs.bits[0], bits[0]);
            sub_borrow64(borrow, bits[1], otherhs.bits[1], bits[1]);
            return *this;
        }

        template<std::integral T>
        friend constexpr T& operator -= (T& lhs, const int128& rhs) noexcept
        {
            lhs = static_cast<T>(int128{ lhs } - rhs);
            return lhs;
        }

        constexpr void strip_negative(bool& _Flip) noexcept
        {
            if ((bits[1] & (1ull << 63)) != 0)
            {
                *this = -*this;
                _Flip = !_Flip;
            }
        }

        friend constexpr int128 operator * (int128 lhs, int128 rhs) noexcept
        {
            bool negative = false;
            lhs.strip_negative(negative);
            rhs.strip_negative(negative);
            int128 result{ basic_128bit::multiply(lhs, rhs) };
            if (negative)
            {
                result = -result;
            }
            return result;
        }

        template<std::integral T>
        constexpr int128& operator *= (const T otherhs) noexcept
        {
            *this = *this * otherhs;
            return *this;
        }

        constexpr int128& operator *= (const int128& otherhs) noexcept
        {
            *this = *this * otherhs;
            return *this;
        }

        constexpr int128& operator *= (const uint128& otherhs) noexcept
        {
            *this = int128{ static_cast<const basic_128bit&>(*this) * otherhs };
            return *this;
        }

        template<std::integral T>
        friend constexpr T& operator *= (T& lhs, const int128& rhs) noexcept
        {
            lhs = static_cast<T>(int128{ lhs } *rhs);
            return lhs;
        }

        template<std::integral T>
        friend constexpr int128 operator / (int128 numb, T dens) noexcept
        {
            bool negative = false;
            numb.strip_negative(negative);
            if constexpr (std::is_signed_v<T>)
            {
                if (dens < 0)
                {
                    dens = -dens;
                    negative = !negative;
                }
            }

            int128 result = int128{ basic_128bit::divide(numb, static_cast<u64>(dens)) };

            if (negative)
            {
                result = -result;
            }
            return result;
        }

        friend constexpr int128 operator / (int128 numb, int128 dens) noexcept
        {
            bool negative = false;
            numb.strip_negative(negative);
            dens.strip_negative(negative);
            int128 result{ basic_128bit::divide(numb, dens) };
            if (negative)
            {
                result = -result;
            }
            return result;
        }

        template<std::integral T>
        constexpr int128& operator /= (const T otherhs) noexcept
        {
            *this = *this / otherhs;
            return *this;
        }

        constexpr int128& operator /= (const int128& otherhs) noexcept
        {
            *this = *this / otherhs;
            return *this;
        }

        constexpr int128& operator /= (const uint128& otherhs) noexcept
        {
            *this = int128{ static_cast<basic_128bit&>(*this) / otherhs };
            return *this;
        }

        template<std::integral T>
        friend constexpr T& operator /= (T& lhs, const int128& rhs) noexcept
        {
            lhs = static_cast<T>(int128{ lhs } / rhs);
            return lhs;
        }

        friend constexpr int128 operator % (int128 lhs, int128 rhs) noexcept
        {
            bool negative = false;
            lhs.strip_negative(negative);

            if ((rhs.bits[1] & (1ull << 63)) != 0)
            {
                rhs = -rhs;
            }

            uint128 result{ basic_128bit::modulo(lhs, rhs) };
            if (negative)
            {
                result = -result;
            }
            return int128{ result };
        }

        template<std::integral T>
        friend constexpr int128 operator % (int128 lhs, const T rhs) noexcept
        {
            return lhs % int128{ rhs };
        }

        template<std::integral T>
        constexpr int128& operator %= (const T otherhs) noexcept
        {
            *this = *this % otherhs;
            return *this;
        }

        constexpr int128& operator %= (const int128& otherhs) noexcept
        {
            *this = *this % otherhs;
            return *this;
        }

        constexpr int128& operator %= (const uint128& otherhs) noexcept
        {
            *this = static_cast<const basic_128bit&>(*this) % otherhs;
            return *this;
        }

        template<std::integral T>
        friend constexpr T& operator %= (T& lhs, const int128& rhs) noexcept
        {
            lhs = static_cast<T>(int128{ lhs } % rhs);
            return lhs;
        }

        friend constexpr int128 operator & (const int128& lhs, const int128& rhs) noexcept
        {
            return int128{ lhs.bits[0] & rhs.bits[0], lhs.bits[1] & rhs.bits[1] };
        }

        constexpr int128& operator &= (const basic_128bit& otherhs) noexcept
        {
            bits[0] &= otherhs.bits[0];
            bits[1] &= otherhs.bits[1];
            return *this;
        }

        friend constexpr int128 operator ^ (const int128& lhs, const int128& rhs) noexcept
        {
            return int128{ lhs.bits[0] ^ rhs.bits[0], lhs.bits[1] ^ rhs.bits[1] };
        }

        constexpr int128& operator ^= (const basic_128bit& otherhs) noexcept
        {
            bits[0] ^= otherhs.bits[0];
            bits[1] ^= otherhs.bits[1];
            return *this;
        }

        friend constexpr int128 operator | (const int128& lhs, const int128& rhs) noexcept
        {
            return int128{ lhs.bits[0] | rhs.bits[0], lhs.bits[1] | rhs.bits[1] };
        }

        constexpr int128& operator |= (const basic_128bit& otherhs) noexcept
        {
            bits[0] |= otherhs.bits[0];
            bits[1] |= otherhs.bits[1];
            return *this;
        }
    };
}