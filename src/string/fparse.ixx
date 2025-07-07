export module foye.fparse;

export import foye.foye_core;
export import foye.funicode_cvt;
export import foye.fbytearray;
export import foye.extensionType.int128;
import std;

export namespace fy
{
    namespace numeric_character
    {
        template<character_t Char_t>
        inline constexpr Char_t zero = []() constexpr -> Char_t
            {
                if constexpr (std::is_same_v<Char_t, char32_t>) return U'0';
                else if constexpr (std::is_same_v<Char_t, char16_t>) return u'0';
                else if constexpr (std::is_same_v<Char_t, wchar_t>) return L'0';
                else if constexpr (std::is_same_v<Char_t, char8_t>) return u8'0';
                else return '0';
            }();

        template<character_t Char_t>
        inline constexpr Char_t minus = []() constexpr -> Char_t
            {
                if constexpr (std::is_same_v<Char_t, char32_t>) return U'-';
                else if constexpr (std::is_same_v<Char_t, char16_t>) return u'-';
                else if constexpr (std::is_same_v<Char_t, wchar_t>) return L'-';
                else if constexpr (std::is_same_v<Char_t, char8_t>) return u8'-';
                else return '-';
            }();

        template<character_t Char_t>
        inline constexpr Char_t plus = []() constexpr -> Char_t
            {
                if constexpr (std::is_same_v<Char_t, char32_t>) return U'+';
                else if constexpr (std::is_same_v<Char_t, char16_t>) return u'+';
                else if constexpr (std::is_same_v<Char_t, wchar_t>) return L'+';
                else if constexpr (std::is_same_v<Char_t, char8_t>) return u8'+';
                else return '-';
            }();

        template<character_t Char_t>
        inline constexpr Char_t dot = []() constexpr -> Char_t
            {
                if constexpr (std::is_same_v<Char_t, char32_t>) return U'.';
                else if constexpr (std::is_same_v<Char_t, char16_t>) return u'.';
                else if constexpr (std::is_same_v<Char_t, wchar_t>) return L'.';
                else if constexpr (std::is_same_v<Char_t, char8_t>) return u8'.';
                else return '.';
            }();

        template<character_t Char_t>
        inline constexpr Char_t numbers[10] =
        {
            Char_t(48), Char_t(49), Char_t(50), Char_t(51), Char_t(52),
            Char_t(53), Char_t(54), Char_t(55), Char_t(56), Char_t(57)
        };
    }
}

export namespace fy
{
    template<character_t Char_t>
    struct floating_point_to_chararray_Invoker
    {
        constexpr floating_point_to_chararray_Invoker() noexcept = default;

        Char_t buffer[64] = {};

        template <Floating_arithmetic Value_t>
        constexpr usize check_floating_point(Value_t value) noexcept
        {
            constexpr char8_t nan[] = u8"Nan";
            constexpr char8_t ninf[] = u8"-inf";
            constexpr char8_t pinf[] = u8"inf";
            constexpr char8_t nzero[] = u8"Negative Zero";

            constexpr string_convert_Invoker<char8_t, Char_t> invoker{};

            if (std::isnan(value))
            {
                return invoker(nan, 3, buffer);
            }
            else if (std::isinf(value))
            {
                if (value > 0)
                {
                    return invoker(ninf, 4, buffer);
                }
                else
                {
                    return invoker(pinf, 3, buffer);
                }
            }
            else if (value == 0.0 && std::signbit(value))
            {
                return invoker(nzero, 13, buffer);
            }
            else
            {
                return 0;
            }
        }

        template<Floating_arithmetic Value_t>
        constexpr usize operator()(Value_t value_) noexcept
        {
            using namespace numeric_character;

            usize res = check_floating_point<Value_t>(value_);
            if (res != 0)
            {
                return res;
            }

            constexpr Value_t epsilon = std::numeric_limits<Value_t>::epsilon();
            f64 value = static_cast<f64>(value_);
            Char_t* buf_ptr = buffer;
            usize count = 0;

            if (value < 0)
            {
                *buf_ptr++ = minus<Char_t>;
                value = -value;
                ++count;
            }

            i64 integerPart = static_cast<i64>(value);
            f64 fractionalPart = value - integerPart;

            Char_t temp[32] = {};
            ssize index = 0;

            usize integer_digits = 0;
            if (integerPart == 0)
            {
                integer_digits = 1;
            }
            else
            {
                i64 temp_int = integerPart;
                while (temp_int > 0)
                {
                    temp_int /= 10;
                    integer_digits++;
                }
            }

            do
            {
                temp[index++] = zero<Char_t> +(integerPart % 10);
                integerPart /= 10;
            } while (integerPart > 0);

            while (index > 0)
            {
                *buf_ptr++ = temp[--index];
                ++count;
            }

            if (fractionalPart > epsilon)
            {
                *buf_ptr++ = dot<Char_t>;
                ++count;

                usize precision = static_cast<ssize>(std::numeric_limits<Value_t>::digits10) - static_cast<ssize>(integer_digits);

                Char_t* decimal_start = buf_ptr;

                for (ssize i = 0; i < precision; ++i)
                {
                    fractionalPart *= 10;
                    i64 digit = static_cast<i64>(fractionalPart);
                    *buf_ptr++ = zero<Char_t> +digit;
                    fractionalPart -= digit;

                    ++count;

                    if (std::abs(fractionalPart) < epsilon)
                    {
                        break;
                    }
                }

                if (fractionalPart > epsilon)
                {
                    fractionalPart *= 10;
                    i64 next_digit = static_cast<i64>(fractionalPart);

                    if (next_digit >= 5)
                    {
                        Char_t* p = buf_ptr - 1;
                        while (p >= decimal_start)
                        {
                            if (*p < zero<Char_t> +9)
                            {
                                (*p)++;
                                break;
                            }
                            else
                            {
                                *p = zero<Char_t>;
                                if (p == decimal_start)
                                {
                                    p = buf_ptr - 1;
                                    while (p >= buffer)
                                    {
                                        if (*p == dot<Char_t>)
                                        {
                                            p--;
                                        }
                                        if (*p < zero<Char_t> +9)
                                        {
                                            (*p)++;
                                            break;
                                        }
                                        else
                                        {
                                            *p = zero<Char_t>;
                                            p--;
                                        }
                                    }
                                    break;
                                }
                                p--;
                            }
                        }
                    }
                }

                while (buf_ptr > decimal_start && *(buf_ptr - 1) == zero<Char_t>)
                {
                    --buf_ptr;
                    --count;
                }

                if (buf_ptr == decimal_start)
                {
                    --buf_ptr;
                    --count;
                }
            }

            return count;
        }
    };

    template<character_t Char_t>
    struct integral_to_chararray_Invoker
    {
        constexpr integral_to_chararray_Invoker() noexcept = default;

        static constexpr usize offset = 20;

        Char_t data[offset];

        template<typename Value_t> requires std::is_signed_v<Value_t>
        constexpr usize operator () (Value_t value) noexcept
        {
            using namespace numeric_character;
            if (value == 0)
            {
                data[19] = zero<Char_t>;
                return 1;
            }

            if (value < 0)
            {
                usize index = 19;
                auto abs_value = static_cast<std::make_unsigned_t<Value_t>>(std::abs(value));

                const usize start_index = index;
                while (abs_value > 0)
                {
                    data[index] = zero<Char_t> +(abs_value % 10);
                    abs_value /= 10;
                    index--;
                }

                data[index] = minus<Char_t>;
                return start_index - index + 1;
            }

            return (*this)(static_cast<std::make_unsigned_t<Value_t>>(value));
        }

        template<typename Value_t> requires std::is_unsigned_v<Value_t>
        usize operator () (Value_t value) noexcept
        {
            using namespace numeric_character;
            usize index = 19;

            if (value == 0)
            {
                data[index] = zero<Char_t>;
                return 1;
            }

            const usize start_index = index;

            while (value > 0)
            {
                data[index] = zero<Char_t> +(value % 10);
                value /= 10;
                index--;
            }

            return start_index - index;
        }
    };

    template<character_t Char_t>
    struct integral128_to_chararray_Invoker
    {
        Char_t buffer[41];

        usize operator () (int128 value) noexcept
        {
            constexpr usize buffer_size = sizeof(buffer) / sizeof(Char_t);
            constexpr int128 min_int128 = int128{ 1 } << 127;
            const Char_t minus_sign = static_cast<Char_t>('-');
            const Char_t zero_char = static_cast<Char_t>('0');

            if (value == min_int128)
            {
                const char min_str_char[] = "-170141183460469231731687303715884105728";
                const usize len = sizeof(min_str_char) - 1;

                if (buffer_size < len)
                {
                    return 0;
                }

                for (usize i = 0; i < len; ++i)
                {
                    buffer[i] = static_cast<Char_t>(min_str_char[i]);
                }
                return len;
            }

            bool negative = false;
            if (value < 0)
            {
                negative = true;
                value = -value;
            }

            usize digit_count = 0;
            uint128 abs_value = static_cast<uint128>(value);

            if (abs_value == 0)
            {
                if (buffer_size < 1)
                {
                    return 0;
                }
                buffer[0] = zero_char;
                digit_count = 1;
            }
            else
            {
                constexpr u32 base = 1000000000;
                constexpr i32 max_chunk_count = 5;
                u32 chunks[max_chunk_count];
                i32 chunk_count = 0;

                while (abs_value != 0)
                {
                    uint128 quotient = abs_value / base;
                    uint128 remainder = abs_value % base;
                    abs_value = quotient;

                    if (chunk_count >= max_chunk_count)
                        return 0;

                    chunks[chunk_count++] = static_cast<u32>(remainder);
                }

                u32 highest_chunk = chunks[chunk_count - 1];
                u8 highest_digits = 1;
                while (highest_chunk >= 10)
                {
                    highest_digits++;
                    highest_chunk /= 10;
                }
                digit_count = highest_digits + 9 * (chunk_count - 1);

                if (buffer_size < digit_count + (negative ? 1 : 0))
                {
                    return 0;
                }

                Char_t* out_ptr = negative ? buffer + 1 : buffer;
                for (i32 i = chunk_count - 1; i >= 0; i--)
                {
                    u32 chunk_val = chunks[i];
                    u8 chunk_digits = (i == chunk_count - 1) ? highest_digits : 9;

                    if (i < chunk_count - 1)
                    {
                        for (u8 j = 0; j < 9 - chunk_digits; j++)
                        {
                            *out_ptr++ = zero_char;
                        }
                    }

                    u32 divisor = 1;
                    for (u8 d = 1; d < chunk_digits; d++)
                    {
                        divisor *= 10;
                    }

                    for (u8 d = 0; d < chunk_digits; d++)
                    {
                        u32 digit = chunk_val / divisor;
                        *out_ptr++ = static_cast<Char_t>('0' + digit);
                        chunk_val %= divisor;
                        divisor /= 10;
                    }
                }
            }

            if (negative)
            {
                buffer[0] = minus_sign;
                return digit_count + 1;
            }

            return digit_count;
        }

        usize operator () (uint128 value) noexcept
        {
            constexpr usize buffer_size = sizeof(buffer) / sizeof(Char_t);
            if (value == 0)
            {
                if (buffer_size >= 1)
                {
                    buffer[0] = static_cast<Char_t>('0');
                    return 1;
                }
                return 0;
            }

            constexpr u32 base = 1000000000;
            constexpr i32 max_chunk_count = 5;
            u32 chunks[max_chunk_count];
            u8  bit_counts[max_chunk_count];
            i32 chunk_count = 0;

            while (value != 0)
            {
                uint128 quotient = value / base;
                uint128 remainder = value % base;
                value = quotient;

                u32 chunk_val = static_cast<u32>(remainder);
                u8 bit_count = 0;
                u32 temp = chunk_val;
                do
                {
                    bit_count++;
                    temp /= 10;
                } while (temp != 0);

                if (chunk_count >= max_chunk_count)
                {
                    return 0;
                }

                chunks[chunk_count] = chunk_val;
                bit_counts[chunk_count] = bit_count;
                chunk_count++;
            }

            usize total_digits = 0;
            if (chunk_count > 1)
            {
                total_digits += 9 * (chunk_count - 1);
            }
            if (chunk_count > 0)
            {
                total_digits += bit_counts[chunk_count - 1];
            }

            if (buffer_size < total_digits)
            {
                return 0;
            }

            Char_t* out_ptr = buffer;
            for (i32 i = chunk_count - 1; i >= 0; i--)
            {
                u32 chunk_val = chunks[i];
                u8 bc = bit_counts[i];

                if (i != chunk_count - 1)
                {
                    for (u8 j = 0; j < 9 - bc; j++)
                    {
                        *out_ptr++ = static_cast<Char_t>('0');
                    }
                }

                u32 divisor = 1;
                for (u8 k = 1; k < bc; k++)
                {
                    divisor *= 10;
                }

                for (u8 k = 0; k < bc; k++)
                {
                    u32 digit = chunk_val / divisor;
                    *out_ptr++ = static_cast<Char_t>('0' + digit);
                    chunk_val %= divisor;
                    divisor /= 10;
                }
            }

            return static_cast<usize>(out_ptr - buffer);
        }
    };






    template<character_t Char_t>
    struct string_to_numeric_Invoker
    {
        constexpr string_to_numeric_Invoker() = default;

        constexpr static f64 as_floating_point(const Char_t* src_ptr, usize count_Char_t) noexcept
        {
            using namespace numeric_character;

            if (count_Char_t == 0)
            {
                return 0.0;
            }

            f64 result = 0.0;
            f64 fraction = 0.1;
            bool is_negative = false;
            bool has_decimal_point = false;
            usize i = 0;

            while (i < count_Char_t && src_ptr[i] == ' ')
            {
                ++i;
            }

            if (i < count_Char_t)
            {
                if (src_ptr[i] == minus<Char_t>)
                {
                    is_negative = true;
                    ++i;
                }
                else if (src_ptr[i] == plus<Char_t>)
                {
                    ++i;
                }
            }

            while (i < count_Char_t)
            {
                Char_t ch = src_ptr[i];
                if (ch >= numbers<Char_t>[0] && ch <= numbers<Char_t>[9])
                {
                    if (has_decimal_point)
                    {
                        result += (ch - numbers<Char_t>[0]) * fraction;
                        fraction *= 0.1;
                    }
                    else
                    {
                        result = result * 10.0 + (ch - numbers<Char_t>[0]);
                    }
                    ++i;
                }
                else if (ch == dot<Char_t> && !has_decimal_point)
                {
                    has_decimal_point = true;
                    ++i;
                }
                else
                {
                    break;
                }
            }

            return is_negative ? -result : result;
        }

        constexpr static u64 as_unsigned_integer(const Char_t* src_ptr, usize count_Char_t) noexcept
        {
            using namespace numeric_character;
            u64 result = 0;
            for (usize i = 0; i < count_Char_t; ++i)
            {
                Char_t ch = src_ptr[i];
                if (ch >= numbers<Char_t>[0] && ch <= numbers<Char_t>[9])
                {
                    result = result * 10 + (ch - numbers<Char_t>[0]);
                }
                else
                {
                    break;
                }
            }
            return result;
        }

        constexpr static i64 as_signed_integer(const Char_t* src_ptr, usize count_Char_t) noexcept
        {
            using namespace numeric_character;
            if (count_Char_t == 0)
            {
                return 0;
            }

            bool is_negative = false;
            usize start_index = 0;

            if (src_ptr[0] == minus<Char_t>)
            {
                is_negative = true;
                start_index = 1;
            }
            else if (src_ptr[0] == plus<Char_t>)
            {
                start_index = 1;
            }

            u64 unsigned_result = as_unsigned_integer(src_ptr + start_index, count_Char_t - start_index);
            i64 res = is_negative ? (i64(0) - unsigned_result) : static_cast<i64>(unsigned_result);
            return res;
        }
    };
}




export namespace fy
{
    template<typename Char_Ty, usize N>
    struct static_string
    {
        constexpr static_string(const Char_Ty(&Str)[N])
        {
            std::copy(std::begin(Str), std::end(Str), std::begin(data));
            data[N - 1] = 0;
        }

        constexpr decltype(auto) operator[](usize index) const
        {
            return data[index];
        }

        static constexpr usize size = N;
        Char_Ty data[N];
    };

    template<typename Char_Ty, usize N>
    static_string(const Char_Ty(&)[N]) -> static_string<Char_Ty, N>;

    template<auto Value> 
    struct value_package
    {
        constexpr static auto value = Value;
    };

    template<static_string Str>
    consteval value_package<Str> operator ""_fmt()
    {
        return {};
    }

    template<static_string Str, typename ... Args>
    void format_test(value_package<Str>, Args... args)
    {
        constexpr auto check_token_matched = [&]() constexpr -> bool
            {
                usize count = 0;
                for (usize i = 0; i < Str.size; ++i)
                {
                    if (Str[i] == '{') ++count;
                    if (Str[i] == '}') --count;
                }
                return count == 0;
            };
        
        constexpr auto check_numof_args = [&]() constexpr -> bool
            {
                usize count = 0;
                for (usize i = 0; i < Str.size; ++i)
                {
                    if (Str[i] == '{') ++count;
                }
                return count == sizeof...(args);
            };

        static_assert(check_token_matched(), "Mismatched '{' or '}' in the format string!");
        static_assert(check_numof_args(), "Number of arguments does not match the number of '{}' pairs!");

        static_assert(Str.size > 1);

        using Char_t = decltype(Str[0]);

        std::cout << typeid(Char_t).name() << std::endl;

        
        for (usize i = 0; i < Str.size; ++i)
        {
            std::cout << Str[i];
        }
        std::cout << "\n";
    }
}