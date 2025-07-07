export module foye.funicode_cvt;

export import foye.foye_core;
import std;

export namespace fy
{
    inline constexpr char32_t UNICODE_MAX = 0x10FFFF;
    inline constexpr char32_t SURROGATE_START = 0xD800;
    inline constexpr char32_t SURROGATE_END = 0xDFFF;
    inline constexpr char32_t BMP_MAX = 0x10000;
    inline constexpr char32_t UTF8_BYTES_NEEDED_1 = 0x80;
    inline constexpr char32_t UTF8_BYTES_NEEDED_2 = 0x800;
    inline constexpr char32_t UTF8_BYTES_NEEDED_3 = 0x10000;
    inline constexpr char32_t UTF8_BYTES_NEEDED_4 = 0x110000;

    inline constexpr char8_t UTF8_1_BYTE_MASK = 0x7F;
    inline constexpr char8_t UTF8_2_BYTE_MASK = 0x1F;
    inline constexpr char8_t UTF8_3_BYTE_MASK = 0x0F;
    inline constexpr char8_t UTF8_4_BYTE_MASK = 0x07;
    inline constexpr char8_t UTF8_CONTINUATION_MASK = 0x3F;
    inline constexpr char8_t UTF8_2_BYTE_OFFSET = 0xC0;
    inline constexpr char8_t UTF8_3_BYTE_OFFSET = 0xE0;
    inline constexpr char8_t UTF8_4_BYTE_OFFSET = 0xF0;
    inline constexpr char8_t UTF8_CONTINUATION_OFFSET = 0x80;

    inline constexpr char8_t UTF8_CONTINUATION_CHECK_MASK = 0xC0;
    inline constexpr char8_t UTF8_4_BYTE_CHECK_MASK = 0xF8;

    inline constexpr char16_t UTF16_HIGH_SURROGATE_MASK = 0x3FF;
    inline constexpr char16_t UTF16_HIGH_SURROGATE_OFFSET = 0xD800;
    inline constexpr char16_t UTF16_LOW_SURROGATE_OFFSET = 0xDC00;

    inline constexpr char16_t UTF16_HIGH_SURROGATE_MAX = 0xDBFF;
    inline constexpr char16_t UTF16_LOW_SURROGATE_MAX = 0xDFFF;
}

export namespace fy
{
    template<character_t Src_Char_t, character_t Dst_Char_t>
    struct character_convert_Invoker
    {
        constexpr character_convert_Invoker() = default;
        constexpr usize operator () (char32_t src, char32_t& dst) const noexcept
        {
            if constexpr (sizeof(Src_Char_t) == sizeof(Dst_Char_t))
            {
                dst = src;
                return 1;
            }
            else
            {
                static_assert(false, "Not supported yet");
            }
        }

        constexpr usize operator() (char32_t src, char32_t& dst) const noexcept
            requires(std::is_same_v<Src_Char_t, char32_t>&& std::is_same_v<Dst_Char_t, char16_t>)
        {
            if (src > UNICODE_MAX || (src >= SURROGATE_START && src <= SURROGATE_END))
            {
                return 0;
            }

            dst = 0;

            if (src <= 0xFFFF)
            {
                dst = src;
                return 1;
            }
            else
            {
                dst = (UTF16_HIGH_SURROGATE_OFFSET |
                    ((src - 0x10000) >> 10) << 16) |
                    UTF16_LOW_SURROGATE_OFFSET |
                    ((src - 0x10000) & UTF16_HIGH_SURROGATE_MASK);
                return 2;
            }
        }

        constexpr usize operator() (char32_t src, char32_t& dst) const noexcept
            requires(std::is_same_v<Src_Char_t, char32_t>&& std::is_same_v<Dst_Char_t, char8_t>)
        {
            if (src > UNICODE_MAX || (src >= SURROGATE_START && src <= SURROGATE_END))
            {
                return 0;
            }

            dst = 0;

            if (src <= UTF8_1_BYTE_MASK)
            {
                dst = src;
                return 1;
            }
            else if (src <= 0x7FF)
            {
                dst = ((src >> 6) | UTF8_2_BYTE_OFFSET) |
                    ((src & UTF8_CONTINUATION_MASK) | UTF8_CONTINUATION_OFFSET) << 8;
                return 2;
            }
            else if (src <= 0xFFFF)
            {
                dst = ((src >> 12) | UTF8_3_BYTE_OFFSET) |
                    ((src >> 6 & UTF8_CONTINUATION_MASK) |
                        UTF8_CONTINUATION_OFFSET) << 8 |
                    ((src & UTF8_CONTINUATION_MASK) |
                        UTF8_CONTINUATION_OFFSET) << 16;
                return 3;
            }
            else
            {
                dst = ((src >> 18) | UTF8_4_BYTE_OFFSET) |
                    ((src >> 12 & UTF8_CONTINUATION_MASK) |
                        UTF8_CONTINUATION_OFFSET) << 8 |
                    ((src >> 6 & UTF8_CONTINUATION_MASK) |
                        UTF8_CONTINUATION_OFFSET) << 16 |
                    ((src & UTF8_CONTINUATION_MASK) |
                        UTF8_CONTINUATION_OFFSET) << 24;
                return 4;
            }
        }

        constexpr usize operator() (char32_t src, char32_t& dst) const noexcept
            requires(std::is_same_v<Src_Char_t, char16_t>&& std::is_same_v<Dst_Char_t, char8_t>)
        {
            if (src <= 0xFFFF)
            {
                if (src >= SURROGATE_START && src <= SURROGATE_END)
                {
                    return 0;
                }

                dst = 0;

                if (src <= UTF8_1_BYTE_MASK)
                {
                    dst = src;
                    return 1;
                }
                else if (src <= 0x7FF)
                {
                    dst = ((src >> 6) | UTF8_2_BYTE_OFFSET) |
                        ((src & UTF8_CONTINUATION_MASK) |
                            UTF8_CONTINUATION_OFFSET) << 8;
                    return 2;
                }
                else
                {
                    dst = ((src >> 12) | UTF8_3_BYTE_OFFSET) |
                        ((src >> 6 & UTF8_CONTINUATION_MASK) |
                            UTF8_CONTINUATION_OFFSET) << 8 |
                        ((src & UTF8_CONTINUATION_MASK) |
                            UTF8_CONTINUATION_OFFSET) << 16;
                    return 3;
                }
            }
            else
            {
                char32_t unicode = src;

                if (unicode > UNICODE_MAX)
                {
                    return 0;
                }

                dst = ((unicode >> 18) | UTF8_4_BYTE_OFFSET) |
                    ((unicode >> 12 & UTF8_CONTINUATION_MASK) |
                        UTF8_CONTINUATION_OFFSET) << 8 |
                    ((unicode >> 6 & UTF8_CONTINUATION_MASK) |
                        UTF8_CONTINUATION_OFFSET) << 16 |
                    ((unicode & UTF8_CONTINUATION_MASK) |
                        UTF8_CONTINUATION_OFFSET) << 24;
                return 4;
            }
        }

        constexpr usize operator() (char32_t src, char32_t& dst) const noexcept
            requires(std::is_same_v<Src_Char_t, char16_t>&& std::is_same_v<Dst_Char_t, char32_t>)
        {
            if (src <= 0xFFFF)
            {
                if (src >= UTF16_HIGH_SURROGATE_OFFSET &&
                    src < UTF16_LOW_SURROGATE_OFFSET)
                {
                    return 0;
                }
                else if (src >= UTF16_LOW_SURROGATE_OFFSET &&
                    src <= SURROGATE_END)
                {
                    return 0;
                }

                dst = src;
                return 1;
            }
            else
            {
                char16_t high = (src >> 16) & 0xFFFF;
                char16_t low = src & 0xFFFF;

                if (high < UTF16_HIGH_SURROGATE_OFFSET ||
                    high >= UTF16_LOW_SURROGATE_OFFSET ||
                    low < UTF16_LOW_SURROGATE_OFFSET ||
                    low > SURROGATE_END)
                {
                    return 0;
                }

                dst = (((high - UTF16_HIGH_SURROGATE_OFFSET) << 10) |
                    (low - UTF16_LOW_SURROGATE_OFFSET)) + 0x10000;
                return 2;
            }
        }

        constexpr usize operator() (char32_t src, char32_t& dst) const noexcept
            requires(std::is_same_v<Src_Char_t, char8_t>&& std::is_same_v<Dst_Char_t, char16_t>)
        {
            char32_t codepoint = 0;
            usize bytes_read = 0;

            if ((src & 0x80) == 0)
            {
                codepoint = src;
                bytes_read = 1;
            }
            else if ((src & 0xE0) == 0xC0)
            {
                codepoint = (src & UTF8_2_BYTE_MASK) << 6;
                bytes_read = 2;
            }
            else if ((src & 0xF0) == 0xE0)
            {
                codepoint = (src & UTF8_3_BYTE_MASK) << 12;
                bytes_read = 3;
            }
            else if ((src & 0xF8) == 0xF0)
            {
                codepoint = (src & UTF8_4_BYTE_MASK) << 18;
                bytes_read = 4;
            }
            else
            {
                return 0;
            }

            for (usize i = 1; i < bytes_read; ++i)
            {
                char8_t continuation_byte = *(reinterpret_cast<const char8_t*>(&src) + i);
                if ((continuation_byte & 0xC0) != 0x80)
                {
                    return 0;
                }
                codepoint |= (continuation_byte & UTF8_CONTINUATION_MASK) << (6 * (bytes_read - 1 - i));
            }

            if (codepoint > UNICODE_MAX || (codepoint >= SURROGATE_START && codepoint <= SURROGATE_END))
            {
                return 0;
            }

            if (codepoint <= 0xFFFF)
            {
                dst = codepoint;
                return 1;
            }
            else
            {
                dst = ((UTF16_HIGH_SURROGATE_OFFSET |
                    ((codepoint - 0x10000) >> 10)) << 16) |
                    (UTF16_LOW_SURROGATE_OFFSET | ((codepoint - 0x10000) & UTF16_HIGH_SURROGATE_MASK));
                return 2;
            }
        }

        constexpr usize operator() (char32_t src, char32_t& dst) const noexcept
            requires(std::is_same_v<Src_Char_t, char8_t>&& std::is_same_v<Dst_Char_t, char32_t>)
        {
            char32_t codepoint = 0;
            usize bytes_read = 0;

            if ((src & 0x80) == 0)
            {
                codepoint = src;
                bytes_read = 1;
            }
            else if ((src & 0xE0) == 0xC0)
            {
                codepoint = (src & UTF8_2_BYTE_MASK) << 6;
                bytes_read = 2;
            }
            else if ((src & 0xF0) == 0xE0)
            {
                codepoint = (src & UTF8_3_BYTE_MASK) << 12;
                bytes_read = 3;
            }
            else if ((src & 0xF8) == 0xF0)
            {
                codepoint = (src & UTF8_4_BYTE_MASK) << 18;
                bytes_read = 4;
            }
            else
            {
                return 0;
            }

            for (usize i = 1; i < bytes_read; ++i)
            {
                char8_t continuation_byte = *(reinterpret_cast<const char8_t*>(&src) + i);
                if ((continuation_byte & 0xC0) != 0x80)
                {
                    return 0;
                }
                codepoint |= (continuation_byte & UTF8_CONTINUATION_MASK) << (6 * (bytes_read - 1 - i));
            }

            if (codepoint > UNICODE_MAX || (codepoint >= SURROGATE_START && codepoint <= SURROGATE_END))
            {
                return 0;
            }

            dst = codepoint;
            return 1;
        }
    };


    template<character_t Src_Char_t, character_t Dst_Char_t>
    struct string_convert_Invoker : private character_convert_Invoker<Src_Char_t, Dst_Char_t>
    {
        using cvt_Invoker = character_convert_Invoker<Src_Char_t, Dst_Char_t>;
        constexpr string_convert_Invoker() noexcept = default;
        constexpr usize operator () (const Src_Char_t* const src, usize size, Dst_Char_t* dst) const noexcept
        {
            if constexpr (sizeof(Src_Char_t) == sizeof(Dst_Char_t))
            {
                if (std::is_constant_evaluated())
                {
                    for (usize i = 0; i < size; ++i)
                    {
                        dst[i] = src[i];
                    }
                }
                else
                {
                    std::memcpy(dst, src, size * sizeof(Src_Char_t));
                }
                return size;
            }
            else
            {
                static_assert(false, "Not supported yet");
            }
        }
    };

    template<character_t Dst_Char_t>
    struct string_convert_Invoker<char32_t, Dst_Char_t> : private character_convert_Invoker<char32_t, Dst_Char_t>
    {
        using cvt_Invoker = character_convert_Invoker<char32_t, Dst_Char_t>;
        static constexpr std::make_unsigned_t<Dst_Char_t> mask = std::numeric_limits<std::make_unsigned_t<Dst_Char_t>>::max();
        constexpr string_convert_Invoker() = default;

        constexpr usize operator () (const char32_t* const src_str, usize src_length, Dst_Char_t* dst) const noexcept
        {
            if (!src_str || !dst)
            {
                return 0;
            }
            else
            {
                usize dst_pos = 0;

                for (usize i = 0; i < src_length; ++i)
                {
                    char32_t dst_char{};
                    usize units = cvt_Invoker::operator()(src_str[i], dst_char);

                    for (usize j = 0; j < units; ++j)
                    {
                        dst[dst_pos++] = (dst_char >> (j * (sizeof(Dst_Char_t) * std::numeric_limits<unsigned char>::digits))) & mask;
                    }
                }

                return dst_pos;
            }
        }
    };

    template<character_t Dst_Char_t>
    struct string_convert_Invoker<char16_t, Dst_Char_t> : private character_convert_Invoker<char16_t, Dst_Char_t>
    {
        using cvt_Invoker = character_convert_Invoker<char16_t, Dst_Char_t>;
        static constexpr std::make_unsigned_t<Dst_Char_t> mask = std::numeric_limits<std::make_unsigned_t<Dst_Char_t>>::max();
        constexpr string_convert_Invoker() = default;

        constexpr usize operator () (const char16_t* const src_str, usize src_length, Dst_Char_t* dst) const noexcept
        {
            const constexpr usize dst_char_bit_width = (sizeof(Dst_Char_t) * std::numeric_limits<unsigned char>::digits);
            if (!src_str || !dst)
            {
                return 0;
            }
            else
            {
                usize dst_pos = 0;
                usize src_pos = 0;

                while (src_pos < src_length)
                {
                    if (src_str[src_pos] >= UTF16_HIGH_SURROGATE_OFFSET &&
                        src_str[src_pos] < UTF16_LOW_SURROGATE_OFFSET)
                    {
                        if (src_pos + 1 < src_length &&
                            src_str[src_pos + 1] >= UTF16_LOW_SURROGATE_OFFSET &&
                            src_str[src_pos + 1] <= SURROGATE_END)
                        {
                            char32_t combined = ((src_str[src_pos] - UTF16_HIGH_SURROGATE_OFFSET) << 10) |
                                (src_str[src_pos + 1] - UTF16_LOW_SURROGATE_OFFSET) + 0x10000;

                            char32_t dst_char{};
                            usize units = cvt_Invoker::operator()(combined, dst_char);

                            for (usize j = 0; j < units; ++j)
                            {
                                dst[dst_pos++] = (dst_char >> (j * dst_char_bit_width)) & mask;
                            }
                            src_pos += 2;
                        }
                        else
                        {
                            return 0;
                        }
                    }
                    else
                    {
                        char32_t dst_char{};
                        usize units = cvt_Invoker::operator()(src_str[src_pos], dst_char);

                        for (usize j = 0; j < units; ++j)
                        {
                            dst[dst_pos++] = (dst_char >> (j * dst_char_bit_width)) & mask;
                        }
                        ++src_pos;
                    }
                }

                return dst_pos;
            }
        }
    };

    template<character_t Dst_Char_t>
    struct string_convert_Invoker<char8_t, Dst_Char_t> : private character_convert_Invoker<char8_t, Dst_Char_t>
    {
        using cvt_Invoker = character_convert_Invoker<char8_t, Dst_Char_t>;
        static constexpr std::make_unsigned_t<Dst_Char_t> mask = std::numeric_limits<std::make_unsigned_t<Dst_Char_t>>::max();
        constexpr string_convert_Invoker() = default;

        constexpr void handle_utf16_encoding(char32_t codepoint, Dst_Char_t* dst, usize& dst_pos) const noexcept
        {
            if (codepoint < UTF8_BYTES_NEEDED_3)
            {
                dst[dst_pos++] = static_cast<Dst_Char_t>(codepoint);
            }
            else
            {
                dst[dst_pos++] = static_cast<Dst_Char_t>((UTF16_HIGH_SURROGATE_OFFSET | ((codepoint - UTF8_BYTES_NEEDED_3) >> 10)));
                dst[dst_pos++] = static_cast<Dst_Char_t>((UTF16_LOW_SURROGATE_OFFSET | ((codepoint - UTF8_BYTES_NEEDED_3) & UTF16_HIGH_SURROGATE_MASK)));
            }
        }

        constexpr void handle_utf32_encoding(char32_t codepoint, Dst_Char_t* dst, usize& dst_pos) const noexcept
        {
            dst[dst_pos++] = static_cast<Dst_Char_t>(codepoint);
        }

        constexpr usize operator() (const char8_t* const src_str, usize src_length, Dst_Char_t* dst) const noexcept
        {
            if (!src_str || !dst)
            {
                return 0;
            }

            usize dst_pos = 0;
            usize src_pos = 0;

            while (src_pos < src_length)
            {
                char32_t first_byte = src_str[src_pos];
                char32_t codepoint = 0;
                usize bytes_read = 0;

                if ((first_byte & UTF8_CONTINUATION_OFFSET) == 0)
                {
                    codepoint = first_byte;
                    bytes_read = 1;
                }
                else if ((first_byte & UTF8_3_BYTE_OFFSET) == UTF8_2_BYTE_OFFSET)
                {
                    codepoint = (first_byte & UTF8_2_BYTE_MASK) << 6;
                    bytes_read = 2;
                }
                else if ((first_byte & UTF8_4_BYTE_OFFSET) == UTF8_3_BYTE_OFFSET)
                {
                    codepoint = (first_byte & UTF8_3_BYTE_MASK) << 12;
                    bytes_read = 3;
                }
                else if ((first_byte & UTF8_4_BYTE_CHECK_MASK) == UTF8_4_BYTE_OFFSET)
                {
                    codepoint = (first_byte & UTF8_4_BYTE_MASK) << 18;
                    bytes_read = 4;
                }
                else
                {
                    return 0;
                }

                for (usize i = 1; i < bytes_read; ++i)
                {
                    if (src_pos + i >= src_length)
                    {
                        return 0;
                    }
                    char8_t continuation_byte = src_str[src_pos + i];
                    if ((continuation_byte & UTF8_2_BYTE_OFFSET) != UTF8_CONTINUATION_OFFSET)
                    {
                        return 0;
                    }
                    codepoint |= (continuation_byte & UTF8_CONTINUATION_MASK) << (6 * (bytes_read - 1 - i));
                }

                if (codepoint > UNICODE_MAX ||
                    (codepoint >= SURROGATE_START &&
                        codepoint <= SURROGATE_END))
                {
                    return 0;
                }

                if constexpr (sizeof(Dst_Char_t) == sizeof(char16_t))
                {
                    handle_utf16_encoding(codepoint, dst, dst_pos);
                }
                else if constexpr (sizeof(Dst_Char_t) == sizeof(char32_t))
                {
                    handle_utf32_encoding(codepoint, dst, dst_pos);
                }
                else
                {
                    std::unreachable();
                }

                src_pos += bytes_read;
            }

            return dst_pos;
        }
    };

    template<character_t Dst_Char_t> struct string_convert_Invoker<wchar_t, Dst_Char_t>
    {
        constexpr string_convert_Invoker() = default;
        constexpr usize operator () (const wchar_t* const src_str, usize src_length, Dst_Char_t* dst) const noexcept
        {
            string_convert_Invoker<char16_t, Dst_Char_t> invoker;
            return invoker(reinterpret_cast<const char16_t* const>(src_str), src_length, dst);
        }
    };

    template<character_t Dst_Char_t> struct string_convert_Invoker<char, Dst_Char_t>
    {
        constexpr string_convert_Invoker() = default;
        constexpr usize operator () (const char* const src_str, usize src_length, Dst_Char_t* dst) const noexcept
        {
            string_convert_Invoker<char8_t, Dst_Char_t> invoker;
            return invoker(reinterpret_cast<const char8_t* const>(src_str), src_length, dst);
        }
    };

    template<character_t Src_Char_t, character_t Dst_Char_t>
    struct calc_buffer_size_needs_for_conversion_Invoker
    {
        constexpr calc_buffer_size_needs_for_conversion_Invoker() = default;
        constexpr usize operator () (const Src_Char_t* const, usize) const noexcept
        {
            std::unreachable();
        }
    };

    template<> struct calc_buffer_size_needs_for_conversion_Invoker<char32_t, char16_t>
    {
        constexpr calc_buffer_size_needs_for_conversion_Invoker() = default;
        constexpr usize operator () (const char32_t* input, usize length) const noexcept
        {
            usize utf16_length = 0;
            for (usize i = 0; i < length; ++i)
            {
                char32_t ch = input[i];

                if (ch <= 0xFFFF)
                {
                    utf16_length += 1;
                }
                else if (ch <= UNICODE_MAX)
                {
                    utf16_length += 2;
                }
                else
                {
                    return 0;
                }
            }
            return utf16_length;
        }
    };

    template<> struct calc_buffer_size_needs_for_conversion_Invoker<char32_t, char8_t>
    {
        constexpr calc_buffer_size_needs_for_conversion_Invoker() = default;
        constexpr usize operator () (const char32_t* input, usize length) const noexcept
        {
            usize utf8_length = 0;
            for (usize i = 0; i < length; ++i)
            {
                char32_t ch = input[i];

                if (ch <= UTF8_1_BYTE_MASK)
                {
                    utf8_length += 1;
                }
                else if (ch < UTF8_BYTES_NEEDED_2)
                {
                    utf8_length += 2;
                }
                else if (ch < UTF8_BYTES_NEEDED_3)
                {
                    utf8_length += 3;
                }
                else if (ch < UTF8_BYTES_NEEDED_4)
                {
                    utf8_length += 4;
                }
                else
                {
                    return 0;
                }
            }
            return utf8_length;
        }
    };

    template<> struct calc_buffer_size_needs_for_conversion_Invoker<char8_t, char32_t>
    {
        constexpr calc_buffer_size_needs_for_conversion_Invoker() = default;
        constexpr usize operator () (const char8_t* input, usize length) const noexcept
        {
            usize utf32_length = 0;
            for (usize i = 0; i < length;)
            {
                unsigned char ch = static_cast<unsigned char>(input[i]);

                if (ch <= UTF8_1_BYTE_MASK)
                {
                    utf32_length += 1;
                    i += 1;
                }
                else if ((ch & UTF8_3_BYTE_OFFSET) == UTF8_2_BYTE_OFFSET)
                {
                    utf32_length += 1;
                    i += 2;
                }
                else if ((ch & UTF8_4_BYTE_OFFSET) == UTF8_3_BYTE_OFFSET)
                {
                    utf32_length += 1;
                    i += 3;
                }
                else if ((ch & UTF8_4_BYTE_CHECK_MASK) == UTF8_4_BYTE_OFFSET)
                {
                    utf32_length += 1;
                    i += 4;
                }
                else
                {
                    return 0;
                }
            }
            return utf32_length;
        }
    };

    template<> struct calc_buffer_size_needs_for_conversion_Invoker<char8_t, char16_t>
    {
        constexpr calc_buffer_size_needs_for_conversion_Invoker() = default;
        constexpr usize operator () (const char8_t* input, usize length) const noexcept
        {
            usize utf16_length = 0;
            for (usize i = 0; i < length;)
            {
                u8 ch = static_cast<u8>(input[i]);

                if (ch <= UTF8_1_BYTE_MASK)
                {
                    utf16_length += 1;
                    i += 1;
                }
                else if ((ch & UTF8_3_BYTE_OFFSET) == UTF8_2_BYTE_OFFSET)
                {
                    utf16_length += 1;
                    i += 2;
                }
                else if ((ch & UTF8_4_BYTE_OFFSET) == UTF8_3_BYTE_OFFSET)
                {
                    utf16_length += 1;
                    i += 3;
                }
                else if ((ch & UTF8_4_BYTE_CHECK_MASK) == UTF8_4_BYTE_OFFSET)
                {
                    utf16_length += 2;
                    i += 4;
                }
                else
                {
                    return 0;
                }
            }
            return utf16_length;
        }
    };

    template<> struct calc_buffer_size_needs_for_conversion_Invoker<char16_t, char32_t>
    {
        constexpr calc_buffer_size_needs_for_conversion_Invoker() = default;
        constexpr usize operator () (const char16_t* input, usize length) const noexcept
        {
            usize utf32_length = 0;
            for (usize i = 0; i < length; ++i)
            {
                char16_t ch = input[i];

                if (ch >= UTF16_HIGH_SURROGATE_OFFSET && ch <= UTF16_HIGH_SURROGATE_MAX)
                {
                    if (i + 1 < length)
                    {
                        char16_t next_ch = input[i + 1];
                        if (next_ch >= UTF16_LOW_SURROGATE_OFFSET && next_ch <= UTF16_LOW_SURROGATE_MAX)
                        {
                            utf32_length += 1;
                            ++i;
                        }
                        else
                        {
                            return 0;
                        }
                    }
                    else
                    {
                        return 0;
                    }
                }
                else if (ch >= UTF16_LOW_SURROGATE_OFFSET && ch <= UTF16_LOW_SURROGATE_MAX)
                {
                    return 0;
                }
                else
                {
                    utf32_length += 1;
                }
            }
            return utf32_length;
        }
    };

    template<> struct calc_buffer_size_needs_for_conversion_Invoker<char16_t, char8_t>
    {
        constexpr calc_buffer_size_needs_for_conversion_Invoker() = default;
        constexpr usize operator () (const char16_t* const input, usize length) const noexcept
        {
            usize utf8_length = 0;
            for (usize i = 0; i < length; ++i)
            {
                char16_t ch = input[i];

                if (ch >= UTF16_HIGH_SURROGATE_OFFSET && ch <= UTF16_HIGH_SURROGATE_MAX)
                {
                    if (i + 1 < length)
                    {
                        char16_t next_ch = input[i + 1];
                        if (next_ch >= UTF16_LOW_SURROGATE_OFFSET && next_ch <= UTF16_LOW_SURROGATE_MAX)
                        {
                            utf8_length += 4;
                            ++i;
                        }
                        else
                        {
                            return 0;
                        }
                    }
                    else
                    {
                        return 0;
                    }
                }
                else if (ch >= UTF16_LOW_SURROGATE_OFFSET && ch <= UTF16_LOW_SURROGATE_MAX)
                {
                    return 0;
                }
                else
                {
                    if (ch <= UTF8_1_BYTE_MASK)
                    {
                        utf8_length += 1;
                    }
                    else if (ch < UTF8_BYTES_NEEDED_2)
                    {
                        utf8_length += 2;
                    }
                    else
                    {
                        utf8_length += 3;
                    }
                }
            }
            return utf8_length;
        }
    };

    template<character_t Dst_Char_t> struct calc_buffer_size_needs_for_conversion_Invoker<wchar_t, Dst_Char_t>
    : private calc_buffer_size_needs_for_conversion_Invoker<char16_t, Dst_Char_t>
    {
        using Base = calc_buffer_size_needs_for_conversion_Invoker<char16_t, Dst_Char_t>;
        constexpr calc_buffer_size_needs_for_conversion_Invoker() = default;
        constexpr usize operator () (const wchar_t* const input, usize length) const noexcept
        {
            return Base::operator () (reinterpret_cast<const char16_t* const>(input), length);
        }
    };

    template<character_t Dst_Char_t> struct calc_buffer_size_needs_for_conversion_Invoker<char, Dst_Char_t>
    : private calc_buffer_size_needs_for_conversion_Invoker<char8_t, Dst_Char_t>
    {
        using Base = calc_buffer_size_needs_for_conversion_Invoker<char16_t, Dst_Char_t>;
        constexpr calc_buffer_size_needs_for_conversion_Invoker() = default;
        constexpr usize operator () (const char* const input, usize length) const noexcept
        {
            return Base::operator () (reinterpret_cast<const char8_t* const>(input), length);
        }
    };
}

