export module foye.fstring_view;

export import foye.foye_core;
export import foye.funicode_cvt;
export import foye.fbytearray;
import std;

#pragma warning(disable: 4309)
#pragma warning(disable: 4244)
#pragma warning(disable: 4552)
#pragma warning(disable: 4552)
#pragma warning(disable: 4018)

export namespace fy
{
    template<character_t Left_t, character_t Right_t> struct string_equal_Invoker
    {
        bool is_equal;
        string_equal_Invoker(
            const Left_t* const left, const Right_t* const right,
            const usize left_count_char, const usize right_count_char) noexcept
        {
            static_assert(sizeof(Left_t) == sizeof(Right_t));
            if (std::is_constant_evaluated())
            {
                if (left_count_char == right_count_char)
                {
                    for (usize i = 0; i < left_count_char; ++i)
                    {
                        if (left[i] != right[i])
                        {
                            is_equal = false;
                            return;
                        }
                    }
                    is_equal = true;
                }
                is_equal = false;
            }
            else
            {
                is_equal = left_count_char == right_count_char && std::memcmp(left, right, left_count_char * sizeof(Left_t)) == 0;
            }

        }

        operator bool() const noexcept { return is_equal; }
    };

    template<> struct string_equal_Invoker<char8_t, char16_t>
    {
        bool is_equal;

        string_equal_Invoker(
            const char8_t* const left, const char16_t* const right,
            const usize left_count_char, const usize right_count_char) noexcept
        {
            if ((left_count_char == 0 && right_count_char != 0) || (right_count_char == 0 && left_count_char != 0))
            {
                is_equal = false;
                return;
            }

            const char8_t* p = left;
            const char8_t* const end = left + left_count_char;
            const char16_t* q = right;
            const char16_t* const q_end = right + right_count_char;

            while (p < end && q < q_end)
            {
                char32_t code_point = 0;
                char8_t ch = *p;
                usize extra_bytes = 0;

                if (ch <= UTF8_1_BYTE_MASK)
                {
                    code_point = ch;
                    p += 1;

                    if (code_point != *q)
                    {
                        is_equal = false;
                        return;
                    }
                    q += 1;
                    continue;
                }
                else if ((ch & UTF8_3_BYTE_OFFSET) == UTF8_2_BYTE_OFFSET)
                {
                    code_point = ch & UTF8_2_BYTE_MASK;
                    extra_bytes = 1;
                }
                else if ((ch & UTF8_4_BYTE_OFFSET) == UTF8_3_BYTE_OFFSET)
                {
                    code_point = ch & UTF8_3_BYTE_MASK;
                    extra_bytes = 2;
                }
                else if ((ch & UTF8_4_BYTE_CHECK_MASK) == UTF8_4_BYTE_OFFSET)
                {
                    code_point = ch & UTF8_4_BYTE_MASK;
                    extra_bytes = 3;
                }
                else
                {
                    is_equal = false;
                    return;
                }

                p++;
                if (p + extra_bytes > end)
                {
                    is_equal = false;
                    return;
                }

                for (usize i = 0; i < extra_bytes; ++i)
                {
                    ch = *p;
                    if ((ch & UTF8_CONTINUATION_CHECK_MASK) != UTF8_CONTINUATION_OFFSET)
                    {
                        is_equal = false;
                        return;
                    }
                    code_point = (code_point << 6) | (ch & UTF8_CONTINUATION_MASK);
                    p++;
                }

                if (code_point < BMP_MAX)
                {
                    if (code_point != *q)
                    {
                        is_equal = false;
                        return;
                    }
                    q++;
                }
                else if (code_point <= UNICODE_MAX)
                {
                    if (q + 1 >= q_end)
                    {
                        is_equal = false;
                        return;
                    }

                    char16_t high = static_cast<char16_t>(((code_point - BMP_MAX) >> 10) + UTF16_HIGH_SURROGATE_OFFSET);
                    char16_t low = static_cast<char16_t>(((code_point - BMP_MAX) & UTF16_HIGH_SURROGATE_MASK) + UTF16_LOW_SURROGATE_OFFSET);

                    if (*q != high || *(q + 1) != low)
                    {
                        is_equal = false;
                        return;
                    }
                    q += 2;
                }
                else
                {
                    is_equal = false;
                    return;
                }
            }

            is_equal = (p == end && q == q_end);
        }

        operator bool() const noexcept { return is_equal; }
    };

    template<> struct string_equal_Invoker<char8_t, char32_t>
    {
        bool is_equal;

        string_equal_Invoker(
            const char8_t* const left, const char32_t* const right,
            const usize left_count_char, const usize right_count_char) noexcept
        {
            if ((left_count_char == 0 && right_count_char != 0) || (right_count_char == 0 && left_count_char != 0))
            {
                is_equal = false;
                return;
            }

            const char8_t* p = left;
            const char8_t* const end = left + left_count_char;
            const char32_t* q = right;
            const char32_t* const q_end = right + right_count_char;

            while (p < end && q < q_end)
            {
                char32_t code_point = 0;
                char8_t ch = *p;
                usize extra_bytes = 0;

                if (ch <= UTF8_1_BYTE_MASK)
                {
                    code_point = ch;
                    p += 1;
                }
                else if ((ch & UTF8_3_BYTE_OFFSET) == UTF8_2_BYTE_OFFSET)
                {
                    code_point = ch & UTF8_2_BYTE_MASK;
                    extra_bytes = 1;
                }
                else if ((ch & UTF8_4_BYTE_OFFSET) == UTF8_3_BYTE_OFFSET)
                {
                    code_point = ch & UTF8_3_BYTE_MASK;
                    extra_bytes = 2;
                }
                else if ((ch & UTF8_4_BYTE_CHECK_MASK) == UTF8_4_BYTE_OFFSET)
                {
                    code_point = ch & UTF8_4_BYTE_MASK;
                    extra_bytes = 3;
                }
                else
                {
                    is_equal = false;
                    return;
                }

                p++;
                if (p + extra_bytes > end)
                {
                    is_equal = false;
                    return;
                }

                for (usize i = 0; i < extra_bytes; ++i)
                {
                    ch = *p;
                    if ((ch & UTF8_CONTINUATION_CHECK_MASK) != UTF8_CONTINUATION_OFFSET)
                    {
                        is_equal = false;
                        return;
                    }
                    code_point = (code_point << 6) | (ch & UTF8_CONTINUATION_MASK);
                    p++;
                }

                if (code_point != *q)
                {
                    is_equal = false;
                    return;
                }
                q++;
            }

            is_equal = (p == end && q == q_end);
        }

        operator bool() const noexcept { return is_equal; }
    };


    template<> struct string_equal_Invoker<char16_t, char32_t>
    {
        bool is_equal;

        string_equal_Invoker(
            const char16_t* const left, const char32_t* const right,
            const usize left_count_char, const usize right_count_char) noexcept
        {
            if ((left_count_char == 0 && right_count_char != 0) || (right_count_char == 0 && left_count_char != 0))
            {
                is_equal = false;
                return;
            }

            const char16_t* p = left;
            const char16_t* const end = left + left_count_char;
            const char32_t* q = right;
            const char32_t* const q_end = right + right_count_char;

            while (p < end && q < q_end)
            {
                char32_t code_point = 0;
                char16_t wch = *p;

                if (wch < SURROGATE_START || wch > SURROGATE_END)
                {
                    code_point = wch;
                    p++;
                }
                else if (wch >= UTF16_HIGH_SURROGATE_OFFSET &&
                    wch < (UTF16_HIGH_SURROGATE_OFFSET + (SURROGATE_START - UTF16_HIGH_SURROGATE_OFFSET)))
                {
                    if (p + 1 >= end)
                    {
                        is_equal = false;
                        return;
                    }
                    char16_t low_wch = *(p + 1);
                    if (low_wch >= UTF16_LOW_SURROGATE_OFFSET &&
                        low_wch < (UTF16_LOW_SURROGATE_OFFSET + (SURROGATE_END - UTF16_LOW_SURROGATE_OFFSET + 1)))
                    {
                        code_point = static_cast<char32_t>(((wch - UTF16_HIGH_SURROGATE_OFFSET) << 10) | (low_wch - UTF16_LOW_SURROGATE_OFFSET)) + BMP_MAX;
                        p += 2;
                    }
                    else
                    {
                        is_equal = false;
                        return;
                    }
                }
                else
                {
                    is_equal = false;
                    return;
                }

                if (code_point != *q)
                {
                    is_equal = false;
                    return;
                }
                q++;
            }

            is_equal = (p == end && q == q_end);
        }

        operator bool() const noexcept { return is_equal; }
    };

    template<> struct string_equal_Invoker<char16_t, char8_t> : private string_equal_Invoker<char8_t, char16_t>
    {
        string_equal_Invoker(
            const char16_t* const left, const char8_t* const right,
            const usize left_count_char, const usize right_count_char) noexcept
            : string_equal_Invoker<char8_t, char16_t>(right, left, right_count_char, left_count_char) {}

        operator bool() const noexcept { return string_equal_Invoker<char8_t, char16_t>::is_equal; }
    };

    template<> struct string_equal_Invoker<char32_t, char8_t> : private string_equal_Invoker<char8_t, char32_t>
    {
        string_equal_Invoker(
            const char32_t* const left, const char8_t* const right,
            const usize left_count_char, const usize right_count_char) noexcept
            : string_equal_Invoker<char8_t, char32_t>(right, left, right_count_char, left_count_char) {}

        operator bool() const noexcept { return string_equal_Invoker<char8_t, char32_t>::is_equal; }
    };

    template<> struct string_equal_Invoker<char32_t, char16_t> : private string_equal_Invoker<char16_t, char32_t>
    {
        string_equal_Invoker(
            const char32_t* const left, const char16_t* const right,
            const usize left_count_char, const usize right_count_char) noexcept
            : string_equal_Invoker<char16_t, char32_t>(right, left, right_count_char, left_count_char) {}

        operator bool() const noexcept { return string_equal_Invoker<char16_t, char32_t>::is_equal; }
    };

    template<character_t Char_t>
    usize calc_complete_characters(const Char_t* const c_str, usize count_char_t = 0)
    {
        if (count_char_t == 0)
        {
            const Char_t* p = c_str;
            while (*p)
            {
                ++count_char_t;
                ++p;
            }
        }

        if constexpr (std::is_same_v<Char_t, char32_t>)
        {
            return count_char_t;
        }

        if constexpr (std::_Is_any_of_v<Char_t, char, char8_t>)
        {
            usize char_count = 0;
            const Char_t* p = c_str;
            const Char_t* end = c_str + count_char_t;

            while (p < end)
            {
                unsigned char first = static_cast<unsigned char>(*p);
                if (first < 0x80)
                {
                    ++p;
                }
                else if (first < 0xE0)
                {
                    p += 2;
                }
                else if (first < 0xF0)
                {
                    p += 3;
                }
                else
                {
                    p += 4;
                }
                ++char_count;
            }
            return char_count;
        }
        else if constexpr (std::_Is_any_of_v<Char_t, wchar_t, char16_t>)
        {
            usize char_count = 0;
            const Char_t* p = c_str;
            const Char_t* end = c_str + count_char_t;

            while (p < end)
            {
                if ((*p & 0xFC00) == 0xD800 && (p + 1) < end)
                {
                    p += 2;
                }
                else
                {
                    ++p;
                }
                ++char_count;
            }

            return char_count;
        }
        else if constexpr (std::is_same_v<Char_t, char32_t>)
        {
            return count_char_t;
        }
        else
        {
            std::unreachable();
        }
    }

    template<character_t Char_t>
    ssize find_character_Char_t_begin(const Char_t* const, usize, usize)
    {
        std::unreachable();
    }

    template<>
    ssize find_character_Char_t_begin(const char32_t* const, usize, usize targetCharIndex)
    {
        return static_cast<ssize>(targetCharIndex);
    }

    template<>
    ssize find_character_Char_t_begin(const char16_t* const str, usize strLen, usize targetCharIndex)
    {
        usize currentIndex = 0;
        usize charCount = 0;

        while (currentIndex < strLen && charCount < targetCharIndex)
        {
            if (currentIndex + 1 < strLen &&
                (str[currentIndex] >= UTF16_HIGH_SURROGATE_OFFSET && str[currentIndex] <= UTF16_HIGH_SURROGATE_MAX) &&
                (str[currentIndex + 1] >= UTF16_LOW_SURROGATE_OFFSET && str[currentIndex + 1] <= UTF16_LOW_SURROGATE_MAX))
            {
                currentIndex += 2;
            }
            else
            {
                currentIndex++;
            }
            charCount++;
        }

        if (charCount == targetCharIndex)
        {
            return static_cast<ssize>(currentIndex);
        }

        return -1;
    }

    template<>
    ssize find_character_Char_t_begin(const char8_t* const str, usize strLen, usize targetCharIndex)
    {
        usize currentIndex = 0;
        usize charCount = 0;

        while (currentIndex < strLen && charCount < targetCharIndex)
        {
            u8 currentByte = static_cast<u8>(str[currentIndex]);

            usize charBytes = 0;
            if ((currentByte & UTF8_CONTINUATION_OFFSET) == 0)
            {
                charBytes = 1;
            }
            else if ((currentByte & UTF8_3_BYTE_OFFSET) == UTF8_2_BYTE_OFFSET)
            {
                charBytes = 2;
            }
            else if ((currentByte & UTF8_4_BYTE_OFFSET) == UTF8_3_BYTE_OFFSET)
            {
                charBytes = 3;
            }
            else if ((currentByte & UTF8_4_BYTE_CHECK_MASK) == UTF8_4_BYTE_OFFSET)
            {
                charBytes = 4;
            }
            else
            {
                currentIndex++;
                continue;
            }

            if (currentIndex + charBytes > strLen)
            {
                break;
            }

            currentIndex += charBytes;
            charCount++;
        }

        if (charCount == targetCharIndex)
        {
            return static_cast<ssize>(currentIndex);
        }

        return -1;
    }

    template<character_t Char_t>
    usize count_char_cstyle_string(const Char_t* const c_str, usize loop_max = usize(std::numeric_limits<ssize>::max()))
    {
        usize count_char_without_end{ 0 };
        do
        {
            ++count_char_without_end;

            if (count_char_without_end >= loop_max)
            {
                return loop_max;
            }

        } while (c_str[count_char_without_end - 1] != Char_t{ 0 });
        --count_char_without_end;

        return count_char_without_end;
    }

    template<character_t Char_t>
    bool is_string_all_ASCII(const Char_t* const str, usize count_Char_t_without_end_symbol)
    {
        for (usize i = 0; i < count_Char_t_without_end_symbol; ++i)
        {
            if (str[i] > Char_t(127))
            {
                return false;
            }
            else
            {
                continue;
            }
        }

        return true;
    }
}





#define STRVIEW_EQUAL(STR_0, STR_1) \
    dispatch_char_t((STR_0).char_size_(), \
        [&]<character_t Self_t>(Self_t) -> bool \
        { \
            return dispatch_char_t((STR_1).char_size_(), \
                [&]<character_t Other_t>(Other_t) -> bool \
                { \
                    string_equal_Invoker<Self_t, Other_t> invoker( \
                        (STR_0).ptr_<Self_t>(), (STR_1).ptr_<Other_t>(), \
                        (STR_0).count_char_(), (STR_1).count_char_() \
                    ); \
                    return invoker; \
                } \
            ); \
        } \
    )

export namespace fy
{
    class fstring_view
    {
    private:
        struct
        {
            struct
            {
                char count_char_t[6];
                char sizeof_char_t[2];
            } info;
            const void* ptr;
        } data_;

    public:
        static constexpr usize npos{ static_cast<usize>(-1) };
        static constexpr usize max_count_char = (1ULL << 48) - 1;
        static constexpr usize max_char_size = std::numeric_limits<u16>::max();

        template<character_t Char_t, usize N>
        consteval fstring_view(const Char_t(&string_literal)[N]) noexcept
        {
            constexpr usize length = N - 1;

            static_assert(length <= max_count_char, "String length exceeds maximum supported size (48-bit)");
            static_assert(sizeof(Char_t) <= max_char_size, "Character type size exceeds maximum supported size (2 bytes)");

            for (usize i = 0; i < sizeof(data_.info.count_char_t); ++i)
            {
                data_.info.count_char_t[i] = static_cast<char>((length >> (i * 8)) & 0xFF);
            }

            constexpr usize char_size = sizeof(Char_t);
            data_.info.sizeof_char_t[0] = static_cast<char>(char_size & 0xFF);
            data_.info.sizeof_char_t[1] = static_cast<char>((char_size >> 8) & 0xFF);

            data_.ptr = string_literal;
        }

        constexpr fstring_view(const fstring_view& other_view) noexcept : data_(other_view.data_) {}

        constexpr fstring_view(fstring_view&& other_view) noexcept : data_(other_view.data_)
        {
            other_view.data_.ptr = nullptr;
            for (usize i = 0; i < sizeof(other_view.data_.info.count_char_t); ++i)
            {
                other_view.data_.info.count_char_t[i] = 0;
            }
            for (usize i = 0; i < sizeof(other_view.data_.info.sizeof_char_t); ++i)
            {
                other_view.data_.info.sizeof_char_t[i] = 0;
            }
        }

        fstring_view& operator = (const fstring_view& other_view) noexcept
        {
            if (this != &other_view)
            {
                data_ = other_view.data_;
            }
            return *this;
        }

        fstring_view& operator = (fstring_view&& other_view) noexcept
        {
            if (this != &other_view)
            {
                data_ = other_view.data_;

                other_view.data_.ptr = nullptr;
                for (usize i = 0; i < sizeof(other_view.data_.info.count_char_t); ++i)
                {
                    other_view.data_.info.count_char_t[i] = 0;
                }
                for (usize i = 0; i < sizeof(other_view.data_.info.sizeof_char_t); ++i)
                {
                    other_view.data_.info.sizeof_char_t[i] = 0;
                }
            }
            return *this;
        }

        template<character_t Char_t>
        constexpr fstring_view(const Char_t* const ptr, usize N) noexcept(!is_debug_mode)
        {
            if constexpr (is_debug_mode)
            {
                if (N > max_count_char)
                {
                    throw std::invalid_argument("String length exceeds maximum supported size (48-bit)");
                }
                if (sizeof(Char_t) > max_char_size)
                {
                    throw std::invalid_argument("Character type size exceeds maximum supported size (2 bytes)");
                }
                if (ptr == nullptr)
                {
                    throw std::invalid_argument("Input character pointer is null");
                }
            }
            
            for (usize i = 0; i < sizeof(data_.info.count_char_t); ++i)
            {
                data_.info.count_char_t[i] = static_cast<char>((N >> (i * 8)) & 0xFF);
            }

            constexpr usize char_size = sizeof(Char_t);
            data_.info.sizeof_char_t[0] = static_cast<char>(char_size & 0xFF);
            data_.info.sizeof_char_t[1] = static_cast<char>((char_size >> 8) & 0xFF);

            data_.ptr = ptr;
        }

        constexpr fstring_view() noexcept
        {
            data_.ptr = nullptr;
            for (usize i = 0; i < sizeof(data_.info.count_char_t); ++i)
            {
                data_.info.count_char_t[i] = 0;
            }
            for (usize i = 0; i < sizeof(data_.info.sizeof_char_t); ++i)
            {
                data_.info.sizeof_char_t[i] = 0;
            }
        }

        constexpr fstring_view substr(usize index_begin, usize str_size = 0) const noexcept
        {
            str_size = str_size == 0 ? size() - index_begin : str_size;

            return dispatch_char_t(char_size_(),
                [this, index_begin, str_size]<character_t Char_t>(Char_t) -> fstring_view
                {
                    const usize count_char = count_char_();
                    const Char_t* const data = this->ptr_<Char_t>();

                    const ssize index_begin_ = find_character_Char_t_begin(data, count_char, index_begin);
                    const ssize index_end_ = find_character_Char_t_begin(data, count_char, index_begin + str_size);

                    return fstring_view(data + index_begin_, index_end_ - index_begin_);
                }
            );
        }

        constexpr bool is_begin_with(fstring_view prefix) const noexcept
        {
            if (prefix.empty())
            {
                return false;
            }
            else
            {
                if (empty())
                {
                    return false;
                }
                else
                {
                    if (prefix.size() > size())
                    {
                        return false;
                    }
                }
            }

            fstring_view temp_part = substr(0, prefix.size());

            return dispatch_char_t(char_size_(),
                [this, prefix, temp_part]<character_t Self_t>(Self_t) -> bool
                {
                    return dispatch_char_t(prefix.char_size_(),
                        [this, prefix, temp_part]<character_t Other_t>(Other_t) -> bool
                        {
                            string_equal_Invoker<Self_t, Other_t> invoker(
                                temp_part.ptr_<Self_t>(), prefix.ptr_<Other_t>(),
                                temp_part.count_char_(), prefix.count_char_()
                            );

                            return invoker;
                        }
                    );
                }
            );

            return false;
        }

        constexpr bool is_end_with(fstring_view suffix) const noexcept
        {
            if (suffix.empty())
            {
                return false;
            }
            else
            {
                if (empty())
                {
                    return false;
                }
                else
                {
                    if (suffix.size() > size())
                    {
                        return false;
                    }
                }
            }

            fstring_view temp_part = substr(size() - suffix.size());

            return dispatch_char_t(char_size_(),
                [this, suffix, temp_part]<character_t Self_t>(Self_t) -> bool
                {
                    return dispatch_char_t(suffix.char_size_(),
                        [this, suffix, temp_part]<character_t Other_t>(Other_t) -> bool
                        {
                            string_equal_Invoker<Self_t, Other_t> invoker(
                                temp_part.ptr_<Self_t>(), suffix.ptr_<Other_t>(),
                                temp_part.count_char_(), suffix.count_char_()
                            );

                            return invoker;
                        }
                    );
                }
            );
        }

        constexpr fstring_view operator [] (usize index) const noexcept
        {
            if (index >= size())
            {
                return fstring_view();
            }

            return dispatch_char_t(char_size_(),
                [this, index]<character_t Char_t>(Char_t) -> fstring_view
                {
                    if constexpr (std::is_same_v<Char_t, char32_t>)
                    {
                        return fstring_view(ptr_<char32_t>() + index, 1);
                    }
                    else
                    {
                        const usize count_char = count_char_();
                        const Char_t* const data_ptr = ptr_<Char_t>();
                        const ssize index_begin = find_character_Char_t_begin(data_ptr, count_char, index);

                        const Char_t* char_start = data_ptr + index_begin;

                        if (index_begin < 0)
                        {
                            return fstring_view();
                        }

                        if constexpr (std::is_same_v<Char_t, char8_t>)
                        {
                            u8 currentByte = static_cast<u8>(*char_start);
                            usize charBytes{};
                            if ((currentByte & UTF8_CONTINUATION_OFFSET) == 0) { charBytes = 1; }
                            else if ((currentByte & UTF8_3_BYTE_OFFSET) == UTF8_2_BYTE_OFFSET) { charBytes = 2; }
                            else if ((currentByte & UTF8_4_BYTE_OFFSET) == UTF8_3_BYTE_OFFSET) { charBytes = 3; }
                            else if ((currentByte & UTF8_4_BYTE_CHECK_MASK) == UTF8_4_BYTE_OFFSET) { charBytes = 4; }
                            else { return fstring_view(); }

                            return fstring_view(char_start, charBytes);
                        }
                        else if constexpr (std::is_same_v<Char_t, char16_t>)
                        {
                            usize count_char16 = 1;
                            if (*char_start >= UTF16_HIGH_SURROGATE_OFFSET &&
                                *char_start <= UTF16_HIGH_SURROGATE_MAX &&
                                index_begin + 1 < count_char &&
                                data_ptr[index_begin + 1] >= UTF16_LOW_SURROGATE_OFFSET &&
                                data_ptr[index_begin + 1] <= UTF16_LOW_SURROGATE_MAX)
                            {
                                count_char16 = 2;
                            }
                            return fstring_view(char_start, count_char16);
                        }
                        else
                        {
                            std::unreachable();
                        }
                    }
                }
            );
        }

        constexpr bool operator == (fstring_view other_view) const noexcept
        {
            const usize self_count_char = count_char_();
            const usize other_count_char = other_view.count_char_();

            const usize self_char_size = char_size_();
            const usize other_char_size = other_view.char_size_();

            const void* self_data = this->ptr_();
            const void* other_data = other_view.ptr_();

            if (self_char_size == other_char_size && self_count_char != other_count_char)
            {
                return false;
            }

            return dispatch_char_t(self_char_size,
                [&]<character_t Self_t>(Self_t) -> bool
                {
                    return dispatch_char_t(other_char_size,
                        [&]<character_t Other_t>(Other_t) -> bool
                        {
                            string_equal_Invoker<Self_t, Other_t> invoker(
                                reinterpret_cast<const Self_t* const>(self_data),
                                reinterpret_cast<const Other_t* const>(other_data),
                                self_count_char,
                                other_count_char
                            );

                            return invoker;
                        }
                    );
                }
            );
        }

        constexpr bool empty() const noexcept
        {
            return data_.ptr == nullptr || count_char_() == 0;
        }

        constexpr usize size() const noexcept
        {
            return dispatch_char_t(char_size_(),
                [this]<character_t Char_t>(Char_t) -> usize
                {
                    return calc_complete_characters<Char_t>(ptr_<Char_t>(), count_char_());
                }
            );
        }

        template<typename Func>
        static constexpr auto dispatch_char_t(const usize char_size, Func&& func)
        {
            switch (char_size)
            {
            case 1:  return func(char8_t{});
            case 2:  return func(char16_t{});
            case 4:  return func(char32_t{});
            default: { std::unreachable(); }
            }
        }

        constexpr usize count_char_() const noexcept
        {
            usize result = 0;
            for (usize i = 0; i < 6; ++i)
            {
                result |= static_cast<usize>(static_cast<u8>(data_.info.count_char_t[i])) << (i * 8);
            }
            return result;
        }

        constexpr usize char_size_() const noexcept
        {
            return static_cast<usize>(static_cast<u8>(data_.info.sizeof_char_t[1]) << 8)
                | static_cast<usize>(static_cast<u8>(data_.info.sizeof_char_t[0]));
        }

        template<typename Pointer_t = void>
        constexpr const Pointer_t* ptr_() const noexcept
        {
            return reinterpret_cast<const Pointer_t*>(data_.ptr);
        }

        constexpr fbytearray encode_as_UTF8() const noexcept
        {
            return dispatch_char_t(char_size_(),
                [this]<character_t Char_t>(Char_t) -> fbytearray
                {
                    fbytearray result(need_capacity_byte_to_convert_to_UTF8__());
                    constexpr string_convert_Invoker<Char_t, char8_t> invoker{};
                    result.resize(invoker(ptr_<Char_t>(), count_char_(), result.data<char8_t>()) * sizeof(char8_t));

                    return result;
                }
            );
        }

        template<character_t Char_t>
        constexpr fbytearray c_str() const noexcept
        {
            if constexpr (sizeof(Char_t) == sizeof(char8_t))
            {
                fbytearray res = encode_as_UTF8();
                res.push_back<Char_t>(Char_t{ 0 });
                return res;
            }
            else if constexpr (sizeof(Char_t) == sizeof(char16_t))
            {
                fbytearray res = encode_as_UTF16();
                res.push_back<Char_t>(Char_t{ 0 });
                return res;
            }
            else if constexpr (sizeof(Char_t) == sizeof(char32_t))
            {
                fbytearray res = encode_as_UTF32();
                res.push_back<Char_t>(Char_t{ 0 });
                return res;
            }
            else
            {
                std::unreachable();
            }
        }

        constexpr fbytearray encode_as_UTF16() const noexcept
        {
            return dispatch_char_t(char_size_(),
                [this]<character_t Char_t>(Char_t) -> fbytearray
                {
                    fbytearray result(need_capacity_byte_to_convert_to_UTF16__());
                    constexpr string_convert_Invoker<Char_t, char16_t> invoker{};
                    result.resize(invoker(ptr_<Char_t>(), count_char_(), result.data<char16_t>()) * sizeof(char16_t));

                    return result;
                }
            );
        }

        constexpr fbytearray encode_as_UTF32() const noexcept
        {
            return dispatch_char_t(char_size_(),
                [this]<character_t Char_t>(Char_t) -> fbytearray
                {
                    fbytearray result(need_capacity_byte_to_convert_to_UTF32__());
                    constexpr string_convert_Invoker<Char_t, char32_t> invoker{};
                    result.resize(invoker(ptr_<Char_t>(), count_char_(), result.data<char32_t>()) * sizeof(char32_t));

                    return result;
                }
            );
        }

        constexpr usize need_capacity_byte_to_convert_to_UTF8__() const noexcept
        {
            return dispatch_char_t(char_size_(),
                [this]<character_t Char_t>(Char_t) -> usize
                {
                    if constexpr (std::is_same_v<Char_t, char8_t>)
                    {
                        return count_char_() * sizeof(Char_t);
                    }
                    constexpr calc_buffer_size_needs_for_conversion_Invoker<Char_t, char8_t> invoker;
                    return invoker(ptr_<Char_t>(), count_char_());
                }
            );
        }

        constexpr usize need_capacity_byte_to_convert_to_UTF16__() const noexcept
        {
            return dispatch_char_t(char_size_(),
                [this]<character_t Char_t>(Char_t) -> usize
                {
                    if constexpr (std::is_same_v<Char_t, char16_t>)
                    {
                        return count_char_() * sizeof(Char_t);
                    }
                    constexpr calc_buffer_size_needs_for_conversion_Invoker<Char_t, char16_t> invoker;
                    return invoker(ptr_<Char_t>(), count_char_()) * sizeof(char16_t);
                }
            );
        }

        constexpr usize need_capacity_byte_to_convert_to_UTF32__() const noexcept
        {
            return dispatch_char_t(char_size_(),
                [this]<character_t Char_t>(Char_t) -> usize
                {
                    if constexpr (std::is_same_v<Char_t, char32_t>)
                    {
                        return count_char_() * sizeof(Char_t);
                    }
                    constexpr calc_buffer_size_needs_for_conversion_Invoker<Char_t, char32_t> invoker;
                    return invoker(ptr_<Char_t>(), count_char_()) * sizeof(char32_t);
                }
            );
        }

    };

}