export module foye.string_algorithm;

export import foye.foye_core;
export import foye.farray;
export import foye.funicode_cvt;
export import foye.farray_view;
import std;

export namespace fy
{
    u8 parse_UTF8_bytecount(char8_t first_byte)
    {
        if ((first_byte & UTF8_CONTINUATION_OFFSET) == 0) return 1;
        if ((first_byte & UTF8_3_BYTE_OFFSET) == UTF8_2_BYTE_OFFSET) return 2;
        if ((first_byte & UTF8_4_BYTE_OFFSET) == UTF8_3_BYTE_OFFSET) return 3;
        if ((first_byte & UTF8_4_BYTE_CHECK_MASK) == UTF8_4_BYTE_OFFSET) return 4;
        else [[unlikely]] return 0;
    }

    bool is_valid_utf8_char(const char8_t* str, usize max_length)
    {
        if (!str || max_length == 0) return false;

        usize char_size = parse_UTF8_bytecount(str[0]);
        if (char_size == 0 || char_size > max_length) return false;

        for (usize i = 1; i < char_size; ++i)
        {
            if ((str[i] & UTF8_CONTINUATION_CHECK_MASK) != UTF8_CONTINUATION_OFFSET)
            {
                return false;
            }
        }

        return true;
    }

    template<character_t Main_t, character_t Match_t> struct brute_force_matching_Invoker
    {
        brute_force_matching_Invoker(farray_view<Main_t> main_str_, farray_view<Match_t> match_str_) noexcept
            : main_ref(main_str_), match_ref(match_str_) { std::unreachable(); }

        farray<usize> operator () () const noexcept { std::unreachable(); }

        const farray_view<Main_t>& main_ref;
        const farray_view<Match_t>& match_ref;
    };

    template<> struct brute_force_matching_Invoker<char8_t, char8_t>
    {
        constexpr brute_force_matching_Invoker(farray_view<char8_t> main_str_, farray_view<char8_t> match_str_) noexcept
            : main_ref(main_str_), match_ref(match_str_) { }

        template<bool FirstMatchOnly = false>
        constexpr typename std::conditional<FirstMatchOnly, ssize, farray<usize>>::type index_of_char_t_offset() const noexcept
        {
            return match_impl<FirstMatchOnly, false>();
        }

        template<bool FirstMatchOnly = false>
        constexpr typename std::conditional<FirstMatchOnly, ssize, farray<usize>>::type index_of_logical_character_offset() const noexcept
        {
            return match_impl<FirstMatchOnly, true>();
        }

    private:
        template<bool FirstMatchOnly, bool return_char_index>
        constexpr typename std::conditional<FirstMatchOnly, ssize, farray<usize>>::type match_impl() const noexcept
        {
            using ReturnType = typename std::conditional<FirstMatchOnly, ssize, farray<usize>>::type;
            ReturnType result;

            usize i = 0;
            usize char_index = 0;

            const usize count_char_main_str = main_ref.size();
            const usize count_char_match_str = match_ref.size();

            const char8_t* main_str = main_ref.data();
            const char8_t* match_str = match_ref.data();

            while (i < count_char_main_str)
            {
                if (!is_valid_utf8_char(main_str + i, count_char_main_str - i))
                {
                    ++i;
                    if constexpr (return_char_index)
                    {
                        ++char_index;
                    }
                    continue;
                }

                bool found = true;
                usize main_pos = i;
                usize match_pos = 0;

                while (match_pos < count_char_match_str)
                {
                    usize main_char_size = parse_UTF8_bytecount(main_str[main_pos]);
                    usize match_char_size = parse_UTF8_bytecount(match_str[match_pos]);

                    if (main_char_size == 0 || match_char_size == 0 ||
                        main_pos + main_char_size > count_char_main_str ||
                        match_pos + match_char_size > count_char_match_str)
                    {
                        found = false;
                        break;
                    }

                    for (usize j = 0; j < main_char_size; ++j)
                    {
                        if (main_str[main_pos + j] != match_str[match_pos + j])
                        {
                            found = false;
                            break;
                        }
                    }

                    if (!found)
                    {
                        break;
                    }

                    main_pos += main_char_size;
                    match_pos += match_char_size;
                }

                if (found)
                {
                    if constexpr (FirstMatchOnly)
                    {
                        if constexpr (return_char_index)
                        {
                            return static_cast<ssize>(char_index);
                        }
                        else
                        {
                            return static_cast<ssize>(i);
                        }
                    }
                    else
                    {
                        if constexpr (return_char_index)
                        {
                            result.push_back(char_index);
                        }
                        else
                        {
                            result.push_back(i);
                        }
                    }
                }

                usize char_size = parse_UTF8_bytecount(main_str[i]);
                i += (char_size > 0) ? char_size : 1;
                if constexpr (return_char_index)
                {
                    ++char_index;
                }
            }

            if constexpr (FirstMatchOnly)
            {
                return -1;
            }
            else
            {
                return result;
            }
        }

        const farray_view<char8_t>& main_ref;
        const farray_view<char8_t>& match_ref;
    };


    template<> struct brute_force_matching_Invoker<char16_t, char16_t>
    {
        constexpr brute_force_matching_Invoker(farray_view<char16_t> main_str_, farray_view<char16_t> match_str_) noexcept
            : main_ref(main_str_), match_ref(match_str_) { }

        template<bool FirstMatchOnly = false>
        constexpr typename std::conditional<FirstMatchOnly, ssize, farray<usize>>::type index_of_char_t_offset() const noexcept
        {
            return match_impl<FirstMatchOnly, false>();
        }

        template<bool FirstMatchOnly = false>
        constexpr typename std::conditional<FirstMatchOnly, ssize, farray<usize>>::type index_of_logical_character_offset() const noexcept
        {
            return match_impl<FirstMatchOnly, true>();
        }

    private:
        template<bool FirstMatchOnly, bool ReturnLogicalIndex>
        constexpr typename std::conditional<FirstMatchOnly, ssize, farray<usize>>::type match_impl() const noexcept
        {
            using ReturnType = typename std::conditional<FirstMatchOnly, ssize, farray<usize>>::type;
            ReturnType result;

            const usize count_char_main_str = main_ref.size();
            const usize count_char_match_str = match_ref.size();

            const char16_t* main_str = main_ref.data();
            const char16_t* match_str = match_ref.data();

            usize logical_index = 0;

            for (usize i = 0; i <= count_char_main_str - count_char_match_str; ++i)
            {
                bool found = true;

                usize main_pos = i;
                usize match_pos = 0;

                while (match_pos < count_char_match_str)
                {
                    usize main_char_size = utf16_char_size(main_str + main_pos, count_char_main_str - main_pos);
                    usize match_char_size = utf16_char_size(match_str + match_pos, count_char_match_str - match_pos);

                    if (main_char_size == 0 || match_char_size == 0 ||
                        main_pos + main_char_size > count_char_main_str ||
                        match_pos + match_char_size > count_char_match_str)
                    {
                        found = false;
                        break;
                    }

                    for (usize j = 0; j < main_char_size; ++j)
                    {
                        if (main_str[main_pos + j] != match_str[match_pos + j])
                        {
                            found = false;
                            break;
                        }
                    }

                    if (!found)
                    {
                        break;
                    }

                    main_pos += main_char_size;
                    match_pos += match_char_size;
                }

                if (found)
                {
                    if constexpr (FirstMatchOnly)
                    {
                        if constexpr (ReturnLogicalIndex)
                        {
                            return static_cast<ssize>(logical_index);
                        }
                        else
                        {
                            return static_cast<ssize>(i);
                        }
                    }
                    else
                    {
                        if constexpr  (ReturnLogicalIndex)
                        {
                            result.push_back(logical_index);
                        }
                        else
                        {
                            result.push_back(i);
                        }
                    }
                }

                if constexpr (ReturnLogicalIndex)
                {
                    logical_index += utf16_char_size(main_str + i, count_char_main_str - i);
                }
            }

            if constexpr (FirstMatchOnly)
            {
                return -1;
            }
            else
            {
                return result;
            }
        }

        constexpr static usize utf16_char_size(const char16_t* str, usize remaining) noexcept
        {
            if (remaining == 0)
            {
                return 0;
            }

            char16_t first = str[0];
            if (first >= 0xD800 && first <= 0xDBFF)
            {
                if (remaining >= 2 && str[1] >= 0xDC00 && str[1] <= 0xDFFF)
                {
                    return 2;
                }
                return 0;
            }
            else if (first >= 0xDC00 && first <= 0xDFFF)
            {
                return 0;
            }
            else
            {
                return 1;
            }
        }

        const farray_view<char16_t>& main_ref;
        const farray_view<char16_t>& match_ref;
    };



    template<> struct brute_force_matching_Invoker<char32_t, char32_t>
    {
        constexpr brute_force_matching_Invoker(farray_view<char32_t> main_str_, farray_view<char32_t> match_str_) noexcept
            : main_ref(main_str_), match_ref(match_str_)  { }

        template<bool FirstMatchOnly = false>
        constexpr typename std::conditional<FirstMatchOnly, ssize, farray<usize>>::type index_of_char_t_offset() const noexcept
        {
            return match_impl<FirstMatchOnly, false>();
        }

        template<bool FirstMatchOnly = false>
        constexpr typename std::conditional<FirstMatchOnly, ssize, farray<usize>>::type index_of_logical_character_offset() const noexcept
        {
            return match_impl<FirstMatchOnly, true>();
        }

    private:
        template<bool FirstMatchOnly, bool ReturnLogicalIndex>
        constexpr typename std::conditional<FirstMatchOnly, ssize, farray<usize>>::type match_impl() const noexcept
        {
            using ReturnType = typename std::conditional<FirstMatchOnly, ssize, farray<usize>>::type;
            ReturnType result;

            const usize count_char_main_str = main_ref.size();
            const usize count_char_match_str = match_ref.size();

            const char32_t* main_str = main_ref.data();
            const char32_t* match_str = match_ref.data();

            usize logical_index = 0;

            for (usize i = 0; i <= count_char_main_str - count_char_match_str; ++i)
            {
                bool found = true;

                for (usize j = 0; j < count_char_match_str; ++j)
                {
                    if (main_str[i + j] != match_str[j])
                    {
                        found = false;
                        break;
                    }
                }

                if (found)
                {
                    if constexpr (FirstMatchOnly)
                    {
                        return ReturnLogicalIndex ? static_cast<ssize>(logical_index) : static_cast<ssize>(i);
                    }
                    else
                    {
                        if (ReturnLogicalIndex)
                            result.push_back(logical_index);
                        else
                            result.push_back(i);
                    }
                }

                if (ReturnLogicalIndex)
                {
                    ++logical_index;
                }
            }

            if constexpr (FirstMatchOnly)
            {
                return -1;
            }
            else
            {
                return result;
            }
        }

        const farray_view<char32_t>& main_ref;
        const farray_view<char32_t>& match_ref;
    };

    template<character_t Char_t>
    struct boyer_moore_string_match_Invoker 
    {
        static_assert(false);
    };

    template<> struct boyer_moore_string_match_Invoker<char32_t>
    {
        const char32_t* pattern_ptr;
        usize pattern_length;
        std::unordered_map<char32_t, usize> bad_char_table;

        farray<usize> good_suffix_shift;
        farray<usize> matched_prefix_position;
        farray<usize> suffix_length;

        boyer_moore_string_match_Invoker(farray_view<char32_t> pattern) noexcept
            : pattern_ptr(pattern.data()), pattern_length(pattern.size())
            , good_suffix_shift(farray<usize>(pattern_length, 0))
            , matched_prefix_position(farray<usize>(pattern_length, 0))
            , suffix_length(farray<usize>(pattern_length + 1, 0))
        {
            for (usize i = 0; i < pattern_length - 1; ++i)
            {
                bad_char_table[pattern[i]] = pattern_length - 1 - i;
            }

            usize lastPrefixPos = pattern_length;

            for (usize i = pattern_length; i > 0; --i)
            {
                if (i == pattern_length)
                {
                    suffix_length[i] = pattern_length;
                    continue;
                }

                usize j = i;
                while (j > 0 && pattern_ptr[j - 1] == pattern_ptr[pattern_length - 1 - (i - j)])
                {
                    --j;
                }
                suffix_length[i] = i - j;

                if (j == 0)
                {
                    lastPrefixPos = i;
                }
            }

            for (usize i = 0; i < pattern_length; ++i)
            {
                good_suffix_shift[i] = pattern_length;
            }

            for (usize i = pattern_length - 1; i != static_cast<usize>(-1); --i)
            {
                if (suffix_length[i + 1] == i + 1)
                {
                    for (usize j = 0; j < pattern_length - 1 - i; ++j)
                    {
                        if (good_suffix_shift[j] == pattern_length)
                        {
                            good_suffix_shift[j] = pattern_length - 1 - i;
                        }
                    }
                }
            }

            for (usize i = 0; i < pattern_length - 1; ++i)
            {
                good_suffix_shift[pattern_length - 1 - suffix_length[i + 1]] =
                    pattern_length - 1 - i;
            }

            matched_prefix_position[pattern_length - 1] = lastPrefixPos;
        }

        farray<usize> match_all(farray_view<char32_t> source) noexcept
        {
            farray<usize> positions;
            usize source_len = source.size();

            usize shift = 0;
            while (shift <= source_len - pattern_length)
            {
                usize j = pattern_length - 1;

                while (j != static_cast<usize>(-1) && pattern_ptr[j] == source[shift + j])
                {
                    --j;
                }

                if (j == static_cast<usize>(-1))
                {
                    positions.push_back(shift);
                    shift += (shift + pattern_length < source_len) ?
                        pattern_length - bad_char_table[source[shift + pattern_length]] : 1;
                }
                else
                {
                    char32_t badChar = source[shift + j];
                    usize badCharShift = bad_char_table.count(badChar) ?
                        pattern_length - 1 - bad_char_table[badChar] : pattern_length;

                    usize goodSuffixShiftDistance = good_suffix_shift[j];

                    shift += std::max(badCharShift, goodSuffixShiftDistance);
                }
            }

            return positions;
        }

        ssize match_first(farray_view<char32_t> source) noexcept
        {
            usize source_len = source.size();

            usize shift = 0;
            while (shift <= source_len - pattern_length)
            {
                usize j = pattern_length - 1;

                while (j != static_cast<usize>(-1) && pattern_ptr[j] == source[shift + j])
                {
                    --j;
                }

                if (j == static_cast<usize>(-1))
                {
                    return shift;
                }
                else
                {
                    char32_t badChar = source[shift + j];
                    usize badCharShift = bad_char_table.count(badChar) ?
                        pattern_length - 1 - bad_char_table[badChar] : pattern_length;

                    usize goodSuffixShiftDistance = good_suffix_shift[j];

                    shift += std::max(badCharShift, goodSuffixShiftDistance);
                }
            }

            return -1;
        }
    };

    template<character_t Char_t>
    void to_lowercase(Char_t* str, std::size_t length)
    {
        constexpr Char_t A = [ ]() {
            if constexpr (std::is_same_v<Char_t, char8_t>) return u8'A';
            else if constexpr (std::is_same_v<Char_t, char16_t>) return u'A';
            else return U'A';
            }();

        constexpr Char_t Z = [ ]() {
            if constexpr (std::is_same_v<Char_t, char8_t>) return u8'Z';
            else if constexpr (std::is_same_v<Char_t, char16_t>) return u'Z';
            else return U'Z';
            }();

        constexpr Char_t case_diff = ([ ]() {
            if constexpr (std::is_same_v<Char_t, char8_t>) return u8'a';
            else if constexpr (std::is_same_v<Char_t, char16_t>) return u'a';
            else return U'a';
            }()) - A;

        for (usize i = 0; i < length; ++i)
        {
            if (str[i] >= A && str[i] <= Z)
            {
                str[i] = str[i] + case_diff;
            }
        }
    }

    template<character_t Char_t>
    void to_uppercase(Char_t* str, std::size_t length)
    {
        constexpr Char_t a = [ ]() {
            if constexpr (std::is_same_v<Char_t, char8_t>) return u8'a';
            else if constexpr (std::is_same_v<Char_t, char16_t>) return u'a';
            else return U'a';
            }();

        constexpr Char_t z = [ ]() {
            if constexpr (std::is_same_v<Char_t, char8_t>) return u8'z';
            else if constexpr (std::is_same_v<Char_t, char16_t>) return u'z';
            else return U'z';
            }();

        constexpr Char_t case_diff = a - ([ ]() {
            if constexpr (std::is_same_v<Char_t, char8_t>) return u8'A';
            else if constexpr (std::is_same_v<Char_t, char16_t>) return u'A';
            else return U'A';
            }());

        for (usize i = 0; i < length; ++i)
        {
            if (str[i] >= a && str[i] <= z)
            {
                str[i] = str[i] - case_diff;
            }
        }
    }












    template<character_t Char_t>
    Char_t control_character_array[] = {
        Char_t('\0'),
        Char_t('\n'),
        Char_t('\r'),
        Char_t('\t'),
        Char_t('\v'),
        Char_t('\f'),
        Char_t('\b'),
        Char_t('\a'),
        Char_t('\x00'),
        Char_t('\x01'),
        Char_t('\x02'),
        Char_t('\x03'),
        Char_t('\x04'),
        Char_t('\x05'),
        Char_t('\x06'),
        Char_t('\x07'),
        Char_t('\x08'),
        Char_t('\x09'),
        Char_t('\x0A'),
        Char_t('\x0B'),
        Char_t('\x0C'),
        Char_t('\x0D'),
        Char_t('\x0E'),
        Char_t('\x0F'),
        Char_t('\x10'),
        Char_t('\x11'),
        Char_t('\x12'),
        Char_t('\x13'),
        Char_t('\x14'),
        Char_t('\x15'),
        Char_t('\x16'),
        Char_t('\x17'),
        Char_t('\x18'),
        Char_t('\x19'),
        Char_t('\x1A'),
        Char_t('\x1B'),
        Char_t('\x1C'),
        Char_t('\x1D'),
        Char_t('\x1E'),
        Char_t('\x1F'),
        Char_t('\x7F'),
        Char_t('\x80'),
        Char_t('\x81'),
        Char_t('\x82'),
        Char_t('\x83'),
        Char_t('\x84'),
        Char_t('\x85'),
        Char_t('\x86'),
        Char_t('\x87'),
        Char_t('\x88'),
        Char_t('\x89'),
        Char_t('\x8A'),
        Char_t('\x8B'),
        Char_t('\x8C'),
        Char_t('\x8D'),
        Char_t('\x8E'),
        Char_t('\x8F'),
        Char_t('\x90'),
        Char_t('\x91'),
        Char_t('\x92'),
        Char_t('\x93'),
        Char_t('\x94'),
        Char_t('\x95'),
        Char_t('\x96'),
        Char_t('\x97'),
        Char_t('\x98'),
        Char_t('\x99'),
        Char_t('\x9A'),
        Char_t('\x9B'),
        Char_t('\x9C'),
        Char_t('\x9D'),
        Char_t('\x9E'),
        Char_t('\x9F')
    };


    template<character_t Char_t>
    constexpr bool is_whitespace(Char_t c)
    {
        for (usize i = 0; i < (sizeof(control_character_array<Char_t>) / sizeof(Char_t)); ++i)
        {
            if (c == control_character_array<Char_t>[i])
            {
                return true;
            }
            else
            {
                continue;
            }
        }

        return false;
    }

    template<character_t Char_t>
    usize trim(Char_t* str, usize len, bool remove_internal_whitespace)
    {
        usize start = 0;

        while (start < len && is_whitespace(str[start]))
        {
            ++start;
        }

        usize end = len;
        while (end > start && is_whitespace(str[end - 1]))
        {
            --end;
        }

        usize new_len = end - start;

        if (remove_internal_whitespace)
        {
            usize write_index = 0;
            for (usize i = start; i < end; ++i)
            {
                if (!is_whitespace(str[i]))
                {
                    str[write_index++] = str[i];
                }
            }
            return write_index;
        }
        else
        {
            if (start > 0 && new_len > 0)
            {
                for (usize i = 0; i < new_len; ++i)
                {
                    str[i] = str[start + i];
                }
            }
            return new_len;
        }
    }



    template<character_t Char_t>
    usize count_alpha_in_string(const Char_t* const strptr, usize count_char)
    {
        static_assert(std::is_same_v<Char_t, char32_t>);

        static constexpr Char_t upper_begin = Char_t(65);
        static constexpr Char_t upper_end = Char_t(90);

        static constexpr Char_t lower_begin = Char_t(97);
        static constexpr Char_t lower_end = Char_t(122);

        usize count = 0;
        for (usize i = 0; i < count_char; ++i)
        {
            Char_t ch = strptr[i];
            if ((ch >= upper_begin && ch <= upper_end) || (ch >= lower_begin && ch <= lower_end))
            {
                ++count;
            }
        }
        return count;
    }

    template<>
    usize count_alpha_in_string<char16_t>(const char16_t* const strptr, usize count_char)
    {
        usize count = 0;

        for (usize i = 0; i < count_char;)
        {
            char16_t first_word = strptr[i];

            if (first_word >= UTF16_HIGH_SURROGATE_OFFSET && first_word <= UTF16_HIGH_SURROGATE_MAX)
            {
                if (i + 1 < count_char)
                {
                    char16_t second_word = strptr[i + 1];
                    if (second_word >= UTF16_LOW_SURROGATE_OFFSET && second_word <= UTF16_LOW_SURROGATE_MAX)
                    {
                        char32_t codepoint = ((first_word - UTF16_HIGH_SURROGATE_OFFSET) << 10) +
                            (second_word - UTF16_LOW_SURROGATE_OFFSET) + BMP_MAX;

                        if ((codepoint >= 'A' && codepoint <= 'Z') || (codepoint >= 'a' && codepoint <= 'z'))
                        {
                            ++count;
                        }

                        i += 2;
                        continue;
                    }
                }
            }

            if ((first_word >= 'A' && first_word <= 'Z') || (first_word >= 'a' && first_word <= 'z'))
            {
                ++count;
            }

            ++i;
        }

        return count;
    }

    template<>
    usize count_alpha_in_string<char8_t>(const char8_t* const strptr, usize count_char)
    {
        usize count = 0;

        for (usize i = 0; i < count_char;)
        {
            char8_t first_byte = strptr[i];
            u8 byte_count = parse_UTF8_bytecount(first_byte);

            if (byte_count == 0 || i + byte_count > count_char)
            {
                ++i;
                continue;
            }

            if (byte_count == 1)
            {
                if ((first_byte >= 'A' && first_byte <= 'Z') || (first_byte >= 'a' && first_byte <= 'z'))
                {
                    ++count;
                }
            }
            else
            {
                char32_t codepoint = 0;

                if (byte_count == 2)
                {
                    codepoint = ((first_byte & UTF8_2_BYTE_MASK) << 6) |
                        (strptr[i + 1] & UTF8_CONTINUATION_MASK);
                }
                else if (byte_count == 3)
                {
                    codepoint = ((first_byte & UTF8_3_BYTE_MASK) << 12) |
                        ((strptr[i + 1] & UTF8_CONTINUATION_MASK) << 6) |
                        (strptr[i + 2] & UTF8_CONTINUATION_MASK);
                }
                else if (byte_count == 4)
                {
                    codepoint = ((first_byte & UTF8_4_BYTE_MASK) << 18) |
                        ((strptr[i + 1] & UTF8_CONTINUATION_MASK) << 12) |
                        ((strptr[i + 2] & UTF8_CONTINUATION_MASK) << 6) |
                        (strptr[i + 3] & UTF8_CONTINUATION_MASK);
                }

                if (codepoint <= UNICODE_MAX && !(codepoint >= SURROGATE_START && codepoint <= SURROGATE_END))
                {
                    if ((codepoint >= 'A' && codepoint <= 'Z') || (codepoint >= 'a' && codepoint <= 'z'))
                    {
                        ++count;
                    }
                }
            }

            i += byte_count;
        }

        return count;
    }



}

