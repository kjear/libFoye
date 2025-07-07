export module foye.fstring;

import foye.foye_core;
import foye.fstring_view;
import foye.farray;
import foye.fparse;
import foye.funicode_cvt;
import foye.string_algorithm;

import std;

#pragma warning(disable: 4309)
#pragma warning(disable: 4244)

export namespace fy
{
	class fstring
	{
	public:
        template<character_t Char_t, typename traits = std::char_traits<Char_t>, typename alloc = std::allocator<Char_t>>
        fstring(const std::basic_string<Char_t, traits, alloc>& stl_string) noexcept
            : fstring(stl_string.data(), stl_string.length()) { }

        template<character_t Char_t, typename traits = std::char_traits<Char_t>>
        fstring(const std::basic_string_view<Char_t, traits>& stl_string) noexcept
            : fstring(stl_string.data(), stl_string.length()) { }

        template<character_t Char_t, usize N>
        fstring(const Char_t(&string_literal)[N]) noexcept
            : fstring(&string_literal[0], N - 1) { }

        template<character_t Char_t>
        fstring(Char_t character) noexcept
            : fstring(&character, 1) { }

        template<character_t Char_t>
        fstring(const Char_t* const ptr, usize count_char_without_end_symbol) noexcept
        {
            string_convert_Invoker<Char_t, char8_t> cvtU8_invoker{};
            string_convert_Invoker<Char_t, char32_t> cvtU32_invoker{};

            if constexpr (std::is_same_v<Char_t, char> || std::is_same_v<Char_t, char8_t>)
            {
                if (count_char_without_end_symbol <= maximum_stack_capacity)
                {
                    std::memcpy(data_.stack.data, ptr, count_char_without_end_symbol * sizeof(char8_t));
                    data_.stack.used_length = count_char_without_end_symbol;
                }
                else
                {
                    data_.heap.data = new farray<char32_t>(count_char_without_end_symbol);
                    data_.heap.ref = new std::atomic<usize>(1);

                    const usize converted_length = cvtU32_invoker(ptr, count_char_without_end_symbol, data_.heap.data->data());
                    data_.heap.data->resize(converted_length);
                    data_.stack.used_length = -1;
                }
            }
            else
            {
                const usize need_u8 = fstring_view(ptr, count_char_without_end_symbol).need_capacity_byte_to_convert_to_UTF8__();
                if (need_u8 <= maximum_stack_capacity)
                {
                    string_convert_Invoker<Char_t, char8_t> cvtU8_invoker{};
                    data_.stack.used_length = cvtU8_invoker(ptr, count_char_without_end_symbol, data_.stack.data);
                }
                else
                {
                    if constexpr (std::is_same_v<Char_t, char32_t>)
                    {
                        data_.heap.data = new farray<char32_t>(ptr, count_char_without_end_symbol);
                    }
                    else
                    {
                        data_.heap.data = new farray<char32_t>(count_char_without_end_symbol);
                        const usize converted_length = cvtU32_invoker(ptr, count_char_without_end_symbol, data_.heap.data->data());
                        data_.heap.data->resize(converted_length);
                    }

                    data_.stack.used_length = -1;
                    data_.heap.ref = new std::atomic<usize>(1);
                }
            }
        }

        fstring(fstring_view str) noexcept
        {
            if (str.empty())
            {
                return;
            }

            fstring_view::dispatch_char_t(str.char_size_(),
                [this, str]<character_t Char_t>(Char_t) -> void
                {
                    string_convert_Invoker<Char_t, char8_t> cvtU8_invoker{};
                    string_convert_Invoker<Char_t, char32_t> cvtU32_invoker{};

                    const usize str_count_char = str.count_char_();

                    if constexpr (std::is_same_v<Char_t, char8_t>)
                    {
                        if (str_count_char > maximum_stack_capacity)
                            goto utf8_heap_path;

                        data_.stack.used_length = str_count_char;
                        std::memcpy(data_.stack.data, str.ptr_<char8_t>(), str_count_char * sizeof(char8_t));
                        return;

                    utf8_heap_path:
                        data_.heap.data = new farray<char32_t>(str.need_capacity_byte_to_convert_to_UTF32__());
                        data_.heap.ref = new std::atomic<usize>(1);

                        const usize converted_length = cvtU32_invoker(str.ptr_<Char_t>(), str_count_char, data_.heap.data->data());
                        data_.heap.data->resize(converted_length);
                        data_.stack.used_length = -1;
                        return;
                    }
                    else
                    {
                        const usize needs4cvt_UTF8 = str.need_capacity_byte_to_convert_to_UTF8__();
                        if (needs4cvt_UTF8 > maximum_stack_capacity)
                            goto non_utf8_heap_path;

                        data_.stack.used_length = cvtU8_invoker(str.ptr_<Char_t>(), str_count_char, data_.stack.data);
                        return;

                    non_utf8_heap_path:
                        if constexpr (std::is_same_v<Char_t, char32_t>)
                        {
                            data_.heap.data = new farray<char32_t>(str.ptr_<char32_t>(), str_count_char);
                        }
                        else
                        {
                            data_.heap.data = new farray<char32_t>(str.need_capacity_byte_to_convert_to_UTF32__());
                            const usize converted_length = cvtU32_invoker(str.ptr_<Char_t>(), str_count_char, data_.heap.data->data());
                            data_.heap.data->resize(converted_length);
                        }
                        data_.stack.used_length = -1;
                        data_.heap.ref = new std::atomic<usize>(1);
                    }
                }
            );
        }

        fstring(const fstring& other) noexcept
        {
            if (other.data_.stack.used_length >= 0)
            {
                if (other.empty())
                {
                    data_.stack.used_length = 0;
                }
                else
                {
                    data_.stack.used_length = other.data_.stack.used_length;
                    std::memcpy(data_.stack.data, other.data_.stack.data, other.data_.stack.used_length * sizeof(char8_t));
                }
            }
            else
            {
                data_.heap.data = other.data_.heap.data;
                data_.heap.ref = other.data_.heap.ref;
                data_.stack.used_length = -1;

                data_.heap.ref->fetch_add(1, std::memory_order_acq_rel);
            }
        }

        fstring& operator = (const fstring& other) noexcept
        {
            detach__();

            if (other.data_.stack.used_length >= 0)
            {
                if (other.empty())
                {
                    data_.stack.used_length = 0;
                }
                else
                {
                    data_.stack.used_length = other.data_.stack.used_length;
                    std::memcpy(data_.stack.data, other.data_.stack.data, other.data_.stack.used_length * sizeof(char8_t));
                }
            }
            else
            {
                other.data_.heap.ref->fetch_add(1, std::memory_order_acq_rel);
                data_.heap.data = other.data_.heap.data;
                data_.heap.ref = other.data_.heap.ref;
                data_.stack.used_length = -1;
            }

            return *this;
        }

        fstring(fstring&& other) noexcept
        {
            if (other.data_.stack.used_length >= 0)
            {
                if (other.empty())
                {
                    data_.stack.used_length = 0;
                }
                else
                {
                    data_.stack.used_length = other.data_.stack.used_length;
                    std::memcpy(data_.stack.data, other.data_.stack.data, other.data_.stack.used_length * sizeof(char8_t));
                }
            }
            else
            {
                data_.heap.data = other.data_.heap.data;
                data_.heap.ref = other.data_.heap.ref;
                data_.stack.used_length = -1;
                
                other.data_.heap.data = nullptr;
                other.data_.heap.ref = nullptr;
                other.data_.stack.used_length = 0;
            }
        }

        fstring& operator = (fstring&& other) noexcept
        {
            detach__();

            if (other.data_.stack.used_length >= 0)
            {
                if (other.empty())
                {
                    data_.stack.used_length = 0;
                }
                else
                {
                    data_.stack.used_length = other.data_.stack.used_length;
                    std::memcpy(data_.stack.data, other.data_.stack.data, other.data_.stack.used_length * sizeof(char8_t));
                }
            }
            else
            {
                data_.heap.data = other.data_.heap.data;
                data_.heap.ref = other.data_.heap.ref;
                data_.stack.used_length = -1;

                other.data_.heap.data = nullptr;
                other.data_.heap.ref = nullptr;
                other.data_.stack.used_length = 0;
            }

            return *this;
        }

        explicit operator bool() const noexcept
        {
            return !empty();
        }

        fstring() noexcept
        {
            data_.stack.used_length = 0;
            data_.heap.data = nullptr;
            data_.heap.ref = nullptr;
        }

        ~fstring() noexcept
        {
            release__();
        }

        template<typename First_t, typename ... Args>
        fstring& join(const First_t& first, Args&& ... args) requires (std::convertible_to<Args, fstring> && ...)
        {
            fstring result(first);
            ((result.append_inplace(*this).append_inplace(fstring(std::forward<Args>(args)))), ...);

            *this = std::move(result);
            return *this;
        }

        fstring_view subview(usize index_begin, usize str_size = 0) const noexcept
        {
            if (empty())
            {
                return fstring_view();
            }
            return as_view().substr(index_begin, str_size);
        }

        fstring substr(usize index_begin, usize str_size = 0) const noexcept
        {
            if (empty())
            {
                return fstring();
            }
            else if (index_begin == 0 && str_size == 0)
            {
                return fstring();
            }

            str_size = str_size == 0 ? size() - index_begin : str_size;
            if (is_using_stack__())
            {
                ssize index_begin_ = find_character_Char_t_begin(data_.stack.data, data_.stack.used_length, index_begin);
                ssize index_end_ = find_character_Char_t_begin(data_.stack.data, data_.stack.used_length, index_begin + str_size);

                fstring res;
                usize res_length = index_end_ - index_begin_;

                res.data_.stack.used_length = res_length;
                std::memcpy(res.data_.stack.data, data_.stack.data + index_begin_, res_length);
                return res;
            }
            else
            {
                return fstring(data_.heap.data->data() + index_begin, str_size);
            }
        }

        usize count(const fstring& to_repeat) const noexcept
        {
            if (empty())
            {
                return 0;
            }
            return find(to_repeat).size();
        }

        farray<fstring> split(const fstring& token) const noexcept
        {
            if (empty())
            {
                return farray<fstring>();
            }
            else if (token.empty())
            {
                return farray<fstring>{*this};
            }

            farray<usize> token_indices = find(token);

            farray<fstring> result;
            usize start = 0;

            for (usize index : token_indices)
            {
                result.emplace_back(substr(start, index - start));
                start = index + token.size();
            }

            if (start < size())
            {
                result.emplace_back(substr(start));
            }

            return result;
        }

        fstring remove(const fstring& to_remove) const noexcept
        {
            if (empty())
            {
                return fstring();
            }
            else if (to_remove.empty())
            {
                return copy();
            }

            fstring result;
            farray<fstring> sub_string = split(to_remove);

            for (fstring& str : sub_string)
            {
                result.append_inplace(str);
            }

            return result;
        }
        
        fstring& remove_inplace(const fstring& to_remove) noexcept
        {
            if (to_remove.empty())
            {
                return *this;
            }

            detach__();

            if (is_using_stack__())
            {
                farray<usize> sub_string;
                fstring temp;

                if (to_remove.is_using_stack__())
                {
                    temp = to_remove;
                }
                else [[unlikely]]
                {
                    string_convert_Invoker<char32_t, char8_t> Invoker{ };
                    temp.data_.stack.used_length = Invoker(to_remove.data_.heap.data->data(), to_remove.data_.heap.data->size(), temp.data_.stack.data);
                }

                sub_string = find(temp);
                if (sub_string.empty())
                {
                    return *this;
                }
                
                usize write_pos = 0;
                usize read_pos = 0;

                for (usize match_index : sub_string)
                {
                    usize match_byte_pos = 0;
                    for (usize i = 0; i < match_index; ++i)
                    {
                        match_byte_pos += parse_UTF8_bytecount(data_.stack.data[match_byte_pos]);
                    }

                    std::memmove(data_.stack.data + write_pos, data_.stack.data + read_pos, match_byte_pos - read_pos);
                    write_pos += match_byte_pos - read_pos;

                    read_pos = match_byte_pos + temp.data_.stack.used_length;
                }

                std::memmove(data_.stack.data + write_pos, data_.stack.data + read_pos, data_.stack.used_length - read_pos);
                write_pos += data_.stack.used_length - read_pos;

                data_.stack.used_length = write_pos;

            }
            else
            {
                fstring temp;

                if (to_remove.is_using_stack__())
                {
                    temp = to_remove.deepcopy();
                    temp.switch_to_heap__();
                }
                else
                {
                    temp = to_remove;
                }

                farray<usize> sub_string = find(temp);

                if (sub_string.empty())
                {
                    return *this;
                }

                usize persize = temp.size();
                usize removed_count = 0;
                for (usize i : sub_string)
                {
                    usize adjusted_index = i - removed_count;
                    data_.heap.data->pop_range(adjusted_index, persize);
                    removed_count += persize;
                }
            }

            return *this;
        }

        fstring trimmed(bool remove_internal_whitespace = false) const noexcept
        {
            if (empty())
            {
                return fstring();
            }
            fstring result = deepcopy();
            result.trimmed_inplace(remove_internal_whitespace);
            return result;
        }
        
        fstring& trimmed_inplace(bool remove_internal_whitespace = false) noexcept
        {
            if (empty())
            {
                return *this;
            }

            detach__();
            if (is_using_stack__())
            {
                data_.stack.used_length = trim(
                    data_.stack.data, data_.stack.used_length,
                    remove_internal_whitespace
                );
            }
            else
            {
                data_.heap.data->resize(
                    trim(
                        data_.heap.data->data(), data_.heap.data->size(),
                        remove_internal_whitespace
                    )
                );
            }

            return *this;
        }

        fstring& replace_inplace(const fstring& replace_from, const fstring& replace_to) noexcept
        {
            detach__();
            farray<usize> from = find(replace_from);

            for (usize v : from)
            {
                replace_inplace(replace_to, v, replace_to.size());
            }

            return *this;
        }

        fstring replace(const fstring& replace_from, const fstring& replace_to) const noexcept
        {
            farray<usize> from = find(replace_from);
            fstring result = deepcopy();

            for (usize v : from)
            {
                result.replace_inplace(replace_to, v, replace_to.size());
            }

            return result;
        }

        fstring replace(const fstring& to_replace, usize to_replace_index, usize to_replace_length = 0) const noexcept
        {
            if (to_replace.empty())
            {
                return empty() ? fstring() : copy();
            }

            to_replace_length = to_replace_length == 0 ? to_replace.size() : to_replace_length;
            fstring prefix = substr(0, to_replace_index);
            prefix.append_inplace(to_replace);
            prefix.append_inplace(substr(to_replace_index + to_replace_length));

            return prefix;
        }

        fstring& replace_inplace(const fstring& to_replace, usize to_replace_index, usize to_replace_length = 0) noexcept
        {
            detach__();

            to_replace_length = to_replace_length == 0 ? to_replace.size() : to_replace_length;

            if (is_using_stack__())
            {
                if (to_replace.is_using_stack__())
                {
                    const ssize index_begin_ = find_character_Char_t_begin(data_.stack.data, data_.stack.used_length, to_replace_index);
                    const ssize index_end_ = find_character_Char_t_begin(data_.stack.data, data_.stack.used_length, to_replace_index + to_replace_length);

                    const usize range_bytes = index_end_ - index_end_;
                    const usize total_bytes = (data_.stack.used_length - range_bytes) + to_replace.data_.stack.used_length;

                    if (total_bytes <= maximum_stack_capacity)
                    {
                        usize range_size = index_end_ - index_begin_;
                        if (to_replace.data_.stack.used_length != range_size)
                        {
                            std::memmove(data_.stack.data + index_begin_ + to_replace.data_.stack.used_length, data_.stack.data + index_end_, data_.stack.used_length - index_end_);
                        }

                        std::memcpy(data_.stack.data + index_begin_, to_replace.data_.stack.data, to_replace.data_.stack.used_length);
                        data_.stack.used_length = data_.stack.used_length - range_size + to_replace.data_.stack.used_length;
                    }
                    else
                    {
                        switch_to_heap__();

                        string_convert_Invoker<char8_t, char32_t> Invoker{};
                        char32_t temp_buffer[maximum_stack_capacity] = { };
                        
                        const usize conveted_length = Invoker(to_replace.data_.stack.data, to_replace.data_.stack.used_length, temp_buffer);
                        data_.heap.data->replace_range(temp_buffer, conveted_length, to_replace_index, to_replace_length);
                    }
                }
                else
                {
                    switch_to_heap__();
                    data_.heap.data->replace_range(*to_replace.data_.heap.data, to_replace_index, to_replace_length);
                }
            }
            else
            {
                if (to_replace.is_using_stack__())
                {
                    string_convert_Invoker<char8_t, char32_t> Invoker{};
                    char32_t temp_buffer[maximum_stack_capacity] = { };

                    const usize conveted_length = Invoker(to_replace.data_.stack.data, to_replace.data_.stack.used_length, temp_buffer);
                    data_.heap.data->replace_range(temp_buffer, conveted_length, to_replace_index, to_replace_length);
                }
                else
                {
                    data_.heap.data->replace_range(*to_replace.data_.heap.data, to_replace_index, to_replace_length);
                }
            }

            return *this;
        }

        fstring insert(const fstring& to_insert, usize to_insert_logical_character_index_begin) const noexcept
        {
            if (empty())
            {
                return to_insert.empty() ? fstring() : to_insert.copy();
            }
            else
            {
                if (to_insert.empty())
                {
                    return copy();
                }
            }

            if (to_insert_logical_character_index_begin > size())
            {
                to_insert_logical_character_index_begin = size();
            }

            fstring result;
            result = deepcopy();

            if (is_using_stack__())
            {
                usize total_bytes = data_.stack.used_length;
                if (to_insert.is_using_stack__())
                {
                    total_bytes += to_insert.data_.stack.used_length;
                    if (total_bytes <= maximum_stack_capacity)
                    {
                        usize byte_pos = 0;
                        usize char_count = 0;

                        while (char_count < to_insert_logical_character_index_begin && byte_pos < result.data_.stack.used_length)
                        {
                            byte_pos += parse_UTF8_bytecount(result.data_.stack.data[byte_pos]);
                            char_count++;
                        }

                        std::memmove(result.data_.stack.data + byte_pos + to_insert.data_.stack.used_length,
                            result.data_.stack.data + byte_pos,
                            result.data_.stack.used_length - byte_pos);

                        std::memcpy(result.data_.stack.data + byte_pos, to_insert.data_.stack.data, to_insert.data_.stack.used_length);
                        result.data_.stack.used_length = total_bytes;
                    }
                    else
                    {
                        result.switch_to_heap__();

                        string_convert_Invoker<char8_t, char32_t> Invoker{};
                        Invoker(to_insert.data_.stack.data, to_insert.data_.stack.used_length,
                            result.data_.heap.data->expand_at__(
                                to_insert_logical_character_index_begin,
                                to_insert.as_view().need_capacity_byte_to_convert_to_UTF32__() / sizeof(char32_t)
                            )
                        );
                    }
                }
                else
                {
                    result.switch_to_heap__();
                    result.data_.heap.data->insert(*to_insert.data_.heap.data,to_insert_logical_character_index_begin);
                }
            }
            else
            {
                if (to_insert.is_using_stack__())
                {
                    char32_t temp_buffer[maximum_stack_capacity] = { };
                    string_convert_Invoker<char8_t, char32_t> Invoker{};

                    result.data_.heap.data->insert(
                        temp_buffer, 
                        Invoker(to_insert.data_.stack.data, to_insert.data_.stack.used_length, temp_buffer),
                        to_insert_logical_character_index_begin);

                }
                else
                {
                    result.data_.heap.data->insert(*to_insert.data_.heap.data, to_insert_logical_character_index_begin);
                }
            }

            return result;
        }

        fstring& insert_inplace(const fstring& to_insert, usize to_insert_logical_character_index_begin) noexcept
        {
            detach__();

            if (is_using_stack__())
            {
                if (to_insert.is_using_stack__())
                {
                    const usize total_bytes = data_.stack.used_length + to_insert.data_.stack.used_length;
                    if (total_bytes <= maximum_stack_capacity)
                    {
                        usize byte_pos = 0;
                        usize char_count = 0;

                        while (char_count < to_insert_logical_character_index_begin && byte_pos < data_.stack.used_length)
                        {
                            byte_pos += parse_UTF8_bytecount(data_.stack.data[byte_pos]);
                            char_count++;
                        }

                        std::memmove(data_.stack.data + byte_pos + to_insert.data_.stack.used_length,
                            data_.stack.data + byte_pos,
                            data_.stack.used_length - byte_pos);

                        std::memcpy(data_.stack.data + byte_pos, to_insert.data_.stack.data, to_insert.data_.stack.used_length);
                        data_.stack.used_length = total_bytes;

                    }
                    else
                    {
                        switch_to_heap__();

                        string_convert_Invoker<char8_t, char32_t> Invoker{};
                        Invoker(to_insert.data_.stack.data, to_insert.data_.stack.used_length,
                            data_.heap.data->expand_at__(
                                to_insert_logical_character_index_begin,
                                to_insert.as_view().need_capacity_byte_to_convert_to_UTF32__() / sizeof(char32_t)
                            )
                        );

                    }
                }
                else
                {
                    switch_to_heap__();
                    data_.heap.data->insert(*to_insert.data_.heap.data, to_insert_logical_character_index_begin);
                }
            }
            else
            {
                if (to_insert.is_using_stack__())
                {
                    char32_t temp_buffer[maximum_stack_capacity] = { };
                    string_convert_Invoker<char8_t, char32_t> Invoker{};

                    data_.heap.data->insert(
                        temp_buffer,
                        Invoker(to_insert.data_.stack.data, to_insert.data_.stack.used_length, temp_buffer),
                        to_insert_logical_character_index_begin);
                }
                else
                {
                    data_.heap.data->insert(*to_insert.data_.heap.data, to_insert_logical_character_index_begin);
                }
            }

            return *this;
        }

        bool contains(const fstring& to_find) const noexcept
        {
            return find_first_of(to_find) > -1;
        }

        bool is_begin_with(const fstring& to_match) const noexcept
        {
            return substr(0, to_match.size()).is_consistent_with(to_match);
        }

        bool is_end_with(const fstring& to_match) const noexcept
        {
            return substr(size() - to_match.size()).is_consistent_with(to_match);
        }

        ssize find_first_of(const fstring& to_find) const noexcept
        {
            if (to_find.empty() || empty() || (to_find.size() > size()))
            {
                return -1;
            }
            else if (to_find.size() == size())
            {
                if (to_find.is_consistent_with(*this))
                {
                    return 0;
                }

                return -1;
            }

            if (!is_using_stack__() && size() >= enable_boyer_moore_string_match_length_threshold)
            {
                if (to_find.is_using_stack__())
                {
                    char32_t temp_buffer[maximum_stack_capacity] = { };
                    string_convert_Invoker<char8_t, char32_t> Invoker{};

                    farray_view<char32_t> to_find_view(temp_buffer, temp_buffer + Invoker(to_find.data_.stack.data, to_find.data_.stack.used_length, temp_buffer));
                    farray_view<char32_t> src_view(data_.heap.data->data(), data_.heap.data->data() + data_.heap.data->size());
                    boyer_moore_string_match_Invoker<char32_t> match_invoker(to_find_view);

                    return match_invoker.match_first(src_view);
                }
                else
                {
                    farray_view<char32_t> to_find_view(to_find.data_.heap.data->data(), to_find.data_.heap.data->data() + to_find.data_.heap.data->size());
                    farray_view<char32_t> src_view(data_.heap.data->data(), data_.heap.data->data() + data_.heap.data->size());
                    boyer_moore_string_match_Invoker<char32_t> match_invoker(to_find_view);

                    return match_invoker.match_first(src_view);
                }
            }

            return basic_string_match_impl__<true>(to_find);
        }

        ssize find_last_of(const fstring& to_find) const noexcept
        {
            farray<usize> all = find(to_find);
            return all.empty() ? -1 : all.back();
        }

        farray<usize> find(const fstring& to_find) const noexcept
        {
            if (to_find.empty() || empty() || (to_find.size() > size()))
            {
                return farray<usize>();
            }
            else if (to_find.size() == size())
            {
                if (to_find.is_consistent_with(*this))
                {
                    return farray<usize>(1, 0);
                }

                return farray<usize>();
            }

            if (!is_using_stack__() && size() >= enable_boyer_moore_string_match_length_threshold)
            {
                if (to_find.is_using_stack__())
                {
                    char32_t temp_buffer[maximum_stack_capacity] = { };
                    string_convert_Invoker<char8_t, char32_t> Invoker{};

                    farray_view<char32_t> to_find_view(temp_buffer, temp_buffer + Invoker(to_find.data_.stack.data, to_find.data_.stack.used_length, temp_buffer));
                    farray_view<char32_t> src_view(data_.heap.data->data(), data_.heap.data->data() + data_.heap.data->size());
                    boyer_moore_string_match_Invoker<char32_t> match_invoker(to_find_view);

                    return match_invoker.match_all(src_view);
                }
                else
                {
                    farray_view<char32_t> to_find_view(to_find.data_.heap.data->data(), to_find.data_.heap.data->data() + to_find.data_.heap.data->size());
                    farray_view<char32_t> src_view(data_.heap.data->data(), data_.heap.data->data() + data_.heap.data->size());
                    boyer_moore_string_match_Invoker<char32_t> match_invoker(to_find_view);

                    return match_invoker.match_all(src_view);
                }
            }

            return basic_string_match_impl__<false>(to_find);
        }

        fstring copy() const noexcept
        {
            return fstring(*this);
        }
        
        fstring deepcopy() const noexcept
        {
            fstring result = this->copy();
            if (!result.is_using_stack__())
            {
                result.detach__();
            }
            return result;
        }

        fstring prepend(const fstring& prefix) const noexcept
        {
            if (empty())
            {
                if (prefix.empty())
                {
                    return fstring();
                }
                else
                {
                    return prefix.copy();
                }
            }
            else
            {
                if (prefix.empty())
                {
                    return copy();
                }
            }

            if (is_using_stack__())
            {
                fstring result = prefix.deepcopy();

                if (result.is_using_stack__())
                {
                    usize total_byte = result.data_.stack.used_length + data_.stack.used_length;

                    if (total_byte <= maximum_stack_capacity)
                    {
                        std::memcpy(result.data_.stack.data + result.data_.stack.used_length, data_.stack.data, data_.stack.used_length);
                        result.data_.stack.used_length += data_.stack.used_length;
                        return result;
                    }
                    else
                    {
                        result.switch_to_heap__();
                    }
                }

                char32_t temp_buffer[maximum_stack_capacity] = { };
                string_convert_Invoker<char8_t, char32_t> Invoker{ };
                usize converted_length = Invoker(data_.stack.data, data_.stack.used_length, temp_buffer);
                result.data_.heap.data->push_back(temp_buffer, converted_length);

                return result;
            }
            else
            {
                fstring result = deepcopy();

                if (prefix.is_using_stack__())
                {
                    char32_t temp_buffer[maximum_stack_capacity] = { };
                    string_convert_Invoker<char8_t, char32_t> Invoker{ };
                    usize converted_length = Invoker(prefix.data_.stack.data, prefix.data_.stack.used_length, temp_buffer);
                    result.data_.heap.data->push_front(temp_buffer, converted_length);
                }
                else
                {
                    result.data_.heap.data->push_front(*prefix.data_.heap.data);
                }

                return result;
            }
        }

        fstring& prepend_inplace(const fstring& prefix) noexcept
        {
            if (prefix.empty())
            {
                return *this;
            }
            else if (empty() && !prefix.empty())
            {
                *this = prefix.copy();
                return *this;
            }

            char32_t temp_buffer[maximum_stack_capacity] = { };
            string_convert_Invoker<char8_t, char32_t> Invoker{};

            detach__();

            if (is_using_stack__())
            {
                if (prefix.is_using_stack__())
                {
                    usize count_bytes = data_.stack.used_length + prefix.data_.stack.used_length;
                    if (count_bytes <= maximum_stack_capacity)
                    {
                        std::memmove(data_.stack.data + prefix.data_.stack.used_length, data_.stack.data, data_.stack.used_length);
                        std::memcpy(data_.stack.data, prefix.data_.stack.data, prefix.data_.stack.used_length);
                        data_.stack.used_length += prefix.data_.stack.used_length;
                    }
                    else
                    {
                        switch_to_heap__();

                        data_.heap.data->push_front(
                            temp_buffer, 
                            Invoker(prefix.data_.stack.data, prefix.data_.stack.used_length, temp_buffer)
                        );
                    }
                }
                else
                {
                    switch_to_heap__();
                    data_.heap.data->push_front(*prefix.data_.heap.data);
                }
            }
            else
            {
                if (prefix.is_using_stack__())
                {
                    data_.heap.data->push_front(temp_buffer, 
                        Invoker(prefix.data_.stack.data, prefix.data_.stack.used_length, temp_buffer)
                    );
                }
                else
                {
                    data_.heap.data->push_front(*prefix.data_.heap.data);
                }
            }

            return *this;
        }

        fstring& append_inplace(const fstring& suffix) noexcept
        {
            if (suffix.empty())
            {
                return *this;
            }
            else if (empty() && !suffix.empty())
            {
                *this = suffix.copy();
                return *this;
            }

            char32_t temp_buffer[maximum_stack_capacity] = { };
            string_convert_Invoker<char8_t, char32_t> Invoker{};

            detach__();

            if (is_using_stack__())
            {
                if (suffix.is_using_stack__())
                {
                    usize count_bytes = data_.stack.used_length + suffix.data_.stack.used_length;
                    if (count_bytes <= maximum_stack_capacity)
                    {
                        std::memcpy(data_.stack.data + data_.stack.used_length, suffix.data_.stack.data, suffix.data_.stack.used_length);
                        data_.stack.used_length += suffix.data_.stack.used_length;
                    }
                    else
                    {
                        switch_to_heap__();

                        const usize converted_length = Invoker(suffix.data_.stack.data, suffix.data_.stack.used_length, temp_buffer);
                        data_.heap.data->push_back(temp_buffer, converted_length);
                    }
                }
                else
                {
                    switch_to_heap__();
                    data_.heap.data->push_back(*suffix.data_.heap.data);
                }
            }
            else
            {
                if (suffix.is_using_stack__())
                {
                    data_.heap.data->push_back(temp_buffer,
                        Invoker(suffix.data_.stack.data, suffix.data_.stack.used_length, temp_buffer)
                    );

                }
                else
                {
                    data_.heap.data->push_back(*suffix.data_.heap.data);
                }
            }

            return *this;
        }

        fstring append(const fstring& suffix) const noexcept
        {
            if (empty())
            {
                if (suffix.empty())
                {
                    return fstring();
                }
                else
                {
                    return suffix.copy();
                }
            }
            else
            {
                if (suffix.empty())
                {
                    return copy();
                }
            }

            if (is_using_stack__())
            {
                fstring result = deepcopy();

                if (suffix.is_using_stack__())
                {
                    const usize total_byte = data_.stack.used_length + suffix.data_.stack.used_length;
                    if (total_byte <= maximum_stack_capacity)
                    {
                        std::memcpy(
                            result.data_.stack.data + result.data_.stack.used_length, 
                            suffix.data_.stack.data, data_.stack.used_length);

                        result.data_.stack.used_length += data_.stack.used_length;
                        return result;
                    }
                    else
                    {
                        result.switch_to_heap__();
                    }
                }

                char32_t temp_buffer[maximum_stack_capacity] = { };
                string_convert_Invoker<char8_t, char32_t> Invoker{};
                usize converted_length = Invoker(suffix.data_.stack.data, suffix.data_.stack.used_length, temp_buffer);
                result.data_.heap.data->push_back(temp_buffer, converted_length);

                return result;
            }
            else
            {
                fstring result = deepcopy();

                if (suffix.is_using_stack__())
                {
                    char32_t temp_buffer[maximum_stack_capacity] = { };
                    string_convert_Invoker<char8_t, char32_t> Invoker{ };
                    usize converted_length = Invoker(suffix.data_.stack.data, suffix.data_.stack.used_length, temp_buffer);
                    result.data_.heap.data->push_back(temp_buffer, converted_length);
                }
                else
                {
                    result.data_.heap.data->push_back(*suffix.data_.heap.data);
                }

                return result;
            }
        }

        bool is_consistent_with(const fstring& other) const noexcept
        {
            if (empty())
            {
                return other.empty();
            }

            if (size() != other.size())
            {
                return false;
            }

            if (is_using_stack__())
            {
                if (other.is_using_stack__())
                {
                    return std::memcmp(data_.stack.data, other.data_.stack.data, data_.stack.used_length) == 0;
                }
                else
                {
                    return string_equal_Invoker<char8_t, char32_t>(
                        data_.stack.data,
                        other.data_.heap.data->data(),
                        data_.stack.used_length,
                        other.data_.heap.data->size()
                    );
                }
            }
            else
            {
                if (other.is_using_stack__())
                {
                    return string_equal_Invoker<char32_t, char8_t>(
                        data_.heap.data->data(),
                        other.data_.stack.data,
                        data_.heap.data->size(),
                        other.data_.stack.used_length
                    );
                }
                else
                {
                    return std::memcmp(data_.heap.data->data(), other.data_.heap.data->data(), data_.heap.data->size() * sizeof(char32_t)) == 0;
                }
            }
        }

        usize size() const noexcept
        {
            return data_.stack.used_length >= 0 ?
                calc_complete_characters<char8_t>(data_.stack.data, data_.stack.used_length) :
                data_.heap.data ? data_.heap.data->size() : 0;
        }

        bool empty() const noexcept
        {
            return count_char__() == 0;
        }

        fstring to_lower() const noexcept
        {
            fstring res = deepcopy();
            res.to_lower_inplace();
            return res;
        }

        fstring& to_lower_inplace() noexcept
        {
            if (!empty())
            {
                if (is_using_stack__())
                {
                    to_lowercase(data_.stack.data, data_.stack.used_length);
                }
                else
                {
                    to_lowercase(data_.heap.data->data(), data_.heap.data->size());
                }
            }

            return *this;
        }

        fstring to_upper() const noexcept
        {
            fstring res = deepcopy();
            res.to_upper_inplace();
            return res;
        }

        fstring& to_upper_inplace() noexcept
        {
            if (!empty())
            {
                if (is_using_stack__())
                {
                    to_uppercase(data_.stack.data, data_.stack.used_length);
                }
                else
                {
                    to_uppercase(data_.heap.data->data(), data_.heap.data->size());
                }
            }

            return *this;
        }

        template<bool writeable> class fstring_cell_proxy__
        {
        public:
            operator fstring_view() const noexcept 
            {
                return parasitifer.subview(logical_character_index_offset, 1);
            }

            operator fstring() const noexcept
            {
                return parasitifer.substr(logical_character_index_offset, 1);
            }

            fstring_cell_proxy__& operator = (fstring_view right) noexcept requires(writeable)
            {
                parasitifer.replace_inplace(fstring(right), logical_character_index_offset, 1);
                return *this;
            }

        private:
            friend class fstring;
            fstring_cell_proxy__(std::conditional_t<writeable, fstring&, const fstring&> input_parasitifer, usize index) 
                noexcept : parasitifer(input_parasitifer), logical_character_index_offset(index) 
            {
                if constexpr (writeable) parasitifer.detach__();
            }

            usize logical_character_index_offset;
            std::conditional_t<writeable, fstring&, const fstring&> parasitifer;
        };

        using writeable_fstring_cell_proxy = fstring_cell_proxy__<true>;
        using readonly_fstring_cell_proxy = fstring_cell_proxy__<false>;

        readonly_fstring_cell_proxy operator [] (usize index) const noexcept
        {
            return readonly_fstring_cell_proxy(*this, index);
        }

        writeable_fstring_cell_proxy operator [] (usize index) noexcept
        {
            return writeable_fstring_cell_proxy(*this, index);
        }

        fstring_view as_view() const noexcept
        {
            return data_.stack.used_length >= 0 ?
                fstring_view(data_.stack.data, data_.stack.used_length) :
                data_.heap.data ? fstring_view(data_.heap.data->data(), data_.heap.data->size()) : fstring_view();
        }

        operator fstring_view() const noexcept
        {
            return as_view();
        }

        fbytearray encode_as_UTF8() const noexcept
        {
            return as_view().encode_as_UTF8();
        }

        fbytearray encode_as_UTF16() const noexcept
        {
            return as_view().encode_as_UTF16();
        }

        fbytearray encode_as_UTF32() const noexcept
        {
            return as_view().encode_as_UTF32();
        }

        template<BasicArithmetic numeric_t>
        explicit operator numeric_t() const noexcept
        {
            if constexpr (Floating_arithmetic<numeric_t>)
            {
                return is_using_stack__() ?
                    static_cast<numeric_t>(
                        string_to_numeric_Invoker<char8_t>::as_floating_point(data_.stack.data, data_.stack.used_length)) :
                    static_cast<numeric_t>(
                        string_to_numeric_Invoker<char32_t>::as_floating_point(data_.heap.data->data(), data_.heap.data->size()));
            }
            else if constexpr (integral_arithmetic<numeric_t>)
            {
                if constexpr (std::is_unsigned_v<numeric_t>)
                {
                    return is_using_stack__() ?
                        static_cast<numeric_t>(
                            string_to_numeric_Invoker<char8_t>::as_unsigned_integer(data_.stack.data, data_.stack.used_length)) :
                        static_cast<numeric_t>(
                            string_to_numeric_Invoker<char32_t>::as_unsigned_integer(data_.heap.data->data(), data_.heap.data->size()));
                }
                else
                {
                    return is_using_stack__() ?
                        static_cast<numeric_t>(
                            string_to_numeric_Invoker<char8_t>::as_signed_integer(data_.stack.data, data_.stack.used_length)) :
                        static_cast<numeric_t>(
                            string_to_numeric_Invoker<char32_t>::as_signed_integer(data_.heap.data->data(), data_.heap.data->size()));
                }
            }
            else
            {
                std::unreachable();
            }
        }

        template<character_t Char_t = char, typename Traits = std::char_traits<Char_t>, typename Alloc = std::allocator<Char_t>>
        std::basic_string<Char_t, Traits, Alloc> stl_string() const noexcept
        {
            using stl_string = std::basic_string<Char_t, Traits, Alloc>;

            if constexpr (std::is_same_v<Char_t, char8_t> || std::is_same_v<Char_t, char>)
            {
                if (is_using_stack__())
                {
                    return stl_string(reinterpret_cast<const Char_t*>(data_.stack.data), data_.stack.used_length);
                }
                else
                {
                    fbytearray temp_utf8 = encode_as_UTF8();
                    return stl_string(temp_utf8.data<Char_t>(), temp_utf8.size());
                }
            }
            else if constexpr (std::is_same_v<Char_t, char16_t> || std::is_same_v<Char_t, wchar_t>)
            {
                fbytearray temp_utf16 = encode_as_UTF16();
                return stl_string(temp_utf16.data<Char_t>(), temp_utf16.size() / sizeof(Char_t));
            }
            else if constexpr (std::is_same_v<Char_t, char32_t>)
            {
                if (!is_using_stack__())
                {
                    return stl_string(data_.heap.data->data(), data_.heap.data->size());
                }
                else
                {
                    fbytearray temp_utf32 = encode_as_UTF32();
                    return stl_string(temp_utf32.data<char32_t>(), temp_utf32.size() / sizeof(char32_t));
                }
            }
            else
            {
                std::unreachable();
            }
        }

    private:
        bool is_using_stack__() const noexcept
        {
            return data_.stack.used_length >= 0;
        }

        bool is_sharing__() const noexcept
        {
            return (!is_using_stack__()) && (data_.heap.ref && data_.heap.ref->load(std::memory_order_seq_cst) > 1);
        }

        usize count_char__() const noexcept
        {
            return data_.stack.used_length >= 0 ?
                data_.stack.used_length :
                data_.heap.data ? data_.heap.data->size() : 0;
        }

        void detach__() noexcept
        {
            if (data_.stack.used_length < 0)
            {
                if (data_.heap.ref && data_.heap.ref->load() > 1 && data_.heap.data)
                {
                    farray<char32_t>* new_data = new farray<char32_t>(data_.heap.data->data(), data_.heap.data->size());
                    std::atomic<usize>* new_ref = new std::atomic<usize>(1);

                    if (data_.heap.ref->fetch_sub(1, std::memory_order_seq_cst) == 1)
                    {
                        delete data_.heap.ref;
                        delete data_.heap.data;
                    }

                    data_.heap.data = new_data;
                    data_.heap.ref = new_ref;
                }
            }
        }

        void clear__() noexcept
        {
            if (data_.stack.used_length >= 0)
            {
                data_.stack.used_length = 0;
            }
            else
            {
                data_.heap.data->clear();
            }
        }
        
        void release__() noexcept
        {
            if (data_.stack.used_length < 0)
            {
                if (data_.heap.ref && data_.heap.ref->fetch_sub(1, std::memory_order_seq_cst) == 1)
                {
                    if (data_.heap.data) delete data_.heap.data;
                    delete data_.heap.ref;

                    data_.heap.data = nullptr;
                    data_.heap.ref = nullptr;
                }
            }
        }

        void switch_to_heap__() noexcept
        {
            if (data_.stack.used_length >= 0)
            {
                farray<char32_t>* new_heap = new farray<char32_t>(maximum_stack_capacity * 4);
                std::atomic<usize>* new_ref = new std::atomic<usize>(1);

                string_convert_Invoker<char8_t, char32_t> invoker{};
                const usize converted_length = invoker(data_.stack.data, data_.stack.used_length, new_heap->data());
                new_heap->resize(converted_length);
                
                data_.heap.data = new_heap;
                data_.heap.ref = new_ref;

                data_.stack.used_length = -1;
            }
        }

        template<bool only_match_first>
        typename std::conditional_t<only_match_first, ssize, farray<usize>> basic_string_match_impl__(const fstring& to_find) const noexcept
        {
            if (is_using_stack__())
            {
                farray_view<char8_t> self_view(data_.stack.data, data_.stack.data + data_.stack.used_length);
                if (to_find.is_using_stack__())
                {
                    brute_force_matching_Invoker<char8_t, char8_t> Invoker(self_view,
                        farray_view<char8_t>(to_find.data_.stack.data, to_find.data_.stack.data + to_find.data_.stack.used_length));
                    return Invoker.index_of_logical_character_offset<only_match_first>();
                }
                else [[unlikely]]
                    {
                        fbytearray temp = to_find.encode_as_UTF8();

                        brute_force_matching_Invoker<char8_t, char8_t> Invoker(self_view,
                            farray_view<char8_t>(temp.data<char8_t>(), temp.data<char8_t>() + temp.size()));
                        return Invoker.index_of_logical_character_offset<only_match_first>();
                    }
            }
            else
            {
                const usize self_length = count_char__();
                farray_view<char32_t> self_view(data_.heap.data->data(), data_.heap.data->data() + data_.heap.data->size());
                if (to_find.is_using_stack__())
                {
                    fbytearray temp = to_find.encode_as_UTF32();
                    farray_view<char32_t> to_find_view(farray_view<char32_t>(temp.data<char32_t>(), temp.data<char32_t>() + (temp.size() / sizeof(char32_t))));
                    brute_force_matching_Invoker<char32_t, char32_t> Invoker(self_view, to_find_view);

                    return Invoker.index_of_logical_character_offset<only_match_first>();
                }
                else
                {
                    farray_view<char32_t> to_find_view(to_find.data_.heap.data->data(), to_find.data_.heap.data->data() + to_find.data_.heap.data->size());
                    brute_force_matching_Invoker<char32_t, char32_t> Invoker(self_view, to_find_view);

                    return Invoker.index_of_logical_character_offset<only_match_first>();
                }
            }
        }

    protected:
		static inline constexpr usize maximum_stack_capacity = 31;
        static inline constexpr usize enable_boyer_moore_string_match_length_threshold = 100;

		union
		{
			struct
			{
				alignas(32) char8_t data[maximum_stack_capacity];
				i8 used_length;
			} stack;

			struct
			{
				farray<char32_t>* data;
				std::atomic<usize>* ref;
			} heap;
		} data_;
	};
}

export namespace fy
{
    template<integral_arithmetic Numeric_t>
    fstring to_fstring(Numeric_t numeric)
    {
        integral_to_chararray_Invoker<char8_t> invoker;
        usize length = invoker(numeric);

        char8_t temp[20] = {};
        usize start_index = 20 - length;
        for (usize i = 0; i < length; ++i)
        {
            temp[i] = invoker.data[start_index + i];
        }

        return fstring(temp, length);
    }

    template<Floating_arithmetic Numeric_t>
    fstring to_fstring(Numeric_t numeric)
    {
        floating_point_to_chararray_Invoker<char8_t> invoker{};
        return fstring(invoker.buffer, invoker(numeric));
    }

    fstring to_fstring(bool boolalpha)
    {
        return boolalpha ? fstring("true") : fstring("false");
    }

    fstring& operator << (fstring& mstr, fstring_view suffix)
    {
        mstr.append_inplace(fstring(suffix));
        return mstr;
    }

    fstring& operator >> (fstring_view lhs, fstring& rhs)
    {
        rhs.prepend_inplace(fstring(lhs));
        return rhs;
    }

    template<BasicArithmetic numeric_t>
    fstring& operator << (fstring& mstr, numeric_t str)
    {
        mstr.append_inplace(to_fstring(str));
        return mstr;
    }

    template<BasicArithmetic numeric_t>
    fstring& operator >> (numeric_t lhs, fstring& rhs)
    {
        rhs.prepend_inplace(to_fstring(lhs));
        return rhs;
    }

    bool operator == (const fstring& mstr, const fstring& other)
    {
        return mstr.is_consistent_with(other);
    }

    bool operator != (const fstring& mstr, const fstring& other)
    {
        return !mstr.is_consistent_with(other);
    }

    fstring operator + (const fstring& mstr, const fstring& suffix)
    {
        return mstr.append(suffix);
    }

    fstring& operator += (fstring& mstr, const fstring& suffix)
    {
        return mstr.append_inplace(suffix);
    }

    template<character_t Char_t, typename traits = std::char_traits<Char_t>>
    std::basic_ostream<Char_t, traits>& operator << (std::basic_ostream<Char_t, traits>& os, const fstring& str)
    {
        os << str.stl_string<Char_t>();
        return os;
    }
}