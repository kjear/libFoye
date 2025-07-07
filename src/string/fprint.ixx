module;
#include <Windows.h>
#undef min
#undef max

export module foye.fprint;

export import foye.foye_core;
export import foye.fstring_view;
export import foye.farray;
export import foye.fbytearray;
export import foye.funicode_cvt;
export import foye.fparse;
export import foye.extensionType.int128;
export import foye.extensionType.float16;
export import foye.extensionType.bfloat16;
import std;

#pragma warning(disable: 4309)
#pragma warning(disable: 4244)
#pragma warning(disable: 4552)
#pragma warning(disable: 4552)
#pragma warning(disable: 4018)

export namespace fy
{
    struct fTerminal_print_Invoker final
    {
    public:
        enum class stream_type : i32
        {
            Error,
            Output
        };

        fTerminal_print_Invoker(stream_type h = stream_type::Error)
            : capacity(0),
            type(h),
            hConsole(GetStdHandle(h == stream_type::Error ? STD_ERROR_HANDLE : STD_OUTPUT_HANDLE))
        {
            std::memset(buffer, 0, sizeof(char8_t) * BUFFER_SIZE);
            static bool init_UTF8 = false;
            if (!init_UTF8)
            {
                if (!SetConsoleOutputCP(CP_UTF8))
                {
                    throw std::runtime_error("print error: Failed to set UTF-8 console output");
                }
                init_UTF8 = true;
            }

            if (hConsole == INVALID_HANDLE_VALUE)
            {
                throw std::runtime_error("print error: Failed to get std output handle");
            }
        }

        void reset_buffer() noexcept
        {
            capacity = 0;
        }

        void flush_buffer() noexcept
        {
            if (capacity > 0 && type == stream_type::Output)
            {
                DWORD written;
                if (WriteFile(hConsole, buffer, static_cast<DWORD>(capacity), &written, nullptr))
                {
                    FlushFileBuffers(hConsole);
                    reset_buffer();
                }
            }
        }

        bool write_in(const char8_t* const buffer_ptr, usize count_char_without_end_symbol)
        {
            if (!buffer_ptr || count_char_without_end_symbol == 0)
            {
                return false;
            }

            if (type == stream_type::Error)
            {
                DWORD written;
                return WriteFile(hConsole, buffer_ptr,
                    static_cast<DWORD>(count_char_without_end_symbol), &written, nullptr) != 0;
            }

            if (capacity + count_char_without_end_symbol > BUFFER_SIZE)
            {
                flush_buffer();

                if (count_char_without_end_symbol > BUFFER_SIZE)
                {
                    DWORD written;
                    return WriteFile(hConsole, buffer_ptr,
                        static_cast<DWORD>(count_char_without_end_symbol), &written, nullptr) != 0;
                }
            }

            std::memcpy(buffer + capacity, buffer_ptr, count_char_without_end_symbol);
            capacity += count_char_without_end_symbol;
            return true;
        }

        bool write_in(const char16_t* const buffer_ptr, usize count_char_without_end_symbol)
        {
            if (!buffer_ptr || count_char_without_end_symbol == 0)
            {
                return false;
            }

            if (type == stream_type::Error)
            {
                DWORD written;
                return WriteConsoleW(hConsole,
                    buffer_ptr,
                    static_cast<DWORD>(count_char_without_end_symbol),
                    &written,
                    nullptr) != 0;
            }

            flush_buffer();

            DWORD written;
            return WriteConsoleW(hConsole,
                buffer_ptr,
                static_cast<DWORD>(count_char_without_end_symbol),
                &written,
                nullptr) != 0;
        }

        fTerminal_print_Invoker(const fTerminal_print_Invoker&) = delete;
        fTerminal_print_Invoker(fTerminal_print_Invoker&&) = delete;

        fTerminal_print_Invoker& operator = (const fTerminal_print_Invoker&) = delete;
        fTerminal_print_Invoker& operator = (fTerminal_print_Invoker&&) = delete;

        HANDLE hConsole;
        static constexpr usize BUFFER_SIZE = 256;

        char8_t buffer[BUFFER_SIZE];
        usize capacity;
        stream_type type;
    };
}

namespace fy
{
    static fTerminal_print_Invoker global_Terminal_Output_Invoker(fTerminal_print_Invoker::stream_type::Output);
    static fTerminal_print_Invoker global_Terminal_Error_Invoker(fTerminal_print_Invoker::stream_type::Error);

    void basic_print(fstring_view sv, fTerminal_print_Invoker::stream_type out_type)
    {
        if (sv.empty())
        {
            return;
        }

        fstring_view::dispatch_char_t(sv.char_size_(),
            [&]<character_t Char_t>(Char_t) -> void
            {
                const Char_t* const data = sv.ptr_<Char_t>();

                if constexpr (std::_Is_any_of_v<Char_t, char8_t, char16_t>)
                {
                    (out_type == fTerminal_print_Invoker::stream_type::Output ?
                        global_Terminal_Output_Invoker : global_Terminal_Error_Invoker)
                        .write_in(data, sv.count_char_());
                }
                else
                {
                    char16_t temp_buffer[2] = {};
                    string_convert_Invoker<char32_t, char16_t> invoker;

                    for (usize i = 0; i < sv.size(); ++i)
                    {
                        fstring_view sub = sv.substr(i, 1);
                        (out_type == fTerminal_print_Invoker::stream_type::Output ?
                            global_Terminal_Output_Invoker : global_Terminal_Error_Invoker)
                            .write_in(
                                temp_buffer,
                                invoker(sub.ptr_<Char_t>(), 1, temp_buffer)
                            );
                    }
                }
            }
        );

        global_Terminal_Output_Invoker.flush_buffer();
    }

    void basic_print(bool boolalpha, fTerminal_print_Invoker::stream_type out_type)
    {
        static constexpr fstring_view true_string_view__ = u8"true";
        static constexpr fstring_view false_string_view__ = u8"false";
        return basic_print(boolalpha ? true_string_view__ : false_string_view__, out_type);
    }

    template<character_t Char_t, usize N>
    void basic_print(const Char_t(&string_literal)[N], fTerminal_print_Invoker::stream_type out_type)
    {
        return basic_print(fstring_view(string_literal, N - 1), out_type);
    }

    template<character_t Char_t>
    void basic_print(Char_t chr, fTerminal_print_Invoker::stream_type out_type)
    {
        const Char_t data[1] = { chr };
        return basic_print(fstring_view(data, 1), out_type);
    }

    template<Floating_arithmetic T> requires(!std::is_same_v<T, f16>)
    void basic_print(T floating, fTerminal_print_Invoker::stream_type out_type)
    {
        floating_point_to_chararray_Invoker<char16_t> invoker{};
        basic_print(fstring_view(invoker.buffer, invoker(floating)), out_type);
    }

    void basic_print(int128 i128v, fTerminal_print_Invoker::stream_type out_type)
    {
        integral128_to_chararray_Invoker<char16_t> invoker;
        basic_print(fstring_view(invoker.buffer, invoker(i128v)), out_type);
    }

    void basic_print(uint128 u128v, fTerminal_print_Invoker::stream_type out_type)
    {
        integral128_to_chararray_Invoker<char16_t> invoker;
        basic_print(fstring_view(invoker.buffer, invoker(u128v)), out_type);
    }

    void basic_print(f16 floating, fTerminal_print_Invoker::stream_type out_type)
    {
        f32 temp_val = static_cast<f32>(floating);
        floating_point_to_chararray_Invoker<char16_t> invoker{};
        basic_print(fstring_view(invoker.buffer, invoker(temp_val)), out_type);
    }

    void basic_print(bf16 floating, fTerminal_print_Invoker::stream_type out_type)
    {
        f32 temp_val = static_cast<f32>(floating);
        floating_point_to_chararray_Invoker<char16_t> invoker{};
        basic_print(fstring_view(invoker.buffer, invoker(temp_val)), out_type);
    }

    template<integral_arithmetic T>
    void basic_print(T integral, fTerminal_print_Invoker::stream_type out_type)
    {
        integral_to_chararray_Invoker<char16_t> invoker;
        usize length = invoker(integral);

        char16_t temp[20] = {};
        usize start_index = 20 - length;
        for (usize i = 0; i < length; ++i)
        {
            temp[i] = invoker.data[start_index + i];
        }
        return basic_print(fstring_view(temp, length), out_type);
    }

    template<typename Type>
    inline constexpr bool has_basic_print_overload_v = requires(Type t, fTerminal_print_Invoker::stream_type o)
    {
        basic_print(t, o);
    };

    template<character_t Char_t, typename traits = std::char_traits<Char_t>, typename alloc = std::allocator<Char_t>>
    void basic_print(const std::basic_string<Char_t, traits, alloc>& stl_string, fTerminal_print_Invoker::stream_type out_type)
    {
        basic_print(fstring_view(stl_string.data(), stl_string.length()), out_type);
    }

    template<character_t Char_t, typename traits = std::char_traits<Char_t>>
    void basic_print(const std::basic_string_view<Char_t, traits>& stl_stringview, fTerminal_print_Invoker::stream_type out_type)
    {
        basic_print(fstring_view(stl_stringview.data(), stl_stringview.length()), out_type);
    }

    template<typename Element_t>
    void basic_print(const std::vector<Element_t>& farr, fTerminal_print_Invoker::stream_type out_type)
    {
        static_assert(has_basic_print_overload_v<Element_t>, "Unprintable data types");

        const usize size = farr.size();

        basic_print("[", out_type);
        for (usize i = 0; i < size; ++i)
        {
            if constexpr (character_t<Element_t>)
            {
                basic_print(static_cast<i32>(farr[i]), out_type);
            }
            else
            {
                basic_print(farr[i], out_type);
            }
            if (i + 1 < size)
            {
                basic_print(", ", out_type);
            }
        }
        basic_print("]", out_type);
    }

    template<typename Element_t>
    void basic_print(const fvalarray<Element_t>& farr, fTerminal_print_Invoker::stream_type out_type)
    {
        const usize size = farr.size();

        basic_print("[", out_type);
        for (usize i = 0; i < size; ++i)
        {
            if constexpr (character_t<Element_t>)
            {
                basic_print(static_cast<i32>(farr[i]), out_type);
            }
            else
            {
                basic_print(farr[i], out_type);
            }
            if (i + 1 < size)
            {
                basic_print(", ", out_type);
            }
        }
        basic_print("]", out_type);
    }

    template<typename Element_t>
    void basic_print(const farray<Element_t>& farr, fTerminal_print_Invoker::stream_type out_type)
    {
        static_assert(has_basic_print_overload_v<Element_t>, "Unprintable data types");

        const usize size = farr.size();

        basic_print("[", out_type);
        for (usize i = 0; i < size; ++i)
        {
            if constexpr (character_t<Element_t>)
            {
                basic_print(static_cast<i32>(farr[i]), out_type);
            }
            else
            {
                basic_print(farr[i], out_type);
            }
            if (i + 1 < size)
            {
                basic_print(", ", out_type);
            }
        }
        basic_print("]", out_type);
    }

    template<typename Element_t, usize N>
    void basic_print(const std::array<Element_t, N>& stl_arr, fTerminal_print_Invoker::stream_type out_type)
    {
        static_assert(has_basic_print_overload_v<Element_t>, "Unprintable data types");

        basic_print("[", out_type);
        for (usize i = 0; i < N; ++i)
        {
            if constexpr (character_t<Element_t>)
            {
                basic_print(static_cast<i32>(stl_arr[i]), out_type);
            }
            else
            {
                basic_print(stl_arr[i], out_type);
            }
            if (i + 1 < N)
            {
                basic_print(", ", out_type);
            }
        }
        basic_print("]", out_type);
    }

    void basic_print(const fbytearray& bytearr, fTerminal_print_Invoker::stream_type out_type)
    {
        const usize size = bytearr.size();

        basic_print("[", out_type);
        for (usize i = 0; i < size; ++i)
        {
            basic_print(static_cast<usize>(bytearr[i]), out_type);
            if (i + 1 < size)
            {
                basic_print(", ", out_type);
            }
        }
        basic_print("]", out_type);
    }

    template<typename Element_t, usize N> requires((!character_t<Element_t>) && has_basic_print_overload_v<Element_t>)
    void basic_print(const Element_t(&array)[N], fTerminal_print_Invoker::stream_type out_type)
    {
        return basic_print(farray<Element_t>(array), out_type);
    }

    template<bool Error_handle, typename... Args>
    void basic_print(Args&&... args)
    {
        constexpr fTerminal_print_Invoker::stream_type hdt = Error_handle ?
            fTerminal_print_Invoker::stream_type::Error :
            fTerminal_print_Invoker::stream_type::Output;

        [&]<usize... index>(std::index_sequence<index...>)
        {
            ((basic_print(std::forward<Args>(args), hdt),
                index < sizeof...(Args) - 1 ? basic_print(" ", hdt) : void()), ...);
        }(std::index_sequence_for<Args...>{});
    }
}

export namespace fy
{
    template<typename... Args>
    void print(Args&&... args)
    {
        basic_print<false>(std::forward<Args>(args)...);
    }

    template<typename... Args>
    void println(Args&&... args)
    {
        print(std::forward<Args>(args)...);
        print("\n");
    }

    template<typename... Args>
    void error(Args&&... args)
    {
        basic_print<true>(std::forward<Args>(args)...);
    }
}

export namespace fy
{
    enum class ConsoleColor : WORD
    {
        Black = 0,
        DarkBlue = 1,
        DarkGreen = 2,
        DarkCyan = 3,
        DarkRed = 4,
        DarkMagenta = 5,
        DarkYellow = 6,
        Gray = 7,
        DarkGray = 8,
        Blue = 9,
        Green = 10,
        Cyan = 11,
        Red = 12,
        Magenta = 13,
        Yellow = 14,
        White = 15,
        Default
    };

    void set_print_color(ConsoleColor foreground, ConsoleColor background = ConsoleColor::Black)
    {
        WORD attribute = static_cast<WORD>(foreground) | (static_cast<WORD>(background) << 4);
        SetConsoleTextAttribute(global_Terminal_Output_Invoker.hConsole, attribute) != 0;
    }

    void reset_print_color()
    {
        SetConsoleTextAttribute(global_Terminal_Output_Invoker.hConsole,
            static_cast<WORD>(ConsoleColor::Gray) |
            (static_cast<WORD>(ConsoleColor::Black) << 4)) != 0;
    }

}