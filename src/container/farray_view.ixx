export module foye.farray_view;

import foye.foye_core;
import std;

export namespace fy
{
    template<typename Element_t>
    class farray_view
    {
    public:
        using value_type = Element_t;
        using reference = value_type&;
        using const_reference = const value_type&;
        using pointer = value_type*;
        using const_pointer = const value_type*;

        template<usize N>
        constexpr farray_view(const Element_t(&array)[N]) noexcept
            : farray_view(&array[0], &array[N]) { }

        constexpr farray_view(const Element_t* const& begin, const Element_t* const& end) noexcept
            : buffer_begin(begin), buffer_end(end) {  }

        constexpr usize size() const noexcept
        {
            return buffer_end - buffer_begin;
        }

        constexpr bool empty() const noexcept
        {
            return buffer_begin == buffer_end;
        }

        constexpr const_reference operator[](usize pos) const noexcept
        {
            return buffer_begin[pos];
        }

        constexpr const_reference front() const noexcept
        {
            return *buffer_begin;
        }

        constexpr const_reference back() const noexcept
        {
            return *(buffer_end - 1);
        }

        constexpr const_pointer begin() const noexcept
        {
            return buffer_begin;
        }

        constexpr const_pointer end() const noexcept
        {
            return buffer_end;
        }

        constexpr void set_end(const Element_t* const& new_end) noexcept
        {
            buffer_end = new_end;
        }

        constexpr void set_begin(const Element_t* const& new_begin) noexcept
        {
            buffer_begin = new_begin;
        }

        constexpr farray_view<Element_t> subview(usize offset, usize count) const
        {
            return farray_view<Element_t>(buffer_begin + offset, buffer_begin + offset + count);
        }

        constexpr const_pointer data() const noexcept
        {
            return buffer_begin;
        }

    private:
        const Element_t* const& buffer_begin;
        const Element_t* const& buffer_end;
    };

}