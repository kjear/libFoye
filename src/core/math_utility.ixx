export module foye.math_utility;

import std;
import foye.alias;
import foye.type_characteristics;

export namespace fy
{
    template<Floating_arithmetic Element_t>
    constexpr bool isnan(Element_t value) noexcept
    {
        if consteval
        {
            using Int_t = typename fixed_width_types<sizeof(Element_t) * 8>::unsigned_integral_t;
            static_assert(!std::is_same_v<Int_t, void>, "Unsupported floating-point size");

            constexpr auto exp_bits = std::numeric_limits<Element_t>::digits - 1;
            constexpr u64 total_bits = sizeof(Element_t) * 8;
            constexpr auto exp_mask = Int_t((1ULL << exp_bits) - 1) << (total_bits - exp_bits - 1);
            constexpr auto man_mask = (Int_t(1) << (total_bits - exp_bits - 1)) - 1;

            Int_t bits;
            if constexpr (std::endian::native == std::endian::little)
            {
                bits = *reinterpret_cast<Int_t*>(&value);
            }
            else
            {
                char* ptr = reinterpret_cast<char*>(&value);
                char bytes[sizeof(Element_t)];
                std::reverse_copy(ptr, ptr + sizeof(Element_t), bytes);
                bits = *reinterpret_cast<Int_t*>(bytes);
            }

            return (bits & exp_mask) == exp_mask && (bits & man_mask) != 0;
        }
        else
        {
            return std::isnan(value);
        }
    }

    template<integral_arithmetic Element_t>
    constexpr bool isnan(Element_t value) noexcept
    {
        return fy::isnan<f64>(static_cast<f64>(value));
    }

}