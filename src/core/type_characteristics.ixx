export module foye.type_characteristics;

export import foye.extensionType.float16;
export import foye.extensionType.bfloat16;
export import foye.extensionType.int128;
export import foye.alias;
import std;

export namespace fy
{
	template<typename T> concept BasicArithmetic = std::disjunction_v<
		std::is_same<T, u8>, std::is_same<T, u16>, std::is_same<T, u32>, std::is_same<T, u64>,
		std::is_same<T, i8>, std::is_same<T, i16>, std::is_same<T, i32>, std::is_same<T, i64>,
		std::is_same<T, bf16>, std::is_same<T, f16>, std::is_same<T, f32>, std::is_same<T, f64>,
		std::is_same<T, u128>, std::is_same<T, i128>
	>;

	template<typename T>
	concept Floating_arithmetic = std::disjunction_v<std::is_same<T, bf16>, std::is_same<T, f16>, std::is_same<T, f32>, std::is_same<T, f64>>;

	template<typename T>
	concept integral_arithmetic = std::disjunction_v<
		std::is_same<T, u8>, std::is_same<T, u16>, std::is_same<T, u32>, std::is_same<T, u64>,
		std::is_same<T, i8>, std::is_same<T, i16>, std::is_same<T, i32>, std::is_same<T, i64>,
		std::is_same<T, u128>, std::is_same<T, i128>
	>;

	template<typename T>
	concept Signed_integral_arithmetic = std::disjunction_v<
		std::is_same<T, i8>, std::is_same<T, i16>, std::is_same<T, i32>, std::is_same<T, i64>, std::is_same<T, i128>
	>;

	template<typename T>
	concept Unsigned_integral_arithmetic = std::disjunction_v<
		std::is_same<T, u8>, std::is_same<T, u16>, std::is_same<T, u32>, std::is_same<T, u64>, std::is_same<T, u128>
	>;

	template<typename T>
	using extended_t = typename std::conditional_t<
		std::is_unsigned_v<T>, u64,
		typename std::conditional_t<std::is_integral_v<T>, i64, f64>
	>;

	template<typename T>
	constexpr bool is_basic_arithmetic = BasicArithmetic<T>;

	template<typename Char_t>
	concept character_t =
		std::is_same_v<Char_t, char8_t> ||
		std::is_same_v<Char_t, char16_t> ||
		std::is_same_v<Char_t, char32_t> ||
		std::is_same_v<Char_t, char> ||
		std::is_same_v<Char_t, wchar_t>;

	template<typename T>
	concept bitcomparable = std::is_trivially_copyable_v<T>;

	template<typename T>
	concept memcpyable = std::is_trivially_copyable_v<T> && 
		std::is_standard_layout_v<T> && !std::is_pointer_v<T>;

	template<usize N_bits> struct fixed_width_types
	{
		using signed_integral_t = void;
		using unsigned_integral_t = void;
		using floating_t = void;
	};

	template<> struct fixed_width_types<8>
	{
		using signed_integral_t = i8;
		using unsigned_integral_t = u8;
		using floating_t = void;
	};

	template<> struct fixed_width_types<16>
	{
		using signed_integral_t = i16;
		using unsigned_integral_t = u16;
		using floating_t = f16;
	};

	template<> struct fixed_width_types<32>
	{
		using signed_integral_t = i32;
		using unsigned_integral_t = u32;
		using floating_t = f32;
	};

	template<> struct fixed_width_types<64>
	{
		using signed_integral_t = i64;
		using unsigned_integral_t = u64;
		using floating_t = f64;
	};


	template<Floating_arithmetic Element_t>
	struct limits_floating
	{
		static constexpr Element_t min() noexcept { return std::numeric_limits<Element_t>::min(); }
		static constexpr Element_t max() noexcept { return std::numeric_limits<Element_t>::max(); }
		static constexpr Element_t lowest() noexcept { return std::numeric_limits<Element_t>::lowest(); }
		static constexpr Element_t epsilon() noexcept { return std::numeric_limits<Element_t>::epsilon(); }

		static constexpr Element_t round_error() noexcept { return std::numeric_limits<Element_t>::round_error(); }
		static constexpr Element_t denorm_min() noexcept { return std::numeric_limits<Element_t>::denorm_min(); }
		static constexpr Element_t infinity() noexcept { return std::numeric_limits<Element_t>::infinity(); }
		static constexpr Element_t quiet_NaN() noexcept { return std::numeric_limits<Element_t>::quiet_NaN(); }
		static constexpr Element_t signaling_NaN() noexcept { return std::numeric_limits<Element_t>::signaling_NaN(); }

		static constexpr usize digits = usize(std::numeric_limits<Element_t>::digits);
		static constexpr usize digits10 = usize(std::numeric_limits<Element_t>::digits10);
		static constexpr usize max_digits10 = usize(std::numeric_limits<Element_t>::max_digits10);
		static constexpr usize max_exponent = usize(std::numeric_limits<Element_t>::max_exponent);
		static constexpr usize max_exponent10 = usize(std::numeric_limits<Element_t>::max_exponent10);
		static constexpr usize min_exponent = usize(std::numeric_limits<Element_t>::min_exponent);
		static constexpr usize min_exponent10 = usize(std::numeric_limits<Element_t>::min_exponent10);
	};

	template<integral_arithmetic Element_t>
	struct limits_integral
	{
		static constexpr Element_t min() noexcept { return std::numeric_limits<Element_t>::min(); }
		static constexpr Element_t max() noexcept { return std::numeric_limits<Element_t>::max(); }
		static constexpr Element_t lowest() noexcept { return std::numeric_limits<Element_t>::lowest(); }
		static constexpr Element_t epsilon() noexcept { return std::numeric_limits<Element_t>::epsilon(); }
		static constexpr Element_t round_error() noexcept { return std::numeric_limits<Element_t>::round_error(); }
		static constexpr Element_t denorm_min() noexcept { return std::numeric_limits<Element_t>::denorm_min(); }
		static constexpr Element_t infinity() noexcept { return std::numeric_limits<Element_t>::infinity(); }
		static constexpr Element_t quiet_NaN() noexcept { return std::numeric_limits<Element_t>::quiet_NaN(); }
		static constexpr Element_t signaling_NaN() noexcept { return std::numeric_limits<Element_t>::signaling_NaN(); }

		static constexpr usize digits = usize(std::numeric_limits<Element_t>::digits);
		static constexpr usize digits10 = usize(std::numeric_limits<Element_t>::digits10);
		static constexpr bool is_signed = std::is_signed_v<Element_t>;
	};
}

export namespace fy
{
	template<> struct limits_floating<bfloat16>
	{
		static constexpr bfloat16 min() noexcept { return bfloat16::bfloatFromBits(0x0080); }
		static constexpr bfloat16 max() noexcept { return bfloat16::bfloatFromBits(0x7F7F); }
		static constexpr bfloat16 lowest() noexcept { return bfloat16::bfloatFromBits(0xFF7F); }
		static constexpr bfloat16 epsilon() noexcept { return bfloat16::bfloatFromBits(0x3C00); }
		static constexpr bfloat16 round_error() noexcept { return bfloat16(0.5f); }
		static constexpr bfloat16 denorm_min() noexcept { return bfloat16::bfloatFromBits(0x0001); }
		static constexpr bfloat16 infinity() noexcept { return bfloat16::bfloatFromBits(0x7F80); }
		static constexpr bfloat16 quiet_NaN() noexcept { return bfloat16::bfloatFromBits(0x7FC0); }
		static constexpr bfloat16 signaling_NaN() noexcept { return bfloat16::bfloatFromBits(0x7F80 | 0x0040); }

		static constexpr usize digits = 8;
		static constexpr usize digits10 = 2;
		static constexpr usize max_digits10 = 4;
		static constexpr usize  min_exponent = -126;
		static constexpr usize  min_exponent10 = -37;
		static constexpr usize  max_exponent = 127;
		static constexpr usize  max_exponent10 = 38;
	};

	template<> struct limits_floating<float16>
	{
		static constexpr float16 min() noexcept { return float16(1.0f / 65504.0f); }
		static constexpr float16 max() noexcept { return float16(65504.0f); }
		static constexpr float16 lowest() noexcept { return float16(-65504.0f); }
		static constexpr float16 epsilon() noexcept { return float16::hfloatFromBits(0x1400); }
		static constexpr float16 round_error() noexcept { return float16(0.5f); }
		static constexpr float16 denorm_min() noexcept { return float16::hfloatFromBits(0x0001); }
		static constexpr float16 infinity() noexcept { return float16::hfloatFromBits(0x7C00); }
		static constexpr float16 quiet_NaN() noexcept { return float16::hfloatFromBits(0x7E00); }
		static constexpr float16 signaling_NaN() noexcept { return float16::hfloatFromBits(0x7D00); }

		static constexpr usize digits = 11;
		static constexpr usize digits10 = 3;
		static constexpr usize max_digits10 = 5;
		static constexpr usize min_exponent = -13;
		static constexpr usize min_exponent10 = -4;
		static constexpr usize max_exponent = 16;
		static constexpr usize max_exponent10 = 4;
	};

	template<> struct limits_integral<uint128>
	{
		static constexpr uint128 min() noexcept { return 0; }
		static constexpr uint128 max() noexcept { return uint128{ ~0ull, ~0ull }; }
		static constexpr uint128 lowest() noexcept { return limits_integral<uint128>::min(); }
		static constexpr uint128 epsilon() noexcept { return 0; }
		static constexpr uint128 round_error() noexcept { return 0; }
		static constexpr uint128 denorm_min() noexcept { return 0; }
		static constexpr uint128 infinity() noexcept { return 0; }
		static constexpr uint128 quiet_NaN() noexcept { return 0; }
		static constexpr uint128 signaling_NaN() noexcept { return 0; }

		static constexpr bool is_signed = false;
		static constexpr usize digits = 128;
		static constexpr usize digits10 = 38;
	};

	template<> struct limits_integral<int128>
	{
		static constexpr int128 min() noexcept { return int128{ 0ull, 1ull << 63 }; }
		static constexpr int128 max() noexcept { return int128{ ~0ull, ~0ull >> 1 }; }
		static constexpr int128 lowest() noexcept { return limits_integral<int128>::min(); }
		static constexpr int128 epsilon() noexcept { return 0; }
		static constexpr int128 round_error() noexcept { return 0; }
		static constexpr int128 denorm_min() noexcept { return 0; }
		static constexpr int128 infinity() noexcept { return 0; }
		static constexpr int128 quiet_NaN() noexcept { return 0; }
		static constexpr int128 signaling_NaN() noexcept { return 0; }

		static constexpr bool is_signed = true;
		static constexpr usize digits = 127;
		static constexpr usize digits10 = 38;
	};
}