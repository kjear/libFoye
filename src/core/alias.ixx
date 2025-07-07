export module foye.alias;

import std;

export namespace fy
{
	using u8 = std::uint8_t;
	using u16 = std::uint16_t;
	using u32 = std::uint32_t;
	using u64 = std::uint64_t;

	using i8 = std::int8_t;
	using i16 = std::int16_t;
	using i32 = std::int32_t;
	using i64 = std::int64_t;

	using f32 = std::float_t;
	using f64 = std::double_t;

	using usize = std::uintmax_t;
	using ssize = std::intmax_t;

	using uptr_t = std::uintptr_t;
	using sptr_t = std::intptr_t;
}