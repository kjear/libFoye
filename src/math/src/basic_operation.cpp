module;
#include <immintrin.h>

module foye.algorithm;
import foye.foye_core;
import foye.simd;
import std;

namespace fy
{
	using namespace simd;

	template<BasicArithmetic value_t> struct vector_basic_arithmetic_invoker
	{
		enum class operation_type
		{
			add, sub, mul, div,
			AND, OR, XOR,
			remainder
		};

		template<operation_type Op> struct operation {};

		template<> struct operation<operation_type::add>
		{
			static constexpr auto simd_op = v_add<AVX_t<value_t>>;
			static constexpr const std::plus<> scalar_op = std::plus<>{};
		};

		template<> struct operation<operation_type::sub>
		{
			static constexpr auto simd_op = v_sub<AVX_t<value_t>>;
			static constexpr const std::minus<> scalar_op = std::minus<>{};
		};

		template<> struct operation<operation_type::mul>
		{
			static constexpr auto simd_op = v_mul<AVX_t<value_t>>;
			static constexpr const std::multiplies<> scalar_op = std::multiplies<>{};
		};

		template<> struct operation<operation_type::div>
		{
			static constexpr auto simd_op = v_div<AVX_t<value_t>>;
			static constexpr const std::divides<> scalar_op = std::divides<>{};
		};

		template<> struct operation<operation_type::AND>
		{
			static constexpr auto simd_op = v_bitwise_AND<AVX_t<value_t>>;
			static constexpr const std::bit_and<> scalar_op = std::bit_and<>{};
		};

		template<> struct operation<operation_type::OR>
		{
			static constexpr auto simd_op = v_bitwise_OR<AVX_t<value_t>>;
			static constexpr const std::bit_or<> scalar_op = std::bit_or<>{};
		};

		template<> struct operation<operation_type::XOR>
		{
			static constexpr auto simd_op = v_bitwise_XOR<AVX_t<value_t>>;
			static constexpr const std::bit_xor<> scalar_op = std::bit_xor<>{};
		};

		template<> struct operation<operation_type::remainder>
		{
			static constexpr auto simd_op = v_remainder<AVX_t<value_t>>;
			static constexpr struct scalar_remainder
			{
				template <typename T>
				constexpr T operator()(const T& x, const T& y) const
				{
					return x % y;
				}
			} scalar_op{};
		};

		const value_t* left_;
		usize length_;

		vector_basic_arithmetic_invoker(const value_t* left, usize length) noexcept : left_(left), length_(length) {}

		template<operation_type op_t> void process(const value_t* right, value_t* result) noexcept
		{
			static constexpr usize simd_batch_size = AVX_t<value_t>::batch_size;

			usize i = 0;
			const auto& op = operation<op_t>{};

			for (; i + simd_batch_size <= length_; i += simd_batch_size)
			{
				op.simd_op(AVX_t<value_t>(&left_[i]), AVX_t<value_t>(&right[i])).download(&result[i]);
			}
			for (; i < length_; ++i)
			{
				result[i] = op.scalar_op(left_[i], right[i]);
			}
		}

		template<operation_type op_t, BasicArithmetic scalar_t> void process(scalar_t right, value_t* result) noexcept
		{
			static constexpr usize simd_batch_size = AVX_t<value_t>::batch_size;

			usize i = 0;
			const auto& op = operation<op_t>{};

			const value_t right_ = saturate_cast<value_t>(right);
			const AVX_t<value_t> v_right(right_);

			for (; i + simd_batch_size <= length_; i += simd_batch_size)
			{
				op.simd_op(AVX_t<value_t>(&left_[i]), v_right).download(&result[i]);
			}
			for (; i < length_; ++i)
			{
				result[i] = op.scalar_op(left_[i], right_);
			}
		}
	};
}

namespace fy
{
	template<BasicArithmetic Element_t>
	void addition(const Element_t* left, const Element_t* right, Element_t* dst, usize length) noexcept
	{
		vector_basic_arithmetic_invoker<Element_t> invoker(left, length);
		invoker.process<vector_basic_arithmetic_invoker<Element_t>::operation_type::add>(right, dst);
	}

	template<BasicArithmetic Element_t>
	void subtraction(const Element_t* left, const Element_t* right, Element_t* dst, usize length) noexcept
	{
		vector_basic_arithmetic_invoker<Element_t> invoker(left, length);
		invoker.process<vector_basic_arithmetic_invoker<Element_t>::operation_type::sub>(right, dst);
	}

	template<BasicArithmetic Element_t>
	void multiplication(const Element_t* left, const Element_t* right, Element_t* dst, usize length) noexcept
	{
		vector_basic_arithmetic_invoker<Element_t> invoker(left, length);
		invoker.process<vector_basic_arithmetic_invoker<Element_t>::operation_type::mul>(right, dst);
	}

	template<BasicArithmetic Element_t>
	void division(const Element_t* left, const Element_t* right, Element_t* dst, usize length) noexcept
	{
		vector_basic_arithmetic_invoker<Element_t> invoker(left, length);
		invoker.process<vector_basic_arithmetic_invoker<Element_t>::operation_type::div>(right, dst);
	}

	template<BasicArithmetic Element_t>
	void bit_and(const Element_t* left, const Element_t* right, Element_t* dst, usize length) noexcept
	{
		vector_basic_arithmetic_invoker<Element_t> invoker(left, length);
		invoker.process<vector_basic_arithmetic_invoker<Element_t>::operation_type::AND>(right, dst);
	}

	template<BasicArithmetic Element_t>
	void bit_or(const Element_t* left, const Element_t* right, Element_t* dst, usize length) noexcept
	{
		vector_basic_arithmetic_invoker<Element_t> invoker(left, length);
		invoker.process<vector_basic_arithmetic_invoker<Element_t>::operation_type::OR>(right, dst);
	}

	template<BasicArithmetic Element_t>
	void bit_xor(const Element_t* left, const Element_t* right, Element_t* dst, usize length) noexcept
	{
		vector_basic_arithmetic_invoker<Element_t> invoker(left, length);
		invoker.process<vector_basic_arithmetic_invoker<Element_t>::operation_type::XOR>(right, dst);
	}

	template<BasicArithmetic Element_t>
	void remainder(const Element_t* left, const Element_t* right, Element_t* dst, usize length) noexcept
	{
		vector_basic_arithmetic_invoker<Element_t> invoker(left, length);
		invoker.process<vector_basic_arithmetic_invoker<Element_t>::operation_type::remainder>(right, dst);
	}
}

namespace fy
{
	template<BasicArithmetic Element_t, BasicArithmetic Right>
	void addition(const Element_t* left, Right right, Element_t* dst, usize length) noexcept
	{
		vector_basic_arithmetic_invoker<Element_t> invoker(left, length);
		invoker.process<vector_basic_arithmetic_invoker<Element_t>::operation_type::add>(saturate_cast<Element_t>(right), dst);
	}

	template<BasicArithmetic Element_t, BasicArithmetic Right>
	void subtraction(const Element_t* left, Right right, Element_t* dst, usize length) noexcept
	{
		vector_basic_arithmetic_invoker<Element_t> invoker(left, length);
		invoker.process<vector_basic_arithmetic_invoker<Element_t>::operation_type::sub>(saturate_cast<Element_t>(right), dst);
	}

	template<BasicArithmetic Element_t, BasicArithmetic Right>
	void multiplication(const Element_t* left, Right right, Element_t* dst, usize length) noexcept
	{
		vector_basic_arithmetic_invoker<Element_t> invoker(left, length);
		invoker.process<vector_basic_arithmetic_invoker<Element_t>::operation_type::mul>(saturate_cast<Element_t>(right), dst);
	}

	template<BasicArithmetic Element_t, BasicArithmetic Right>
	void division(const Element_t* left, Right right, Element_t* dst, usize length) noexcept
	{
		vector_basic_arithmetic_invoker<Element_t> invoker(left, length);
		invoker.process<vector_basic_arithmetic_invoker<Element_t>::operation_type::div>(saturate_cast<Element_t>(right), dst);
	}

	template<BasicArithmetic Element_t, BasicArithmetic Right>
	void bit_and(const Element_t* left, Right right, Element_t* dst, usize length) noexcept
	{
		vector_basic_arithmetic_invoker<Element_t> invoker(left, length);
		invoker.process<vector_basic_arithmetic_invoker<Element_t>::operation_type::AND>(saturate_cast<Element_t>(right), dst);
	}

	template<BasicArithmetic Element_t, BasicArithmetic Right>
	void bit_or(const Element_t* left, Right right, Element_t* dst, usize length) noexcept
	{
		vector_basic_arithmetic_invoker<Element_t> invoker(left, length);
		invoker.process<vector_basic_arithmetic_invoker<Element_t>::operation_type::OR>(saturate_cast<Element_t>(right), dst);
	}

	template<BasicArithmetic Element_t, BasicArithmetic Right>
	void bit_xor(const Element_t* left, Right right, Element_t* dst, usize length) noexcept
	{
		vector_basic_arithmetic_invoker<Element_t> invoker(left, length);
		invoker.process<vector_basic_arithmetic_invoker<Element_t>::operation_type::XOR>(saturate_cast<Element_t>(right), dst);
	}

	template<BasicArithmetic Element_t, BasicArithmetic Right>
	void remainder(const Element_t* left, Right right, Element_t* dst, usize length) noexcept
	{
		vector_basic_arithmetic_invoker<Element_t> invoker(left, length);
		invoker.process<vector_basic_arithmetic_invoker<Element_t>::operation_type::remainder>(saturate_cast<Element_t>(right), dst);
	}


	template<usize N = 2>
	f64 rsqrt_double(f64 x)
	{
		if (x < 0.0) return std::nan("");
		if (x == 0.0) return std::numeric_limits<f64>::infinity();

		if (std::isnan(x)) return std::nan("");
		if (std::isinf(x)) return 0.0;

		constexpr u64 MAGIC = 0x5FE6EB50C7B537AA;
		constexpr f64 HALF = 0.5;
		constexpr f64 ONE_HALF = 1.5;

		f64 initial_guess;
		{
			u64 i;
			std::memcpy(&i, &x, sizeof(x));

			i = MAGIC - (i >> 1);
			std::memcpy(&initial_guess, &i, sizeof(i));
		}

		f64 y = initial_guess;

		for (usize i = 0; i < N; ++i)
		{
			y *= (ONE_HALF - (x * HALF) * y * y);
		}
		return y;
	}
	
}
