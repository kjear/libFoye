export module foye.fbytearray;

import foye.foye_core;
import std;

#pragma warning(disable: 4309)
#pragma warning(disable: 4244)
#pragma warning(disable: 4552)
#pragma warning(disable: 4552)
#pragma warning(disable: 4018)

export namespace fy
{
	class fbytearray
	{
	public:
		static inline constexpr usize BITS_PER_BYTE = std::numeric_limits<unsigned char>::digits;
		static inline constexpr usize BASE64_BITS = 6;
		static inline constexpr usize BASE64_GROUP_SIZE = 4;
		static inline constexpr usize INPUT_GROUP_SIZE = 3;
		static inline constexpr usize BYTE_MASK = 0xFF;
		static inline constexpr usize BASE64_MASK = 0x3F;
		static inline constexpr usize FIRST_CHAR_SHIFT = 18;
		static inline constexpr usize SECOND_CHAR_SHIFT = 12;
		static inline constexpr usize THIRD_CHAR_SHIFT = 6;
		static inline constexpr u8 INVALID_CHAR = 64;

		static inline constexpr char base64_chars[] =
			"ABCDEFGHIJKLMNOPQRSTUVWXYZ"
			"abcdefghijklmnopqrstuvwxyz"
			"0123456789+/";

		static inline constexpr u8 decoding_table[] = 
		{
				INVALID_CHAR, INVALID_CHAR, INVALID_CHAR, INVALID_CHAR, INVALID_CHAR, INVALID_CHAR, INVALID_CHAR, INVALID_CHAR,
				INVALID_CHAR, INVALID_CHAR, INVALID_CHAR, INVALID_CHAR, INVALID_CHAR, INVALID_CHAR, INVALID_CHAR, INVALID_CHAR,
				INVALID_CHAR, INVALID_CHAR, INVALID_CHAR, INVALID_CHAR, INVALID_CHAR, INVALID_CHAR, INVALID_CHAR, INVALID_CHAR,
				INVALID_CHAR, INVALID_CHAR, INVALID_CHAR, INVALID_CHAR, INVALID_CHAR, INVALID_CHAR, INVALID_CHAR, INVALID_CHAR,
				INVALID_CHAR, INVALID_CHAR, INVALID_CHAR, INVALID_CHAR, INVALID_CHAR, INVALID_CHAR, INVALID_CHAR, INVALID_CHAR,
				INVALID_CHAR, INVALID_CHAR, INVALID_CHAR, 62, INVALID_CHAR, INVALID_CHAR, INVALID_CHAR, 63,
				52, 53, 54, 55, 56, 57, 58, 59, 60, 61,
				INVALID_CHAR, INVALID_CHAR, INVALID_CHAR, INVALID_CHAR, INVALID_CHAR, INVALID_CHAR, INVALID_CHAR,
				0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
				16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
				INVALID_CHAR, INVALID_CHAR, INVALID_CHAR, INVALID_CHAR, INVALID_CHAR, INVALID_CHAR,
				26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
				36, 37, 38, 39, 40, 41, 42, 43, 44, 45,
				46, 47, 48, 49, 50, 51,
				INVALID_CHAR, INVALID_CHAR, INVALID_CHAR, INVALID_CHAR, INVALID_CHAR
		};

	public:
		fbytearray() noexcept
		{
			data_.stack.used_length = 0;
			data_.heap.buffer_begin = nullptr;
			data_.heap.buffer_end = nullptr;
			data_.heap.data_end = nullptr;
		}

		template<typename Element_t> requires(memcpyable<Element_t>)
		fbytearray(std::initializer_list<Element_t> init_list) noexcept
			: fbytearray(init_list.size() * sizeof(Element_t))
		{
			const Element_t* src = init_list.begin();
			const usize total_bytes = init_list.size() * sizeof(Element_t);
			std::memcpy(data(), src, total_bytes);
		}

		explicit fbytearray(usize init_sizebyte) noexcept
		{
			data_.stack.used_length = (init_sizebyte <= maximum_stack_capacity) ? init_sizebyte : -1;
			if (!is_using_stack__())
			{
				data_.heap.buffer_begin = memory_alloc<u8>(init_sizebyte);
				data_.heap.buffer_end = data_.heap.buffer_begin + init_sizebyte;
				data_.heap.data_end = data_.heap.buffer_end;
			}
		}

		template<typename Element_t> requires(memcpyable<Element_t>)
		explicit fbytearray(const Element_t* const ptr, usize count) noexcept
		{
			const usize total_bytes = sizeof(Element_t) * count;
			data_.stack.used_length = (total_bytes <= maximum_stack_capacity) ? total_bytes : -1;

			if (!is_using_stack__())
			{
				data_.heap.buffer_begin = memory_alloc<u8>(total_bytes);
				data_.heap.buffer_end = data_.heap.buffer_begin + total_bytes;
				data_.heap.data_end = data_.heap.buffer_end;
			}

			std::memcpy(data(), ptr, total_bytes);
		}

		~fbytearray() noexcept
		{
			if (!is_using_stack__() && data_.heap.buffer_begin)
			{
				memory_free(reinterpret_cast<void*>(data_.heap.buffer_begin));
			}

			data_.stack.used_length = 0;
			data_.heap.buffer_begin = nullptr;
			data_.heap.buffer_end = nullptr;
			data_.heap.data_end = nullptr;
		}

		fbytearray(const fbytearray& other) noexcept
		{
			const usize other_size = other.size();
			if (other_size <= maximum_stack_capacity)
			{
				std::memcpy(data_.stack.data, other.data_.stack.data, other_size);
				data_.stack.used_length = other_size;
			}
			else
			{
				data_.heap.buffer_begin = memory_alloc<u8>(other_size);
				data_.heap.buffer_end = data_.heap.buffer_begin + other_size;
				data_.heap.data_end = data_.heap.buffer_end;

				std::memcpy(data_.heap.buffer_begin, other.data_.heap.buffer_begin, other_size);
				data_.stack.used_length = -1;
			}
		}

		fbytearray(fbytearray&& other) noexcept
		{
			data_.stack.used_length = other.data_.stack.used_length;
			if (other.is_using_stack__())
			{
				std::memcpy(data_.stack.data, other.data_.stack.data, other.data_.stack.used_length);
			}
			else
			{
				data_.heap.buffer_begin = other.data_.heap.buffer_begin;
				data_.heap.buffer_end = other.data_.heap.buffer_end;
				data_.heap.data_end = other.data_.heap.data_end;
			}

			other.data_.heap.buffer_begin = nullptr;
			other.data_.heap.buffer_end = nullptr;
			other.data_.heap.data_end = nullptr;
			other.data_.stack.used_length = 0;
		}

		fbytearray& operator = (const fbytearray& other) noexcept
		{
			if (this != &other)
			{
				const usize other_size = other.size();

				if (other.is_using_stack__())
				{
					if (is_using_stack__())
					{
						std::memcpy(data_.stack.data, other.data_.stack.data, other_size);
						data_.stack.used_length = other_size;
					}
					else
					{
						if (memory_size__() >= other_size)
						{
							std::memcpy(data_.heap.buffer_begin, other.data_.stack.data, other_size);
							data_.heap.data_end = data_.heap.buffer_begin + other_size;
						}
						else
						{
							u8* new_buffer = memory_alloc<u8>(other_size);
							if (data_.heap.buffer_begin)
							{
								memory_free(reinterpret_cast<void*>(data_.heap.buffer_begin));
							}
							std::memcpy(new_buffer, other.data_.stack.data, other_size);
							data_.heap.buffer_begin = new_buffer;
							data_.heap.buffer_end = new_buffer + other_size;
							data_.heap.data_end = new_buffer + other_size;
							data_.stack.used_length = -1;
						}
					}
				}
				else
				{
					if (!is_using_stack__())
					{
						if (memory_size__() >= other_size)
						{
							std::memcpy(data_.heap.buffer_begin, other.data_.heap.buffer_begin, other_size);
							data_.heap.data_end = data_.heap.buffer_begin + other_size;
						}
						else
						{
							u8* new_buffer = memory_alloc<u8>(other_size);
							std::memcpy(new_buffer, other.data_.heap.buffer_begin, other_size);
							memory_free(reinterpret_cast<void*>(data_.heap.buffer_begin));
							data_.heap.buffer_begin = new_buffer;
							data_.heap.buffer_end = new_buffer + other_size;
							data_.heap.data_end = new_buffer + other_size;
						}
					}
					else
					{
						if (other_size <= maximum_stack_capacity)
						{
							std::memcpy(data_.stack.data, other.data_.heap.buffer_begin, other_size);
							data_.stack.used_length = other_size;
						}
						else if (memory_size__() >= other_size)
						{
							data_.stack.used_length = -1;
							if (!data_.heap.buffer_begin)
							{
								data_.heap.buffer_begin = memory_alloc<u8>(other_size);
								data_.heap.buffer_end = data_.heap.buffer_begin + other_size;
							}
							std::memcpy(data_.heap.buffer_begin, other.data_.heap.buffer_begin, other_size);
							data_.heap.data_end = data_.heap.buffer_begin + other_size;
						}
						else
						{
							u8* new_buffer = memory_alloc<u8>(other_size);
							std::memcpy(new_buffer, other.data_.heap.buffer_begin, other_size);
							data_.stack.used_length = -1;
							data_.heap.buffer_begin = new_buffer;
							data_.heap.buffer_end = new_buffer + other_size;
							data_.heap.data_end = new_buffer + other_size;
						}
					}
				}
			}

			return *this;
		}

		fbytearray& operator = (fbytearray&& other) noexcept
		{
			if (this != &other)
			{
				if (!is_using_stack__())
				{
					memory_free(reinterpret_cast<void*>(data_.heap.buffer_begin));
				}

				data_.stack.used_length = other.data_.stack.used_length;
				if (other.is_using_stack__())
				{
					std::memcpy(data_.stack.data, other.data_.stack.data, other.data_.stack.used_length);
				}
				else
				{
					data_.heap.buffer_begin = other.data_.heap.buffer_begin;
					data_.heap.buffer_end = other.data_.heap.buffer_end;
					data_.heap.data_end = other.data_.heap.data_end;
				}

				other.data_.stack.used_length = 0;
				other.data_.heap.buffer_begin = nullptr;
				other.data_.heap.buffer_end = nullptr;
				other.data_.heap.data_end = nullptr;
			}

			return *this;
		}

		bool operator == (const fbytearray& other) const noexcept
		{
			return size() == other.size() && std::memcmp(data<void>(), other.data<void>(), size()) == 0;
		}
		
		template<typename Element_t> requires(memcpyable<Element_t>)
		void assign(usize byte_offset, const Element_t* const array_begin, usize count_elem)
		{
			const usize total_bytes = sizeof(Element_t) * count_elem;

			if (byte_offset + total_bytes > memory_size__())
			{
				usize new_memory_size = static_cast<usize>((byte_offset + total_bytes) * factory);
				memory_expand(new_memory_size);
			}

			std::memcpy(data<u8>() + byte_offset, array_begin, total_bytes);
		}

		template<typename Element_t> requires(memcpyable<Element_t>)
		void assign(usize byte_offset, const fbytearray& other_arr)
		{
			const usize total_bytes = other_arr.size();

			if (byte_offset + total_bytes > memory_size__())
			{
				usize new_memory_size = static_cast<usize>((byte_offset + total_bytes) * factory);
				memory_expand(new_memory_size);
			}

			std::memcpy(data<u8>() + byte_offset, other_arr.data<u8>(), total_bytes);
		}

		u8& operator [] (usize index) noexcept
		{
			return (is_using_stack__() ? data_.stack.data : data_.heap.buffer_begin)[index];
		}

		const u8& operator [] (usize index) const noexcept
		{
			return (is_using_stack__() ? data_.stack.data : data_.heap.buffer_begin)[index];
		}

		void clear() noexcept
		{
			if (is_using_stack__())
			{
				data_.stack.used_length = 0;
			}
			else
			{
				data_.heap.data_end = data_.heap.buffer_begin;
			}
		}

		void broadcast(u8 value) noexcept
		{
			if (is_using_stack__())
			{
				std::memset(data_.stack.data, value, data_.stack.used_length);
			}
			else
			{
				std::memset(data_.heap.buffer_begin, value, data_.heap.data_end - data_.heap.buffer_begin);
			}
		}

		usize capacity() const noexcept
		{
			return is_using_stack__() ? maximum_stack_capacity - data_.stack.used_length
				: data_.heap.buffer_begin ? (data_.heap.buffer_end - data_.heap.buffer_begin) - size() : 0;
		}

		usize size() const noexcept
		{
			return is_using_stack__() ? data_.stack.used_length :
				data_.heap.buffer_begin ? data_.heap.data_end - data_.heap.buffer_begin : 0;
		}

		bool empty() const noexcept
		{
			return size() == 0;
		}

		void resize(usize new_capacity) noexcept
		{
			const usize current_size = size();
			if (!is_using_stack__())
			{
				if (new_capacity < current_size)
				{
					data_.heap.data_end = data_.heap.buffer_begin + new_capacity;
				}
				else
				{
					memory_expand(new_capacity);
				}
			}
			else
			{
				if (new_capacity <= maximum_stack_capacity)
				{
					data_.stack.used_length = new_capacity;
				}
				else
				{
					memory_expand(new_capacity);
				}
			}
		}

		void reserve(usize new_capacity) noexcept
		{
			if (new_capacity > memory_size__())
			{
				if (is_using_stack__())
				{
					const usize current_size = size();
					u8* new_memory = memory_alloc<u8>(new_capacity);

					std::memcpy(new_memory, data_.stack.data, current_size);

					data_.stack.used_length = -1;
					data_.heap.buffer_begin = new_memory;
					data_.heap.buffer_end = new_memory + new_capacity;
					data_.heap.data_end = new_memory + current_size;
				}
				else
				{
					memory_expand(new_capacity);
				}
			}
		}

		template<typename Element_t> requires(memcpyable<Element_t>)
		void push_back(Element_t&& elment) noexcept
		{
			if (capacity() < sizeof(Element_t))
			{
				memory_expand(static_cast<f64>(size() + sizeof(Element_t)) * factory);
			}

			if (is_using_stack__())
			{
				Element_t& ref = *reinterpret_cast<Element_t*>(data_.stack.data + size());
				ref = std::bit_cast<Element_t>(elment);
				data_.stack.used_length += sizeof(Element_t);
			}
			else
			{
				data_.heap.data_end += sizeof(Element_t);
				*reinterpret_cast<Element_t*>(data_.heap.data_end - sizeof(Element_t)) = std::bit_cast<Element_t>(elment);;
			}
		}

		template<typename T = u8>
		T& front() noexcept
		{
			return *reinterpret_cast<T*>(data());
		}

		template<typename T = u8>
		const T& front() const noexcept
		{
			return *reinterpret_cast<const T*>(data());
		}

		template<typename T = u8>
		T& back() noexcept
		{
			return *reinterpret_cast<T*>(data() + size() - sizeof(T));
		}

		template<typename T = u8>
		const T& back() const noexcept
		{
			return *reinterpret_cast<const T*>(data() + size() - sizeof(T));
		}

		template<typename T>
		T& reinterpret_at(usize byte_offset) noexcept
		{
			return *reinterpret_cast<T*>(data<u8>() + byte_offset);
		}

		template<typename T, typename ... Args>
		T& construct_at(usize byte_offset, Args&& ... args) noexcept
		{
			constexpr usize Element_size = sizeof(T);

			if (byte_offset + Element_size > memory_size__())
			{
				usize new_memory_size = static_cast<usize>((byte_offset + Element_size) * factory);
				memory_expand(new_memory_size);
			}

			T* obj_ptr = new (data<u8>() + byte_offset) T(std::forward<Args>(args)...);

			return *obj_ptr;
		}

		fbytearray encode_as_base64() const noexcept
		{
			const u8* input_data = data<u8>();
			usize input_size = size();

			usize output_size = ((input_size + (INPUT_GROUP_SIZE - 1)) / INPUT_GROUP_SIZE) * BASE64_GROUP_SIZE;

			fbytearray output_array(output_size);
			u8* output_data = output_array.data<u8>();

			usize i = 0, j = 0;

			while (i < input_size)
			{
				u32 octet_a = i < input_size ? input_data[i++] : 0;
				u32 octet_b = i < input_size ? input_data[i++] : 0;
				u32 octet_c = i < input_size ? input_data[i++] : 0;

				u32 triple = (octet_a << (BITS_PER_BYTE * 2)) +
					(octet_b << BITS_PER_BYTE) +
					octet_c;

				output_data[j++] = base64_chars[(triple >> FIRST_CHAR_SHIFT) & BASE64_MASK];
				output_data[j++] = base64_chars[(triple >> SECOND_CHAR_SHIFT) & BASE64_MASK];
				output_data[j++] = base64_chars[(triple >> THIRD_CHAR_SHIFT) & BASE64_MASK];
				output_data[j++] = base64_chars[triple & BASE64_MASK];
			}

			constexpr usize mod_table[] = { 0, 2, 1 };
			usize pad = mod_table[input_size % INPUT_GROUP_SIZE];

			for (usize k = 0; k < pad; ++k)
			{
				output_data[output_size - 1 - k] = '=';
			}

			return output_array;
		}

		fbytearray decode_from_base64() const
		{
			const u8* input_data = data<u8>();
			usize input_size = size();

			if (input_size % BASE64_GROUP_SIZE != 0)
			{
				return fbytearray();
			}

			usize padding = 0;
			constexpr usize MIN_SIZE_FOR_PADDING = 2;
			if (input_size >= MIN_SIZE_FOR_PADDING)
			{
				if (input_data[input_size - 1] == '=') padding++;
				if (input_data[input_size - 2] == '=') padding++;
			}

			usize output_size = (input_size / BASE64_GROUP_SIZE) * INPUT_GROUP_SIZE - padding;

			fbytearray output_array(output_size);
			u8* output_data = output_array.data<u8>();

			usize i = 0, j = 0;
			constexpr usize ASCII_MASK = 0x7F;

			while (i < input_size)
			{
				u32 sextet_a = input_data[i] == '=' ? 0 & i++ : decoding_table[input_data[i++] & ASCII_MASK];
				u32 sextet_b = input_data[i] == '=' ? 0 & i++ : decoding_table[input_data[i++] & ASCII_MASK];
				u32 sextet_c = input_data[i] == '=' ? 0 & i++ : decoding_table[input_data[i++] & ASCII_MASK];
				u32 sextet_d = input_data[i] == '=' ? 0 & i++ : decoding_table[input_data[i++] & ASCII_MASK];

				u32 triple = (sextet_a << FIRST_CHAR_SHIFT) +
					(sextet_b << SECOND_CHAR_SHIFT) +
					(sextet_c << THIRD_CHAR_SHIFT) +
					sextet_d;

				if (j < output_size) output_data[j++] = (triple >> (BITS_PER_BYTE * 2)) & BYTE_MASK;
				if (j < output_size) output_data[j++] = (triple >> BITS_PER_BYTE) & BYTE_MASK;
				if (j < output_size) output_data[j++] = triple & BYTE_MASK;
			}

			return output_array;
		}


		template<typename T = u8>
		const T* data() const noexcept
		{
			return reinterpret_cast<const T*>(is_using_stack__() ? data_.stack.data : data_.heap.buffer_begin);
		}

		template<typename T = u8>
		T* data() noexcept
		{
			return reinterpret_cast<T*>(is_using_stack__() ? data_.stack.data : data_.heap.buffer_begin);
		}

	private:
		usize memory_size__() const noexcept
		{
			return is_using_stack__() ? maximum_stack_capacity :
				data_.heap.buffer_begin ? data_.heap.buffer_end - data_.heap.buffer_begin : 0;
		}

		bool is_using_stack__() const noexcept
		{
			return data_.stack.used_length >= 0;
		}

		void memory_expand(usize new_memory_size) noexcept
		{
			u8* raw_memory = memory_alloc<u8>(new_memory_size);

			const usize size_ = size();

			if (is_using_stack__())
			{
				std::memcpy(raw_memory, data_.stack.data, size_ * sizeof(u8));
				data_.stack.used_length = -1;
			}
			else
			{
				std::memcpy(raw_memory, data_.heap.buffer_begin, size_ * sizeof(u8));
				memory_free(reinterpret_cast<void*>(data_.heap.buffer_begin));
			}

			data_.heap.buffer_begin = raw_memory;
			data_.heap.buffer_end = data_.heap.buffer_begin + new_memory_size;
			data_.heap.data_end = data_.heap.buffer_begin + size_;
		}

		static constexpr usize maximum_stack_capacity = 31;
		static constexpr f64 factory = 2.0;

		union
		{
			struct
			{
				u8 data[31];
				i8 used_length;
			} stack;

			struct
			{
				u8* buffer_begin;
				u8* buffer_end;
				u8* data_end;
			} heap;

		} data_;
	};
}