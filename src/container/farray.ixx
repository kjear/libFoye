export module foye.farray;

import foye.foye_core;
import std;

#pragma warning(disable: 4309)
#pragma warning(disable: 4244)

export namespace fy
{
	template<typename Element_t> class farray;
	template<typename Element_t> class fmdarray;
	template<BasicArithmetic Element_t> class fvalarray;

	template<BasicArithmetic Element_t>
	class fvalarray
	{
	public:
		fvalarray() noexcept : begin_(nullptr), end_(nullptr) { }

		explicit fvalarray(usize num) : begin_(nullptr), end_(nullptr)
		{
			Element_t* raw_ptr = memory_alloc<Element_t>(num);
			begin_ = write_tag(raw_ptr, false);
			end_ = raw_ptr + num;
		}

		explicit fvalarray(Element_t* ptr, usize num)
			: begin_(ptr ? write_tag(ptr, true) : nullptr)
			, end_(ptr ? ptr + num : nullptr)
		{
			if (ptr && (!((reinterpret_cast<std::uintptr_t>(ptr) & tag_mask) == 0)))
			{
				throw std::runtime_error("Low bits attempt to cover behavior is illegal");
			}
		}

		explicit fvalarray(farray<Element_t>& farr) noexcept;
		explicit fvalarray(std::vector<Element_t>& farr) noexcept;

		explicit fvalarray(Element_t* external_begin, Element_t* external_end) noexcept
			: fvalarray(external_begin, external_end - external_begin) { }

		template<typename Input_element_t>
		explicit fvalarray(std::initializer_list<Input_element_t> init_list)
			: fvalarray(init_list.size())
		{
			if (!begin_)
			{
				return;
			}

			Element_t* dest = clean_tag(begin_);
			const auto* src = init_list.begin();

			if constexpr (std::is_same_v<Element_t, Input_element_t>)
			{
				std::memcpy(dest, src, sizeof(Element_t) * init_list.size());
			}
			else
			{
				std::transform(src, src + init_list.size(), dest,
					[ ](Input_element_t val) {
						return static_cast<Element_t>(val);
					}
				);
			}
		}

		~fvalarray()
		{
			reset();
		}

		void reset() noexcept
		{
			if (begin_ && is_owner())
			{
				memory_free(reinterpret_cast<void*>(clean_tag(begin_)));
			}
			begin_ = nullptr;
			end_ = nullptr;
		}

		fvalarray(fvalarray&& other) noexcept
			: begin_(other.begin_)
			, end_(other.end_)
		{
			other.begin_ = nullptr;
			other.end_ = nullptr;
		}

		fvalarray& operator = (fvalarray&& other) noexcept
		{
			if (this != &other)
			{
				reset();
				begin_ = other.begin_;
				end_ = other.end_;
				other.begin_ = nullptr;
				other.end_ = nullptr;
			}
			return *this;
		}

		fvalarray(const fvalarray& other) noexcept
		{
			if (other.is_owner())
			{
				Element_t* raw_ptr = memory_alloc<Element_t>(other.size());

				begin_ = write_tag(raw_ptr, false);
				end_ = raw_ptr + other.size();

				std::memcpy(data(), other.data(), sizeof(Element_t) * size());
			}
			else
			{
				begin_ = other.begin_;
				end_ = other.end_;
			}
		}

		fvalarray& operator = (const fvalarray& other) noexcept
		{
			if (this == &other)
			{
				return *this;
			}

			if (other.empty())
			{
				reset();
				return *this;
			}

			if (is_owner() && (size() >= other.size()))
			{
				std::memcpy(data(), other.data(), sizeof(Element_t) * other.size());
				end_ = data() + other.size();
			}
			else
			{
				reset();

				if (other.is_owner())
				{
					const usize n = other.size();
					Element_t* new_mem = memory_alloc<Element_t>(n);
					std::memcpy(new_mem, other.data(), sizeof(Element_t) * n);
					begin_ = write_tag(new_mem, false);
					end_ = new_mem + n;
				}
				else
				{
					begin_ = other.begin_;
					end_ = other.end_;
				}
			}

			return *this;
		}

		Element_t& operator [] (usize i) noexcept
		{
			return clean_tag(begin_)[i];
		}

		const Element_t& operator [] (usize i) const noexcept
		{
			return clean_tag(begin_)[i];
		}

		usize size() const noexcept
		{
			return end_ ? end_ - clean_tag(begin_) : 0;
		}

		bool empty() const noexcept
		{
			return begin_ == end_ || !begin_;
		}

		Element_t* data() noexcept
		{
			return begin_ ? clean_tag(begin_) : nullptr;
		}

		const Element_t* data() const noexcept
		{
			return begin_ ? clean_tag(begin_) : nullptr;
		}

		Element_t& front() noexcept
		{
			return *clean_tag(begin_);
		}

		const Element_t& front() const noexcept
		{
			return *clean_tag(begin_);
		}

		Element_t& back() noexcept
		{
			return *(end_ - 1);
		}

		const Element_t& back() const noexcept
		{
			return *(end_ - 1);
		}

		Element_t* begin() noexcept
		{
			return clean_tag(begin_);
		}

		Element_t* end() noexcept
		{
			return end_;
		}

		Element_t* begin() const noexcept
		{
			return clean_tag(begin_);
		}

		const Element_t* end() const noexcept
		{
			return end_;
		}

		const Element_t* cbegin() const noexcept
		{
			return clean_tag(begin_);
		}

		const Element_t* cend() const noexcept
		{
			return end_;
		}

		fvalarray make_copy(usize index_begin = 0, usize subsize = 0) const noexcept
		{
			return make_view().as_alone();
		}

		fvalarray make_view(usize index_begin = 0, usize subsize = 0) const noexcept
		{
			if (empty() || index_begin >= size())
			{
				return fvalarray();
			}

			usize actual_size = (subsize == 0) ?
				(size() - index_begin) :
				std::min(subsize, size() - index_begin);

			Element_t* start_ptr = clean_tag(begin_) + index_begin;
			return fvalarray(start_ptr, actual_size);
		}

		fvalarray& as_alone() noexcept
		{
			if (empty())
			{
				return *this;
			}

			if (!is_owner())
			{
				const usize n = size();
				Element_t* new_mem = memory_alloc<
					Element_t,
					std::max(usize{ alignof(Element_t) }, usize{ 2 })
				>(n);

				const Element_t* src = clean_tag(begin_);
				std::memcpy(new_mem, clean_tag(begin_), sizeof(Element_t) * n);

				reset();
				begin_ = write_tag(new_mem, false);
				end_ = new_mem + n;
			}

			return *this;
		}

	private:
		static constexpr std::uintptr_t tag_mask = 0x1;

		bool is_owner() const noexcept
		{
			return begin_ && !is_external();
		}

		bool is_external() const noexcept
		{
			return begin_ && (reinterpret_cast<std::uintptr_t>(begin_) & tag_mask);
		}

		static Element_t* clean_tag(Element_t* ptr) noexcept
		{
			return reinterpret_cast<Element_t*>(reinterpret_cast<std::uintptr_t>(ptr) & ~tag_mask);
		}

		static Element_t* write_tag(Element_t* ptr, bool is_external) noexcept
		{
			const std::uintptr_t tag = is_external ? tag_mask : 0;
			return reinterpret_cast<Element_t*>(reinterpret_cast<std::uintptr_t>(ptr) | tag);
		}

	private:
		Element_t* begin_;
		Element_t* end_;
	};

	template<typename Element_t>
	class farray
	{
	public:
		farray() noexcept : buffer_begin_(nullptr), buffer_end_(nullptr), data_begin_(nullptr), data_end_(nullptr) {}

		template<usize N>
		explicit farray(const std::array<Element_t, N>& stl_array) noexcept
		{
			const usize init_capacity = N;
			buffer_begin_ = memory_alloc<Element_t, std::max(alignof(Element_t), usize(64))>(init_capacity);
			data_begin_ = buffer_begin_;

			buffer_end_ = buffer_begin_ + init_capacity;
			data_end_ = data_begin_ + init_capacity;

			if constexpr (std::is_trivially_copyable_v<Element_t>)
			{
				std::memcpy(data_begin_, stl_array.data(), init_capacity * sizeof(Element_t));
			}
			else
			{
				for (usize i = 0; i < init_capacity; ++i)
				{
					new (data_begin_ + i) Element_t(stl_array[i]);
				}
			}
		}

		explicit farray(const std::vector<Element_t>& stl_vector) noexcept
		{
			const usize init_capacity = stl_vector.size();
			buffer_begin_ = memory_alloc<Element_t, std::max(alignof(Element_t), usize(64))>(init_capacity);
			data_begin_ = buffer_begin_;

			buffer_end_ = buffer_begin_ + init_capacity;
			data_end_ = data_begin_ + init_capacity;

			if constexpr (std::is_trivially_copyable_v<Element_t>)
			{
				std::memcpy(data_begin_, stl_vector.data(), init_capacity * sizeof(Element_t));
			}
			else
			{
				for (usize i = 0; i < init_capacity; ++i)
				{
					new (data_begin_ + i) Element_t(stl_vector[i]);
				}
			}
		}

		explicit farray(std::initializer_list<Element_t> init_list) noexcept
			: farray(init_list.size())
		{
			
			usize i = 0;
			for (const Element_t& elem : init_list)
			{
				new (data_begin_ + i) Element_t(elem);
				++i;
			}
			data_end_ = data_begin_ + init_list.size();
		}

		explicit farray(const Element_t* const elements, usize count) noexcept
		{
			if (count == 0)
			{
				buffer_begin_ = buffer_end_ = data_begin_ = data_end_ = nullptr;
				return;
			}

			buffer_begin_ = memory_alloc<Element_t, std::max(alignof(Element_t), usize(64))>(count);
			buffer_end_ = buffer_begin_ + count;

			data_begin_ = buffer_begin_;
			data_end_ = buffer_end_;

			if constexpr (std::is_trivially_copyable_v<Element_t>)
			{
				std::memcpy(data_begin_, elements, count * sizeof(Element_t));
			}
			else
			{
				for (usize i = 0; i < count; ++i)
				{
					new (data_begin_ + i) Element_t(elements[i]);
				}
			}
		}

		explicit farray(usize init_capacity) noexcept
		{
			buffer_begin_ = memory_alloc<Element_t, std::max(alignof(Element_t), usize(64))>(init_capacity);
			buffer_end_ = buffer_begin_ + init_capacity;

			data_begin_ = buffer_begin_;
			data_end_ = buffer_end_;
		}

		template<typename array_element_t, usize N> requires(std::is_convertible_v<array_element_t, Element_t>)
		explicit farray(const array_element_t(&array)[N])
		{
			buffer_begin_ = memory_alloc<Element_t, std::max(alignof(Element_t), usize(64))>(N);
			buffer_end_ = buffer_begin_ + N;

			data_begin_ = buffer_begin_;
			data_end_ = buffer_end_;

			if constexpr (std::is_same_v<array_element_t, Element_t> && std::is_trivially_copyable_v<array_element_t>)
			{
				std::memcpy(data_begin_, &array[0], N * sizeof(array_element_t));
			}
			else
			{
				for (usize i = 0; i < N; ++i)
				{
					data_begin_[i] = Element_t(array[i]);
				}
			}
		}

		template<typename... Args>
		explicit farray(usize init_size, Args&&... args) noexcept
		{
			buffer_begin_ = memory_alloc<Element_t, std::max(alignof(Element_t), usize(64))>(init_size);
			buffer_end_ = buffer_begin_ + init_size;

			data_begin_ = buffer_begin_;
			data_end_ = buffer_end_;

			for (usize i = 0; i < init_size; ++i)
			{
				new (data_begin_ + i) Element_t(std::forward<Args>(args)...);
			}
		}

		~farray() noexcept
		{
			destroy_elements();
			memory_free(reinterpret_cast<void*>(buffer_begin_));
		}

		farray(const farray& other) noexcept
			: buffer_begin_(nullptr),
			buffer_end_(nullptr),
			data_begin_(nullptr),
			data_end_(nullptr)
		{
			usize new_capacity = other.capacity();
			buffer_begin_ = memory_alloc<Element_t, std::max(alignof(Element_t), usize(64))>(new_capacity);
			buffer_end_ = buffer_begin_ + new_capacity;

			data_begin_ = buffer_begin_ + (other.data_begin_ - other.buffer_begin_);
			data_end_ = data_begin_ + other.size();

			if constexpr (std::is_trivially_copyable_v<Element_t>)
			{
				std::memcpy(data_begin_, other.data_begin_, other.size() * sizeof(Element_t));
			}
			else
			{
				for (usize i = 0; i < other.size(); ++i)
				{
					new (data_begin_ + i) Element_t(other[i]);
				}
			}
		}

		farray& operator = (const farray& other) noexcept
		{
			if (this == &other)
			{
				return *this;
			}

			const usize other_size = other.size();
			const usize other_capacity = other.capacity();
			const Element_t* other_data = other.data_begin_;

			if (capacity() >= other_size)
			{
				destroy_elements();

				const usize available_capacity = buffer_end_ - buffer_begin_;
				data_begin_ = buffer_begin_ + (available_capacity - other_size) / 2;
				data_end_ = data_begin_ + other_size;

				if constexpr (std::is_trivially_copyable_v<Element_t>)
				{
					std::memcpy(data_begin_, other_data, other_size * sizeof(Element_t));
				}
				else
				{
					for (usize i = 0; i < other_size; ++i)
					{
						new (data_begin_ + i) Element_t(other_data[i]);
					}
				}
			}
			else
			{
				destroy_elements();
				if (buffer_begin_)
				{
					memory_free(buffer_begin_);
				}

				const usize new_capacity = other_capacity + (other_capacity & 1);
				buffer_begin_ = memory_alloc<Element_t, std::max(alignof(Element_t), usize(64))>(new_capacity);
				buffer_end_ = buffer_begin_ + new_capacity;

				data_begin_ = buffer_begin_ + (new_capacity - other_size) / 2;
				data_end_ = data_begin_ + other_size;

				if constexpr (std::is_trivially_copyable_v<Element_t>)
				{
					std::memcpy(data_begin_, other_data, other_size * sizeof(Element_t));
				}
				else
				{
					for (usize i = 0; i < other_size; ++i)
					{
						new (data_begin_ + i) Element_t(other_data[i]);
					}
				}
			}

			return *this;
		}

		farray(farray&& other) noexcept
			: buffer_begin_(other.buffer_begin_),
			buffer_end_(other.buffer_end_),
			data_begin_(other.data_begin_),
			data_end_(other.data_end_)
		{
			other.buffer_begin_ = other.buffer_end_ = other.data_begin_ = other.data_end_ = nullptr;
		}

		farray& operator = (farray&& other) noexcept
		{
			if (this != &other)
			{
				destroy_elements();

				if (buffer_begin_)
				{
					memory_free(reinterpret_cast<void*>(buffer_begin_));
				}

				buffer_begin_ = other.buffer_begin_;
				buffer_end_ = other.buffer_end_;
				data_begin_ = other.data_begin_;
				data_end_ = other.data_end_;

				other.buffer_begin_ = other.buffer_end_ = other.data_begin_ = other.data_end_ = nullptr;
			}
			return *this;
		}

		template<typename Other_t> 
		requires(std::is_convertible_v<Other_t, Element_t> && (!is_basic_arithmetic<Other_t>) && (!is_basic_arithmetic<Other_t>))
		operator farray<Other_t>() const noexcept
		{
			const usize current_size = size();

			farray<Other_t> result(current_size);
			for (usize i = 0; i < current_size; ++i)
			{
				result[i] = static_cast<Other_t>(data_begin_[i]);
			}

			return result;
		}

		operator std::vector<Element_t>() const noexcept
		{
			return std::vector<Element_t>(data_begin_, data_begin_ + size());
		}

		Element_t& operator [] (usize index) noexcept
		{
			return data_begin_[index];
		}

		const Element_t& operator [] (usize index) const noexcept
		{
			return data_begin_[index];
		}

		Element_t& front() noexcept
		{
			return *data_begin_;
		}

		const Element_t& front() const noexcept
		{
			return *data_begin_;
		}

		Element_t& back() noexcept
		{
			return *(data_end_ - 1);
		}

		const Element_t& back() const noexcept
		{
			return *(data_end_ - 1);
		}

		farray subarray(usize index_begin, usize subsize = 0) const noexcept
		{
			const usize length = subsize == 0 ? size() - index_begin : subsize;
			if (index_begin + length > size())
			{
				return farray();
			}
			return farray(data_begin_ + index_begin, length);
		}

		template<typename T> requires std::is_convertible_v<T, Element_t>
		void insert(usize pos, T&& value) noexcept
		{
			if (pos > size())
			{
				return;
			}

			const usize curr_size = size();

			if (curr_size >= capacity())
			{
				memory_expand();
			}

			const usize left_shift_cost = pos;
			const usize right_shift_cost = curr_size - pos;

			if (left_shift_cost <= right_shift_cost)
			{
				--data_begin_;
				if constexpr (std::is_trivially_copyable_v<Element_t>)
				{
					std::memmove(data_begin_, data_begin_ + 1, pos * sizeof(Element_t));
				}
				else
				{
					for (usize i = 0; i < pos; ++i)
					{
						new (data_begin_ + i) Element_t(std::move(*(data_begin_ + i + 1)));
						(data_begin_ + i + 1)->~Element_t();
					}
				}
				new (data_begin_ + pos) Element_t(std::forward<T>(value));
			}
			else
			{
				if constexpr (std::is_trivially_copyable_v<Element_t>)
				{
					std::memmove(data_begin_ + pos + 1, data_begin_ + pos, right_shift_cost * sizeof(Element_t));
				}
				else
				{
					for (usize i = curr_size; i > pos; --i)
					{
						new (data_begin_ + i) Element_t(std::move(*(data_begin_ + i - 1)));
						(data_begin_ + i - 1)->~Element_t();
					}
				}
				new (data_begin_ + pos) Element_t(std::forward<T>(value));
				++data_end_;
			}
		}

		void insert(const farray other, usize insert_begin_index ) noexcept
		{
			return insert(other.data(), other.size(), insert_begin_index );
		}

		void insert(const Element_t* ptr, usize count, usize insert_begin_index ) noexcept
		{
			if (ptr == nullptr || count == 0 || insert_begin_index > size())
			{
				return;
			}

			usize old_size = size();
			usize new_size = old_size + count;

			if (new_size > capacity())
			{
				usize new_capacity = new_size;
				new_capacity += (new_capacity & 1);

				Element_t* new_buffer = memory_alloc<Element_t, std::max(alignof(Element_t), usize(64))>(new_capacity );
				Element_t* new_data_begin = new_buffer + (new_capacity - new_size) / 2;
				Element_t* new_data_end = new_data_begin + new_size;

				if constexpr (std::is_trivially_copyable_v<Element_t>)
				{
					std::memcpy(new_data_begin, data_begin_, insert_begin_index * sizeof(Element_t));
					std::memcpy(new_data_begin + insert_begin_index, ptr, count * sizeof(Element_t));
					std::memcpy(new_data_begin + insert_begin_index + count, data_begin_ + insert_begin_index, (old_size - insert_begin_index) * sizeof(Element_t));
				}
				else
				{
					for (usize i = 0; i < insert_begin_index; ++i)
					{
						new (new_data_begin + i) Element_t(std::move(data_begin_[i]));
						data_begin_[i].~Element_t();
					}
					for (usize i = 0; i < count; ++i)
					{
						new (new_data_begin + insert_begin_index + i) Element_t(ptr[i]);
					}
					for (usize i = insert_begin_index; i < old_size; ++i)
					{
						new (new_data_begin + i + count) Element_t(std::move(data_begin_[i]));
						data_begin_[i].~Element_t();
					}
				}

				memory_free(reinterpret_cast<void*>(buffer_begin_));

				buffer_begin_ = new_buffer;
				buffer_end_ = new_buffer + new_capacity;
				data_begin_ = new_data_begin;
				data_end_ = new_data_end;
			}
			else
			{
				if constexpr (std::is_trivially_copyable_v<Element_t>)
				{
					std::memmove(data_begin_ + insert_begin_index + count, data_begin_ + insert_begin_index, (old_size - insert_begin_index) * sizeof(Element_t));
					std::memcpy(data_begin_ + insert_begin_index, ptr, count * sizeof(Element_t));
				}
				else
				{
					for (usize i = old_size; i > insert_begin_index; --i)
					{
						new (data_begin_ + i + count - 1) Element_t(std::move(data_begin_[i - 1]));
						data_begin_[i - 1].~Element_t();
					}
					for (usize i = 0; i < count; ++i)
					{
						new (data_begin_ + insert_begin_index + i) Element_t(ptr[i]);
					}
				}
				data_end_ += count;
			}
		}

		void erase(usize pos) noexcept
		{
			if (pos >= size())
			{
				return;
			}

			const usize curr_size = size();
			const usize left_shift_cost = pos + 1;
			const usize right_shift_cost = curr_size - pos - 1;

			if (left_shift_cost <= right_shift_cost)
			{
				if constexpr (std::is_trivially_copyable_v<Element_t>)
				{
					std::memmove(data_begin_ + 1, data_begin_, pos * sizeof(Element_t));
				}
				else
				{
					(data_begin_ + pos)->~Element_t();
					for (usize i = pos; i > 0; --i)
					{
						new (data_begin_ + i) Element_t(std::move(*(data_begin_ + i - 1)));
						(data_begin_ + i - 1)->~Element_t();
					}
				}
				++data_begin_;
			}
			else
			{
				if constexpr (std::is_trivially_copyable_v<Element_t>)
				{
					std::memmove(data_begin_ + pos, data_begin_ + pos + 1, right_shift_cost * sizeof(Element_t));
				}
				else
				{
					(data_begin_ + pos)->~Element_t();
					for (usize i = pos; i < curr_size - 1; ++i)
					{
						new (data_begin_ + i) Element_t(std::move(*(data_begin_ + i + 1)));
						(data_begin_ + i + 1)->~Element_t();
					}
				}
				--data_end_;
			}
		}

		void to_appropriate_size(usize extra_capacity = 0 ) noexcept
		{
			if (buffer_begin_)
			{
				const usize current_size = size();
				const usize current_capacity = capacity();
				const usize desired_capacity = current_size + extra_capacity + (extra_capacity & 1);

				if (desired_capacity < current_capacity)
				{
					Element_t* new_buffer = memory_alloc<Element_t, std::max(alignof(Element_t), usize(64))>(desired_capacity );
					Element_t* new_data_begin = new_buffer + (desired_capacity - current_size) / 2;
					Element_t* new_data_end = new_data_begin + current_size;

					if constexpr (std::is_arithmetic_v<Element_t>)
					{
						std::memcpy(new_data_begin, data_begin_, current_size * sizeof(Element_t));
					}
					else
					{
						for (usize i = 0; i < current_size; ++i)
						{
							new (new_data_begin + i) Element_t(std::move(data_begin_[i]));
						}
						destroy_elements();
					}

					memory_free(reinterpret_cast<void*>(buffer_begin_));

					buffer_begin_ = new_buffer;
					buffer_end_ = buffer_begin_ + desired_capacity;
					data_begin_ = new_data_begin;
					data_end_ = new_data_end;
				}
			}
		}

		void clear() noexcept
		{
			if (buffer_begin_)
			{
				if constexpr (!std::is_trivially_destructible_v<Element_t>)
				{
					for (auto it = data_begin_; it != data_end_; ++it)
					{
						it->~Element_t();
					}
				}

				data_begin_ = data_end_ = buffer_begin_ + (capacity() / 2);
			}
		}

		void resize(usize new_size) noexcept
		{
			usize old_size = size();
			if (new_size > old_size)
			{
				if (new_size > capacity())
				{
					reserve(new_size );
				}

				for (usize i = old_size; i < new_size; ++i)
				{
					new (data_end_++) Element_t();
				}
			}
			else if (new_size < old_size)
			{
				for (usize i = new_size; i < old_size; ++i)
				{
					data_begin_[i].~Element_t();
				}
				data_end_ = data_begin_ + new_size;
			}
		}

		void reserve(usize new_capacity) noexcept
		{
			if (new_capacity > capacity())
			{
				new_capacity += (new_capacity & 1);

				Element_t* new_buffer = memory_alloc<Element_t, std::max(alignof(Element_t), usize(64))>(new_capacity );
				Element_t* new_data_begin = new_buffer + (new_capacity - size()) / 2;
				Element_t* new_data_end = new_data_begin + size();

				if (buffer_begin_)
				{
					if constexpr (std::is_arithmetic_v<Element_t>)
					{
						std::memcpy(new_data_begin, data_begin_, size() * sizeof(Element_t));
					}
					else
					{
						for (usize i = 0; i < size(); ++i)
						{
							new (new_data_begin + i) Element_t(std::move(data_begin_[i]));
						}

						destroy_elements();
					}
					
					memory_free(reinterpret_cast<void*>(buffer_begin_));
				}

				buffer_begin_ = new_buffer;
				buffer_end_ = buffer_begin_ + new_capacity;
				data_begin_ = new_data_begin;
				data_end_ = new_data_end;
			}
		}

		void pop_front() noexcept
		{
			if (empty())
			{
				return;
			}

			if constexpr (!std::is_trivially_destructible_v<Element_t>)
			{
				data_begin_->~Element_t();
			}

			++data_begin_;
		}

		void pop_back() noexcept
		{
			if (empty())
			{
				return;
			}

			if constexpr (!std::is_trivially_destructible_v<Element_t>)
			{
				(data_end_ - 1)->~Element_t();
			}

			--data_end_;
		}

		void pop_range(usize pos, usize count) noexcept
		{
			if (pos >= size() || count == 0)
			{
				return;
			}

			if (pos + count > size())
			{
				count = size() - pos;
			}

			const usize old_size = size();
			const usize new_size = old_size - count;

			if constexpr (std::is_trivially_copyable_v<Element_t>)
			{
				std::memmove(data_begin_ + pos, data_begin_ + pos + count, (old_size - pos - count) * sizeof(Element_t));
			}
			else
			{
				for (usize i = pos; i < pos + count; ++i)
				{
					(data_begin_ + i)->~Element_t();
				}

				for (usize i = pos; i < new_size; ++i)
				{
					new (data_begin_ + i) Element_t(std::move(data_begin_[i + count]));
					(data_begin_ + i + count)->~Element_t();
				}
			}

			data_end_ = data_begin_ + new_size;
		}

		void push_back(const farray& other) noexcept
		{
			if (other.empty())
			{
				return;
			}

			usize old_size = size();
			usize new_size = old_size + other.size();

			if (new_size > capacity())
			{
				reserve(new_size );
			}

			if constexpr (std::is_trivially_copyable_v<Element_t>)
			{
				std::memcpy(data_end_, other.data_begin_, other.size() * sizeof(Element_t));
				data_end_ += other.size();
			}
			else
			{
				for (usize i = 0; i < other.size(); ++i)
				{
					new (data_end_++) Element_t(other[i]);
				}
			}
		}

		void push_front(const farray& other) noexcept
		{
			if (other.empty()) return;

			usize old_size = size();
			usize new_size = old_size + other.size();

			if (new_size > capacity())
			{
				reserve(new_size );
			}

			if constexpr (std::is_trivially_copyable_v<Element_t>)
			{
				std::memmove(data_begin_ + other.size(), data_begin_, old_size * sizeof(Element_t));
				std::memcpy(data_begin_, other.data_begin_, other.size() * sizeof(Element_t));
				data_end_ += other.size();
			}
			else
			{
				data_begin_ -= other.size();
				for (usize i = 0; i < other.size(); ++i)
				{
					new (data_begin_ + i) Element_t(other[i]);
				}
			}
		}

		template<typename T> requires std::is_convertible_v<T, Element_t>
		void push_front(T&& value) noexcept
		{
			if ((buffer_begin_ ? data_end_ - buffer_begin_ : 0) == 0)
			{
				memory_expand();
			}

			--data_begin_;
			new (data_begin_) Element_t(std::forward<T>(value));
		}

		template<typename T> requires std::is_convertible_v<T, Element_t>
		void push_back(T&& value) noexcept
		{
			if ((buffer_begin_ ? buffer_end_ - data_end_ : 0) == 0)
			{
				memory_expand();
			}

			new (data_end_) Element_t(std::forward<T>(value));
			++data_end_;
		}

		template<typename T> requires std::is_convertible_v<T, Element_t>
		void push_back(const T* const arr, usize count_elemnt) noexcept
		{
			if (count_elemnt == 0)
			{
				return;
			}

			usize old_size = size();
			usize new_size = old_size + count_elemnt;

			if (new_size > capacity())
			{
				reserve(new_size );
			}

			if constexpr (std::is_trivially_copyable_v<Element_t>)
			{
				std::memcpy(data_end_, arr, count_elemnt * sizeof(Element_t));
				data_end_ += count_elemnt;
			}
			else
			{
				for (usize i = 0; i < count_elemnt; ++i)
				{
					new (data_end_++) Element_t(arr[i]);
				}
			}
		}

		template<typename T> requires std::is_convertible_v<T, Element_t>
		void push_front(const T* const arr, usize count_elemnt ) noexcept
		{
			if (count_elemnt == 0)
			{
				return;
			}

			usize old_size = size();
			usize new_size = old_size + count_elemnt;

			if (new_size > capacity())
			{
				reserve(new_size );
			}

			if constexpr (std::is_trivially_copyable_v<Element_t>)
			{
				std::memmove(data_begin_ + count_elemnt, data_begin_, old_size * sizeof(Element_t));
				std::memcpy(data_begin_, arr, count_elemnt * sizeof(Element_t));
				data_end_ += count_elemnt;
			}
			else
			{
				data_begin_ -= count_elemnt;
				for (usize i = 0; i < count_elemnt; ++i)
				{
					new (data_begin_ + i) Element_t(arr[i]);
				}
			}
		}

		template<typename... Args>
		void emplace_back(Args&&... args) noexcept
		{
			if ((buffer_begin_ ? buffer_end_ - data_end_ : 0) == 0)
			{
				memory_expand();
			}

			new (data_end_) Element_t(std::forward<Args>(args)...);
			++data_end_;
		}

		template<typename... Args>
		void emplace_front(Args&&... args) noexcept
		{
			if ((buffer_begin_ ? data_end_ - buffer_begin_ : 0) == 0)
			{
				memory_expand();
			}

			--data_begin_;
			new (data_begin_) Element_t(std::forward<Args>(args)...);
		}

		void replace_range(const farray& other, usize index, usize count_to_replace ) noexcept
		{
			return replace_range(other.data(), other.size(), index, count_to_replace );
		}

		void replace_range(const Element_t* ptr, usize count_new, usize index, usize count_to_replace ) noexcept
		{
			if (!buffer_begin_ || !ptr || (count_new == 0 && count_to_replace == 0))
			{
				return;
			}
			if (index > size())
			{
				return;
			}

			if (index + count_to_replace > size())
			{
				count_to_replace = size() - index;
			}

			usize old_size = size();
			usize new_size = old_size - count_to_replace + count_new;

			if (new_size > capacity())
			{
				usize new_capacity = new_size;
				new_capacity += (new_capacity & 1);

				Element_t* new_buffer = memory_alloc<Element_t, std::max(alignof(Element_t), usize(64))>(new_capacity );
				Element_t* new_data_begin = new_buffer + (new_capacity - new_size) / 2;
				Element_t* new_data_end = new_data_begin + new_size;

				if constexpr (std::is_trivially_copyable_v<Element_t>)
				{
					std::memcpy(new_data_begin, data_begin_, index * sizeof(Element_t));
					std::memcpy(new_data_begin + index, ptr, count_new * sizeof(Element_t));
					std::memcpy(new_data_begin + index + count_new,
						data_begin_ + index + count_to_replace,
						(old_size - (index + count_to_replace)) * sizeof(Element_t));
				}
				else
				{

					for (usize i = 0; i < index; ++i)
					{
						new (new_data_begin + i) Element_t(std::move(data_begin_[i]));
						data_begin_[i].~Element_t();
					}

					for (usize i = 0; i < count_new; ++i)
					{
						new (new_data_begin + index + i) Element_t(ptr[i]);
					}

					for (usize i = index + count_to_replace; i < old_size; ++i)
					{
						new (new_data_begin + (index + count_new + (i - (index + count_to_replace)))) Element_t(std::move(data_begin_[i]));
						data_begin_[i].~Element_t();
					}
				}

				destroy_elements();
				if (buffer_begin_)
				{
					memory_free(reinterpret_cast<void*>(buffer_begin_));
				}

				buffer_begin_ = new_buffer;
				buffer_end_ = new_buffer + new_capacity;
				data_begin_ = new_data_begin;
				data_end_ = new_data_end;
			}
			else
			{
				if (count_new < count_to_replace)
				{
					if constexpr (std::is_trivially_copyable_v<Element_t>)
					{
						std::memcpy(data_begin_ + index, ptr, count_new * sizeof(Element_t));
					}
					else
					{
						for (usize i = 0; i < count_new; ++i)
						{
							data_begin_[index + i].~Element_t();
							new (data_begin_ + index + i) Element_t(ptr[i]);
						}
					}
					for (usize i = count_new; i < count_to_replace; ++i)
					{
						data_begin_[index + i].~Element_t();
					}

					usize move_count = old_size - (index + count_to_replace);
					if constexpr (std::is_trivially_copyable_v<Element_t>)
					{
						std::memmove(
							data_begin_ + index + count_new,
							data_begin_ + index + count_to_replace,
							move_count * sizeof(Element_t)
						);
					}
					else
					{
						for (usize i = 0; i < move_count; ++i)
						{
							new (data_begin_ + index + count_new + i) Element_t(std::move(data_begin_[index + count_to_replace + i]));
							data_begin_[index + count_to_replace + i].~Element_t();
						}
					}

					data_end_ -= (count_to_replace - count_new);
				}
				else if (count_new > count_to_replace)
				{

					usize move_count = old_size - (index + count_to_replace);
					if constexpr (std::is_trivially_copyable_v<Element_t>)
					{
						std::memmove(
							data_begin_ + index + count_new,
							data_begin_ + index + count_to_replace,
							move_count * sizeof(Element_t)
						);
					}
					else
					{
						for (ssize i = move_count - 1; i >= 0; --i)
						{
							new (data_begin_ + index + count_new + i) Element_t(std::move(data_begin_[index + count_to_replace + i]));
							data_begin_[index + count_to_replace + i].~Element_t();
						}
					}

					if constexpr (std::is_trivially_copyable_v<Element_t>)
					{
						std::memcpy(data_begin_ + index, ptr, count_new * sizeof(Element_t));
					}
					else
					{
						for (usize i = 0; i < count_new; ++i)
						{
							if (i < count_to_replace)
							{
								data_begin_[index + i].~Element_t();
							}

							new (data_begin_ + index + i) Element_t(ptr[i]);
						}
					}

					data_end_ += (count_new - count_to_replace);
				}
				else
				{
					if constexpr (std::is_trivially_copyable_v<Element_t>)
					{
						std::memcpy(data_begin_ + index, ptr, count_new * sizeof(Element_t));
					}
					else
					{
						for (usize i = 0; i < count_to_replace; ++i)
						{
							data_begin_[index + i].~Element_t();
							new (data_begin_ + index + i) Element_t(ptr[i]);
						}
					}

				}
			}
		}

		void push_front(farray&& other) noexcept
		{
			if (this == &other || other.empty())
			{
				return;
			}

			usize old_size = size();
			usize new_size = old_size + other.size();

			if (new_size > capacity())
			{
				reserve(new_size);
			}

			if constexpr (std::is_trivially_copyable_v<Element_t>)
			{
				std::memmove(data_begin_ + other.size(), data_begin_, old_size * sizeof(Element_t));
				std::memcpy(data_begin_, other.data_begin_, other.size() * sizeof(Element_t));
				data_end_ += other.size();
			}
			else
			{
				data_begin_ -= other.size();
				for (usize i = 0; i < other.size(); ++i)
				{
					new (data_begin_ + i) Element_t(std::move(other[i]));
				}
			}

			other.clear();
		}

		void push_back(farray&& other) noexcept
		{
			if (this == &other || other.empty())
			{
				return;
			}

			usize old_size = size();
			usize new_size = old_size + other.size();

			if (new_size > capacity())
			{
				reserve(new_size);
			}

			if constexpr (std::is_trivially_copyable_v<Element_t>)
			{
				std::memcpy(data_end_, other.data_begin_, other.size() * sizeof(Element_t));
				data_end_ += other.size();
			}
			else
			{
				for (usize i = 0; i < other.size(); ++i)
				{
					new (data_end_ + i) Element_t(std::move(other[i]));
				}
				data_end_ += other.size();
			}

			other.clear();
		}

		void push_at(usize index, farray&& other) noexcept
		{
			if (other.empty() || index > size())
			{
				return;
			}

			usize old_size = size();
			usize new_size = old_size + other.size();

			if (new_size > capacity())
			{
				usize new_capacity = new_size;
				new_capacity += (new_capacity & 1);

				Element_t* new_buffer = memory_alloc<Element_t, std::max(alignof(Element_t), usize(64))>(new_capacity );
				Element_t* new_data_begin = new_buffer + (new_capacity - new_size) / 2;
				Element_t* new_data_end = new_data_begin + new_size;

				if constexpr (std::is_trivially_copyable_v<Element_t>)
				{
					std::memcpy(new_data_begin, data_begin_, index * sizeof(Element_t));
					std::memcpy(new_data_begin + index, other.data_begin_, other.size() * sizeof(Element_t));
					std::memcpy(new_data_begin + index + other.size(), data_begin_ + index, (old_size - index) * sizeof(Element_t));
				}
				else
				{
					for (usize i = 0; i < index; ++i)
					{
						new (new_data_begin + i) Element_t(std::move(data_begin_[i]));
						data_begin_[i].~Element_t();
					}
					for (usize i = 0; i < other.size(); ++i)
					{
						new (new_data_begin + index + i) Element_t(std::move(other[i]));
					}
					for (usize i = index; i < old_size; ++i)
					{
						new (new_data_begin + i + other.size()) Element_t(std::move(data_begin_[i]));
						data_begin_[i].~Element_t();
					}
				}

				destroy_elements();
				if (buffer_begin_)
				{
					memory_free(reinterpret_cast<void*>(buffer_begin_));
				}

				buffer_begin_ = new_buffer;
				buffer_end_ = buffer_begin_ + new_capacity;
				data_begin_ = new_data_begin;
				data_end_ = new_data_end;
			}
			else
			{
				if constexpr (std::is_trivially_copyable_v<Element_t>)
				{
					std::memmove(data_begin_ + index + other.size(), data_begin_ + index, (old_size - index) * sizeof(Element_t));
					std::memcpy(data_begin_ + index, other.data_begin_, other.size() * sizeof(Element_t));
				}
				else
				{
					for (usize i = old_size; i > index; --i)
					{
						new (data_begin_ + i + other.size() - 1) Element_t(std::move(data_begin_[i - 1]));
						data_begin_[i - 1].~Element_t();
					}
					for (usize i = 0; i < other.size(); ++i)
					{
						new (data_begin_ + index + i) Element_t(std::move(other[i]));
					}
				}
				data_end_ += other.size();
			}

			other.clear();
		}

		void push_at(usize index, const farray& other ) noexcept
		{
			if (other.empty() || index > size())
			{
				return;
			}

			usize old_size = size();
			usize new_size = old_size + other.size();

			if (new_size > capacity())
			{
				usize new_capacity = new_size;
				new_capacity += (new_capacity & 1);

				Element_t* new_buffer = memory_alloc<Element_t, std::max(alignof(Element_t), usize(64))>(new_capacity );
				Element_t* new_data_begin = new_buffer + (new_capacity - new_size) / 2;
				Element_t* new_data_end = new_data_begin + new_size;

				if constexpr (std::is_trivially_copyable_v<Element_t>)
				{
					std::memcpy(new_data_begin, data_begin_, index * sizeof(Element_t));
					std::memcpy(new_data_begin + index, other.data_begin_, other.size() * sizeof(Element_t));
					std::memcpy(new_data_begin + index + other.size(), data_begin_ + index, (old_size - index) * sizeof(Element_t));
				}
				else
				{
					for (usize i = 0; i < index; ++i)
					{
						new (new_data_begin + i) Element_t(std::move(data_begin_[i]));
						data_begin_[i].~Element_t();
					}
					for (usize i = 0; i < other.size(); ++i)
					{
						new (new_data_begin + index + i) Element_t(other[i]);
					}
					for (usize i = index; i < old_size; ++i)
					{
						new (new_data_begin + i + other.size()) Element_t(std::move(data_begin_[i]));
						data_begin_[i].~Element_t();
					}
				}

				memory_free(reinterpret_cast<void*>(buffer_begin_));

				buffer_begin_ = new_buffer;
				buffer_end_ = buffer_begin_ + new_capacity;
				data_begin_ = new_data_begin;
				data_end_ = new_data_end;
			}
			else
			{
				if constexpr (std::is_trivially_copyable_v<Element_t>)
				{
					std::memmove(data_begin_ + index + other.size(), data_begin_ + index, (old_size - index) * sizeof(Element_t));
					std::memcpy(data_begin_ + index, other.data_begin_, other.size() * sizeof(Element_t));
				}
				else
				{
					for (usize i = old_size; i > index; --i)
					{
						new (data_begin_ + i + other.size() - 1) Element_t(std::move(data_begin_[i - 1]));
						data_begin_[i - 1].~Element_t();
					}
					for (usize i = 0; i < other.size(); ++i)
					{
						new (data_begin_ + index + i) Element_t(other[i]);
					}
				}
				data_end_ += other.size();
			}
		}
		
		usize capacity() const noexcept
		{
			return capacity_front() + capacity_back();
		}

		usize capacity_front() const noexcept
		{
			return buffer_begin_ ? data_end_ - buffer_begin_ : 0;
		}

		usize capacity_back() const noexcept
		{
			return buffer_begin_ ? buffer_end_ - data_end_ : 0;
		}

		usize size() const noexcept
		{
			return buffer_begin_ ? data_end_ - data_begin_ : 0;
		}

		bool empty() const noexcept
		{
			return size() == 0;
		}

		const Element_t* data() const noexcept
		{
			return reinterpret_cast<const Element_t*>(data_begin_);
		}

		Element_t* data() noexcept
		{
			return reinterpret_cast<Element_t*>(data_begin_);
		}

		Element_t* begin() noexcept
		{
			return data_begin_;
		}

		Element_t* end() noexcept
		{
			return data_end_;
		}

		Element_t* begin() const noexcept
		{
			return data_begin_;
		}

		const Element_t* end() const noexcept
		{
			return data_end_;
		}

		const Element_t* cbegin() const noexcept
		{
			return data_begin_;
		}

		const Element_t* cend() const noexcept
		{
			return data_end_;
		}

		template<typename ... Args>
		void broadcast(Args&& ... args) noexcept
		{
			for (Element_t* it = data_begin_; it != data_end_; ++it)
			{
				*it = Element_t(std::forward<Args>(args)...);
			}
		}

		template<typename Input_t>
		usize count(const Input_t& to_repeat) const noexcept
		{
			static_assert(requires(const Element_t& a, const Input_t& b) { { a == b } -> std::convertible_to<bool>; },
				"Element_t and Other_t must support operator==.");

			usize count_{ 0 };
			for (const Element_t& elm : *this)
			{
				if (elm == static_cast<Element_t>(to_repeat))
				{
					++count_;
				}
			}

			return count_;
		}

		template<typename Expression> requires std::is_invocable_v<Expression, Element_t&>
		void for_each(Expression&& expr) noexcept
		{
			for (Element_t* it = data_begin_; it != data_end_; ++it)
			{
				std::invoke(expr, *it);
			}
		}

		template<typename Expression> requires std::is_invocable_v<Expression, const Element_t&>
		void for_each(Expression&& expr) const noexcept
		{
			for (const Element_t* it = data_begin_; it != data_end_; ++it)
			{
				std::invoke(expr, *it);
			}
		}

		farray& swap(farray& other) noexcept
		{
			std::swap(buffer_begin_, other.buffer_begin_);
			std::swap(buffer_end_, other.buffer_end_);
			std::swap(data_begin_, other.data_begin_);
			std::swap(data_end_, other.data_end_);

			return *this;
		}

		template<typename Other_t>
		bool operator == (const Other_t& other) const noexcept
		{
			if (empty())
			{
				return false;
			}

			static_assert(requires(const Element_t & a, const Other_t & b) { { a == b } -> std::convertible_to<bool>; },
				"Element_t and Other_t must support operator==.");

			for (const Element_t& v : *this)
			{
				if (v == other)
				{
					continue;
				}
				else
				{
					return false;
				}
			}

			return true;
		}

		template<typename Other_t>
		bool operator == (const farray<Other_t>& other) const noexcept
		{
			if (size() != other.size() || empty() || other.empty())
			{
				return false;
			}
			if constexpr (std::is_arithmetic_v<Other_t> && std::is_same_v<Other_t, Element_t>)
			{
				return std::memcmp(data_begin_, other.data_begin_, size() * sizeof(Other_t)) == 0;
			}
			else
			{
				static_assert(requires(const Element_t& a, const Other_t& b) { { a == b } -> std::convertible_to<bool>; },
					"Element_t and Other_t must support operator==.");

				const usize count_element = size();
				for (usize i = 0; i < count_element; ++i)
				{
					if (this->operator[](i) == other[i])
					{
						continue;
					}
					else
					{
						return false;
					}
				}

				return true;
			}
		}

		usize memory_size__() const noexcept
		{
			return buffer_end_ - buffer_begin_;
		}

		Element_t* expand_at__(usize index, usize expand_size) noexcept
		{
			if (expand_size == 0 || index > size())
			{
				return nullptr;
			}

			usize old_size = size();
			usize new_size = old_size + expand_size;

			if (new_size > capacity())
			{
				usize new_capacity = capacity() + expand_size;
				new_capacity += (new_capacity & 1);

				Element_t* new_buffer = memory_alloc<Element_t, std::max(alignof(Element_t), usize(64))>(new_capacity );
				Element_t* new_data_begin = new_buffer + (new_capacity - new_size) / 2;
				Element_t* new_data_end = new_data_begin + new_size;

				if constexpr (std::is_trivially_copyable_v<Element_t>)
				{
					std::memcpy(new_data_begin, data_begin_, index * sizeof(Element_t));
					std::memcpy(new_data_begin + index + expand_size, data_begin_ + index, (old_size - index) * sizeof(Element_t));
				}
				else
				{
					for (usize i = 0; i < index; ++i)
					{
						new (new_data_begin + i) Element_t(std::move(data_begin_[i]));
						data_begin_[i].~Element_t();
					}
					for (usize i = index; i < old_size; ++i)
					{
						new (new_data_begin + i + expand_size) Element_t(std::move(data_begin_[i]));
						data_begin_[i].~Element_t();
					}
				}

				memory_free(reinterpret_cast<void*>(buffer_begin_));

				buffer_begin_ = new_buffer;
				buffer_end_ = new_buffer + new_capacity;
				data_begin_ = new_data_begin;
				data_end_ = new_data_end;
			}
			else
			{
				if constexpr (std::is_trivially_copyable_v<Element_t>)
				{
					std::memmove(data_begin_ + index + expand_size, data_begin_ + index, (old_size - index) * sizeof(Element_t));
				}
				else
				{
					for (usize i = old_size; i > index; --i)
					{
						new (data_begin_ + i + expand_size - 1) Element_t(std::move(data_begin_[i - 1]));
						data_begin_[i - 1].~Element_t();
					}
				}
				data_end_ += expand_size;
			}

			return data_begin_ + index;
		}

		Element_t* expand_back__(usize expand_size) noexcept
		{
			if (expand_size == 0)
			{
				return data_end_;
			}
			
			usize old_size = size();
			usize new_size = old_size + expand_size;

			usize new_capacity = capacity() + expand_size;

			if (new_capacity > capacity())
			{
				reserve(new_capacity );
			}

			Element_t* old_data_end = data_end_;
			data_end_ += expand_size;

			return old_data_end;
		}

		Element_t* expand_front__(usize expand_size) noexcept
		{
			if (expand_size == 0)
			{
				return data_begin_;
			}

			usize old_size = size();
			usize new_size = old_size + expand_size;

			usize new_capacity = capacity() + expand_size;

			if (new_capacity > capacity())
			{
				reserve(new_capacity );
			}

			data_begin_ -= expand_size;

			return data_begin_;
		}

		Element_t* raw_buffer_begin__() noexcept { return buffer_begin_; }
		Element_t* raw_buffer_end__() noexcept { return buffer_end_; }
		Element_t* raw_data_begin__() noexcept { return data_begin_; }
		Element_t* raw_data_end__() noexcept { return data_end_; }

		const Element_t* raw_buffer_begin__() const noexcept { return reinterpret_cast<const Element_t*>(buffer_begin_); }
		const Element_t* raw_buffer_end__() const noexcept { return reinterpret_cast<const Element_t*>(buffer_end_); }
		const Element_t* raw_data_begin__() const noexcept { return reinterpret_cast<const Element_t*>(data_begin_); }
		const Element_t* raw_data_end__() const noexcept { return reinterpret_cast<const Element_t*>(data_end_); }

	private:
		void memory_expand() noexcept
		{
			const usize old_size = size();
			const usize old_capacity = buffer_end_ - buffer_begin_;
			const usize new_capacity = old_capacity == 0 ? default_init : static_cast<usize>(old_capacity * factory);

			const usize adjusted_capacity = new_capacity + (new_capacity & 1);

			Element_t* new_buffer = memory_alloc<Element_t, std::max(alignof(Element_t), usize(64))>(adjusted_capacity);
			Element_t* new_data_begin = new_buffer + (adjusted_capacity - old_size) / 2;
			Element_t* new_data_end = new_data_begin + old_size;

			if (buffer_begin_ && old_size > 0)
			{
				if constexpr (std::is_arithmetic_v<Element_t>)
				{
					std::memcpy(new_data_begin, data_begin_, old_size * sizeof(Element_t));
				}
				else if constexpr (std::is_nothrow_move_constructible_v<Element_t>)
				{
					for (usize i = 0; i < old_size; ++i)
					{
						new (new_data_begin + i) Element_t(std::move(data_begin_[i]));
					}
				}
				else
				{
					for (usize i = 0; i < old_size; ++i)
					{
						new (new_data_begin + i) Element_t(data_begin_[i]);
					}
				}

				destroy_elements();
				memory_free(reinterpret_cast<void*>(buffer_begin_));
			}

			buffer_begin_ = new_buffer;
			buffer_end_ = new_buffer + adjusted_capacity;
			data_begin_ = new_data_begin;
			data_end_ = new_data_end;
		}

		void destroy_elements() noexcept
		{
			if constexpr (!std::is_trivially_destructible_v<Element_t>)
			{
				for (Element_t* it = data_begin_; it != data_end_; ++it)
				{
					it->~Element_t();
				}
			}
		}

		Element_t* buffer_begin_;
		Element_t* buffer_end_;
		Element_t* data_begin_;
		Element_t* data_end_;

		static constexpr usize default_init = 4;
		static constexpr f64 factory = 2.0;
	};
}


namespace fy
{
	template<BasicArithmetic Element_t>
	fvalarray<Element_t>::fvalarray(farray<Element_t>& farr) noexcept
		: fvalarray<Element_t>(farr.begin(), farr.end()) { }

	template<BasicArithmetic Element_t>
	fvalarray<Element_t>::fvalarray(std::vector<Element_t>& farr) noexcept
		: fvalarray<Element_t>(farr.begin(), farr.end()) { }
}