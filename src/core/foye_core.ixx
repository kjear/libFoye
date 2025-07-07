module;
#include <Windows.h>

#undef min
#undef max

#include <malloc.h>
#include <immintrin.h>

//#define _TEST_

#ifdef _TEST_
#pragma warning(error: 4834)
#define FY_NODISCARD [[nodiscard]]
#else
#define FY_NODISCARD
#endif

export module foye.foye_core;
export import foye.extensionType.float16;
export import foye.extensionType.bfloat16;
export import foye.extensionType.float8;
export import foye.extensionType.int128;
export import foye.type_characteristics;
export import foye.alias;
import std;

#pragma warning(disable: 4309)
#pragma warning(disable: 4244)
#pragma warning(disable: 4552)
#pragma warning(disable: 4552)
#pragma warning(disable: 4018)

#ifdef _TEST_
export namespace fy
{
	inline constexpr bool is_debug_mode = true;
}
#else
export namespace fy
{
	inline constexpr bool is_debug_mode = false;
}
#endif

export namespace fy
{
	class fallocator
	{
	public:
#ifdef _TEST_
		struct alloc_info
		{
			const char* file_name;
			const char* funtion_name;
			const char* alloc_type;
			usize line;
			usize column;
			usize count_bytes;
			usize num;
			usize aligned;

			const char* release_file_name = nullptr;
			const char* release_funtion_name = nullptr;
			usize release_line = 0;
			usize release_column = 0;
			bool is_released = false;
		};
#endif
		template<typename Element_t, usize aligned_size = std::max(usize(64), alignof(Element_t))>
		__declspec(allocator) FY_NODISCARD static Element_t* alloc(usize num_elements
#ifdef _TEST_
			, const std::source_location location
#endif
		)
		{
			const usize element_size = std::max(num_elements * sizeof(Element_t), aligned_size);
#ifdef _TEST_
			const usize total_size = element_size + sizeof(usize);
			usize* memory = reinterpret_cast<usize*>(_aligned_malloc(total_size, aligned_size));
			*memory = element_size;
			allocated_size += element_size;
			total_allocated += element_size;
			Element_t* raw_memory = reinterpret_cast<Element_t*>(memory + 1);

			alloc_info info{
				location.file_name(),
				location.function_name(),
				typeid(Element_t).name(),
				location.line(),
				location.column(),
				num_elements * sizeof(Element_t),
				num_elements,
				aligned_size
			};

			alloc_position[raw_memory] = info;
#else
			Element_t* raw_memory = reinterpret_cast<Element_t*>(_aligned_malloc(element_size, aligned_size));
#endif
			return raw_memory;
		}

		static void release(void* memory
#ifdef _TEST_
			, const std::source_location location = std::source_location::current()
#endif
		)
		{
			if (!memory) return;
#ifdef _TEST_
			auto it = alloc_position.find(memory);
			if (it != alloc_position.end())
			{
				it->second.release_file_name = location.file_name();
				it->second.release_funtion_name = location.function_name();
				it->second.release_line = location.line();
				it->second.release_column = location.column();
				it->second.is_released = true;

				static std::atomic<usize> counter{ 1 };
				void* new_key = reinterpret_cast<void*>(counter++);

				alloc_position[new_key] = it->second;
				alloc_position.erase(it);

				usize* real_memory = reinterpret_cast<usize*>(memory) - 1;
				allocated_size -= *real_memory;
				_aligned_free(real_memory);
			}
#else
			_aligned_free(memory);
#endif
		}

#ifdef _TEST_
		static inline usize allocated_size = 0;
		static inline usize total_allocated = 0;
		static inline std::unordered_map<void*, alloc_info> alloc_position;
#endif
	};

	FY_NODISCARD usize current_surviving()
	{
#ifdef _TEST_
		return fallocator::allocated_size;
#else
		return usize{};
#endif
	}

	FY_NODISCARD usize current_allocated()
	{
#ifdef _TEST_
		return fallocator::total_allocated;
#else
		return usize{};
#endif
	}

	void write_allocated_info(const char* file_path)
	{
		std::ofstream file(file_path, std::ios::out | std::ios::trunc);

		if (!file.is_open())
		{
			std::cerr << "Error: Could not open or create file: " << file_path << std::endl;
			return;
		}
#ifdef _TEST_
		file << "Current surviving: " << fallocator::allocated_size << " bytes\n";
		file << "Current allocated: " << fallocator::total_allocated << " bytes\n";

		if (fallocator::allocated_size > 0)
		{
			file << "\n\nWarning! Please pay attention to the recorded unreleased memorys\n";
			file << "Warning! Please pay attention to the recorded unreleased memorys\n";
			file << "Warning! Please pay attention to the recorded unreleased memorys\n";
		}

		file << "\n\n----------------------------------------------------\n\n";

		for (const auto& [ptr, info] : fallocator::alloc_position)
		{
			file
				<< "Allocation Info:\n"
				<< "	File: " << info.file_name << "\n"
				<< "	Function: " << info.funtion_name << "\n"
				<< "	Line: " << info.line << "\n"
				<< "	Column: " << info.column << "\n"
				<< "	Address: " << ptr << "\n"
				<< "	type: " << info.alloc_type << "\n"
				<< "	Num of element: " << info.num << "\n"
				<< "	Bytes: " << info.count_bytes << "\n"
				<< "	Aligned: " << info.aligned << "\n";
			if (info.is_released)
			{
				file
					<< "Release Info:\n"
					<< "	File: " << info.release_file_name << "\n"
					<< "	Function: " << info.release_funtion_name << "\n"
					<< "	Line: " << info.release_line << "\n"
					<< "	Column: " << info.release_column << "\n\n";
			}
			else
			{
				file << "	Not released yet\n\n";
			}
		}
#else
		file << "Not available in release mode.\n";
#endif
		file.close();
	}


	template<typename T, usize aligned_size = std::max(alignof(T), usize(64))>
	FY_NODISCARD T* memory_alloc(usize num_elements = 1
#ifdef _TEST_
		, const std::source_location location = std::source_location::current()
#endif
	)
	{
#ifdef _TEST_
		static bool flag = false;
		if (!flag)
		{
			std::atexit(
				[ ]() -> void
				{
					write_allocated_info("memory_track.txt");
				}
			);

			flag = true;
		}
#endif
		return fallocator::alloc<T, aligned_size>(num_elements
#ifdef _TEST_
			, location
#endif
		);
	}

	void memory_free(void* memory
#ifdef _TEST_
		, const std::source_location location = std::source_location::current()
#endif
	)
	{
		fallocator::release(memory
#ifdef _TEST_
			, location
#endif
		);
	}

}
