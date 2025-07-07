module;
#include <Windows.h>
#include <immintrin.h>
#undef min
#undef max

export module foye.time;

export import foye.foye_core;
import std;

#pragma warning(disable: 4309)
#pragma warning(disable: 4244)
#pragma warning(disable: 4552)
#pragma warning(disable: 4552)
#pragma warning(disable: 4018)

export namespace fy
{
    class Timer
    {
    private:
        LARGE_INTEGER start_time;
        LARGE_INTEGER frequency;
        bool is_running;

    public:
        Timer() : is_running(false)
        {
            QueryPerformanceFrequency(&frequency);
        }

        void begin()
        {
            QueryPerformanceCounter(&start_time);
            is_running = true;
        }

        double end()
        {
            if (!is_running)
            {
                return 0.0;
            }

            LARGE_INTEGER end_time;
            QueryPerformanceCounter(&end_time);

            LONGLONG elapsed = end_time.QuadPart - start_time.QuadPart;

            double elapsed_ms = (elapsed * 1000.0) / frequency.QuadPart;

            is_running = false;
            return elapsed_ms;
        }
    };

	void wait(usize milliseconds)
	{
        LARGE_INTEGER frequency;
        if (!QueryPerformanceFrequency(&frequency) || frequency.QuadPart == 0)
            return;

        const ssize original_priority = GetThreadPriority(GetCurrentThread());
        SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_TIME_CRITICAL);

        const DWORD_PTR original_affinity = SetThreadAffinityMask(GetCurrentThread(), 1);

        LARGE_INTEGER start;
        QueryPerformanceCounter(&start);
        const usize target = start.QuadPart +
            (milliseconds * frequency.QuadPart) / 1000LL;

        LARGE_INTEGER current;
        do
        {
            QueryPerformanceCounter(&current);

            if (current.QuadPart < start.QuadPart)
            {
                break;
            }
        } while (current.QuadPart < target);

        SetThreadAffinityMask(GetCurrentThread(), original_affinity);
        SetThreadPriority(GetCurrentThread(), original_priority);

        if (current.QuadPart < target) 
        {
            const usize remaining = target - current.QuadPart;
            wait(static_cast<ssize>(remaining * 1000 / frequency.QuadPart));
        }
        
	}



}