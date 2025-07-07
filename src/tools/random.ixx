export module foye.random;

import foye.foye_core;
import foye.farray;
import std;

export namespace fy
{
    class rand_LCG
    {
        static constexpr u32 MULTIPLIER = 4164903690U;
        static constexpr f32 FLOAT_MULTIPLIER = 2.3283064365386962890625e-10f;
        static constexpr f64 DOUBLE_MULTIPLIER = 5.4210108624275221700372640043497e-20;

        u64 state;

    public:
        rand_LCG() noexcept : state(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count()) {}
        explicit rand_LCG(u64 _state) noexcept : state(_state ? _state : std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count()) {}

        bool operator == (const rand_LCG& other) const
        {
            return state == other.state;
        }

        u32 next() noexcept
        {
            state = static_cast<u64>(static_cast<u32>(state)) * MULTIPLIER + static_cast<u32>(state >> 32);
            return static_cast<u32>(state);
        }

        i32 uniform(i32 a, i32 b)
        {
            return a == b ? a : static_cast<i32>(next() % static_cast<u32>(b - a)) + a;
        }

        f32 uniform(f32 a, f32 b)
        {
            return static_cast<f32>(*this) * (b - a) + a;
        }

        f64 uniform(f64 a, f64 b)
        {
            return static_cast<f64>(*this) * (b - a) + a;
        }

        template<BasicArithmetic Dst_t> explicit operator Dst_t() noexcept
        {
            if constexpr (std::is_integral_v<Dst_t>)
            {
                if constexpr (sizeof(Dst_t) <= sizeof(u32))
                {
                    return static_cast<Dst_t>(next());
                }
                else
                {
                    if constexpr (std::is_unsigned_v<Dst_t>)
                    {
                        return (static_cast<u64>(next()) << 32) | next();
                    }
                    else
                    {
                        return static_cast<i64>((static_cast<u64>(next()) << 32) | next());
                    }
                }
            }
            else
            {
                if constexpr (std::is_same_v<Dst_t, f32>)
                {
                    return static_cast<f32>(next()) * FLOAT_MULTIPLIER;
                }
                else if constexpr (std::is_same_v<Dst_t, f64>)
                {
                    u32 t = next();
                    return (((static_cast<u64>(t) << 32) | next()) * DOUBLE_MULTIPLIER);
                }
            }
        }
    };

    class rand_MT19937
    {
    public:
        rand_MT19937() noexcept { seed(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count()); }
        rand_MT19937(u32 seed_) noexcept { seed(seed_); }

        void seed(u32 seed_) noexcept
        {
            state[0] = seed_;
            for (mti = 1; mti < N; mti++)
            {
                state[mti] = (1812433253U * (state[mti - 1] ^ (state[mti - 1] >> 30)) + mti);
            }
        }

        u32 next() noexcept
        {
            static u32 mag01[2] = { 0x0U, 0x9908b0dfU };

            const u32 UPPER_MASK = 0x80000000U;
            const u32 LOWER_MASK = 0x7fffffffU;

            if (mti >= N)
            {
                i32 kk = 0;
                for (; kk < N - M; ++kk)
                {
                    u32 y = (state[kk] & UPPER_MASK) | (state[kk + 1] & LOWER_MASK);
                    state[kk] = state[kk + M] ^ (y >> 1) ^ mag01[y & 0x1U];
                }

                for (; kk < N - 1; ++kk)
                {
                    u32 y = (state[kk] & UPPER_MASK) | (state[kk + 1] & LOWER_MASK);
                    state[kk] = state[kk + (M - N)] ^ (y >> 1) ^ mag01[y & 0x1U];
                }

                u32 y = (state[N - 1] & UPPER_MASK) | (state[0] & LOWER_MASK);
                state[N - 1] = state[M - 1] ^ (y >> 1) ^ mag01[y & 0x1U];

                mti = 0;
            }

            u32 y = state[mti++];

            y ^= (y >> 11);
            y ^= (y << 7) & 0x9d2c5680U;
            y ^= (y << 15) & 0xefc60000U;
            y ^= (y >> 18);

            return y;
        }

        i32 uniform(i32 a, i32 b) noexcept { return static_cast<i32>(next() % (b - a) + a); }

        f32 uniform(f32 a, f32 b) noexcept { return (static_cast<f32>(*this)) * (b - a) + a; }

        f64 uniform(f64 a, f64 b) noexcept { return (static_cast<f64>(*this)) * (b - a) + a; }

        operator i32() noexcept { return static_cast<i32>(next()); }
        operator u32() noexcept { return next(); }

        operator f32() noexcept { return next() * (1.f / 4294967296.f); }
        operator f64() noexcept
        {
            u32 a = next() >> 5;
            u32 b = next() >> 6;
            return (a * 67108864.0 + b) * (1.0 / 9007199254740992.0);
        }

    protected:
        enum PeriodParameters { N = 624, M = 397 };
        u32 state[N];
        i32 mti;
    };


}