export module foye;

export import foye.foye_core;
export import foye.time;

export import foye.algorithm;

export import foye.farray;
export import foye.fbytearray;
export import foye.flist;

export import foye.fstring_view;
export import foye.farray_view;

export import foye.fstring;
export import foye.fprint;

export import foye.extensionType.bfloat16;
export import foye.extensionType.float8;
export import foye.extensionType.float16;
export import foye.extensionType.int128;

export import foye.random;


export namespace fy
{
	template<usize rounds, typename Expr_0, typename Expr_1>
	std::pair<f64, f64> bench_mark(Expr_0&& expr_0, Expr_1&& expr_1)
	{
      /*  for (volatile usize r = 0; r < 10; ++r)
        {
            expr_0();
            expr_1();
        }*/

        f64 cost_0{ 0. };
        f64 cost_1{ 0. };
        {
            Timer timer_0{};
            Timer timer_1{};

            for (volatile usize r = 0; r < rounds; ++r)
            {

                timer_0.begin();
                expr_0();
                cost_0 += timer_0.end();

                timer_1.begin();
                expr_1();
                cost_1 += timer_1.end();

            }

            cost_0 /= rounds;
            cost_1 /= rounds;
        }

        return std::make_pair(cost_0, cost_1);
	}
}