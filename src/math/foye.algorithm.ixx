export module foye.algorithm;

export import foye.foye_core;
export import foye.farray;

export namespace fy
{
    template<BasicArithmetic Element_t>
    void addition(const Element_t* left, const Element_t* right, Element_t* dst, usize length) noexcept;
    
    template<BasicArithmetic Element_t>
    void subtraction(const Element_t* left, const Element_t* right, Element_t* dst, usize length) noexcept;

    template<BasicArithmetic Element_t>
    void multiplication(const Element_t* left, const Element_t* right, Element_t* dst, usize length) noexcept;

    template<BasicArithmetic Element_t>
    void division(const Element_t* left, const Element_t* right, Element_t* dst, usize length) noexcept;

	template<BasicArithmetic Element_t, BasicArithmetic Right>
    void addition(const Element_t* left, Right right, Element_t* dst, usize length) noexcept;

	template<BasicArithmetic Element_t, BasicArithmetic Right>
    void subtraction(const Element_t* left, Right right, Element_t* dst, usize length) noexcept;

	template<BasicArithmetic Element_t, BasicArithmetic Right>
    void multiplication(const Element_t* left, Right right, Element_t* dst, usize length) noexcept;

	template<BasicArithmetic Element_t, BasicArithmetic Right>
    void division(const Element_t* left, Right right, Element_t* dst, usize length) noexcept;

    template<BasicArithmetic Element_t>
    void bit_and(const Element_t* left, const Element_t* right, Element_t* dst, usize length) noexcept;

    template<BasicArithmetic Element_t>
    void bit_or(const Element_t* left, const Element_t* right, Element_t* dst, usize length) noexcept;

    template<BasicArithmetic Element_t>
    void bit_xor(const Element_t* left, const Element_t* right, Element_t* dst, usize length) noexcept;

    template<BasicArithmetic Element_t>
    void remainder(const Element_t* left, const Element_t* right, Element_t* dst, usize length) noexcept;

	template<BasicArithmetic Element_t, BasicArithmetic Right>
    void bit_and(const Element_t* left, Right right, Element_t* dst, usize length) noexcept;

	template<BasicArithmetic Element_t, BasicArithmetic Right>
    void bit_or(const Element_t* left, Right right, Element_t* dst, usize length) noexcept;

	template<BasicArithmetic Element_t, BasicArithmetic Right>
    void bit_xor(const Element_t* left, Right right, Element_t* dst, usize length) noexcept;

	template<BasicArithmetic Element_t, BasicArithmetic Right>
	void remainder(const Element_t* left, Right right, Element_t* dst, usize length) noexcept;



	template<BasicArithmetic Element_t> 
	Element_t min_what(const Element_t* ptr, usize count) noexcept;

	template<BasicArithmetic Element_t> 
	Element_t max_what(const Element_t* ptr, usize count) noexcept;

	template<BasicArithmetic Element_t> 
	void minmax_what(const Element_t* ptr, usize count, Element_t& minval, Element_t& maxval) noexcept;



    template<Floating_arithmetic Element_t>
    void sine(const Element_t* input, Element_t* output, usize count) noexcept;

    template<Floating_arithmetic Element_t>
    void cosine(const Element_t* input, Element_t* output, usize count) noexcept;

    template<Floating_arithmetic Element_t>
    void tangent(const Element_t* input, Element_t* output, usize count) noexcept;

    template<Floating_arithmetic Element_t>
    void arctangent(const Element_t* input, Element_t* output, usize count) noexcept;

    template<Floating_arithmetic Element_t>
    void arccosine(const Element_t* input, Element_t* output, usize count) noexcept;

    template<Floating_arithmetic Element_t>
    void arcsine(const Element_t* input, Element_t* output, usize count) noexcept;

    template<Floating_arithmetic Element_t>
    void floor(const Element_t* input, Element_t* output, usize count) noexcept;

    template<Floating_arithmetic Element_t>
    void ceil(const Element_t* input, Element_t* output, usize count) noexcept;

    template<Floating_arithmetic Element_t>
    void round(const Element_t* input, Element_t* output, usize count) noexcept;

    template<Floating_arithmetic Element_t>
    void trunc(const Element_t* input, Element_t* output, usize count) noexcept;


    /* 数组操作 */
    template<BasicArithmetic Element_t>
    void setall(Element_t* ptr, Element_t toset, usize count) noexcept;

    //template<BasicArithmetic Element_t>
    //void chunky_to_plane(const Element_t* chunky_src, Element_t* plane_dst, usize total, usize ndims) noexcept;

    template<BasicArithmetic Src_t, BasicArithmetic Dst_t>
    void cast_towardszero(const Src_t* src_ptr, Dst_t* dst_ptr, usize count) noexcept;

    template<BasicArithmetic Element_t>
    void clamp(const Element_t* src_ptr, Element_t* dst_ptr, usize count, Element_t minVal, Element_t maxVal) noexcept;

    //template<BasicArithmetic To, BasicArithmetic From>
    //void convert_saturation(const From* from, To* to, usize count) noexcept;

    //template<BasicArithmetic To, BasicArithmetic From>
    //void convert_scale(const From* from, To* to, usize count) noexcept;


    template<BasicArithmetic To, BasicArithmetic From>
    void convert(const From* from, To* to, usize count) noexcept;

    template<BasicArithmetic To, BasicArithmetic From>
    void convert_saturation(const From* from, To* to, usize count) noexcept;



    template<BasicArithmetic Element_t> inline Element_t mask_equal = ~Element_t(0);
    template<BasicArithmetic Element_t> inline Element_t mask_nonequal = Element_t(0);

    template<> inline f16 mask_equal<f16> = f16::hfloatFromBits(~u16{ 0 });
    template<> inline f32 mask_equal<f32> = std::bit_cast<f32>(~u32{ 0 });
    template<> inline f64 mask_equal<f64> = std::bit_cast<f64>(~u64{ 0 });

    template<> inline f16 mask_nonequal<f16> = f16::hfloatFromBits(u16{ 0 });
    template<> inline f32 mask_nonequal<f32> = std::bit_cast<f32>(u32{ 0 });
    template<> inline f64 mask_nonequal<f64> = std::bit_cast<f64>(u64{ 0 });

    template<BasicArithmetic Element_t>
    void equal(const Element_t* cmp_0, const Element_t* cmp_1, Element_t* res_mask, usize count, Element_t mask_equal_val = mask_equal<Element_t>, Element_t mask_nonequal_val = mask_nonequal<Element_t>) noexcept;

    template<BasicArithmetic Element_t>
    void not_equal(const Element_t* cmp_0, const Element_t* cmp_1, Element_t* res_mask, usize count, Element_t mask_equal_val = mask_equal<Element_t>, Element_t mask_nonequal_val = mask_nonequal<Element_t>) noexcept;

    template<BasicArithmetic Element_t>
    void less(const Element_t* cmp_0, const Element_t* cmp_1, Element_t* res_mask, usize count, Element_t mask_equal_val = mask_equal<Element_t>, Element_t mask_nonequal_val = mask_nonequal<Element_t>) noexcept;

    template<BasicArithmetic Element_t>
    void less_equal(const Element_t* cmp_0, const Element_t* cmp_1, Element_t* res_mask, usize count, Element_t mask_equal_val = mask_equal<Element_t>, Element_t mask_nonequal_val = mask_nonequal<Element_t>) noexcept;

    template<BasicArithmetic Element_t>
    void greater(const Element_t* cmp_0, const Element_t* cmp_1, Element_t* res_mask, usize count, Element_t mask_equal_val = mask_equal<Element_t>, Element_t mask_nonequal_val = mask_nonequal<Element_t>) noexcept;

    template<BasicArithmetic Element_t>
    void greater_equal(const Element_t* cmp_0, const Element_t* cmp_1, Element_t* res_mask, usize count, Element_t mask_equal_val = mask_equal<Element_t>, Element_t mask_nonequal_val = mask_nonequal<Element_t>) noexcept;


    template<BasicArithmetic Element_t>
    void equal(const Element_t* cmp_0, Element_t cmp_1, Element_t* res_mask, usize count, Element_t mask_equal_val = mask_equal<Element_t>, Element_t mask_nonequal_val = mask_nonequal<Element_t>) noexcept;

    template<BasicArithmetic Element_t>
    void not_equal(const Element_t* cmp_0, Element_t cmp_1, Element_t* res_mask, usize count, Element_t mask_equal_val = mask_equal<Element_t>, Element_t mask_nonequal_val = mask_nonequal<Element_t>) noexcept;

    template<BasicArithmetic Element_t>
    void less(const Element_t* cmp_0, Element_t cmp_1, Element_t* res_mask, usize count, Element_t mask_equal_val = mask_equal<Element_t>, Element_t mask_nonequal_val = mask_nonequal<Element_t>) noexcept;

    template<BasicArithmetic Element_t>
    void less_equal(const Element_t* cmp_0, Element_t cmp_1, Element_t* res_mask, usize count, Element_t mask_equal_val = mask_equal<Element_t>, Element_t mask_nonequal_val = mask_nonequal<Element_t>) noexcept;

    template<BasicArithmetic Element_t>
    void greater(const Element_t* cmp_0, Element_t cmp_1, Element_t* res_mask, usize count, Element_t mask_equal_val = mask_equal<Element_t>, Element_t mask_nonequal_val = mask_nonequal<Element_t>) noexcept;

    template<BasicArithmetic Element_t>
    void greater_equal(const Element_t* cmp_0, Element_t cmp_1, Element_t* res_mask, usize count, Element_t mask_equal_val = mask_equal<Element_t>, Element_t mask_nonequal_val = mask_nonequal<Element_t>) noexcept;


    template<Floating_arithmetic Element_t>
    void close(const Element_t* cmp_0, const Element_t* cmp_1, Element_t* res_mask, usize count,
        Element_t mask_equal_val = mask_equal<Element_t>, Element_t mask_nonequal_val = mask_nonequal<Element_t>,
        f64 rtol = 0.00001, f64 atol = 0.00000001, bool nan_is_eq = false) noexcept;

    template<integral_arithmetic Element_t> // u8 u16 u32
    void close(const Element_t* cmp_0, const Element_t* cmp_1, Element_t* res_mask, usize count, Element_t tolerance = 0,
        Element_t mask_equal_val = mask_equal<Element_t>, Element_t mask_nonequal_val = mask_nonequal<Element_t>) noexcept;

    template<BasicArithmetic Element_t>
    usize count_diff(const Element_t* cmp_0, const Element_t* cmp_1, usize count) noexcept;

    //template<integral_arithmetic Element_t>
    //usize count_close(const Element_t* cmp_0, const Element_t* cmp_1, usize count, Element_t tolerance) noexcept;

    //template<Floating_arithmetic Element_t>
    //usize count_close(const Element_t* cmp_0, const Element_t* cmp_1, usize count, f64 rtol = 0.00001, f64 atol = 0.00000001) noexcept;

    template<BasicArithmetic Element_t>
    usize count_repeat(const Element_t* cmp_0, Element_t repeat, usize count) noexcept;

    template<BasicArithmetic Element_t>
    bool any_diff(const Element_t* cmp_0, const Element_t* cmp_1, usize count) noexcept;

    template<BasicArithmetic Element_t>
    bool any_same(const Element_t* cmp0, const Element_t* cmp1, size_t count) noexcept;

    template<BasicArithmetic Element_t> // unstable
    void compare(const Element_t* src, usize count, Element_t* dst, Element_t to_compare,
        Element_t mask_lez_val, Element_t mask_eqz_val, Element_t mask_gtz_val);

    template<BasicArithmetic Element_t>
    void abs_diff(const Element_t* src_0, const Element_t* src_1, Element_t* dst, usize count) noexcept;



    template<BasicArithmetic Element_t>
    void abs(const Element_t* input, Element_t* output, usize count) noexcept;

    template<BasicArithmetic Element_t> // only supports u8, u16
    void avg(const Element_t* input_0, const Element_t* input_1, Element_t* output, usize count) noexcept;

    template<Floating_arithmetic Element_t>
    void exp(const Element_t* input, Element_t* output, usize count) noexcept;

    template<Floating_arithmetic Element_t>
    void exp2(const Element_t* input, Element_t* output, usize count) noexcept;

    template<Floating_arithmetic Element_t>
    void exp10(const Element_t* input, Element_t* output, usize count) noexcept;

    template<Floating_arithmetic Element_t>
    void log(const Element_t* input, Element_t* output, usize count) noexcept;

    template<Floating_arithmetic Element_t>
    void log2(const Element_t* input, Element_t* output, usize count) noexcept;

    template<Floating_arithmetic Element_t>
    void log10(const Element_t* input, Element_t* output, usize count) noexcept;

    template<Floating_arithmetic Element_t>
    void sqrt(const Element_t* input, Element_t* output, usize count) noexcept;
    
    template<Floating_arithmetic Element_t>
    void rsqrt(const Element_t* input, Element_t* output, usize count) noexcept;

    template<Floating_arithmetic Element_t>
    void rcp(const Element_t* input, Element_t* output, usize count) noexcept;

    template<Floating_arithmetic Element_t>
    void rcp(const Element_t* input, Element_t* output, usize count) noexcept;


    
    template<BasicArithmetic Element_t>
    void arg_where(Element_t to_find, const Element_t* src, usize count, farray<usize>& resbuf) noexcept;

    template<BasicArithmetic Element_t>
    void argset_where(const Element_t* const to_find, usize count_to_find, const Element_t* src, usize count_count, farray<usize>& resbuf) noexcept;

    template<Floating_arithmetic Element_t>
    void arg_where(Element_t to_find, const Element_t* src, usize count, farray<usize>& resbuf, Element_t epsilon = Element_t(1e-9f)) noexcept;

    template<Floating_arithmetic Element_t>
    void argset_where(const Element_t* const to_find, usize count_to_find, const Element_t* src, usize count_src, farray<usize>& resbuf, Element_t epsilon = Element_t(1e-9f)) noexcept;



    template<BasicArithmetic Element_t>
    extended_t<Element_t> sum(const Element_t* src, usize count) noexcept;

    template<BasicArithmetic Element_t>
    Element_t median(const Element_t* src, usize count) noexcept;

    template<integral_arithmetic Element_t> // only u8 and 65537 * u16
    Element_t mean(const Element_t* src, usize count) noexcept;

    template<Floating_arithmetic Element_t>
    f64 mean(const Element_t* src, usize count) noexcept;

    //template<BasicArithmetic Element_t>
    //Element_t std(const Element_t* src, usize count) noexcept;

    //template<BasicArithmetic Element_t>
    //Element_t var(const Element_t* src, usize count) noexcept;


    template<BasicArithmetic Element_t>
    extended_t<Element_t> dot_product(const Element_t* a, const Element_t* b, usize count) noexcept;
}