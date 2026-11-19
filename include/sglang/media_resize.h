#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace sglang::media_detail {

struct Size {
    int height;
    int width;
};

enum class ResizeFilter { Bilinear, Bicubic };

std::size_t rgb_bytes(Size size);

class Resize {
public:
    Resize(Size input, Size output, ResizeFilter filter, std::size_t byte_limit);
    const std::uint8_t* apply(const std::uint8_t* input);

private:
    template <typename Weight>
    struct Tap {
        int first;
        std::vector<Weight> weights;
    };

    template <typename Weight>
    static std::vector<Tap<Weight>> make_weights(int input, int output, ResizeFilter filter);
    static int make_fixed_weights(int input, int output, std::vector<Tap<int>>& axis);
    static void apply_axis(const std::uint8_t* input, Size source, std::uint8_t* output,
                           Size target, const std::vector<Tap<int>>& axis, int precision,
                           bool horizontal);
    template <typename Input, typename Output>
    static void apply_bilinear(const Input* input, Size source, Output* output, Size target,
                               const std::vector<Tap<float>>& axis, bool horizontal);

    Size input_;
    Size middle_;
    Size output_;
    std::vector<Tap<int>> horizontal_;
    std::vector<Tap<int>> vertical_;
    std::vector<Tap<float>> bilinear_horizontal_;
    std::vector<Tap<float>> bilinear_vertical_;
    int horizontal_precision_ = 0;
    int vertical_precision_ = 0;
    std::vector<std::uint8_t> temporary_;
    std::vector<float> bilinear_temporary_;
    std::vector<std::uint8_t> result_;
};

}
