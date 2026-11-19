#include "sglang/media_resize.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <type_traits>

namespace sglang::media_detail {
namespace {

double filter_weight(double distance, ResizeFilter filter) {
    const double offset = std::abs(distance);
    if (filter == ResizeFilter::Bilinear) {
        return std::max(1.0 - offset, 0.0);
    }
    constexpr double cubic = -0.5;
    if (offset < 1.0) {
        return ((cubic + 2.0) * offset - (cubic + 3.0)) * offset * offset + 1.0;
    }
    if (offset < 2.0) {
        return ((cubic * offset - 5.0 * cubic) * offset + 8.0 * cubic) * offset - 4.0 * cubic;
    }
    return 0.0;
}

std::uint8_t round_to_byte(float value) {
    const int whole = static_cast<int>(value);
    const float fraction = value - whole;
    return static_cast<std::uint8_t>(
        std::clamp(whole + (fraction > 0.5f || (fraction == 0.5f && whole % 2)), 0, 255));
}

}

std::size_t rgb_bytes(Size size) {
    return static_cast<std::size_t>(size.height) * size.width * 3;
}

Resize::Resize(Size input, Size output, ResizeFilter filter, std::size_t byte_limit)
    : input_(input), middle_{input.height, output.width}, output_(output) {
    if (rgb_bytes(middle_) > byte_limit || rgb_bytes(output_) > byte_limit) {
        throw std::length_error("Media resize workspace exceeds the byte limit");
    }
    const bool change_width = input.width != output.width;
    const bool change_height = input.height != output.height;
    if (filter == ResizeFilter::Bilinear) {
        if (change_width) {
            bilinear_horizontal_ = make_weights<float>(input.width, output.width, filter);
        }
        if (change_height) {
            bilinear_vertical_ = make_weights<float>(input.height, output.height, filter);
        }
        if (change_width && change_height) {
            if (rgb_bytes(middle_) > byte_limit / sizeof(float)) {
                throw std::length_error("Media resize workspace exceeds the byte limit");
            }
            bilinear_temporary_.resize(rgb_bytes(middle_));
        }
    } else {
        if (change_width) {
            horizontal_precision_ = make_fixed_weights(input.width, output.width, horizontal_);
        }
        if (change_height) {
            vertical_precision_ = make_fixed_weights(input.height, output.height, vertical_);
        }
        if (change_width && change_height) {
            temporary_.resize(rgb_bytes(middle_));
        }
    }
    if (change_width || change_height) {
        result_.resize(rgb_bytes(output));
    }
}

template <typename Weight>
std::vector<Resize::Tap<Weight>> Resize::make_weights(int input, int output, ResizeFilter filter) {
    const Weight scale = static_cast<Weight>(input) / output;
    const Weight stretch = std::max(Weight{1}, scale);
    const Weight inv_scale = 1.0 / stretch;
    const Weight support = stretch * (filter == ResizeFilter::Bilinear ? 1 : 2);
    const int max_taps = 2 * static_cast<int>(std::ceil(support)) + 1;
    std::vector<Tap<Weight>> axis;
    axis.reserve(output);
    for (int index = 0; index < output; ++index) {
        const Weight center = scale * (index + 0.5);
        const int first = std::max(0, static_cast<int>(center - support + 0.5));
        const int end = std::min({input, first + max_taps,
                                  static_cast<int>(center + support + 0.5)});
        auto& tap = axis.emplace_back(Tap<Weight>{first, {}});
        tap.weights.reserve(end - first);
        Weight sum = 0;
        for (int source = first; source < end; ++source) {
            const Weight distance = (source - center + 0.5) * inv_scale;
            const Weight weight = filter_weight(distance, filter);
            tap.weights.push_back(weight);
            sum += weight;
        }
        for (auto& weight : tap.weights) {
            weight /= sum;
        }
    }
    return axis;
}

int Resize::make_fixed_weights(int input, int output, std::vector<Tap<int>>& axis) {
    const auto coefficients = make_weights<double>(input, output, ResizeFilter::Bicubic);
    double max_weight = 0;
    for (const auto& tap : coefficients) {
        const auto maximum = std::max_element(tap.weights.begin(), tap.weights.end());
        max_weight = std::max(max_weight, *maximum);
    }
    int precision = 0;
    while (precision < 22 && std::round(std::ldexp(max_weight, precision + 1)) < 32768) {
        ++precision;
    }
    axis.reserve(output);
    for (const auto& coeff : coefficients) {
        auto& tap = axis.emplace_back(Tap<int>{coeff.first, {}});
        tap.weights.reserve(coeff.weights.size());
        for (double weight : coeff.weights) {
            tap.weights.push_back(static_cast<int>(std::round(std::ldexp(weight, precision))));
        }
    }
    return precision;
}

void Resize::apply_axis(const std::uint8_t* input, Size source, std::uint8_t* output, Size target,
                        const std::vector<Tap<int>>& axis, int precision, bool horizontal) {
    const auto row_bytes = static_cast<std::size_t>(source.width) * 3;
    const std::size_t stride = horizontal ? 3 : row_bytes;
    for (int row = 0; row < target.height; ++row) {
        for (int column = 0; column < target.width; ++column) {
            const auto& tap = axis[horizontal ? column : row];
            const int from_y = horizontal ? row : tap.first;
            const int from_x = horizontal ? tap.first : column;
            const auto* start = input + from_y * row_bytes + from_x * 3;
            for (int channel = 0; channel < 3; ++channel) {
                std::int64_t value = std::int64_t{1} << (precision - 1);
                for (std::size_t index = 0; index < tap.weights.size(); ++index) {
                    value += start[index * stride + channel] * tap.weights[index];
                }
                const auto pixel = std::min<std::int64_t>(
                    std::max(value, std::int64_t{0}) >> precision, 255);
                *output++ = static_cast<std::uint8_t>(pixel);
            }
        }
    }
}

template <typename Input, typename Output>
void Resize::apply_bilinear(const Input* input, Size source, Output* output, Size target,
                            const std::vector<Tap<float>>& axis, bool horizontal) {
    const auto row_bytes = static_cast<std::size_t>(source.width) * 3;
    const std::size_t stride = horizontal ? 3 : row_bytes;
    for (int row = 0; row < target.height; ++row) {
        for (int column = 0; column < target.width; ++column) {
            const auto& tap = axis[horizontal ? column : row];
            const int from_y = horizontal ? row : tap.first;
            const int from_x = horizontal ? tap.first : column;
            const auto* start = input + from_y * row_bytes + from_x * 3;
            for (int channel = 0; channel < 3; ++channel) {
                float value = start[channel] * tap.weights.front();
                for (std::size_t index = 1; index < tap.weights.size(); ++index) {
                    value = std::fma(start[index * stride + channel], tap.weights[index], value);
                }
                if constexpr (std::is_same_v<Output, float>) {
                    *output++ = value;
                } else {
                    *output++ = round_to_byte(value);
                }
            }
        }
    }
}

const std::uint8_t* Resize::apply(const std::uint8_t* input) {
    if (!bilinear_horizontal_.empty() && !bilinear_vertical_.empty()) {
        auto* stage = bilinear_temporary_.data();
        apply_bilinear(input, input_, stage, middle_, bilinear_horizontal_, true);
        apply_bilinear(stage, middle_, result_.data(), output_, bilinear_vertical_, false);
        return result_.data();
    }
    if (!bilinear_horizontal_.empty() || !bilinear_vertical_.empty()) {
        const bool horizontal = !bilinear_horizontal_.empty();
        const auto& axis = horizontal ? bilinear_horizontal_ : bilinear_vertical_;
        apply_bilinear(input, input_, result_.data(), output_, axis, horizontal);
        return result_.data();
    }
    if (!horizontal_.empty()) {
        auto* stage = vertical_.empty() ? result_.data() : temporary_.data();
        apply_axis(input, input_, stage, middle_, horizontal_, horizontal_precision_, true);
        input = stage;
    }
    if (!vertical_.empty()) {
        apply_axis(input, middle_, result_.data(), output_, vertical_, vertical_precision_, false);
        input = result_.data();
    }
    return input;
}

}
