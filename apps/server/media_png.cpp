#include "sglang/server/media_codec.h"

#include <png.h>
#include <cstring>
#include <stdexcept>

namespace sglang::server {
namespace {

struct PngImage {
    png_image header{};
    PngImage() { header.version = PNG_IMAGE_VERSION; }
    ~PngImage() { png_image_free(&header); }
};

std::uint32_t big_endian(const std::uint8_t* bytes) {
    return (static_cast<std::uint32_t>(bytes[0]) << 24) |
        (static_cast<std::uint32_t>(bytes[1]) << 16) |
        (static_cast<std::uint32_t>(bytes[2]) << 8) |
        static_cast<std::uint32_t>(bytes[3]);
}

}

DecodedMedia decode_png(const std::vector<std::uint8_t>& bytes) {
    PngImage image;
    auto& header = image.header;
    if (bytes.size() < 33 || png_sig_cmp(bytes.data(), 0, 8) ||
        big_endian(bytes.data() + 8) != 13 || std::memcmp(bytes.data() + 12, "IHDR", 4)) {
        throw std::invalid_argument("Invalid PNG image");
    }
    const auto width = big_endian(bytes.data() + 16);
    const auto height = big_endian(bytes.data() + 20);
    if (width < min_media_dimension || height < min_media_dimension ||
        width > max_media_dimension || height > max_media_dimension) {
        throw std::invalid_argument("PNG dimensions must be in [16, 1024]");
    }
    if (!png_image_begin_read_from_memory(&header, bytes.data(), bytes.size()) ||
        header.width != width || header.height != height) {
        throw std::invalid_argument("Invalid PNG header");
    }
    header.format = PNG_FORMAT_RGBA;
    std::vector<std::uint8_t> rgba(PNG_IMAGE_SIZE(header));
    if (!png_image_finish_read(&header, nullptr, rgba.data(), 0, nullptr)) {
        throw std::invalid_argument("Invalid PNG pixels");
    }
    DecodedMedia decoded{static_cast<int>(header.height), static_cast<int>(header.width), {}, {}};
    decoded.rgb.reserve(rgba.size() / 4 * 3);
    for (std::size_t offset = 0; offset < rgba.size(); offset += 4) {
        decoded.rgb.insert(decoded.rgb.end(), rgba.data() + offset, rgba.data() + offset + 3);
    }
    return decoded;
}

}
