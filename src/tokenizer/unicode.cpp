#include "sglang/tokenizer/unicode.h"

#include "sglang/utf8.h"

#include <unicode/bytestream.h>
#include <unicode/normalizer2.h>
#include <unicode/uchar.h>
#include <unicode/uniset.h>
#include <unicode/ustring.h>

#include <algorithm>
#include <array>
#include <limits>
#include <stdexcept>
#include <vector>

namespace sglang::tokenizer {
namespace {

constexpr unsigned letter = 1;
constexpr unsigned mark = 2;
constexpr unsigned number = 4;
constexpr unsigned space = 8;
constexpr unsigned word = letter | mark;

static_assert(U_ICU_VERSION_MAJOR_NUM == 78 && U_ICU_VERSION_MINOR_NUM == 3);

template <unsigned Major> bool age_at_most(UChar32 scalar) {
    UVersionInfo age{};
    u_charAge(scalar, age);
    return age[0] < Major || (age[0] == Major && age[1] == 0);
}

const icu::Normalizer2& nfc() {
    static const auto* instance = [] {
        UErrorCode status = U_ZERO_ERROR;
        const auto* result = icu::Normalizer2::getNFCInstance(status);
        if (U_FAILURE(status) != 0) {
            throw std::runtime_error(u_errorName(status));
        }
        return result;
    }();
    return *instance;
}

const icu::FilteredNormalizer2& nfc9() {
    static const auto filter = [] {
        UErrorCode status = U_ZERO_ERROR;
        icu::UnicodeSet result(icu::UnicodeString(u"[:age=9.0:]"), status);
        if (U_FAILURE(status) != 0) {
            throw std::runtime_error(u_errorName(status));
        }
        result.freeze();
        return result;
    }();
    static const icu::FilteredNormalizer2 normalizer(nfc(), filter);
    return normalizer;
}

struct CanonicalScalar {
    char32_t point;
    unsigned combining_class;
};

bool ascii(std::string_view bytes) {
    return std::all_of(bytes.begin(), bytes.end(), [](unsigned char byte) { return byte < 128; });
}

unsigned properties(char32_t scalar) {
    if (scalar < 128) {
        if ((scalar | 32) >= U'a' && (scalar | 32) <= U'z') {
            return letter;
        }
        if (scalar >= U'0' && scalar <= U'9') {
            return number;
        }
        return scalar == U' ' || (scalar >= 9 && scalar <= 13) ? space : 0;
    }
    const auto point = static_cast<UChar32>(scalar);
    const auto category = U_GET_GC_MASK(point);
    unsigned flags = 0;
    if ((category & U_GC_L_MASK) != 0) {
        flags = letter;
    } else if ((category & U_GC_M_MASK) != 0) {
        flags = mark;
    } else if ((category & U_GC_N_MASK) != 0) {
        flags = number;
    } else if (u_isUWhiteSpace(point) != 0) {
        flags = space;
    }
    return flags != 0 && age_at_most<16>(point) ? flags : 0;
}

char32_t fold_contraction(char32_t scalar) {
    if (scalar == 0x17f) {
        return U's';
    }
    return scalar >= U'A' && scalar <= U'Z' ? scalar + 32 : scalar;
}

bool newline(char32_t scalar) {
    return scalar == U'\r' || scalar == U'\n';
}

void decompose(const icu::Normalizer2& normalizer, UChar32 point,
               std::vector<CanonicalScalar>& output) {
    icu::UnicodeString mapping;
    if (normalizer.getDecomposition(point, mapping) == 0) {
        output.push_back({static_cast<char32_t>(point), normalizer.getCombiningClass(point)});
        return;
    }
    for (int i = 0; i < mapping.length();) {
        const auto scalar = mapping.char32At(i);
        output.push_back({static_cast<char32_t>(scalar), normalizer.getCombiningClass(scalar)});
        i += U16_LENGTH(scalar);
    }
}

std::u32string normalize_large(std::u32string input) {
    const auto& normalizer = nfc9();
    std::vector<CanonicalScalar> decomposed;
    decomposed.reserve(input.size());
    for (const auto point : input) {
        decompose(normalizer, static_cast<UChar32>(point), decomposed);
    }
    for (auto first = decomposed.begin(); first != decomposed.end();) {
        if (first->combining_class == 0) {
            ++first;
            continue;
        }
        const auto end = std::find_if(first, decomposed.end(), [](const auto& scalar) {
            return scalar.combining_class == 0;
        });
        if (end - first > 1) {
            std::stable_sort(first, end, [](const auto& left, const auto& right) {
                return left.combining_class < right.combining_class;
            });
        }
        first = end;
    }
    input.clear();
    auto starter = std::u32string::npos;
    unsigned previous_class = 0;
    for (const auto [point, combining_class] : decomposed) {
        if (starter != std::u32string::npos &&
            (previous_class == 0 || previous_class < combining_class)) {
            const auto composite = normalizer.composePair(static_cast<UChar32>(input[starter]),
                                                          static_cast<UChar32>(point));
            if (composite >= 0) {
                input[starter] = static_cast<char32_t>(composite);
                continue;
            }
        }
        if (combining_class == 0) {
            starter = input.size();
        }
        input.push_back(point);
        previous_class = combining_class;
    }
    return input;
}

bool normalizer_fits(std::string_view bytes) {
    return bytes.size() <= static_cast<std::size_t>(std::numeric_limits<int32_t>::max()) / 4;
}

std::string normalize_checked(std::string_view bytes) {
    if (!valid_utf8(bytes)) {
        throw std::invalid_argument("Invalid UTF-8 input");
    }
    std::string output;
    output.reserve(bytes.size());
    icu::StringByteSink<std::string> sink(&output);
    UErrorCode status = U_ZERO_ERROR;
    nfc9().normalizeUTF8(0, {bytes.data(), static_cast<int32_t>(bytes.size())}, sink, nullptr,
                         status);
    if (U_FAILURE(status) != 0) {
        throw std::runtime_error(u_errorName(status));
    }
    return output;
}

std::u32string normalized_scalars(std::string_view bytes) {
    if (ascii(bytes)) {
        return read_utf8(bytes);
    }
    if (!normalizer_fits(bytes)) {
        return normalize_large(read_utf8(bytes));
    }
    return read_utf8(normalize_checked(bytes));
}

void append_folded(std::u32string& output, char32_t point) {
    if (point < 128) {
        output += point >= U'A' && point <= U'Z' ? point + 32 : point;
    } else if (point > 0x10ffff || U_IS_SURROGATE(point) ||
               !age_at_most<15>(static_cast<UChar32>(point))) {
        output += point;
    } else {
        UChar input[2];
        UChar folded[6];
        int32_t size = 0;
        U16_APPEND_UNSAFE(input, size, point);
        UErrorCode status = U_ZERO_ERROR;
        const auto length = u_strFoldCase(folded, 6, input, size, U_FOLD_CASE_DEFAULT, &status);
        if (U_FAILURE(status) != 0) {
            throw std::runtime_error(u_errorName(status));
        }
        for (int32_t offset = 0; offset < length;) {
            UChar32 scalar = 0;
            U16_NEXT_UNSAFE(folded, offset, scalar);
            output += static_cast<char32_t>(scalar);
        }
    }
}

std::size_t contraction_end(std::u32string_view text, std::size_t start) {
    constexpr std::array<std::u32string_view, 7> options = {
        U"s", U"t", U"re", U"ve", U"m", U"ll", U"d",
    };
    const auto suffix = text.substr(start + 1);
    for (const auto option : options) {
        if (suffix.size() >= option.size() &&
            std::equal(option.begin(), option.end(), suffix.begin(),
                [](char32_t expected, char32_t input) {
                    return expected == fold_contraction(input);
                })) {
            return start + 1 + option.size();
        }
    }
    return start;
}

std::size_t piece_end(std::u32string_view text, const std::vector<unsigned char>& flags,
                      std::size_t start) {
    const auto size = text.size();
    if (text[start] == U'\'') {
        const auto end = contraction_end(text, start);
        if (end != start) {
            return end;
        }
    }
    const auto has = [&](std::size_t at, unsigned wanted) { return (flags[at] & wanted) != 0; };
    const auto symbols = [&](std::size_t at) { return flags[at] == 0; };
    if (has(start, word) ||
        (!newline(text[start]) && !has(start, letter | number) &&
         start + 1 < size && has(start + 1, word))) {
        auto end = start + 1;
        while (end < size && has(end, word)) {
            ++end;
        }
        return end;
    }
    if (has(start, number)) {
        return start + 1;
    }
    auto end = start + static_cast<unsigned>(text[start] == U' ');
    if (end < size && symbols(end)) {
        do {
            ++end;
        } while (end < size && symbols(end));
        while (end < size && newline(text[end])) {
            ++end;
        }
        return end;
    }
    end = start;
    auto line_end = start;
    while (end < size && has(end, space)) {
        if (newline(text[end])) {
            line_end = end + 1;
        }
        ++end;
    }
    if (line_end != start) {
        return line_end;
    }
    return end == size ? end : std::max(start + 1, end - 1);
}

}

std::string normalize_nfc(std::string_view bytes) {
    if (ascii(bytes)) {
        return std::string(bytes);
    }
    if (normalizer_fits(bytes)) {
        return normalize_checked(bytes);
    }
    std::string output;
    output.reserve(bytes.size());
    for (const auto scalar : normalize_large(read_utf8(bytes))) {
        append_utf8(output, scalar);
    }
    return output;
}

std::u32string casefold(std::u32string_view text) {
    std::u32string output;
    output.reserve(text.size());
    for (const auto point : text) {
        append_folded(output, point);
    }
    return output;
}

void split_pieces(std::string_view bytes,
                  const std::function<void(std::string_view)>& consume) {
    const auto text = normalized_scalars(bytes);
    std::vector<unsigned char> flags;
    flags.reserve(text.size());
    for (const auto point : text) {
        flags.push_back(static_cast<unsigned char>(properties(point)));
    }
    std::string piece;
    for (std::size_t start = 0; start < text.size();) {
        const auto end = piece_end(text, flags, start);
        piece.clear();
        while (start < end) {
            append_utf8(piece, text[start++]);
        }
        consume(piece);
    }
}

}
