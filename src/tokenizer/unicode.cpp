#include "sglang/tokenizer/unicode.h"

#include "sglang/utf8.h"

#include <algorithm>
#include <array>
#include <iterator>
#include <utility>
#include <vector>

namespace sglang::tokenizer {
namespace {

struct PropertyRange {
    char32_t first;
    char32_t last;
    unsigned char flags;
};

struct ScalarPair {
    char32_t point;
    char32_t value;
};

struct Mapping {
    char32_t first;
    char32_t second;
    char32_t third;
};

#include "sglang/tokenizer/unicode/character_properties_16.h"
#include "sglang/tokenizer/unicode/combining_classes_9.h"
#include "sglang/tokenizer/unicode/canonical_decompositions_9.h"
#include "sglang/tokenizer/unicode/canonical_compositions_9.h"

constexpr unsigned letter = 1;
constexpr unsigned mark = 2;
constexpr unsigned number = 4;
constexpr unsigned space = 8;
constexpr unsigned word = letter | mark;

constexpr char32_t syllable_base = 0xac00;
constexpr char32_t leading_base = 0x1100;
constexpr char32_t vowel_base = 0x1161;
constexpr char32_t trailing_base = 0x11a7;
constexpr char32_t leading_count = 19;
constexpr char32_t vowel_count = 21;
constexpr char32_t trailing_count = 28;
constexpr char32_t syllable_group = vowel_count * trailing_count;
constexpr char32_t syllable_count = leading_count * syllable_group;

struct CanonicalScalar {
    char32_t point;
    unsigned combining_class;
};

template <class Entry, std::size_t Size, class Field>
const Entry* lookup(const Entry (&table)[Size], char32_t point, Field field) {
    const auto entry = std::lower_bound(std::begin(table), std::end(table), point,
        [field](const Entry& value, char32_t wanted) { return value.*field < wanted; });
    return entry != std::end(table) && (*entry).*field == point ? entry : nullptr;
}

unsigned properties(char32_t scalar) {
    const auto after = std::upper_bound(std::begin(properties_16), std::end(properties_16),
        scalar, [](char32_t point, const PropertyRange& range) { return point < range.first; });
    return after != std::begin(properties_16) && scalar <= (after - 1)->last
        ? (after - 1)->flags : 0;
}

char32_t fold_contraction(char32_t scalar) {
    const auto* entry = lookup(contraction_folds_16, scalar, &ScalarPair::point);
    return entry ? entry->value : scalar;
}

bool newline(char32_t scalar) {
    return scalar == U'\r' || scalar == U'\n';
}

void decompose(char32_t point, std::vector<CanonicalScalar>& output) {
    const auto syllable = point - syllable_base;
    if (syllable < syllable_count) {
        output.push_back({leading_base + syllable / syllable_group, 0});
        output.push_back({vowel_base + (syllable % syllable_group) / trailing_count, 0});
        if (const auto trailing = syllable % trailing_count) {
            output.push_back({trailing_base + trailing, 0});
        }
    } else if (const auto* mapping = lookup(decompositions_9, point, &Mapping::first)) {
        decompose(mapping->second, output);
        if (mapping->third) {
            decompose(mapping->third, output);
        }
    } else {
        const auto* entry = lookup(classes_9, point, &ScalarPair::point);
        output.push_back({point, entry ? entry->value : 0});
    }
}

char32_t compose(char32_t first, char32_t second) {
    const auto leading = first - leading_base;
    const auto vowel = second - vowel_base;
    if (leading < leading_count && vowel < vowel_count) {
        return syllable_base + (leading * vowel_count + vowel) * trailing_count;
    }
    const auto syllable = first - syllable_base;
    const auto trailing = second - trailing_base;
    if (syllable < syllable_count && syllable % trailing_count == 0 &&
        trailing > 0 && trailing < trailing_count) {
        return first + trailing;
    }
    const auto key = std::pair(first, second);
    const auto found = std::lower_bound(std::begin(compositions_9), std::end(compositions_9),
        key, [](const Mapping& value, const auto& wanted) {
            return std::pair(value.first, value.second) < wanted;
        });
    return found != std::end(compositions_9) && std::pair(found->first, found->second) == key
        ? found->third : 0;
}

std::u32string normalized_scalars(std::string_view bytes) {
    auto input = read_utf8(bytes);
    if (std::all_of(input.begin(), input.end(), [](char32_t point) { return point < 0x80; })) {
        return input;
    }
    std::vector<CanonicalScalar> decomposed;
    decomposed.reserve(input.size());
    for (const auto point : input) {
        decompose(point, decomposed);
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
            if (const auto composite = compose(input[starter], point)) {
                input[starter] = composite;
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
    std::string output;
    output.reserve(bytes.size());
    for (const auto scalar : normalized_scalars(bytes)) {
        append_utf8(output, scalar);
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
