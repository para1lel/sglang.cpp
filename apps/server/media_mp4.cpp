#include "sglang/server/media_codec.h"

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/common.h>
#include <libavutil/mathematics.h>
#include <libswscale/swscale.h>
}

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <new>
#include <stdexcept>

namespace sglang::server {
namespace {

constexpr int max_source_frames = 256;
constexpr int max_packets = 4096;
constexpr std::size_t max_rgb_bytes = 16 * 1024 * 1024;
constexpr int io_buffer_bytes = 4096;
constexpr unsigned mp4v_tag = MKTAG('m', 'p', '4', 'v');

struct Mp4Reader {
    const std::vector<std::uint8_t>& bytes;
    std::int64_t offset = 0;
};

int read_bytes(void* opaque, std::uint8_t* destination, int size) {
    auto& source = *static_cast<Mp4Reader*>(opaque);
    const auto length = static_cast<std::int64_t>(source.bytes.size());
    const auto count = std::min<std::int64_t>(size, length - source.offset);
    if (count <= 0) {
        return AVERROR_EOF;
    }
    std::memcpy(destination, source.bytes.data() + source.offset, count);
    source.offset += count;
    return static_cast<int>(count);
}

std::int64_t seek_bytes(void* opaque, std::int64_t offset, int mode) {
    auto& source = *static_cast<Mp4Reader*>(opaque);
    const auto length = static_cast<std::int64_t>(source.bytes.size());
    mode &= ~AVSEEK_FORCE;
    if (mode == AVSEEK_SIZE) {
        return length;
    }
    const auto base = mode == SEEK_SET ? 0 : mode == SEEK_CUR ? source.offset : length;
    if ((mode != SEEK_SET && mode != SEEK_CUR && mode != SEEK_END) ||
        offset < -base || offset > length - base) {
        return AVERROR(EINVAL);
    }
    return source.offset = base + offset;
}

int deny_io(AVFormatContext*, AVIOContext**, const char*, int, AVDictionary**) {
    return AVERROR(EPERM);
}

bool valid_dimensions(int height, int width) {
    return height >= min_media_dimension && height <= max_media_dimension &&
        width >= min_media_dimension && width <= max_media_dimension;
}

int allocate_frame(AVCodecContext* codec, AVFrame* frame, int flags) {
    if (!valid_dimensions(frame->height, frame->width) ||
        frame->height != codec->height || frame->width != codec->width) {
        return AVERROR_INVALIDDATA;
    }
    return avcodec_default_get_buffer2(codec, frame, flags);
}

struct Mp4Decoder {
    Mp4Reader source;
    AVIOContext* io = nullptr;
    AVFormatContext* format = nullptr;
    AVCodecContext* codec = nullptr;
    AVPacket* packet = nullptr;
    AVFrame* frame = nullptr;
    SwsContext* colors = nullptr;
    bool opened = false;

    explicit Mp4Decoder(const std::vector<std::uint8_t>& bytes) : source{bytes} {}

    ~Mp4Decoder() {
        sws_freeContext(colors);
        av_frame_free(&frame);
        av_packet_free(&packet);
        avcodec_free_context(&codec);
        if (opened) {
            avformat_close_input(&format);
        } else {
            avformat_free_context(format);
        }
        if (io) {
            av_freep(&io->buffer);
            avio_context_free(&io);
        }
    }

    void open() {
        const auto* demuxer = av_find_input_format("mov");
        const auto* implementation = avcodec_find_decoder(AV_CODEC_ID_MPEG4);
        if (!demuxer || !implementation) {
            throw std::runtime_error("FFmpeg requires the MOV demuxer and MPEG-4 Part 2 decoder");
        }
        auto* buffer = static_cast<unsigned char*>(av_malloc(io_buffer_bytes));
        if (!buffer) {
            throw std::bad_alloc();
        }
        io = avio_alloc_context(buffer, io_buffer_bytes, 0, &source, read_bytes, nullptr,
                                seek_bytes);
        if (!io) {
            av_free(buffer);
            throw std::bad_alloc();
        }
        format = avformat_alloc_context();
        codec = avcodec_alloc_context3(implementation);
        packet = av_packet_alloc();
        frame = av_frame_alloc();
        if (!format || !codec || !packet || !frame) {
            throw std::bad_alloc();
        }
        format->pb = io;
        format->flags |= AVFMT_FLAG_CUSTOM_IO;
        format->io_open = deny_io;
        format->max_streams = 8;
        if (avformat_open_input(&format, nullptr, demuxer, nullptr) < 0) {
            throw std::invalid_argument("Invalid MP4 container");
        }
        opened = true;
    }

    void open_codec(const AVCodecParameters& parameters) {
        if (avcodec_parameters_to_context(codec, &parameters) < 0) {
            throw std::invalid_argument("Invalid MPEG-4 video parameters");
        }
        codec->thread_count = 1;
        codec->max_pixels = max_media_dimension * max_media_dimension;
        codec->get_buffer2 = allocate_frame;
        codec->err_recognition = AV_EF_EXPLODE;
        if (avcodec_open2(codec, nullptr, nullptr) < 0) {
            throw std::invalid_argument("Cannot decode MPEG-4 Part 2 video");
        }
    }

    void copy_rgb(DecodedMedia& decoded, std::size_t index, std::size_t frame_bytes) {
        const auto pixels = static_cast<AVPixelFormat>(frame->format);
        if (pixels != AV_PIX_FMT_YUV420P && pixels != AV_PIX_FMT_YUVJ420P) {
            throw std::invalid_argument("Expected 8-bit YUV420 MPEG-4 Part 2 video");
        }
        colors = sws_getCachedContext(colors, frame->width, frame->height, pixels,
                                      frame->width, frame->height, AV_PIX_FMT_RGB24,
                                      SWS_BICUBIC, nullptr, nullptr, nullptr);
        if (!colors) {
            throw std::runtime_error("Cannot initialize video RGB conversion");
        }
        std::uint8_t* destination[] = {decoded.rgb.data() + index * frame_bytes, nullptr,
                                      nullptr, nullptr};
        const int stride[] = {frame->width * 3, 0, 0, 0};
        if (sws_scale(colors, frame->data, frame->linesize, 0, frame->height,
                      destination, stride) != frame->height) {
            throw std::invalid_argument("Cannot convert MPEG-4 frame to RGB8");
        }
    }
};

const AVStream& video_stream(const AVFormatContext& format) {
    const AVStream* video = nullptr;
    for (unsigned index = 0; index < format.nb_streams; ++index) {
        const auto* stream = format.streams[index];
        if (stream->codecpar->codec_type != AVMEDIA_TYPE_VIDEO) {
            continue;
        }
        const auto& codec = *stream->codecpar;
        if (video || codec.codec_id != AV_CODEC_ID_MPEG4 || codec.codec_tag != mp4v_tag ||
            av_packet_side_data_get(codec.coded_side_data, codec.nb_coded_side_data,
                                    AV_PKT_DATA_DISPLAYMATRIX)) {
            throw std::invalid_argument("Expected one unrotated MPEG-4 Part 2 video track");
        }
        video = stream;
    }
    if (!video || video->nb_frames < 2 || video->nb_frames > max_source_frames ||
        video->time_base.num <= 0 || video->time_base.den <= 0) {
        throw std::invalid_argument("MP4 needs 2 to 256 source frames and a valid time base");
    }
    return *video;
}

std::size_t allocate_rgb(DecodedMedia& decoded, int height, int width) {
    if (!valid_dimensions(height, width)) {
        throw std::invalid_argument("MP4 dimensions must be in [16, 1024]");
    }
    const auto frame_bytes = std::size_t{3} * height * width;
    const auto frames = decoded.video.frame_indices.size();
    if (frame_bytes > max_rgb_bytes / frames) {
        throw std::invalid_argument("MP4 exceeds the 16 MiB selected RGB8 budget");
    }
    decoded.height = height;
    decoded.width = width;
    decoded.rgb.resize(frames * frame_bytes + AV_INPUT_BUFFER_PADDING_SIZE);
    return frame_bytes;
}

struct FrameSelection {
    const AVStream& stream;
    DecodedMedia& decoded;
    std::size_t frame_bytes = 0;
    int received = 0;
    std::size_t selected = 0;
    std::int64_t first_pts = AV_NOPTS_VALUE;

    void receive(Mp4Decoder& decoder) {
        while (true) {
            const int status = avcodec_receive_frame(decoder.codec, decoder.frame);
            if (status == AVERROR(EAGAIN) || status == AVERROR_EOF) {
                return;
            }
            const auto& frame = *decoder.frame;
            if (status < 0 || received == decoded.video.source_frame_count ||
                frame.best_effort_timestamp == AV_NOPTS_VALUE ||
                (frame.flags & AV_FRAME_FLAG_CORRUPT)) {
                throw std::invalid_argument("Invalid or inconsistent MPEG-4 video frame");
            }
            if (received == 0) {
                first_pts = frame.best_effort_timestamp;
                frame_bytes = allocate_rgb(decoded, frame.height, frame.width);
            } else if (frame.height != decoded.height || frame.width != decoded.width) {
                throw std::invalid_argument("MPEG-4 frame dimensions changed during decode");
            }
            const auto ticks = av_sat_sub64(frame.best_effort_timestamp, first_pts);
            const auto expected = av_rescale_q(received, av_inv_q(stream.avg_frame_rate),
                                                stream.time_base);
            if (ticks != expected) {
                throw std::invalid_argument("MP4 requires constant-rate presentation timestamps");
            }
            if (selected < decoded.video.frame_indices.size() &&
                received == decoded.video.frame_indices[selected]) {
                decoder.copy_rgb(decoded, selected++, frame_bytes);
            }
            ++received;
            av_frame_unref(decoder.frame);
        }
    }
};

void read_video(Mp4Decoder& decoder, FrameSelection& selection) {
    for (int count = 0; count < max_packets; ++count) {
        const int status = av_read_frame(decoder.format, decoder.packet);
        if (status == AVERROR_EOF) {
            if (avcodec_send_packet(decoder.codec, nullptr) < 0) {
                break;
            }
            selection.receive(decoder);
            if (selection.received == selection.decoded.video.source_frame_count &&
                selection.selected == selection.decoded.video.frame_indices.size()) {
                return;
            }
            break;
        }
        if (status < 0) {
            break;
        }
        if (decoder.packet->stream_index == selection.stream.index) {
            if (avcodec_send_packet(decoder.codec, decoder.packet) < 0) {
                break;
            }
            selection.receive(decoder);
        }
        av_packet_unref(decoder.packet);
    }
    throw std::invalid_argument("MP4 frame count, packet budget or video data is invalid");
}

}

DecodedMedia decode_mp4(const std::vector<std::uint8_t>& bytes, int selected_frames) {
    if (bytes.size() < 16 || std::memcmp(bytes.data() + 4, "ftyp", 4) ||
        selected_frames < 2 || selected_frames > max_selected_frames) {
        throw std::invalid_argument("Invalid MP4 data URL or selected frame count");
    }
    Mp4Decoder decoder(bytes);
    decoder.open();
    const auto& stream = video_stream(*decoder.format);
    const auto fps = av_q2d(stream.avg_frame_rate);
    if (!(fps >= 0.001 && fps <= 1000)) {
        throw std::invalid_argument("MP4 needs a source rate in [0.001, 1000] frames per second");
    }
    const int source_frames = static_cast<int>(stream.nb_frames);
    const int frames = std::min(selected_frames, source_frames);
    DecodedMedia decoded{0, 0, {}, {source_frames, fps, {}}};
    for (int index = 0; index < frames; ++index) {
        decoded.video.frame_indices.push_back(index * (source_frames - 1) / (frames - 1));
    }
    decoder.open_codec(*stream.codecpar);
    FrameSelection selection{stream, decoded};
    read_video(decoder, selection);
    decoded.rgb.resize(frames * selection.frame_bytes);
    return decoded;
}

}
