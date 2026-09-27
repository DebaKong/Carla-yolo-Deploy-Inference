#include "protocol.h"

#include <cstring>
#include <stdexcept>

namespace
{
    void appendBytes(
        std::vector<uint8_t>& buffer,
        const void* data,
        size_t size
    )
    {
        const uint8_t* ptr =
            static_cast<const uint8_t*>(data);

        buffer.insert(
            buffer.end(),
            ptr,
            ptr + size
        );
    }
}

std::vector<uint8_t> DetectionSerializer::serialize(
    const std::vector<TrackResult>& results
)
{
    constexpr size_t BYTES_PER_DETECTION = 28;

    const uint32_t count =
        static_cast<uint32_t>(results.size());

    std::vector<uint8_t> buffer;

    buffer.reserve(
        sizeof(uint32_t)
        + results.size()
        * BYTES_PER_DETECTION
    );

    // Detection 数量
    appendBytes(
        buffer,
        &count,
        sizeof(count)
    );

    for (const auto& result : results)
    {
        const int32_t class_id =
            result.class_id;

	const int32_t track_id =
            result.track_id;

        const float confidence =
            result.confidence;

        const int32_t x =
            result.box.x;

        const int32_t y =
            result.box.y;

        const int32_t width =
            result.box.width;

        const int32_t height =
            result.box.height;

        appendBytes(
            buffer,
            &class_id,
            sizeof(class_id)
        );

	appendBytes(
            buffer,
            &track_id,
            sizeof(track_id)
        );

        appendBytes(
            buffer,
            &confidence,
            sizeof(confidence)
        );

        appendBytes(
            buffer,
            &x,
            sizeof(x)
        );

        appendBytes(
            buffer,
            &y,
            sizeof(y)
        );

        appendBytes(
            buffer,
            &width,
            sizeof(width)
        );

        appendBytes(
            buffer,
            &height,
            sizeof(height)
        );
    }

    return buffer;
}
