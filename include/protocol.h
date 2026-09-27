#ifndef PROTOCOL_H
#define PROTOCOL_H

#include "tracker.h"

#include <cstdint>
#include <vector>

class DetectionSerializer
{
public:
    static std::vector<uint8_t> serialize(
        const std::vector<TrackResult>& results
    );
};

#endif
