#ifndef TRACKER_H
#define TRACKER_H

#include "yolov8.h"
#include "BYTETracker.h"

#include <memory>
#include <unordered_map>
#include <vector>

struct TrackResult
{
    int class_id;
    int track_id;
    float confidence;
    cv::Rect box;
};

namespace mytrack {
class Tracker
{
public:
    Tracker(
	int frame_rate = 30,
	int track_buffer = 30
    );

    std::vector<TrackResult> update(
	const std::vector<Detection>& detections
    );

private:
    std::unordered_map<
	int,
	std::unique_ptr<BYTETracker>
    > trackers_;

    int frame_rate_;
    int track_buffer_;

private:
    BYTETracker& getTracker(int class_id);
};
}

#endif
