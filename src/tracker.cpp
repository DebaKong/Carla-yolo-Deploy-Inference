#include "tracker.h"

#include <cmath>
#include <vector>


mytrack::Tracker::Tracker(
    int frame_rate,
    int track_buffer
)
    : frame_rate_(frame_rate),
      track_buffer_(track_buffer)
{
}


BYTETracker& mytrack::Tracker::getTracker(int class_id)
{
    auto it = trackers_.find(class_id);

    if (it == trackers_.end())
    {
	auto tracker = std::make_unique<BYTETracker>(frame_rate_, track_buffer_);

	auto result = trackers_.emplace(class_id, std::move(tracker));

	return *result.first->second;
    }

    return *it->second;
}


std::vector<TrackResult> mytrack::Tracker::update(
    const std::vector<Detection>& detections
)
{
    std::unordered_map<
	int,
	std::vector<Object>
    > objects_by_class;

    // Detection -> ByteTrack Object  按类别拆分
    for (const auto& detection : detections)
    {
	Object object;

	object.rect = cv::Rect_<float>(
	    static_cast<float>(detection.box.x),
	    static_cast<float>(detection.box.y),
	    static_cast<float>(detection.box.width),
	    static_cast<float>(detection.box.height)
	);

	object.label = detection.class_id;
	object.prob = detection.confidence;

	objects_by_class[detection.class_id].push_back(object);
    }

    std::vector<TrackResult> results;

    // 对每个类别独立执行ByteTrack
    for (auto& item : objects_by_class)
    {
	const int class_id = item.first;

	BYTETracker& tracker = getTracker(class_id);

	std::vector<STrack> tracks = tracker.update(item.second);

	// STrack -> TrackResult
	for (const auto& track : tracks)
	{
	    if (!track.is_activated)
            {
                continue;
            }

            if (track.state != TrackState::Tracked)
            {
                continue;
            }

            const int x =
                static_cast<int>(
                    std::round(track.tlwh[0])
                );

            const int y =
                static_cast<int>(
                    std::round(track.tlwh[1])
                );

            const int width =
                static_cast<int>(
                    std::round(track.tlwh[2])
                );

            const int height =
                static_cast<int>(
                    std::round(track.tlwh[3])
                );

            TrackResult result;

            result.class_id = class_id;
            result.track_id = track.track_id;
            result.confidence = track.score;

            result.box = cv::Rect(
                x,
                y,
                width,
                height
            );

            results.push_back(result);
        }
    }

    return results;
}
