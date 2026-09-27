#include "tcp_server.h"
#include "yolov8.h"
#include "protocol.h"
#include "tracker.h"

#include <opencv2/opencv.hpp>

#include <iostream>
#include <vector>
#include <string>
#include <chrono>

int main(int argc, char* argv[])
{
    constexpr uint16_t PORT = 5000;

    if (argc < 2)
    {
        std::cerr
            << "Usage: ./tcp_server <engine_path>\n";

        return -1;
    }

    std::string ENGINE_PATH = argv[1];

    std::cout
        << "Using engine: "
        << ENGINE_PATH
        << std::endl;

    try
    {
        YOLOv8 detector(ENGINE_PATH);

	mytrack::Tracker tracker(30, 30);

        // ========================================
        // TCP Server
        // ========================================

        TcpServer server(
            "0.0.0.0",
            PORT
        );

        if (!server.start())
        {
            std::cerr
                << "[ERROR] Failed to start TCP server."
                << std::endl;

            return 1;
        }

        while (true)
        {
            std::cout
                << "[TCP] Waiting for client..."
                << std::endl;

            if (!server.acceptClient())
            {
                continue;
            }

            while (true)
            {
                std::vector<uint8_t> jpeg_data;

                // ========================================
                // 1. 接收 JPEG
                // ========================================

                if (!server.receiveMessage(jpeg_data))
                {
                    std::cout
                        << "[TCP] Client disconnected."
                        << std::endl;

                    server.closeClient();
                    break;
                }

                // ========================================
                // 2. JPEG → cv::Mat
                // ========================================

                cv::Mat jpeg_buffer(
                    1,
                    static_cast<int>(
                        jpeg_data.size()
                    ),
                    CV_8UC1,
                    jpeg_data.data()
                );

                cv::Mat image =
                    cv::imdecode(
                        jpeg_buffer,
                        cv::IMREAD_COLOR
                    );

                if (image.empty())
                {
                    std::cerr
                        << "[OpenCV] JPEG decode failed."
                        << std::endl;

                    continue;
                }

                // ========================================
                // 3. YOLOv8 TensorRT 推理
                // ========================================

                auto start =
                    std::chrono::high_resolution_clock::now();

                std::vector<Detection> detections =
                    detector.infer(image);

		std::vector<TrackResult> tracks =
		    tracker.update(detections);

                auto end =
                    std::chrono::high_resolution_clock::now();

                const double inference_ms =
                    std::chrono::duration<double, std::milli>(
                        end - start
                    ).count();

                // ========================================
                // 4. 输出日志
                // ========================================

                std::cout
                    << "[Inference] "
                    << image.cols
                    << "x"
                    << image.rows
                    << " | detections="
                    << tracks.size()
                    << " | inference="
                    << inference_ms
                    << " ms"
                    << std::endl;

                for (const auto& track : tracks)
                {
                    std::cout
                        << "  class="
                        << track.class_id
                        << " track_id="
			<< track.track_id
			<< " confidence="
                        << track.confidence
                        << " box=("
                        << track.box.x
                        << ", "
                        << track.box.y
                        << ", "
                        << track.box.width
                        << ", "
                        << track.box.height
                        << ")"
                        << std::endl;
                }

                // ========================================
                // 5. 序列化
                // ========================================

                std::vector<uint8_t> result =
                    DetectionSerializer::serialize(
                        tracks
                    );

                // ========================================
                // 6. 返回 Python
                // ========================================

                if (!server.sendMessage(result))
                {
                    std::cerr
                        << "[TCP] Failed to send result."
                        << std::endl;

                    server.closeClient();
                    break;
                }
            }
        }
    }
    catch (const std::exception& e)
    {
        std::cerr
            << "[FATAL] "
            << e.what()
            << std::endl;

        return 1;
    }

    return 0;
}
