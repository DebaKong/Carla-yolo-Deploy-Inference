#include "yolov8.h"

#include <opencv2/opencv.hpp>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;


// ============================================================
// COCO 80 类
// ============================================================

constexpr int NUM_CLASSES = 80;


// ============================================================
// Ground Truth
// ============================================================

struct GroundTruth
{
    int class_id;

    cv::Rect2f box;
};


// ============================================================
// Prediction
// ============================================================

struct Prediction
{
    int class_id;

    float confidence;

    cv::Rect2f box;

    bool matched = false;
};


// ============================================================
// 计算 IoU
// ============================================================

float calculateIoU(
    const cv::Rect2f& a,
    const cv::Rect2f& b)
{
    float x1 = std::max(a.x, b.x);
    float y1 = std::max(a.y, b.y);

    float x2 = std::min(
        a.x + a.width,
        b.x + b.width
    );

    float y2 = std::min(
        a.y + a.height,
        b.y + b.height
    );

    float w = std::max(
        0.0f,
        x2 - x1
    );

    float h = std::max(
        0.0f,
        y2 - y1
    );

    float intersection = w * h;

    float union_area =
        a.area() +
        b.area() -
        intersection;

    if (union_area <= 0.0f)
        return 0.0f;

    return intersection / union_area;
}


// ============================================================
// 读取 YOLO Ground Truth
//
// 格式：
// class x_center y_center width height
// ============================================================

std::vector<GroundTruth> loadGroundTruth(
    const std::string& label_path,
    int image_width,
    int image_height)
{
    std::vector<GroundTruth> result;

    std::ifstream file(label_path);

    if (!file.is_open())
    {
        return result;
    }

    std::string line;

    while (std::getline(file, line))
    {
        if (line.empty())
            continue;

        std::stringstream ss(line);

        int class_id;

        float cx;
        float cy;
        float w;
        float h;

        ss >> class_id
           >> cx
           >> cy
           >> w
           >> h;

        if (ss.fail())
            continue;

        float x =
            (cx - w / 2.0f)
            * image_width;

        float y =
            (cy - h / 2.0f)
            * image_height;

        float width =
            w * image_width;

        float height =
            h * image_height;

        x = std::max(
            0.0f,
            std::min(
                x,
                static_cast<float>(
                    image_width - 1
                )
            )
        );

        y = std::max(
            0.0f,
            std::min(
                y,
                static_cast<float>(
                    image_height - 1
                )
            )
        );

        width = std::min(
            width,
            static_cast<float>(
                image_width - x
            )
        );

        height = std::min(
            height,
            static_cast<float>(
                image_height - y
            )
        );

        if (width <= 0.0f ||
            height <= 0.0f)
        {
            continue;
        }

        GroundTruth gt;

        gt.class_id = class_id;

        gt.box = cv::Rect2f(
            x,
            y,
            width,
            height
        );

        result.push_back(gt);
    }

    return result;
}


// ============================================================
// 将 Detection 转换成 Prediction
// ============================================================

std::vector<Prediction> convertPredictions(
    const std::vector<Detection>& detections)
{
    std::vector<Prediction> result;

    result.reserve(
        detections.size()
    );

    for (const auto& detection :
         detections)
    {
        Prediction prediction;

        prediction.class_id =
            detection.class_id;

        prediction.confidence =
            detection.confidence;

        prediction.box =
            cv::Rect2f(
                static_cast<float>(
                    detection.box.x
                ),
                static_cast<float>(
                    detection.box.y
                ),
                static_cast<float>(
                    detection.box.width
                ),
                static_cast<float>(
                    detection.box.height
                )
            );

        prediction.matched = false;

        result.push_back(
            prediction
        );
    }

    return result;
}


// ============================================================
// 单个 IoU 阈值下的 AP
// ============================================================

struct DetectionRecord
{
    float confidence;

    bool true_positive;
};


double calculateAP(
    std::vector<DetectionRecord> records,
    int total_ground_truth)
{
    if (total_ground_truth == 0)
    {
        return -1.0;
    }

    if (records.empty())
    {
        return 0.0;
    }

    // 按 confidence 从高到低
    std::sort(
        records.begin(),
        records.end(),
        [](const DetectionRecord& a,
           const DetectionRecord& b)
        {
            return a.confidence >
                   b.confidence;
        }
    );

    std::vector<double> precision(
        records.size()
    );

    std::vector<double> recall(
        records.size()
    );

    int tp = 0;
    int fp = 0;

    for (size_t i = 0;
         i < records.size();
         ++i)
    {
        if (records[i].true_positive)
            tp++;
        else
            fp++;

        precision[i] =
            static_cast<double>(tp) /
            static_cast<double>(tp + fp);

        recall[i] =
            static_cast<double>(tp) /
            static_cast<double>(
                total_ground_truth
            );
    }

    // 101-point interpolation
    double ap = 0.0;

    for (int r = 0; r <= 100; ++r)
    {
        double recall_threshold =
            r / 100.0;

        double max_precision = 0.0;

        for (size_t i = 0;
             i < precision.size();
             ++i)
        {
            if (recall[i] >=
                recall_threshold)
            {
                max_precision =
                    std::max(
                        max_precision,
                        precision[i]
                    );
            }
        }

        ap += max_precision;
    }

    ap /= 101.0;

    return ap;
}


// ============================================================
// 计算指定 IoU 阈值下的 TP / FP / FN
// ============================================================

void evaluateImage(
    const std::vector<GroundTruth>& gt,
    const std::vector<Prediction>& predictions,
    float iou_threshold,
    std::vector<std::vector<DetectionRecord>>& records,
    std::vector<int>& gt_count)
{
    std::vector<bool> gt_used(
        gt.size(),
        false
    );

    // 每个类别分别处理
    for (int class_id = 0;
         class_id < NUM_CLASSES;
         ++class_id)
    {
        std::vector<int> gt_indices;

        for (size_t i = 0;
             i < gt.size();
             ++i)
        {
            if (gt[i].class_id == class_id)
            {
                gt_indices.push_back(
                    static_cast<int>(i)
                );
            }
        }

        gt_count[class_id] +=
            static_cast<int>(gt_indices.size());

        std::vector<int> pred_indices;

        for (size_t i = 0;
             i < predictions.size();
             ++i)
        {
            if (predictions[i].class_id == class_id)
            {
                pred_indices.push_back(
                    static_cast<int>(i)
                );
            }
        }

        // 按置信度从高到低排序
        std::sort(
            pred_indices.begin(),
            pred_indices.end(),
            [&](int a, int b)
            {
                return predictions[a].confidence >
                       predictions[b].confidence;
            }
        );

        for (int pred_index : pred_indices)
        {
            float best_iou = 0.0f;
            int best_gt = -1;

            for (int gt_index : gt_indices)
            {
                if (gt_used[gt_index])
                    continue;

                float iou =
                    calculateIoU(
                        predictions[pred_index].box,
                        gt[gt_index].box
                    );

                if (iou > best_iou)
                {
                    best_iou = iou;
                    best_gt = gt_index;
                }
            }

            DetectionRecord record;

            record.confidence =
                predictions[pred_index].confidence;

            if (best_gt >= 0 &&
                best_iou >= iou_threshold)
            {
                gt_used[best_gt] = true;
                record.true_positive = true;
            }
            else
            {
                record.true_positive = false;
            }

            // 关键修改：
            // 按类别保存 DetectionRecord
            records[class_id].push_back(record);
        }
    }
}


// ============================================================
// 计算 Precision / Recall
// ============================================================

struct PRResult
{
    long long TP = 0;
    long long FP = 0;
    long long FN = 0;

    double precision = 0.0;
    double recall = 0.0;
};


PRResult calculatePR(
    const std::vector<
        std::vector<GroundTruth>
    >& all_gt,

    const std::vector<
        std::vector<Prediction>
    >& all_predictions,

    float iou_threshold)
{
    PRResult result;

    for (size_t image_index = 0;
         image_index < all_gt.size();
         ++image_index)
    {
        const auto& gt =
            all_gt[image_index];

        const auto& predictions =
            all_predictions[image_index];

        std::vector<bool> gt_used(
            gt.size(),
            false
        );

        std::vector<int> pred_indices(
            predictions.size()
        );

        std::iota(
            pred_indices.begin(),
            pred_indices.end(),
            0
        );

        std::sort(
            pred_indices.begin(),
            pred_indices.end(),
            [&](int a, int b)
            {
                return predictions[a].confidence >
                       predictions[b].confidence;
            }
        );

        for (int pred_index :
             pred_indices)
        {
            float best_iou = 0.0f;
            int best_gt = -1;

            for (size_t j = 0;
                 j < gt.size();
                 ++j)
            {
                if (gt_used[j])
                    continue;

                if (
                    gt[j].class_id !=
                    predictions[pred_index].class_id
                )
                {
                    continue;
                }

                float iou =
                    calculateIoU(
                        gt[j].box,
                        predictions[pred_index].box
                    );

                if (iou > best_iou)
                {
                    best_iou = iou;
                    best_gt =
                        static_cast<int>(j);
                }
            }

            if (best_gt >= 0 &&
                best_iou >= iou_threshold)
            {
                gt_used[best_gt] = true;

                result.TP++;
            }
            else
            {
                result.FP++;
            }
        }

        for (bool used : gt_used)
        {
            if (!used)
                result.FN++;
        }
    }

    const double denominator_precision =
        static_cast<double>(
            result.TP + result.FP
        );

    const double denominator_recall =
        static_cast<double>(
            result.TP + result.FN
        );

    if (denominator_precision > 0.0)
    {
        result.precision =
            result.TP /
            denominator_precision;
    }

    if (denominator_recall > 0.0)
    {
        result.recall =
            result.TP /
            denominator_recall;
    }

    return result;
}


// ============================================================
// 主程序
// ============================================================

int main(
    int argc,
    char* argv[])
{
    if (argc < 2)
    {
        std::cerr
            << "Usage:\n"
            << "  ./evaluate_map <engine_path>\n\n"
            << "Example:\n"
            << "  ./evaluate_map assets/yolov8n.engine\n"
            << "  ./evaluate_map assets/yolov8n_int8.engine\n";

        return -1;
    }

    const std::string engine_path =
        argv[1];

    const std::string image_dir =
        "assets/Calibration/images";

    const std::string label_dir =
        "assets/Calibration/labels";


    std::cout
        << "========================================\n"
        << "YOLOv8 Detection Accuracy Evaluation\n"
        << "========================================\n";

    std::cout
        << "Engine : "
        << engine_path
        << "\n";

    std::cout
        << "Images : "
        << image_dir
        << "\n";

    std::cout
        << "Labels : "
        << label_dir
        << "\n\n";


    // --------------------------------------------------------
    // 收集图片
    // --------------------------------------------------------

    std::vector<std::string> image_paths;

    for (const auto& entry :
         fs::directory_iterator(image_dir))
    {
        if (!entry.is_regular_file())
            continue;

        std::string ext =
            entry.path()
            .extension()
            .string();

        std::transform(
            ext.begin(),
            ext.end(),
            ext.begin(),
            ::tolower
        );

        if (ext == ".jpg" ||
            ext == ".jpeg" ||
            ext == ".png")
        {
            image_paths.push_back(
                entry.path().string()
            );
        }
    }

    std::sort(
        image_paths.begin(),
        image_paths.end()
    );

    if (image_paths.empty())
    {
        std::cerr
            << "No evaluation images found.\n";

        return -1;
    }


    // --------------------------------------------------------
    // 加载 TensorRT Engine
    // --------------------------------------------------------

    YOLOv8 detector(
        engine_path
    );


    // --------------------------------------------------------
    // 保存所有 GT / Prediction
    // --------------------------------------------------------

    std::vector<
        std::vector<GroundTruth>
    > all_gt;

    std::vector<
        std::vector<Prediction>
    > all_predictions;


    // --------------------------------------------------------
    // 推理
    // --------------------------------------------------------

    for (size_t i = 0;
         i < image_paths.size();
         ++i)
    {
        cv::Mat image =
            cv::imread(
                image_paths[i]
            );

        if (image.empty())
        {
            std::cerr
                << "\nFailed to read: "
                << image_paths[i]
                << "\n";

            continue;
        }


        // Ground Truth
        fs::path image_path(
            image_paths[i]
        );

        std::string filename =
            image_path
            .stem()
            .string();

        std::string label_path =
            (
                fs::path(label_dir) /
                (filename + ".txt")
            ).string();

        std::vector<GroundTruth> gt =
            loadGroundTruth(
                label_path,
                image.cols,
                image.rows
            );


        // TensorRT 推理
        std::vector<Detection> detections =
            detector.infer(
                image
            );


        std::vector<Prediction> predictions =
            convertPredictions(
                detections
            );


        all_gt.push_back(
            std::move(gt)
        );

        all_predictions.push_back(
            std::move(predictions)
        );


        std::cout
            << "\rProcessed: "
            << i + 1
            << " / "
            << image_paths.size()
            << std::flush;
    }

    std::cout << "\n\n";


    // ========================================================
    // Precision / Recall @ IoU 0.50
    // ========================================================

    PRResult pr50 =
        calculatePR(
            all_gt,
            all_predictions,
            0.50f
        );


    // ========================================================
    // mAP@0.50
    // ========================================================

    std::vector<
        std::vector<DetectionRecord>
    > records50(
        NUM_CLASSES
    );

    std::vector<int> gt_count50(
        NUM_CLASSES,
        0
    );


    for (size_t i = 0;
         i < all_gt.size();
         ++i)
    {
        evaluateImage(
            all_gt[i],
            all_predictions[i],
            0.50f,
            records50,
            gt_count50
        );
    }


    double map50 = 0.0;
    int valid_classes50 = 0;

    for (int c = 0;
         c < NUM_CLASSES;
         ++c)
    {
        if (gt_count50[c] == 0)
            continue;

        double ap =
            calculateAP(
                records50[c],
                gt_count50[c]
            );

        if (ap >= 0.0)
        {
            map50 += ap;
            valid_classes50++;
        }
    }

    if (valid_classes50 > 0)
    {
        map50 /=
            valid_classes50;
    }


    // ========================================================
    // mAP@0.50:0.95
    // ========================================================

    double map5095 = 0.0;
    int valid_map_thresholds = 0;

    for (int threshold_index = 0;
         threshold_index < 10;
         ++threshold_index)
    {
        float iou_threshold =
            0.50f +
            threshold_index * 0.05f;


        std::vector<
            std::vector<DetectionRecord>
        > records(
            NUM_CLASSES
        );

        std::vector<int> gt_count(
            NUM_CLASSES,
            0
        );


        for (size_t i = 0;
             i < all_gt.size();
             ++i)
        {
            evaluateImage(
                all_gt[i],
                all_predictions[i],
                iou_threshold,
                records,
                gt_count
            );
        }


        double map =
            0.0;

        int valid_classes = 0;


        for (int c = 0;
             c < NUM_CLASSES;
             ++c)
        {
            if (gt_count[c] == 0)
                continue;

            double ap =
                calculateAP(
                    records[c],
                    gt_count[c]
                );

            if (ap >= 0.0)
            {
                map += ap;
                valid_classes++;
            }
        }


        if (valid_classes > 0)
        {
            map /= valid_classes;

            map5095 += map;

            valid_map_thresholds++;
        }
    }


    if (valid_map_thresholds > 0)
    {
        map5095 /=
            valid_map_thresholds;
    }


    // ========================================================
    // 输出
    // ========================================================

    std::cout
        << "\n"
        << "========================================\n"
        << "Evaluation Result\n"
        << "========================================\n";


    std::cout
        << std::fixed
        << std::setprecision(4);


    std::cout
        << "Images                 : "
        << all_gt.size()
        << "\n";


    std::cout
        << "TP @ IoU=0.50          : "
        << pr50.TP
        << "\n";


    std::cout
        << "FP @ IoU=0.50          : "
        << pr50.FP
        << "\n";


    std::cout
        << "FN @ IoU=0.50          : "
        << pr50.FN
        << "\n";


    std::cout
        << "Precision @ IoU=0.50   : "
        << pr50.precision
        << "\n";


    std::cout
        << "Recall @ IoU=0.50      : "
        << pr50.recall
        << "\n";


    std::cout
        << "mAP@0.50               : "
        << map50
        << "\n";


    std::cout
        << "mAP@0.50:0.95          : "
        << map5095
        << "\n";


    std::cout
        << "========================================\n";

    return 0;
}