#ifndef __RETINAFACE_H__
#define __RETINAFACE_H__
#include <stdio.h>
#include <fstream>
#include <map>
#include <sys/time.h>
#include <opencv2/opencv.hpp>
#include <tvm/runtime/module.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/packed_func.h>
#include <dlpack/dlpack.h>

const size_t inWidth = 1024;
const size_t inHeight = 1024;
const size_t maxSize = 1980;
constexpr int dtype_code = kDLFloat;
constexpr int dtype_bits = 32;
constexpr int dtype_lanes = 1;
constexpr int device_type = kDLCPU;
constexpr int device_id = 0;
constexpr int in_ndim = 4;
constexpr int out_num = 9;

const char* outputs[out_num] = {"face_rpn_cls_prob_reshape_stride32_output", "face_rpn_bbox_pred_stride32_output", "face_rpn_landmark_pred_stride32_output", "face_rpn_cls_prob_reshape_stride16_output", "face_rpn_bbox_pred_stride16_output", "face_rpn_landmark_pred_stride16_output", "face_rpn_cls_prob_reshape_stride8_output", "face_rpn_bbox_pred_stride8_output", "face_rpn_landmark_pred_stride8_output"};
const float WHRatio = inWidth / (float)inHeight;
bool nocrop = false;

// Retinaface Deploy of TVM
class FR_RFN_Deploy{
    private:
        void *handle;
    public:
        FR_RFN_Deploy(std::string);
        ~FR_RFN_Deploy();

        std::map<const char*, cv::Mat> forward(cv::Mat& img, float threshold, std::vector<float>& scales, bool do_flip);
};

inline float CosineDistance(const cv::Mat &v1, const cv::Mat &v2)
{
    return static_cast<float>(v1.dot(v2));
}
#endif /*__RETINAFACE_H__*/
