#include <opencv2/opencv.hpp>
#include "retinaface.h"
#include "time.h"

int main(int argc, char *argv[])
{
    if(argc != 3) {
        fprintf(stderr, "args is not enough, we need 2 args: modelpath dir, imagepath");
        return -1;
    }

    const char* modelpath = argv[1];
    const char* imagepath = argv[2];
    float threshold = 0.8;
    std::vector<float> scales;
    float imscale;
    struct timeval start, end;
    long long start_time, end_time;
    bool do_flip = false;

    cv::Mat img = cv::imread(imagepath, CV_LOAD_IMAGE_COLOR);
    printf("Image size is rows: %d, cols: %d, dims: %d, channels: %d.\n", img.rows, img.cols, img.dims, img.channels());
    if(img.empty()){ 
        fprintf(stderr, "cv::imread %s failed\n", imagepath);
        return -1;
    }
    //cv::Mat img2 = cv::imread("./face1.jpg");
    FR_RFN_Deploy deploy(modelpath);

    int im_size_min = cv::min(img.rows, img.cols);
    int im_size_max = cv::max(img.rows, img.cols);
    float im_scale = float(inWidth) / float(im_size_min);
    if(round(im_scale * im_size_max) > maxSize) {
        im_scale = float(maxSize) / float(im_size_max);
    }
    
    printf("im_scale: %f", im_scale);
    // in c++11, we can change push_back to emplace_back to improve the performance.
    scales.push_back(im_scale);
    gettimeofday(&start, NULL);
    start_time = (long long)start.tv_sec*1000 + start.tv_usec/1000;
    std::map<const char*, cv::Mat> result = deploy.forward(img, threshold, scales, do_flip);
    gettimeofday(&end, NULL);
    end_time = (long long)end.tv_sec*1000 + end.tv_usec/1000;
    printf("Infrence time: %f ms\n", (double(end_time- start_time)));
    //cv::Mat result2 = deploy.forward(img2);
    //std::cout<<CosineDistance(result1,result2)<<std::endl;
    cv::Mat result_mat = result["face_rpn_cls_prob_reshape_stride32_output"];
    for (int i = 0; i < 128; i++) {
        printf("%f \t\t", result_mat.at<float>(i,0));
    }
    return 0;
}
