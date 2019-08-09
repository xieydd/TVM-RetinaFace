#include "retinaface.h"

// Retinaface Deploy of TVM
FR_RFN_Deploy::FR_RFN_Deploy(std::string modelFolder)
{
    tvm::runtime::Module mod_syslib = tvm::runtime::Module::LoadFromFile(modelFolder + "/deploy_lib.so");
    //load graph
    std::ifstream json_in(modelFolder + "/deploy_graph.json");
    std::string json_data((std::istreambuf_iterator<char>(json_in)), std::istreambuf_iterator<char>());
    json_in.close();

    int device_type = kDLCPU;
    int device_id = 0;
    // get global function module for graph runtime 
    tvm::runtime::Module module = (*tvm::runtime::Registry::Get("tvm.graph_runtime.create"))(json_data, mod_syslib, device_type, device_id);
    FR_RFN_Deploy::handle = new tvm::runtime::Module(module);

    // load params
	std::ifstream params_in(modelFolder + "/deploy_param.params", std::ios::binary);
    std::string params_data((std::istreambuf_iterator<char>(params_in)), std::istreambuf_iterator<char>());
    params_in.close();

	TVMByteArray params_arr;
    params_arr.data = params_data.c_str();
    params_arr.size = params_data.length();
    tvm::runtime::PackedFunc load_params = module.GetFunction("load_params");
    load_params(params_arr);
}

std::map<const char*, cv::Mat> FR_RFN_Deploy::forward(cv::Mat& img, float threshold, std::vector<float> & scales, bool do_flip)
{
    std::vector<int> flips;
    cv::Mat im;
    cv::Mat dst;
    cv::Mat tensor; 
    DLTensor* input;
    std::map<const char* , cv::Mat> output_map;

    flips.push_back(0);
    if (do_flip) {
        flips.push_back(1);
    }

    for (int i = 0; i < scales.size(); i++) {
        for (int j = 0; j < flips.size(); j++) {
            if (scales[i] != 1.0) {
		// set CV_INTER_LINEAR to 1
                cv::resize(img, im,cv::Size(0,0), scales[i], scales[i], 1);
            } else 
            {
                im = img.clone();
            }

            if (flips[j]) {
                cv::flip(im, im, -1); // <0 x,y flip >0 x:<0 y:>0  
            }

            int h = im.rows;
            int w = im.cols;
            // keep rows and cols be multipe of 32, dosen`t Impact coordinates.
            if(nocrop)
            {
                if (im.rows % 32 == 0) { h = im.rows;}
                else {h = (im.rows / 32 + 1)*32;}

                if (im.cols % 32 == 0) { w = im.cols;}
                else {h = (im.cols / 32 + 1)*32;}
               
		// set  cv::BORDER_CONSTANT to 0
                cv::copyMakeBorder(im, dst, 0, h - im.rows, 0,w - im.cols, cv::BORDER_CONSTANT, cv::Scalar(0,0,0));
                im = dst.clone();
            }
            //for batch image files // cv::Mat tensor = cv::dnn::blobFromImages(inputImagesAligned,1.0,cv::Size(112,112),cv::Scalar(0,0,0),true);
            /*
            1. mean subtraction: Mean subtraction is used to help combat illumination changes in the input images in our dataset.For ImageNet dataset, R=103.93, G=116.77, and B=123.68, we use 0 0 0 
            2. standard deviation set to 1.0
            3. swapRB: Change RGB to BGR for opencv
            */
            tensor = cv::dnn::blobFromImage(im,1.0,cv::Size(im.rows, im.cols),cv::Scalar(0,0,0),true);
            // change uint8 input to float32 and convert to RGB via opencv dnn function
            const int64_t in_shape[in_ndim] = {1, 3, im.rows, im.cols};
            TVMArrayAlloc(in_shape, in_ndim, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &input);
            TVMArrayCopyFromBytes(input,tensor.data,im.rows*3*im.cols*4);
            tvm::runtime::Module* mod = (tvm::runtime::Module*)FR_RFN_Deploy::handle;
            tvm::runtime::PackedFunc set_input = mod->GetFunction("set_input");
            set_input("data", input);
            tvm::runtime::PackedFunc run = mod->GetFunction("run");
            run();
            tvm::runtime::PackedFunc get_output = mod->GetFunction("get_output");
            for (int i = 0; i < out_num; i++) {
                tvm::runtime::NDArray res = get_output(i);
                //std::vector<int64_t> shapes= res.Shape();
                //for(int64_t shape: shapes){
                //	printf("%ld\n", shape);
                //}
                        //DLTensor dltensor;
                        //res.CopyTo(&dltensor);
                //printf("%ld\n", tvm::runtime::GetDataSize(res->data_->dl_tensor));
                size_t size = 1;
                printf("dims: %d \n", res->ndim);
                for (int i = 0; i < res->ndim; i++) {
                    size *= static_cast<size_t>(res->shape[i]);
                }
                //printf("%ld\n", tvm::runtime::GetDataSize(res->ndim));
                printf("%ld\n", size);
                cv::Mat vector(128,1,CV_32F);
                memcpy(vector.data, res->data, 128*4);
                output_map[outputs[i]] = vector;
            }
            TVMArrayFree(input);
        }
    }
    return output_map;

        //            cv::Size img_size = inputImageAligned.size();
        //            cv::Size cropSize;
        //            if (img_size.width / (float)img_size.height > WHRatio) {
        //                cropSize = cv::Size(img_size.width, static_cast<int>(img_size.width / WHRatio));
        //            }
        //            else
        //            {
        //                cropSize = cv::Size(static_cast<int>(img_size.height / WHRatio), img_size.width);
        //            }
        //            cv::Rect crop(cv::Point((img_size.width - cropSize.width)/2, (img_size.height - cropSize.height)/2), cropSize);
        //
                    //for batch image files // cv::Mat tensor = cv::dnn::blobFromImages(inputImagesAligned,1.0,cv::Size(112,112),cv::Scalar(0,0,0),true);
                    //cv::Mat tensor = cv::dnn::blobFromImage(inputImageAligned,1.0,cv::Size(inHeight,inWidth),cv::Scalar(0,0,0),true);
    // change uint8 input to float32 and convert to RGB via opencv dnn function
    //DLTensor* input;
    //constexpr int dtype_code = kDLFloat;
    //constexpr int dtype_bits = 32;
    //constexpr int dtype_lanes = 1;
    //constexpr int device_type = kDLCPU;
    //constexpr int device_id = 0;
    //constexpr int in_ndim = 4;
    //constexpr int out_num = 9;
    //const int64_t in_shape[in_ndim] = {1, 3, inHeight, inWidth};
    //TVMArrayAlloc(in_shape, in_ndim, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &input);
    //TVMArrayCopyFromBytes(input,tensor.data,inHeight*3*inWidth*4);
    //tvm::runtime::Module* mod = (tvm::runtime::Module*)handle;
    //tvm::runtime::PackedFunc set_input = mod->GetFunction("set_input");
    //set_input("data", input);
    //tvm::runtime::PackedFunc run = mod->GetFunction("run");
    //run();
    //tvm::runtime::PackedFunc get_output = mod->GetFunction("get_output");
    //tvm::runtime::NDArray res = get_output(0);
    //cv::Mat vector(128,1,CV_32F);
    //memcpy(vector.data, res->data, 128*4);
    //cv::Mat _l2;
    // normlize
    //cv::multiply(vector,vector,_l2);
    //float l2 = cv::sqrt(cv::sum(_l2).val[0]);
    //vector = vector / l2;

    //std::map<const char* , cv::Mat> output_map;
    //const char* outputs[out_num] = {"face_rpn_cls_prob_reshape_stride32_output", "face_rpn_bbox_pred_stride32_output", "face_rpn_landmark_pred_stride32_output", "face_rpn_cls_prob_reshape_stride16_output", "face_rpn_bbox_pred_stride16_output", "face_rpn_landmark_pred_stride16_output", "face_rpn_cls_prob_reshape_stride8_output", "face_rpn_bbox_pred_stride8_output", "face_rpn_landmark_pred_stride8_output"};
    //for (int i = 0; i < out_num; i++) {
    //    tvm::runtime::NDArray res = get_output(i);
    //    //std::vector<int64_t> shapes= res.Shape();
    //    //for(int64_t shape: shapes){
    //    //	printf("%ld\n", shape);
    //    //}
    //            //DLTensor dltensor;
    //            //res.CopyTo(&dltensor);
    //    //printf("%ld\n", tvm::runtime::GetDataSize(res->data_->dl_tensor));
    //    size_t size = 1;
    //    printf("dims: %d \n", res->ndim);
    //    for (int i = 0; i < res->ndim; i++) {
    //        size *= static_cast<size_t>(res->shape[i]);
    //    }
    //    //printf("%ld\n", tvm::runtime::GetDataSize(res->ndim));
    //    printf("%ld\n", size);
    //        cv::Mat vector(128,1,CV_32F);
    //        memcpy(vector.data, res->data, 128*4);
    //        output_map[outputs[i]] = vector;
    //    }
    //
    //TVMArrayFree(input);
    //return output_map;
}

FR_RFN_Deploy::~FR_RFN_Deploy() {FR_RFN_Deploy::handle = NULL;}
