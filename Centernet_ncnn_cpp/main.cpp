#include <opencv2/opencv.hpp>
#include <net.h>
#include <mat.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <cpu.h>
#include <chrono>
#include <filesystem>

namespace fs = std::filesystem;
using namespace std;
using namespace cv;

const int input_width = 320;
const int input_height = 320;
const int MODEL_SCALE = 4;
const float threshold_ = 0.2;

struct Box {
    float xmin, ymin, xmax, ymax, score, label;
};


// Calculate Intersection over Union (IoU)
float iou(const Box& a, const Box& b) {
    float xx1 = std::max(a.xmin, b.xmin);
    float yy1 = std::max(a.ymin, b.ymin);
    float xx2 = std::min(a.xmax, b.xmax);
    float yy2 = std::min(a.ymax, b.ymax);

    float width = std::max(0.0f, xx2 - xx1);
    float height = std::max(0.0f, yy2 - yy1);

    float intersection = width * height;
    float union_area = (a.xmax - a.xmin) * (a.ymax - a.ymin) + (b.xmax - b.xmin) * (b.ymax - b.ymin) - intersection;

    return intersection / union_area;  // IoU formula
}

float calculateArea(const Box& box) {
    return (box.xmax - box.xmin) * (box.ymax - box.ymin);
}

// Non-Maximum Suppression
std::vector<Box> non_maximum_suppression(const std::vector<Box>& boxes, float iou_threshold) {
    std::vector<Box> result;
    std::vector<Box> boxes_sorted = boxes;

    // Sort boxes by score
    std::sort(boxes_sorted.begin(), boxes_sorted.end(), [](const Box& a, const Box& b) {
        return a.score > b.score;
    });

    while (!boxes_sorted.empty()) {
        Box current = boxes_sorted.front();
        result.push_back(current);

        // Remove boxes with IoU greater than the threshold
        std::vector<Box> remaining_boxes;
        for (size_t i = 1; i < boxes_sorted.size(); ++i) {
            if (iou(current, boxes_sorted[i]) <= iou_threshold) {
                remaining_boxes.push_back(boxes_sorted[i]);
            }
        }
        boxes_sorted = remaining_boxes;
    }

    return result;
}



ncnn::Mat preprocess_image(const string& img_path, const Size& target_size = Size(input_width, input_height)) {
    Mat img = imread(img_path);
    if (img.empty()) {
        throw runtime_error("Image not found at path: " + img_path);
    }

    // Get original image dimensions
    int h = img.rows;
    int w = img.cols;

    // Convert the image to ncnn Mat and resize
    ncnn::Mat ncnn_img = ncnn::Mat::from_pixels_resize(
        img.data, ncnn::Mat::PIXEL_BGR, w, h, target_size.width, target_size.height
    );

    // Define mean and standard deviation values
    const float mean_vals[3] = {0.485f * 255.0f, 0.456f * 255.0f, 0.406f * 255.0f};
    const float norm_vals[3] = {1.0f / (0.229f * 255.0f), 1.0f / (0.224f * 255.0f), 1.0f / (0.225f * 255.0f)};

    // Apply mean subtraction and normalization
    ncnn_img.substract_mean_normalize(mean_vals, norm_vals);

    return ncnn_img;
}

int main() {
    ncnn::Option opt;
    opt.use_int8_storage = true;
    opt.use_int8_arithmetic = true;
    opt.num_threads = 4;

    ncnn::Net net;
    net.opt = opt;

    // Load the NCNN model
    if (net.load_param("best_epoch_weights_mbv2_shufflenet_fe_traced.ncnn-int8.param") != 0 ||
        net.load_model("best_epoch_weights_mbv2_shufflenet_fe_traced.ncnn-int8.bin") != 0) {
        cerr << "Failed to load model files." << endl;
        return -1;
    }

    // vector<string> test_folder = {
    //     "/mnt/e/derpetv5_xml/val_images/57_vlcsnap-2023-04-06-13h50m25s895.png"
    // };
    vector<string> test_folder;
    string dir_path = "/mnt/e/ComfyUI_windows_portable_nvidia/ComfyUI_windows_portable/ComfyUI/output";
    // Iterate through the directory
    for (const auto& entry : fs::directory_iterator(dir_path)) {
        // Check if the file is a regular file and has a .jpg extension
        if (entry.is_regular_file() && entry.path().extension() == ".jpg") {
            // Append the file path to the vector
            test_folder.push_back(entry.path().string());
        }

        if (entry.is_regular_file() && entry.path().extension() == ".png") {
            // Append the file path to the vector
            test_folder.push_back(entry.path().string());
        }

        if (entry.is_regular_file() && entry.path().extension() == ".JPG") {
            // Append the file path to the vector
            test_folder.push_back(entry.path().string());
        }
    }

    int photo_count = 0;
    for (const auto& img_path : test_folder) {
        std::cout<< img_path;
        Mat img = imread(img_path);
        Mat image = img.clone();
        resize(img,img,cv::Size(input_width,input_height));
        ncnn::Mat mat_in = preprocess_image(img_path);
        vector<vector<Box>> boxes;
        auto start_time = std::chrono::high_resolution_clock::now();
        ncnn::Extractor ex = net.create_extractor();
        ex.input("in0", mat_in);

        ncnn::Mat hm, wh, offset;
        ex.extract("out0", hm);
        ex.extract("out1", wh);
        ex.extract("out2", offset);
        auto end_time = std::chrono::high_resolution_clock::now();
        double elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
        double fps = 1 / (elapsed_time / 1000.0);
        std::cout << "FPS: " << fps << " frames/second" << std::endl;

        float* hm_data = (float*)hm.data;
        float* wh_data = (float*)wh.data;
        float* offset_data = (float*)offset.data;


        vector<Box> box_vec;
        for (int y = 0; y < hm.h; y++) {
            for (int x=0; x < hm.w; x++) {
                for (int c=0; c < hm.c; c++) {
                if  (hm_data[c * hm.h * hm.w + y * hm.w + x] > threshold_ ) {
                    float score =  hm_data[c * hm.h * hm.w + y * hm.w + x];
                    float width = wh_data[0 * hm.h * hm.w + y * hm.w + x];  // Predicted width
                    float height = wh_data[1 * hm.h * hm.w + y * hm.w + x];  // Predicted height

                    float offset_x = offset_data[0 * hm.h * hm.w + y * hm.w + x];  // Offset in x
                    float offset_y = offset_data[1 * hm.h * hm.w + y * hm.w + x];  // Offset in y
                    float label = c;

                    // Calculate bounding box coordinates
                    //std::cout << "x:" << x << std::endl;
                    float xmin = ( (x + offset_x) - (width / 2) ) * 4;  // striding factor (if applicable)
                    float ymin = ( (y + offset_y) - (height / 2) ) * 4;
                    float xmax = ( (x + offset_x) + (width / 2) ) * 4;
                    float ymax = ( (y + offset_y) + (height / 2) ) * 4;

                    // Print the bounding box and score
                    // std::cout << "Score: " << score << " - " << label << "-" 
                    //           << "xmin: " << xmin << ", ymin: " << ymin << ", xmax: " << xmax << ", ymax: " << ymax << std::endl;
                    // rectangle(img, Point(xmin,ymin), Point(xmax,ymax), 
                    // Scalar(255, 0, 0), 
                    // 3, LINE_8);
                    //boxes.push_back({xmin,ymin,xmax,ymax,label,score});
                    Box box_;
                    box_.xmin = xmin;
                    box_.ymin = ymin;
                    box_.xmax = xmax;
                    box_.ymax = ymax;
                    box_.score = score;
                    box_.label = label;
                    box_vec.push_back(box_);
                    boxes.push_back(box_vec);
                }
            }
            
        }
        }
        //vector<vector<Box>> boxes_sorted = boxes;
         vector<Box> final_boxes = non_maximum_suppression(box_vec, 0.3);
         for (const auto& box : final_boxes) {
             float xmin = (box.xmin * image.cols) / input_width ;
             float ymin = (box.ymin * image.rows) / input_height ;
             float xmax = (box.xmax * image.cols) / input_width ;
             float ymax = (box.ymax * image.rows) / input_height ;
             int label = (int) box.label;
             float score = box.score;
        // std::cout << "Box: [" << box.xmin << ", " << box.ymin << ", " << box.xmax << ", " << box.ymax << "] Score: " << box.score << std::endl;
        // rectangle(img, Point(box.xmin,box.ymin), Point(box.xmax,box.ymax), 
        //             Scalar(255, 0, 0), 
        //             3, LINE_8);
        rectangle(image, Point(xmin,ymin), Point(xmax,ymax), 
                    Scalar(255, 0, 0), 
                    3, LINE_8);
        putText(image, //target image
            to_string(label) + " Score: " + to_string(score), //text
            Point(xmin,ymin-4), //top-left position
            FONT_HERSHEY_DUPLEX,
            1.0,
            CV_RGB(118, 185, 0), //font color
            2);

    }

        
        //imshow("Image", image);
        //cv::imwrite("res/"+ to_string(photo_count)+".jpg",img);
        cv::imwrite("res/"+ to_string(photo_count)+"_orig.jpg",image);
        //waitKey(0);   
        photo_count += 1;
    }

    return 0;
}

