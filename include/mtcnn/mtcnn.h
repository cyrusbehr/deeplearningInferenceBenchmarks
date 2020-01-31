#ifndef __MTCNN_H__
#define __MTCNN_H__

#include <vector>
#include "ncnn/net.h"

using namespace std;

struct Bbox
{
    float score;
    int x1;
    int y1;
    int x2;
    int y2;
    float area;
    bool exist;
    float ppoint[10];
    float regreCoord[4];
};

struct orderScore
{
    float score;
    int oriOrder;
};


class MTCNN{
public:
    MTCNN();
    void detect(ncnn::Mat& img_, std::vector<Bbox>& finalBbox, const float newThresholds[3]);

private:
    void generateBbox(ncnn::Mat score, ncnn::Mat location, vector<Bbox>& boundingBox_, vector<orderScore>& bboxScore_, float scale);
    void nms(vector<Bbox> &boundingBox_, std::vector<orderScore> &bboxScore_, const float overlap_threshold, string modelname="Union");
    void refineAndSquareBbox(vector<Bbox> &vecBbox, const int &height, const int &width);

    ncnn::Net pnet_, rnet_, onet_;
    ncnn::Mat img;

    float threshold[3] = {0.04, 0.04, 0.04};
    const float nms_threshold[3] = {0.5, 0.7, 0.7};
    const float mean_vals[3] = {127.5, 127.5, 127.5};
    const float norm_vals[3] = {0.0078125, 0.0078125, 0.0078125};
    std::vector<Bbox> firstBbox_, secondBbox_,thirdBbox_;
    
    std::vector<orderScore> firstOrderScore_, secondBboxScore_, thirdBboxScore_;
    int img_w, img_h;
};


#endif