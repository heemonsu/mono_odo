#ifndef MONO_VO_H
#define MONO_VO_H


#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/features2d.hpp"


using namespace cv;
using namespace std;

class mono_vo {

public:

	mono_vo(){}
	virtual ~mono_vo(){}
	mono_vo(const Mat& _k);

	void Detect_Features(Mat img_1, vector<Point2f>& points1, vector<KeyPoint>& keypoints_1, Mat descriptors_1, string type);

	void Detect_Match_Features(Mat img_1, Mat img_2, vector<Point2f>& match_points1, vector<Point2f>& match_points2, string type);

	vector<Point3f> get3D_Points_0(const vector<Point2f>& feature_p1, const vector<Point2f>& feature_p2);
	
	vector<Point3f> get3D_Points_1(const vector<Point2f>& feature_p1, const vector<Point2f>& feature_p2, vector<Point2f>& prevFeatures, Mat inv_transform);

	void featureTracking_0(Mat img_1, Mat img_2, vector<Point2f>& points1, vector<Point2f>& points2, vector<uchar>& status);

	void featureTracking_1(Mat img_1, Mat img_2, vector<Point2f>& points1, vector<Point2f>& points2, vector<uchar>& status, const vector<Point3f>& landmarks, vector<Point3f>& landmarks_ref, 		vector<Point2f>&featurePoints_ref);

	void Continious(const string& filename1, const string& filename2,  const vector<vector<float>>& poses);

	cv::Mat getK() const {return K;}
	void setK(const cv::Mat& _K) {K = _K;}
	void set_Max_frame(int maxframe) {max_frame = maxframe;}


private:

	Mat K;
	int max_frame;

};


#endif
