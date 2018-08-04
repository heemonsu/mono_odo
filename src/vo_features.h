#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"

#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"

#include "opencv2/xfeatures2d/nonfree.hpp"
#include "opencv2/xfeatures2d.hpp"

#include <iostream>
#include <ctype.h>
#include <algorithm> // for copy
#include <iterator> // for ostream_iterator
#include <vector>
#include <ctime>
#include <sstream>
#include <fstream>
#include <string>

using namespace cv;
using namespace std;
using namespace xfeatures2d;




/////////////////////////////////////////////////////////////- FEATURE DETECTION & MATCHING -//////////////////////////////////////////////////////////////////////////////

void Match_Features(Mat descriptors_1, Mat descriptors_2, vector<KeyPoint>& keypoints1, vector<KeyPoint>& keypoints2, vector<Point2f>& match_points1, vector<Point2f>& match_points2)  {

  FlannBasedMatcher matcher;
  std::vector< DMatch > matches;
  descriptors_1.convertTo(descriptors_1, CV_32F);
  descriptors_2.convertTo(descriptors_2, CV_32F);
  matcher.match( descriptors_1, descriptors_2, matches, 2 );
  //matcher.knnMatch(descriptors_1, descriptors_2, matches, 2);

  // cout << matches.size() << endl;

//  for (int i = 0; i < matches.size(); i++) {

//  const DMatch& bestMatch = matches[i][0];  
//  const DMatch& betterMatch = matches[i][1];  

//  if (bestMatch.distance < 0.7*betterMatch.distance) {
//	match_points1.push_back(keypoints1[bestMatch.queryIdx].pt);
//        match_points2.push_back(keypoints2[bestMatch.trainIdx].pt);
//        //bestMatches.push_back(bestMatch);
//	}
//
//    }

}


void Detect_Features(Mat img_1, vector<Point2f>& points1, vector<KeyPoint>& keypoints_1, Mat descriptors_1, string type)	{


 int fast_threshold = 20;
 bool nonmaxSuppression = true;

 if(type == "SIFT"){
  cv::Ptr<Feature2D> f2d = SIFT::create(2000, 3, 0.05, 20, 1);
  f2d->detect( img_1, keypoints_1 );
  f2d->compute( img_1, keypoints_1, descriptors_1 );
  KeyPoint::convert(keypoints_1, points1, vector<int>());
 }
 else if(type == "BRIEF"){
  cv::Ptr<Feature2D> extractor = BriefDescriptorExtractor::create();
  FAST(img_1, keypoints_1, fast_threshold, nonmaxSuppression);
  extractor->compute( img_1, keypoints_1, descriptors_1 );
  KeyPoint::convert(keypoints_1, points1, vector<int>());
 }
 else if(type == "ORB"){
  // C++: ORB::ORB(int nfeatures=500, float scaleFactor=1.2f, int nlevels=8, int edgeThreshold=31, int firstLevel=0, int WTA_K=2, int scoreType=ORB::HARRIS_SCORE, int patchSize=31)
  static Ptr<FeatureDetector> orb = ORB::create(500,1.2f,8,30,0,2,ORB::FAST_SCORE,30); 
  orb->detectAndCompute(img_1,cv::noArray(), keypoints_1, descriptors_1);
  KeyPoint::convert(keypoints_1, points1, vector<int>());
 }
 else if(type == "BRISK"){
  static Ptr<FeatureDetector> brisk = BRISK::create(20, 3, 1.0f);
  brisk->detectAndCompute(img_1,cv::noArray(), keypoints_1, descriptors_1);
  KeyPoint::convert(keypoints_1, points1, vector<int>());
 }
 else if(type == "AKAZE"){
  static Ptr<FeatureDetector> akaze = AKAZE::create(); //failing at 3rd turn, needs to be tuned
  akaze->detectAndCompute(img_1,cv::noArray(), keypoints_1, descriptors_1);
  KeyPoint::convert(keypoints_1, points1, vector<int>()); 
 }
}

void FAST_featureDetection(Mat img_1, vector<KeyPoint>& keypoints_1, vector<Point2f>& points1)	{   //uses FAST
  
  int fast_threshold = 20;
  bool nonmaxSuppression = true;
  FAST(img_1, keypoints_1, fast_threshold, nonmaxSuppression);
  KeyPoint::convert(keypoints_1, points1, vector<int>());
}

void SURF_featureDetection(Mat img_1, vector<Point2f>& points1)	{   //uses FAST
  vector<KeyPoint> keypoints_1;
  Mat descriptors_1;
  cv::Ptr<Feature2D> f2d = SIFT::create();
  f2d->detect( img_1, keypoints_1 );
  f2d->compute( img_1, keypoints_1, descriptors_1 );
  KeyPoint::convert(keypoints_1, points1, vector<int>());
}

void BRIEF_featureDetection(Mat img_1, vector<Point2f>& points1){   //uses BRIEF
  vector<KeyPoint> keypoints_1;
  Mat descriptors_1;
  cv::Ptr<Feature2D> extractor = BriefDescriptorExtractor::create();
  int fast_threshold = 20;
  bool nonmaxSuppression = true;
  FAST(img_1, keypoints_1, fast_threshold, nonmaxSuppression);
  extractor->compute( img_1, keypoints_1, descriptors_1 );
  KeyPoint::convert(keypoints_1, points1, vector<int>());
}

void ORB_featureDetection(Mat img_1, vector<Point2f>& points1)	{   //uses ORB 
  vector<KeyPoint> keypoints_1;
  Mat descriptors_1;
  static Ptr<FeatureDetector> orb = ORB::create(500,1.5f,8,31,0,2,ORB::HARRIS_SCORE,31); 
  orb->detectAndCompute(img_1,cv::noArray(), keypoints_1, descriptors_1);
  KeyPoint::convert(keypoints_1, points1, vector<int>());
}

void BRISK_featureDetection(Mat img_1, vector<Point2f>& points1) { //uses BRISK
  vector<KeyPoint> keypoints_1;
  Mat descriptors_1;
  static Ptr<FeatureDetector> brisk = BRISK::create(20, 3, 1.0f);
  brisk->detectAndCompute(img_1,cv::noArray(), keypoints_1, descriptors_1);
  KeyPoint::convert(keypoints_1, points1, vector<int>());  
}

void AKAZE_featureDetection(Mat img_1, vector<Point2f>& points1) { //uses AKAZE
  vector<KeyPoint> keypoints_1;
  Mat descriptors_1;
  static Ptr<FeatureDetector> akaze = AKAZE::create(); //failing at 3rd turn, needs to be tuned
  akaze->detectAndCompute(img_1,cv::noArray(), keypoints_1, descriptors_1);
  KeyPoint::convert(keypoints_1, points1, vector<int>());  
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////- ELLIPTICAL KEYPOINTS CONVERTION FOR BECHMARKING -/////////////////////////////////////////////////////////////

template<typename _Tp> static int solveQuadratic(_Tp a, _Tp b, _Tp c, _Tp& x1, _Tp& x2)
{
    if( a == 0 )
    {
        if( b == 0 )
        {
            x1 = x2 = 0;
            return c == 0;
        }
        x1 = x2 = -c/b;
        return 1;
    }

    _Tp d = b*b - 4*a*c;
    if( d < 0 )
    {
        x1 = x2 = 0;
        return 0;
    }
    if( d > 0 )
    {
        d = std::sqrt(d);
        double s = 1/(2*a);
        x1 = (-b - d)*s;
        x2 = (-b + d)*s;
        if( x1 > x2 )
            std::swap(x1, x2);
        return 2;
    }
    x1 = x2 = -b/(2*a);
    return 1;
}

class EllipticKeyPoint
{
public:
    EllipticKeyPoint();
    EllipticKeyPoint( const Point2f& _center, const Scalar& _ellipse );

    static void convert( const std::vector<KeyPoint>& src, std::vector<EllipticKeyPoint>& dst );
    static void convert( const std::vector<EllipticKeyPoint>& src, std::vector<KeyPoint>& dst );

    static Mat_<double> getSecondMomentsMatrix( const Scalar& _ellipse );
    Mat_<double> getSecondMomentsMatrix() const;

    void calcProjection( const Mat_<double>& H, EllipticKeyPoint& projection ) const;
    static void calcProjection( const std::vector<EllipticKeyPoint>& src, const Mat_<double>& H, std::vector<EllipticKeyPoint>& dst );

    Point2f center;
    Scalar ellipse; // 3 elements a, b, c: ax^2+2bxy+cy^2=1
    Size_<float> axes; // half length of ellipse axes
    Size_<float> boundingBox; // half sizes of bounding box which sides are parallel to the coordinate axes
};

EllipticKeyPoint::EllipticKeyPoint()
{
    *this = EllipticKeyPoint(Point2f(0,0), Scalar(1, 0, 1) );
}

EllipticKeyPoint::EllipticKeyPoint( const Point2f& _center, const Scalar& _ellipse )
{
    center = _center;
    ellipse = _ellipse;

    double a = ellipse[0], b = ellipse[1], c = ellipse[2];
    double ac_b2 = a*c - b*b;
    double x1, x2;
    solveQuadratic(1., -(a+c), ac_b2, x1, x2);
    axes.width = (float)(1/sqrt(x1));
    axes.height = (float)(1/sqrt(x2));

    boundingBox.width = (float)sqrt(ellipse[2]/ac_b2);
    boundingBox.height = (float)sqrt(ellipse[0]/ac_b2);
}

void EllipticKeyPoint::convert( const std::vector<KeyPoint>& src, std::vector<EllipticKeyPoint>& dst )
{
    //CV_INSTRUMENT_REGION()

    if( !src.empty() )
    {
        dst.resize(src.size());
        for( size_t i = 0; i < src.size(); i++ )
        {
            float rad = src[i].size/2;
            CV_Assert( rad );
            float fac = 1.f/(rad*rad);
            dst[i] = EllipticKeyPoint( src[i].pt, Scalar(fac, 0, fac) );
        }
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////



/////////////////////////////////////////////////////////////- KLT FEATURE TRACKING -//////////////////////////////////////////////////////////////////////////////////////

void featureTracking_1(Mat img_1, Mat img_2, vector<Point2f>& points1, vector<Point2f>& points2, vector<uchar>& status, const vector<Point3f>& landmarks, vector<Point3f>& landmarks_ref, vector<Point2f>&featurePoints_ref)	{ 

  vector<float> err;					
  Size winSize=Size(21,21);																							
  //TermCriteria termcrit=TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 30, 0.01);

  //calcOpticalFlowPyrLK(img_1, img_2, points1, points2, status, err, winSize, 3, termcrit, 0, 0.001);
  calcOpticalFlowPyrLK(img_1, img_2, points1, points2, status, err);


  int indexCorrection = 0;
  for( int i=0; i<status.size(); i++)
     {  	
	if (status[i] == 1) 
	{ 
	featurePoints_ref.push_back(points2[i]);
        landmarks_ref.push_back(landmarks[i]);
	}

     }

}


void featureTracking_0(Mat img_1, Mat img_2, vector<Point2f>& points1, vector<Point2f>& points2, vector<uchar>& status)	{ 

//this function automatically gets rid of points for which tracking fails

  vector<float> err;					
  Size winSize=Size(21,21);																								
  TermCriteria termcrit=TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 30, 0.01);

  calcOpticalFlowPyrLK(img_1, img_2, points1, points2, status, err, winSize, 3, termcrit, 0, 0.001);

  	//getting rid of points for which the KLT tracking failed or those who have gone outside the frame
  	int indexCorrection = 0;
  	for( int i=0; i<status.size(); i++)
     	{  Point2f pt = points2.at(i- indexCorrection);
     		if ((status.at(i) == 0)||(pt.x<0)||(pt.y<0))	{
     		  if((pt.x<0)||(pt.y<0))	{
     		  	status.at(i) = 0;
     		  }
     		  points1.erase (points1.begin() + (i - indexCorrection));
     		  points2.erase (points2.begin() + (i - indexCorrection));
     		  indexCorrection++;
     		}

     	}

}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////- TRIANGULATION STUFF -/////////////////////////////////////////////////////////////////////////////////////


vector<Point3f> get3D_Points_0(const vector<Point2f>& feature_p1, const vector<Point2f>& feature_p2, Mat H) {

    // This is to generate the 3D points

    vector<Point2f> featurePoints; 
    Mat inv_transform;  
    
    Mat K = (Mat_<double>(3, 3) << 7.188560000000e+02, 0, 6.071928000000e+02,0, 7.188560000000e+02, 1.852157000000e+02,0, 0, 1);

    Mat M_1 = Mat::zeros(3,4, CV_64F);
    Mat M_2 = Mat::zeros(3,4, CV_64F);
    M_1.at<double>(0,0)  =1; M_1.at<double>(1,1)  =1; M_1.at<double>(2,2)  =1;
    M_2.at<double>(0,0)  =1; M_2.at<double>(1,1)  =1; M_2.at<double>(2,2)  =1;

    M_1  = K*M_1;
    M_2  = K*H;

    Mat landmarks;

    triangulatePoints(M_1, M_2, feature_p1, feature_p2, landmarks); 

    //cout << M_1 << endl;
    //cout << M_2 << endl;

    std::vector<Point3f> output;

    for (int i = 0; i < landmarks.cols; i++) {
    	Point3f p;
    	p.x = landmarks.at<float>(0, i)/landmarks.at<float>(3, i);
	p.y = landmarks.at<float>(1, i)/landmarks.at<float>(3, i);
	p.z = landmarks.at<float>(2, i)/landmarks.at<float>(3, i);
	output.push_back(p);
    }

    return output;
}

vector<Point3f> get3D_Points_1(const vector<Point2f>& feature_p1, const vector<Point2f>& feature_p2, Mat H, Mat J) {

    // This is to generate the 3D points

    vector<Point2f> featurePoints; 
    Mat inv_transform;  
    
    Mat K = (Mat_<double>(3, 3) << 7.188560000000e+02, 0, 6.071928000000e+02,0, 7.188560000000e+02, 1.852157000000e+02,0, 0, 1);

    Mat M_1 = Mat::zeros(3,4, CV_64F);
    Mat M_2 = Mat::zeros(3,4, CV_64F);
    M_1.at<double>(0,0)  =1; M_1.at<double>(1,1)  =1; M_1.at<double>(2,2)  =1;
    M_2.at<double>(0,0)  =1; M_2.at<double>(1,1)  =1; M_2.at<double>(2,2)  =1;

    M_1  = K*H;
    M_2  = K*J;

    Mat landmarks;

    triangulatePoints(M_1, M_2, feature_p1, feature_p2, landmarks); 

    //cout << M_1 << endl;
    //cout << M_2 << endl;

    std::vector<Point3f> output;

    for (int i = 0; i < landmarks.cols; i++) {
    	Point3f p;
    	p.x = landmarks.at<float>(0, i)/landmarks.at<float>(3, i);
	p.y = landmarks.at<float>(1, i)/landmarks.at<float>(3, i);
	p.z = landmarks.at<float>(2, i)/landmarks.at<float>(3, i);
	output.push_back(p);
    }

    return output;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


