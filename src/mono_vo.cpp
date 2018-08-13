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
#include <algorithm>
#include <iterator> 
#include <vector>
#include <ctime>
#include <sstream>
#include <fstream>
#include <string>

#include "mono_vo.h"

using namespace cv;
using namespace std;
using namespace xfeatures2d;

mono_vo::mono_vo(const Mat& _K):K(_K){

}

/////////////////////////////////////////////////////////////- FEATURE DETECTION & MATCHING -//////////////////////////////////////////////////////////////////////////////

void mono_vo::Detect_Features(Mat img_1, vector<Point2f>& points1, vector<KeyPoint>& keypoints_1, Mat descriptors_1, string type)	{

 int fast_threshold = 20;
 bool nonmaxSuppression = true;

 if(type == "SIFT"){
  //C++: SIFT::SIFT(int nfeatures=0, int nOctaveLayers=3, double contrastThreshold=0.04, double edgeThreshold=10, double sigma=1.6)
  cv::Ptr<Feature2D> f2d = SIFT::create();
  f2d->detect( img_1, keypoints_1 );
  f2d->compute( img_1, keypoints_1, descriptors_1 );
  KeyPoint::convert(keypoints_1, points1, vector<int>());
 }
 else if(type == "BRIEF"){
  //static Ptr<BriefDescriptorExtractor> cv::xfeatures2d::BriefDescriptorExtractor::create(int bytes = 32, bool use_orientation = false) 	
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
  //static Ptr<BRISK> cv::BRISK::create( int thresh = 30, int octaves = 3, float patternScale = 1.0f) 	
  static Ptr<FeatureDetector> brisk = BRISK::create(20, 3, 1.0f);
  brisk->detectAndCompute(img_1,cv::noArray(), keypoints_1, descriptors_1);
  KeyPoint::convert(keypoints_1, points1, vector<int>());
 }
 else if(type == "AKAZE"){
  //static Ptr<AKAZE> cv::AKAZE::create( int descriptor_type = AKAZE::DESCRIPTOR_MLDB, int descriptor_size = 0, int descriptor_channels = 3, float threshold = 0.001f, int nOctaves = 4, int nOctaveLayers = 4, int diffusivity = KAZE::DIFF_PM_G2) 	
  static Ptr<FeatureDetector> akaze = AKAZE::create(); //failing at 3rd turn, needs to be tuned
  akaze->detectAndCompute(img_1,cv::noArray(), keypoints_1, descriptors_1);
  KeyPoint::convert(keypoints_1, points1, vector<int>()); 
 }

}


void mono_vo::Detect_Match_Features(Mat img_1, Mat img_2, vector<Point2f>& match_points1, vector<Point2f>& match_points2, string type)  {

 int fast_threshold = 20;
 bool nonmaxSuppression = true;
 Mat descriptors_1, descriptors_2;
 vector<KeyPoint> keypoints_1, keypoints_2;

 if(type == "SIFT"){
  //C++: SIFT::SIFT(int nfeatures=0, int nOctaveLayers=3, double contrastThreshold=0.04, double edgeThreshold=10, double sigma=1.6)
  cv::Ptr<Feature2D> f2d = SIFT::create();
  f2d->detect( img_1, keypoints_1 );
  f2d->compute( img_1, keypoints_1, descriptors_1);
  f2d->detect( img_2, keypoints_2 );
  f2d->compute( img_2, keypoints_2, descriptors_2);
  }
 else if(type == "BRIEF"){
  //static Ptr<BriefDescriptorExtractor> cv::xfeatures2d::BriefDescriptorExtractor::create(int bytes = 32, bool use_orientation = false) 	
  cv::Ptr<Feature2D> extractor = BriefDescriptorExtractor::create();
  FAST(img_1, keypoints_1, fast_threshold, nonmaxSuppression);
  extractor->compute( img_1, keypoints_1, descriptors_1 );
  FAST(img_2, keypoints_2, fast_threshold, nonmaxSuppression);
  extractor->compute( img_2, keypoints_2, descriptors_2 );
  }
 else if(type == "ORB"){
  // C++: ORB::ORB(int nfeatures=500, float scaleFactor=1.2f, int nlevels=8, int edgeThreshold=31, int firstLevel=0, int WTA_K=2, int scoreType=ORB::HARRIS_SCORE, int patchSize=31)
  static Ptr<FeatureDetector> orb = ORB::create(500,1.2f,8,30,0,2,ORB::FAST_SCORE,30); 
  orb->detectAndCompute(img_1,cv::noArray(), keypoints_1, descriptors_1); 
  orb->detectAndCompute(img_2,cv::noArray(), keypoints_2, descriptors_2);
  }
 else if(type == "BRISK"){
  //static Ptr<BRISK> cv::BRISK::create( int thresh = 30, int octaves = 3, float patternScale = 1.0f) 	
  static Ptr<FeatureDetector> brisk = BRISK::create(20, 3, 1.0f);
  brisk->detectAndCompute(img_1,cv::noArray(), keypoints_1, descriptors_1);
  brisk->detectAndCompute(img_2,cv::noArray(), keypoints_2, descriptors_2);
  }
 else if(type == "AKAZE"){
  //static Ptr<AKAZE> cv::AKAZE::create( int descriptor_type = AKAZE::DESCRIPTOR_MLDB, int descriptor_size = 0, int descriptor_channels = 3, float threshold = 0.001f, int nOctaves = 4, int nOctaveLayers = 4, int diffusivity = KAZE::DIFF_PM_G2) 	
  static Ptr<FeatureDetector> akaze = AKAZE::create(); //failing at 3rd turn, needs to be tuned
  akaze->detectAndCompute(img_1,cv::noArray(), keypoints_1, descriptors_1);
  akaze->detectAndCompute(img_2,cv::noArray(), keypoints_2, descriptors_2);
  }

  FlannBasedMatcher matcher;
  vector<vector< DMatch > > matches;

  descriptors_1.convertTo(descriptors_1, CV_32F);
  descriptors_2.convertTo(descriptors_2, CV_32F);

  matcher.knnMatch(descriptors_1, descriptors_2, matches, 2);
	
  for (int i = 0; i < matches.size(); i++) {

  const DMatch& bestMatch = matches[i][0];  
  const DMatch& betterMatch = matches[i][1];  

  if (bestMatch.distance < 0.7*betterMatch.distance) {
	match_points1.push_back(keypoints_1[bestMatch.queryIdx].pt);
        match_points2.push_back(keypoints_2[bestMatch.trainIdx].pt);
	}
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


/////////////////////////////////////////////////////////////- KLT FEATURE TRACKING -//////////////////////////////////////////////////////////////////////////////////////

void mono_vo::featureTracking_0(Mat img_1, Mat img_2, vector<Point2f>& points1, vector<Point2f>& points2, vector<uchar>& status)	{ 

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
     		  //	status.at(i) = 0;
     		  }
     		  //points1.erase (points1.begin() + (i - indexCorrection));
     		  //points2.erase (points2.begin() + (i - indexCorrection));
     		  indexCorrection++;
     		}

     	}

}

void mono_vo::featureTracking_1(Mat img_1, Mat img_2, vector<Point2f>& points1, vector<Point2f>& points2, vector<uchar>& status, const vector<Point3f>& landmarks, vector<Point3f>& landmarks_ref, vector<Point2f>&featurePoints_ref)	{ 

  vector<float> err;					
  Size winSize=Size(21,21);																							
  TermCriteria termcrit=TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 30, 0.01);
  
  calcOpticalFlowPyrLK(img_1, img_2, points1, points2, status, err, winSize, 3, termcrit, 0, 0.001);

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


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////- TRIANGULATION -/////////////////////////////////////////////////////////////////////////////////////


vector<Point3f> mono_vo::get3D_Points_0(const vector<Point2f>& feature_p1, const vector<Point2f>& feature_p2, Mat H) {

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

    vector<Point3f> output;

    for (int i = 0; i < landmarks.cols; i++) {
    	Point3f p;
    	p.x = landmarks.at<float>(0, i)/landmarks.at<float>(3, i);
	p.y = landmarks.at<float>(1, i)/landmarks.at<float>(3, i);
	p.z = landmarks.at<float>(2, i)/landmarks.at<float>(3, i);
	output.push_back(p);
    }

    return output;
}

vector<Point3f> mono_vo::get3D_Points_1(const vector<Point2f>& feature_p1, const vector<Point2f>& feature_p2, Mat H, vector<Point2f>& prevFeatures, Mat inv_transform) {

  vector<Point3f> landmarks_new = get3D_Points_0(feature_p1, feature_p2, H);
  vector<Point3f> landmarks;

   	for (int k = 0; k < landmarks_new.size(); k++) {

	   	const Point3f& pt = landmarks_new[k];

	   	Point3f p;
		
	   	p.x = inv_transform.at<double>(0, 0)*pt.x + inv_transform.at<double>(0, 1)*pt.y + inv_transform.at<double>(0, 2)*pt.z + inv_transform.at<double>(0, 3);
	   	p.y = inv_transform.at<double>(1, 0)*pt.x + inv_transform.at<double>(1, 1)*pt.y + inv_transform.at<double>(1, 2)*pt.z + inv_transform.at<double>(1, 3);
	   	p.z = inv_transform.at<double>(2, 0)*pt.x + inv_transform.at<double>(2, 1)*pt.y + inv_transform.at<double>(2, 2)*pt.z + inv_transform.at<double>(2, 3);

	   	if (p.z > 0) {

			landmarks.push_back(p);
			prevFeatures.push_back(feature_p2[k]);
	   	}

	}

    return landmarks;

}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


//////////////////////////////////////////////////////////////////- CONTINOUS PART HERE -/////////////////////////////////////////////////////////////////////////////////

void mono_vo::Continious(const string& filename1, const string& filename2, const vector<vector<float>>& poses){
  
  vector<Point2f> match_points1;
  vector<Point2f> match_points2;
  vector<KeyPoint> keypoints_1;
  vector<KeyPoint> keypoints_2;
  vector<Point3f> landmarks;
  vector<Point3f> landmarks_new;
  Mat descriptors_1;
  Mat R, t, E, mask;
  vector<uchar> status;
  string type = "SIFT";

  double focal = 7.188560000000e+02;
  cv::Point2d pp(6.071928000000e+02, 1.852157000000e+02);

  //read the first two frames from the dataset
  Mat img_1_c = imread(filename1);
  Mat img_2_c = imread(filename2);
  Mat img_1;
  Mat img_2;

  if ( !img_1_c.data || !img_2_c.data ) { 
    std::cout<< " --(!) Error reading images " << std::endl;
  }

  // we work with grayscale images
  cvtColor(img_1_c, img_1, COLOR_BGR2GRAY);
  cvtColor(img_2_c, img_2, COLOR_BGR2GRAY);

  assert(landmarks.size() == 0);
  assert(match_points1.size() == 0);

  //FAST_featureDetection(img_1, keypoints_1, match_points1);
  Detect_Features(img_1, match_points1, keypoints_1, descriptors_1, type);
  
  featureTracking_0(img_1,img_2,match_points1,match_points2, status);
 	
  E = findEssentialMat(match_points2, match_points1, focal, pp, RANSAC, 0.999, 1.0, mask);

  recoverPose(E, match_points2, match_points1, R, t, focal, pp, mask);
  
  Mat H = Mat::zeros(3,4,CV_64F);
  R.col(0).copyTo(H.col(0));
  R.col(1).copyTo(H.col(1));
  R.col(2).copyTo(H.col(2));
  t.copyTo(H.col(3));

  Mat distCoeffs = Mat::zeros(4,1,CV_64F);
  Mat rvec;
  vector<Point2f> imagePoints;
  Rodrigues(R, rvec);
  
  landmarks = get3D_Points_0(match_points1, match_points2, H);

  //cout << landmarks << endl;

  cout << "--------------------------- INITIALIZATION ---------------------------------------" << endl;
  cout << " " << endl;
  cout << "match_points1 size = " << match_points1.size() << endl;
  cout << "match_points2 size = " << match_points2.size() << endl;
  cout << "landmarks size = " << landmarks.size() << endl;
  cout << "translation = " << t << endl;
  cout << "rotation = " << R << endl;
  cout << "----------------------------------------------------------------------------------" << endl;
  cout << " " << endl;
  
  Mat prevImage = img_2;
  Mat currImage, currImage_1;
  vector<Point2f> prevFeatures = match_points2;
  vector<Point2f> currFeatures, currFeatures_1;

  char filename[100];
  Mat traj = Mat::zeros(600, 600, CV_8UC3);
  namedWindow( "Trajectory", WINDOW_AUTOSIZE );

  cout << "--------------------------- CONTININOUS ---------------------------------------" << endl;

  for(int numFrame=4; numFrame < 15; numFrame++)	{ 

	cout << "numFrame = " << numFrame << endl;

	sprintf(filename, "/media/mahdi/Bulk1/Ubuntu/dataset-kitti-odom/sequences/00/image_0/%06d.png", numFrame);
  	Mat currImage_c = imread(filename);
  	cvtColor(currImage_c, currImage, COLOR_BGR2GRAY);

  	vector<Point3f> landmarks_ref;
  	vector<Point2f> featurePoints_ref;

	//featureTracking_1(prevImage, currImage, prevFeatures, currFeatures, status, landmarks, landmarks_ref, featurePoints_ref);
	featureTracking_0(prevImage, currImage, prevFeatures, currFeatures_1, status);	

	cout << "prevFeatures size = " <<  prevFeatures.size() << endl;
	cout << "currFeatures size = " <<  currFeatures_1.size() << endl;
	cout << "landmarks size = " <<  landmarks.size() << endl;
	//cout << "landmarks_ref size = " <<  landmarks_ref.size() << endl;
	//cout << "featurePoints_ref size = " <<  featurePoints_ref.size() << endl;

	vector<int> inliers;
	inliers.clear();
	
 	Mat dist_coeffs = Mat::zeros(4,1,CV_64F);
	Mat rv, tv;
	
  	//solvePnPRansac(landmarks_ref, featurePoints_ref, K, dist_coeffs, rv, tv, false, 100, 4.0, 0.99, inliers);
	//solvePnP(landmarks_ref, featurePoints_ref, K, dist_coeffs, rv, tv, false, CV_P3P);
	solvePnPRansac(landmarks, currFeatures_1, K, dist_coeffs, rv, tv, false, 100, 4.0, 0.99, inliers);

	currFeatures.clear();
	landmarks_new.clear();

  	for( int i=0; i<inliers.size(); i++)
     	{  	
		int k = inliers[i];
		//currFeatures.push_back(featurePoints_ref[k]);
        	//landmarks_new.push_back(landmarks_ref[k]);
		currFeatures.push_back(currFeatures_1[k]);
        	landmarks_new.push_back(landmarks[k]);
    	}

	//if (inliers.size() < 5) continue;
	
	float inliers_ratio = inliers.size()/float(landmarks.size());

	cout << "currFeatures size = " << currFeatures.size() << endl;
	cout << "landmarks_new size = " << landmarks_new.size() << endl;	
	cout << "inliers size = " << inliers.size() << endl;
	cout << "inliers ratio = " << inliers_ratio << endl;	

	cout << inliers.size() << endl;
	
	Mat R_mat, t_vec;

	Rodrigues(rv, R_mat);

	Mat J = Mat::zeros(3,4,CV_64F);
	R_mat.col(0).copyTo(J.col(0));
	R_mat.col(1).copyTo(J.col(1));
	R_mat.col(2).copyTo(J.col(2));
	tv.copyTo(J.col(3));

  	R_mat = R_mat.t();
	t_vec = -R_mat*tv;

	cout << "translation = " << t_vec.t() << endl;
	cout << "GT 	    = " << "["<<poses[numFrame][3] << ", " << poses[numFrame][7] << ", " << poses[numFrame][11] <<"]"<<endl;
	cout << " " << endl;

	cout << "||||||||||||||||||| NEXT |||||||||||||||||" << endl;
	cout << " " << endl;
	
	Mat inv_transform = Mat::zeros(3,4,CV_64F);
	R_mat.col(0).copyTo(inv_transform.col(0));
	R_mat.col(1).copyTo(inv_transform.col(1));
	R_mat.col(2).copyTo(inv_transform.col(2));
	t_vec.copyTo(inv_transform.col(3));
	J = inv_transform;

	vector<Point2f> mpoints1;
	vector<Point2f> mpoints2;
  	
	cout << "TRIGGERED RE-DETECTION" << endl;

	prevFeatures.clear();
	landmarks.clear();
	landmarks = landmarks_new;

	if(numFrame > 4) {
	cout << "STEPED IN" << endl;
   	for (int k = 0; k < landmarks_new.size(); k++) {
	 	const Point3f& pt = landmarks_new[k];
	  	Point3f p;
	  	p.x = J.at<double>(0, 0)*pt.x + J.at<double>(0, 1)*pt.y + J.at<double>(0, 2)*pt.z + J.at<double>(0, 3);
	 	p.y = J.at<double>(1, 0)*pt.x + J.at<double>(1, 1)*pt.y + J.at<double>(1, 2)*pt.z + J.at<double>(1, 3);
	  	p.z = J.at<double>(2, 0)*pt.x + J.at<double>(2, 1)*pt.y + J.at<double>(2, 2)*pt.z + J.at<double>(2, 3);
	   	if (p.z > 0) {
			//landmarks.push_back(p);
			//prevFeatures.push_back(currFeatures[k]);
	  		}
		}
	}

	//TODO : detect+match+triangulation in one big function

	//Detect_Match_Features(prevImage, currImage, mpoints1, mpoints2, type);

	//cout << "mpoints1 size = " << mpoints1.size() << endl;
	//cout << "mpoints2 size = " << mpoints2.size() << endl;

	//cout << "NEW POINTS TRIANGULATION" << endl;
	//landmarks = get3D_Points_1(mpoints1, mpoints2, J, prevFeatures, inv_transform);
	//landmarks = get3D_Points_0(mpoints1, mpoints2, J);
	//cout << "landmarks AFTER CLEANING = " << landmarks.size() << endl;

	prevImage = currImage;
	prevFeatures = currFeatures;
	currFeatures.clear();

		// plot the information
		string text  = "Red color: estimated trajectory";
		string text2  = "Green color: GT";

		t_vec.convertTo(t_vec, CV_32F);
		Point2f center = Point2f(int(t_vec.at<float>(0)) + 300, int(t_vec.at<float>(2)) + 100);
		Point2f t_center = Point2f(int(poses[numFrame][3]) + 300, int(poses[numFrame][11]) + 100);
		circle(traj, center ,1, Scalar(0,0,255), 2);
		circle(traj, t_center,1, Scalar(255,0,0), 2);
		rectangle(traj, Point2f(10, 30), Point2f(550, 50),  Scalar(0,0,0), cv::FILLED);
		putText(traj, text, Point2f(10,50), cv::FONT_HERSHEY_PLAIN, 1, Scalar(0, 0,255), 1, 5);
		putText(traj, text2, Point2f(10,70), cv::FONT_HERSHEY_PLAIN, 1, Scalar(255,0,0), 1, 5);
		imshow( "Trajectory", traj);
		waitKey(1);

	}
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
