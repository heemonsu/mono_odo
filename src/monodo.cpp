#include "vo_features.h"
#include "groundtruth.h"

using namespace cv;
using namespace std;



int main( int argc, char** argv )	{

  Mat img_1, img_2;

  int fontFace = FONT_HERSHEY_PLAIN;
  double fontScale = 1;
  int thickness = 0.5;  
  cv::Point textOrg(10, 50);

  double scale = 1.00;
  double x0=0, y0=0, z0=0;
  double px=0, py=0, pz=0;  
  char filename1[200];
  char filename2[200];
  string type = "SIFT";

  vector<Point3f> prelandmarks, landmarks;

  vector<Point2f> points1, points2;
  vector<uchar> status;

  Mat E, R, t, mask;
  Mat img_keypoints_1; 
  Mat img_keypoints_2;
  Mat traj = Mat::zeros(600, 600, CV_8UC3);

  // load the gt
  vector<vector<float> > poses = get_Pose();

  Mat K = (Mat_<double>(3, 3) << 7.188560000000e+02, 0, 6.071928000000e+02,0, 7.188560000000e+02, 1.852157000000e+02,0, 0, 1);
  Mat old_R = (Mat_<double>(3,3) << 1,0,0,0,1,0,0,0,1);

  double focal = 7.188560000000e+02;
  cv::Point2d pp(6.071928000000e+02, 1.852157000000e+02);


  //Theese need to be modified depending on where the dataset is located
  sprintf(filename1, "/media/mahdi/Bulk1/Ubuntu/dataset-kitti-odom/sequences/00/image_0/%06d.png", 1);
  sprintf(filename2, "/media/mahdi/Bulk1/Ubuntu/dataset-kitti-odom/sequences/00/image_0/%06d.png", 3);

  //read the first two frames from the dataset
  Mat img_1_c = imread(filename1);
  Mat img_2_c = imread(filename2);

  //namedWindow( "1", WINDOW_AUTOSIZE );
  //namedWindow( "Road facing camera", WINDOW_AUTOSIZE );
  namedWindow( "Trajectory", WINDOW_AUTOSIZE );

  if ( !img_1_c.data || !img_2_c.data ) { 
    std::cout<< " --(!) Error reading images " << std::endl; return -1;
  }

  // we work with grayscale images
  cvtColor(img_1_c, img_1, COLOR_BGR2GRAY);
  cvtColor(img_2_c, img_2, COLOR_BGR2GRAY);

/////////////////////////////////////- INITIALIZATION PART HERE -///////////////////////////////////////////////////////


	Mat descriptors_1i;
  	Mat descriptors_2i;
  	vector<Point2f> match_points1;
  	vector<Point2f> match_points2i;
  	vector<Point2f> match_points2;
  	vector<KeyPoint> keypoints_1;
  	vector<KeyPoint> keypoints_2;


  assert(landmarks.size() == 0);
  assert(match_points1.size() == 0);

  //FAST_featureDetection(img_1, keypoints_1, match_points1);
  //featureTracking_0(img_1,img_2,match_points1,match_points2, status);

	//cout << "I AM HERE" << endl;	

	//Detect_Features(prevImage, prevFeatures, prevKeypoints, descriptors_1, type);
	//Detect_Features(currImage, currFeatures, currKeypoints, descriptors_2, type);

  	cv::Ptr<Feature2D> f2d = SIFT::create();
  	f2d->detect( img_1, keypoints_1 );
  	f2d->compute( img_1, keypoints_1, descriptors_1i);
  	//KeyPoint::convert(keypoints_1, match_points1, vector<int>());
  	f2d->detect( img_2, keypoints_2 );
  	f2d->compute( img_2, keypoints_2, descriptors_2i );
  	//KeyPoint::convert(keypoints_2, match_points2, vector<int>());
	
  	//featureTracking_0(prevImage,currImage,prevFeatures,currFeatures, status);
	//BruteForceMatcher<L2<float> > matcher;
  	FlannBasedMatcher matcher_1;
  	vector<vector< DMatch > > matches_1;

  	descriptors_1i.convertTo(descriptors_1i, CV_32F);
  	descriptors_2i.convertTo(descriptors_2i, CV_32F);

  	matcher_1.knnMatch(descriptors_1i, descriptors_2i, matches_1, 2);
	
  	for (int i = 0; i < matches_1.size(); i++) {

  	const DMatch& bestMatch = matches_1[i][0];  
  	const DMatch& betterMatch = matches_1[i][1];  

  	if (bestMatch.distance < 0.7*betterMatch.distance) {
		match_points1.push_back(keypoints_1[bestMatch.queryIdx].pt);
        	match_points2.push_back(keypoints_2[bestMatch.trainIdx].pt);
        	//bestMatches.push_back(bestMatch);
		}

    	}
	
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


  cout << landmarks << endl;

  cout << "--------------------------- INITIALIZATION ---------------------------------------" << endl;
  cout << " " << endl;
  cout << "match_points1 size = " << match_points1.size() << endl;
  cout << "match_points2 size = " << match_points2.size() << endl;
  cout << "landmarks size = " << landmarks.size() << endl;
  cout << "translation = " << t << endl;
  cout << "rotation = " << R << endl;
  cout << "----------------------------------------------------------------------------------" << endl;
  cout << " " << endl;

  //////////////////////////////////////- FOR DEBUG PURPOSES -////////////////////////////////////////////

  projectPoints(landmarks, rvec, t, K, distCoeffs, imagePoints);

  Mat img_keypoints_3;
  vector<KeyPoint> keypoints_3;
  for( size_t i = 0; i < imagePoints.size(); i++ ) {
  keypoints_3.push_back(KeyPoint(imagePoints[i], 1.f));
  }

  drawKeypoints( img_2, keypoints_3, img_keypoints_3, Scalar::all(-1), DrawMatchesFlags::DEFAULT );

  //imshow( "1", img_keypoints_3 );

  //waitKey();

  /////////////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


  Mat prevImage = img_2;
  Mat currImage;
  Mat H1 = H;
  Mat R_f = R.clone();

  vector<Point2f> prevFeatures = match_points2;
  vector<Point2f> currFeatures;



  char filename[100];

  clock_t begin = clock();

  cout << "--------------------------- CONTININOUS ---------------------------------------" << endl;

  for(int numFrame=4; numFrame < 2000; numFrame++)	{ 

/////////////////////////////////////- CONTINOUS PART HERE -///////////////////////////////////////////////////////

	cout << "numFrame = " << numFrame << endl;

	sprintf(filename, "/media/mahdi/Bulk1/Ubuntu/dataset-kitti-odom/sequences/00/image_0/%06d.png", numFrame);
  	Mat currImage_c = imread(filename);
  	cvtColor(currImage_c, currImage, COLOR_BGR2GRAY);
  	vector<Point3f> landmarks_ref;
  	vector<Point2f> featurePoints_ref;
  	
	featureTracking_1(prevImage, currImage, prevFeatures, currFeatures, status, landmarks, landmarks_ref, featurePoints_ref);

	cout << "prevFeatures size = " <<  prevFeatures.size() << endl;
	cout << "currFeatures size = " <<  currFeatures.size() << endl;
	cout << "landmarks_ref size = " <<  landmarks_ref.size() << endl;
	cout << "featurePoints_ref size = " <<  featurePoints_ref.size() << endl;
	//cout << landmarks_ref.size() << endl;
	

 	Mat dist_coeffs = Mat::zeros(4,1,CV_64F);
	Mat rv, tv;
	vector<int> inliers;
  	solvePnPRansac(landmarks_ref, featurePoints_ref, K, dist_coeffs, rv, tv, false, 100, 4.0, 0.99, inliers);
	//solvePnP(landmarks_ref, featurePoints_ref, K, dist_coeffs, rv, tv, false, CV_P3P );
	//solvePnPRansac(landmarks, currFeatures, K, dist_coeffs, rv, tv, false, 100, 4.0, 0.99, inliers);

	//if (inliers.size() < 5) continue;

	
	float inliers_ratio = inliers.size()/float(landmarks_ref.size());


	cout << "inliers size = " << inliers.size() << endl;
	cout << "inliers ratio: " << inliers_ratio << endl;
	
	Mat R_mat, t_vec;

	Rodrigues(rv, R_mat);

	Mat J = Mat::zeros(3,4,CV_64F);
	R_mat.col(0).copyTo(J.col(0));
	R_mat.col(1).copyTo(J.col(1));
	R_mat.col(2).copyTo(J.col(2));
	tv.copyTo(J.col(3));


  	R_mat = R_mat.t();
	//R_f = R_f*R_mat;
	t_vec = -R_mat*tv;
  	//t_vec = -R_f*tv;


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

	Mat H2 = inv_transform;

	vector<KeyPoint> prevKeypoints, currKeypoints;
	vector<Point3f> landmarks_new;
	vector<Point3f> landmarks_ref_new;
	vector<Point2f> featurePoints_ref_new;
  	
	//if(prevFeatures.size() < 500){
	cout << "TRIGGERED RE-DETECTION" << endl;

	prevFeatures.clear();
	currFeatures.clear();

	Mat descriptors_1;
  	Mat descriptors_2;

	//Detect_Features(prevImage, prevFeatures, prevKeypoints, descriptors_1, type);
	//Detect_Features(currImage, currFeatures, currKeypoints, descriptors_2, type);

  	cv::Ptr<Feature2D> f2d = SIFT::create();
  	f2d->detect( prevImage, prevKeypoints );
  	f2d->compute( prevImage, prevKeypoints, descriptors_1 );
  	KeyPoint::convert(prevKeypoints, prevFeatures, vector<int>());
  	f2d->detect( currImage, currKeypoints );
  	f2d->compute( currImage, currKeypoints, descriptors_2 );
  	KeyPoint::convert(currKeypoints, currFeatures, vector<int>());

	cout << "prevFeatures size = " << prevFeatures.size() << endl;
	cout << "currFeatures size = " << currFeatures.size() << endl;

  	//featureTracking_0(prevImage,currImage,prevFeatures,currFeatures, status);
	//BruteForceMatcher<L2<float> > matcher;
  	FlannBasedMatcher matcher;
  	vector<vector< DMatch > > matches;

  	descriptors_1.convertTo(descriptors_1, CV_32F);
  	descriptors_2.convertTo(descriptors_2, CV_32F);

  	matcher.knnMatch(descriptors_1, descriptors_2, matches, 2);

	cout << "matches size = " << matches.size() << endl;

	vector<Point2f> mpoints1;
	vector<Point2f> mpoints2;

  	for (int i = 0; i < matches.size(); i++) {

  	const DMatch& bestMatch = matches[i][0];  
  	const DMatch& betterMatch = matches[i][1];  

  	if (bestMatch.distance < 0.7*betterMatch.distance) {
		mpoints1.push_back(prevKeypoints[bestMatch.queryIdx].pt);
        	mpoints2.push_back(currKeypoints[bestMatch.trainIdx].pt);
        	//bestMatches.push_back(bestMatch);
		}

    	}
	//}


	cout << "NEW POINTS TRIANGULATION" << endl;
  	//landmarks_new = get3D_Points_0(prevFeatures, currFeatures, H2);
  	landmarks_new = get3D_Points_0(mpoints1, mpoints2, H2); //what matrices should i use here, H2 or J ??
	cout << "NEW LANDMARKS SIZE = " << landmarks_new.size() << endl;


	prevFeatures.clear();
	landmarks.clear();

   	for (int k = 0; k < landmarks_new.size(); k++) {
//
	   	const Point3f& pt = landmarks_new[k];

	   	Point3f p;
		
//
	   	p.x = inv_transform.at<double>(0, 0)*pt.x + inv_transform.at<double>(0, 1)*pt.y + inv_transform.at<double>(0, 2)*pt.z + inv_transform.at<double>(0, 3);
	   	p.y = inv_transform.at<double>(1, 0)*pt.x + inv_transform.at<double>(1, 1)*pt.y + inv_transform.at<double>(1, 2)*pt.z + inv_transform.at<double>(1, 3);
	   	p.z = inv_transform.at<double>(2, 0)*pt.x + inv_transform.at<double>(2, 1)*pt.y + inv_transform.at<double>(2, 2)*pt.z + inv_transform.at<double>(2, 3);

//	   	cout << p << endl;
	   	if (p.z > 0) {

			landmarks.push_back(p);
			prevFeatures.push_back(mpoints2[k]);
	   	}
//
	}
//
//	cout << numFrame << endl;	
	cout << "landmarks AFTER CLEANING = " << landmarks.size() << endl;
//	cout << prevFeatures.size() << endl;
	prevImage = currImage;
//  	prevFeatures = currFeatures;
	H1 = H2;
// 	landmarks = landmarks_new; 
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


 /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// 

  clock_t end = clock();
  double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
  cout << "Total time taken: " << elapsed_secs << "s" << endl;
  cout << "------------------------------------------------------------------------------------------" << endl;
  waitKey(0);
  return 0;
}
