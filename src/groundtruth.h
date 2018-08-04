#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"


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


vector<vector<float> > get_Pose() {

  vector<vector<float> > poses;
  //Theese need to be modified depending on where the dataset is located
  ifstream myfile("/media/mahdi/Bulk1/Ubuntu/dataset-kitti-odom/poses/00.txt");
  //ifstream myfile(path);
  string line;
  if (myfile.is_open())
  {
    while ( getline (myfile,line) )
    {
          char * dup = strdup(line.c_str());
   		  char * token = strtok(dup, " ");
   		  std::vector<float> v;
	   	  while(token != NULL){
	        	v.push_back(atof(token));
	        	token = strtok(NULL, " ");
	    	}
	    	poses.push_back(v);
	    	free(dup);
    }
    myfile.close();
  } else {
  	cout << "Unable to open file"; 
  }	

  return poses;

}

double getAbsoluteScale(int frame_id, int sequence_id, double z_cal)	{
  

  string line;
  int i = 0;
  ifstream myfile ("/media/mahdi/Bulk1/Ubuntu/dataset-kitti-odom/poses/00.txt");
  double x=0, y=0, z=0;
  double x_prev, y_prev, z_prev;
  if (myfile.is_open())
  {
    while (( getline (myfile,line) ) && (i<=frame_id))
    {
      z_prev = z;
      x_prev = x;
      y_prev = y;
      std::istringstream in(line);
      //cout << line << '\n';
      for (int j=0; j<12; j++)  {
        in >> z ;
        if (j==7) y=z;
        if (j==3)  x=z;
      }
      
      i++;
    }
    myfile.close();
  }

  else {
    cout << "Unable to open file";
    return 0;
  }

  return sqrt((x-x_prev)*(x-x_prev) + (y-y_prev)*(y-y_prev) + (z-z_prev)*(z-z_prev)) ;

}
