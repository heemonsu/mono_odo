#include "mono_vo.h"
#include "groundtruth.h"

using namespace std;

vector<vector<float> > get_Pose(string path) {

  vector<vector<float> > poses;
  //Theese need to be modified depending on where the dataset is located
  ifstream myfile(path);
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


int main(int argc, char** argv) {

  int max_frame= 1000;


  //Theese need to be modified depending on where the dataset is located
  string filename1 = "/media/mahdi/Bulk1/Ubuntu/dataset-kitti-odom/sequences/00/image_0/000000.png";
  string filename2 = "/media/mahdi/Bulk1/Ubuntu/dataset-kitti-odom/sequences/00/image_1/000000.png";
  string pose_path = "/media/mahdi/Bulk1/Ubuntu/dataset-kitti-odom/poses/00.txt";


  cout <<"Program starts!"<<endl;

  cv::Mat K = (cv::Mat_<double>(3, 3) << 7.188560000000e+02, 0, 6.071928000000e+02,
                         0, 7.188560000000e+02, 1.852157000000e+02,
                         0, 0, 1);

  mono_vo Mvo(K);

  //we have only 1000 paris of pictures
  Mvo.set_Max_frame(max_frame);

  //This is to get the ground truth pose data
  vector<vector<float>> poses = get_Pose(pose_path);

  Mvo.Continious(filename1, filename2, poses);
  
  cout << "You come to the end!" << endl;

	return 0;
}
