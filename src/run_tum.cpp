#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include <pangolin/pangolin.h>

#include <opencv2/core/eigen.hpp>
#include <stdio.h>      /* printf, fopen */
#include <stdlib.h>     /* exit, EXIT_FAILURE */
#include <thread>

#include <time.h>
#include "System.h"

using namespace std;
using namespace cv;

void LoadImages(const string &path, 
				vector<string> &imgName, 
				vector<double> &vTimestamps)
{
	ifstream f;
	f.open(path.c_str());
	string s;
	getline(f,s);
	getline(f,s);
	getline(f,s);
	
	while(!f.eof()) // end of the file (eof)
	{
		getline(f,s);
		if(!s.empty())
		{
			stringstream ss;
			ss << s;
			
			double t;
			string sRGB;
			
			ss >> t;
			vTimestamps.push_back(t); // retrieve timestamp from rgb.txt
			
			ss >> sRGB;
			imgName.push_back(sRGB); // retrieve img name from rgb.txt
        }
    }
}

void printHelp(char ** argv)
{
	cout	<< "\n\nNOT ENOUGH ARGUMENT PROVIDED!!\n\nSyntax:"		
			<< argv[0]	
			<< " [/path/2/image_folder] [/path/2/setting file]"
			<< "\n\nEXAMPLE:\n\n" 	
			<< argv[0]
			<< " /home/xenial/Datasets/TUM/rgbd_dataset_freiburg1_xyz /home/xenial/WS_Farid/orb_slam2_TUT/Examples/Monocular/TUM1.yaml\n\n"
			<< endl;
}

int main( int argc, char** argv )
{
	if(argc != 3)
	{
		printHelp(argv);
		return -1; 
	}
    
	//Mat img;
	string imgFile = string(argv[1])+"/rgb.txt"; // open rgb.txt from the img folder
	
	vector<double> vTimestamps;		// retrieve ts 
	vector<string> imgName; 		// retrieve img file names ex: rgb/frame_145.jpg
    LoadImages(imgFile, imgName, vTimestamps);
    int nImages = imgName.size();

    float frame_scale = 0.9f;
    int window_sz_BM = 11;
    float ssd_th = 10.0f;
    float ssd_ratio_th = .8f;
    
	ORB_VISLAM::System mySLAM(argv[2], frame_scale, window_sz_BM, ssd_th, ssd_ratio_th);
	vector<size_t> keyIMG;
	for(int ni = 0; ni < nImages; ni++) 
	{
		if(ni%1 == 0) 
		{
			keyIMG.push_back(ni);
		}
	}
	cout 	<< "\nMatching process of " 		<< keyIMG.size() 
			<< " frames out of " << nImages 	<< " frames ..." 
			<< endl;
	// TODO: the following path does not exist =>> modify it ...
	string traj_cam = string(argv[1])+"frames/VO_Trajectory_EuRoC.txt";
	
	ofstream f_cam;
	f_cam.open(traj_cam.c_str());
	
	f_cam << fixed;
	f_cam << "x,y,z"<< endl;

	clock_t tStart = clock();
	for(size_t ni = 0; ni < keyIMG.size(); ni++)
	//for(int ni = 0; ni < nImages; ni++) 
	{
		cout 	<<"\n\nReading Frame ["	<< imgName[keyIMG[ni]] << "]" << endl;
		string frame_name = imgName[keyIMG[ni]];
		Mat img = imread(string(argv[1]) +"/"+ imgName[keyIMG[ni]], 
								CV_LOAD_IMAGE_GRAYSCALE);
		if(img.empty())
		{
			cerr	<<"\nFailed loading frame["
					<< string(argv[1]) + "/" +imgName[keyIMG[ni]] <<"]"
					<< endl;
			return 1;
		}
		
		if(img.channels() < 3) //this should be always true
		{
			cvtColor(img, img, CV_GRAY2BGR);
		}
		mySLAM.run(img, frame_name, f_cam);
	}
	clock_t tEnd = clock();
    
    double runTime;
    runTime = (double)(tEnd - tStart)/CLOCKS_PER_SEC;
    
    cout << "\nAlgorithm time: "<< runTime << " sec.\n" << endl;
    //mySLAM.shutdown();
	return 0;
}
