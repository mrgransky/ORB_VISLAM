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
#define PI 3.1415926f

void import_seq(const string &path, 
				string &imgFiles, 
				vector<double> &vTimestamps)
{
	imgFiles = path + "/image_0/%06d.png";
	string tsFile = path +"/times.txt";
	ifstream f;
	f.open(tsFile.c_str());

	while(!f.eof()) // end of the file (eof)
	{
		string s;
		getline(f,s);
		if(!s.empty())
		{
			stringstream ss;
			ss << s;
			
			double t;
			
			ss >> t;
			vTimestamps.push_back(t); // retrieve timestamp from rgb.txt
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
			<< " /home/xenial/Datasets/KITTI/sequences/00 /home/xenial/WS_Farid/orb_slam2_TUT/Examples/Monocular/KITTI00-02.yaml\n\n"
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
	string seqPath = argv[1];
	
	vector<double> vTimestamps;		// retrieve ts
	string imgFolder;
    import_seq(seqPath, imgFolder, vTimestamps);
    int nImages = vTimestamps.size();
    
    float frame_scale = 0.74f;
    int window_sz_BM = 11;
    float ssd_th = 50.0f;
    float ssd_ratio_th = .8f;
    
    cout << "no images = " << nImages << endl;
	
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
	string traj_cam = string(argv[1])+"frames/VO_Trajectory_KITTI.txt";
	
	ofstream f_cam;
	f_cam.open(traj_cam.c_str());
	
	f_cam << fixed;
	f_cam << "x,y,z"<< endl;

	char filename[100];
	Mat imgT;
	clock_t tStart = clock();
	
	for(size_t ni = 0; ni < keyIMG.size(); ni++)
	//for(int ni = 0; ni < nImages; ni++) 
	{
		sprintf(filename, imgFolder.c_str(), ni);
		
		cout 	<<"\n\nReading Frame:\t"	<< filename << endl;
		string frame_name = filename;
		
		Mat img = imread(filename, CV_LOAD_IMAGE_GRAYSCALE);
		
		if(img.empty())
		{
			cerr	<<"\nFailed loading frame: \t"
					<< filename
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
