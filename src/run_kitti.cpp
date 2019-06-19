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
    int window_sz_BM 	= 3;
    float ssd_th 		= 12.0f;
    float ssd_ratio_th 	= 0.7f;
    size_t MIN_NUM_FEAT 	= 10;
	
    cout << "no images = " << nImages << endl;
	
	ORB_VISLAM::System mySLAM(argv[2], frame_scale, 
								window_sz_BM, ssd_th, ssd_ratio_th, MIN_NUM_FEAT);
	
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
	
	string vo_file 		= string(argv[1])	+ "/VO.txt";
	string homog_file 	= string(argv[1])	+ "/Homography.txt";
	
	ofstream file_vo, file_homography;	
	
	file_vo.open(vo_file.c_str());
	file_homography.open(homog_file.c_str());
	
	file_vo << fixed;
	file_vo << "matches12,matches21,matchesCCM,sol0_rvec_x,sol0_rvec_y,sol0_rvec_z,sol0_R00,sol0_R01,sol0_R02,sol0_tx,sol0_R10,sol0_R11,sol0_R12,sol0_ty,sol0_R20,sol0_R21,sol0_R22,sol0_tz,sol1_rvec_x,sol1_rvec_y,sol1_rvec_z,sol1_R00,sol1_R01,sol1_R02,sol1_tx,sol1_R10,sol1_R11,sol1_R12,sol1_ty,sol1_R20,sol1_R21,sol1_R22,sol1_tz,sol2_rvec_x,sol2_rvec_y,sol2_rvec_z,sol2_R00,sol2_R01,sol2_R02,sol2_tx,sol2_R10,sol2_R11,sol2_R12,sol2_ty,sol2_R20,sol2_R21,sol2_R22,sol2_tz,sol3_rvec_x,sol3_rvec_y,sol3_rvec_z,sol3_R00,sol3_R01,sol3_R02,sol3_tx,sol3_R10,sol3_R11,sol3_R12,sol3_ty,sol3_R20,sol3_R21,sol3_R22,sol3_tz" << endl;

	file_homography << fixed;
	file_homography << "H_00,H_01,H_02,H_10,H_11,H_12,H_20,H_21,H_22" << endl;


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
		mySLAM.run(img, frame_name, file_vo, file_homography);
	}
	clock_t tEnd = clock();
    
	double runTime;
	runTime = (double)(tEnd - tStart)/CLOCKS_PER_SEC;
    
	cout << "\nAlgorithm time: "<< runTime << " sec.\n" << endl;
	//mySLAM.shutdown();
	return 0;
}
