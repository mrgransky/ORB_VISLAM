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

	while(!f.eof()) // end of the file (eof)
	{
		string s;
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
			<< " /home/xenial/Datasets/E-Wing640x480/ /home/xenial/Datasets/E-Wing640x480/huawei.yaml\n\n"
			<< endl;
}

int main( int argc, char** argv )
{
	if(argc != 3)
	{
		printHelp(argv);
		return -1; 
	}

	Mat I_3x3 = Mat::eye(3, 3, CV_64F);
	Mat I_4x4 = Mat::eye(4, 4, CV_32F);
	Mat Z_3x1 = Mat::zeros(3, 1, CV_64F);
	
	string imgFile = string(argv[1])+"frames/rgb.txt"; // open rgb.txt from the img folder

	// import image files:
	vector<double> vTimestamps;		// retrieve ts 
	vector<string> imgName; 		// retrieve img file names ex: rgb/frame_145.jpg
    LoadImages(imgFile, imgName, vTimestamps);
    int nImages = imgName.size();

	// improt ground truth:
	vector<Mat> T_GT;
	vector<float> scale_GT;
	for(int i = 0; i < nImages; i++)
	{
		T_GT.push_back(I_4x4.clone());
	}

	scale_GT.push_back(1);
	for(int i = 0; i < nImages - 1; i++)
	{
		scale_GT.push_back(1);
	}
    float frame_scale 	= 1.0f;
    int window_sz_BM 	= 11.0;
    float ssd_th 		= 25.0f;
    float ssd_ratio_th	= 0.4f;
	size_t MIN_NUM_FEAT = 8.0;
    float MIN_GT_SCALE 	= 0.1f;
	
	ORB_VISLAM::System mySLAM(argv[2], frame_scale, 
								window_sz_BM, ssd_th, ssd_ratio_th, 
								MIN_NUM_FEAT, MIN_GT_SCALE);
	
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

	string vo_file 		= string(argv[1])	+ "VO.txt";
	string loc_vo_file 	= string(argv[1])	+ "VO_loc.txt";
	
	string gt_file 		= string(argv[1])	+ "T_GT.txt";
	string rvec_file 	= string(argv[1])	+ "rvec_GT.txt";
	
	
	ofstream f_vo, f_gt, f_rvec_abs, vo_loc;
	
	f_vo.open(vo_file.c_str());
	f_gt.open(gt_file.c_str());
	f_rvec_abs.open(rvec_file.c_str());
	vo_loc.open(loc_vo_file.c_str());
	
	f_vo << fixed;
	f_vo << "sol0_rvec_x,sol0_rvec_y,sol0_rvec_z,sol0_R00,sol0_R01,sol0_R02,sol0_tx,sol0_R10,sol0_R11,sol0_R12,sol0_ty,sol0_R20,sol0_R21,sol0_R22,sol0_tz,sol1_rvec_x,sol1_rvec_y,sol1_rvec_z,sol1_R00,sol1_R01,sol1_R02,sol1_tx,sol1_R10,sol1_R11,sol1_R12,sol1_ty,sol1_R20,sol1_R21,sol1_R22,sol1_tz,sol2_rvec_x,sol2_rvec_y,sol2_rvec_z,sol2_R00,sol2_R01,sol2_R02,sol2_tx,sol2_R10,sol2_R11,sol2_R12,sol2_ty,sol2_R20,sol2_R21,sol2_R22,sol2_tz,sol3_rvec_x,sol3_rvec_y,sol3_rvec_z,sol3_R00,sol3_R01,sol3_R02,sol3_tx,sol3_R10,sol3_R11,sol3_R12,sol3_ty,sol3_R20,sol3_R21,sol3_R22,sol3_tz,E_rvec_x,E_rvec_y,E_rvec_z,E_R00,E_R01,E_R02,E_tx,E_R10,E_R11,E_R12,E_ty,E_R20,E_R21,E_R22,E_tz" << endl;

	f_gt << fixed;
	f_gt << "T_00,T_01,T_02,T_03,T_10,T_11,T_12,T_13,T_20,T_21,T_22,T_23,T_30,T_31,T_32,T_33" << endl;

	f_rvec_abs << fixed;
	f_rvec_abs << "rvec_x,rvec_y,rvec_z" << endl;

	vo_loc << fixed;
	vo_loc << "sol0_rvec_x,sol0_rvec_y,sol0_rvec_z,sol0_R00,sol0_R01,sol0_R02,sol0_tx,sol0_R10,sol0_R11,sol0_R12,sol0_ty,sol0_R20,sol0_R21,sol0_R22,sol0_tz,sol1_rvec_x,sol1_rvec_y,sol1_rvec_z,sol1_R00,sol1_R01,sol1_R02,sol1_tx,sol1_R10,sol1_R11,sol1_R12,sol1_ty,sol1_R20,sol1_R21,sol1_R22,sol1_tz,sol2_rvec_x,sol2_rvec_y,sol2_rvec_z,sol2_R00,sol2_R01,sol2_R02,sol2_tx,sol2_R10,sol2_R11,sol2_R12,sol2_ty,sol2_R20,sol2_R21,sol2_R22,sol2_tz,sol3_rvec_x,sol3_rvec_y,sol3_rvec_z,sol3_R00,sol3_R01,sol3_R02,sol3_tx,sol3_R10,sol3_R11,sol3_R12,sol3_ty,sol3_R20,sol3_R21,sol3_R22,sol3_tz,E_rvec_x,E_rvec_y,E_rvec_z,E_R00,E_R01,E_R02,E_tx,E_R10,E_R11,E_R12,E_ty,E_R20,E_R21,E_R22,E_tz" << endl;

	clock_t tStart = clock();
	for(size_t ni = 0; ni < keyIMG.size(); ni++)
	//for(int ni = 0; ni < nImages; ni++) 
	{
		cout 	<<"\nReading Frame ["	<< imgName[keyIMG[ni]] << "]" << endl;
		string frame_name = imgName[keyIMG[ni]];
		Mat img = imread(string(argv[1]) + "frames/" + imgName[keyIMG[ni]], 
								CV_LOAD_IMAGE_GRAYSCALE);
		if(img.empty())
		{
			cerr	<<"\nFailed loading frame["
					<< string(argv[1]) 				<< "frames/"
					<< imgName[keyIMG[ni]] <<"]"
					<< endl;
			return 1;
		}
		if(img.channels() < 3) //this should be always true
		{
			cvtColor(img, img, CV_GRAY2BGR);
		}
		
		mySLAM.run(img, frame_name, f_vo, f_gt, f_rvec_abs, vo_loc, 
					T_GT[keyIMG[ni]], scale_GT[keyIMG[ni]]);
	}
	clock_t tEnd = clock();
    
    double runTime;
    runTime = (double)(tEnd - tStart)/CLOCKS_PER_SEC;
    
    cout << "\nAlgorithm time: "<< runTime << " sec.\n" << endl;
	string pc_file_name = string(argv[1]) + "point_cloud.pcd";
	mySLAM.savePointCloud(pc_file_name);
    //mySLAM.shutdown();
	return 0;
}
