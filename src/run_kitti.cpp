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

void import_GT(const string &path, vector<Mat> &T_abs, vector<float> &scale_abs)
{
	vector <float> T_00, T_01, T_02, T_03,
					T_10, T_11, T_12, T_13,
					T_20, T_21, T_22, T_23;

	Mat T_tmp = Mat::eye(4, 4, CV_32F);
	ifstream csvFile;
	csvFile.open(path.c_str());

	if (!csvFile.is_open())
	{
		cout << "Wrong Path To GROUD TRUTH FILE!!!!" << endl;
		exit(EXIT_FAILURE);
	}
	string line;
	//getline(csvFile, line); // skip the 1st line (header)

    while (getline(csvFile,line))
    {
        if (line.empty()) // skip empty lines:
        {
            //cout << "empty line!" << endl;
            continue;
        }

        istringstream iss(line);
        string lineStream;
        string::size_type sz;

		//vector <double> T_line;
		vector <float> T_line;

		while (getline(iss, lineStream, ' '))
		{
			//T_line.push_back(stold(lineStream,&sz)); // convert to double
			T_line.push_back(stof(lineStream, &sz)); // convert to float
		}
		
		T_00.push_back(T_line[0]);	
		T_01.push_back(T_line[1]);	
		T_02.push_back(T_line[2]);
		T_03.push_back(T_line[3]);	// x
				
		T_10.push_back(T_line[4]);	
		T_11.push_back(T_line[5]);	
		T_12.push_back(T_line[6]);
		T_13.push_back(T_line[7]);	// y

		T_20.push_back(T_line[8]);	
		T_21.push_back(T_line[9]);	
		T_22.push_back(T_line[10]);
		T_23.push_back(T_line[11]);	// z	
    }
    for(size_t j = 0; j < T_00.size(); j++)
    {
		T_tmp.at<float>(0,0) = T_00[j];
		T_tmp.at<float>(0,1) = T_01[j];
		T_tmp.at<float>(0,2) = T_02[j];
		T_tmp.at<float>(0,3) = T_03[j];	// x
		
		T_tmp.at<float>(1,0) = T_10[j];
		T_tmp.at<float>(1,1) = T_11[j];
		T_tmp.at<float>(1,2) = T_12[j];
		T_tmp.at<float>(1,3) = T_13[j];	// y

		T_tmp.at<float>(2,0) = T_20[j];
		T_tmp.at<float>(2,1) = T_21[j];
		T_tmp.at<float>(2,2) = T_22[j];
		T_tmp.at<float>(2,3) = T_23[j];	// z

		T_tmp.at<float>(3,0) = 0;
		T_tmp.at<float>(3,1) = 0;
		T_tmp.at<float>(3,2) = 0;
		T_tmp.at<float>(3,3) = 1;
		
    	//cout << "\n\ntmp_T [" << j <<"]= \n" << T_tmp << endl;
		T_abs.push_back(T_tmp.clone());
	}	
	scale_abs.push_back(1); 
	for(size_t k = 0; k < T_00.size()-1; k++)
	{
		scale_abs.push_back(
							sqrt(
							(T_03[k+1] - T_03[k]) * (T_03[k+1] - T_03[k]) + 
							(T_13[k+1] - T_13[k]) * (T_13[k+1] - T_13[k]) + 
							(T_23[k+1] - T_23[k]) * (T_23[k+1] - T_23[k])
							)
							);
	}
}

void import_seq(const string &path, 
				string &imgFiles, 
				vector<double> &vTimestamps)
{
	imgFiles = path + "/image_0/%06d.png";
	string tsFile = path +"/times.txt";
	ifstream f;
	f.open(tsFile.c_str());

	if (!f.is_open())
	{
		cout << "Wrong Path To Image Folder!" << endl;
		exit(EXIT_FAILURE);
	}

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
			<< " /home/xenial/Datasets/KITTI/ /home/xenial/WS_Farid/orb_slam2_TUT/Examples/Monocular/KITTI00-02.yaml\n\n"
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
	string seqPath = string(argv[1]) + "sequences/05";
	string GT_Path = string(argv[1]) + "poses/05.txt";
	
	// improt ground truth:
	vector<Mat> T_GT;
	vector<float> scale_GT;
	import_GT(GT_Path, T_GT, scale_GT);
	
	cout << "T_GT sz = \t" << T_GT.size() << "\tscale_GT sz = \t"<< scale_GT.size()<< endl;
	// import image files:
	vector<double> vTimestamps;		// retrieve ts
	string imgFolder;
    import_seq(seqPath, imgFolder, vTimestamps);
    int nImages = vTimestamps.size();
    
    
    float frame_scale = 0.74f;
    int window_sz_BM 	= 3;
    float ssd_th 		= 12.0f;
    float ssd_ratio_th 	= 0.7f;
    size_t MIN_NUM_FEAT = 8;
    float MIN_GT_SCALE = 0.1;
    
    cout << "no images = " << nImages << endl;
	
	ORB_VISLAM::System mySLAM(argv[2], frame_scale, 
								window_sz_BM, ssd_th, 
								ssd_ratio_th, MIN_NUM_FEAT, MIN_GT_SCALE);
	
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
	
	string vo_file 		= seqPath	+ "/VO.txt";
	string loc_vo_file 	= seqPath	+ "/VO_loc.txt";
	
	string gt_file 		= seqPath	+ "/T_GT.txt";
	string rvec_file 	= seqPath	+ "/rvec_GT.txt";
	string pc_file 		= seqPath	+ "/point_cloud.txt";
	
	
	ofstream f_vo, f_gt, f_rvec_abs, vo_loc, f_pc;
	
	f_vo.open(vo_file.c_str());
	f_gt.open(gt_file.c_str());
	f_rvec_abs.open(rvec_file.c_str());
	vo_loc.open(loc_vo_file.c_str());
	f_pc.open(pc_file.c_str());
	
	f_pc << fixed;
	f_pc << "p3d_x,p3d_y,p3d_z" << endl;
	
	f_vo << fixed;
	f_vo << "sol0_rvec_x,sol0_rvec_y,sol0_rvec_z,sol0_R00,sol0_R01,sol0_R02,sol0_tx,sol0_R10,sol0_R11,sol0_R12,sol0_ty,sol0_R20,sol0_R21,sol0_R22,sol0_tz,sol1_rvec_x,sol1_rvec_y,sol1_rvec_z,sol1_R00,sol1_R01,sol1_R02,sol1_tx,sol1_R10,sol1_R11,sol1_R12,sol1_ty,sol1_R20,sol1_R21,sol1_R22,sol1_tz,sol2_rvec_x,sol2_rvec_y,sol2_rvec_z,sol2_R00,sol2_R01,sol2_R02,sol2_tx,sol2_R10,sol2_R11,sol2_R12,sol2_ty,sol2_R20,sol2_R21,sol2_R22,sol2_tz,sol3_rvec_x,sol3_rvec_y,sol3_rvec_z,sol3_R00,sol3_R01,sol3_R02,sol3_tx,sol3_R10,sol3_R11,sol3_R12,sol3_ty,sol3_R20,sol3_R21,sol3_R22,sol3_tz,E_rvec_x,E_rvec_y,E_rvec_z,E_R00,E_R01,E_R02,E_tx,E_R10,E_R11,E_R12,E_ty,E_R20,E_R21,E_R22,E_tz" << endl;

	f_gt << fixed;
	f_gt << "T_00,T_01,T_02,T_03,T_10,T_11,T_12,T_13,T_20,T_21,T_22,T_23,T_30,T_31,T_32,T_33" << endl;

	f_rvec_abs << fixed;
	f_rvec_abs << "rvec_x,rvec_y,rvec_z" << endl;

	vo_loc << fixed;
	vo_loc << "sol0_rvec_x,sol0_rvec_y,sol0_rvec_z,sol0_R00,sol0_R01,sol0_R02,sol0_tx,sol0_R10,sol0_R11,sol0_R12,sol0_ty,sol0_R20,sol0_R21,sol0_R22,sol0_tz,sol1_rvec_x,sol1_rvec_y,sol1_rvec_z,sol1_R00,sol1_R01,sol1_R02,sol1_tx,sol1_R10,sol1_R11,sol1_R12,sol1_ty,sol1_R20,sol1_R21,sol1_R22,sol1_tz,sol2_rvec_x,sol2_rvec_y,sol2_rvec_z,sol2_R00,sol2_R01,sol2_R02,sol2_tx,sol2_R10,sol2_R11,sol2_R12,sol2_ty,sol2_R20,sol2_R21,sol2_R22,sol2_tz,sol3_rvec_x,sol3_rvec_y,sol3_rvec_z,sol3_R00,sol3_R01,sol3_R02,sol3_tx,sol3_R10,sol3_R11,sol3_R12,sol3_ty,sol3_R20,sol3_R21,sol3_R22,sol3_tz,E_rvec_x,E_rvec_y,E_rvec_z,E_R00,E_R01,E_R02,E_tx,E_R10,E_R11,E_R12,E_ty,E_R20,E_R21,E_R22,E_tz" << endl;


	char filename[400];
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
		mySLAM.run(img, frame_name, f_vo, f_gt, f_rvec_abs, vo_loc, f_pc,
					T_GT[keyIMG[ni]], scale_GT[keyIMG[ni]]);
	}
	clock_t tEnd = clock();
	double runTime;
	runTime = (double)(tEnd - tStart)/CLOCKS_PER_SEC;
	cout << "\nAlgorithm time: "<< runTime << " sec.\n" << endl;
	
	//mySLAM.shutdown();
	return 0;
}
