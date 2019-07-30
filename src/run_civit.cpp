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

struct Angle { vector<double> roll, pitch, heading;};
struct Geodesy {vector<double> lat, lng, alt;};

void load_GNSS_INS(const string &file_path, 
					vector<double> &gpsTime, 
					Angle &angle, Geodesy &geodecy)
{
	ifstream csvFile;
	csvFile.open(file_path.c_str());

	if (!csvFile.is_open())
	{
		cout << "Wrong Path!!!!" << endl;
		exit(EXIT_FAILURE);
	}
	string line;
	vector <string> vec;
	getline(csvFile, line); // skip the 1st line (header)

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

		vector <double> lineGNSS_INS;

		while (getline(iss, lineStream, ','))
		{
			lineGNSS_INS.push_back(stold(lineStream,&sz)); // convert to double
			//lineGNSS_INS.push_back(stof(lineStream,&sz)); // convert to float
		}
		gpsTime.push_back(lineGNSS_INS[0]);
		geodecy.lat.push_back(lineGNSS_INS[3]);
		geodecy.lng.push_back(lineGNSS_INS[4]);
		geodecy.alt.push_back(lineGNSS_INS[5]);

		angle.roll.push_back(lineGNSS_INS[8]);
		angle.pitch.push_back(lineGNSS_INS[7]);
		angle.heading.push_back(lineGNSS_INS[6]);
    }
}

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
			<< " /home/xenial/Datasets/CIVIT/Dec_14/VideoFiles/seq1_short/ /home/xenial/WS_Farid/orb_slam2_TUT/settingFiles/civit.yaml\n\n"
			<< endl;
}

int main( int argc, char** argv )
{
	if(argc != 3)
	{
		printHelp(argv);
		return -1; 
	}
	
	Angle ang;
	Geodesy geo;
	vector<double> gpsT;
	
	string gnss_insFile = string(argv[1])+"/matchedNovatelData.csv";
    load_GNSS_INS(gnss_insFile, gpsT, ang, geo);
    
    
    //cout << "ang roll sz =\t" << ang.roll.size()<< endl;
    //cout << "gpsT sz =\t" << gpsT.size()<< endl;
	//Mat img;
	string imgFile = string(argv[1])+"frames/rgb.txt"; // open rgb.txt from the img folder
	
	vector<double> vTimestamps;		// retrieve ts 
	vector<string> imgName; 		// retrieve img file names ex: rgb/frame_145.jpg
    LoadImages(imgFile, imgName, vTimestamps);
    int nImages = imgName.size();

    float frame_scale 	= 0.48f;
    int window_sz_BM 	= 7;
    float ssd_th 		= 100.0f;
    float ssd_ratio_th 	= 0.6f;
    size_t MIN_NUM_FEAT = 8;
    float MIN_GT_SCALE = 0.0f;
    float distTo3DPts = 70.0f;
	
	ORB_VISLAM::System mySLAM(argv[2], frame_scale, 
								window_sz_BM, ssd_th, ssd_ratio_th, 
								MIN_NUM_FEAT, MIN_GT_SCALE, distTo3DPts,
								geo.lat[0], geo.lng[0], geo.alt[0]);
	
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
	string loc_vo_file 	= string(argv[1])	+ "VO_LOC.txt";
	
	string gt_file 		= string(argv[1])	+ "T_GT.txt";
	string util_file 	= string(argv[1])	+ "util.txt";
	
	ofstream f_vo, f_gt, f_util, f_vo_loc, f_pc;
	
	f_vo.open(vo_file.c_str());
	f_gt.open(gt_file.c_str());
	f_util.open(util_file.c_str());
	f_vo_loc.open(loc_vo_file.c_str());
	
	f_vo << fixed;
	f_vo << "E_rvec_x,E_rvec_y,E_rvec_z,E_R00,E_R01,E_R02,E_tx,E_R10,E_R11,E_R12,E_ty,E_R20,E_R21,E_R22,E_tz" << endl;
	
	f_vo_loc << fixed;
	f_vo_loc << "E_rvec_x,E_rvec_y,E_rvec_z,E_R00,E_R01,E_R02,E_tx,E_R10,E_R11,E_R12,E_ty,E_R20,E_R21,E_R22,E_tz" << endl;
	
	f_gt << fixed;
	f_gt << "T_00,T_01,T_02,T_03,T_10,T_11,T_12,T_13,T_20,T_21,T_22,T_23,T_30,T_31,T_32,T_33" << endl;

	f_util << fixed;
	f_util << "rvecGTx,rvecGTy,rvecGTz,scale,front3DPtsOPCV,front3DPtsOWN" << endl;

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
		
		mySLAM.run(img, frame_name, gpsT[keyIMG[ni]], 
					geo.lat[keyIMG[ni]], geo.lng[keyIMG[ni]], geo.alt[keyIMG[ni]], 
					ang.roll[keyIMG[ni]], ang.pitch[keyIMG[ni]], ang.heading[keyIMG[ni]], 
					f_vo, f_gt, f_util, f_vo_loc);
		
	}
	clock_t tEnd = clock();
    
    double runTime;
    runTime = (double)(tEnd - tStart)/CLOCKS_PER_SEC;
    
    cout << "\nAlgorithm time: "<< runTime << " sec.\n" << endl;
    //mySLAM.shutdown();
	return 0;
}
