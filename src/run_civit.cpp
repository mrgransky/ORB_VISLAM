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
	string imgFile = string(argv[1])+"/frames/rgb.txt"; // open rgb.txt from the img folder
	
	vector<double> vTimestamps;		// retrieve ts 
	vector<string> imgName; 		// retrieve img file names ex: rgb/frame_145.jpg
    LoadImages(imgFile, imgName, vTimestamps);
    int nImages = imgName.size();

    float frame_scale 	= 0.48f;
    int window_sz_BM 	= 15;
    float ssd_th 		= 15.0f;
    float ssd_ratio_th 	= 0.6f;
    
	ORB_VISLAM::System mySLAM(argv[2], frame_scale, window_sz_BM, ssd_th, ssd_ratio_th,
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
	
	string traj_GT = string(argv[1])+"frames/GT.txt";
	string vo_file = string(argv[1])+"frames/VO.txt";
	
	ofstream f_GT, f_cam;
	f_GT.open(traj_GT.c_str());
	f_cam.open(vo_file.c_str());
	
	f_GT << fixed;
	f_GT << "rvec_x[rad],rvec_y[rad],rvec_z[rad],R00,R01,R02,tx,R10,R11,R12,ty,R20,R21,R22,tz"<< endl;

	f_cam << fixed;
	f_cam 	<< "GpsTime,Lat[deg],lng[deg],alt[m],roll[deg],pitch[deg],heading[deg],sol1_rvec_x[rad],sol1_rvec_y[rad],sol1_rvec_z[rad], sol1_R00,sol1_R01,sol1_R02,sol1_tx,sol1_R10,sol1_R11,sol1_R12,sol1_ty,sol1_R20,sol1_R21,sol1_R22,sol1_tz,sol2_rvec_x[rad],sol2_rvec_y[rad],sol2_rvec_z[rad],sol2_R00,sol2_R01,sol2_R02,sol2_tx,sol2_R10,sol2_R11,sol2_R12,sol2_ty,sol2_R20,sol2_R21,sol2_R22,sol2_tz,sol3_rvec_x[rad],sol3_rvec_y[rad],sol3_rvec_z[rad],sol3_R00,sol3_R01,sol3_R02,sol3_tx,sol3_R10,sol3_R11,sol3_R12,sol3_ty,sol3_R20,sol3_R21,sol3_R22,sol3_tz,sol4_rvec_x[rad],sol4_rvec_y[rad],sol4_rvec_z[rad],sol4_R00,sol4_R01,sol4_R02,sol4_tx,sol4_R10,sol4_R11,sol4_R12,sol4_ty,sol4_R20,sol4_R21,sol4_R22,sol4_tz" 
			<< endl;

	clock_t tStart = clock();
	for(size_t ni = 0; ni < keyIMG.size(); ni++)
	//for(int ni = 0; ni < nImages; ni++) 
	{
		cout 	<<"\n\nReading Frame ["	<< imgName[keyIMG[ni]] << "]" << endl;
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
					f_GT, f_cam);
		
	}
	clock_t tEnd = clock();
    
    double runTime;
    runTime = (double)(tEnd - tStart)/CLOCKS_PER_SEC;
    
    cout << "\nAlgorithm time: "<< runTime << " sec.\n" << endl;
    //mySLAM.shutdown();
	return 0;
}
