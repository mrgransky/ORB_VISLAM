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


void load_GNSS_INS(const string &file_path, Angle &angle, Geodesy &geodecy)
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
	
	string gnss_insFile = string(argv[1])+"/matchedNovatelData.csv";
    load_GNSS_INS(gnss_insFile, ang, geo);
    
	//Mat img;
	string imgFile = string(argv[1])+"/frames/rgb.txt"; // open rgb.txt from the img folder
	
	vector<double> vTimestamps;		// retrieve ts 
	vector<string> imgName; 		// retrieve img file names ex: rgb/frame_145.jpg
    LoadImages(imgFile, imgName, vTimestamps);
    int nImages = imgName.size();

	ORB_VISLAM::System mySLAM(argv[2], geo.lat[0], geo.lng[0], geo.alt[0]);
	
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
	
	string traj_GT = string(argv[1])+"frames/GNSS_INS_Trajectory.txt";
	string traj_cam = string(argv[1])+"frames/VO_Trajectory.txt";
	
	ofstream f_GT, f_cam;
	f_GT.open(traj_GT.c_str());
	f_cam.open(traj_cam.c_str());
	
	f_GT << fixed;
	f_GT << "x,y,z"<< endl;

	f_cam << fixed;
	f_cam << "x,y,z"<< endl;

	clock_t tStart = clock();
	for(size_t ni = 0; ni < keyIMG.size(); ni++)
	//for(int ni = 0; ni < nImages; ni++) 
	{
		cout 	<<"\n\nReading Frame["	<< imgName[keyIMG[ni]] << "]" << endl;
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
			cout << "ch B4 = \t" << img.channels() 
			<< " , depth B4 = \t" << img.depth()
			<< " , type B4 = \t" << img.type() 
			<< endl;
			cvtColor(img, img, CV_GRAY2BGR);
		}
		cout << "ch after = \t" << img.channels() 
		<< " , depth after = \t" << img.depth()
		<< " , type after = \t" << img.type() 
		<< endl;
		
		/*mySLAM.run(img, frame_name, geo.lat[keyIMG[ni]], geo.lng[keyIMG[ni]], geo.alt[keyIMG[ni]],
					ang.roll[keyIMG[ni]], ang.pitch[keyIMG[ni]], ang.heading[keyIMG[ni]], 
					f_GT, f_cam);*/
		
	}
	clock_t tEnd = clock();
    
    double runTime;
    runTime = (double)(tEnd - tStart)/CLOCKS_PER_SEC;
    
    cout << "\nAlgorithm time: "<< runTime << " sec.\n" << endl;
    //mySLAM.shutdown();
	return 0;
}
