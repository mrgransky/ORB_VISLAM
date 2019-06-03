#include "System.h"

using namespace std;
using namespace cv;
#define PI 3.1415926f

namespace ORB_VISLAM
{

System::System(	const string &settingFilePath, float scale,
						int win_sz, float ssd_th, float ssd_ratio_th)
{
	cout << "\n\n" << endl;
	cout << "#########################################################################" << endl;
	cout << "\t\t\t\tSYSTEN V"															<< endl;
	cout << "#########################################################################" << endl;
	
	frame_avl = true;
	
	// init vision class:
	visionPtr		= new Vision(settingFilePath, win_sz, ssd_th, ssd_ratio_th);
	
	// initialize visualizer class
	visualizerPtr 	= new Visualizer(visionPtr->IMG_, visionPtr->T_cam, 
										visionPtr->fps, scale, frame_avl);
	
	// run visualizer thread
	visThread 		= new thread(&Visualizer::run, visualizerPtr);
}

System::System(	const string &settingFilePath, float scale,
				int win_sz, float ssd_th, float ssd_ratio_th,
				double &ref_lat, 
				double &ref_lng, 
				double &ref_alt)/*:	init_absPose(ref_lat, ref_lng, ref_alt),
									init_visualizer()*/
{
	cout << "\n\n" << endl;
	cout << "#########################################################################" << endl;
	cout << "\t\t\t\tSYSTEN VI"															<< endl;
	cout << "#########################################################################" << endl;
	frame_avl = true;
	
	// initialize absPose class:
	absPosePtr 		= new AbsolutePose(ref_lat, ref_lng, ref_alt);
	
	// init vision class:
	visionPtr		= new Vision(settingFilePath, win_sz, ssd_th, ssd_ratio_th);
	
	// initialize visualizer class
	visualizerPtr 	= new Visualizer(	visionPtr->IMG_, visionPtr->T_cam, visionPtr->fps, 
										absPosePtr->T_abs, scale, frame_avl);
	
	
	// run visualizer thread
	visThread 		= new thread(&Visualizer::run, visualizerPtr);
}

System::~System()
{
	visThread->join();
}

void System::run(Mat &raw_frame, string &frame_name, ofstream &file_cam)
{

	vector<pair<int,int>> matches;
	vector<KeyPoint> KP;
	
	visionPtr->Analyze(raw_frame, KP, matches);
	visualizerPtr->show(raw_frame, KP, matches, frame_name);
	
	saveTraj(visionPtr->T_cam, file_cam);
}

void System::run(Mat &raw_frame, string &frame_name, double &lat, double &lng, double &alt, 
					double &roll, double &pitch, double &heading, 
					ofstream &file_GT, ofstream &file_cam)
{

	vector<pair<int,int>> matches;
	vector<KeyPoint> KP;
	
	visionPtr->Analyze(raw_frame, KP, matches);
	visualizerPtr->show(raw_frame, KP, matches, frame_name);
	absPosePtr->calcPose(lat, lng, alt, roll, pitch, heading);
	saveTraj(absPosePtr->T_abs, file_GT);
	saveTraj(visionPtr->T_cam, file_cam);
}

void System::saveTraj(Mat T, ofstream &file_)
{
	file_	<< setprecision(15)	<< T.at<float>(0,3) 	<< ","
			<< setprecision(15)	<< T.at<float>(1,3) 	<< ","
			<< setprecision(15)	<< T.at<float>(2,3) 	<< endl;
}
void System::shutdown()
{
	frame_avl = false; 	// if activated in main.cpp:
						// TODO: deteriorate the while loop in visualizer class,
						//-->>>>> (last frame invisible)
	visualizerPtr->hasFrame = false;
	cout << "system is shutting down!" << endl;
}
}
