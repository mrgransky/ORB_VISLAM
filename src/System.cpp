#include "System.h"

using namespace std;
using namespace cv;
#define PI 3.1415926f

namespace ORB_VISLAM
{

System::System(	const string &settingFilePath, 
				double &ref_lat, 
				double &ref_lng, 
				double &ref_alt)/*:	init_absPose(ref_lat, ref_lng, ref_alt),
									init_visualizer()*/
{
	cout << "\n\n" << endl;
	cout << "#########################################################################" << endl;
	cout << "\t\t\t\tSYSTEN"															<< endl;
	cout << "#########################################################################" << endl;
	frame_avl = true;
	
	// initialize absPose class:
	absPosePtr 		= new AbsolutePose(ref_lat, ref_lng, ref_alt);
	
	// init vision class:
	visionPtr		= new Vision(settingFilePath);
	
	// initialize visualizer class
	visualizerPtr 	= new Visualizer(visionPtr->IMG_, absPosePtr->T_abs, frame_avl);
	
	// run visualizer thread
	visThread 		= new thread(&Visualizer::run, visualizerPtr);
}

System::~System()
{
	visThread->join();
}

void System::run(Mat &raw_frame, string &frame_name, double &lat, double &lng, double &alt, 
					double &roll, double &pitch, double &heading, 
					ofstream &file_)
{	
	int sFPS = visionPtr->fps;
	/*Mat AnalyzedFrame = visionPtr->Analyze(raw_frame);
	visualizerPtr->show(AnalyzedFrame, frame_name, sFPS);*/
	

	vector<pair<int,int>> matches;
	vector<KeyPoint> KP;
	
	visionPtr->Analyze(raw_frame, KP, matches);
	visualizerPtr->show(raw_frame, KP, matches, frame_name, sFPS);
	
	
	
	
	
	
	
	
	
	
	
	absPosePtr->calcPose(lat, lng, alt, roll, pitch, heading);
	saveTraj(absPosePtr->T_abs, file_);
	
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

