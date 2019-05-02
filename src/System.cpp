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
	
	FileStorage fSettings(settingFilePath, FileStorage::READ);
    float fx 			= fSettings["Camera.fx"];
    float fy 			= fSettings["Camera.fy"];
    float cx 			= fSettings["Camera.cx"];
    float cy 			= fSettings["Camera.cy"];
	
	Mat K = Mat::eye(3, 3, CV_32F);
	
	K.at<float>(0,0) 		= fx;
    K.at<float>(1,1) 		= fy;
    K.at<float>(0,2) 		= cx;
    K.at<float>(1,2) 		= cy;

	K.copyTo(mK);
	
	Mat DistCoef(4, 1, CV_32F);
	
    DistCoef.at<float>(0) 	= fSettings["Camera.k1"];
    DistCoef.at<float>(1) 	= fSettings["Camera.k2"];
    DistCoef.at<float>(2) 	= fSettings["Camera.p1"];
    DistCoef.at<float>(3) 	= fSettings["Camera.p2"];
    const float k3 			= fSettings["Camera.k3"];

    if(k3!=0)
    {
        DistCoef.resize(5);
        DistCoef.at<float>(4) = k3;
    }
    DistCoef.copyTo(mDistCoef);
    
	cout << "\n\n" << endl;
	cout << "#########################################################################" << endl;
	cout << "\t\t\tCAMERA PARAMETERS"													<< endl;
	cout << "#########################################################################" << endl;
	
    cout << "- fx: " << fx << endl;
    cout << "- fy: " << fy << endl;
    cout << "- cx: " << cx << endl;
    cout << "- cy: " << cy << endl;
    cout << "- k1: " << DistCoef.at<float>(0) << endl;
    cout << "- k2: " << DistCoef.at<float>(1) << endl;
    if(DistCoef.rows == 5)
        cout << "- k3: " << DistCoef.at<float>(4) << endl;
    cout << "- p1: " << DistCoef.at<float>(2) << endl;
    cout << "- p2: " << DistCoef.at<float>(3) << endl;
	
	// initialize absPose class:
	absPosePtr 		= new AbsolutePose(ref_lat, ref_lng, ref_alt);
	
	// init vision class:
	visionPtr		= new Vision();
	
	// initialize visualizer class
	visualizerPtr 	= new Visualizer(visionPtr->IMG_, absPosePtr->T_abs);
	
	// run visualizer thread
	visThread 		= new thread(&Visualizer::run, visualizerPtr);
}

System::~System()
{
	visThread->join();
}

void System::run(Mat &raw_frame, double &lat, double &lng, double &alt, 
					double &roll, double &pitch, double &heading, ofstream &file_)
{
	Mat AnalyzedFrame = visionPtr->Analyze(raw_frame);
	
	visualizerPtr->show(AnalyzedFrame);
	
	
	absPosePtr->calcPose(lat, lng, alt, roll, pitch, heading);
	saveTraj(absPosePtr->T_abs, file_);
	
}
void System::saveTraj(Mat T, ofstream &file_)
{
	file_	<< setprecision(15)	<< T.at<float>(0,3) 	<< ","
			<< setprecision(15)	<< T.at<float>(1,3) 	<< ","
			<< setprecision(15)	<< T.at<float>(2,3) 	<< endl;
}
}

