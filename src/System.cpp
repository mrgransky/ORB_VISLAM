#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include <pangolin/pangolin.h>

#include <opencv2/core/eigen.hpp>
#include <stdio.h>      /* printf, fopen */
#include <stdlib.h>     /* exit, EXIT_FAILURE */
#include <thread>
#include "System.h"

using namespace std;
using namespace cv;
#define PI 3.1415926f

namespace ORB_VISLAM
{

System::System(const std::string &settingFilePath, 
				double &ref_lat, 
				double &ref_lng, 
				double &ref_alt):absPose(ref_lat, ref_lng, ref_alt)
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
	
}

Mat System::run_abs_pose(double &lat, double &lng, double &alt, 
							double &roll, double &pitch, double &heading)
{
	return absPose.getPose(lat, lng, alt, roll, pitch, heading);

}

}

