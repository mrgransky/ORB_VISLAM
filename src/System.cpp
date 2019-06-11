#include "System.h"

using namespace std;
using namespace cv;
#define PI 3.1415926f

namespace ORB_VISLAM
{

System::System(	const string &settingFilePath, float scale,
						int win_sz, float ssd_th, float ssd_ratio_th)
{
	cout << "" << endl;
	cout << "#########################################################################" << endl;
	cout << "\t\t\t\tSYSTEN V"															<< endl;
	cout << "#########################################################################" << endl;
	
	frame_avl = true;
	
	Mat identityMAT = Mat::eye(3, 3, CV_64F);
	Mat zeroMAT = Mat::zeros(3, 1, CV_64F);

	R_prev 		= vector<Mat>{identityMAT, identityMAT, identityMAT, identityMAT};
	t_prev 		= vector<Mat>{zeroMAT, zeroMAT, zeroMAT, zeroMAT};
	rvec_prev 	= vector<Mat>{zeroMAT, zeroMAT, zeroMAT, zeroMAT};
	
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
	cout << "" << endl;
	cout << "#########################################################################" << endl;
	cout << "\t\t\t\tSYSTEN VI"															<< endl;
	cout << "#########################################################################" << endl;
	frame_avl = true;
	
	Mat identityMAT = Mat::eye(3, 3, CV_64F);
	Mat zeroMAT = Mat::zeros(3, 1, CV_64F);

	R_prev 		= vector<Mat>{identityMAT, identityMAT, identityMAT, identityMAT};
	t_prev 		= vector<Mat>{zeroMAT, zeroMAT, zeroMAT, zeroMAT};
	rvec_prev 	= vector<Mat>{zeroMAT, zeroMAT, zeroMAT, zeroMAT};
	
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

}

void System::run(Mat &raw_frame, string &frame_name, double &gpsT,
					double &lat, double &lng, double &alt, 
					double &roll, double &pitch, double &heading, 
					ofstream &file_GT, ofstream &file_cam)
{

	vector<pair<int,int>> matches;
	vector<KeyPoint> KP;
	
	visionPtr->Analyze(raw_frame, KP, matches);
	visualizerPtr->show(raw_frame, KP, matches, frame_name);
	absPosePtr->calcPose(lat, lng, alt, roll, pitch, heading);
	
	saveTraj(absPosePtr->T_abs, absPosePtr->rvec_GT, file_GT);
	
	//saveVOFile(gpsT, lat, lng, alt, roll, pitch, heading, visionPtr->T_cam, file_cam);
	saveVOFile(gpsT, lat, lng, alt, roll, pitch, heading, 
				visionPtr->Rdec, visionPtr->tdec, visionPtr->vrvec_dec, file_cam);

	saveVOFile(gpsT, lat, lng, alt, roll, pitch, heading, 
				visionPtr->Rdec, visionPtr->tdec, visionPtr->vrvec_dec, file_cam);
	

}

void System::saveTraj(Mat &T, Mat &rvec, ofstream &file_)
{
	file_	<< setprecision(8)	<< rvec.at<float>(0,0) 	<< ","
			<< setprecision(8)	<< rvec.at<float>(1,0) 	<< ","
			<< setprecision(8)	<< rvec.at<float>(2,0) 	<< ","
			<< setprecision(8)	<< T.at<float>(0,0) 	<< ","
			<< setprecision(8)	<< T.at<float>(0,1) 	<< ","
			<< setprecision(8)	<< T.at<float>(0,2)		<< ","
			<< setprecision(8)	<< T.at<float>(0,3) 	<< ","
			<< setprecision(8)	<< T.at<float>(1,0) 	<< ","
			<< setprecision(8)	<< T.at<float>(1,1) 	<< ","
			<< setprecision(8)	<< T.at<float>(1,2) 	<< ","
			<< setprecision(8)	<< T.at<float>(1,3) 	<< ","
			<< setprecision(8)	<< T.at<float>(2,0) 	<< ","
			<< setprecision(8)	<< T.at<float>(2,1) 	<< ","
			<< setprecision(8)	<< T.at<float>(2,2) 	<< ","
			<< setprecision(8)	<< T.at<float>(2,3) 	<< endl;
}

void System::saveVOFile(double &gpsT, double &lat, double &lng, double &alt, 
						double &roll, double &pitch, double &heading,
						Mat &T, ofstream &file_)
{
	file_	<< setprecision(15)	<< gpsT 			<< ","
			<< setprecision(15)	<< lat 				<< ","
			<< setprecision(15)	<< lng 				<< ","
			<< setprecision(15)	<< alt 				<< ","
			<< setprecision(15)	<< roll 			<< ","
			<< setprecision(15)	<< pitch 			<< ","
			<< setprecision(15)	<< heading 			<< ","
			<< setprecision(15)	<< T.at<float>(0,0) << ","
			<< setprecision(15)	<< T.at<float>(0,1) << ","
			<< setprecision(15)	<< T.at<float>(0,2)	<< ","
			<< setprecision(15)	<< T.at<float>(0,3) << ","
			<< setprecision(15)	<< T.at<float>(1,0) << ","
			<< setprecision(15)	<< T.at<float>(1,1) << ","
			<< setprecision(15)	<< T.at<float>(1,2) << ","
			<< setprecision(15)	<< T.at<float>(1,3) << ","
			<< setprecision(15)	<< T.at<float>(2,0) << ","
			<< setprecision(15)	<< T.at<float>(2,1) << ","
			<< setprecision(15)	<< T.at<float>(2,2) << ","
			<< setprecision(15)	<< T.at<float>(2,3) << endl;
}

void System::saveVOFile(double &gpsT, double &lat, double &lng, double &alt, 
						double &roll, double &pitch, double &heading,
						vector<Mat> &R_dc, vector<Mat> &t_dc, vector<Mat> &rvec, ofstream &file_)
{	
	if(R_dc.size() != 4)
	{
		//cout << "rvec.size() =\t" << rvec.size() << endl;
		R_dc = R_prev;
		t_dc = t_prev;
		rvec = rvec_prev;	
	}
	
	
	/*for (size_t i = 0; i < R_dc.size(); i++)
	{
		cout << "R_dc [" << i << "] = \n"<< R_dc[i] << endl;
	}
	
	for (size_t i = 0; i < rvec.size(); i++)
	{
		cout << "rvec [" << i << "] = \n"<< rvec[i] << endl;
	}*/
	
	
	
		file_		<< setprecision(15)	<< gpsT 			<< ","
					<< setprecision(15)	<< lat 				<< ","
					<< setprecision(15)	<< lng 				<< ","
					<< setprecision(15)	<< alt 				<< ","
					<< setprecision(15)	<< roll 			<< ","
					<< setprecision(15)	<< pitch 			<< ","
					<< setprecision(15)	<< heading 			<< ","
					
					<< setprecision(8)	<< rvec[0].at<double>(0,0) 	<< ","
					<< setprecision(8)	<< rvec[0].at<double>(1,0) 	<< ","
					<< setprecision(8)	<< rvec[0].at<double>(2,0) 	<< ","
					<< setprecision(8)	<< R_dc[0].at<double>(0,0) 	<< ","
					<< setprecision(8)	<< R_dc[0].at<double>(0,1) 	<< ","
					<< setprecision(8)	<< R_dc[0].at<double>(0,2)	<< ","
					<< setprecision(8)	<< t_dc[0].at<double>(0,0) 	<< ","
					<< setprecision(8)	<< R_dc[0].at<double>(1,0) 	<< ","
					<< setprecision(8)	<< R_dc[0].at<double>(1,1) 	<< ","
					<< setprecision(8)	<< R_dc[0].at<double>(1,2) 	<< ","
					<< setprecision(8)	<< t_dc[0].at<double>(1,0) 	<< ","
					<< setprecision(8)	<< R_dc[0].at<double>(2,0) 	<< ","
					<< setprecision(8)	<< R_dc[0].at<double>(2,1) 	<< ","
					<< setprecision(8)	<< R_dc[0].at<double>(2,2) 	<< ","
					<< setprecision(8)	<< t_dc[0].at<double>(2,0) 	<< ","
					
					<< setprecision(8)	<< rvec[1].at<double>(0,0) 	<< ","
					<< setprecision(8)	<< rvec[1].at<double>(1,0) 	<< ","
					<< setprecision(8)	<< rvec[1].at<double>(2,0) 	<< ","
					<< setprecision(8)	<< R_dc[1].at<double>(0,0) 	<< ","
					<< setprecision(8)	<< R_dc[1].at<double>(0,1) 	<< ","
					<< setprecision(8)	<< R_dc[1].at<double>(0,2)	<< ","
					<< setprecision(8)	<< t_dc[1].at<double>(0,0) 	<< ","
					<< setprecision(8)	<< R_dc[1].at<double>(1,0) 	<< ","
					<< setprecision(8)	<< R_dc[1].at<double>(1,1) 	<< ","
					<< setprecision(8)	<< R_dc[1].at<double>(1,2) 	<< ","
					<< setprecision(8)	<< t_dc[1].at<double>(1,0) 	<< ","
					<< setprecision(8)	<< R_dc[1].at<double>(2,0) 	<< ","
					<< setprecision(8)	<< R_dc[1].at<double>(2,1) 	<< ","
					<< setprecision(8)	<< R_dc[1].at<double>(2,2) 	<< ","
					<< setprecision(8)	<< t_dc[1].at<double>(2,0) 	<< ","
					
					<< setprecision(8)	<< rvec[2].at<double>(0,0) 	<< ","
					<< setprecision(8)	<< rvec[2].at<double>(1,0) 	<< ","
					<< setprecision(8)	<< rvec[2].at<double>(2,0) 	<< ","
					<< setprecision(8)	<< R_dc[2].at<double>(0,0) 	<< ","
					<< setprecision(8)	<< R_dc[2].at<double>(0,1) 	<< ","
					<< setprecision(8)	<< R_dc[2].at<double>(0,2)	<< ","
					<< setprecision(8)	<< t_dc[2].at<double>(0,0) 	<< ","
					<< setprecision(8)	<< R_dc[2].at<double>(1,0) 	<< ","
					<< setprecision(8)	<< R_dc[2].at<double>(1,1) 	<< ","
					<< setprecision(8)	<< R_dc[2].at<double>(1,2) 	<< ","
					<< setprecision(8)	<< t_dc[2].at<double>(1,0) 	<< ","
					<< setprecision(8)	<< R_dc[2].at<double>(2,0) 	<< ","
					<< setprecision(8)	<< R_dc[2].at<double>(2,1) 	<< ","
					<< setprecision(8)	<< R_dc[2].at<double>(2,2) 	<< ","
					<< setprecision(8)	<< t_dc[2].at<double>(2,0) 	<< ","
					
					<< setprecision(8)	<< rvec[3].at<double>(0,0) 	<< ","
					<< setprecision(8)	<< rvec[3].at<double>(1,0) 	<< ","
					<< setprecision(8)	<< rvec[3].at<double>(2,0) 	<< ","
					<< setprecision(8)	<< R_dc[3].at<double>(0,0) 	<< ","
					<< setprecision(8)	<< R_dc[3].at<double>(0,1) 	<< ","
					<< setprecision(8)	<< R_dc[3].at<double>(0,2)	<< ","
					<< setprecision(8)	<< t_dc[3].at<double>(0,0) 	<< ","
					<< setprecision(8)	<< R_dc[3].at<double>(1,0) 	<< ","
					<< setprecision(8)	<< R_dc[3].at<double>(1,1) 	<< ","
					<< setprecision(8)	<< R_dc[3].at<double>(1,2) 	<< ","
					<< setprecision(8)	<< t_dc[3].at<double>(1,0) 	<< ","
					<< setprecision(8)	<< R_dc[3].at<double>(2,0) 	<< ","
					<< setprecision(8)	<< R_dc[3].at<double>(2,1) 	<< ","
					<< setprecision(8)	<< R_dc[3].at<double>(2,2) 	<< ","
					<< setprecision(8)	<< t_dc[3].at<double>(2,0) 	
					<< endl;

	R_prev 		= R_dc;
	t_prev 		= t_dc;
	rvec_prev 	= rvec;
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

