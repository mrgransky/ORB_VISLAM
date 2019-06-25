#include "System.h"

using namespace std;
using namespace cv;

namespace ORB_VISLAM
{

System::System(	const string &settingFilePath, float scale,
						int win_sz, float ssd_th, float ssd_ratio_th, size_t minFeat)
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
	visionPtr		= new Vision(settingFilePath, win_sz, ssd_th, ssd_ratio_th, minFeat);
	
	/*// initialize visualizer class
	// ##################### 1 soultions ##################### //
	visualizerPtr 	= new Visualizer(visionPtr->IMG_, visionPtr->T_cam, 
										visionPtr->fps, scale, frame_avl);*/
	
	// initialize visualizer class
	// ##################### 4 soultions ##################### //
	visualizerPtr 	= new Visualizer(visionPtr->IMG_, visionPtr->T_cam_0, visionPtr->T_cam_1, 
														visionPtr->T_cam_2, visionPtr->T_cam_3, 
														visionPtr->fps, scale, frame_avl);
	
	// run visualizer thread
	visThread 		= new thread(&Visualizer::run, visualizerPtr);
}

System::System(	const string &settingFilePath, float scale,
				int win_sz, float ssd_th, float ssd_ratio_th, size_t minFeat,
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
	visionPtr		= new Vision(settingFilePath, win_sz, ssd_th, ssd_ratio_th, minFeat);
	
	/*// initialize visualizer class
	// ##################### 1 soultions ##################### //
	visualizerPtr 	= new Visualizer(	visionPtr->IMG_, absPosePtr->T_abs, 
										visionPtr->T_cam, visionPtr->fps, 
										 scale, frame_avl);*/
	// initialize visualizer class
	// ##################### 4 soultions ##################### //
	visualizerPtr 	= new Visualizer(	visionPtr->IMG_, absPosePtr->T_abs, 
										visionPtr->T_cam_0, visionPtr->T_cam_1, 
										visionPtr->T_cam_2, visionPtr->T_cam_3,
										visionPtr->fps, scale, frame_avl);
	
	// run visualizer thread
	visThread 		= new thread(&Visualizer::run, visualizerPtr);
}

System::~System()
{
	visThread->join();
}

void System::run(Mat &raw_frame, string &frame_name, ofstream &file_vo, 
									ofstream &file_h, ofstream &file_e)
{

	vector<pair<int,int>> matches;
	vector<KeyPoint> KP;
	
	visionPtr->Analyze(raw_frame, KP, matches);
	visualizerPtr->show(raw_frame, KP, matches, frame_name);
	
	saveMatrix(visionPtr->Homography_Matrix, file_h);
	saveMatrix(visionPtr->Essential_Matrix, file_e);
	
				
	saveVOFile(visionPtr->nmatches12, visionPtr->nmatches21, visionPtr->nmatchesCCM,
				visionPtr->T_cam_0, visionPtr->rvec_0, 
				visionPtr->T_cam_1, visionPtr->rvec_1,
				visionPtr->T_cam_2, visionPtr->rvec_2,
				visionPtr->T_cam_3, visionPtr->rvec_3,
				file_vo);

}

void System::run(Mat &raw_frame, string &frame_name, double &gpsT,
					double &lat, double &lng, double &alt, 
					double &roll, double &pitch, double &heading, 
					ofstream &file_GT, ofstream &file_vo, ofstream &file_h, ofstream &file_e)
{
	vector<pair<int,int>> matches;
	vector<KeyPoint> KP;
	
	visionPtr->Analyze(raw_frame, KP, matches);
	visualizerPtr->show(raw_frame, KP, matches, frame_name);
	absPosePtr->calcPose(lat, lng, alt, roll, pitch, heading);
	
	saveMatrix(visionPtr->Homography_Matrix, file_h);
	saveMatrix(visionPtr->Essential_Matrix, file_e);
	
	saveTraj(absPosePtr->T_abs, file_GT);
	
	saveVO_CIVIT(gpsT, lat, lng, alt, roll, pitch, heading, 
				visionPtr->nmatches12, visionPtr->nmatches21, visionPtr->nmatchesCCM,
				visionPtr->T_cam_0, visionPtr->rvec_0, 
				visionPtr->T_cam_1, visionPtr->rvec_1,
				visionPtr->T_cam_2, visionPtr->rvec_2,
				visionPtr->T_cam_3, visionPtr->rvec_3,
				file_vo);				
				
}

void System::saveTraj(Mat &T, ofstream &file_)
{
	file_	<< setprecision(8)	<< T.at<float>(0,0) 	<< ","
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

void System::saveMatrix(Mat &Matrix, ofstream &file_)
{
	file_	<< setprecision(8)	<< Matrix.at<double>(0,0) 	<< ","
			<< setprecision(8)	<< Matrix.at<double>(0,1) 	<< ","
			<< setprecision(8)	<< Matrix.at<double>(0,2)		<< ","
			
			<< setprecision(8)	<< Matrix.at<double>(1,0) 	<< ","
			<< setprecision(8)	<< Matrix.at<double>(1,1) 	<< ","
			<< setprecision(8)	<< Matrix.at<double>(1,2) 	<< ","
			
			<< setprecision(8)	<< Matrix.at<double>(2,0) 	<< ","
			<< setprecision(8)	<< Matrix.at<double>(2,1) 	<< ","
			<< setprecision(8)	<< Matrix.at<double>(2,2) 	<< endl;
}

void System::saveVOFile(int &m12, int &m21, int &mCCM, 	Mat &Tc_0, Mat &rvec_0,
														Mat &Tc_1, Mat &rvec_1,
														Mat &Tc_2, Mat &rvec_2,
														Mat &Tc_3, Mat &rvec_3,
														ofstream &file_)
{	
	file_	<< 					m12		 				<< ","
			<< 					m21 					<< ","
			<< 					mCCM 					<< ","
			
			<< setprecision(8)	<< rvec_0.at<double>(0,0) 	<< ","
			<< setprecision(8)	<< rvec_0.at<double>(1,0) 	<< ","
			<< setprecision(8)	<< rvec_0.at<double>(2,0) 	<< ","
			
			<< setprecision(8)	<< Tc_0.at<float>(0,0) 	<< ","
			<< setprecision(8)	<< Tc_0.at<float>(0,1) 	<< ","
			<< setprecision(8)	<< Tc_0.at<float>(0,2)	<< ","
			<< setprecision(8)	<< Tc_0.at<float>(0,3) 	<< ","
			
			<< setprecision(8)	<< Tc_0.at<float>(1,0) 	<< ","
			<< setprecision(8)	<< Tc_0.at<float>(1,1) 	<< ","
			<< setprecision(8)	<< Tc_0.at<float>(1,2) 	<< ","
			<< setprecision(8)	<< Tc_0.at<float>(1,3) 	<< ","
			
			<< setprecision(8)	<< Tc_0.at<float>(2,0) 	<< ","
			<< setprecision(8)	<< Tc_0.at<float>(2,1) 	<< ","
			<< setprecision(8)	<< Tc_0.at<float>(2,2) 	<< ","
			<< setprecision(8)	<< Tc_0.at<float>(2,3) 	<< ","
					
			<< setprecision(8)	<< rvec_1.at<double>(0,0) 	<< ","
			<< setprecision(8)	<< rvec_1.at<double>(1,0) 	<< ","
			<< setprecision(8)	<< rvec_1.at<double>(2,0) 	<< ","

			<< setprecision(8)	<< Tc_1.at<float>(0,0) 	<< ","
			<< setprecision(8)	<< Tc_1.at<float>(0,1) 	<< ","
			<< setprecision(8)	<< Tc_1.at<float>(0,2)	<< ","
			<< setprecision(8)	<< Tc_1.at<float>(0,3) 	<< ","
			
			<< setprecision(8)	<< Tc_1.at<float>(1,0) 	<< ","
			<< setprecision(8)	<< Tc_1.at<float>(1,1) 	<< ","
			<< setprecision(8)	<< Tc_1.at<float>(1,2) 	<< ","
			<< setprecision(8)	<< Tc_1.at<float>(1,3) 	<< ","
			
			<< setprecision(8)	<< Tc_1.at<float>(2,0) 	<< ","
			<< setprecision(8)	<< Tc_1.at<float>(2,1) 	<< ","
			<< setprecision(8)	<< Tc_1.at<float>(2,2) 	<< ","
			<< setprecision(8)	<< Tc_1.at<float>(2,3) 	<< ","
			
			<< setprecision(8)	<< rvec_2.at<double>(0,0) 	<< ","
			<< setprecision(8)	<< rvec_2.at<double>(1,0) 	<< ","
			<< setprecision(8)	<< rvec_2.at<double>(2,0) 	<< ","

			<< setprecision(8)	<< Tc_2.at<float>(0,0) 	<< ","
			<< setprecision(8)	<< Tc_2.at<float>(0,1) 	<< ","
			<< setprecision(8)	<< Tc_2.at<float>(0,2)	<< ","
			<< setprecision(8)	<< Tc_2.at<float>(0,3) 	<< ","
			
			<< setprecision(8)	<< Tc_2.at<float>(1,0) 	<< ","
			<< setprecision(8)	<< Tc_2.at<float>(1,1) 	<< ","
			<< setprecision(8)	<< Tc_2.at<float>(1,2) 	<< ","
			<< setprecision(8)	<< Tc_2.at<float>(1,3) 	<< ","
			
			<< setprecision(8)	<< Tc_2.at<float>(2,0) 	<< ","
			<< setprecision(8)	<< Tc_2.at<float>(2,1) 	<< ","
			<< setprecision(8)	<< Tc_2.at<float>(2,2) 	<< ","
			<< setprecision(8)	<< Tc_2.at<float>(2,3) 	<< ","
			
			<< setprecision(8)	<< rvec_3.at<double>(0,0) 	<< ","
			<< setprecision(8)	<< rvec_3.at<double>(1,0) 	<< ","
			<< setprecision(8)	<< rvec_3.at<double>(2,0) 	<< ","

			<< setprecision(8)	<< Tc_3.at<float>(0,0) 	<< ","
			<< setprecision(8)	<< Tc_3.at<float>(0,1) 	<< ","
			<< setprecision(8)	<< Tc_3.at<float>(0,2)	<< ","
			<< setprecision(8)	<< Tc_3.at<float>(0,3) 	<< ","
			
			<< setprecision(8)	<< Tc_3.at<float>(1,0) 	<< ","
			<< setprecision(8)	<< Tc_3.at<float>(1,1) 	<< ","
			<< setprecision(8)	<< Tc_3.at<float>(1,2) 	<< ","
			<< setprecision(8)	<< Tc_3.at<float>(1,3) 	<< ","
			
			<< setprecision(8)	<< Tc_3.at<float>(2,0) 	<< ","
			<< setprecision(8)	<< Tc_3.at<float>(2,1) 	<< ","
			<< setprecision(8)	<< Tc_3.at<float>(2,2) 	<< ","
			<< setprecision(8)	<< Tc_3.at<float>(2,3) 	<< endl;
}

void System::saveVO_CIVIT(double &gpsT, double &lat, double &lng, double &alt, 
							double &roll, double &pitch, double &heading, 
							int &m12, int &m21, int &mCCM,	Mat &Tc_0, Mat &rvec_0,
															Mat &Tc_1, Mat &rvec_1,
															Mat &Tc_2, Mat &rvec_2,
															Mat &Tc_3, Mat &rvec_3,
															ofstream &file_)
{
	file_		<< setprecision(15)	<< 	gpsT 					<< ","
				<< setprecision(15)	<< 	lat 					<< ","
				<< setprecision(15)	<< 	lng 					<< ","
				<< setprecision(15)	<< 	alt 					<< ","
				<< setprecision(15)	<< 	roll 					<< ","
				<< setprecision(15)	<< 	pitch 					<< ","
				<< setprecision(15)	<< 	heading 				<< ","
				
				<< 						m12 					<< ","
				<< 						m21 					<< ","
				<< 						mCCM 					<< ","

				<< setprecision(8)	<< rvec_0.at<double>(0,0) 	<< ","
				<< setprecision(8)	<< rvec_0.at<double>(1,0) 	<< ","
				<< setprecision(8)	<< rvec_0.at<double>(2,0) 	<< ","
				
				<< setprecision(8)	<< Tc_0.at<float>(0,0) 	<< ","
				<< setprecision(8)	<< Tc_0.at<float>(0,1) 	<< ","
				<< setprecision(8)	<< Tc_0.at<float>(0,2)	<< ","
				<< setprecision(8)	<< Tc_0.at<float>(0,3) 	<< ","
			
				<< setprecision(8)	<< Tc_0.at<float>(1,0) 	<< ","
				<< setprecision(8)	<< Tc_0.at<float>(1,1) 	<< ","
				<< setprecision(8)	<< Tc_0.at<float>(1,2) 	<< ","
				<< setprecision(8)	<< Tc_0.at<float>(1,3) 	<< ","
			
				<< setprecision(8)	<< Tc_0.at<float>(2,0) 	<< ","
				<< setprecision(8)	<< Tc_0.at<float>(2,1) 	<< ","
				<< setprecision(8)	<< Tc_0.at<float>(2,2) 	<< ","
				<< setprecision(8)	<< Tc_0.at<float>(2,3) 	<< ","
					
				<< setprecision(8)	<< rvec_1.at<double>(0,0) 	<< ","
				<< setprecision(8)	<< rvec_1.at<double>(1,0) 	<< ","
				<< setprecision(8)	<< rvec_1.at<double>(2,0) 	<< ","

				<< setprecision(8)	<< Tc_1.at<float>(0,0) 	<< ","
				<< setprecision(8)	<< Tc_1.at<float>(0,1) 	<< ","
				<< setprecision(8)	<< Tc_1.at<float>(0,2)	<< ","
				<< setprecision(8)	<< Tc_1.at<float>(0,3) 	<< ","
			
				<< setprecision(8)	<< Tc_1.at<float>(1,0) 	<< ","
				<< setprecision(8)	<< Tc_1.at<float>(1,1) 	<< ","
				<< setprecision(8)	<< Tc_1.at<float>(1,2) 	<< ","
				<< setprecision(8)	<< Tc_1.at<float>(1,3) 	<< ","
			
				<< setprecision(8)	<< Tc_1.at<float>(2,0) 	<< ","
				<< setprecision(8)	<< Tc_1.at<float>(2,1) 	<< ","
				<< setprecision(8)	<< Tc_1.at<float>(2,2) 	<< ","
				<< setprecision(8)	<< Tc_1.at<float>(2,3) 	<< ","
			
				<< setprecision(8)	<< rvec_2.at<double>(0,0) 	<< ","
				<< setprecision(8)	<< rvec_2.at<double>(1,0) 	<< ","
				<< setprecision(8)	<< rvec_2.at<double>(2,0) 	<< ","

				<< setprecision(8)	<< Tc_2.at<float>(0,0) 	<< ","
				<< setprecision(8)	<< Tc_2.at<float>(0,1) 	<< ","
				<< setprecision(8)	<< Tc_2.at<float>(0,2)	<< ","
				<< setprecision(8)	<< Tc_2.at<float>(0,3) 	<< ","
			
				<< setprecision(8)	<< Tc_2.at<float>(1,0) 	<< ","
				<< setprecision(8)	<< Tc_2.at<float>(1,1) 	<< ","
				<< setprecision(8)	<< Tc_2.at<float>(1,2) 	<< ","
				<< setprecision(8)	<< Tc_2.at<float>(1,3) 	<< ","
			
				<< setprecision(8)	<< Tc_2.at<float>(2,0) 	<< ","
				<< setprecision(8)	<< Tc_2.at<float>(2,1) 	<< ","
				<< setprecision(8)	<< Tc_2.at<float>(2,2) 	<< ","
				<< setprecision(8)	<< Tc_2.at<float>(2,3) 	<< ","
			
				<< setprecision(8)	<< rvec_3.at<double>(0,0) 	<< ","
				<< setprecision(8)	<< rvec_3.at<double>(1,0) 	<< ","
				<< setprecision(8)	<< rvec_3.at<double>(2,0) 	<< ","

				<< setprecision(8)	<< Tc_3.at<float>(0,0) 	<< ","
				<< setprecision(8)	<< Tc_3.at<float>(0,1) 	<< ","
				<< setprecision(8)	<< Tc_3.at<float>(0,2)	<< ","
				<< setprecision(8)	<< Tc_3.at<float>(0,3) 	<< ","
			
				<< setprecision(8)	<< Tc_3.at<float>(1,0) 	<< ","
				<< setprecision(8)	<< Tc_3.at<float>(1,1) 	<< ","
				<< setprecision(8)	<< Tc_3.at<float>(1,2) 	<< ","
				<< setprecision(8)	<< Tc_3.at<float>(1,3) 	<< ","
				
				<< setprecision(8)	<< Tc_3.at<float>(2,0) 	<< ","
				<< setprecision(8)	<< Tc_3.at<float>(2,1) 	<< ","
				<< setprecision(8)	<< Tc_3.at<float>(2,2) 	<< ","
				<< setprecision(8)	<< Tc_3.at<float>(2,3) 	<< endl;
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

