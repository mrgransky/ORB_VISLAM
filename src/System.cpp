#include "System.h"

using namespace std;
using namespace cv;

using namespace pcl;
using namespace pcl::io;
using namespace pcl::visualization;
namespace ORB_VISLAM
{

System::System(	const string &settingFilePath, float frameDownScale,
						int win_sz, float ssd_th, 
						float ssd_ratio_th, size_t minFeat, float minScale)
{
	cout << "" << endl;
	cout << "#########################################################################" << endl;
	cout << "\t\t\t\tSYSTEM"															<< endl;
	cout << "#########################################################################" << endl;
	
	frame_avl = true;
	
	
	absPosePtr 		= new AbsolutePose();
	// init vision class:
	visionPtr		= new Vision(settingFilePath, win_sz, ssd_th, ssd_ratio_th, minFeat, minScale);
	// initialize visualizer class
	
	visualizerPtr 	= new Visualizer(visionPtr->IMG_, 	absPosePtr->T_abs,
														visionPtr->T_cam_E,
														visionPtr->T_cam_0, 
														visionPtr->T_cam_1, 
														visionPtr->T_cam_2, 
														visionPtr->T_cam_3, 
														visionPtr->fps, 
														frameDownScale, frame_avl,
														visionPtr->cloud);
	// run visualizer thread
	visThread 		= new thread(&Visualizer::run, visualizerPtr);
}

/* ###################### CIVIT DATASET constructor ###################### */
System::System(	const string &settingFilePath, float frameDownScale,
				int win_sz, float ssd_th, float ssd_ratio_th, size_t minFeat, float minScale,
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
	// initialize absPose class:
	absPosePtr 		= new AbsolutePose(ref_lat, ref_lng, ref_alt);
	
	// init vision class:
	visionPtr		= new Vision(settingFilePath, win_sz, ssd_th, ssd_ratio_th, minFeat, minScale);

	// initialize visualizer class
	visualizerPtr 	= new Visualizer(visionPtr->IMG_, 	absPosePtr->T_abs, 
														visionPtr->T_cam_E,
														visionPtr->T_cam_0, 
														visionPtr->T_cam_1, 
														visionPtr->T_cam_2, 
														visionPtr->T_cam_3,
														visionPtr->fps, 
														frameDownScale, frame_avl,
														visionPtr->cloud);
	// run visualizer thread
	visThread 		= new thread(&Visualizer::run, visualizerPtr);
}

System::~System()
{
	visThread->join();
}

void System::run(Mat &raw_frame, string &frame_name, 
					ofstream &file_vo,
					ofstream &file_gt,
					ofstream &file_rvec_abs,
					ofstream &file_vo_loc,
					Mat &T_GT, double &scale_GT)
{
	vector<pair<int,int>> matches;
	vector<KeyPoint> KP;
	vector<Point3f> map_points;
	visionPtr->sc = scale_GT;
	//visionPtr->sc = 1.0;
	
	visionPtr->Analyze(raw_frame, KP, matches, map_points);
	visualizerPtr->show(raw_frame, KP, matches, map_points, frame_name);
	
	absPosePtr->set(T_GT);
	saveMatrix(T_GT, file_gt);
	saveMatrix(absPosePtr->rvec_abs, file_rvec_abs);
	
	saveVOFile(	visionPtr->T_loc_0, visionPtr->rvec_loc_0, 
				visionPtr->T_loc_1, visionPtr->rvec_loc_1,
				visionPtr->T_loc_2, visionPtr->rvec_loc_2,
				visionPtr->T_loc_3, visionPtr->rvec_loc_3,
				visionPtr->T_loc_E, visionPtr->rvec_loc_E,
				file_vo_loc);
	
	saveVOFile(	visionPtr->T_cam_0, visionPtr->rvec_0, 
				visionPtr->T_cam_1, visionPtr->rvec_1,
				visionPtr->T_cam_2, visionPtr->rvec_2,
				visionPtr->T_cam_3, visionPtr->rvec_3,
				visionPtr->T_cam_E, visionPtr->rvec_E,
				file_vo);
}

void System::run(Mat &raw_frame, string &frame_name, double &gpsT,
					double &lat, double &lng, double &alt, 
					double &roll, double &pitch, double &heading, 
					ofstream &file_vo, 
					ofstream &file_gt, 
					ofstream &file_rvec_abs,
					ofstream &file_vo_loc)
{
	vector<pair<int,int>> matches;
	vector<KeyPoint> KP;
	vector<Point3f> map_points;
	
	visionPtr->Analyze(raw_frame, KP, matches, map_points);
	visualizerPtr->show(raw_frame, KP, matches, visionPtr->map_3D, frame_name);
	absPosePtr->calcPose(lat, lng, alt, roll, pitch, heading);
	
	visionPtr->sc = absPosePtr->AbsScale;
	//visionPtr->sc = 1.0;
	
	
	saveMatrix(absPosePtr->T_abs, file_gt);
	saveMatrix(absPosePtr->rvec_abs, file_rvec_abs);
	
	saveVOFile(	visionPtr->T_loc_0, visionPtr->rvec_loc_0, 
				visionPtr->T_loc_1, visionPtr->rvec_loc_1,
				visionPtr->T_loc_2, visionPtr->rvec_loc_2,
				visionPtr->T_loc_3, visionPtr->rvec_loc_3,
				visionPtr->T_loc_E, visionPtr->rvec_loc_E,
				file_vo_loc);
	
	saveVOFile(	visionPtr->T_cam_0, visionPtr->rvec_0, 
				visionPtr->T_cam_1, visionPtr->rvec_1,
				visionPtr->T_cam_2, visionPtr->rvec_2,
				visionPtr->T_cam_3, visionPtr->rvec_3,
				visionPtr->T_cam_E, visionPtr->rvec_E,
				file_vo);
}

void System::save3Dpoints(ofstream &file_)
{
	for(size_t i = 0; i < visionPtr->map_3D.size(); i++)
	{
		file_	<< visionPtr->map_3D[i].x 	<< ","
				<< visionPtr->map_3D[i].y 	<< ","
				<< visionPtr->map_3D[i].z 	<< endl;
	}
}

void System::savePointCloud()
{
	for(size_t i = 0; i < visionPtr->map_3D.size(); i++)
	{
		uint8_t red 	= rand() * 255;
		uint8_t green 	= rand() * 255;
		uint8_t blue 	= rand() * 255;
		pcl::PointXYZRGB p;
		
		p.x = visionPtr->map_3D[i].x;
		p.y = visionPtr->map_3D[i].y;
		p.z = visionPtr->map_3D[i].z;
		
		p.r = red;
		p.g = green;
		p.b = blue;
		visionPtr->cloud->push_back(p);
	}

	cout 	<< "cloud [w,h,sz] =\t" << visionPtr->cloud->width 
			<< " , "				<< visionPtr->cloud->height 
			<< " , " 				<< visionPtr->cloud->size()
			<< endl;
			
	if (visionPtr->cloud->size() != 0)
	{
		savePCDFileASCII("Point_Cloud.pcd", *visionPtr->cloud);
	}
	else
	{
		cout << "\nNO Point Cloud saved...\n" << endl;
	}
}

void System::saveMatrix(Mat &Matrix, ofstream &file_)
{
	for(int r = 0; r < Matrix.rows; r++)
	{
		for(int c = 0; c < Matrix.cols; c++)
		{
			if (r == Matrix.rows-1 && c == Matrix.cols-1)
			{
				file_	<< setprecision(8)	<< Matrix.at<float>(r,c) << endl;
			}
			else 
			{
				file_	<< setprecision(8)	<< Matrix.at<float>(r,c) << ",";			
			}
		}
	}
}

void System::saveVOFile(Mat &Tc_0, Mat &rvec_0, 
						Mat &Tc_1, Mat &rvec_1, 
						Mat &Tc_2, Mat &rvec_2, 
						Mat &Tc_3, Mat &rvec_3,
						Mat &Tc_E, Mat &rvec_E,						
						ofstream &file_)
{	
	file_	<< setprecision(8)	<< rvec_0.at<double>(0,0) 	<< ","
			<< setprecision(8)	<< rvec_0.at<double>(1,0) 	<< ","
			<< setprecision(8)	<< rvec_0.at<double>(2,0) 	<< ","
			
			<< setprecision(8)	<< Tc_0.at<float>(0,0) 		<< ","
			<< setprecision(8)	<< Tc_0.at<float>(0,1) 		<< ","
			<< setprecision(8)	<< Tc_0.at<float>(0,2)		<< ","
			<< setprecision(8)	<< Tc_0.at<float>(0,3) 		<< ","
			
			<< setprecision(8)	<< Tc_0.at<float>(1,0) 		<< ","
			<< setprecision(8)	<< Tc_0.at<float>(1,1) 		<< ","
			<< setprecision(8)	<< Tc_0.at<float>(1,2) 		<< ","
			<< setprecision(8)	<< Tc_0.at<float>(1,3) 		<< ","
			
			<< setprecision(8)	<< Tc_0.at<float>(2,0) 		<< ","
			<< setprecision(8)	<< Tc_0.at<float>(2,1) 		<< ","
			<< setprecision(8)	<< Tc_0.at<float>(2,2) 		<< ","
			<< setprecision(8)	<< Tc_0.at<float>(2,3) 		<< ","
					
			<< setprecision(8)	<< rvec_1.at<double>(0,0) 	<< ","
			<< setprecision(8)	<< rvec_1.at<double>(1,0) 	<< ","
			<< setprecision(8)	<< rvec_1.at<double>(2,0) 	<< ","

			<< setprecision(8)	<< Tc_1.at<float>(0,0) 		<< ","
			<< setprecision(8)	<< Tc_1.at<float>(0,1) 		<< ","
			<< setprecision(8)	<< Tc_1.at<float>(0,2)		<< ","
			<< setprecision(8)	<< Tc_1.at<float>(0,3) 		<< ","
			
			<< setprecision(8)	<< Tc_1.at<float>(1,0) 		<< ","
			<< setprecision(8)	<< Tc_1.at<float>(1,1) 		<< ","
			<< setprecision(8)	<< Tc_1.at<float>(1,2) 		<< ","
			<< setprecision(8)	<< Tc_1.at<float>(1,3) 		<< ","
			
			<< setprecision(8)	<< Tc_1.at<float>(2,0) 		<< ","
			<< setprecision(8)	<< Tc_1.at<float>(2,1) 		<< ","
			<< setprecision(8)	<< Tc_1.at<float>(2,2) 		<< ","
			<< setprecision(8)	<< Tc_1.at<float>(2,3) 		<< ","
			
			<< setprecision(8)	<< rvec_2.at<double>(0,0) 	<< ","
			<< setprecision(8)	<< rvec_2.at<double>(1,0) 	<< ","
			<< setprecision(8)	<< rvec_2.at<double>(2,0) 	<< ","

			<< setprecision(8)	<< Tc_2.at<float>(0,0) 		<< ","
			<< setprecision(8)	<< Tc_2.at<float>(0,1) 		<< ","
			<< setprecision(8)	<< Tc_2.at<float>(0,2)		<< ","
			<< setprecision(8)	<< Tc_2.at<float>(0,3) 		<< ","
			
			<< setprecision(8)	<< Tc_2.at<float>(1,0) 		<< ","
			<< setprecision(8)	<< Tc_2.at<float>(1,1) 		<< ","
			<< setprecision(8)	<< Tc_2.at<float>(1,2) 		<< ","
			<< setprecision(8)	<< Tc_2.at<float>(1,3) 		<< ","
			
			<< setprecision(8)	<< Tc_2.at<float>(2,0) 		<< ","
			<< setprecision(8)	<< Tc_2.at<float>(2,1) 		<< ","
			<< setprecision(8)	<< Tc_2.at<float>(2,2) 		<< ","
			<< setprecision(8)	<< Tc_2.at<float>(2,3) 		<< ","
			
			<< setprecision(8)	<< rvec_3.at<double>(0,0) 	<< ","
			<< setprecision(8)	<< rvec_3.at<double>(1,0) 	<< ","
			<< setprecision(8)	<< rvec_3.at<double>(2,0) 	<< ","

			<< setprecision(8)	<< Tc_3.at<float>(0,0) 		<< ","
			<< setprecision(8)	<< Tc_3.at<float>(0,1) 		<< ","
			<< setprecision(8)	<< Tc_3.at<float>(0,2)		<< ","
			<< setprecision(8)	<< Tc_3.at<float>(0,3) 		<< ","
			
			<< setprecision(8)	<< Tc_3.at<float>(1,0) 		<< ","
			<< setprecision(8)	<< Tc_3.at<float>(1,1) 		<< ","
			<< setprecision(8)	<< Tc_3.at<float>(1,2) 		<< ","
			<< setprecision(8)	<< Tc_3.at<float>(1,3) 		<< ","
			
			<< setprecision(8)	<< Tc_3.at<float>(2,0) 		<< ","
			<< setprecision(8)	<< Tc_3.at<float>(2,1) 		<< ","
			<< setprecision(8)	<< Tc_3.at<float>(2,2) 		<< ","
			<< setprecision(8)	<< Tc_3.at<float>(2,3) 		<< ","
			
			
			<< setprecision(8)	<< rvec_E.at<double>(0,0) 	<< ","
			<< setprecision(8)	<< rvec_E.at<double>(1,0) 	<< ","
			<< setprecision(8)	<< rvec_E.at<double>(2,0) 	<< ","

			<< setprecision(8)	<< Tc_E.at<float>(0,0) 		<< ","
			<< setprecision(8)	<< Tc_E.at<float>(0,1) 		<< ","
			<< setprecision(8)	<< Tc_E.at<float>(0,2)		<< ","
			<< setprecision(8)	<< Tc_E.at<float>(0,3) 		<< ","
			
			<< setprecision(8)	<< Tc_E.at<float>(1,0) 		<< ","
			<< setprecision(8)	<< Tc_E.at<float>(1,1) 		<< ","
			<< setprecision(8)	<< Tc_E.at<float>(1,2) 		<< ","
			<< setprecision(8)	<< Tc_E.at<float>(1,3) 		<< ","
			
			<< setprecision(8)	<< Tc_E.at<float>(2,0) 		<< ","
			<< setprecision(8)	<< Tc_E.at<float>(2,1) 		<< ","
			<< setprecision(8)	<< Tc_E.at<float>(2,2) 		<< ","
			<< setprecision(8)	<< Tc_E.at<float>(2,3) 		<< endl;
}

void System::shutdown()
{
	frame_avl = false; 	// if activated in main.cpp:
						// TODO: deteriorate the while loop in visualizer class,
						//-->>>>> (last frame invisible)
	visualizerPtr->hasFrame = false;
	cout << "system is shutting down!" << endl;
}

/*void System::saveRMSE(Mat &Tc, Mat &Tgt, )
{
	vector<double> sq_res, sum;
	
	for(size_t j = 0; j < Tc.rows; j++)
	{
		sq_res.push_back(
		(Tgt.at<double>(j,3) - Tc.at<double>(j,3)) * (Tgt.at<double>(j,3) - Tc.at<double>(j,3))
						);
		
		sum = sum + sq_res[j];
	}
	
		
	double xGT 	= Tgt.at<double>(0,3);
	
	sq_RMSE = 
}*/

}
