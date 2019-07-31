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
						float ssd_ratio_th, size_t minFeat, 
						float minScale, float distTo3DPts)
{
	cout << "" << endl;
	cout << "#########################################################################" << endl;
	cout << "\t\t\t\tSYSTEM"															<< endl;
	cout << "#########################################################################" << endl;
	
	frame_avl = true;
	
	absPosePtr 		= new AbsolutePose();
	// init vision class:
	visionPtr		= new Vision(settingFilePath, win_sz, ssd_th, ssd_ratio_th, 
									minFeat, minScale, distTo3DPts, false);
	// initialize visualizer class
	
	visualizerPtr 	= new Visualizer(visionPtr->IMG_, 	absPosePtr->T_abs,
														visionPtr->T_cam_E,
														visionPtr->fps, 
														frameDownScale, frame_avl,
														visionPtr->cloud);
	// run visualizer thread
	visThread 		= new thread(&Visualizer::run, visualizerPtr);
}

/* ###################### CIVIT DATASET constructor ###################### */
System::System(	const string &settingFilePath, float frameDownScale,
				int win_sz, float ssd_th, float ssd_ratio_th, 
				size_t minFeat, float minScale, float distTo3DPts,
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
	visionPtr		= new Vision(settingFilePath, win_sz, ssd_th, ssd_ratio_th, 
									minFeat, minScale, distTo3DPts, false);
	// initialize visualizer class
	visualizerPtr 	= new Visualizer(visionPtr->IMG_, 	absPosePtr->T_abs, 
														visionPtr->T_cam_E,
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
					ofstream &file_util,
					ofstream &file_vo_loc,
					Mat &T_GT, float &scale_GT)
{
	vector<KeyPoint> KP;
	vector<pair<int,int>> matches;
	visionPtr->sc = scale_GT;
	//visionPtr->sc = 1.0;	

	visionPtr->Analyze(raw_frame, KP, matches);
	visualizerPtr->show(raw_frame, KP, matches, 
						visionPtr->visionMap, 
						visionPtr->vPt2D_rep,
						visionPtr->vPt2D_measured,
						frame_name);
	
	absPosePtr->set(T_GT);
	saveMatrix(T_GT, file_gt);
	saveMatrix(absPosePtr->rvec_abs, scale_GT,
				visionPtr->front3DPtsOPCV,
				visionPtr->front3DPtsOWN,
				visionPtr->repErr,
				file_util);
				
	saveVOFile(visionPtr->T_loc_E, visionPtr->rvec_loc_E,
				file_vo_loc);
	
	saveVOFile(visionPtr->T_cam_E, 
				visionPtr->rvec_E,
				file_vo);
}

void System::run(Mat &raw_frame, string &frame_name, double &gpsT,
					double &lat, double &lng, double &alt, 
					double &roll, double &pitch, double &heading, 
					ofstream &file_vo, 
					ofstream &file_gt, 
					ofstream &file_util,
					ofstream &file_vo_loc)
{
	vector<KeyPoint> KP;	
	vector<pair<int,int>> matches;
	
	visionPtr->Analyze(raw_frame, KP, matches);

	visualizerPtr->show(raw_frame, KP, matches, 
						visionPtr->visionMap, 
						visionPtr->vPt2D_rep,
						visionPtr->vPt2D_measured, 
						frame_name);
	
	absPosePtr->calcPose(lat, lng, alt, roll, pitch, heading);
	
	visionPtr->sc = absPosePtr->AbsScale;
	//visionPtr->sc = 1.0;
	
	saveMatrix(absPosePtr->T_abs, file_gt);
	saveMatrix(absPosePtr->rvec_abs, absPosePtr->AbsScale, 
				visionPtr->front3DPtsOPCV,
				visionPtr->front3DPtsOWN,
				visionPtr->repErr,
				file_util);
	
				
	saveVOFile(visionPtr->T_loc_E, visionPtr->rvec_loc_E, file_vo_loc);
	saveVOFile(visionPtr->T_cam_E, visionPtr->rvec_E, file_vo);
}

void System::save3Dpoints(vector<Mat> &p3ds, ofstream &file_)
{
	cout << "sMap sz =\t" << p3ds.size() << endl;
	for(size_t i = 0; i < p3ds.size(); i++)
	{
		file_	<< p3ds[i].at<float>(0) 	<< ","
				<< p3ds[i].at<float>(1) 	<< ","
				<< p3ds[i].at<float>(2) 	<< endl;
	}
}

void System::savePointCloud(string fname_)
{
	pcl::PointCloud<pcl::PointXYZ>::Ptr sCloud(new pcl::PointCloud<pcl::PointXYZ>());
	cout << "visualizerPtr->vMap.size() = \t" << visualizerPtr->vMap.size() << endl;
	
	for(size_t i = 0; i < visualizerPtr->vMap.size(); i++)
	{
		for(int j = 0; j < visualizerPtr->vMap[i].cols; j++)
		{
			pcl::PointXYZ p;

			p.x = visualizerPtr->vMap[i].at<float>(0,j);
			p.y = visualizerPtr->vMap[i].at<float>(1,j);
			p.z = visualizerPtr->vMap[i].at<float>(2,j);
		
			sCloud->push_back(p);
		}	
	}
	if (sCloud->size() != 0)
	{
		cout 	<< "cloud [w,h,sz] =\t" << sCloud->width 
				<< " , "				<< sCloud->height 
				<< " , " 				<< sCloud->size()
				<< endl;
		savePCDFileASCII(fname_, *sCloud);
	}
	else
	{
		cout << "\nNO Point Cloud saved!\n" << endl;
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

void System::saveMatrix(Mat &Matrix, float &scale, float &frontOPCV, 
						float &frontOWN, float &rE, ofstream &file_)
{
	for(int r = 0; r < Matrix.rows; r++)
	{
		for(int c = 0; c < Matrix.cols; c++)
		{
			if (r == Matrix.rows-1 && c == Matrix.cols-1)
			{
				file_	<< setprecision(8)	<< Matrix.at<float>(r,c) 
						<<","<< scale <<","<< frontOPCV <<","<< frontOWN <<","<<rE<< endl;
			}
			else 
			{
				file_	<< setprecision(8)	<< Matrix.at<float>(r,c) << ",";			
			}
		}
	}
}

void System::saveVOFile(Mat &Tc_E, Mat &rvec_E,						
						ofstream &file_)
{	
	file_	<< setprecision(8)	<< rvec_E.at<float>(0,0) 	<< ","
			<< setprecision(8)	<< rvec_E.at<float>(1,0) 	<< ","
			<< setprecision(8)	<< rvec_E.at<float>(2,0) 	<< ","

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
