#ifndef SYSTEM_H
#define SYSTEM_H

#include <pangolin/pangolin.h>

#include <limits>
#include <fstream>
#include <vector>
#include <string>
#include <iostream>
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <stdio.h>
#include <stdlib.h>
#include <thread>
#include <time.h>

#include <pcl/common/common_headers.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/visualization/pcl_visualizer.h>

#include "AbsolutePose.h"
#include "Visualizer.h"
#include "Vision.h"
#include "boost/make_shared.hpp"
namespace ORB_VISLAM
{
	class System
	{
		public:
			System( const std::string &settingFilePath, float frameDownScale,
						int win_sz, float ssd_th, float ssd_ratio_th, 
						size_t minFeat, float minScale, float distTo3DPts);
			
			System( const std::string &settingFilePath, float frameDownScale,
					int win_sz, float ssd_th, float ssd_ratio_th, 
					size_t minFeat, float minScale, float distTo3DPts,
					double &ref_lat, double &ref_lng, double &ref_alt);
					
			~System();
			
			void run(cv::Mat &frame, std::string &frame_name, 
						std::ofstream &file_vo, 
						std::ofstream &file_gt, 
						std::ofstream &file_rvec_abs,
						std::ofstream &file_vo_loc,
						cv::Mat &T_GT,
						float &scale_GT);
			
			void run(cv::Mat &frame, std::string &frame_name, double &gpsT,
						double &lat, double &lng, double &alt, 
						double &roll, double &pitch, double &heading, 
						std::ofstream &file_vo,
						std::ofstream &file_gt,
						std::ofstream &file_rvec_abs,
						std::ofstream &file_vo_loc);
		
			void shutdown();
			void savePointCloud(std::string fname_);
			//std::vector<int> nmatchesCCM, nmatches12, nmatches21;
		private:
			std::vector<cv::Mat> R_prev, t_prev, rvec_prev;
			AbsolutePose* absPosePtr;
			
			void saveMatrix(cv::Mat &Matrix, std::ofstream &file_);
			
			void saveMatrix(cv::Mat &Matrix, float &scale, float &frontOPCV, 
							float &frontOWN, float &rE, std::ofstream &file_);
			
			void saveVOFile(cv::Mat &Tc_E, cv::Mat &rvec_E, std::ofstream &file_);
			
			Visualizer* visualizerPtr;
			Vision*		visionPtr;
			
			clock_t tStart, tEnd;
			bool frame_avl;
			double runTime;
    		
    		
			void save3Dpoints(std::vector<cv::Mat> &p3ds, std::ofstream &file_);
			//std::vector<cv::Mat> map_points;

			std::thread* visThread;
			std::thread* absPoseThread;
	};

}// namespace ORB_VISLAM

#endif // SYSTEM_H
