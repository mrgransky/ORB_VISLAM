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

#include "AbsolutePose.h"
#include "Visualizer.h"
#include "Vision.h"

namespace ORB_VISLAM
{
	class System
	{
		public:
			System( const std::string &settingFilePath, float scale,
						int win_sz, float ssd_th, float ssd_ratio_th, size_t minFeat);
			
			System( const std::string &settingFilePath, float scale,
					int win_sz, float ssd_th, float ssd_ratio_th, size_t minFeat,
					double &ref_lat, double &ref_lng, double &ref_alt);
					
			~System();
			
			void run(cv::Mat &frame, std::string &frame_name, 
						std::ofstream &file_vo, std::ofstream &file_h, std::ofstream &file_e);
			
			void run(cv::Mat &frame, std::string &frame_name, double &gpsT,
						double &lat, double &lng, double &alt, 
						double &roll, double &pitch, double &heading, 
						std::ofstream &file_GT, 
						std::ofstream &file_vo, std::ofstream &file_h, std::ofstream &file_e);
		
			void shutdown();
			
			//std::vector<int> nmatchesCCM, nmatches12, nmatches21;
		private:
			std::vector<cv::Mat> R_prev, t_prev, rvec_prev;
			
			AbsolutePose* absPosePtr;
			
			void saveTraj(cv::Mat &T, std::ofstream &file_);
			
			void saveMatrix(cv::Mat &Matrix, std::ofstream &file_);
			void saveVOFile(int &m12, int &m21, int &mCCM,
							cv::Mat &Tc_0, cv::Mat &rvec_0, 
							cv::Mat &Tc_1, cv::Mat &rvec_1,
							cv::Mat &Tc_2, cv::Mat &rvec_2,
							cv::Mat &Tc_3, cv::Mat &rvec_3,
							std::ofstream &file_);

			void saveVO_CIVIT(double &gpsT, double &lat, double &lng, double &alt, 
								double &roll, double &pitch, double &heading,
								int &m12, int &m21, int &mCCM,
								cv::Mat &Tc_0, cv::Mat &rvec_0, 
								cv::Mat &Tc_1, cv::Mat &rvec_1,
								cv::Mat &Tc_2, cv::Mat &rvec_2,
								cv::Mat &Tc_3, cv::Mat &rvec_3,	
								std::ofstream &file_);
							
			
			Visualizer* visualizerPtr;
			Vision*		visionPtr;
			
			clock_t tStart, tEnd;
			bool frame_avl;
			double runTime;
    		
			std::thread* visThread;
			std::thread* absPoseThread;
	};

}// namespace ORB_VISLAM

#endif // SYSTEM_H
