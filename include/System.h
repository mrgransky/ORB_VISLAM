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
			System( const std::string &settingFilePath, 
					double &ref_lat, double &ref_lng, double &ref_alt);
					
			/*System( const std::string &settingFilePath, 
					double &ref_lat, double &ref_lng, double &ref_alt);*/
			~System();
			
			void run(cv::Mat &frame, double &lat, double &lng, double &alt, 
							double &roll, double &pitch, double &heading, std::ofstream &file_);
		
			void shutdown();
		private:
			//AbsolutePose init_absPose;
			AbsolutePose* absPosePtr;
			void saveTraj(cv::Mat T, std::ofstream &file_);
			//Visualizer init_visualizer;
			Visualizer* visualizerPtr;
			Vision*		visionPtr;
			
			//cv::Mat Tw = cv::Mat::eye(4, 4, CV_32F);
			clock_t tStart, tEnd;
			bool frame_avl;
			double runTime;
    		cv::Mat mK;
    		cv::Mat mDistCoef;
			std::thread* visThread;
			std::thread* absPoseThread;
	};

}// namespace ORB_VISLAM

#endif // SYSTEM_H
