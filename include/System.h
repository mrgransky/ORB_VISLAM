#ifndef SYSTEM_H
#define SYSTEM_H

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
#include <time.h>
#include "AbsolutePose.h"
//#include "Viewer.h"


namespace ORB_VISLAM
{
	class System
	{
		public:
			System(const std::string &settingFilePath, 
					double &ref_lat, double &ref_lng, double &ref_alt);
			cv::Mat run_abs_pose(double &lat, double &lng, double &alt, 
							double &roll, double &pitch, double &heading);
		private:
			AbsolutePose absPose;
			clock_t tStart, tEnd;
			double runTime;
    		cv::Mat mK;
    		cv::Mat mDistCoef;
			
	};

}// namespace ORB_VISLAM

#endif // SYSTEM_H
