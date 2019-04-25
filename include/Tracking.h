#ifndef TRACKING_H
#define TRACKING_H

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
#include "Vision.h"
#include "AbsoltePose.h"

namespace ORB_VISLAM
{
	class Tracking
	{
		public:
			Tracking(const cv::Mat &img, cv::Mat &K, cv::Mat &distCoef);
			cv::Mat T_vis;
			cv::Mat T_abs;
			Vision myVision;
			
			cv::Mat Track_Vision();
			cv::Mat Track_GNSS_INS();
		private:

	};

}// namespace ORB_VISLAM

#endif // TRACKING_H
