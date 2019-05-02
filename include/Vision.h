#ifndef FEATUREMATCHING_H
#define FEATUREMATCHING_H

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


namespace ORB_VISLAM
{
	class Vision
	{
		public:
			Vision();
			
			/*FeatureMatching();
			
			std::vector<cv::KeyPoint> ref_kp, keyP;
			cv::Mat ref_img;
			
			std::string frameWinName;
			
			std::vector<cv::KeyPoint> getKP(cv::Mat img);*/

			cv::Mat IMG_ = cv::Mat::zeros(640, 480, CV_8UC3);			
			cv::Mat Analyze(cv::Mat &image);
			
			
		private:
			cv::Mat imgRef;
			
	};
}// namespace ORB_VISLAM


#endif // FEATUREMATCHING_H
