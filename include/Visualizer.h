#ifndef VISUALIZER_H
#define VISUALIZER_H

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
	class Visualizer
	{
		public:
			Visualizer(cv::Mat TransformationMatrix);
			struct Triplet;
			void visualizeMatches(cv::Mat &output_image, cv::Point2f parent, 
									cv::Point2f match, float sc);
			void visualizeKeyPoints(cv::Mat &output_image, 
									std::vector<cv::KeyPoint> kp, 
									float sc, std::string id_str);
			void draw_wrd_axis();
			void run();
			pangolin::OpenGlMatrix currentPose(cv::Mat T);
		private:
			cv::Mat T_ = cv::Mat::eye(4, 4, CV_32F);
			void draw_path(std::vector<Triplet> &vertices);
			void draw_camera(pangolin::OpenGlMatrix &Tc);
			void draw_KF(std::vector<pangolin::OpenGlMatrix> &KeyFrames);

	};

}// namespace ORB_VISLAM

#endif // VISUALIZER_H
