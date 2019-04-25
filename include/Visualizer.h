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
			Visualizer(double _id);
			/*struct Triplet;
			void visualizeMatches(cv::Mat &output_image, cv::Point2f parent, 
									cv::Point2f match, float sc);
			void draw_camera(pangolin::OpenGlMatrix &Tc);
			void visualizeKeyPoints(cv::Mat &output_image, 
									std::vector<cv::KeyPoint> kp, 
									float sc, std::string id_str);
			//void draw_path(Triplet ref, Triplet cur, float r, float g, float b);
			void draw_wrd_axis();
			void run(cv::Mat T);
			pangolin::OpenGlMatrix getCurrentCameraPose();*/
		private:
	};

}// namespace ORB_VISLAM

#endif // VISUALIZER_H
