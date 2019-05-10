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
			Visualizer(cv::Mat &im, cv::Mat TransformationMatrix, bool &frame_avl);
			
			struct Triplet;
			/*void visualizeMatches(cv::Mat &output_image, cv::Point2f parent, 
									cv::Point2f match, float sc);
			void visualizeKeyPoints(cv::Mat &output_image, 
									std::vector<cv::KeyPoint> kp, 
									float sc, std::string id_str);
			*/
			void draw_wrd_axis();
			void run();
			
			cv::Mat vImg 	= cv::Mat::zeros(640, 480, CV_8UC3);
			std::string vImg_name;
			
			cv::Mat vImgR 	= cv::Mat::zeros(640, 480, CV_8UC3);
			std::string vImgR_name = "NULL";
			std::vector<cv::KeyPoint> vKP_ref;
			
			cv::Mat vImgScaled = cv::Mat::zeros(640, 480, CV_8UC3);
			
			void show(cv::Mat &frame, std::string &frame_name, int fps);
			
			
			void show(cv::Mat &frame, 
						std::vector<cv::KeyPoint> &kp, 
						std::vector<std::pair<int,int>> &matches,
						std::string &frame_name, int fps);
			
			bool hasFrame;
			pangolin::OpenGlMatrix currentPose(cv::Mat T);
		private:
			std::string frameWinName = "frames";
			cv::Mat T_;
			std::mutex visualizerMutex;
			int vImg_W, vImg_H, vImgScaled_W, vImgScaled_H;
			double scale = .48;
			int vFPS;
			void draw_path(std::vector<Triplet> &vertices);
			void draw_camera(pangolin::OpenGlMatrix &Tc);
			void draw_KF(std::vector<pangolin::OpenGlMatrix> &KeyFrames);

			void draw_KP(cv::Mat &scaled_win, std::vector<cv::KeyPoint> &kp);
			void draw_matches(cv::Mat &scaled_win, std::vector<cv::KeyPoint> &kp,
								std::vector<std::pair<int,int>> &matches);
			
			void openCV_();
			void openGL_();
			
			
	};

}// namespace ORB_VISLAM

#endif // VISUALIZER_H
