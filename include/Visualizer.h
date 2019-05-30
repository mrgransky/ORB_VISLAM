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
			Visualizer(cv::Mat &im, cv::Mat T_cam, int fps, 
						cv::Mat T_GT, float scale, bool &frame_avl);
						
			Visualizer(cv::Mat &im, cv::Mat T_cam, int fps, float scale, bool &frame_avl);
			
			struct Triplet;
			void draw_wrd_axis();
			void run();
			
			cv::Mat vImg 	= cv::Mat::zeros(640, 480, CV_8UC3);
			std::string vImg_name;
			
			cv::Mat vImgR 	= cv::Mat::zeros(640, 480, CV_8UC3);
			std::string vImgR_name = "NULL";
			
			
			cv::Mat vImgScaled = cv::Mat::zeros(640, 480, CV_8UC3);
			
			//void show(cv::Mat &frame, std::string &frame_name, int fps);
			std::vector<cv::KeyPoint> vKP_ref;

			void show(cv::Mat &frame, 
						std::vector<cv::KeyPoint> &kp, 
						std::vector<std::pair<int,int>> &matches,
						std::string &frame_name);
						
			
			bool hasFrame;
			pangolin::OpenGlMatrix getCurrentPose(cv::Mat &T);
		private:
			std::string frameWinName = "frames";
			
			cv::Mat vTgt, vTcam;
			
			
			std::mutex visualizerMutex;
			int vImg_W, vImg_H, vImgScaled_W, vImgScaled_H;
			float vScale;
			int vFPS;
			void draw_path(std::vector<Triplet> &vertices, float r, float g, float b);
			void draw(pangolin::OpenGlMatrix &T, float r, float g, float b);
			void draw_KF(std::vector<pangolin::OpenGlMatrix> &KeyFrames);

			void draw_KP(cv::Mat &scaled_win, std::vector<cv::KeyPoint> &kp);
			void draw_matches(cv::Mat &scaled_win, std::vector<cv::KeyPoint> &kp,
								std::vector<std::pair<int,int>> &matches);
			
			void openCV_();
			void openGL_();
			
			
	};

}// namespace ORB_VISLAM

#endif // VISUALIZER_H
