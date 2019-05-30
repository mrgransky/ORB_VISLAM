#ifndef VISION_H
#define VISION_H

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
			Vision(const std::string &settingFilePath,
						int win_sz, float ssd_th, float ssd_ratio_th);
			
			cv::Mat IMG_ = cv::Mat::zeros(640, 480, CV_8UC3);
			
			void Analyze(cv::Mat &rawImg, 
							std::vector<cv::KeyPoint> &kp, 
							std::vector<std::pair<int,int>> &matches);	
			float fps;
			float focal;
			cv::Point2f pp;
			float sc = 1.0f;
			cv::Mat T_cam = cv::Mat::eye(4, 4, CV_32F);		
		private:
		
			std::vector<cv::KeyPoint> getKP(cv::Mat &rawImg);
			cv::Mat ref_img;
    		cv::Mat mK;
    		cv::Mat mDistCoef;
    		std::vector<cv::KeyPoint> ref_kp;
    		cv::Mat R_f, t_f;
    		
			int vWS;
			float vSSD_TH, vSSD_ratio_TH;
			
			void setCurrentPose(cv::Mat &R_, cv::Mat &t_);
    		void matching(cv::Mat &img, std::vector<cv::KeyPoint> &kp,
    									std::vector<std::pair<int,int>> &matches);
    		
			
			float getSSD(cv::Mat &block_1, cv::Mat &block_2);
			
			cv::Mat getBlock(cv::Mat &img, cv::Point2f &point);
			
    		/*
			int getSSD(cv::Mat &block_r, cv::Mat &block_c);
			
			void getMatches(cv::Mat img_1, cv::Mat img_2, 
							std::vector<cv::KeyPoint> keyP1, 
							std::vector<cv::KeyPoint> keyP2,
							std::vector<std::pair<int,int>> &matches);
							
			std::vector<std::pair<int,int>> crossCheckMatching(	
											std::vector <std::pair<int,int>> C2R,
											std::vector <std::pair<int,int>> R2C);*/
							
			void getMatches(cv::Mat img_1, std::vector<cv::KeyPoint> keyP1, 
							cv::Mat img_2, std::vector<cv::KeyPoint> keyP2,	
							std::vector<std::pair<int,int>> &matches);
						
											
			std::vector<std::pair<int,int>> crossCheckMatching(	
											std::vector <std::pair<int,int>> &m_12,
											std::vector <std::pair<int,int>> &m_21);
											
											
			cv::Mat getHomography(	const std::vector<cv::Point2f> &p_ref, 
									const std::vector<cv::Point2f> &p_mtch);
			
			void decomposeHomography(cv::Mat homography);
	};
}// namespace ORB_VISLAM
#endif // VISION_H
