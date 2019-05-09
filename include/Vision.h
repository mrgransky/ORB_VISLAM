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
			Vision(const std::string &settingFilePath);
					
			cv::Mat IMG_;
			cv::Mat Analyze(cv::Mat &rawImg);
			float fps;
			
		private:
		
			std::vector<cv::KeyPoint> getKP(cv::Mat &rawImg);
			cv::Mat ref_img;
    		cv::Mat mK;
    		cv::Mat mDistCoef;
    		std::vector<cv::KeyPoint> ref_kp;
    		
    		void matching(cv::Mat &img, std::vector<cv::KeyPoint> &kp);
    		
			int getSSD(cv::Mat &block_r, cv::Mat &block_c);
			cv::Mat getBlock(cv::Mat &img, cv::Point2f &point, int window_size);
			
    		/*std::vector <std::pair<int,int>> getMatches(cv::Mat &img_1, cv::Mat &img_2, 
														std::vector<cv::KeyPoint> &keyP1, 
														std::vector<cv::KeyPoint> &keyP2);*/
														
			void getMatches(cv::Mat img_1, cv::Mat img_2, 
							std::vector<cv::KeyPoint> keyP1, 
							std::vector<cv::KeyPoint> keyP2,
							std::vector<std::pair<int,int>> matches);
														
														
			std::vector<std::pair<int,int>> crossCheckMatching(	
											std::vector <std::pair<int,int>> C2R,
											std::vector <std::pair<int,int>> R2C);
			
	};
}// namespace ORB_VISLAM


#endif // FEATUREMATCHING_H
