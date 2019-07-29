#ifndef VISUALIZER_H
#define VISUALIZER_H

#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include <pangolin/pangolin.h>



#include <pcl/common/common_headers.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <opencv2/core/eigen.hpp>
#include <stdio.h>      /* printf, fopen */
#include <stdlib.h>     /* exit, EXIT_FAILURE */
#include <thread>


namespace ORB_VISLAM
{
	class Visualizer
	{
		public:
			Visualizer(cv::Mat &im, cv::Mat &T_GT, 	 cv::Mat &T_cam_E,
									cv::Mat T_cam_0, cv::Mat T_cam_1, 
									cv::Mat T_cam_2, cv::Mat T_cam_3, 
									int fps, float scale, bool &frame_avl,
									pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud);
			struct Triplet;
			void run();
			cv::Mat vImg 	= cv::Mat::zeros(640, 480, CV_8UC3);
			std::string vImg_name;
			
			cv::Mat vImgR 	= cv::Mat::zeros(640, 480, CV_8UC3);
			std::string vImgR_name = "NULL";
			cv::Mat vImgScaled = cv::Mat::zeros(640, 480, CV_8UC3);
			std::vector<cv::KeyPoint> vKP_ref;

			void show(cv::Mat &frame, 
						std::vector<cv::KeyPoint> &kp, 
						std::vector<std::pair<int,int>> &matches,
						cv::Mat &loc3Dpts,
						std::string &frame_name);
			std::vector<cv::Mat> vMap;
			bool hasFrame;
			pangolin::OpenGlMatrix getCurrentPose(cv::Mat &T);
		private:
			cv::Mat vTgt, vTcam_E, vTcam_0, vTcam_1, vTcam_2, vTcam_3;
			cv::Mat vglob;
			pcl::PointCloud<pcl::PointXYZ>::Ptr vCloud;
			void getGlobalPTs3D(cv::Mat &loc, cv::Mat &glob);

			std::mutex visualizerMutex;
			int vImg_W, vImg_H, vImgScaled_W, vImgScaled_H;
			float vScale;
			int vFPS;
			void draw_path(std::vector<Triplet> &vertices, float r, float g, float b);
			void draw(pangolin::OpenGlMatrix &T, float r, float g, float b);
			void draw_KF(std::vector<pangolin::OpenGlMatrix> &KeyFrames);
			void drawPC();
			
			void draw_KP(cv::Mat &scaled_win, std::vector<cv::KeyPoint> &kp);
			void draw_matches(cv::Mat &scaled_win, std::vector<cv::KeyPoint> &kp,
								std::vector<std::pair<int,int>> &matches);

			void drawWRLD();
			void openCV_();
			void openGL_();
			void PCL_();
	};

}// namespace ORB_VISLAM

#endif // VISUALIZER_H
