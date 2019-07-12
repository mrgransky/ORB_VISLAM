#ifndef VISION_H
#define VISION_H

#include <iostream>
#include <pangolin/pangolin.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/eigen.hpp>


#include <pcl/common/common_headers.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <stdio.h>      /* printf, fopen */
#include <stdlib.h>     /* exit, EXIT_FAILURE */
#include <thread>
#include <limits>
#include <string>
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <time.h>
#include "boost/make_shared.hpp"

namespace ORB_VISLAM
{
	class Vision
	{
		public:	
			Vision(const std::string &settingFilePath,
						int win_sz, float ssd_th, float ssd_ratio_th, 
						size_t minFeatures, float minScale);
			
			cv::Mat IMG_ = cv::Mat::zeros(640, 480, CV_8UC3);
			
			void Analyze(cv::Mat &rawImg, 
							std::vector<cv::KeyPoint> &kp,
							std::vector<std::pair<int,int>> &matches,
							std::vector<cv::Point3f> &map_points);	
			float fps;
			float FOCAL_LENGTH;
			cv::Point2f pp;
			double sc;
			
    		std::vector<cv::Mat> T_f, R_f, t_f, T_local;
    		
    		cv::Mat R_f_E, R_f_0, R_f_1, R_f_2, R_f_3;
    		cv::Mat t_f_E, t_f_0, t_f_1, t_f_2, t_f_3;
    		
    		cv::Mat rvec_E, rvec_0, rvec_1, rvec_2, rvec_3,
					rvec_loc_E, rvec_loc_0, rvec_loc_1, rvec_loc_2, rvec_loc_3;
    		
			int nmatchesCCM = 0, nmatches12 = 0, nmatches21 = 0;
			
			cv::Mat T_prev_E	= cv::Mat::eye(4, 4, CV_32F);
			cv::Mat T_prev_0	= cv::Mat::eye(4, 4, CV_32F);
			cv::Mat T_prev_1	= cv::Mat::eye(4, 4, CV_32F);
			cv::Mat T_prev_2	= cv::Mat::eye(4, 4, CV_32F);
			cv::Mat T_prev_3	= cv::Mat::eye(4, 4, CV_32F);

			cv::Mat T_cam_E		= cv::Mat::eye(4, 4, CV_32F);			
			cv::Mat T_cam_0		= cv::Mat::eye(4, 4, CV_32F);
			cv::Mat T_cam_1 	= cv::Mat::eye(4, 4, CV_32F);
			cv::Mat T_cam_2 	= cv::Mat::eye(4, 4, CV_32F);
			cv::Mat T_cam_3 	= cv::Mat::eye(4, 4, CV_32F);
			
			cv::Mat T_loc_E		= cv::Mat::eye(4, 4, CV_32F);						
			cv::Mat T_loc_0		= cv::Mat::eye(4, 4, CV_32F);
			cv::Mat T_loc_1 	= cv::Mat::eye(4, 4, CV_32F);
			cv::Mat T_loc_2 	= cv::Mat::eye(4, 4, CV_32F);
			cv::Mat T_loc_3 	= cv::Mat::eye(4, 4, CV_32F);
			
			pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud;
			std::vector<cv::Point3f> map_3D;
		private:
    		std::vector<cv::Mat> R_f_prev, t_f_prev;
    		float vMIN_SCALE;
			void get_ORB_kp(cv::Mat &rawImg, std::vector<cv::KeyPoint> &kp, cv::Mat &desc);
			void get_AKAZE_kp(cv::Mat &rawImg, std::vector<cv::KeyPoint> &kp, cv::Mat &desc);
			std::vector<cv::Point3f> get_map_points();
			cv::Mat ref_img;
    		cv::Mat mK;
    		cv::Mat mDistCoef;
    		std::vector<cv::KeyPoint> ref_kp;
    		cv::Mat ref_desc;
    		
			cv::Mat I_3x3 = cv::Mat::eye(3, 3, CV_64F);
			cv::Mat I_4x4 = cv::Mat::eye(4, 4, CV_64F);
			cv::Mat Z_3x1 = cv::Mat::zeros(3, 1, CV_64F);
    		
			cv::Mat R_f_prev_E, R_f_prev_0, R_f_prev_1, R_f_prev_2, R_f_prev_3;
    		
    		cv::Mat rvec_prev_E, rvec_prev_0, rvec_prev_1, rvec_prev_2, rvec_prev_3;    		

    		cv::Mat t_f_prev_E, t_f_prev_0, t_f_prev_1, t_f_prev_2, t_f_prev_3;
    		
			cv::Mat Essential_Matrix, Fundamental_Matrix, Homography_Matrix;
			
			cv::Mat P_prev 	= cv::Mat::eye(3, 4, CV_64F);
			cv::Mat P 		= cv::Mat::zeros(3, 4, CV_64F);
			
			int vWS;
			size_t vMIN_NUM_FEAT;
			float vSSD_TH, vSSD_ratio_TH;
			void extract3DPoints(std::vector<cv::Point2f> &src, 
								std::vector<cv::Point2f> &dst, 
								cv::Mat &P);
			void setCurrentPose(cv::Mat &R_, cv::Mat &t_, cv::Mat &T_);
			
    		void matching(cv::Mat &img, std::vector<cv::KeyPoint> &kp,
    									std::vector<std::pair<int,int>> &matches);
    		
    		void matching(cv::Mat &img, std::vector<cv::KeyPoint> &kp, 
    					cv::Mat &desc, 	std::vector<std::pair<int,int>> &match_idx);
			
			void Reconstruction(std::vector<cv::Point2f> &dst, 
								std::vector<cv::Point2f> &src, 
								cv::Mat &P_prev, cv::Mat &P,
								std::vector<cv::Point3f> &pt3D_loc);
			
			float getSSD(cv::Mat &block_1, cv::Mat &block_2);
			
			cv::Mat getBlock(cv::Mat &img, cv::Point2f &point);

			void getMatches(cv::Mat img_1, std::vector<cv::KeyPoint> keyP1, 
							cv::Mat img_2, std::vector<cv::KeyPoint> keyP2,	
							std::vector<std::pair<int,int>> &matches);
			
			void decomposeEToRANDt(cv::Mat &E, cv::Mat &R1, cv::Mat &R2, cv::Mat &t1, cv::Mat &t2);
			
			std::vector<std::pair<int,int>> crossCheckMatching(	
											std::vector <std::pair<int,int>> &m_12,
											std::vector <std::pair<int,int>> &m_21);
											
			void Point3D_2_Mat(cv::Point3f &pt, cv::Mat &mat);
			cv::Point3f Mat_2_Point3D(cv::Mat &mat);
			
			cv::Mat getHomography(	const std::vector<cv::Point2f> &p_ref, 
									const std::vector<cv::Point2f> &p_mtch);
			//void PoseFromHomographyMatrix(cv::Mat &H);

											
			void essential_matrix_inliers(std::vector<cv::Point2f> &src, 
											std::vector<cv::Point2f> &dst, 
											std::vector<int> &ref_kp_idx, 
											std::vector<int> &kp_idx, 
											std::vector<std::vector<cv::DMatch>> &all_matches,
											std::vector<std::pair<int,int>> &match_idx);
			
			void homography_matrix_inliers(std::vector<cv::Point2f> &src, 
											std::vector<cv::Point2f> &dst, 
											std::vector<int> &ref_kp_idx, 
											std::vector<int> &kp_idx, 
											std::vector<std::vector<cv::DMatch>> &all_matches,
											std::vector<std::pair<int,int>> &match_idx);
			
			void fundamental_matrix_inliers(std::vector<cv::Point2f> &src, 
											std::vector<cv::Point2f> &dst, 
											std::vector<int> &ref_kp_idx, 
											std::vector<int> &kp_idx, 
											std::vector<std::vector<cv::DMatch>> &all_matches,
											std::vector<std::pair<int,int>> &match_idx);
			
			void PoseFromHomographyMatrix(	std::vector<cv::Point2f> &src, 
											std::vector<cv::Point2f> &dst);
			
			void PoseFromEssentialMatrix(	std::vector<cv::Point2f> &src, 
											std::vector<cv::Point2f> &dst);
			
			void PoseFromFundamentalMatrix(	std::vector<cv::Point2f> &src, 
											std::vector<cv::Point2f> &dst);
			
			void triangulateFcn(std::vector<cv::Point2f> &src, 
								std::vector<cv::Point2f> &dst, 
								cv::Mat &P, cv::Mat &P1, double reprojErr);
			
			void getExtrinsic(cv::Mat &skew, cv::Mat &P, cv::Mat &skewXP);					
			void getSkewMatrix(cv::Point2f &pt2D, cv::Mat &skew);
			
			const double akaze_thresh = 3e-4; 
    		// AKAZE detection threshold to locate about 1000 keypoints

			
			// Nearest-neighbour matching ratio
			const double nn_match_ratio = 0.7f; 
			
			// Minimal number of inliers to draw bounding box
			const int bb_min_inliers = 100; 
			
			// On-screen statistics are updated every 10 frames
			const int stats_update_period = 10; 
			
	};
}// namespace ORB_VISLAM
#endif // VISION_HR
