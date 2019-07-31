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
						size_t minFeatures, float minScale, float distTo3DPts, bool downScale);
			
			cv::Mat IMG_ = cv::Mat::zeros(640, 480, CV_8UC3);
			
			void Analyze(cv::Mat &rawImg, 
							std::vector<cv::KeyPoint> &kp,
							std::vector<std::pair<int,int>> &matches);	
			float fps, foc, sc;
    		float front3DPtsOPCV, front3DPtsOWN;
			cv::Point2f pp;
    		std::vector<cv::Mat> T_f, R_f, t_f, T_local;
    		
    		cv::Mat R_Glob, R_f_E, R_f_0, R_f_1, R_f_2, R_f_3;
    		cv::Mat t_Glob, t_f_E, t_f_0, t_f_1, t_f_2, t_f_3;
    		
    		cv::Mat rvec_Glob, rvec_E, rvec_0, rvec_1, rvec_2, rvec_3,
					rvec_loc, rvec_loc_E, rvec_loc_0, rvec_loc_1, rvec_loc_2, rvec_loc_3;
    		
			int nmatchesCCM = 0, nmatches12 = 0, nmatches21 = 0;
			
			cv::Mat T_cam_E		= cv::Mat::eye(4, 4, CV_32F);
			cv::Mat T_loc_E		= cv::Mat::eye(4, 4, CV_32F);
			
			pcl::PointCloud<pcl::PointXYZ>::Ptr cloud;
			std::vector<cv::Mat> map_3D;
			cv::Mat visionMap, vPt2D_rep, vPt2D_measured;
			
			std::vector<float> reprojection_error;
			float repErr;
		private:
    		std::vector<cv::Mat> R_f_prev, t_f_prev;
    		float vMIN_SCALE, vdistTh_3DPts;
    		bool vDownScaled;
    		
			void get_ORB_kp(cv::Mat &rawImg, std::vector<cv::KeyPoint> &kp, cv::Mat &desc);
			void get_AKAZE_kp(cv::Mat &rawImg, std::vector<cv::KeyPoint> &kp, cv::Mat &desc);
			cv::Mat ref_img;
    		cv::Mat mK, mK_inv, mDistCoef;
    		std::vector<cv::KeyPoint> ref_kp;
    		cv::Mat ref_desc;
    		
			cv::Mat Z_3x1, I_3x3, I_4x4;
			cv::Mat R_Glob_prv, R_f_prev_E, R_f_prev_0, R_f_prev_1, R_f_prev_2, R_f_prev_3;
    		cv::Mat rvec_Glob_prv, rvec_prev_E, rvec_prev_0, rvec_prev_1, rvec_prev_2, rvec_prev_3;
    		cv::Mat t_Glob_prv, t_f_prev_E, t_f_prev_0, t_f_prev_1, t_f_prev_2, t_f_prev_3;
    		
			
			cv::Mat Rt_prev = cv::Mat::eye(3, 4, CV_32F);
			cv::Mat Rt 		= cv::Mat::eye(3, 4, CV_32F);
			int vWS;
			size_t vMIN_NUM_FEAT;
			float vSSD_TH, vSSD_ratio_TH;
			
			void setCurrentPose(cv::Mat &R_, cv::Mat &t_, cv::Mat &T_);
			
    		void matching(cv::Mat &img, std::vector<cv::KeyPoint> &kp,
    									std::vector<std::pair<int,int>> &matches);
    		
    		void matching(cv::Mat &img, std::vector<cv::KeyPoint> &kp, 
    					cv::Mat &desc, 	std::vector<std::pair<int,int>> &match_idx);
			
			void Reconstruction(std::vector<cv::Point2f> &src, 
								std::vector<cv::Point2f> &dst, 
								cv::Mat &Rt,
								std::vector<cv::Mat> &pt3D_loc);

			void Extract4DPts(std::vector<cv::Mat> &src, std::vector<cv::Mat> &dst, 
								cv::Mat &Rt, cv::Mat &Points4D);
			
			void SetR_t(cv::Mat &R_, cv::Mat &t_, cv::Mat &Rt_);
			
			void pose_AND_3dPts(std::vector<cv::Point2f> &dst, std::vector<float> &good_vec, 
								std::vector<cv::Mat> &Rt_vec, std::vector<cv::Mat> &Pts3D_vec, 
								cv::Mat &R_, cv::Mat &t_, cv::Mat &p3_, cv::Mat &measuredPts);
						
			void applyContraints(std::vector<cv::Point2f> &dst, cv::Mat &p3_raw, cv::Mat &Rt, 
								cv::Mat &p3_front, cv::Mat &origMeasPts);
								
			void proj3D_2D(cv::Mat &p3_, cv::Mat &p2_);
			void calcReprojErr(cv::Mat &measured, cv::Mat &reprojected, float &rE);
			void Normalize2DPts(std::vector<cv::Point2f> &src, std::vector<cv::Point2f> &dst,
								std::vector<cv::Mat> &src_normalized, 
								std::vector<cv::Mat> &dst_normalized);
			
			void Normalize2DPts(std::vector<cv::Point2f> &src, std::vector<cv::Point2f> &dst,
								std::vector<cv::Point2f> &src_normalized, 
								std::vector<cv::Point2f> &dst_normalized);
								
			void setGloabalMapPoints(std::vector<cv::Mat> &p3D_loc, cv::Mat &R_, cv::Mat &t_);
			
			void GlobalMapPoints(std::vector<cv::Mat> &p3D_loc, 
								cv::Mat &R_, cv::Mat &t_, 
								cv::Mat &global_pts);
									
			float getSSD(cv::Mat &block_1, cv::Mat &block_2);			
			cv::Mat getBlock(cv::Mat &img, cv::Point2f &point);
			void getMatches(cv::Mat img_1, std::vector<cv::KeyPoint> keyP1, 
							cv::Mat img_2, std::vector<cv::KeyPoint> keyP2,	
							std::vector<std::pair<int,int>> &matches);
			
			void decomE(cv::Mat &E, cv::Mat &R1, cv::Mat &R2, cv::Mat &t1, cv::Mat &t2);
			
			void get_correct_pose(std::vector<cv::Point2f> &src, std::vector<cv::Point2f> &dst, 
								cv::Mat &R1, cv::Mat &R2, cv::Mat &t, 
								cv::Mat &R_correct, cv::Mat &t_correct);
								
			void getDummy(std::vector<cv::Point2f> &src, 
							std::vector<cv::Point2f> &dst, 
							std::vector<int> &ref_kp_idx, 
							std::vector<int> &kp_idx, 
							std::vector<std::vector<cv::DMatch>> &possible_matches,
							std::vector<std::pair<int,int>> &match_idx);
			
			std::vector<std::pair<int,int>> crossCheckMatching(	
											std::vector <std::pair<int,int>> &m_12,
											std::vector <std::pair<int,int>> &m_21);
											
									
			void essential_matrix_inliers(std::vector<cv::Point2f> &src, 
											std::vector<cv::Point2f> &dst, 
											std::vector<int> &ref_kp_idx, 
											std::vector<int> &kp_idx, 
											std::vector<std::vector<cv::DMatch>> &possible_matches,
											std::vector<std::pair<int,int>> &match_idx);
			
			
			void PoseFromEssentialMatrix(	std::vector<cv::Point2f> &src, 
											std::vector<cv::Point2f> &dst,
											cv::Mat &E);
			
			void PoseFromFundamentalMatrix(	std::vector<cv::Point2f> &src, 
											std::vector<cv::Point2f> &dst,
											cv::Mat &F);
			
			void triangulateMyPoints(std::vector<cv::Point2f> &src, 
									std::vector<cv::Point2f> &dst, 
									cv::Mat &Rt, float &acceptance_rate, cv::Mat & P3_H_RAW);
	
			void GetPose(std::vector<cv::Point2f> &src, std::vector<cv::Point2f> &dst, 
								cv::Mat &E);

			void getSkewMatrix(cv::Point2f &pt2D, cv::Mat &skew);
			void getSkewMatrix(cv::Mat &mat, cv::Mat &skew);
			
			void get_info(cv::Mat &matrix, std::string matrix_name);
			const double akaze_thresh = 3e-4; 
    		// AKAZE detection threshold to locate about 1000 keypoints

			void pt2D_to_mat(cv::Point2f &pt2d, cv::Mat &pt2dMat);
			// Nearest-neighbour matching ratio
			const double nn_match_ratio = 0.7f; 
			
			// Minimal number of inliers to draw bounding box
			const int bb_min_inliers = 100; 
			
			// On-screen statistics are updated every 10 frames
			const int stats_update_period = 10; 
			
	};
}// namespace ORB_VISLAM
#endif // VISION_HR
