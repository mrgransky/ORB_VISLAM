#ifndef ABSOLUTEPOSITIONING_H
#define ABSOLUTEPOSITIONING_H

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
	class AbsolutePose
	{
		public:
			AbsolutePose(double &ref_lat, double &ref_lng, double &ref_alt);
			
			AbsolutePose();
					
			void calcPose( 	double &lat, double &lng, double &alt, 
							double &roll, double &pitch, double &heading);
			
			void set(cv::Mat &T_GT);
			
			cv::Mat T_abs;
			cv::Mat rvec_abs;
			float AbsScale;
		private:
			cv::Mat tprev;
			const int earth_rad = 6378137;
			const double f_inv = 298.257224f;
			const double f = 1.0 / f_inv;
			double latRef, lngRef, altRef;
			void setCurrentPose(cv::Mat &R_, cv::Mat &t_);
			cv::Mat lla2ENU(double &inpLat, double &inpLong, double &inpAlt);
			cv::Mat abs_rot(double &phi, double &theta, double &psi);
			void calcRotation(cv::Mat &T);
			void getScale(cv::Mat &t_);
	};
}// namespace ORB_VISLAM


#endif // ABSOLUTEPOSITIONING_H
