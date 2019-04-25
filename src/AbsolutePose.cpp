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

#include "AbsolutePose.h"

using namespace std;
using namespace cv;
#define PI 3.1415926f

namespace ORB_VISLAM
{

AbsolutePose::AbsolutePose(double &ref_lat, double &ref_lng, double &ref_alt)
{
	cout << "\n\n" << endl;
	cout << "#########################################################################" << endl;
	cout << "\t\t\tABSOLUTE POSE {GNSS/INS}"											<< endl;
	cout << "#########################################################################" << endl;
	
	latRef = ref_lat;
	lngRef = ref_lng;
	altRef = ref_alt;
}

/* Absolute Position using GNSS/INS
Given:
*/
Mat AbsolutePose::getPose(double &lat, double &lng, double &alt, 
							double &roll, double &pitch, double &heading)
{
	Mat T;
	Mat t_abs = lla2ENU(lat, lng, alt);
	Mat R_abs = abs_rot(roll, pitch, heading);
	CurrentPose(R_abs, t_abs);
	T_abs.copyTo(T);
	return T; 
}

void AbsolutePose::CurrentPose(Mat &R_abs, Mat &t_abs)
{
//	Mat T_abs = Mat::eye(4,4,CV_32F);
	Mat center = -R_abs.inv()*t_abs;
	
	center.copyTo(T_abs.rowRange(0,3).col(3));
	R_abs.copyTo(T_abs.rowRange(0,3).colRange(0,3));
}

Mat AbsolutePose::lla2ENU(double &inpLat, double &inpLong, double &inpAlt)
{

	//=========================================================
	// Geodetic to ECEF:
	//=========================================================

	// long-lat 2 x y z:
	//float c,e2,s,xECEF,yECEF,zECEF;
	//float cRef,x0,y0,z0, xEast, yNorth, zUp;

	Mat t(3, 1, CV_32F, Scalar(0));

	double c, e2, s, xECEF,yECEF,zECEF;
	double cRef,x0,y0,z0, xEast, yNorth, zUp;


	c = 1 / sqrt(cos(inpLat*PI/180)*cos(inpLat*PI/180)+
					(1-f)*(1-f)*sin(inpLat*PI/180)*sin(inpLat*PI/180));

	s = (1.0 - f) * (1.0 - f) * c;
	e2 = 1 - (1 - f) * (1 - f);

	xECEF = (earth_rad * c + inpAlt) * (cos(inpLat * PI/180)) * cos(inpLong * PI/180);
	yECEF = (earth_rad * c + inpAlt) * (cos(inpLat * PI/180)) * sin(inpLong * PI/180);
	zECEF = (earth_rad * s + inpAlt) * (sin(inpLat * PI/180));

	//=========================================================
	// ECEF 2 ENU
	//=========================================================

	cRef = 1 / sqrt(cos(latRef * PI/180) * cos(latRef * PI/180)+
							(1-f) * (1-f) * sin(latRef * PI/180) * sin(latRef * PI/180));

	x0 = (earth_rad*cRef + altRef)*cos(latRef*PI/180)*cos(lngRef*PI/180);
	y0 = (earth_rad*cRef + altRef)*cos(latRef*PI/180)*sin(lngRef*PI/180);
	z0 = (earth_rad*cRef*(1-e2) + altRef) * sin(latRef*PI/180);

	xEast = (-(xECEF-x0) * sin(lngRef*PI/180)) + ((yECEF-y0)*(cos(lngRef*PI/180)));
	t.at<float>(0,0) = xEast;
	
	yNorth = (-cos(lngRef*PI/180)*sin(latRef*PI/180)*(xECEF-x0)) - 
				(sin(latRef*PI/180)*sin(lngRef*PI/180)*(yECEF-y0)) + 
				(cos(latRef*PI/180)*(zECEF-z0));
	t.at<float>(1,0) = yNorth;

	zUp = (cos(latRef*PI/180)*cos(lngRef*PI/180)*(xECEF-x0)) + 
			(cos(latRef*PI/180)*sin(lngRef*PI/180)*(yECEF-y0)) + 
			(sin(latRef*PI/180)*(zECEF-z0));
	t.at<float>(2,0) = zUp;

	//return Vec3f (xEast, yNorth, zUp);
	return t;
}

//Mat AbsolutePose::abs_rot(	const float &phi, 	/* roll */
//							const float &theta, /* pitch */
//							const float &psi	/* yaw */)
Mat AbsolutePose::abs_rot(	double &phi, 	/* roll */
							double &theta, 	/* pitch */
							double &psi		/* yaw */)

{
	Mat R(3,3,CV_32F);
	
	Mat R_x(3,3,CV_32F);
	Mat R_y(3,3,CV_32F);
	Mat R_z(3,3,CV_32F);
	
	// rotation along x-axis (roll - phi) ccw+
	R_x.at<float>(0,0) = 1.0;
	R_x.at<float>(0,1) = 0.0;
	R_x.at<float>(0,2) = 0.0;
	
	R_x.at<float>(1,0) = 0.0;
	R_x.at<float>(1,1) = cos(phi*PI / 180);
	R_x.at<float>(1,2) = -sin(phi*PI / 180);
	
	R_x.at<float>(2,0) = 0.0;
	R_x.at<float>(2,1) = sin(phi*PI / 180);
	R_x.at<float>(2,2) = cos(phi*PI / 180);
	
	
	// rotation along y-axis (pitch - theta) ccw+
	R_y.at<float>(0,0) = cos(theta*PI / 180);
	R_y.at<float>(0,1) = 0.0;
	R_y.at<float>(0,2) = sin(theta*PI / 180);
	
	R_y.at<float>(1,0) = 0.0;
	R_y.at<float>(1,1) = 1.0;
	R_y.at<float>(1,2) = 0.0;
	
	R_y.at<float>(2,0) = -sin(theta*PI / 180);
	R_y.at<float>(2,1) = 0.0;
	R_y.at<float>(2,2) = cos(theta*PI / 180);
	
	
	// rotation along z-axis (yaw - psi) ccw+
	R_z.at<float>(0,0) = cos(psi*PI / 180);
	R_z.at<float>(0,1) = -sin(psi*PI / 180);
	R_z.at<float>(0,2) = 0.0;
	
	R_z.at<float>(1,0) = sin(psi*PI / 180);
	R_z.at<float>(1,1) = cos(psi*PI / 180);
	R_z.at<float>(1,2) = 0.0;
	
	R_z.at<float>(2,0) = 0.0;
	R_z.at<float>(2,1) = 0.0;
	R_z.at<float>(2,2) = 1.0;
	
	R = R_z*R_y*R_x;
	
	return R;
}
}
