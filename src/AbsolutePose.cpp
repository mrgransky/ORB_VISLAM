#include "AbsolutePose.h"

using namespace std;
using namespace cv;

namespace ORB_VISLAM
{

AbsolutePose::AbsolutePose(double &ref_lat, double &ref_lng, double &ref_alt)
{
	T_abs = Mat::eye(4, 4, CV_32F);
	
	tprev = Mat::zeros(3, 1, CV_32F);
	latRef = ref_lat;
	lngRef = ref_lng;
	altRef = ref_alt;
}

AbsolutePose::AbsolutePose()
{
	tprev = Mat::zeros(3, 1, CV_32F);
	T_abs = Mat::eye(4, 4, CV_32F);
}

void AbsolutePose::set(Mat &T_GT)
{
	T_GT.copyTo(T_abs);
	calcRotation(T_abs);
	//cout << "\nT_abs = \n" <<T_abs<< endl;
}

void AbsolutePose::calcRotation(Mat &T)
{
	Rodrigues(T.rowRange(0,3).colRange(0,3), rvec_abs);
}

void AbsolutePose::calcPose(double &lat, double &lng, double &alt, 
							double &roll, double &pitch, double &heading)
{
	Mat t_abs = lla2ENU(lat, lng, alt);
	Mat R_abs = abs_rot(roll, pitch, heading);
	setCurrentPose(R_abs, t_abs);
	calcRotation(T_abs);
	getScale(t_abs);
}

void AbsolutePose::setCurrentPose(Mat &R_, Mat &t_)
{
	t_.copyTo(T_abs.rowRange(0,3).col(3));
	R_.copyTo(T_abs.rowRange(0,3).colRange(0,3));
	//cout << "\n\nT_abs =\n" << T_abs << endl;
}

void AbsolutePose::getScale(Mat &t_)
{	
	AbsScale = sqrt(
	(t_.at<float>(0,0) - tprev.at<float>(0,0)) * (t_.at<float>(0,0) - tprev.at<float>(0,0)) +
	(t_.at<float>(1,0) - tprev.at<float>(1,0)) * (t_.at<float>(1,0) - tprev.at<float>(1,0)) +
	(t_.at<float>(2,0) - tprev.at<float>(2,0)) * (t_.at<float>(2,0) - tprev.at<float>(2,0))
					);
	tprev = t_;
}

Mat AbsolutePose::lla2ENU(double &inpLat, double &inpLong, double &inpAlt)
{
	//=========================================================
	// Geodetic to ECEF:
	//=========================================================

	// long-lat 2 x y z:
	//float c,e2,s,xECEF,yECEF,zECEF;
	//float cRef,x0,y0,z0, xEast, yNorth, zUp;

	Mat t = Mat::zeros(3, 1, CV_32F);
	
	double c, e2, s, xECEF,yECEF,zECEF;
	double cRef,x0,y0,z0, xEast, yNorth, zUp;


	c = 1 / sqrt(cos(inpLat*CV_PI/180)*cos(inpLat*CV_PI/180)+
					(1-f)*(1-f)*sin(inpLat*CV_PI/180)*sin(inpLat*CV_PI/180));

	s = (1.0 - f) * (1.0 - f) * c;
	e2 = 1 - (1 - f) * (1 - f);

	xECEF = (earth_rad * c + inpAlt) * (cos(inpLat * CV_PI/180)) * cos(inpLong * CV_PI/180);
	yECEF = (earth_rad * c + inpAlt) * (cos(inpLat * CV_PI/180)) * sin(inpLong * CV_PI/180);
	zECEF = (earth_rad * s + inpAlt) * (sin(inpLat * CV_PI/180));

	//=========================================================
	// ECEF 2 ENU
	//=========================================================

	cRef = 1 / sqrt(cos(latRef * CV_PI/180) * cos(latRef * CV_PI/180)+
							(1-f) * (1-f) * sin(latRef * CV_PI/180) * sin(latRef * CV_PI/180));

	x0 = (earth_rad*cRef + altRef)*cos(latRef*CV_PI/180)*cos(lngRef*CV_PI/180);
	y0 = (earth_rad*cRef + altRef)*cos(latRef*CV_PI/180)*sin(lngRef*CV_PI/180);
	z0 = (earth_rad*cRef*(1-e2) + altRef) * sin(latRef*CV_PI/180);

	xEast = (-(xECEF-x0) * sin(lngRef*CV_PI/180)) + ((yECEF-y0)*(cos(lngRef*CV_PI/180)));
	
	t.at<float>(0,0) = xEast;
	
	yNorth = (-cos(lngRef*CV_PI/180)*sin(latRef*CV_PI/180)*(xECEF-x0)) - 
				(sin(latRef*CV_PI/180)*sin(lngRef*CV_PI/180)*(yECEF-y0)) + 
				(cos(latRef*CV_PI/180)*(zECEF-z0));
	t.at<float>(1,0) = yNorth;

	zUp = (cos(latRef*CV_PI/180)*cos(lngRef*CV_PI/180)*(xECEF-x0)) + 
			(cos(latRef*CV_PI/180)*sin(lngRef*CV_PI/180)*(yECEF-y0)) + 
			(sin(latRef*CV_PI/180)*(zECEF-z0));
	t.at<float>(2,0) = zUp;

	return t;
}

Mat AbsolutePose::abs_rot(double &phi, double &theta, double &psi)
{
	Mat R(3 , 3, CV_32F);
	
	Mat R_x(3, 3, CV_32F);
	Mat R_y(3, 3, CV_32F);
	Mat R_z(3, 3, CV_32F);
	
	// rotation along x-axis (roll - phi) ccw+
	R_x.at<float>(0,0) = 1.0;
	R_x.at<float>(0,1) = 0.0;
	R_x.at<float>(0,2) = 0.0;
	
	R_x.at<float>(1,0) = 0.0;
	R_x.at<float>(1,1) = cos(phi * CV_PI/180);
	R_x.at<float>(1,2) = -sin(phi * CV_PI/180);
	
	R_x.at<float>(2,0) = 0.0;
	R_x.at<float>(2,1) = sin(phi * CV_PI/180);
	R_x.at<float>(2,2) = cos(phi * CV_PI/180);
	
	
	// rotation along y-axis (pitch - theta) ccw+
	R_y.at<float>(0,0) = cos(theta * CV_PI/180);
	R_y.at<float>(0,1) = 0.0;
	R_y.at<float>(0,2) = sin(theta * CV_PI/180);
	
	R_y.at<float>(1,0) = 0.0;
	R_y.at<float>(1,1) = 1.0;
	R_y.at<float>(1,2) = 0.0;
	
	R_y.at<float>(2,0) = -sin(theta * CV_PI/180);
	R_y.at<float>(2,1) = 0.0;
	R_y.at<float>(2,2) = cos(theta * CV_PI/180);
	
	
	// rotation along z-axis (yaw - psi) ccw+
	R_z.at<float>(0,0) = cos(psi*CV_PI / 180);
	R_z.at<float>(0,1) = -sin(psi*CV_PI / 180);
	R_z.at<float>(0,2) = 0.0;
	
	R_z.at<float>(1,0) = sin(psi * CV_PI/180);
	R_z.at<float>(1,1) = cos(psi * CV_PI/180);
	R_z.at<float>(1,2) = 0.0;
	
	R_z.at<float>(2,0) = 0.0;
	R_z.at<float>(2,1) = 0.0;
	R_z.at<float>(2,2) = 1.0;
	
	R = R_z*R_y*R_x;
	
	return R;
}
}
