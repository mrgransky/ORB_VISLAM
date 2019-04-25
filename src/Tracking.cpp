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

#include "Tracking.h"

using namespace std;
using namespace cv;

namespace ORB_VISALAM
{

Tracking::Tracking(const Mat &img, Mat &K, Mat &distCoef)
{
	
	Track_Vision(img, K, distCoef);
	
}


Mat Tracking::Track_Vision()
{
	// TODO: provide inputs for Vision class;
	myVision.;
	
	
}


Mat Tracking::Track_GNSS_INS()
{
	// TODO: provide inputs for AbsolutePose class
}



Mat R_abs, t_abs;
		// absolute position of frame:
		absolutePose(	gpsT[selected_images[ni]], ro[selected_images[ni]], 
						pi[selected_images[ni]], h[selected_images[ni]], 
						lat[selected_images[ni]], lng[selected_images[ni]],
						alt[selected_images[ni]], 
						R_abs, t_abs);

		//T_vec.push_back(CurrentCameraPose(R_abs,t_abs));
		T = CurrentCameraPose(R_abs,t_abs);
		
		

		// TODO: move 2 Tracking
		keyP = getKP(img);
		//visualizeKeyPoints(output_image, keyP, scale, img_id_str);

		matching(img, keyP);
		
		ref_kp 	= keyP;
		//ref_img = img;
		img.copyTo(ref_img);
}
