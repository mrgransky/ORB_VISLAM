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
#include "Visualizer.h"


using namespace std;
using namespace cv;

namespace ORB_VISLAM
{

Visualizer::Visualizer(Mat &im, Mat T_cam, int fps, float scale, bool &frame_avl)
{
	cout << "\n\n" << endl;
	cout << "#########################################################################" << endl;
	cout << "\t\t\tVISUALIZER V"														<< endl;
	cout << "#########################################################################" << endl;
	
	cout 	<< "\nOrig Image:\n\nch  = " 	<< im.channels() 
			<< " , depth  = " 			<< im.depth()
			<< " , type  = " 			<< im.type() 
			<< " , dim  = " 			<< im.dims 
			<< " , size  = "			<< im.size() 
			<< endl;
	
	vTcam 	= T_cam;
	
	vFPS 	= fps;
	vScale	= scale;
	hasFrame = frame_avl;
	
	vImg_W = im.cols;
	vImg_H = im.rows;
	
	vImgScaled_W = vImg_W * scale;
	vImgScaled_H = vImg_H * scale;
	
	vImgScaled = Mat::zeros(cv::Size(vImgScaled_W + vImgScaled_W, vImgScaled_H), CV_8UC3);
	
	cout 	<< "\nScaled Image:\n\nch  = " 		<< vImgScaled.channels() 
			<< " , depth  = " 	<< vImgScaled.depth()
			<< " , type  = " 	<< vImgScaled.type() 
			<< " , dim  = " 	<< vImgScaled.dims 
			<< " , size  = "	<< vImgScaled.size() 
			<< endl;
	
	//cout << "has frame init with: \t"<< hasFrame<< endl;
}


Visualizer::Visualizer(Mat &im, Mat T_cam, int fps, Mat T_GT, float scale, bool &frame_avl)
{
	cout << "\n\n" << endl;
	cout << "#########################################################################" << endl;
	cout << "\t\t\tVISUALIZER VI"														<< endl;
	cout << "#########################################################################" << endl;
	
	cout 	<< "\nOrig Image:\n\nch  = " 	<< im.channels() 
			<< " , depth  = " 				<< im.depth()
			<< " , type  = " 				<< im.type() 
			<< " , dim  = " 				<< im.dims 
			<< " , size  = "				<< im.size() 
			<< endl;
	
	
	vTgt 	= T_GT;
	vTcam 	= T_cam;
	vFPS 	= fps;
	vScale	= scale;
	
	hasFrame = frame_avl;
	
	vImg_W = im.cols;
	vImg_H = im.rows;
	
	vImgScaled_W = vImg_W * vScale;
	vImgScaled_H = vImg_H * vScale;
			
	vImgScaled = Mat::zeros(cv::Size(vImgScaled_W + vImgScaled_W, vImgScaled_H), CV_8UC3);
	//cout << "has frame init with: \t"<< hasFrame<< endl;
	
	cout 	<< "\nScaled Image:\n\nch  = " 	<< vImgScaled.channels() 
			<< " , depth  = " 				<< vImgScaled.depth()
			<< " , type  = " 				<< vImgScaled.type() 
			<< " , dim  = " 				<< vImgScaled.dims 
			<< " , size  = "				<< vImgScaled.size() 
			<< endl;
}

struct Visualizer::Triplet
{
	float x, y, z;
};


void Visualizer::draw_KP(Mat &scaled_win, vector<KeyPoint> &kp)
{
	if (!vKP_ref.empty())
	{
		for (size_t i = 0; i < vKP_ref.size(); i++)
		{
			Point2f pt_ref(vScale*vKP_ref[i].pt.x, vScale*vKP_ref[i].pt.y);
			cv::circle(scaled_win, pt_ref, 1, Scalar(1,240,180), FILLED);
		}
	} else
	{
		cout << "ref_kp NOT available!!" << endl;
	}
	
	for (size_t i = 0; i < kp.size(); i++)
	{
		Point2f pt_curr(.5*scaled_win.cols + vScale*kp[i].pt.x, vScale*kp[i].pt.y);
		cv::circle(scaled_win, pt_curr, 1, Scalar(199,199,20), FILLED);
	}
}

void Visualizer::draw_matches(Mat &scaled_win, vector<KeyPoint> &kp, 
								vector<pair<int,int>> &matches)
{
	int min = 0;
	int max = 255;
	if (!vKP_ref.empty())
	{
		for (size_t i = 0; i < matches.size(); i++)
		{
			int parent 	= matches[i].first;
			int match 	= matches[i].second;
		
			Point2f pt_1 = vScale * vKP_ref[parent].pt;
			cv::circle(scaled_win, pt_1, 3, Scalar(1,111,197), LINE_4);
	
			Point2f pt_2(.5*scaled_win.cols + vScale * kp[match].pt.x, vScale * kp[match].pt.y);
			cv::circle(scaled_win, pt_2, 3, Scalar(1,111,197), LINE_4);
	
			cv::line(scaled_win, pt_1, pt_2, Scalar(rand() % max + min, 
						rand() % max + min, rand() % max + min));
		}
	}
}

void Visualizer::show(Mat &frame, 
			vector<KeyPoint> &kp, 
			vector<pair<int,int>> &matches,
			string &frame_name)
{
	vImg 		= frame;
	vImg_name 	= frame_name;
	
	Mat vImg_tmp, vImgR_tmp;
	
	resize(vImg, vImg_tmp, Size(vImgScaled_W, vImgScaled_H));
	vImg_tmp.copyTo(vImgScaled(Rect(vImgScaled_W, 0, vImgScaled_W, vImgScaled_H)));
	
	resize(vImgR, vImgR_tmp, Size(vImgScaled_W, vImgScaled_H));
	vImgR_tmp.copyTo(vImgScaled(Rect(0, 0, vImgScaled_W, vImgScaled_H)));

	stringstream s_img, s_imgR;
	s_img 	<< vImg_name;
	s_imgR 	<< vImgR_name;
	
	draw_KP(vImgScaled, kp);
	draw_matches(vImgScaled, kp, matches);
	
	cv::putText(vImgScaled, s_imgR.str(),
				cv::Point(.01*vImgScaled.cols, .1*vImgScaled.rows),
    			cv::FONT_HERSHEY_PLAIN, 1, Scalar::all(255), 2, LINE_4);
	
	cv::putText(vImgScaled, s_img.str(), 
				cv::Point(.01*vImgScaled.cols + .5*vImgScaled.cols, .1*vImgScaled.rows),
    			cv::FONT_HERSHEY_PLAIN, 1, Scalar::all(255), 2, LINE_4);
    			
	vImgR 		= vImg;
	vImgR_name 	= vImg_name;
	vKP_ref		= kp;
}

void Visualizer::openCV_()
{
	if(!vImg.empty())
	{	
		while(hasFrame)
		{	
			imshow(frameWinName, vImgScaled);
			waitKey(vFPS);

			//waitKey(0);
		}
		//destroyWindow(frameWinName);
		cout << "while opencv ended!" << endl;
	}
	else
	{
		cout << "ref_img EMPTY!!"<< endl;
	}
}

void Visualizer::openGL_()
{
	float width = 1600;
    float heigth = 900;
    
    pangolin::CreateWindowAndBind("ORB_VISLAM", width, heigth);
    glEnable(GL_DEPTH_TEST);
    glEnable (GL_BLEND);
    glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    
    pangolin::OpenGlRenderState s_cam(
        pangolin::ProjectionMatrix(width, heigth, 500, 500, .9*width, .1*heigth, .2, 100),
        pangolin::ModelViewLookAt(0,0,1, 0,0,0, pangolin::AxisY)
        //pangolin::ModelViewLookAt(0,-1,0,0,0,0, 0,0,1) // equivalent
    );

    // Create Interactive View in window
    pangolin::Handler3D handler(s_cam);
    
    pangolin::View& d_cam = pangolin::CreateDisplay()
            .SetBounds(0.0, 1.0, 0.0, 1.0, width/heigth)
            .SetHandler(&handler);


	// GNSS/INS in World Coordinate:
    pangolin::OpenGlMatrix pT_gt;
	pT_gt.SetIdentity();
	
	// camera in World Coordinate:
	pangolin::OpenGlMatrix pTc;
	pTc.SetIdentity();
	
	
	vector<Triplet> vertices_gt;
	vector<Triplet> vertices_cam;
	
	vector<pangolin::OpenGlMatrix> KeyFrames;
	
	int counter_KF = 0;
	while(!pangolin::ShouldQuit())
	{
        // Clear screen and activate view to render into
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
				
		d_cam.Activate(s_cam);
		glClearColor(1,1,1,1);
		

		Triplet current_gt_pt;
		Triplet current_cam_pt;
		
		draw_wrd_axis();

		pT_gt 	= getCurrentPose(vTgt);
		pTc 	= getCurrentPose(vTcam);
		
		if (counter_KF%50 == 0)
		{
			KeyFrames.push_back(pT_gt);
		}
		draw(pT_gt,.1,.1,.1);
		draw_KF(KeyFrames);	
		draw(pTc,.8,0,0);
		
		//s_cam.Follow(T_gt);
		// GNSS/INS:
		current_gt_pt.x = vTgt.at<float>(0,3);
		current_gt_pt.y = vTgt.at<float>(1,3);
		current_gt_pt.z = vTgt.at<float>(2,3);
		
		// camera:
		current_cam_pt.x = vTcam.at<float>(0,3);
		current_cam_pt.y = vTcam.at<float>(1,3);
		current_cam_pt.z = vTcam.at<float>(2,3);
		
		
		
		vertices_gt.push_back(current_gt_pt);
		vertices_cam.push_back(current_cam_pt);
		
		draw_path(vertices_gt, .84, .83, .1);
		counter_KF++;
		draw_path(vertices_cam, .2,.8,.8);
		pangolin::FinishFrame();
	}
}

void Visualizer::run()
{

	openCV_();
	/*thread t1(&Visualizer::openCV_, this);
	thread t2(&Visualizer::openGL_, this);

	t1.join();
	t2.join();*/
}

pangolin::OpenGlMatrix Visualizer::getCurrentPose(Mat &T)
{
	pangolin::OpenGlMatrix curPose;
	
	Mat R(3,3,CV_32F);
	Mat t(3,1,CV_32F);
	
	R = T.rowRange(0,3).colRange(0,3);
	t = T.rowRange(0,3).col(3);
	
		
	curPose.m[0]  = R.at<float>(0,0);
	curPose.m[1]  = R.at<float>(1,0);
	curPose.m[2]  = R.at<float>(2,0);
	curPose.m[3]  = 0.0;
	
	curPose.m[4]  = R.at<float>(0,1);
	curPose.m[5]  = R.at<float>(1,1);
	curPose.m[6]  = R.at<float>(2,1);
	curPose.m[7]  = 0.0;

	curPose.m[8]  = R.at<float>(0,2);
	curPose.m[9]  = R.at<float>(1,2);
	curPose.m[10] = R.at<float>(2,2);
	curPose.m[11] = 0.0;

	curPose.m[12] = t.at<float>(0);
	curPose.m[13] = t.at<float>(1);
	curPose.m[14] = t.at<float>(2);
	curPose.m[15] = 1.0;
	
	return curPose;
}

void Visualizer::draw_wrd_axis()
{
	glColor3f(1,0,0); // red x
	glBegin(GL_LINES);
	glVertex3f(-.05, 0, 0);		glVertex3f(.2, 0,0);

	// arrow
	glVertex3f(.2,0,0);			glVertex3f(.15,.02, 0.0f);
	glVertex3f(.2,0,0);			glVertex3f(.15,-.02, 0.0f);
	
    glEnd();				
    glFlush();

	// y 
	glColor3f(0,1,0); // green y
	glBegin(GL_LINES);
	glVertex3f(0,-.05,0);
	glVertex3f(0,.2,0);

	// arrow
	glVertex3f(0,.2,0);			glVertex3f(.02,.15,0);
	glVertex3f(0,.2,0);			glVertex3f(-.02,.15,0);

	glEnd();
	glFlush();

	// z 
	glColor3f(0,0,1); // blue z
	glBegin(GL_LINES);
	
	glVertex3f(0,0,-.05);		glVertex3f(0,0,.2);

	// arrow
	glVertex3f(0,0,.2);			glVertex3f(0,.02,.15);
	glVertex3f(0,0,.2);			glVertex3f(0,-.02,.15);
	
	glEnd();
	glFlush();
}

void Visualizer::draw_path(vector<Triplet> &vertices, float r, float g, float b)
{
	glLineWidth(.9);
	glColor4f(r,g,b, 1);
	glBegin(GL_LINES);
	
	for (size_t i = 1; i < vertices.size(); i++)
	{
		glVertex3f(vertices[i-1].x, vertices[i-1].y, vertices[i-1].z);
		glVertex3f(vertices[i].x, vertices[i].y, vertices[i].z);
	}
	glEnd();
	glFlush();
}

void Visualizer::draw_KF(vector<pangolin::OpenGlMatrix> &KeyFrames)
{
    const float w = .03;
    const float h = w*1;
    const float z = w*1;

	glLineWidth(.8);
	glBegin(GL_LINES);
	

	for (size_t i = 0; i < KeyFrames.size(); i++)
	{
		glPushMatrix();

#ifdef HAVE_GLES
        glMultMatrixf(KeyFrames[i].m);
#else
        glMultMatrixd(KeyFrames[i].m);
#endif

		// camera axis: X  red.
		glColor3f (1,0,0);
		glVertex3f(KeyFrames[i].m[12], KeyFrames[i].m[13], KeyFrames[i].m[14]);
    	glVertex3f(KeyFrames[i].m[12]+.07, KeyFrames[i].m[13], KeyFrames[i].m[14]);
    	
    	glColor3f (0,1,0);  	//Y green.	
    	glVertex3f(KeyFrames[i].m[12], KeyFrames[i].m[13], KeyFrames[i].m[14]); 	
    	glVertex3f(KeyFrames[i].m[12], KeyFrames[i].m[13]+.07, KeyFrames[i].m[14]);
    	
    	glColor3f (0,0,1);  	//Z  blue.
    	glVertex3f(KeyFrames[i].m[12], KeyFrames[i].m[13], KeyFrames[i].m[14]);
    	glVertex3f(KeyFrames[i].m[12], KeyFrames[i].m[13], KeyFrames[i].m[14]+.07);
		
		
		glColor3f(.93, .44, .9);
		glVertex3f(KeyFrames[i].m[12], 		KeyFrames[i].m[13], 		KeyFrames[i].m[14]);	
		glVertex3f(w+KeyFrames[i].m[12],	h+KeyFrames[i].m[13],		z+KeyFrames[i].m[14]);
		
		glVertex3f(KeyFrames[i].m[12], KeyFrames[i].m[13], KeyFrames[i].m[14]);	
		glVertex3f(w+KeyFrames[i].m[12],-h+KeyFrames[i].m[13],z+KeyFrames[i].m[14]);
		
		glVertex3f(KeyFrames[i].m[12], KeyFrames[i].m[13], KeyFrames[i].m[14]);	
		glVertex3f(-w+KeyFrames[i].m[12],-h+KeyFrames[i].m[13],z+KeyFrames[i].m[14]);
		
		glVertex3f(KeyFrames[i].m[12], KeyFrames[i].m[13], KeyFrames[i].m[14]);	
		glVertex3f(-w+KeyFrames[i].m[12],h+KeyFrames[i].m[13],z+KeyFrames[i].m[14]);
		
		glVertex3f(w+KeyFrames[i].m[12],h+KeyFrames[i].m[13],z+KeyFrames[i].m[14]);
		glVertex3f(w+KeyFrames[i].m[12],-h+KeyFrames[i].m[13],z+KeyFrames[i].m[14]);
		
		glVertex3f(-w+KeyFrames[i].m[12],h+KeyFrames[i].m[13],z+KeyFrames[i].m[14]);
		glVertex3f(-w+KeyFrames[i].m[12],-h+KeyFrames[i].m[13],z+KeyFrames[i].m[14]);
		
		glVertex3f(-w+KeyFrames[i].m[12],h+KeyFrames[i].m[13],z+KeyFrames[i].m[14]);
		glVertex3f(w+KeyFrames[i].m[12],h+KeyFrames[i].m[13],z+KeyFrames[i].m[14]);
		
		glVertex3f(-w+KeyFrames[i].m[12],-h+KeyFrames[i].m[13],z+KeyFrames[i].m[14]);
		glVertex3f(w+KeyFrames[i].m[12],-h+KeyFrames[i].m[13],z+KeyFrames[i].m[14]);
	}
	
    glEnd();
    glFlush();
    glPopMatrix();
}


void Visualizer::draw(pangolin::OpenGlMatrix &T, float r, float g, float b)
{
	glPushMatrix();

#ifdef HAVE_GLES
		glMultMatrixf(T.m);
#else
		glMultMatrixd(T.m);
#endif

    const float w = .02;
    const float h = w*0.7;
    const float z = w*0.5;

	glLineWidth(.8);
	glColor3f(r,g,b);
	
	glBegin(GL_LINES);

    glVertex3f(0,0,0);
	glVertex3f(w,h,z);
    
	glVertex3f(0,0,0);
	glVertex3f(w,-h,z);

	glVertex3f(0,0,0);
	glVertex3f(-w,-h,z);
    
	glVertex3f(0,0,0);
	glVertex3f(-w,h,z);

	glVertex3f(w,h,z);
	glVertex3f(w,-h,z);

	glVertex3f(-w,h,z);
	glVertex3f(-w,-h,z);

	glVertex3f(-w,h,z);
	glVertex3f(w,h,z);

	glVertex3f(-w,-h,z);
	glVertex3f(w,-h,z);

	// axis
    glColor3f (1,0,0);  		glVertex3f (0,0,0);  	glVertex3f (.07,0,0);    // X  red.
    glColor3f (0,1,0);  		glVertex3f (0,0,0);  	glVertex3f (0,.07,0);    // Y green.
    glColor3f (0,0,1);  		glVertex3f (0,0,0);  	glVertex3f (0,0,.07);    // z  blue.
    
    glEnd();
    glFlush();
    glPopMatrix();
}
}
