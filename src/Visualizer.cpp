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

Visualizer::Visualizer(Mat &im, Mat TransformationMatrix, bool &frame_avl)
{
	cout << "\n\n" << endl;
	cout << "#########################################################################" << endl;
	cout << "\t\t\tVISUALIZER"															<< endl;
	cout << "#########################################################################" << endl;
	T_ = TransformationMatrix;
	hasFrame = frame_avl;
	
	vImg_W = im.cols;
	vImg_H = im.rows;
	
	vImgScaled_W = vImg_W * scale;
	vImgScaled_H = vImg_H * scale;
	
	vImgScaled = Mat::zeros(cv::Size(vImgScaled_W + vImgScaled_W, vImgScaled_H), CV_8UC3);
	//cout << "has frame init with: \t"<< hasFrame<< endl;
}

struct Visualizer::Triplet
{
	float x, y, z;
};


/*void Viewer::visualizeKeyPoints(Mat &output_image, vector<KeyPoint> kp, float sc, string id_str)
{	
	if (!ref_kp.empty())
	{
		frameWinName = "Frames";
		namedWindow(frameWinName);
		for (size_t i = 0; i < ref_kp.size(); i++)
		{
			Point2f pt_ref(sc*ref_kp[i].pt.x, sc*ref_kp[i].pt.y);
			cv::circle(output_image, pt_ref, 1, Scalar(1,240,180), FILLED);
		}
	
		for (size_t i = 0; i < kp.size(); i++)
		{
			Point2f pt_curr(.5*output_image.cols + sc*kp[i].pt.x, sc*kp[i].pt.y);
			cv::circle(output_image, pt_curr, 1, Scalar(199,199,20), FILLED);
		}
		
		imshow(frameWinName, output_image);
		waitKey(2000);
		destroyWindow(frameWinName);
	}else
	{
		cout << "Keypoints Visualization cannot proceed!\nref_kp empty!!" << endl;
	}
}

void Viewer::visualizeMatches(Mat &output_image, Point2f parent, Point2f match, float sc)
{
	int min = 0;
	int max = 255;
	
	Point2f pt_1 = sc * parent;
	cv::circle(output_image, pt_1, 3, Scalar(1,111,197), LINE_4);
	
	Point2f pt_2(.5*output_image.cols + sc*match.x, sc*match.y);	
	cv::circle(output_image, pt_2,3, Scalar(1,111,197), LINE_4);
	
	cv::line(output_image, pt_1, pt_2, Scalar(rand() % max + min, 
				rand() % max + min, rand() % max + min));
}*/

void Visualizer::show(Mat &frame, string &frame_name, int fps)
{
	vImg = frame;
	vImg_name = frame_name;
	
	vFPS = fps;
	Mat vImg_tmp, vImgR_tmp;
	
	resize(vImg, vImg_tmp, Size(vImgScaled_W, vImgScaled_H));
	vImg_tmp.copyTo(vImgScaled(Rect(vImgScaled_W, 0, vImgScaled_W, vImgScaled_H)));
	
	
	// TODO: modification required!
	resize(vImgR, vImgR_tmp, Size(vImgScaled_W, vImgScaled_H));
	vImgR_tmp.copyTo(vImgScaled(Rect(0, 0, vImgScaled_W, vImgScaled_H)));

	stringstream s_img, s_imgR;
	s_img 	<< vImg_name;
	s_imgR 	<< vImgR_name;
			
	cv::putText(vImgScaled, s_imgR.str(),
				cv::Point(.01*vImgScaled.cols, .1*vImgScaled.rows),
    			cv::FONT_HERSHEY_PLAIN, 1, Scalar::all(255), 2, LINE_4);
	
	cv::putText(vImgScaled, s_img.str(), 
				cv::Point(.01*vImgScaled.cols + .5*vImgScaled.cols, .1*vImgScaled.rows),
    			cv::FONT_HERSHEY_PLAIN, 1, Scalar::all(255), 2, LINE_4);


	vImgR 		= vImg;
	vImgR_name 	= vImg_name;
}

void Visualizer::openCV_()
{
	if(!vImg.empty())
	{
		while(hasFrame)
		{
				
			//imshow(frameWinName, vImg);
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


	// Camera in World Coordinate:
    pangolin::OpenGlMatrix cw;
    cw.SetIdentity();
	
	vector<Triplet> vertices;
	vector<pangolin::OpenGlMatrix> KeyFrames;
	
	int counter_KF = 0;
	while(!pangolin::ShouldQuit())
	{
        // Clear screen and activate view to render into
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
				
		d_cam.Activate(s_cam);
		glClearColor(1,1,1,1);
		

		Triplet cur_pt;
		draw_wrd_axis();

		cw = currentPose(T_);
		
		if (counter_KF%50 == 0)
		{
			KeyFrames.push_back(cw);
		}
		draw_camera(cw);
		draw_KF(KeyFrames);	
			
		//s_cam.Follow(cw);
		cur_pt.x = T_.at<float>(0,3);
		cur_pt.y = T_.at<float>(1,3);
		cur_pt.z = T_.at<float>(2,3);
		
		vertices.push_back(cur_pt);
		draw_path(vertices);
		counter_KF++;
		pangolin::FinishFrame();
	}
}

void Visualizer::run()
{
	thread t1(&Visualizer::openCV_, this);
	thread t2(&Visualizer::openGL_, this);

	t1.join();
	t2.join();
}

pangolin::OpenGlMatrix Visualizer::currentPose(Mat T)
{
	pangolin::OpenGlMatrix curPose;
	
	Mat Rc(3,3,CV_32F);
	Mat tc(3,1,CV_32F);
	
	Rc = T.rowRange(0,3).colRange(0,3);
	tc = T.rowRange(0,3).col(3);
	
		
	curPose.m[0]  = Rc.at<float>(0,0);
	curPose.m[1]  = Rc.at<float>(1,0);
	curPose.m[2]  = Rc.at<float>(2,0);
	curPose.m[3]  = 0.0;
	
	curPose.m[4]  = Rc.at<float>(0,1);
	curPose.m[5]  = Rc.at<float>(1,1);
	curPose.m[6]  = Rc.at<float>(2,1);
	curPose.m[7]  = 0.0;

	curPose.m[8]  = Rc.at<float>(0,2);
	curPose.m[9]  = Rc.at<float>(1,2);
	curPose.m[10] = Rc.at<float>(2,2);
	curPose.m[11] = 0.0;

	curPose.m[12] = tc.at<float>(0);
	curPose.m[13] = tc.at<float>(1);
	curPose.m[14] = tc.at<float>(2);
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

void Visualizer::draw_path(vector<Triplet> &vertices)
{
	glLineWidth(.9);
	glColor4f(.84, .83, .1,1);
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
		
		
		glColor3f(.1,.91,.95);
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


void Visualizer::draw_camera(pangolin::OpenGlMatrix &Tc)
{
	glPushMatrix();

#ifdef HAVE_GLES
        glMultMatrixf(Tc.m);
#else
        glMultMatrixd(Tc.m);
#endif

    const float w = .02;
    const float h = w*0.7;
    const float z = w*0.5;

	glLineWidth(.8);
	glColor3f(.1,.1,.1);
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

	// camera axis
    glColor3f (1,0,0);  		glVertex3f (0,0,0);  	glVertex3f (.07,0,0);    // X  red.
    glColor3f (0,1,0);  		glVertex3f (0,0,0);  	glVertex3f (0,.07,0);    // Y green.
    glColor3f (0,0,1);  		glVertex3f (0,0,0);  	glVertex3f (0,0,.07);    // z  blue.
    
    glEnd();
    glFlush();
    glPopMatrix();
}

/*Mat Visualizer::visualizeFrames(Mat &frame)
{
	Mat out;
	resize(frame, f, Size(w_scaled, h_scaled));
	f.copyTo(out(Rect(w_scaled, 0, w_scaled, h_scaled)));
	
		
		
	resize(ref, ref_img_tmp, Size(w_scaled, h_scaled));
    ref_img_tmp.copyTo(out(Rect(0, 0, w_scaled, h_scaled)));
}*/

}
