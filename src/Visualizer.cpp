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

Visualizer::Visualizer(double _id)
{
	cout << "\n\n" << endl;
	cout << "#########################################################################" << endl;
	cout << "\t\t\t\tVIEWER"															<< endl;
	cout << "#########################################################################" << endl;
	cout << _id<< endl;
}

/*struct Viewer::Triplet
{
	float x, y, z;
}

void Viewer::visualizeKeyPoints(Mat &output_image, vector<KeyPoint> kp, float sc, string id_str)
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
}

void Viewer::run(Mat T)
{
	float width = 800;
    float heigth = 600;
    
    pangolin::CreateWindowAndBind("ORB_VISLAM", width, heigth);
    glEnable(GL_DEPTH_TEST);
    glEnable (GL_BLEND);
    glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // Define Projection and initial ModelView matrix
    
    
    pangolin::OpenGlRenderState s_cam(
        pangolin::ProjectionMatrix(width, heigth, 500, 500, .1*width, .93*heigth, .2, 100),
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

	while(!pangolin::ShouldQuit())
	{
        // Clear screen and activate view to render into
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
			
		//s_cam.Follow(Tcw);	
		d_cam.Activate(s_cam);
		glClearColor(1,1,1,1);
		
		draw_wrd_axis();

		Triplet ref_pt, cur_pt;

		cw = getCurrentCameraPose(T);
		draw_camera(cw);
			
		cur_pt.x = T.at<float>(0,3);
		cur_pt.y = T.at<float>(1,3);
		cur_pt.z = T.at<float>(2,3);
			
		draw_path(ref_pt, cur_pt , 1 , 1, 0);	
		ref_pt = cur_pt;
		pangolin::FinishFrame();
	}
}

pangolin::OpenGlMatrix Viewer::getCurrentCameraPose()
{
	pangolin::OpenGlMatrix cw;
	return cw;
}

void Viewer::draw_wrd_axis()
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

void Viewer::draw_path(Triplet ref, Triplet cur, float r, float g, float b)
{
	glLineWidth(.9);
	glColor4f(r, g, b,1);
	glBegin(GL_LINES);
	
	glVertex3f(ref.x, ref.y, ref.z);
	glVertex3f(cur.x, cur.y, cur.z);
	
	glEnd();
	glFlush();
}

void Viewer::draw_camera(pangolin::OpenGlMatrix &Tc)
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
}*/

}
