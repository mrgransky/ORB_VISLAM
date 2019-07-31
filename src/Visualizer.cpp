#include "Visualizer.h"


using namespace std;
using namespace cv;

using namespace pcl;
using namespace pcl::io;
using namespace pcl::visualization;
namespace ORB_VISLAM
{

Visualizer::Visualizer(Mat &im, Mat &T_GT, Mat &T_cam_E,
								int fps, float scale, bool &frame_avl,
								PointCloud<PointXYZ>::Ptr &cloud)
{
	cout << "" << endl;
	cout << "#########################################################################" << endl;
	cout << "\t\t\tVISUALIZER"															<< endl;
	cout << "#########################################################################" << endl;

	vTgt 		= T_GT;
	vTcam_E 	= T_cam_E;
	vCloud		= cloud;
	
	vFPS 	= fps;
	vScale	= scale;
	
	hasFrame = frame_avl;
	
	vImg_W = im.cols;
	vImg_H = im.rows;
	
	vImgScaled_W = vImg_W * vScale;
	vImgScaled_H = vImg_H * vScale;
			
	vImgScaled = Mat::zeros(cv::Size(vImgScaled_W + vImgScaled_W, vImgScaled_H), CV_8UC3);
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
			cv::circle(scaled_win, pt_ref, 1, Scalar(1,240,195), FILLED);
 
		}
	} else
	{
		cout << "ref_kp NOT available!!" << endl;
	}
	
	for (size_t i = 0; i < kp.size(); i++)
	{
		Point2f pt_curr(.5*scaled_win.cols + vScale*kp[i].pt.x, vScale*kp[i].pt.y);
		cv::circle(scaled_win, pt_curr, 1, Scalar(1,240,195), FILLED);
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
			cv::circle(scaled_win, pt_1, 2, Scalar(200,7,7), FILLED);
	
			Point2f pt_2(.5*scaled_win.cols + vScale * kp[match].pt.x, vScale * kp[match].pt.y);
			cv::circle(scaled_win, pt_2, 2, Scalar(200,7,7), FILLED);
	
			/*cv::line(scaled_win, pt_1, pt_2, Scalar(rand() % max + min, 
						rand() % max + min, rand() % max + min));*/
		}
	}
}

void Visualizer::drawReprojError(Mat &scaled_win, Mat &measuredPts, Mat &reprojectedPts)
{
	if (!vKP_ref.empty())
	{
		for (int i = 0; i < reprojectedPts.cols; i++)
		{
			Point2f pt_rep(.5*scaled_win.cols 	+ 	vScale*reprojectedPts.at<float>(0,i), 
								0				+	vScale*reprojectedPts.at<float>(1,i));

			Point2f pt_meas(.5*scaled_win.cols 	+ 	vScale*measuredPts.at<float>(0,i), 
								0				+	vScale*measuredPts.at<float>(1,i));
			
			cv::circle(scaled_win, pt_rep, 2, Scalar(180,1,2), FILLED);
			cv::circle(scaled_win, pt_meas, 2, Scalar(1,2,211), FILLED);
			cv::line(scaled_win, pt_meas, pt_rep, Scalar(1,255,1));
		}
	}
}
void Visualizer::show(Mat &frame,
			vector<KeyPoint> &kp,
			vector<pair<int,int>> &matches,
			Mat &locP3d,
			Mat &reprojPts,
			Mat &measuredPts,
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

	//draw_KP(vImgScaled, kp);
	//draw_matches(vImgScaled, kp, matches);
	drawReprojError(vImgScaled, measuredPts, reprojPts);
	
	Mat vloc;
	locP3d.copyTo(vloc);
	vglob = Mat::zeros(3, vloc.cols, vloc.type());
	getGlobalPTs3D(vloc, vglob);
	
	cv::putText(vImgScaled, s_imgR.str(),
				cv::Point(.01*vImgScaled.cols, .1*vImgScaled.rows),
    			cv::FONT_HERSHEY_PLAIN, 1, Scalar(5,200,210), 2, LINE_4);
	
	cv::putText(vImgScaled, s_img.str(), 
				cv::Point(.01*vImgScaled.cols + .5*vImgScaled.cols, .1*vImgScaled.rows),
    			cv::FONT_HERSHEY_PLAIN, 1, Scalar(5,220,210), 2, LINE_4);
    			
	vImgR 		= vImg;
	vImgR_name 	= vImg_name;
	vKP_ref		= kp;
}

void Visualizer::getGlobalPTs3D(Mat &loc, Mat &glob)
{
	if (!loc.empty())
	{
		Mat Rt;
		vTcam_E.rowRange(0,3).colRange(0,4).copyTo(Rt);
		Mat temp = Rt * loc;
		temp.copyTo(glob);
	}
}

void Visualizer::run()
{
	thread t1(&Visualizer::openCV_, this);
	thread t2(&Visualizer::openGL_, this);
	//thread t3(&Visualizer::PCL_, this);

	t1.join();
	t2.join();
	//t3.join();
}

void Visualizer::PCL_()
{
	// visualization
	PCLVisualizer viz ("Point_Cloud");
	viz.setBackgroundColor(0,0,0);
	
	// original cloud -> blue
	PointCloudColorHandlerCustom<PointXYZ> originalCloudColor(vCloud, 0, 0, 250);
	viz.addPointCloud<PointXYZ>(vCloud, originalCloudColor, "originalPointCloud");
	
	viz.addText("Point Cloud", 30, 50, 16, 0, 0, 150);
	
	while (!viz.wasStopped())
	{
		viz.spinOnce();
	}
}

void Visualizer::openCV_()
{
	
	std::string frameWinName = "Image_Frames";
	if(!vImg.empty())
	{	
		while(hasFrame)
		{	
			imshow(frameWinName, vImgScaled);
			waitKey(vFPS);
		}
	}
	else
	{
		cout << "ref_img EMPTY!!"<< endl;
	}
}

void Visualizer::openGL_()
{
	float width 	= 1600;
    float heigth 	= 1000;
    
    pangolin::CreateWindowAndBind("AKAZE_VISLAM", width, heigth);
    
    glEnable(GL_DEPTH_TEST);
    glEnable (GL_BLEND);
    glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	pangolin::OpenGlRenderState s_cam(
			pangolin::ProjectionMatrix(width, heigth, 100, 100, .56*width, .97*heigth, .2, 100),
			pangolin::ModelViewLookAt(0,7,0, 0,0,0, pangolin::AxisZ));
	pangolin::View& d_cam = pangolin::CreateDisplay()
            .SetBounds(0.0, 1.0, 0.0, 1.0, -width/heigth)
            .SetHandler(new pangolin::Handler3D(s_cam));

	// GNSS/INS in World Coordinate:
    pangolin::OpenGlMatrix pT_gt;
	pT_gt.SetIdentity();
	vector<Triplet> vertices_gt;
	vector<pangolin::OpenGlMatrix> KeyFrames;
	
	vector<Triplet>	vertices_cam_Ess;
	int counter_KF = 0;
	while(!pangolin::ShouldQuit())
	{ // Clear screen and activate view to render into
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);				
		d_cam.Activate(s_cam);
		glClearColor(1,1,1,0);
		
		
		drawWRLD();
		// ############ Draw GROUND TRUTH ############
		Triplet current_gt_pt;
		pT_gt 	= getCurrentPose(vTgt);
		
		if (counter_KF%50 == 0)
		{
			KeyFrames.push_back(pT_gt);
		}
		draw(pT_gt, .7, .4, .1);
		//draw_KF(KeyFrames);	
		
		//s_cam.Follow(pT_gt);
		// GNSS/INS:
		current_gt_pt.x = vTgt.at<float>(0,3);
		current_gt_pt.y = vTgt.at<float>(1,3);
		current_gt_pt.z = vTgt.at<float>(2,3);
		vertices_gt.push_back(current_gt_pt);
		draw_path(vertices_gt, .1, .42, .98);
		
		counter_KF++;
		// ############ Draw GROUND TRUTH ############
				
		// ############ Essential Matrix solution ############
		Triplet current_cam_pt_E;
		
		pangolin::OpenGlMatrix pTc_Ess;
		pTc_Ess.SetIdentity();

		pTc_Ess 	= getCurrentPose(vTcam_E);
		draw(pTc_Ess, 0.9, 0.8, 0.1); // yellow
		// camera:
		current_cam_pt_E.x = vTcam_E.at<float>(0,3);
		current_cam_pt_E.y = vTcam_E.at<float>(1,3);
		current_cam_pt_E.z = vTcam_E.at<float>(2,3);
		vertices_cam_Ess.push_back(current_cam_pt_E);
		draw_path(vertices_cam_Ess, .81,.06,.08);
		// ############ Essential Matrix solution ############
		vMap.push_back(vglob);
		drawPC();
		
		pangolin::FinishFrame();
	}
}

void Visualizer::drawPC()
{
	glPointSize(.5f);
	glColor3f(.1,.1,.1);
	glBegin(GL_POINTS);
	
	for(size_t i = 0; i < vMap.size(); i++)
	{
		for(int j = 0; j < vMap[i].cols; j++)
		{
			glVertex3f(vMap[i].at<float>(0,j), vMap[i].at<float>(1,j), vMap[i].at<float>(2,j));
		}
	}
	glEnd();
}

pangolin::OpenGlMatrix Visualizer::getCurrentPose(Mat &T)
{
	pangolin::OpenGlMatrix curPose;
	
	curPose.m[0]  = T.at<float>(0,0);
	curPose.m[1]  = T.at<float>(1,0);
	curPose.m[2]  = T.at<float>(2,0);
	curPose.m[3]  = 0;
	
	curPose.m[4]  = T.at<float>(0,1);
	curPose.m[5]  = T.at<float>(1,1);
	curPose.m[6]  = T.at<float>(2,1);
	curPose.m[7]  = 0;

	curPose.m[8]  = T.at<float>(0,2);
	curPose.m[9]  = T.at<float>(1,2);
	curPose.m[10] = T.at<float>(2,2);
	curPose.m[11] = 0;

	curPose.m[12] = T.at<float>(0,3);
	curPose.m[13] = T.at<float>(1,3);
	curPose.m[14] = T.at<float>(2,3);
	curPose.m[15] = 1.0;
	
	
	return curPose;
}

void Visualizer::drawWRLD()
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
	glLineWidth(5);
	glColor4f(r, g, b, 1);
	glBegin(GL_LINE_STRIP);
	
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
    const float w = .3;
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
    	glVertex3f(KeyFrames[i].m[12]+2.2, KeyFrames[i].m[13], KeyFrames[i].m[14]);
    	
    	glColor3f (0,1,0);  	//Y green.	
    	glVertex3f(KeyFrames[i].m[12], KeyFrames[i].m[13], KeyFrames[i].m[14]); 	
    	glVertex3f(KeyFrames[i].m[12], KeyFrames[i].m[13]+2.2, KeyFrames[i].m[14]);
    	
    	glColor3f (0,0,1);  	//Z  blue.
    	glVertex3f(KeyFrames[i].m[12], KeyFrames[i].m[13], KeyFrames[i].m[14]);
    	glVertex3f(KeyFrames[i].m[12], KeyFrames[i].m[13], KeyFrames[i].m[14]+2.2);
		
		
		glColor3f(.93, .1, .09);
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

    const float w = .9;
    const float h = w*0.7;
    const float z = w*0.5;
    
	glLineWidth(1);
	glBegin(GL_LINES);
	
	// axis
    glColor3f (1,0,0);  		glVertex3f (0, 0, 0);  	glVertex3f (1, 0, 0);    // X  red.
    glColor3f (0,1,0);  		glVertex3f (0, 0, 0);  	glVertex3f (0, 1, 0);    // Y green.
	glColor3f (0,0,1); 			glVertex3f (0, 0, 0); 	glVertex3f (0, 0, 1); // z  blue.
	
	glColor3f(r,g,b);
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

    
    glEnd();
    glFlush();
    glPopMatrix();
}
}
