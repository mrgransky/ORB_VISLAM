#include "Visualizer.h"


using namespace std;
using namespace cv;

using namespace pcl;
using namespace pcl::io;
using namespace pcl::visualization;
namespace ORB_VISLAM
{

Visualizer::Visualizer(Mat &im, Mat &T_GT, Mat &T_cam_E,
								Mat T_cam_0, Mat T_cam_1, 
								Mat T_cam_2, Mat T_cam_3,
								int fps, float scale, bool &frame_avl,
								PointCloud<PointXYZRGB>::Ptr &cloud)
{
	cout << "" << endl;
	cout << "#########################################################################" << endl;
	cout << "\t\t\tVISUALIZER"															<< endl;
	cout << "#########################################################################" << endl;

	vTgt 		= T_GT;
		
	vTcam_E 	= T_cam_E;
	vTcam_0 	= T_cam_0;
	vTcam_1 	= T_cam_1;
	vTcam_2 	= T_cam_2;
	vTcam_3 	= T_cam_3;
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
			vector<Point3f> &map_3D,
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
	
	mapPoints	 = map_3D;
	
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
	PCLVisualizer viz ("Point Cloud");
	viz.setBackgroundColor(0,0,0);
	
	// original cloud -> blue
	PointCloudColorHandlerCustom<PointXYZRGB> originalCloudColor(vCloud, 0, 0, 250);
	viz.addPointCloud<PointXYZRGB>(vCloud, originalCloudColor, "originalPointCloud");
	
	viz.addText("Point Cloud", 30, 50, 16, 0, 0, 150);
	
	while (!viz.wasStopped())
	{
		viz.spinOnce();
	}
}

void Visualizer::openCV_()
{
	
	std::string frameWinName = "frames";
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
    float heigth 	= 900;
    
    pangolin::CreateWindowAndBind("ORB_VISLAM", width, heigth);
    
    glEnable(GL_DEPTH_TEST);
    glEnable (GL_BLEND);
    glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    
    pangolin::OpenGlRenderState s_cam(
        pangolin::ProjectionMatrix(width, heigth, 220, 220, .3*width, .5*heigth, .2, 100),
        pangolin::ModelViewLookAt(0,5,0, 0,0,0, pangolin::AxisZ)
        //pangolin::ModelViewLookAt(-0.8,0.5,-0.5, 0,0,0, pangolin::AxisY)
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
	vector<Triplet> vertices_gt;
	vector<pangolin::OpenGlMatrix> KeyFrames;
	
	vector<Triplet>	vertices_cam_Ess;
	vector<Triplet> vertices_cam_0, vertices_cam_1, vertices_cam_2, vertices_cam_3;
	int counter_KF = 0;
	while(!pangolin::ShouldQuit())
	{
        // Clear screen and activate view to render into
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
				
		d_cam.Activate(s_cam);
		glClearColor(1,1,1,0);
		draw_wrd_axis();
		
		draw_map_points();
		
		// ############ Draw GROUND TRUTH ############
		Triplet current_gt_pt;
		pT_gt 	= getCurrentPose(vTgt);
		
		if (counter_KF%50 == 0)
		{
			KeyFrames.push_back(pT_gt);
		}
		draw(pT_gt,0, .58, .16);
		//draw_KF(KeyFrames);	
		
		//s_cam.Follow(pT_gt);
		// GNSS/INS:
		current_gt_pt.x = vTgt.at<float>(0,3);
		current_gt_pt.y = vTgt.at<float>(1,3);
		current_gt_pt.z = vTgt.at<float>(2,3);
		
		vertices_gt.push_back(current_gt_pt);
		
		draw_path(vertices_gt, .1, .3, .98);
		
		counter_KF++;
		// ############ Draw GROUND TRUTH ############
		
		
		// ############ Homography Matrix solution ############
		Triplet current_cam_pt_0, current_cam_pt_1, current_cam_pt_2, current_cam_pt_3;
		
		pangolin::OpenGlMatrix pTc_0;
		pTc_0.SetIdentity();
		pTc_0 	= getCurrentPose(vTcam_0);
		draw(pTc_0,.91,.02,.01); // red
		//s_cam.Follow(pTc_0);
		current_cam_pt_0.x = vTcam_0.at<float>(0,3);
		current_cam_pt_0.y = vTcam_0.at<float>(1,3);
		current_cam_pt_0.z = vTcam_0.at<float>(2,3);
		vertices_cam_0.push_back(current_cam_pt_0);
		draw_path(vertices_cam_0, .2,.2,.2);
		
		pangolin::OpenGlMatrix pTc_1;
		pTc_1.SetIdentity();
		pTc_1 	= getCurrentPose(vTcam_1);
		draw(pTc_1, .08,.84,.2); // green
		current_cam_pt_1.x = vTcam_1.at<float>(0,3);
		current_cam_pt_1.y = vTcam_1.at<float>(1,3);
		current_cam_pt_1.z = vTcam_1.at<float>(2,3);
		vertices_cam_1.push_back(current_cam_pt_1);
		draw_path(vertices_cam_1, .6, .2, .81);
	
	
		pangolin::OpenGlMatrix pTc_2;
		pTc_2.SetIdentity();
		pTc_2 	= getCurrentPose(vTcam_2);
		draw(pTc_2, .01, .01, .01); // black
		current_cam_pt_2.x = vTcam_2.at<float>(0,3);
		current_cam_pt_2.y = vTcam_2.at<float>(1,3);
		current_cam_pt_2.z = vTcam_2.at<float>(2,3);
		vertices_cam_2.push_back(current_cam_pt_2);
		draw_path(vertices_cam_2, .95,.03, .01);
		
		
		pangolin::OpenGlMatrix pTc_3;
		pTc_3.SetIdentity();
		pTc_3 	= getCurrentPose(vTcam_3);
		draw(pTc_3, .01,.01,.92); // blue
		current_cam_pt_3.x = vTcam_3.at<float>(0,3);
		current_cam_pt_3.y = vTcam_3.at<float>(1,3);
		current_cam_pt_3.z = vTcam_3.at<float>(2,3);
		vertices_cam_3.push_back(current_cam_pt_3);
		draw_path(vertices_cam_3, .09,.91, .61);
		// ############ Homography Matrix solution ############
		
		
		// ############ Essential Matrix solution ############
		Triplet current_cam_pt_E;
		pangolin::OpenGlMatrix pTc_Ess;
		pTc_Ess.SetIdentity();

		pTc_Ess 	= getCurrentPose(vTcam_E);
		draw(pTc_Ess, 0.95, 0, 0.7);	// pink
	
		// camera:
		current_cam_pt_E.x = vTcam_E.at<float>(0,3);
		current_cam_pt_E.y = vTcam_E.at<float>(1,3);
		current_cam_pt_E.z = vTcam_E.at<float>(2,3);
		vertices_cam_Ess.push_back(current_cam_pt_E);
		
		draw_path(vertices_cam_Ess, .8,.6,.08);
		// ############ Essential Matrix solution ############
		
		pangolin::FinishFrame();
	}
}

pangolin::OpenGlMatrix Visualizer::getCurrentPose(Mat &T)
{
	pangolin::OpenGlMatrix curPose;
	
	curPose.m[0]  = T.at<float>(0,0);
	curPose.m[1]  = T.at<float>(1,0);
	curPose.m[2]  = T.at<float>(2,0);
	curPose.m[3]  = 0.0;
	
	curPose.m[4]  = T.at<float>(0,1);
	curPose.m[5]  = T.at<float>(1,1);
	curPose.m[6]  = T.at<float>(2,1);
	curPose.m[7]  = 0.0;

	curPose.m[8]  = T.at<float>(0,2);
	curPose.m[9]  = T.at<float>(1,2);
	curPose.m[10] = T.at<float>(2,2);
	curPose.m[11] = 0.0;

	curPose.m[12] = T.at<float>(3,0);
	curPose.m[13] = T.at<float>(3,1);
	curPose.m[14] = T.at<float>(3,2);
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

    const float w = .5;
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

void Visualizer::draw_map_points()
{
	if(mapPoints.empty())
	{
		cout << "No Map Points..." << endl;
		return;
	}
	cout << "Map Points size [draw_map_points] =\t" << mapPoints.size() << endl;

	float mPointSize = 1.2f;
	glPointSize(mPointSize);
	glColor3f(0,0,0);
	glBegin(GL_POINTS);
	cout << "B4 for loop!" << endl;
	for(size_t i = 0; i < mapPoints.size(); i++)
	{
		float x = mapPoints[i].x;
		float y = mapPoints[i].y;
		float z = mapPoints[i].z;
		
		
		/*cout 	<< "Map Points[" 	<< i 
				<< "] = \t" 		<< mapPoints[i].x 
				<< " , "			<< mapPoints[i].y 
				<< " , "			<< mapPoints[i].z 
				<< endl;*/
		glVertex3f(x, y, z);
	}
    cout << "\n\nMap Points DONE!\n" << endl;
	
	
	glEnd();
    glFlush();
}

}
