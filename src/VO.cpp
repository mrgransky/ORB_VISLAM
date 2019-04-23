#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include <pangolin/pangolin.h>

#include <opencv2/core/eigen.hpp>
#include <stdio.h>      /* printf, fopen */
#include <stdlib.h>     /* exit, EXIT_FAILURE */

using namespace std;
using namespace cv;
#define PI 3.1415926f

string frameWinName;

Mat Rref , tvecR;

float rad2deg = 180 / PI;
float d_inv1 = 1;


Mat camera_matrix = Mat::eye(3,3,CV_32F);

Mat dist_coeffs(4,1,CV_32F);

vector<Point2f> ref_interest_pt, cur_interest_pt;

vector<Mat> T_vec;	
vector<Point3f> tv_loc_3D;

struct Triplet
{
	float x, y, z;
};

struct userClick
{
	Mat im;
	vector<Point2f> points;
};

void draw_path(Triplet ref, Triplet cur, float r, float g, float b)
{
	glLineWidth(.9);
	glColor4f(r, g, b,1);
	glBegin(GL_LINES);
	
	glVertex3f(ref.x, ref.y, ref.z);
	glVertex3f(cur.x, cur.y, cur.z);
	
	glEnd();
	glFlush();
}

void LoadImages(const string &inpFile, 
				vector<string> &imgName, 
				vector<long double> &vTimestamps)
{
	ifstream f;
	f.open(inpFile.c_str());

	while(!f.eof()) // end of the file (eof)
	{
		string s;
		getline(f,s);
		if(!s.empty())
		{
			stringstream ss;
			ss << s;
			
			long double t;
			string sRGB;
			
			ss >> t;
			vTimestamps.push_back(t); // retrieve timestamp from rgb.txt
			
			ss >> sRGB;
			imgName.push_back(sRGB); // retrieve img name from rgb.txt
        }
    }
}

void mouseHandler(int event, int x, int y, int flags, void* data_ptr)
{
	userClick *click = ((userClick *) data_ptr);
	if (event == EVENT_LBUTTONDOWN)
	{
		circle(click->im, Point(x,y),3, Scalar(122,125,155), 5);
		imshow(frameWinName, click->im);
		if (click->points.size() < 4)
		{
			click->points.push_back(Point2f(x,y));
		} 
	}
}
vector<Mat> get3D_cam(vector<Point3f> w_3Dpts, Mat R, Mat t)
{
	// 3d points in camera coordinate systems:
    vector<Mat> pt3D_cam_coord;
    
    Mat A = Mat(w_3Dpts).reshape(1).t();
    cout << "\n\n\n"<< endl;
    for(int i = 0; i < A.cols; i++)
    {
    	cout 	<< "\nw_pt[" << i <<"] =\n"	<< A.rowRange(0,3).col(i)
    			<< "\n\nc_pt =\n"			<< R*A.rowRange(0,3).col(i)+t
    			<< endl;
    	pt3D_cam_coord.push_back(R*A.rowRange(0,3).col(i)+t);
    }
    return pt3D_cam_coord;
}

vector<Point2f> getInterestPoints(Mat img, string id_str)
{
	cout << "\n\n" << endl;
	cout << "#########################################################################" << endl;
	cout << "\t\t\tINTEREST POINTS"														<< endl;
	cout << "#########################################################################" << endl;

	vector<Point2f> intPoints;
	Mat tmpImg = img.clone();
	
	userClick uClick;
    uClick.im = tmpImg;
    
    frameWinName = id_str;
	namedWindow(frameWinName);
	
	cout << "Click on four corners of a picture and then press ENTER" << endl;
	setMouseCallback(frameWinName, mouseHandler, &uClick);
	imshow(frameWinName, tmpImg);
	waitKey();
	destroyWindow(frameWinName);
    
    for (size_t i = 0; i < uClick.points.size(); i++)
    {
    	cout << "pt[" << i << "] =\t" << uClick.points[i]<< endl;
    	intPoints.push_back(uClick.points[i]);
    }
    return intPoints;
}

void printHelp(char ** argv)
{			
	cout	<< "\n\nNOT ENOUGH ARGUMENT PROVIDED!!\n\nSyntax:"		
			<< argv[0]	
			<< " [/path/2/image_folder] [/path/2/setting file]"
			<< "\n\nEXAMPLE:\n\n" 	
			<< argv[0] 
			<< " /home/xenial/Datasets/TV_dataset/ /home/xenial/WS_Farid/orb_slam2_TUT/settingFiles/my_huawei.yaml\n\n"
			<< endl;
}

void draw_wrd_axis()
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

void decomposeHomography(Mat homography)
{
	cout << "\n\n" << endl;
	cout << "#########################################################################" << endl;
	cout << "\t\t\tHOMOGRAPHY DECOMPOSITION"											<< endl;
	cout << "#########################################################################" << endl;
	
	vector<Mat> Rs_decomp, ts_decomp, normals_decomp;
	
	int solutions = decomposeHomographyMat(homography, camera_matrix, 
    										Rs_decomp, ts_decomp, normals_decomp);
    cout << "\n\n" << endl;					
	//! [decompose-homography-from-camera-displacement]
	for (int i = 0; i < solutions; i++)
	{
		float factor_d1 = 1.0f / d_inv1;
		Mat rvec_decomp;
		Rodrigues(Rs_decomp[i], rvec_decomp);
		
		cout << "Solution " << i << ":" << endl;
		cout << "rvec decom \t =" << rvec_decomp.t() << endl;
		//cout << "rvec camera displacement \t =" << rvec_1to2.t() << endl;
		
		cout 	<< "tvec decom \t =" << ts_decomp[i].t() 
      			<< "\nscaled by d \t =" << factor_d1 * ts_decomp[i].t() 
      			<< endl;
		
		//cout << "tvec camera displacement \t =" << t_1to2.t() << endl;
		
		cout << "plane normal decom \t =" << normals_decomp[i].t() << endl;
		//cout << "plane normal cam 1 pose \t =" << normal1.t()<< endl;
		cout << "------------------------------------------------------" << endl;
	}
}

Mat getHomography(const vector<Point2f> &p_ref, const vector<Point2f> &p_mtch)
{
	cout << "\n\n" << endl;
	cout << "#########################################################################" << endl;
	cout << "\t\t\tHOMOGRAPHY"															<< endl;
	cout << "#########################################################################" << endl;
	
	for (size_t i = 0; i < p_ref.size(); i++)
	{
		cout << "ref =\t" << p_ref[i] << endl;
	}


	for (size_t j = 0; j < p_mtch.size(); j++)
	{
		cout << "matched =\t" << p_mtch[j] << endl;
	}


	Mat H;
	int nPoints = p_ref.size();
	Mat A(2*nPoints,9,CV_32F);
	
	if (nPoints == 4)
		A.resize(2*nPoints+1,9);
	
	for (size_t i = 0; i < p_ref.size(); i++)
	{
		// x'_i * Hx_i = 0:
		A.at<float>(2*i,0) = 0.0;
        A.at<float>(2*i,1) = 0.0;
        A.at<float>(2*i,2) = 0.0;
        A.at<float>(2*i,3) = -p_ref[i].x;
        A.at<float>(2*i,4) = -p_ref[i].y;
        A.at<float>(2*i,5) = -1;
        A.at<float>(2*i,6) = p_mtch[i].y*p_ref[i].x;
        A.at<float>(2*i,7) = p_mtch[i].y*p_ref[i].y;
        A.at<float>(2*i,8) = p_mtch[i].y;

        A.at<float>(2*i+1,0) = p_ref[i].x;
        A.at<float>(2*i+1,1) = p_ref[i].y;
        A.at<float>(2*i+1,2) = 1;
        A.at<float>(2*i+1,3) = 0.0;
        A.at<float>(2*i+1,4) = 0.0;
        A.at<float>(2*i+1,5) = 0.0;
        A.at<float>(2*i+1,6) = -p_mtch[i].x*p_ref[i].x;
        A.at<float>(2*i+1,7) = -p_mtch[i].x*p_ref[i].y;
        A.at<float>(2*i+1,8) = -p_mtch[i].x;
    }
    
    //cout << "\nA["<< A.rows << " x "<<A.cols << "]= \n"<< A << endl;
    
    // Add an extra line with zero.
	if (nPoints == 4)
	{
		for (int i = 0; i < 9; i ++) 
		{
			A.at<float>(2*nPoints,i) = 0;
		}
	}
	Mat u,w,vt;
    cv::SVDecomp(A,w,u,vt,cv::SVD::MODIFY_A | cv::SVD::FULL_UV);
    
	float smallestSv = w.at<float>(0,0);
	unsigned int indexSmallestSv = 0 ;
	
	for (int i = 1; i < w.rows; i++) 
	{
		if ((w.at<float>(i, 0) < smallestSv))
		{
			smallestSv = w.at<float>(i, 0);
			indexSmallestSv = i;
		}
	}
    H = vt.row(indexSmallestSv).reshape(0,3);
	
	if (H.at<float>(2,2) < 0) // tz < 0
	{
		H *=-1;	
	}
	
	float norm = H.at<float>(2,2);
	H /= norm;
	
	return H;
}

pangolin::OpenGlMatrix getCurrentCameraPose(Mat T)
{
	pangolin::OpenGlMatrix curr_cam_pose;
	
	Mat Rc(3,3,CV_32F);
	Mat tc(3,1,CV_32F);
	
	Rc = T.rowRange(0,3).colRange(0,3);
	tc = T.rowRange(0,3).col(3);
	
		
	curr_cam_pose.m[0]  = Rc.at<float>(0,0);
	curr_cam_pose.m[1]  = Rc.at<float>(1,0);
	curr_cam_pose.m[2]  = Rc.at<float>(2,0);
	curr_cam_pose.m[3]  = 0.0;
	
	curr_cam_pose.m[4]  = Rc.at<float>(0,1);
	curr_cam_pose.m[5]  = Rc.at<float>(1,1);
	curr_cam_pose.m[6]  = Rc.at<float>(2,1);
	curr_cam_pose.m[7]  = 0.0;

	curr_cam_pose.m[8]  = Rc.at<float>(0,2);
	curr_cam_pose.m[9]  = Rc.at<float>(1,2);
	curr_cam_pose.m[10] = Rc.at<float>(2,2);
	curr_cam_pose.m[11] = 0.0;

	curr_cam_pose.m[12] = tc.at<float>(0);
	curr_cam_pose.m[13] = tc.at<float>(1);
	curr_cam_pose.m[14] = tc.at<float>(2);
	curr_cam_pose.m[15] = 1.0;
	
	return curr_cam_pose;
}

void draw_camera(pangolin::OpenGlMatrix &Tc)
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

Mat CurrentCameraPose(Mat R_abs, Mat t_abs)
{
	Mat Tcw_abs = Mat::eye(4,4,CV_32F);
	Mat camC = -R_abs.inv()*t_abs;
	camC.copyTo(Tcw_abs.rowRange(0,3).col(3));
	R_abs.copyTo(Tcw_abs.rowRange(0,3).colRange(0,3));
    cout << "Tcw_abs =\n" << Tcw_abs << endl;
    return Tcw_abs;
}

void getCameraParamerters(const string &settingFilePath, Mat K, Mat DistCoef)
{
	FileStorage fSettings(settingFilePath, FileStorage::READ);
    float fx 			= fSettings["Camera.fx"];
    float fy 			= fSettings["Camera.fy"];
    float cx 			= fSettings["Camera.cx"];
    float cy 			= fSettings["Camera.cy"];


    K.at<float>(0,0) 		= fx;
    K.at<float>(1,1) 		= fy;
    K.at<float>(0,2) 		= cx;
    K.at<float>(1,2) 		= cy;

    DistCoef.at<float>(0) 	= fSettings["Camera.k1"];
    DistCoef.at<float>(1) 	= fSettings["Camera.k2"];
    DistCoef.at<float>(2) 	= fSettings["Camera.p1"];
    DistCoef.at<float>(3) 	= fSettings["Camera.p2"];
    const float k3 			= fSettings["Camera.k3"];

    if(k3!=0)
    {
        DistCoef.resize(5);
        DistCoef.at<float>(4) = k3;
    }
    
	cout << "\n\n" << endl;
	cout << "#########################################################################" << endl;
	cout << "\t\t\tCAMERA PARAMETERS"													<< endl;
	cout << "#########################################################################" << endl;
	
    cout << "- fx: " << fx << endl;
    cout << "- fy: " << fy << endl;
    cout << "- cx: " << cx << endl;
    cout << "- cy: " << cy << endl;
    cout << "- k1: " << DistCoef.at<float>(0) << endl;
    cout << "- k2: " << DistCoef.at<float>(1) << endl;
    if(DistCoef.rows==5)
        cout << "- k3: " << DistCoef.at<float>(4) << endl;
    cout << "- p1: " << DistCoef.at<float>(2) << endl;
    cout << "- p2: " << DistCoef.at<float>(3) << endl;
}

/* Absolute Position using SolvePnP,
Given: 	2D points 	[img coordinate frame]
		3D points 	[wrd coordinate frame] 
Goal:	R,t			[cam coordinate frame]
*/
void absolutePose(vector<Point2f> pt_2D, Mat &R_abs, Mat &tvec_abs)
{
	cout << "\n\n" << endl;
	cout << "#########################################################################" << endl;
	cout << "\t\t\tABSOLUTE POSE {solvePnP}"											<< endl;
	cout << "#########################################################################" << endl;
   
	Mat rvec, tvec, rdirec;
	solvePnP(tv_loc_3D, pt_2D, camera_matrix, dist_coeffs, rvec, tvec);
	
	Rodrigues(rvec, R_abs);
    tvec_abs = tvec;
    T_vec.push_back(CurrentCameraPose(R_abs,tvec_abs));
    
	float a = rvec.at<float>(0,0); 
	float b = rvec.at<float>(1,0); 
	float c = rvec.at<float>(2,0);
	float rot_angle = sqrt(a*a + b*b + c*c);
    rdirec = (1/rot_angle)*rvec;
    
    cout 	<< "\n\nRot Vector = " 		<< rvec.t()
    		<< "\n\nTrans Vector = "	<< tvec.t()
			<< "\n\nRot angle = " 		<< rot_angle*rad2deg
    		<< "\n\nRot direction = "	<< rdirec.t()
    		<< "\n\nR_abs = \n"			<< R_abs
    		<< endl;
}

//! [compute-homography] relative position: frame 1 --> frame 2:
Mat getEuclideanHomography(	const Mat &R_1to2, const Mat &tvec_1to2, 
						const float d_inv, const Mat &normal)
{
    Mat homography = R_1to2 + d_inv * tvec_1to2*normal.t();
    homography /= homography.at<float>(2,2);
    return homography;
}
//! [compute-homography]


// homography with absolute positions of frame 1 and frame 2:
Mat getEuclideanHomography(const Mat &R1, const Mat &tvec1, const Mat &R2, const Mat &tvec2,
                      const float d_inv, const Mat &normal)
{
	Mat homography = R2 * R1.t() + d_inv * (-R2 * R1.t() * tvec1 + tvec2) * normal.t();
	homography /= homography.at<float>(2,2);
    return homography;
}

//! [compute-c2Mc1]
void computeC2MC1(	const Mat &R1, 	const Mat &tvec1, 
					const Mat &R2, 	const Mat &tvec2,
                  	Mat &R_1to2, 	Mat &tvec_1to2)
{
    //c2Mc1 = c2Mo * oMc1 = c2Mo * c1Mo.inv()
    R_1to2 = R2 * R1.t();
    tvec_1to2 = R2 * (-R1.t()*tvec1) + tvec2;
}
//! [compute-c2Mc1]

Mat getCMR(vector<Point2f> intPt)
{
	Mat T_1to2 = Mat::eye(4, 4, CV_32F);
	Mat R		, tvec;
	//Mat Rref	, tvecR;
	Mat R_1to2	, tvec_1to2;
	
	if (ref_interest_pt.empty())
	{	
		cout << "ref empty!" << endl;
		absolutePose(intPt, Rref, tvecR);
	}
	else
	{
		cout << "ref not empty!!!" << endl;
		absolutePose(intPt, R, tvec);
			
		R_1to2 		= R * Rref.t();
		tvec_1to2 	= R * (-Rref.t()*tvecR) + tvec;
	
		Mat rvec_1to2;
    	Rodrigues(R_1to2, rvec_1to2);
    
    	cout << "\n(1 --> 2):\nR = " 	<< rvec_1to2.t()
			 << " , t = " 				<< tvec_1to2.t()
			 << endl;

    	R_1to2.copyTo(T_1to2.rowRange(0,3).colRange(0,3));
		tvec_1to2.copyTo(T_1to2.rowRange(0,3).col(3));
	}
	return T_1to2;
}

int main(int argc, char **argv)
{
	Mat img;
	
	cout << "\n\n######################################################################"<< endl;
	cout <<  "\t\t\tADVANCED IMPLEMENTATION"											<< endl;
	cout << "\n\n######################################################################"<< endl;
	
	if(argc != 3)
	{
		printHelp(argv);
		return -1;
	}
	
	float tv_width = 0.6f;	float tv_height = 0.4f;

	
	Point3f tv_tl(0.8f, 			0.5f + tv_height, 	0.2f);	
	Point3f tv_tr(0.8f + tv_width, 	0.5f + tv_height, 	0.2f);
	Point3f tv_bl(0.8f, 			0.5f, 				0.2f);
	Point3f tv_br(0.8f + tv_width, 	0.5f, 				0.2f);
	
	tv_loc_3D.push_back(tv_tl);		tv_loc_3D.push_back(tv_tr);
	tv_loc_3D.push_back(tv_bl);		tv_loc_3D.push_back(tv_br);
	
	string imgFile = string(argv[1])+"/frames/rgb.txt"; // open rgb.txt from the img folder
	vector<long double> vTimestamps;		// retrieve ts 
	vector<string> 		imgName; 			// retrieve img file names ex: rgb/frame_145.jpg
    LoadImages(imgFile, imgName, vTimestamps);
    
	getCameraParamerters(argv[2], camera_matrix, dist_coeffs);
	
    int nImages = imgName.size();

	vector<size_t> v;
	for(int ni = 0; ni < nImages; ni++) 
	{
		if(ni%100 == 0) 
		{
			v.push_back(ni);
			cout << "ni =\t" <<ni<< endl;
		}
	}
	
	cout << "v size = " << v.size()<< endl;
	for(size_t ni = 0; ni < v.size(); ni++)
	//for(int ni = 0; ni < nImages; ni++) 
	{
		cout 	<<"\n\nReading Frame["	<< imgName[v[ni]] << "]" << endl;
		img = imread(string(argv[1])+"frames/"+imgName[v[ni]],CV_LOAD_IMAGE_GRAYSCALE);
		
		if(img.empty())
		{
			cerr	<<"\nFailed loading frame["
					<< string(argv[1]) 		<< "frames/"
					<< imgName[v[ni]] 			<<"]"
					<< endl;
			return 1;
		}
		
		if(img.channels() < 3) //this should be always true
		{
			cvtColor(img, img, CV_GRAY2BGR);
		}
		
		string img_id_str = imgName[v[ni]];
		
		cur_interest_pt = getInterestPoints(img, img_id_str);
		
		getCMR(cur_interest_pt);
		if (!ref_interest_pt.empty() && !cur_interest_pt.empty())
		{
			Mat H_openCV = findHomography(ref_interest_pt, cur_interest_pt);
			cout << "\n\nH_openCV = \n" <<H_openCV<< endl;
			Mat H = getHomography(ref_interest_pt, cur_interest_pt);
			cout << "\nH_own = \n" <<H<< endl;
			decomposeHomography(H);
		}
		ref_interest_pt = cur_interest_pt;
	}
	
	
	float w = 800;
    float h = 600;
    
    pangolin::CreateWindowAndBind("Demo",w,h);
    glEnable(GL_DEPTH_TEST);

    // Define Projection and initial ModelView matrix
    
    
    pangolin::OpenGlRenderState s_cam(
        pangolin::ProjectionMatrix(w,h,500,500,.1*w,.93*h,0.2,100),
        pangolin::ModelViewLookAt(0,0,1, 0,0,0, pangolin::AxisY)
        //pangolin::ModelViewLookAt(0,-1,0,0,0,0, 0,0,1) // equivalent
    );

    // Create Interactive View in window
    pangolin::Handler3D handler(s_cam);
    
    pangolin::View& d_cam = pangolin::CreateDisplay()
            .SetBounds(0.0, 1.0, 0.0, 1.0, w/h)
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
		for(size_t i = 0; i < T_vec.size(); i++)
		{
			cw = getCurrentCameraPose(T_vec[i]);
			draw_camera(cw);
			
			cur_pt.x = T_vec[i].at<float>(0,3);
			cur_pt.y = T_vec[i].at<float>(1,3);
			cur_pt.z = T_vec[i].at<float>(2,3);
			
			draw_path(ref_pt, cur_pt , 1 , 1, 0);	

			ref_pt = cur_pt;
		}
		pangolin::FinishFrame();
	}
	return 0;
}
