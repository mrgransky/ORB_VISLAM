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

using namespace std;
using namespace cv;
#define PI 3.1415926f
string frameWinName;
float rad2deg = 180 / PI;
float d_inv1 = 1;
const int earth_rad = 6378137;
const long double f_inv = 298.257224f;
const long double f = 1.0 / f_inv;
Mat camera_matrix = Mat::eye(3,3,CV_32F);
Mat dist_coeffs(4,1,CV_32F);

vector<KeyPoint> ref_kp, keyP;
Mat ref_img;
long double latRef, lngRef, altRef;

vector<Mat> T_vec;

struct Triplet
{
	float x, y, z;
};

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
	Mat T_abs = Mat::eye(4,4,CV_32F);
	Mat camC = -R_abs.inv()*t_abs;
	camC.copyTo(T_abs.rowRange(0,3).col(3));
	R_abs.copyTo(T_abs.rowRange(0,3).colRange(0,3));
    cout << "T_abs =\n" << T_abs << endl;
    return T_abs;
}
Mat lla2ENU(	long double inpLat, long double inpLong, long double inpAlt,
				long double inpLatRef, long double inpLongRef, long double inpAltRef)
				
{

//=========================================================
// Geodetic to ECEF:
//=========================================================

// long-lat 2 x y z:
//float c,e2,s,xECEF,yECEF,zECEF;
//float cRef,x0,y0,z0, xEast, yNorth, zUp;
Mat t(3, 1, CV_32F, Scalar(0));

long double c,e2,s,xECEF,yECEF,zECEF;
long double cRef,x0,y0,z0, xEast, yNorth, zUp;


c=1/sqrt(cos(inpLat*PI/180)*cos(inpLat*PI/180)+(1-f)*(1-f)*sin(inpLat*PI/180)*sin(inpLat*PI/180));

s = (1.0 - f) * (1.0 - f) * c;
e2 = 1 - (1 - f) * (1 - f);

xECEF = (earth_rad*c + inpAlt) *(cos(inpLat*PI/180))*cos(inpLong*PI/180);
yECEF = (earth_rad*c + inpAlt) *(cos(inpLat*PI/180))*sin(inpLong*PI/180);
zECEF = (earth_rad*s + inpAlt) *(sin(inpLat*PI/180));

//=========================================================
// ECEF 2 ENU
//=========================================================

cRef=1/sqrt(cos(inpLatRef*PI/180)*cos(inpLatRef*PI/180)+(1-f)*(1-f)*sin(inpLatRef*PI/180)*sin(inpLatRef*PI/180));

x0 = (earth_rad*cRef + inpAltRef)*cos(inpLatRef*PI/180)*cos(inpLongRef*PI/180);
y0 = (earth_rad*cRef + inpAltRef)*cos(inpLatRef*PI/180)*sin(inpLongRef*PI/180);
z0 = (earth_rad*cRef*(1-e2) + inpAltRef) * sin(inpLatRef*PI/180);

xEast = (-(xECEF-x0) * sin(inpLongRef*PI/180)) + ((yECEF-y0)*(cos(inpLongRef*PI/180)));
t.at<float>(0,0) = xEast;

yNorth = (-cos(inpLongRef*PI/180)*sin(inpLatRef*PI/180)*(xECEF-x0)) - 
			(sin(inpLatRef*PI/180)*sin(inpLongRef*PI/180)*(yECEF-y0)) + 
			(cos(inpLatRef*PI/180)*(zECEF-z0));
t.at<float>(1,0) = yNorth;

zUp = (cos(inpLatRef*PI/180)*cos(inpLongRef*PI/180)*(xECEF-x0)) + 
		(cos(inpLatRef*PI/180)*sin(inpLongRef*PI/180)*(yECEF-y0)) + 
		(sin(inpLatRef*PI/180)*(zECEF-z0));
t.at<float>(0,0) = zUp;

//return Vec3f (xEast, yNorth, zUp);
return t;

}

Mat abs_rot(	const float &phi, 	/* roll */
				const float &theta, /* pitch */
				const float &psi	/* yaw */)
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

/* Absolute Position using GNSS/INS
Given:	

*/
void absolutePose(	long double &gps_time, 
					long double &roll,
					long double &pitch,
					long double &heading,
					long double &lat,
					long double &lng,
					long double &alt,
					Mat &R_abs, Mat &t_abs)
{
	cout << "\n\n" << endl;
	cout << "#########################################################################" << endl;
	cout << "\t\t\tABSOLUTE POSE {GNSS/INS}"											<< endl;
	cout << "#########################################################################" << endl;

	// TODO:
	// outcome: R,t absolute
	t_abs = lla2ENU(lat, lng, alt, latRef, lngRef, altRef);
	R_abs = abs_rot(roll, pitch, heading);
}
void printHelp(char ** argv)
{
	cout	<< "\n\nNOT ENOUGH ARGUMENT PROVIDED!!\n\nSyntax:"		
			<< argv[0]	
			<< " [/path/2/image Left] [/path/2/image Right] [/path/2/setting file]"
			<< "\n\nEXAMPLE:\n\n" 	
			<< argv[0] 
			<< " /home/xenial/Datasets/CIVIT/Dec_14/VideoFiles/seq1_short/ /home/xenial/WS_Farid/orb_slam2_TUT/settingFiles/civit.yaml\n\n"
			<< endl;
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

void load_GNSS_INS(const string 				&file_path,
					vector<long double> 		&gpsTime,
					vector<long double> 		&roll,
					vector<long double> 		&pitch,
					vector<long double> 		&heading,
					vector<long double>			&lat,
					vector<long double>			&lng,
					vector<long double>			&alt)
{
	ifstream csvFile;
	csvFile.open(file_path.c_str());

	 if (!csvFile.is_open())
    {
        cout << "Wrong Path!!!!" << endl;
        exit(EXIT_FAILURE);
    }
    string line;
    vector <string> vec;
    getline(csvFile, line); // skip the 1st line (header)

    while (getline(csvFile,line))
    {
        if (line.empty()) // skip empty lines:
        {
            //cout << "empty line!" << endl;
            continue;
        }

        istringstream iss(line);
        string lineStream;
        string::size_type sz;

		vector <long double> lineGNSS_INS;

		while (getline(iss, lineStream, ','))
		{
			lineGNSS_INS.push_back(stold(lineStream,&sz)); // convert to double
			//lineGNSS_INS.push_back(stof(lineStream,&sz)); // convert to float
		}

		gpsTime.push_back(lineGNSS_INS[0]);
		
		lat.push_back(lineGNSS_INS[3]);
		lng.push_back(lineGNSS_INS[4]);
		alt.push_back(lineGNSS_INS[5]);

		roll.push_back(lineGNSS_INS[8]);
		pitch.push_back(lineGNSS_INS[7]);
		heading.push_back(lineGNSS_INS[6]);
    }
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

void visualizeKeyPoints(Mat &output_image, vector<KeyPoint> kp, float sc, string id_str)
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

vector<KeyPoint> getKP(Mat img)
{
	cout << "\n" << endl;
	cout << "#########################################################################" << endl;
	cout << "\t\t\tKEYPOINTS"															<< endl;
	cout << "#########################################################################" << endl;

	vector<KeyPoint> kp;
	Ptr<FeatureDetector> 		detector 	= ORB::create();
    Ptr<DescriptorExtractor> 	extractor 	= ORB::create();
    
	detector->detect(img, kp);
	return kp;
}

void visualizeMatches(Mat &output_image, Point2f parent, Point2f match, float sc)
{
	int min = 0;
	int max = 255;
	
	Point2f pt_1 = sc * parent;
	cv::circle(output_image, pt_1, 3, Scalar(1,111,197), LINE_4);
	
	Point2f pt_2(.5*output_image.cols + sc*match.x, sc*match.y);	
	cv::circle(output_image, pt_2,3, Scalar(1,111,197), LINE_4);
	
	cv::line(output_image, pt_1, pt_2, Scalar(rand() % max + min, rand() % max + min, rand() % max + min));
}

Mat getBlock(Mat img, Point2f point, int window_size)
{
	Mat Block = Mat::zeros(window_size, window_size, CV_32F);
	int r = 0;
	int c = 0;
	for(int u = -window_size/2; u < window_size/2 + 1; u++)
	{
		for(int v = -window_size/2; v < window_size/2 + 1; v++)
		{
			if(int(point.y)+u >= 0 && int(point.x)+v >= 0)
			{
				uchar intensity = img.at<uchar>(int(point.y)+u,int(point.x)+v);
				Block.at<float>(r,c) = int(intensity);
			}
			c = c + 1;
		}
		r = r + 1;
		c = 0;
	}
	r = 0;
	return Block;
}

int getSSD(Mat block_r, Mat block_c)
{
	int ssd = 0;
	for (int xx = 0; xx < block_c.rows; xx++)
	{
		for(int yy = 0; yy < block_c.cols; yy++)
		{
			int df = block_c.at<float>(xx,yy) - block_r.at<float>(xx,yy);
			int dfSq = df*df;
			ssd = ssd + dfSq;
		}
	}
	return ssd;
}


vector<pair<int,int>> getMatches(Mat img_1, Mat img_2, 
								vector<KeyPoint> keyP1, 
								vector<KeyPoint> keyP2)
{
	vector<pair<int,int>> matches; 

	int window_size = 11;
	double ssd_th = 50;
	
	Mat block_2 	= Mat::zeros(window_size, window_size, CV_32F);
	Mat block_1 	= Mat::zeros(window_size, window_size, CV_32F);
	vector<Point2f> kp2_vec;	
	for(size_t i = 0; i < keyP2.size(); i++)
	//for(size_t i = 0; i < 3; i++)
	{
		kp2_vec.push_back(keyP2[i].pt);
		block_2 = getBlock(img_2, kp2_vec[i], window_size);
		
		//cout << "block_current =\n"<< block_2<< endl;
		vector<int> ssd_vec;
		vector<int> ssd_tmp_vec;
		vector<Point2f> kp1_vec;
		
		//for (size_t j = 0; j < 4; j++)
		for (size_t j = 0; j < keyP1.size(); j++)
		{
			kp1_vec.push_back(keyP1[j].pt);
			block_1 = getBlock(img_1, kp1_vec[j], window_size);
			//cout << "block_ref =\n"<< block_1<< endl;
			
			int ssd = 0;	
			ssd = getSSD(block_1, block_2);
			//cout << "SSD = \t" <<ssd<< endl;
			ssd_vec.push_back(ssd);
		}
		ssd_tmp_vec = ssd_vec;
		sort(ssd_vec.begin(),ssd_vec.end());
		
		double ssd_ratio;
		ssd_ratio = static_cast<double>(ssd_vec[0])/static_cast<double>(ssd_vec[1]);
		//cout << "ssd_ratio =\t"<<ssd_ratio<< endl;
		//cout<<setfill('-')<<setw(80)<<"-"<<endl;

		if(ssd_vec[0] < ssd_th)
		{
			for (size_t k = 0; k < ssd_tmp_vec.size(); k++)
			{
				if (ssd_tmp_vec[k] == ssd_vec[0])
				{
					matches.push_back(make_pair(i,k));
				}
			}
		}
	}
	return matches;
}

vector<pair<int,int>> crossCheckMatching(vector <pair<int,int>> C2R, vector <pair<int,int>> R2C)
{	
	vector<pair<int,int>> CrossCheckedMatches; 
	for (size_t i = 0;  i < min(C2R.size(), R2C.size()); i++)
	{
		for (size_t j = 0;  j < max(C2R.size(), R2C.size()); j++)
		{
			if (C2R[j].second == R2C[i].first && 
				C2R[j].first == R2C[i].second)
			{
				CrossCheckedMatches.push_back(make_pair(R2C[i].first, C2R[j].first));
			}
		}
	}
	return CrossCheckedMatches;
}

void matching(Mat img, vector<KeyPoint> kp)
{
	if (!ref_kp.empty())
	{
		cout << "proceed to:\nmatching...!" << endl;
		// 1. matched c2r
		// current is bigger loop
		cout << "\n\nReference \t\t<<<---\t\t Current\n" << endl;
		vector <pair<int,int>> matchedC2R;
		matchedC2R = getMatches(ref_img, img, ref_kp, kp);
		//cout << "matches C2R =\t"<<matchedC2R.size()<< endl;
	
		for (size_t k = 0; k < matchedC2R.size(); k++)
		{
			int parent 	= matchedC2R[k].first;
			int match 	= matchedC2R[k].second;
		}
	
		//	cout<<setfill('-')<<setw(80)<<"-"<<endl;

		// 2. matched r2c
		// ref is bigger loop
		cout << "\n\nReference \t\t--->>>\t\t Current\n" << endl;
		vector <pair<int,int>> matchedR2C;
		matchedR2C = getMatches(img, ref_img, kp, ref_kp);
		//cout << "matches R2C =\t"<<matchedR2C.size()<< endl;
	
		for (size_t k = 0; k < matchedR2C.size(); k++)
		{
			int parent 	= matchedR2C[k].first;
			int match 	= matchedR2C[k].second;
		}
	

		// 3. cross check matching
		vector <pair<int,int>> ccm;
	
		ccm = crossCheckMatching(matchedC2R, matchedR2C);
	
		cout << "ccm sz =\t" << ccm.size()<< endl;
	
		vector<Point2f> pt_ref;
		vector<Point2f> pt_matched;
		for (size_t i = 0; i < ccm.size(); i++)
		{
			int parent 	= ccm[i].first;
			int match 	= ccm[i].second;
			pt_ref.push_back(ref_kp[parent].pt);
			pt_matched.push_back(kp[match].pt);
			
			//visualizeMatches(output_image, ref_kp[parent].pt, kp[match].pt, scale);
		}
		cout << "----------------------------------------------------------------" << endl;
	}
	else
	{
		cout << "Matching cannot proceed!\nref_kp empty!!" << endl;
	}
}
void run_viewer(Mat T)
{
	float width = 800;
    float heigth = 600;
    
    pangolin::CreateWindowAndBind("Demo",width,heigth);
    glEnable(GL_DEPTH_TEST);

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

int main( int argc, char** argv )
{
	cout << "\n\n######################################################################"<< endl;
	cout <<  "\t\t\tSSD FEATURE MATCHING"											<< endl;
	cout << "\n\n######################################################################"<< endl;
	
	if(argc != 3)
	{
		printHelp(argv);
		return -1; 
	}
	
	
	vector <long double> gpsT, ro, pi, h, lat, lng, alt;
	
	string gnss_insFile = string(argv[1])+"/matchedNovatelData.csv";
    load_GNSS_INS(gnss_insFile, gpsT, ro, pi, h, lat, lng, alt);
    
	// assign ref values for lat, lng, alt:
	latRef = lat[0];	lngRef = lng[0];	altRef = alt[0];
    
	//Mat img;
	string imgFile = string(argv[1])+"/frames/rgb.txt"; // open rgb.txt from the img folder
	getCameraParamerters(argv[2], camera_matrix, dist_coeffs);
	
	 
	vector<long double> vTimestamps;		// retrieve ts 
	vector<string> imgName; 		// retrieve img file names ex: rgb/frame_145.jpg
    LoadImages(imgFile, imgName, vTimestamps);
    int nImages = imgName.size();

	vector<size_t> selected_images;
	for(int ni = 0; ni < nImages; ni++) 
	{
		if(ni%200 == 0) 
		{
			selected_images.push_back(ni);
		}
	}
	
	cout 	<< "\nMatching process of " 		<< selected_images.size() 
			<< " frames out of " << nImages 	<< " frames ..." 
			<< endl;
			
	float scale = .48;
		// imgR:
	Mat T;
	for(size_t ni = 0; ni < selected_images.size(); ni++)
	//for(int ni = 0; ni < nImages; ni++) 
	{
		cout 	<<"\n\nReading Frame["	<< imgName[selected_images[ni]] << "]" << endl;
		Mat img = imread(string(argv[1]) + "frames/" + 
				imgName[selected_images[ni]], CV_LOAD_IMAGE_GRAYSCALE);
		
		if(img.empty())
		{
			cerr	<<"\nFailed loading frame["
					<< string(argv[1]) 				<< "frames/"
					<< imgName[selected_images[ni]] <<"]"
					<< endl;
			return 1;
		}
		
		if(img.channels() < 3) //this should be always true
		{
			cvtColor(img, img, CV_GRAY2BGR);
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
    
		string img_id_str = imgName[selected_images[ni]];
		
		int w = img.cols;
		int h = img.rows;
	
		int w_scaled = w*scale;
		int h_scaled = h*scale;
	

		Mat ref_img_tmp;
    	Mat img_tmp;

		Mat output_image = Mat::zeros(cv::Size(w_scaled+w_scaled, h_scaled), CV_8UC3);
		
		resize(img, img_tmp, Size(w_scaled, h_scaled));
		img_tmp.copyTo(output_image(Rect(w_scaled, 0, w_scaled, h_scaled)));
	
	
		keyP = getKP(img);
		//visualizeKeyPoints(output_image, keyP, scale, img_id_str);

		matching(img, keyP);
		
		ref_kp 	= keyP;
		//ref_img = img;
		img.copyTo(ref_img);
		
		resize(ref_img, ref_img_tmp, Size(w_scaled, h_scaled));
    	ref_img_tmp.copyTo(output_image(Rect(0, 0, w_scaled, h_scaled)));
	}
	
	
	
	thread threadViewer(&run_viewer, T);
	threadViewer.join();
	
	
	return 0;
}
