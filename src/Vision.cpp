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
#include "Vision.h"

using namespace std;
using namespace cv;
#define PI 3.1415926f

namespace ORB_VISLAM
{

Vision::Vision(const string &settingFilePath)
{
	cout << "\n\n" << endl;
	cout << "#########################################################################" << endl;
	cout << "\t\t\tVISION"																<< endl;
	cout << "#########################################################################" << endl;
	
	FileStorage fSettings(settingFilePath, FileStorage::READ);
    float fx 			= fSettings["Camera.fx"];
    float fy 			= fSettings["Camera.fy"];
    float cx 			= fSettings["Camera.cx"];
    float cy 			= fSettings["Camera.cy"];
	
	fps 				= fSettings["Camera.fps"];
	
	Mat K = Mat::eye(3, 3, CV_32F);
	
	K.at<float>(0,0) 		= fx;
    K.at<float>(1,1) 		= fy;
    K.at<float>(0,2) 		= cx;
    K.at<float>(1,2) 		= cy;

	K.copyTo(mK);
	
	Mat DistCoef(4, 1, CV_32F);
	
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
    DistCoef.copyTo(mDistCoef);
    
	cout << "\n\n" << endl;
	cout << "#########################################################################" << endl;
	cout << "\t\t\tCAMERA PARAMETERS"													<< endl;
	cout << "#########################################################################" << endl;
	
    cout << "- fx: " << fx << endl;
    cout << "- fy: " << fy << endl;
    cout << "- cx: " << cx << endl;
    cout << "- cy: " << cy << endl;
    cout << "- k1: " << DistCoef.at<float>(0) 	<< endl;
    cout << "- k2: " << DistCoef.at<float>(1) 	<< endl;
    if(DistCoef.rows == 5)
        cout << "- k3: " << DistCoef.at<float>(4) << endl;
    cout << "- p1: " << DistCoef.at<float>(2) 	<< endl;
    cout << "- p2: " << DistCoef.at<float>(3) 	<< endl;
	cout << "- FPS:" <<	fps						<< endl;
	
	IMG_ = cv::Mat::zeros(fSettings["Camera.height"], fSettings["Camera.width"], CV_8UC3);
}

Mat Vision::Analyze(Mat &rawImg)
{
	vector<KeyPoint> kp;
	
	kp = getKP(rawImg);
	matching(rawImg, kp);
	
	ref_kp = kp;
	ref_img = rawImg;
	
	
	return rawImg;
}

vector<KeyPoint> Vision::getKP(Mat &rawImg)
{
	/*cout << "\n" << endl;
	cout << "#########################################################################" << endl;
	cout << "\t\t\tKEYPOINTS"															<< endl;
	cout << "#########################################################################" << endl;*/

	vector<KeyPoint> kp;
	Ptr<FeatureDetector> 		detector 	= ORB::create();
    Ptr<DescriptorExtractor> 	extractor 	= ORB::create();
    
	detector->detect(rawImg, kp);
	return kp;
}

/*pangolin::OpenGlMatrix Vision::getCurrentCameraPose(Mat T)
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


Mat Vision::CurrentCameraPose(Mat R_abs, Mat t_abs)
{
	Mat T_abs = Mat::eye(4,4,CV_32F);
	Mat camC = -R_abs.inv()*t_abs;
	camC.copyTo(T_abs.rowRange(0,3).col(3));
	R_abs.copyTo(T_abs.rowRange(0,3).colRange(0,3));
    cout << "T_abs =\n" << T_abs << endl;
    return T_abs;
}*/

Mat Vision::getBlock(Mat &img, Point2f &point, int window_size)
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

int Vision::getSSD(Mat &block_r, Mat &block_c)
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


vector<pair<int,int>> Vision::crossCheckMatching(vector <pair<int,int>> C2R, vector <pair<int,int>> R2C)
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


void Vision::getMatches(Mat img_1, Mat img_2, 
				vector<KeyPoint> keyP1, 
				vector<KeyPoint> keyP2,
				vector<pair<int,int>> matches)
{
	// TODO:
	
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
}

/*vector<pair<int,int>> Vision::getMatches(Mat &img_1, Mat &img_2, 
								vector<KeyPoint> &keyP1, 
								vector<KeyPoint> &keyP2)
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
}*/

void Vision::matching(Mat &img, vector<KeyPoint> &kp)
{
	if (!ref_kp.empty())
	{
		vector <pair<int,int>> matchedC2R, matchedR2C;
		thread t1(&Vision::getMatches, this, ref_img, img, ref_kp, kp, matchedC2R);
    	thread t2(&Vision::getMatches, this, img, ref_img, kp, ref_kp, matchedR2C);

    	t1.join();
    	t2.join();
    	
		cout 	<< "matches:\nC2R =\t"	<< matchedC2R.size()	
				<< "\tR2C =\t"			<< matchedR2C.size()
				<< endl;
	
	
		/*//cout << "proceed to:\nmatching...!" << endl;
		// 1. matched c2r
		// current is bigger loop
		//cout << "\n\nReference \t\t<<<---\t\t Current\n" << endl;
		vector <pair<int,int>> matchedC2R;
		matchedC2R = getMatches(ref_img, img, ref_kp, kp);
		cout << "matches C2R =\t"<<matchedC2R.size()<< endl;
	
		for (size_t k = 0; k < matchedC2R.size(); k++)
		{
			int parent 	= matchedC2R[k].first;
			int match 	= matchedC2R[k].second;
		}
	
		//	cout<<setfill('-')<<setw(80)<<"-"<<endl;

		// 2. matched r2c
		// ref is bigger loop
		//cout << "\n\nReference \t\t--->>>\t\t Current\n" << endl;
		vector <pair<int,int>> matchedR2C;
		matchedR2C = getMatches(img, ref_img, kp, ref_kp);
		cout << "matches R2C =\t"<<matchedR2C.size()<< endl;
	
		for (size_t k = 0; k < matchedR2C.size(); k++)
		{
			int parent 	= matchedR2C[k].first;
			int match 	= matchedR2C[k].second;
		}
	

		// 3. cross check matching
		vector <pair<int,int>> ccm;
	
		ccm = crossCheckMatching(matchedC2R, matchedR2C);
	
		cout << "Matches =\t" << ccm.size()<< endl;
	
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
		//cout << "----------------------------------------------------------------" << endl;*/
	}
}

}//namespace ORB_VISLAM
