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

namespace ORB_VISLAM
{
Vision::Vision(const string &settingFilePath,
						int win_sz, float ssd_th, float ssd_ratio_th, int minFeatures)
{
	cout << "" << endl;
	cout << "#########################################################################" << endl;
	cout << "\t\t\t\tVISION"															<< endl;
	cout << "#########################################################################" << endl;
	
	FileStorage fSettings(settingFilePath, FileStorage::READ);
    float fx 			= fSettings["Camera.fx"];
    float fy 			= fSettings["Camera.fy"];
    float cx 			= fSettings["Camera.cx"];
    float cy 			= fSettings["Camera.cy"];
	
	focal = fx;
	pp = Point2f(cx,cy);
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
    
	cout << "" << endl;
	cout << "---------------------" << endl;
	cout << "CAMERA PARAMETERS"		<< endl;
	cout << "---------------------" << endl;
	
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
	
	vWS 			= win_sz;
	vSSD_TH 		= ssd_th;
	vSSD_ratio_TH 	= ssd_ratio_th;
	vMIN_NUM_FEAT	= minFeatures;
	
	Mat identityMAT = Mat::eye(3, 3, CV_64F);
	Mat zeroMAT = Mat::zeros(3, 1, CV_64F);
	
	R_dec_prev 		= vector<Mat>{identityMAT, identityMAT, identityMAT, identityMAT};
	t_dec_prev 		= vector<Mat>{zeroMAT, zeroMAT, zeroMAT, zeroMAT};
	
	vrvec_dec 	= vector<Mat>{zeroMAT, zeroMAT, zeroMAT, zeroMAT};
	Rdec 		= vector<Mat>{identityMAT, identityMAT, identityMAT, identityMAT};
	tdec 		= vector<Mat>{zeroMAT, zeroMAT, zeroMAT, zeroMAT};
	
	//R_f 		= vector<Mat>{identityMAT, identityMAT, identityMAT, identityMAT};
	//R_f_prev 	= vector<Mat>{identityMAT, identityMAT, identityMAT, identityMAT};
	
	//t_f 		= vector<Mat>{zeroMAT, zeroMAT, zeroMAT, zeroMAT};
	//t_f_prev	= vector<Mat>{zeroMAT, zeroMAT, zeroMAT, zeroMAT};
	
	R_f_0 = Mat::eye(3, 3, CV_64F);
	R_f_1 = Mat::eye(3, 3, CV_64F);
	R_f_2 = Mat::eye(3, 3, CV_64F);
	R_f_3 = Mat::eye(3, 3, CV_64F);
	
	rvec_0 = Mat::zeros(3, 1, CV_64F);
	rvec_1 = Mat::zeros(3, 1, CV_64F);
	rvec_2 = Mat::zeros(3, 1, CV_64F);
	rvec_3 = Mat::zeros(3, 1, CV_64F);
	
	t_f_0 = Mat::zeros(3, 1, CV_64F);
	t_f_1 = Mat::zeros(3, 1, CV_64F);
	t_f_2 = Mat::zeros(3, 1, CV_64F);
	t_f_3 = Mat::zeros(3, 1, CV_64F);
	
	R_f_prev_0 = Mat::eye(3, 3, CV_64F);
	R_f_prev_1 = Mat::eye(3, 3, CV_64F);
	R_f_prev_2 = Mat::eye(3, 3, CV_64F);
	R_f_prev_3 = Mat::eye(3, 3, CV_64F);
	
	rvec_prev_0 = Mat::zeros(3, 1, CV_64F);
	rvec_prev_1 = Mat::zeros(3, 1, CV_64F);
	rvec_prev_2 = Mat::zeros(3, 1, CV_64F);
	rvec_prev_3 = Mat::zeros(3, 1, CV_64F);
	
	
	t_f_prev_0 = Mat::zeros(3, 1, CV_64F);
	t_f_prev_1 = Mat::zeros(3, 1, CV_64F);
	t_f_prev_2 = Mat::zeros(3, 1, CV_64F);
	t_f_prev_3 = Mat::zeros(3, 1, CV_64F);
}

void Vision::Analyze(Mat &rawImg, vector<KeyPoint> &kp, vector<pair<int,int>> &matches)
{
	kp = getKP(rawImg);
	matching(rawImg, kp, matches);
	ref_kp = kp;
	ref_img = rawImg;
}
			
vector<KeyPoint> Vision::getKP(Mat &rawImg)
{
	vector<KeyPoint> kp;
	Ptr<FeatureDetector> 		detector 	= ORB::create(350,1.2,8,31,0,2, ORB::FAST_SCORE,31,20);
	Ptr<DescriptorExtractor> 	extractor 	= ORB::create();
    
	detector->detect(rawImg, kp);
	return kp;
}

Mat Vision::getBlock(Mat &img, Point2f &point)
{
	Mat Block = Mat::zeros(vWS, vWS, CV_32F);
	int r = 0;
	int c = 0;
	for(int u = -vWS/2; u < vWS/2 + 1; u++)
	{
		for(int v = -vWS/2; v < vWS/2 + 1; v++)
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

float Vision::getSSD(Mat &block_1, Mat &block_2)
{
	float ssd = 0.0f;
	for (int xx = 0; xx < block_2.rows; xx++)
	{
		for(int yy = 0; yy < block_2.cols; yy++)
		{
			float df = block_2.at<float>(xx,yy) - block_1.at<float>(xx,yy);
			float dfSq = df*df/(vWS*vWS);
			ssd = ssd + dfSq;
		}
	}
	return ssd;
}

void Vision::getMatches(Mat img_1, vector<KeyPoint> keyP1, 
						Mat img_2, vector<KeyPoint> keyP2,
						vector<pair<int,int>> &matches)			
{
	Mat block_2 	= Mat::zeros(vWS, vWS, CV_32F);
	Mat block_1 	= Mat::zeros(vWS, vWS, CV_32F);
	vector<Point2f> kp1_vec;	
	for(size_t i = 0; i < keyP1.size(); i++)
	{
		kp1_vec.push_back(keyP1[i].pt);
		block_1 = getBlock(img_1, kp1_vec[i]);
		
		vector<float> ssd_tot;
		vector<float> ssd_unsorted_vec;
		vector<Point2f> kp2_vec;
		
		for (size_t j = 0; j < keyP2.size(); j++)
		{
			kp2_vec.push_back(keyP2[j].pt);
			block_2 = getBlock(img_2, kp2_vec[j]);
			
			float ssd = 0.0f;
			ssd = getSSD(block_1, block_2);
			
			ssd_tot.push_back(ssd);
		}
		
		//cout << "----------------------------------------------------" << endl;
		ssd_unsorted_vec = ssd_tot;
		
		// sort ssd_tot from less to high:
		sort(ssd_tot.begin(), ssd_tot.end());
		
		double ssd_ratio;
		ssd_ratio = static_cast<double>(ssd_tot[0])/static_cast<double>(ssd_tot[1]);
		
		if(ssd_tot[0] < vSSD_TH)
		{
			for (size_t k = 0; k < ssd_unsorted_vec.size(); k++)
			{
				if (ssd_unsorted_vec[k] == ssd_tot[0] && ssd_ratio < vSSD_ratio_TH)
				{
					matches.push_back(make_pair(i,k));
				}
			}
		}
	}
}

vector<pair<int,int>> Vision::crossCheckMatching(	vector<pair<int,int>> &m_12, 
													vector<pair<int,int>> &m_21)
{	
	vector<pair<int,int>> CrossCheckedMatches;
	for (size_t i = 0;  i < min(m_12.size(), m_21.size()); i++)
	{
		for (size_t j = 0;  j < max(m_12.size(), m_21.size()); j++)
		{
			if (m_12[j].second == m_21[i].first && m_12[j].first == m_21[i].second)
			{
				CrossCheckedMatches.push_back(make_pair(m_12[j].first, m_21[i].first));
			}
		}
	}
	return CrossCheckedMatches;
}

void Vision::matching(Mat &img, vector<KeyPoint> &kp, vector <pair<int,int>> &matches)
{
	if (!ref_kp.empty())
	{    	
		vector <pair<int,int>> matches12, matches21;
    	thread t1(&Vision::getMatches, this, ref_img, ref_kp, img, kp, ref(matches12));
    	thread t2(&Vision::getMatches, this, img, kp, ref_img, ref_kp, ref(matches21));

    	t1.join();
    	t2.join();
    	
		cout 	<< "Matching:\nIMAGE_1 \t VS. \t IMAGE_2 : \t"	<< matches12.size()	
				<< "\nIMAGE_2 \t VS. \t IMAGE_1 : \t"			<< matches21.size()
				<< endl;
	
		// 3. cross check matching
		matches = crossCheckMatching(matches12, matches21);
		cout << "Matches (CCM) =\t" << matches.size()<< endl;
		
		nmatches12 = matches12.size();
		nmatches21 = matches21.size();
		nmatchesCCM = matches.size();
		
		vector<Point2f> pt_ref;
		vector<Point2f> pt_matched;
	
		for (size_t i = 0; i < matches.size(); i++)
		{
			int parent 	= matches[i].first;
			int match 	= matches[i].second;
			
			pt_ref.push_back(ref_kp[parent].pt);
			pt_matched.push_back(kp[match].pt);
		}
		
		if (!pt_ref.empty() && !pt_matched.empty() 
							&& matches12.size() >= vMIN_NUM_FEAT 
							&& matches21.size() >= vMIN_NUM_FEAT
							&& matches.size() >= vMIN_NUM_FEAT)
		{
			/*Homography_Matrix = getHomography(pt_ref, pt_matched);
			cout << "\nH_own = \n" << Homography_Matrix << endl;*/
			
			Homography_Matrix = findHomography(pt_ref, pt_matched, RANSAC);
			cout << "\nH_RANSAC = \n" << Homography_Matrix << endl;
			cout << "----------------------------------------------------------------" << endl;
			calcPose(Homography_Matrix);
		}
	}
}

void Vision::setCurrentPose(Mat &R_, Mat &t_)
{
	//Mat center = -R_.inv()*t_;
	Mat center = t_;
	
	center.copyTo(T_cam.rowRange(0,3).col(3));
	R_.copyTo(T_cam.rowRange(0,3).colRange(0,3));
	
	cout << "\nT_cam =\n" << T_cam << endl;
}

Mat Vision::getHomography(const vector<Point2f> &p_ref, const vector<Point2f> &p_mtch)
{
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

void Vision::calcPose(Mat &homog)
{
	vector<Mat> Rs_decomp, ts_decomp, normals_decomp;
	
	int solutions = decomposeHomographyMat(homog, mK, Rs_decomp, ts_decomp, normals_decomp);
	
	//cout << "founded solutions = \t"<< solutions << endl;
	if (solutions >= 2)
	{
		Rdec = Rs_decomp;
		tdec = ts_decomp;
		
		R_f_0 = Rdec[0] * R_f_prev_0;
		Rodrigues(R_f_0, rvec_0);
		
		R_f_1 = Rdec[1] * R_f_prev_1;
		Rodrigues(R_f_1, rvec_1);
		
		R_f_2 = Rdec[2] * R_f_prev_2;
		Rodrigues(R_f_2, rvec_2);
		
		R_f_3 = Rdec[3] * R_f_prev_3;
		Rodrigues(R_f_3, rvec_3);
		
		t_f_0 = t_f_prev_0 + sc*(R_f_0*tdec[0]);
		t_f_1 = t_f_prev_1 + sc*(R_f_1*tdec[1]);
		t_f_2 = t_f_prev_2 + sc*(R_f_2*tdec[2]);
		t_f_3 = t_f_prev_3 + sc*(R_f_3*tdec[3]);
	}
	
	t_f_0.copyTo(T_cam_0.rowRange(0,3).col(3));
	R_f_0.copyTo(T_cam_0.rowRange(0,3).colRange(0,3));
	cout << "\nrvec_0 =\n" << rvec_0.t() << endl;

	
	//cout << "\nT_cam_0 =\n" << T_cam_0 << endl;

	t_f_1.copyTo(T_cam_1.rowRange(0,3).col(3));
	R_f_1.copyTo(T_cam_1.rowRange(0,3).colRange(0,3));
	cout << "\nrvec_1 =\n" << rvec_1.t() << endl;

	//cout << "\nT_cam_1 =\n" << T_cam_1 << endl;

	t_f_2.copyTo(T_cam_2.rowRange(0,3).col(3));
	R_f_2.copyTo(T_cam_2.rowRange(0,3).colRange(0,3));
	cout << "\nrvec_2 =\n" << rvec_2.t() << endl;

	//cout << "\nT_cam_2 =\n" << T_cam_2 << endl;

	t_f_3.copyTo(T_cam_3.rowRange(0,3).col(3));
	R_f_3.copyTo(T_cam_3.rowRange(0,3).colRange(0,3));
	cout << "\nrvec_3 =\n" << rvec_3.t() << endl;

	//cout << "\nT_cam_3 =\n" << T_cam_3 << endl;
	
	R_f_prev_0 	= R_f_0;
	rvec_prev_0 = rvec_0;

	R_f_prev_1 	= R_f_1;
	rvec_prev_1 = rvec_1;

	R_f_prev_2 	= R_f_2;
	rvec_prev_2 = rvec_2;

	R_f_prev_3 	= R_f_3;
	rvec_prev_3 = rvec_3;
	
	
	t_f_prev_0 = t_f_0;
	t_f_prev_1 = t_f_1;
	t_f_prev_2 = t_f_2;
	t_f_prev_3 = t_f_3;
	
	
	R_dec_prev 	= Rdec;
	t_dec_prev 	= tdec;
}

}//namespace ORB_VISLAM
