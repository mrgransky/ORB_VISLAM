#include "Vision.h"

using namespace std;
using namespace cv;

using namespace pcl;
using namespace pcl::io;
using namespace pcl::visualization;

namespace ORB_VISLAM{

Vision::Vision(const string &settingFilePath,
						int win_sz, float ssd_th, float ssd_ratio_th, 
						size_t minFeatures, float minScale, float distTo3DPts, bool downScale){
	cout << "" << endl;
	cout << "#########################################################################" << endl;
	cout << "\t\t\t\tVISION"															<< endl;
	cout << "#########################################################################" << endl;
	
	FileStorage fSettings(settingFilePath, FileStorage::READ);
    float fx 			= fSettings["Camera.fx"];
    float fy 			= fSettings["Camera.fy"];
    float cx 			= fSettings["Camera.cx"];
    float cy 			= fSettings["Camera.cy"];
	
	foc = fx;
	
	pp = Point2f(cx,cy);
	fps 				= fSettings["Camera.fps"];
	vMIN_SCALE = minScale;
	Mat K = Mat::eye(3, 3, CV_32F);
	
	K.at<float>(0,0) 		= fx;
    K.at<float>(1,1) 		= fy;
    K.at<float>(0,2) 		= cx;
    K.at<float>(1,2) 		= cy;

	K.copyTo(mK);
	Mat tmp_inv = mK.inv();
	tmp_inv.copyTo(mK_inv);
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
    
    if(DistCoef.rows == 5) cout << "- k3: " << DistCoef.at<float>(4) << endl;
    
    cout << "- p1: " << DistCoef.at<float>(2) 	<< endl;
    cout << "- p2: " << DistCoef.at<float>(3) 	<< endl;
	cout << "- FPS:" <<	fps						<< endl;
	
	vdistTh_3DPts = distTo3DPts;
	vDownScaled = downScale;
	if (vDownScaled) cout << "Raw Frames being down Scaled..." << endl;
	
	IMG_ = cv::Mat::zeros(fSettings["Camera.height"], fSettings["Camera.width"], CV_8UC3);
	
	vWS 			= win_sz;
	vSSD_TH 		= ssd_th;
	vSSD_ratio_TH 	= ssd_ratio_th;
	vMIN_NUM_FEAT	= minFeatures;
	
	I_3x3 	= Mat::eye(3, 3, CV_32F);
	I_4x4 	= Mat::eye(4, 4, CV_32F);	
	Z_3x1 	= Mat::zeros(3, 1, CV_32F);
	
	T_local = vector<Mat>{I_4x4, I_4x4, I_4x4, I_4x4};
	T_f 	= vector<Mat>{I_4x4, I_4x4, I_4x4, I_4x4};
	
	R_f 		= vector<Mat>{I_3x3, I_3x3, I_3x3, I_3x3};
	R_f_prev 	= vector<Mat>{I_3x3, I_3x3, I_3x3, I_3x3};
	
	t_f 		= vector<Mat>{Z_3x1, Z_3x1, Z_3x1, Z_3x1};
	t_f_prev 	= vector<Mat>{Z_3x1, Z_3x1, Z_3x1, Z_3x1};
	
	cloud = boost::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
		
	R_f_E = Mat::eye(3, 3, CV_32F);
	R_f_0 = Mat::eye(3, 3, CV_32F);
	R_f_1 = Mat::eye(3, 3, CV_32F);
	R_f_2 = Mat::eye(3, 3, CV_32F);
	R_f_3 = Mat::eye(3, 3, CV_32F);

	rvec_E = Mat::zeros(3, 1, CV_32F);
	rvec_0 = Mat::zeros(3, 1, CV_32F);
	rvec_1 = Mat::zeros(3, 1, CV_32F);
	rvec_2 = Mat::zeros(3, 1, CV_32F);
	rvec_3 = Mat::zeros(3, 1, CV_32F);
	
	rvec_loc_E = Mat::zeros(3, 1, CV_32F);
	rvec_loc_0 = Mat::zeros(3, 1, CV_32F);
	rvec_loc_1 = Mat::zeros(3, 1, CV_32F);
	rvec_loc_2 = Mat::zeros(3, 1, CV_32F);
	rvec_loc_3 = Mat::zeros(3, 1, CV_32F);
	
	t_f_E = Mat::zeros(3, 1, CV_32F);
	t_f_0 = Mat::zeros(3, 1, CV_32F);
	t_f_1 = Mat::zeros(3, 1, CV_32F);
	t_f_2 = Mat::zeros(3, 1, CV_32F);
	t_f_3 = Mat::zeros(3, 1, CV_32F);
	
	R_f_prev_E = Mat::eye(3, 3, CV_32F);
	R_f_prev_0 = Mat::eye(3, 3, CV_32F);
	R_f_prev_1 = Mat::eye(3, 3, CV_32F);
	R_f_prev_2 = Mat::eye(3, 3, CV_32F);
	R_f_prev_3 = Mat::eye(3, 3, CV_32F);
	
	rvec_prev_E = Mat::zeros(3, 1, CV_32F);
	rvec_prev_0 = Mat::zeros(3, 1, CV_32F);
	rvec_prev_1 = Mat::zeros(3, 1, CV_32F);
	rvec_prev_2 = Mat::zeros(3, 1, CV_32F);
	rvec_prev_3 = Mat::zeros(3, 1, CV_32F);
	
	t_f_prev_E = Mat::zeros(3, 1, CV_32F);
	t_f_prev_0 = Mat::zeros(3, 1, CV_32F);
	t_f_prev_1 = Mat::zeros(3, 1, CV_32F);
	t_f_prev_2 = Mat::zeros(3, 1, CV_32F);
	t_f_prev_3 = Mat::zeros(3, 1, CV_32F);
}

void Vision::Analyze(Mat &rawImg, vector<KeyPoint> &kp, 
						vector<pair<int,int>> &matches)
{
	int tgt_width = 600; // pixel
	// TODO: downscale raw images:
	if (vDownScaled && (rawImg.cols > 2*tgt_width))
	{
		Mat tempDS = rawImg.clone();
		pyrDown(tempDS, rawImg, Size(tempDS.cols/2, tempDS.rows/2));
		cout << "raw Img r,c =\t" << rawImg.rows << " , " << rawImg.cols<< endl;
	}
	
	Mat temp = rawImg.clone();
	undistort(temp, rawImg, mK, mDistCoef);
	
	Mat descriptor;
	get_AKAZE_kp(rawImg, kp, descriptor);
	//get_ORB_kp(rawImg, kp, descriptor);
	
	
	//matching(rawImg, kp, matches);
	matching(rawImg, kp, descriptor, matches);
	
	ref_kp 		= kp;
	ref_desc 	= descriptor;
	ref_img 	= rawImg;
}

void Vision::setCurrentPose(Mat &R_, Mat &t_, Mat &T_)
{
	t_.copyTo(T_.rowRange(0,3).col(3));
	R_.copyTo(T_.rowRange(0,3).colRange(0,3));
}

void Vision::get_ORB_kp(Mat &rawImg, vector<KeyPoint> &kp, Mat &desc)
{
	Ptr<ORB> 					feature		= ORB::create();
    kp.clear();
    
    // TODO: mask must be changed to cv::Mat to manipulate KP:
    // https://stackoverflow.com/questions/42346761/opencv-python-feature-detection-how-to-provide-a-mask-sift
	vector<uchar> kpMASK;
	feature->detectAndCompute(rawImg, kpMASK, kp, desc);
	
	//int nonZero = countNonZero(kpMASK);
	//cout << "kpMasK nonZero =\t" << nonZero << endl;
	
	
}

void Vision::get_AKAZE_kp(Mat &rawImg, vector<KeyPoint> &kp, Mat &desc)
{
	Ptr<AKAZE> 					feature 	= AKAZE::create();
	kp.clear();
	// TODO: mask must be changed to cv::Mat to manipulate KP:
    // https://stackoverflow.com/questions/42346761/opencv-python-feature-detection-how-to-provide-a-mask-sift
	vector<uchar> kpMASK;
	feature->detectAndCompute(rawImg, kpMASK, kp, desc);
	
	//int nonZero = countNonZero(kpMASK);
	//cout << "kpMasK nonZero =\t" << nonZero << endl;
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

void Vision::matching(Mat &img, vector<KeyPoint> &kp, Mat &desc, 
					vector<pair<int,int>> &match_idx)
{	
	Ptr<DescriptorMatcher> matcher 	= DescriptorMatcher::create("BruteForce-Hamming");
	
	if (!ref_kp.empty())
	{
		vector<vector<DMatch>> all_matches;
		vector<Point2f> src, dst;
		vector<int> ref_kp_idx, kp_idx;
		
		matcher->knnMatch(ref_desc, desc, all_matches, 2);
		for(size_t i = 0; i < all_matches.size(); i++) 
		{
			DMatch first = all_matches[i][0];
			float dist1 = all_matches[i][0].distance;
			float dist2 = all_matches[i][1].distance;
			if(dist1 < nn_match_ratio * dist2) 
			{
				ref_kp_idx.push_back(first.queryIdx);
				kp_idx.push_back(first.trainIdx);
			
				src.push_back(ref_kp[first.queryIdx].pt);
				dst.push_back(kp[first.trainIdx].pt);
			}
		}
		getDummy(src, dst, ref_kp_idx, kp_idx, all_matches, match_idx);
		//essential_matrix_inliers(src, dst, ref_kp_idx, kp_idx, all_matches, match_idx);
		//homography_matrix_inliers(src, dst, ref_kp_idx, kp_idx, all_matches, match_idx);
	}
}

void Vision::getDummy(vector<Point2f> &src, vector<Point2f> &dst,
										vector<int> &ref_kp_idx, vector<int> &kp_idx,
										vector<vector<DMatch>> &all_matches,
										vector<pair<int,int>> &match_idx)
{
	double minVal,maxVal;
	cv::minMaxIdx(src, &minVal, &maxVal);
	vector<Point2f> srcInlier, dstInlier;
	
	//double minDis2EpiLine = 0.006 * maxVal;
	double minDis2EpiLine = .5; 
	double confid = .99;
	vector<uchar> mask;
	
	Mat E = findEssentialMat(dst, src, mK, RANSAC, confid, minDis2EpiLine, mask);
	
	int inliers_num = countNonZero(mask);
	for(size_t nm = 0; nm < mask.size(); nm++)
	{
		if(mask[nm])
		{
			match_idx.push_back(make_pair(ref_kp_idx[nm], kp_idx[nm]));
			srcInlier.push_back(src[nm]);
			dstInlier.push_back(dst[nm]);
		}
	}
	
	cout << "2D Matches:\n"; 
	cout << "\tALL\tMASK\tINLIERS\n";
	cout 	<< '\t' << all_matches.size() 	<< '\t' 	<< mask.size() 
			<< '\t' << inliers_num 			<< '\n';

	if (!srcInlier.empty() && !dstInlier.empty() && match_idx.size() >= vMIN_NUM_FEAT)
	{
		GetPose(srcInlier, dstInlier, E);		
		cout << "----------------------------------------------------------------" << endl;
	}
}

void Vision::essential_matrix_inliers(vector<Point2f> &src, vector<Point2f> &dst,
										vector<int> &ref_kp_idx, vector<int> &kp_idx,
										vector<vector<DMatch>> &all_matches,
										vector<pair<int,int>> &match_idx)
{
	double minVal,maxVal;
	cv::minMaxIdx(src, &minVal, &maxVal);
	vector<Point2f> srcInlier, dstInlier;
	
	//double minDis2EpiLine = 0.006 * maxVal;
	double minDis2EpiLine = .5; 
	double confid = .99;
	vector<uchar> mask;
	
	Mat E = findEssentialMat(dst, src, mK, RANSAC, confid, minDis2EpiLine, mask);
	
	int inliers_num = countNonZero(mask);
	for(size_t nm = 0; nm < mask.size(); nm++)
	{
		if(mask[nm])
		{
			match_idx.push_back(make_pair(ref_kp_idx[nm], kp_idx[nm]));
			srcInlier.push_back(src[nm]);
			dstInlier.push_back(dst[nm]);
		}
	}
	
	cout << "2D Matches:\n"; 
	cout << "\tALL\tMASK\tINLIERS\n";
	cout 	<< '\t' << all_matches.size() 	<< '\t' 	<< mask.size() 
			<< '\t' << inliers_num 			<< '\n';

	if (!srcInlier.empty() && !dstInlier.empty() && match_idx.size() >= vMIN_NUM_FEAT)
	{
		PoseFromEssentialMatrix(srcInlier, dstInlier, E);
		cout << "----------------------------------------------------------------" << endl;
	}
}

void Vision::GetPose(vector<Point2f> &src, vector<Point2f> &dst, Mat &E)
{		
	Mat R1, R2, t1, t2, R_local, t_local;

	vector<float> good_vec(4);
	vector<Mat> Rt_vec(4);
	
	// 1. Decompose Essential Matrix to obtain R1, R2, t:
	decomE(E, R1, R2, t1, t2);
	
	SetR_t(R1, t1, Rt_vec[0]);
	SetR_t(R2, t1, Rt_vec[1]);
	SetR_t(R1, t2, Rt_vec[2]);	
	SetR_t(R2, t2, Rt_vec[3]);
	
	for(size_t i = 0; i < Rt_vec.size(); i++)
	{
		triangulateMyPoints(src, dst, Rt_vec[i], good_vec[i]);
	}
	
	cout << "\nFront 3D Pts:\n"; 
	cout << "\tSOL_0\tSOL_1\tSOL_2\tSOL_3\n";
	cout 	<< '\t' << setprecision(3) << good_vec[0] 
			<< '\t' << setprecision(3) << good_vec[1] 
			<< '\t' << setprecision(3) << good_vec[2] 
			<< '\t' << setprecision(3) << good_vec[3] 
			<< '\n';
	
	// 2. Obtain correct Pose: // cheirality constraint:
	ChooseCorrectPose(good_vec, Rt_vec, R_local, t_local);
	
	if(t_local.at<float>(2) < 0)
	{
		t_local.at<float>(2) *= -1.0f;
	}
	setCurrentPose(R_local, t_local, T_loc_0);
	//cout << "\nT_loc = \n" << T_loc_0 << endl;
	
	if (sc < vMIN_SCALE ||	t_local.at<float>(2) < t_local.at<float>(0) ||
							t_local.at<float>(2) < t_local.at<float>(1))
	{
		cerr 	<< "\n\n\n#########################ERROR:#########################\n" 
				<< "\n\nconstraints NOT fullfiled!!!\n"
				<< endl;
		
		cout << "scale = " 		<< sc			<< endl;
		cout << "t_local = \t" 	<< t_local.t()	<< endl;
	}
	if (sc > vMIN_SCALE &&	t_local.at<float>(2,0) > t_local.at<float>(0,0) &&
							t_local.at<float>(2,0) > t_local.at<float>(1,0))
	{
		t_f_0 = t_f_prev_0 + sc*(R_f_0 * t_local);
		R_f_0 = R_f_prev_0 * R_local;
	}
	Rodrigues(R_f_0, rvec_0);
	setCurrentPose(R_f_0, t_f_0, T_cam_0);
	//cout << "\nT_0 = \n" << T_cam_0 << endl;
	R_f_prev_0 		= R_f_0.clone();
	rvec_prev_0 	= rvec_0.clone();	
	t_f_prev_0 		= t_f_0.clone();
}

void Vision::ChooseCorrectPose(vector<float> &good_vec, vector<Mat> &Rt_vec, Mat &R_, Mat &t_)
{
	R_.create(3, 3, CV_32F);
	t_.create(3, 1, CV_32F);
	
	if( good_vec[0] >= good_vec[1] && 
		good_vec[0] >= good_vec[2] && 
		good_vec[0] >= good_vec[3])
	{
		front3DPtsOWN = good_vec[0];
		//cout << "sol_0 is correct!" << endl;
		Rt_vec[0].rowRange(0,3).colRange(0,3).copyTo(R_);
		Rt_vec[0].col(3).copyTo(t_);
	}
	else if(good_vec[1] >= good_vec[0] && 
			good_vec[1] >= good_vec[2] && 
			good_vec[1] >= good_vec[3])
	{
		front3DPtsOWN = good_vec[1];
		//cout << "sol_1 is correct!" << endl;
		Rt_vec[1].rowRange(0,3).colRange(0,3).copyTo(R_);
		Rt_vec[1].col(3).copyTo(t_);
	}
	else if(good_vec[2] >= good_vec[0] && 
			good_vec[2] >= good_vec[1] && 
			good_vec[2] >= good_vec[3])
	{
		front3DPtsOWN = good_vec[2];
		//cout << "sol_2 is correct!" << endl;
		Rt_vec[2].rowRange(0,3).colRange(0,3).copyTo(R_);
		//Rt_vec[2].col(3).copyTo(t_);
		Mat TEM = Rt_vec[2].col(3) * -1;
		TEM.copyTo(t_);
	}
	else if(good_vec[3] >= good_vec[0] && 
			good_vec[3] >= good_vec[1] && 
			good_vec[3] >= good_vec[2])
	{
		front3DPtsOWN = good_vec[2];
		//cout << "sol_3 is correct!" << endl;
		Rt_vec[3].rowRange(0,3).colRange(0,3).copyTo(R_);
		//Rt_vec[3].col(3).copyTo(t_);
		Mat TEMP_ = Rt_vec[3].col(3) * -1;
		TEMP_.copyTo(t_);
	}
	else
	{
		cout << "No SOLUTION FOUND!!!" << endl;
		exit(EXIT_FAILURE);
	}
}


void Vision::get_correct_pose(vector<Point2f> &src, vector<Point2f> &dst, 
								Mat &R1, Mat &R2, Mat &t, Mat &R_correct, Mat &t_correct)
{

	R_correct.create(3, 3, CV_32F);
	t_correct.create(3, 1, CV_32F);
	// Normalize Points:
	vector<Point2f> src_norm, dst_norm;
	Normalize2DPts(src, dst, src_norm, dst_norm);
	
	/*vector<Point2f> src_norm2, dst_norm2;
	for(size_t i = 0; i < src.size(); i++)
	{
		float x = (src[i].x - pp.x)/foc;
		float y = (src[i].y - pp.y)/foc;
		
		src_norm2.push_back(Point2f(x,y));
		
		float xx = (dst[i].x - pp.x)/foc;
		float yy = (dst[i].y - pp.y)/foc;
		
		dst_norm2.push_back(Point2f(xx,yy));
	}
	
	for(size_t i = 0; i < src_norm.size(); i++)
	{
		cout << "\n\nsrc_norm1 [" <<i << "] = "<< src_norm[i]<< endl;
		cout << "src_norm2 [" <<i << "] = "<< src_norm2[i]<< endl;
	
		cout << "\n\ndst_norm1 [" <<i << "] = "<< dst_norm[i]<< endl;
		cout << "dst_norm2 [" <<i << "] = "<< dst_norm2[i]<< endl;
		cout << "----------------------------------------" << endl;
	
	}*/
	
	// P0, P1, P2, P3, P4:
	Mat P0 = Mat::eye(3, 4, R1.type());
    Mat P1(3, 4, R1.type()), P2(3, 4, R1.type()), P3(3, 4, R1.type()), P4(3, 4, R1.type());
    P1(Range::all(), Range(0, 3)) = R1 * 1.0; P1.col(3) = t * 1.0;
    P2(Range::all(), Range(0, 3)) = R2 * 1.0; P2.col(3) = t * 1.0;
    P3(Range::all(), Range(0, 3)) = R1 * 1.0; P3.col(3) = -t * 1.0;
    P4(Range::all(), Range(0, 3)) = R2 * 1.0; P4.col(3) = -t * 1.0;

	Mat Q;
	
	// triangulation sol_1:
	triangulatePoints(P0, P1, src_norm, dst_norm, Q);
	Mat mask1 = Q.row(2).mul(Q.row(3)) > 0;
	Q.row(0) /= Q.row(3);
	Q.row(1) /= Q.row(3);
	Q.row(2) /= Q.row(3);
	Q.row(3) /= Q.row(3);
	mask1 = (Q.row(2) < 50) & mask1;
	Q = P1 * Q;
	mask1 = (Q.row(2) > 0) & mask1;
	mask1 = (Q.row(2) < 50) & mask1;

	// triangulation sol_2:
	triangulatePoints(P0, P2, src_norm, dst_norm, Q);
	Mat mask2 = Q.row(2).mul(Q.row(3)) > 0;
	Q.row(0) /= Q.row(3);
	Q.row(1) /= Q.row(3);
	Q.row(2) /= Q.row(3);
	Q.row(3) /= Q.row(3);
	mask2 = (Q.row(2) < 50) & mask2;
	Q = P2 * Q;
	mask2 = (Q.row(2) > 0) & mask2;
	mask2 = (Q.row(2) < 50) & mask2;

	// triangulation sol_3:
	triangulatePoints(P0, P3, src_norm, dst_norm, Q);
	Mat mask3 = Q.row(2).mul(Q.row(3)) > 0;
	Q.row(0) /= Q.row(3);
	Q.row(1) /= Q.row(3);
	Q.row(2) /= Q.row(3);
	Q.row(3) /= Q.row(3);
	mask3 = (Q.row(2) < 50) & mask3;
	Q = P3 * Q;
	mask3 = (Q.row(2) > 0) & mask3;
	mask3 = (Q.row(2) < 50) & mask3;

	// triangulation sol_4:
	triangulatePoints(P0, P4, src_norm, dst_norm, Q);
	Mat mask4 = Q.row(2).mul(Q.row(3)) > 0;
	Q.row(0) /= Q.row(3);
	Q.row(1) /= Q.row(3);
	Q.row(2) /= Q.row(3);
	Q.row(3) /= Q.row(3);
	mask4 = (Q.row(2) < 50) & mask4;
	Q = P4 * Q;
	mask4 = (Q.row(2) > 0) & mask4;
	mask4 = (Q.row(2) < 50) & mask4;


	mask1 = mask1.t();
	mask2 = mask2.t();
	mask3 = mask3.t();
	mask4 = mask4.t();

	int good1 = countNonZero(mask1);
	int good2 = countNonZero(mask2);
	int good3 = countNonZero(mask3);
	int good4 = countNonZero(mask4);

	cout << "\nGOOD:\n"; 
	cout << "\tSOL_0\tSOL_1\tSOL_2\tSOL_3\n";
	cout 	<< '\t' << good1 
			<< '\t' << good2 
			<< '\t' << good3 
			<< '\t' << good4 
			<< '\n';

	if (good1 >= good2 && good1 >= good3 && good1 >= good4)
	{
		cout << "sol_0 correct!" << endl;
		R1.copyTo(R_correct);
		t.copyTo(t_correct);
	}
	else if (good2 >= good1 && good2 >= good3 && good2 >= good4)
	{
		cout << "sol_1 correct!" << endl;
		R2.copyTo(R_correct);
		t.copyTo(t_correct);
	}
	else if (good3 >= good1 && good3 >= good2 && good3 >= good4)
	{
		cout << "sol_2 correct!" << endl;
		t = -t;
		R1.copyTo(R_correct);
		t.copyTo(t_correct);
	}
	else
	{
		cout << "sol_3 correct!" << endl;
		t = -t;
		R2.copyTo(R_correct);
		t.copyTo(t_correct);
	}
}

void Vision::calcGlobalPose(Mat &R_local, Mat &t_local)
{
	// TODO: T_CAM must be the output??????????????!!
	// FIX it later---
	if (sc > vMIN_SCALE &&	t_local.at<float>(2,0) > t_local.at<float>(0,0) &&
							t_local.at<float>(2,0) > t_local.at<float>(1,0))
	{
		/*Reconstruction(src, dst, Rt_prev, Rt, p3D_loc);
		Mat visLOCAL(3, p3D_loc.size(), CV_32F);
		for (size_t i = 0; i < p3D_loc.size(); i++)
		{
			p3D_loc[i].copyTo(visLOCAL.col(i));
		}*/
		t_f_E = t_f_prev_E + sc*(R_f_E * t_local);
		R_f_E = R_f_prev_E * R_local;
		//R_f_E = R_local * R_f_prev_E;
		//visLOCAL.copyTo(visionMap);
	}
	Rodrigues(R_f_E, rvec_E);
	setCurrentPose(R_f_E, t_f_E, T_cam_E);
	R_f_prev_E 		= R_f_E.clone();
	rvec_prev_E 	= rvec_E.clone();	
	t_f_prev_E 		= t_f_E.clone();
	Rt_prev			= Rt.clone();
	
}

void Vision::homography_matrix_inliers(vector<Point2f> &src, vector<Point2f> &dst, 
										vector<int> &ref_kp_idx, vector<int> &kp_idx, 
										vector<vector<DMatch>> &all_matches,
										vector<pair<int,int>> &match_idx)
{
	const double ransac_thresh = 1.0f; 
	vector<uchar> inlier_mask;
	vector<Point2f> srcInlier, dstInlier;
	
	Mat H = findHomography(dst, src, RANSAC, ransac_thresh, inlier_mask);
	
	int inliers_num = countNonZero(inlier_mask);
	for(size_t nm = 0; nm < inlier_mask.size(); nm++)
	{
		if(inlier_mask[nm])
		{
			match_idx.push_back(make_pair(ref_kp_idx[nm], kp_idx[nm]));
			
			srcInlier.push_back(src[nm]);
			dstInlier.push_back(dst[nm]);
		}
	}
	cout 	<< "2D Matches:\n"; 
	cout 	<< "\tALL\tMASK\tINLIERS\n";
	cout 	<< '\t' << all_matches.size() << '\t' 
			<< inlier_mask.size() << '\t' 
			<< inliers_num << '\n';
		
	if (!srcInlier.empty() && !dstInlier.empty() && match_idx.size() >= vMIN_NUM_FEAT)
	{
		PoseFromHomographyMatrix(srcInlier, dstInlier, H);
		cout << "----------------------------------------------------------------" << endl;
	}
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
			//PoseFromHomographyMatrix(pt_ref, pt_matched);
			//PoseFromEssentialMatrix(pt_ref, pt_matched);
			cout << "----------------------------------------------------------------" << endl;
		}
	}
}

void Vision::PoseFromEssentialMatrix(vector<Point2f> &src, vector<Point2f> &dst, Mat &E)
{

	Mat R_local, t_local, mask;
	vector<Mat> p3D_loc;
	
	int frontiers = recoverPose(E, dst, src, mK, R_local, t_local, mask);
	front3DPtsOPCV = ((float)frontiers) / ((float)src.size());
	cout << "front 3DPts OPENCV = \t" << front3DPtsOPCV << endl;
	
	R_local.convertTo(R_local, CV_32F);
	t_local.convertTo(t_local, CV_32F);
	
	Rodrigues(R_local, rvec_loc_E);
	setCurrentPose(R_local, t_local, T_loc_E);
	
	SetR_t(R_local, t_local, Rt);
	//cout << "\nT_loc_E = \n" << T_loc_E << endl;
	if (sc < vMIN_SCALE ||	fabs(t_local.at<float>(2,0)) < t_local.at<float>(0,0) ||
							fabs(t_local.at<float>(2,0)) < t_local.at<float>(1,0))
	{
		cerr 	<< "\n\n*******************************ERROR:********************************\n" 
				<< "constraints NOT fullfiled!!!\n"
				<< endl;
		cout << "scale = " <<sc<< endl;
		cout << "t_local = \t" << t_local.t()<< endl;
		//exit(EXIT_FAILURE);
	}
	if (sc > vMIN_SCALE &&	fabs(t_local.at<float>(2,0)) > t_local.at<float>(0,0) &&
							fabs(t_local.at<float>(2,0)) > t_local.at<float>(1,0))
	{
		Reconstruction(src, dst, Rt_prev, Rt, p3D_loc);
		Mat visLOCAL(3, p3D_loc.size(), CV_32F);
		for (size_t i = 0; i < p3D_loc.size(); i++)
		{
			p3D_loc[i].copyTo(visLOCAL.col(i));
		}
		t_f_E = sc*(R_f_E * t_local) + t_f_prev_E;
		R_f_E = R_f_prev_E * R_local;
		visLOCAL.copyTo(visionMap);
	}
	Rodrigues(R_f_E, rvec_E);
	setCurrentPose(R_f_E, t_f_E, T_cam_E);
	//cout << "\nT_E = \n" << T_cam_E << endl;
	
	R_f_prev_E 		= R_f_E.clone();
	rvec_prev_E 	= rvec_E.clone();	
	t_f_prev_E 		= t_f_E.clone();
	Rt_prev			= Rt.clone(); // TODO: this must be commented in theory eye(3,4)
}

void Vision::Normalize2DPts(vector<Point2f> &src, vector<Point2f> &dst,
							vector<Mat> &src_normalized, vector<Mat> &dst_normalized)
{
	for (size_t i = 0; i < src.size(); i++)
	{
		Mat src_u(3, 1, CV_32F);
		Mat dst_u(3, 1, CV_32F);
		
		pt2D_to_mat(src[i], src_u);
		pt2D_to_mat(dst[i], dst_u);
		
		Mat src_n = mK_inv * src_u;
		Mat dst_n = mK_inv * dst_u;
		
		src_normalized.push_back(src_n);
		dst_normalized.push_back(dst_n);
	}
}

void Vision::Normalize2DPts(vector<Point2f> &src, vector<Point2f> &dst,
							vector<Point2f> &src_normalized, vector<Point2f> &dst_normalized)
{
	for (size_t i = 0; i < src.size(); i++)
	{
		Mat src_u(3, 1, CV_32F);
		Mat dst_u(3, 1, CV_32F);
		pt2D_to_mat(src[i], src_u);
		pt2D_to_mat(dst[i], dst_u);
		
		Mat src_n_matrix = mK_inv * src_u;
		Mat dst_n_matrix = mK_inv * dst_u;
		
		Point2f src_n = Point2f(src_n_matrix.rowRange(0,2));
		Point2f dst_n = Point2f(dst_n_matrix.rowRange(0,2));
		
		src_normalized.push_back(src_n);
		dst_normalized.push_back(dst_n);
	}
}

void Vision::triangulateMyPoints(vector<Point2f> &src, vector<Point2f> &dst, 
									Mat &Rt, float &acceptedPts)
{
	vector<Mat> src_norm, dst_norm;
	Normalize2DPts(src, dst, src_norm, dst_norm);
	Mat pt3D_cam0(3, src_norm.size(), CV_32F), pt3D_cam1(3, dst_norm.size(), CV_32F);
	Mat Pts4D;
	Extract3DPts(src_norm, dst_norm, Rt, Pts4D);
	
	// paper: Nister:
	// cam_0:
	Mat constraints = Pts4D.row(2).mul(Pts4D.row(3)) > 0;
	
	Pts4D.row(0) /= Pts4D.row(3);
	Pts4D.row(1) /= Pts4D.row(3);
	Pts4D.row(2) /= Pts4D.row(3);
	Pts4D.row(3) /= Pts4D.row(3);
	
	constraints = (Pts4D.row(2) < vdistTh_3DPts) & constraints; 
	
	// cam_1:
	Mat Pts4D_cam1 = Rt * Pts4D;
	constraints = (Pts4D_cam1.row(2) > 0) & constraints;
	constraints = (Pts4D_cam1.row(2) < vdistTh_3DPts) & constraints;
	
	int nz3d = countNonZero(constraints);
	acceptedPts = ((float)nz3d / (float)constraints.cols);
}

void Vision::Extract3DPts(vector<Mat> &src, vector<Mat> &dst, Mat &Rt, Mat &Points4D)
{
	Points4D.create(4, src.size(), Rt.type());
	Mat P_prev 	= Rt_prev.clone();
	Mat P 		= Rt.clone();
	//cout << "Pprev =\n" << P_prev << "\nP =\n" << P << endl;
	for (size_t i = 0; i < src.size(); i++)
	{
		Mat pt3D_cam0 = Mat::zeros(3, 1, CV_32F);
		Mat pt3D_cam1 = Mat::zeros(3, 1, CV_32F);
		
		Mat A(6, 4, CV_32F);
		Mat src_skew(3, 3, CV_32F);
		Mat dst_skew(3, 3, CV_32F);
		
		getSkewMatrix(src[i], src_skew);
		getSkewMatrix(dst[i], dst_skew);
		
		Mat src_skewXP = src_skew * P_prev;
		Mat dst_skewXP = dst_skew * P;
		
		A.at<float>(0,0) = src_skewXP.at<float>(0,0);
		A.at<float>(1,0) = src_skewXP.at<float>(1,0);
		A.at<float>(2,0) = src_skewXP.at<float>(2,0);
		A.at<float>(3,0) = dst_skewXP.at<float>(0,0);
		A.at<float>(4,0) = dst_skewXP.at<float>(1,0);
		A.at<float>(5,0) = dst_skewXP.at<float>(2,0);
		
		A.at<float>(0,1) = src_skewXP.at<float>(0,1);
		A.at<float>(1,1) = src_skewXP.at<float>(1,1);
		A.at<float>(2,1) = src_skewXP.at<float>(2,1);
		A.at<float>(3,1) = dst_skewXP.at<float>(0,1);
		A.at<float>(4,1) = dst_skewXP.at<float>(1,1);
		A.at<float>(5,1) = dst_skewXP.at<float>(2,1);

		A.at<float>(0,2) = src_skewXP.at<float>(0,2);
		A.at<float>(1,2) = src_skewXP.at<float>(1,2);
		A.at<float>(2,2) = src_skewXP.at<float>(2,2);
		A.at<float>(3,2) = dst_skewXP.at<float>(0,2);
		A.at<float>(4,2) = dst_skewXP.at<float>(1,2);
		A.at<float>(5,2) = dst_skewXP.at<float>(2,2);

		A.at<float>(0,3) = src_skewXP.at<float>(0,3);
		A.at<float>(1,3) = src_skewXP.at<float>(1,3);
		A.at<float>(2,3) = src_skewXP.at<float>(2,3);
		A.at<float>(3,3) = dst_skewXP.at<float>(0,3);
		A.at<float>(4,3) = dst_skewXP.at<float>(1,3);
		A.at<float>(5,3) = dst_skewXP.at<float>(2,3);
		
		Mat u,w,vt;
		cv::SVDecomp(A,w,u,vt, cv::SVD::FULL_UV);
		vt.row(3).reshape(0,4).copyTo(Points4D.col(i));
	}
}

void Vision::decomE(Mat &E, Mat &R1, Mat &R2, Mat &t1, Mat &t2)
{
	R1.create(3, 3, CV_32F);
	R2.create(3, 3, CV_32F);
	t1.create(3, 1, CV_32F);
	t2.create(3, 1, CV_32F);

	// Textbook Implementation: page 258: W
	Mat u,w,vt;
	cv::SVDecomp(E,w,u,vt, cv::SVD::FULL_UV);
	
	w.convertTo(w, CV_32F);
	u.convertTo(u, CV_32F);
	vt.convertTo(vt, CV_32F);

	if (determinant(u) < 0) u *= -1.0f;
    if (determinant(vt) < 0) vt *= -1.0f;
	
	Mat W = Mat::zeros(3, 3, CV_32F);
	
	W.at<float>(0,1) = -1;
	W.at<float>(1,0) = 1;
	W.at<float>(2,2) = 1;
	
	R1 = u * W 		* vt;
	R2 = u * W.t() 	* vt;
	
	t1 = u.col(2);
	t2 = -1.0 * u.col(2);
}

void Vision::Reconstruction(vector<Point2f> &src, vector<Point2f> &dst, Mat &Rt_prev, Mat &Rt,
							vector<Mat> &pt3D_loc)
{
	Mat P_prev, P;
	P_prev = mK * Rt_prev;
	P = mK * Rt;
	//cout << "\n\nP_prv =\n" << P_prev<< "\nP = \n"<< P << endl;
	for (size_t i = 0; i < src.size(); i++)
	{
		Mat pt3D_cam0 = Mat::zeros(3, 1, CV_32F);
		Mat pt3D_cam1 = Mat::zeros(3, 1, CV_32F);
		
		Mat pt4D(4, 1, CV_32F);
		Mat A(6, 4, CV_32F);
		Mat src_skew(3, 3, CV_32F);
		Mat dst_skew(3, 3, CV_32F);
		
		getSkewMatrix(src[i], src_skew);
		getSkewMatrix(dst[i], dst_skew);
		
		Mat src_skewXP = src_skew * P_prev;
		Mat dst_skewXP = dst_skew * P;
		
		A.at<float>(0,0) = src_skewXP.at<float>(0,0);
		A.at<float>(1,0) = src_skewXP.at<float>(1,0);
		A.at<float>(2,0) = src_skewXP.at<float>(2,0);
		A.at<float>(3,0) = dst_skewXP.at<float>(0,0);
		A.at<float>(4,0) = dst_skewXP.at<float>(1,0);
		A.at<float>(5,0) = dst_skewXP.at<float>(2,0);
		
		A.at<float>(0,1) = src_skewXP.at<float>(0,1);
		A.at<float>(1,1) = src_skewXP.at<float>(1,1);
		A.at<float>(2,1) = src_skewXP.at<float>(2,1);
		A.at<float>(3,1) = dst_skewXP.at<float>(0,1);
		A.at<float>(4,1) = dst_skewXP.at<float>(1,1);
		A.at<float>(5,1) = dst_skewXP.at<float>(2,1);

		A.at<float>(0,2) = src_skewXP.at<float>(0,2);
		A.at<float>(1,2) = src_skewXP.at<float>(1,2);
		A.at<float>(2,2) = src_skewXP.at<float>(2,2);
		A.at<float>(3,2) = dst_skewXP.at<float>(0,2);
		A.at<float>(4,2) = dst_skewXP.at<float>(1,2);
		A.at<float>(5,2) = dst_skewXP.at<float>(2,2);

		A.at<float>(0,3) = src_skewXP.at<float>(0,3);
		A.at<float>(1,3) = src_skewXP.at<float>(1,3);
		A.at<float>(2,3) = src_skewXP.at<float>(2,3);
		A.at<float>(3,3) = dst_skewXP.at<float>(0,3);
		A.at<float>(4,3) = dst_skewXP.at<float>(1,3);
		A.at<float>(5,3) = dst_skewXP.at<float>(2,3);

		Mat u,w,vt;
		cv::SVDecomp(A,w,u,vt, cv::SVD::FULL_UV);
		vt.row(3).reshape(0,4).copyTo(pt4D);
		
		for(int r = 0; r < pt3D_cam0.rows; r++)
		{
			pt3D_cam0.at<float>(r) = pt4D.at<float>(r) / pt4D.at<float>(3);
		}
		
		pt3D_cam1 = (Rt.rowRange(0,3).colRange(0,3) * pt3D_cam0) + Rt.rowRange(0,3).col(3);
		
		if(		pt3D_cam0.at<float>(2) > 0 	&& pt3D_cam0.at<float>(2) < 50 
			&& 	pt3D_cam1.at<float>(2) > 0 	&& pt3D_cam1.at<float>(2) < 50)
		{		
			pt3D_loc.push_back(pt3D_cam0);
		}
	}
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
    cv::SVDecomp(A,w,u,vt, cv::SVD::FULL_UV);
    
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

void Vision::PoseFromHomographyMatrix(vector<Point2f> &src, vector<Point2f> &dst, Mat &H)
{
	vector<Mat> R_local, t_local, n_local;
	int solutions = decomposeHomographyMat(H, mK, R_local, t_local, n_local);
	
	/*// TODO: put 4 matrices in std::vector<cv::Mat>
	for(int i = 0; i < solutions; i++)
	{
		R_local[i].convertTo(R_local[i], CV_32F);
		t_local[i].convertTo(t_local[i], CV_32F);
	
		cout << "\nR_loc[" <<i<<"] =\n"<< R_local[i]<< endl;
		cout << "\nt_loc[" <<i<<"] =\t"<< t_local[i].t()<< endl;
	
		cout << "\n#################_BEFORE_#################"<< endl;
		cout << "R_f[" 			<<i<<"] =\n"<< R_f[i]
			 << "\n\nR_f_prev[" <<i<<"] =\n"<< R_f_prev[i]<< endl;
		if (t_local[i].at<float>(2,0) > t_local[i].at<float>(0,0) &&
			t_local[i].at<float>(2,0) > t_local[i].at<float>(1,0))
		{
			
			cout << "\nt_f[" 		<<i<<"] =\t"<< t_f[i].t()
				 << "\nt_f_prev[" 	<<i<<"] =\t"<< t_f_prev[i].t()<< endl;

			t_f[i] = t_f_prev[i] + sc*(R_f[i]*t_local[i]);
			R_f[i] = R_f_prev[i] * R_local[i];
			setCurrentPose(R_f[i], t_f[i], T_f[i]);
		}
		cout << "\n#################_AFTER_#################"<< endl;	
		cout << "R_f[" 			<<i<<"] =\n"<< R_f[i]
			 << "\n\nR_f_prev[" <<i<<"] =\n"<< R_f_prev[i]<< endl;
		
		cout << "\nt_f[" 		<<i<<"] =\t"<< t_f[i].t()
			 << "\nt_f_prev[" 	<<i<<"] =\t"<< t_f_prev[i].t()<< endl;
		//t_f_prev = t_f;
		//R_f_prev = R_f;
		
		t_f[i].copyTo(t_f_prev[i]);
		R_f[i].copyTo(R_f_prev[i]);
		cout << "-----------------------------------------------------" << endl;
		
	}
	for(size_t i = 0; i < T_f.size(); i++)
	{
		cout << "R_f[" <<i<<"] =\n"<< R_f[i]<< endl;
		cout << "t_f[" <<i<<"] =\t"<< t_f[i].t()<< endl;
		//cout << "T_f[" <<i << "] =\n"<< T_f[i]<< endl;
	}
	cout << "-----------------------------------------------------" << endl;*/
	
	R_local[0].convertTo(R_local[0], CV_32F);
	t_local[0].convertTo(t_local[0], CV_32F);

	R_local[1].convertTo(R_local[1], CV_32F);
	t_local[1].convertTo(t_local[1], CV_32F);

	R_local[2].convertTo(R_local[2], CV_32F);
	t_local[2].convertTo(t_local[2], CV_32F);

	R_local[3].convertTo(R_local[3], CV_32F);
	t_local[3].convertTo(t_local[3], CV_32F);

	if (solutions >= 2)
	{
		setCurrentPose(R_local[0], t_local[0], T_loc_0);
		Rodrigues(R_local[0], rvec_loc_0);
		
		setCurrentPose(R_local[1], t_local[1], T_loc_1);
		Rodrigues(R_local[1], rvec_loc_1);
		
		setCurrentPose(R_local[2], t_local[2], T_loc_2);
		Rodrigues(R_local[2], rvec_loc_2);
		
		setCurrentPose(R_local[3], t_local[3], T_loc_3);
		Rodrigues(R_local[3], rvec_loc_3);
		
		if (sc > vMIN_SCALE &&	t_local[0].at<float>(2,0) > t_local[0].at<float>(0,0) &&
								t_local[0].at<float>(2,0) > t_local[0].at<float>(1,0))
		{
			t_f_0 = t_f_prev_0 + sc*(R_f_0*t_local[0]);
			//R_f_0 = R_local[0] * R_f_prev_0;
			R_f_0 = R_f_prev_0 * R_local[0];
			Rodrigues(R_f_0, rvec_0);
		}
		if (sc > vMIN_SCALE &&	t_local[1].at<float>(2,0) > t_local[1].at<float>(0,0) &&
								t_local[1].at<float>(2,0) > t_local[1].at<float>(1,0))
		{			
			t_f_1 = t_f_prev_1 + sc*(R_f_1*t_local[1]);
			R_f_1 = R_f_prev_1 * R_local[1];
			//R_f_1 = R_local[1] * R_f_prev_1;
			Rodrigues(R_f_1, rvec_1);
		}
		if (sc > vMIN_SCALE &&	t_local[2].at<float>(2,0) > t_local[2].at<float>(0,0) &&
								t_local[2].at<float>(2,0) > t_local[2].at<float>(1,0))
		{			
			t_f_2 = t_f_prev_2 + sc*(R_f_2*t_local[2]);
			R_f_2 = R_f_prev_2 * R_local[2];
			//R_f_2 = R_local[2] * R_f_prev_2;
			Rodrigues(R_f_2, rvec_2);
		}
		if (sc > vMIN_SCALE &&	t_local[3].at<float>(2,0) > t_local[3].at<float>(0,0) &&
								t_local[3].at<float>(2,0) > t_local[3].at<float>(1,0))
		{
			t_f_3 = t_f_prev_3 + sc*(R_f_3*t_local[3]);
			R_f_3 = R_f_prev_3 * R_local[3];
			//R_f_3 = R_local[3] * R_f_prev_3;
			Rodrigues(R_f_3, rvec_3);
		}	
	}
	setCurrentPose(R_f_0, t_f_0, T_cam_0);	setCurrentPose(R_f_1, t_f_1, T_cam_1);
	setCurrentPose(R_f_2, t_f_2, T_cam_2);	setCurrentPose(R_f_3, t_f_3, T_cam_3);
	
	R_f_prev_0 = R_f_0;	R_f_prev_1 = R_f_1;	R_f_prev_2 = R_f_2;	R_f_prev_3 = R_f_3;
	t_f_prev_0 = t_f_0;	t_f_prev_1 = t_f_1;	t_f_prev_2 = t_f_2;	t_f_prev_3 = t_f_3;
	rvec_prev_0 = rvec_0;	rvec_prev_1 = rvec_1;	rvec_prev_2 = rvec_2;	rvec_prev_3 = rvec_3;
}

void Vision::getEssentialMatrix(vector<Mat> &n_src, vector<Mat> &n_dst, Mat &E)
{	
	size_t nPoints = n_src.size();
	Mat A(nPoints, 9, CV_32F);
	
	for (size_t i = 0; i < nPoints; i++)
	{
		float x0 = n_src[i].at<float>(0);	float y0 = n_src[i].at<float>(1);
		float x1 = n_dst[i].at<float>(0);	float y1 = n_dst[i].at<float>(1);
		
		A.at<float>(i,0) = x0 * x1;
		A.at<float>(i,1) = x0 * y1;
		A.at<float>(i,2) = x0;
		A.at<float>(i,3) = y0 * x1;
		A.at<float>(i,4) = y0 * y1;
		A.at<float>(i,5) = y0;
		A.at<float>(i,6) = x1;
		A.at<float>(i,7) = y1;
		A.at<float>(i,8) = 1;
	}
	Mat u,w,vt;
	Mat e;
	cv::SVDecomp(A,w,u,vt, cv::SVD::FULL_UV);
	
	cout /*<< "\n\nu =\n" <<u << "\n\nw = \n"<<w */<< "\n\nvt = \n"<<vt<< endl;
	
	
	e = vt.row(8).reshape(3,3).clone();
}

void Vision::GlobalMapPoints(vector<Mat> &p3D_loc, Mat &R_, Mat &t_, Mat &global_pts)
{
	for(size_t i = 0; i < p3D_loc.size(); i++)
	{
		Mat temp = (R_ * p3D_loc[i]) + t_;
		temp.copyTo(global_pts.col(i));
	}	
}

void Vision::getSkewMatrix(Point2f &pt2D, Mat &skew)
{
	skew.at<float>(0,0) = 0;
	skew.at<float>(0,1) = -1;
	skew.at<float>(0,2) = pt2D.y;
		
	skew.at<float>(1,0) = 1;
	skew.at<float>(1,1) = 0;
	skew.at<float>(1,2) = -pt2D.x;

	skew.at<float>(2,0) = -pt2D.y;
	skew.at<float>(2,1) = pt2D.x;
	skew.at<float>(2,2) = 0;
}

void Vision::getSkewMatrix(Mat &mat, Mat &skew)
{
	skew.at<float>(0,0) = 0;
	skew.at<float>(0,1) = -mat.at<float>(2);
	skew.at<float>(0,2) = mat.at<float>(1);//pt2D.y;
		
	skew.at<float>(1,0) = mat.at<float>(2);
	skew.at<float>(1,1) = 0;
	skew.at<float>(1,2) = -mat.at<float>(0);//pt2D.x;

	skew.at<float>(2,0) = -mat.at<float>(1);//pt2D.y;
	skew.at<float>(2,1) = mat.at<float>(0);//pt2D.x;
	skew.at<float>(2,2) = 0;
}

void Vision::pt2D_to_mat(Point2f &pt2d, Mat &pt2dMat)
{
	pt2dMat.at<float>(0) = pt2d.x;
	pt2dMat.at<float>(1) = pt2d.y;
	pt2dMat.at<float>(2) = 1;
}

void Vision::SetR_t(Mat &R_, Mat &t_, Mat &Rt_)
{
	Rt_.create(3, 4, R_.type());
	R_.copyTo(Rt_.rowRange(0,3).colRange(0,3));
	t_.copyTo(Rt_.rowRange(0,3).col(3));
}

void Vision::get_info(Mat &matrix, string matrix_name)
{
	cout << matrix_name 	<<"\ntype = "		<< matrix.type() 
							<< "\tdepth = "		<< matrix.depth() 
							<< "\tch = " 		<< matrix.channels()
							<< endl;	
}

}//namespace ORB_VISLAM
