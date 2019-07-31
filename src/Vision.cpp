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
		
	R_Glob = Mat::eye(3, 3, CV_32F);
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
	
	rvec_Glob_prv 	= Mat::zeros(3, 1, CV_32F);
	rvec_loc_E 		= Mat::zeros(3, 1, CV_32F);
	rvec_loc_0 		= Mat::zeros(3, 1, CV_32F);
	rvec_loc_1 		= Mat::zeros(3, 1, CV_32F);
	rvec_loc_2 		= Mat::zeros(3, 1, CV_32F);
	rvec_loc_3 		= Mat::zeros(3, 1, CV_32F);
	
	t_Glob = Mat::eye(3, 1, CV_32F);
	t_f_E = Mat::zeros(3, 1, CV_32F);
	t_f_0 = Mat::zeros(3, 1, CV_32F);
	t_f_1 = Mat::zeros(3, 1, CV_32F);
	t_f_2 = Mat::zeros(3, 1, CV_32F);
	t_f_3 = Mat::zeros(3, 1, CV_32F);
	
	R_f_prev_E = Mat::eye(3, 3, CV_32F);
	R_Glob_prv = Mat::eye(3, 3, CV_32F);
	R_f_prev_0 = Mat::eye(3, 3, CV_32F);
	R_f_prev_1 = Mat::eye(3, 3, CV_32F);
	R_f_prev_2 = Mat::eye(3, 3, CV_32F);
	R_f_prev_3 = Mat::eye(3, 3, CV_32F);
	
	rvec_prev_E = Mat::zeros(3, 1, CV_32F);
	rvec_Glob_prv = Mat::eye(3, 3, CV_32F);
	rvec_prev_0 = Mat::zeros(3, 1, CV_32F);
	rvec_prev_1 = Mat::zeros(3, 1, CV_32F);
	rvec_prev_2 = Mat::zeros(3, 1, CV_32F);
	rvec_prev_3 = Mat::zeros(3, 1, CV_32F);
	
	t_f_prev_E = Mat::zeros(3, 1, CV_32F);
	t_Glob_prv = Mat::eye(3, 1, CV_32F);
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
		vector<vector<DMatch>> possible_matches;
		vector<Point2f> src, dst;
		vector<int> ref_kp_idx, kp_idx;
		
		matcher->knnMatch(ref_desc, desc, possible_matches, 2); // size = size(KP)
		
		for(size_t i = 0; i < possible_matches.size(); i++) 
		{
			DMatch first = possible_matches[i][0];
			float dist1 = possible_matches[i][0].distance;
			float dist2 = possible_matches[i][1].distance;
			if(dist1 < nn_match_ratio * dist2) 
			{
				ref_kp_idx.push_back(first.queryIdx);
				kp_idx.push_back(first.trainIdx);
			
				src.push_back(ref_kp[first.queryIdx].pt);
				dst.push_back(kp[first.trainIdx].pt);
			}
		}
		getDummy(src, dst, ref_kp_idx, kp_idx, possible_matches, match_idx);
		//essential_matrix_inliers(src, dst, ref_kp_idx, kp_idx, possible_matches, match_idx);
	}
}

void Vision::getDummy(vector<Point2f> &src, vector<Point2f> &dst,
										vector<int> &ref_kp_idx, vector<int> &kp_idx,
										vector<vector<DMatch>> &possible_matches,
										vector<pair<int,int>> &match_idx)
{
	double minVal,maxVal;
	cv::minMaxIdx(src, &minVal, &maxVal);
	vector<Point2f> srcInlier, dstInlier;
	
	//double minDis2EpiLine = 0.006 * maxVal;
	double minDis2EpiLine = .5; 
	double confid = .99;
	vector<uchar> matching_mask; // ransac
	
	Mat E = findEssentialMat(dst, src, mK, RANSAC, confid, minDis2EpiLine, matching_mask);
	
	int inliers_num = countNonZero(matching_mask);
	for(size_t nm = 0; nm < matching_mask.size(); nm++)
	{
		if(matching_mask[nm])
		{
			match_idx.push_back(make_pair(ref_kp_idx[nm], kp_idx[nm]));
			srcInlier.push_back(src[nm]);
			dstInlier.push_back(dst[nm]);
		}
	}
	cout << "Matching:\n"; 
	cout << "\tKPS\tMATCHES\tINLIERS\n";
	cout 	<< '\t' << possible_matches.size() 	<< '\t' 	<< matching_mask.size() 
			<< '\t' << inliers_num 			<< '\n';

	if (!srcInlier.empty() && !dstInlier.empty() && match_idx.size() >= vMIN_NUM_FEAT)
	{
		GetPose(srcInlier, dstInlier, E);		
		cout << "----------------------------------------------------------------" << endl;
	}
}

void Vision::essential_matrix_inliers(vector<Point2f> &src, vector<Point2f> &dst,
										vector<int> &ref_kp_idx, vector<int> &kp_idx,
										vector<vector<DMatch>> &possible_matches,
										vector<pair<int,int>> &match_idx)
{
	double minVal,maxVal;
	cv::minMaxIdx(src, &minVal, &maxVal);
	vector<Point2f> srcInlier, dstInlier;
	
	//double minDis2EpiLine = 0.006 * maxVal;
	double minDis2EpiLine = .5; 
	double confid = .99;
	vector<uchar> matching_mask; // ransac
	
	Mat E = findEssentialMat(dst, src, mK, RANSAC, confid, minDis2EpiLine, matching_mask);
	
	int inliers_num = countNonZero(matching_mask);
	for(size_t nm = 0; nm < matching_mask.size(); nm++)
	{
		if(matching_mask[nm])
		{
			match_idx.push_back(make_pair(ref_kp_idx[nm], kp_idx[nm]));
			srcInlier.push_back(src[nm]);
			dstInlier.push_back(dst[nm]);
		}
	}
	
	cout << "2D Matches:\n"; 
	cout << "\tKPS\tMATCHES\tINLIERS\n";
	cout 	<< '\t' << possible_matches.size() 	<< '\t' 	<< matching_mask.size() 
			<< '\t' << inliers_num 			<< '\n';

	if (!srcInlier.empty() && !dstInlier.empty() && match_idx.size() >= vMIN_NUM_FEAT)
	{
		PoseFromEssentialMatrix(srcInlier, dstInlier, E);
		cout << "----------------------------------------------------------------" << endl;
	}
}
void Vision::proj3D_2D(Mat &p3_, Mat &p2_)
{
	if(!p3_.empty())
	{
		// x = P * X
		Mat P_prev = mK * Rt_prev;
		Mat p2_homogen = Mat::ones(3, p3_.cols, p3_.type());
		
		p2_homogen = P_prev * p3_;
	
		p2_homogen.row(0) /= p2_homogen.row(2);
		p2_homogen.row(1) /= p2_homogen.row(2);
		p2_homogen.row(2) /= p2_homogen.row(2);
		
		p2_homogen.copyTo(p2_);
		//get_info(p2_, "\npt2D_h:");
	}
}

void Vision::GetPose(vector<Point2f> &src, vector<Point2f> &dst, Mat &E)
{		
	Mat R1, R2, t1, t2, R_local, t_local, P3d, reprojPts, measuredPts;
	vector<Mat> p3D_loc;
	vector<float> good_vec(4);
	vector<Mat> Rt_vec(4);
	vector<Mat> Pts3D_raw(4);
	
	// 1. Decompose Essential Matrix to obtain R1, R2, t:
	decomE(E, R1, R2, t1, t2);
	
	SetR_t(R1, t1, Rt_vec[0]);
	SetR_t(R2, t1, Rt_vec[1]);
	SetR_t(R1, t2, Rt_vec[2]);	
	SetR_t(R2, t2, Rt_vec[3]);
	
	for(size_t i = 0; i < Rt_vec.size(); i++)
	{
		triangulateMyPoints(src, dst, Rt_vec[i], good_vec[i], Pts3D_raw[i]);
	}
	
	cout << "\nFront 3D Pts:\n"; 
	cout << "\tSOL_0\tSOL_1\tSOL_2\tSOL_3\n";
	cout 	<< '\t' << setprecision(3) << good_vec[0] 
			<< '\t' << setprecision(3) << good_vec[1] 
			<< '\t' << setprecision(3) << good_vec[2] 
			<< '\t' << setprecision(3) << good_vec[3] 
			<< '\n';

	// 2. Obtain correct Pose and 3D pts: 
	pose_AND_3dPts(dst, good_vec, Rt_vec, Pts3D_raw, R_local, t_local, P3d, measuredPts);
	
	proj3D_2D(P3d, reprojPts);
		
	calcReprojErr(measuredPts, reprojPts, repErr);
	
	if(t_local.at<float>(2) < 0)
	{
		t_local.at<float>(2) *= -1.0f;
	}
	setCurrentPose(R_local, t_local, T_loc_E);
	SetR_t(R_local, t_local, Rt);
	Rodrigues(R_local, rvec_loc_E);
	
	if (sc > vMIN_SCALE &&	t_local.at<float>(2) > t_local.at<float>(0) &&
							t_local.at<float>(2) > t_local.at<float>(1))
	{
		P3d.copyTo(visionMap);
		t_f_E = sc*(R_f_E * t_local) + t_f_prev_E;
		R_f_E = R_f_prev_E * R_local;
	}
	Rodrigues(R_f_E, rvec_E);
	setCurrentPose(R_f_E, t_f_E, T_cam_E);
	//cout << "\nT_E = \n" << T_cam_E << endl;
	
	R_f_prev_E 		= R_f_E.clone();
	rvec_prev_E 	= rvec_E.clone();	
	t_f_prev_E 		= t_f_E.clone();
}

void Vision::calcReprojErr(Mat &measured, Mat &reprojected, float &rE)
{
	if(!measured.empty())
	{
		reprojected.copyTo(vPt2D_rep);
		measured.copyTo(vPt2D_measured);
		rE = cv::norm(reprojected, measured, NORM_L2)/ (float)reprojected.cols;
		cout << "\n reprojection error = "<< rE << endl;
	}
}

void Vision::applyContraints(vector<Point2f> &dst, Mat &p3_raw, Mat &Rt, Mat &p3_front, Mat &origMeasPts)
{
	// cheirality constraint:
	Mat p3_raw_cam1 = Rt * p3_raw;
	vector<int> p3_front_idx;
	for(int i = 0; i < p3_raw.cols; i++)
	{
		if(	(p3_raw.at<float>(2,i) 		> 0) 				&& 
			(p3_raw.at<float>(2,i) 		< vdistTh_3DPts) 	&&
			(p3_raw_cam1.at<float>(2,i) 	> 0) 				&& 
			(p3_raw_cam1.at<float>(2,i) 	< vdistTh_3DPts))
		{
			p3_front_idx.push_back(i);
		}		
	}
	

	p3_front.create(4, p3_front_idx.size(), CV_32F);
	origMeasPts.create(3, p3_front_idx.size(), CV_32F);
	
	for(size_t k = 0; k < p3_front_idx.size(); k++)
	{
		Mat temp = p3_raw.col(p3_front_idx[k]);
		temp.copyTo(p3_front.col(k));
		
		origMeasPts.at<float>(0,k) = dst[p3_front_idx[k]].x;
		origMeasPts.at<float>(1,k) = dst[p3_front_idx[k]].y;
		origMeasPts.at<float>(2,k) = 1;
	}
}

void Vision::pose_AND_3dPts(vector<Point2f> &dst, vector<float> &good_vec, vector<Mat> &Rt_vec, 
								vector<Mat> &Pts3D_vec, 
								Mat &R_, Mat &t_, Mat &p3_, Mat &measuredPts)
{
	R_.create(3, 3, CV_32F);
	t_.create(3, 1, CV_32F);
	
	if( good_vec[0] >= good_vec[1] && 
		good_vec[0] >= good_vec[2] && 
		good_vec[0] >= good_vec[3])
	{
		front3DPtsOWN = good_vec[0];
		Rt_vec[0].rowRange(0,3).colRange(0,3).copyTo(R_);
		Rt_vec[0].col(3).copyTo(t_);
		
		Mat constrained_3DPts, measured2DPts;
		applyContraints(dst, Pts3D_vec[0], Rt_vec[0], constrained_3DPts, measured2DPts);
		constrained_3DPts.copyTo(p3_);
		measured2DPts.copyTo(measuredPts);
	}
	else if(good_vec[1] >= good_vec[0] && 
			good_vec[1] >= good_vec[2] && 
			good_vec[1] >= good_vec[3])
	{
		front3DPtsOWN = good_vec[1];
		Rt_vec[1].rowRange(0,3).colRange(0,3).copyTo(R_);
		Rt_vec[1].col(3).copyTo(t_);
		
		Mat constrained_3DPts, measured2DPts;
		applyContraints(dst, Pts3D_vec[1], Rt_vec[1], constrained_3DPts, measured2DPts);
		constrained_3DPts.copyTo(p3_);
		measured2DPts.copyTo(measuredPts);
	}
	else if(good_vec[2] >= good_vec[0] && 
			good_vec[2] >= good_vec[1] && 
			good_vec[2] >= good_vec[3])
	{
		front3DPtsOWN = good_vec[2];
		Rt_vec[2].rowRange(0,3).colRange(0,3).copyTo(R_);
		Mat TEM = Rt_vec[2].col(3) * -1;
		TEM.copyTo(t_);
		
		Mat constrained_3DPts, measured2DPts;
		applyContraints(dst, Pts3D_vec[2], Rt_vec[2], constrained_3DPts, measured2DPts);
		constrained_3DPts.copyTo(p3_);
		measured2DPts.copyTo(measuredPts);
	}
	else if(good_vec[3] >= good_vec[0] && 
			good_vec[3] >= good_vec[1] && 
			good_vec[3] >= good_vec[2])
	{
		front3DPtsOWN = good_vec[2];
		Rt_vec[3].rowRange(0,3).colRange(0,3).copyTo(R_);
		Mat TEMP_ = Rt_vec[3].col(3) * -1;
		TEMP_.copyTo(t_);
		
		Mat constrained_3DPts, measured2DPts;
		applyContraints(dst, Pts3D_vec[3], Rt_vec[3], constrained_3DPts, measured2DPts);
		constrained_3DPts.copyTo(p3_);
		measured2DPts.copyTo(measuredPts);
	}
	else
	{
		cout << "No SOLUTION FOUND!!!" << endl;
		exit(EXIT_FAILURE);
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

	if (sc > vMIN_SCALE &&	fabs(t_local.at<float>(2)) > t_local.at<float>(0) &&
							fabs(t_local.at<float>(2)) > t_local.at<float>(1))
	{
		Reconstruction(src, dst, Rt, p3D_loc);
		Mat visLOCAL(3, p3D_loc.size(), CV_32F);
		for (size_t i = 0; i < p3D_loc.size(); i++)
		{
			p3D_loc[i].copyTo(visLOCAL.col(i));
		}
		t_f_E = sc*(R_f_E * t_local) + t_f_prev_E;
		R_f_E = R_f_prev_E * R_local;
		// TODO: size of visionMap has been changed! seg fault in visualize...
		// TODO: make 3dpt cv::Mat and homogenous!
		visLOCAL.copyTo(visionMap);
	}
	Rodrigues(R_f_E, rvec_E);
	setCurrentPose(R_f_E, t_f_E, T_cam_E);
	//cout << "\nT_E = \n" << T_cam_E << endl;
	
	R_f_prev_E 		= R_f_E.clone();
	rvec_prev_E 	= rvec_E.clone();	
	t_f_prev_E 		= t_f_E.clone();
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
									Mat &Rt, float &acceptance_rate, Mat &P3_H_RAW)
{
	vector<Mat> src_norm, dst_norm;
	Normalize2DPts(src, dst, src_norm, dst_norm);
	Mat Pts4D, constraints;
	Extract4DPts(src_norm, dst_norm, Rt, Pts4D);
	
	// paper: Nister:
	// cam_0:
	constraints = Pts4D.row(2).mul(Pts4D.row(3)) > 0; // Z * W > 0
	
	Pts4D.row(0) /= Pts4D.row(3);
	Pts4D.row(1) /= Pts4D.row(3);
	Pts4D.row(2) /= Pts4D.row(3);
	Pts4D.row(3) /= Pts4D.row(3);
	
	//constraints = Pts4D.row(2) > 0; // Z / W > 0
	constraints = (Pts4D.row(2) < vdistTh_3DPts) & constraints; 
	
	// cam_1:
	Mat Pts4D_cam1 = Rt * Pts4D;
	constraints = (Pts4D_cam1.row(2) > 0) & constraints;
	constraints = (Pts4D_cam1.row(2) < vdistTh_3DPts) & constraints;
	int nz = countNonZero(constraints);
	acceptance_rate = ((float)nz / (float)constraints.cols);
	Pts4D.copyTo(P3_H_RAW);
}

void Vision::Extract4DPts(vector<Mat> &src, vector<Mat> &dst, Mat &Rt, Mat &Points4D)
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

void Vision::Reconstruction(vector<Point2f> &src, vector<Point2f> &dst, Mat &Rt,
							vector<Mat> &pt3D_loc)
{
	Mat P_prev = mK * Rt_prev;
	Mat P = mK * Rt;
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
		
		if(		pt3D_cam0.at<float>(2) > 0 	&& pt3D_cam0.at<float>(2) < vdistTh_3DPts 
			&& 	pt3D_cam1.at<float>(2) > 0 	&& pt3D_cam1.at<float>(2) < vdistTh_3DPts)
		{		
			pt3D_loc.push_back(pt3D_cam0);
		}
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
							<< "\tr = "			<< matrix.rows
							<< "\tc = "			<< matrix.cols
							<< endl;	
}

}//namespace ORB_VISLAM
