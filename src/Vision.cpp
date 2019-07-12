#include "Vision.h"

using namespace std;
using namespace cv;

using namespace pcl;
using namespace pcl::io;
using namespace pcl::visualization;

namespace ORB_VISLAM{

Vision::Vision(const string &settingFilePath,
						int win_sz, float ssd_th, float ssd_ratio_th, 
						size_t minFeatures, float minScale){
	cout << "" << endl;
	cout << "#########################################################################" << endl;
	cout << "\t\t\t\tVISION"															<< endl;
	cout << "#########################################################################" << endl;
	
	FileStorage fSettings(settingFilePath, FileStorage::READ);
    float fx 			= fSettings["Camera.fx"];
    float fy 			= fSettings["Camera.fy"];
    float cx 			= fSettings["Camera.cx"];
    float cy 			= fSettings["Camera.cy"];
	
	FOCAL_LENGTH = fx;
	
	pp = Point2f(cx,cy);
	fps 				= fSettings["Camera.fps"];
	vMIN_SCALE = minScale;
	Mat K = Mat::eye(3, 3, CV_64F);
	K.at<double>(0,0) 		= fx;
    K.at<double>(1,1) 		= fy;
    K.at<double>(0,2) 		= cx;
    K.at<double>(1,2) 		= cy;

	K.copyTo(mK);
	
	mK.copyTo(P_prev.rowRange(0,3).colRange(0,3));
	mK.copyTo(P.rowRange(0,3).colRange(0,3));
	
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
	
	I_3x3 	= Mat::eye(3, 3, CV_64F);
	I_4x4 	= Mat::eye(4, 4, CV_64F);	
	Z_3x1 	= Mat::zeros(3, 1, CV_64F);
	
	T_local = vector<Mat>{I_4x4, I_4x4, I_4x4, I_4x4};
	T_f 	= vector<Mat>{I_4x4, I_4x4, I_4x4, I_4x4};
	
	R_f 		= vector<Mat>{I_3x3, I_3x3, I_3x3, I_3x3};
	R_f_prev 	= vector<Mat>{I_3x3, I_3x3, I_3x3, I_3x3};
	
	t_f 		= vector<Mat>{Z_3x1, Z_3x1, Z_3x1, Z_3x1};
	t_f_prev 	= vector<Mat>{Z_3x1, Z_3x1, Z_3x1, Z_3x1};
	
	R_f_E = Mat::eye(3, 3, CV_64F);
	R_f_0 = Mat::eye(3, 3, CV_64F);
	R_f_1 = Mat::eye(3, 3, CV_64F);
	R_f_2 = Mat::eye(3, 3, CV_64F);
	R_f_3 = Mat::eye(3, 3, CV_64F);

	cloud = boost::make_shared<pcl::PointCloud<pcl::PointXYZRGB>>();
	
	rvec_E = Mat::zeros(3, 1, CV_64F);
	rvec_0 = Mat::zeros(3, 1, CV_64F);
	rvec_1 = Mat::zeros(3, 1, CV_64F);
	rvec_2 = Mat::zeros(3, 1, CV_64F);
	rvec_3 = Mat::zeros(3, 1, CV_64F);
	
	rvec_loc_E = Mat::zeros(3, 1, CV_64F);
	rvec_loc_0 = Mat::zeros(3, 1, CV_64F);
	rvec_loc_1 = Mat::zeros(3, 1, CV_64F);
	rvec_loc_2 = Mat::zeros(3, 1, CV_64F);
	rvec_loc_3 = Mat::zeros(3, 1, CV_64F);
	
	t_f_E = Mat::zeros(3, 1, CV_64F);
	t_f_0 = Mat::zeros(3, 1, CV_64F);
	t_f_1 = Mat::zeros(3, 1, CV_64F);
	t_f_2 = Mat::zeros(3, 1, CV_64F);
	t_f_3 = Mat::zeros(3, 1, CV_64F);
	
	R_f_prev_E = Mat::eye(3, 3, CV_64F);
	R_f_prev_0 = Mat::eye(3, 3, CV_64F);
	R_f_prev_1 = Mat::eye(3, 3, CV_64F);
	R_f_prev_2 = Mat::eye(3, 3, CV_64F);
	R_f_prev_3 = Mat::eye(3, 3, CV_64F);
	
	rvec_prev_E = Mat::zeros(3, 1, CV_64F);
	rvec_prev_0 = Mat::zeros(3, 1, CV_64F);
	rvec_prev_1 = Mat::zeros(3, 1, CV_64F);
	rvec_prev_2 = Mat::zeros(3, 1, CV_64F);
	rvec_prev_3 = Mat::zeros(3, 1, CV_64F);
	
	t_f_prev_E = Mat::zeros(3, 1, CV_64F);
	t_f_prev_0 = Mat::zeros(3, 1, CV_64F);
	t_f_prev_1 = Mat::zeros(3, 1, CV_64F);
	t_f_prev_2 = Mat::zeros(3, 1, CV_64F);
	t_f_prev_3 = Mat::zeros(3, 1, CV_64F);
}

void Vision::Analyze(Mat &rawImg, vector<KeyPoint> &kp, 
						vector<pair<int,int>> &matches, 
						vector<Point3f> &map_points)
{
	Mat descriptor;
	
	get_AKAZE_kp(rawImg, kp, descriptor);
	//cout << "current kp sz = \t " << kp.size()<< endl;
	// TODO: descriptor is different from AKAZE ??????????????????
	//get_ORB_kp(rawImg, kp, descriptor);
	
	
	//matching(rawImg, kp, matches);
	
	matching(rawImg, kp, descriptor, matches);
	map_points = get_map_points();
	cout << "map sz [Vision::Analyze] =\t" << map_points.size() << endl;
	ref_kp 		= kp;
	ref_desc 	= descriptor;
	ref_img 	= rawImg;
}

vector<Point3f> Vision::get_map_points()
{
	return map_3D;
}

void Vision::setCurrentPose(Mat &R_, Mat &t_, Mat &T_)
{
	t_.copyTo(T_.rowRange(0,3).col(3));
	R_.copyTo(T_.rowRange(0,3).colRange(0,3));
}

void Vision::get_ORB_kp(Mat &rawImg, vector<KeyPoint> &kp, Mat &desc)
{
	//Ptr<FeatureDetector> 		feature 	= ORB::create();
	Ptr<ORB> 					feature		= ORB::create();
	//Ptr<DescriptorExtractor> 	matcher 	= ORB::create();
    kp.clear();
	feature->detect(rawImg, kp);
	feature->compute(rawImg, kp, desc);
}

void Vision::get_AKAZE_kp(Mat &rawImg, vector<KeyPoint> &kp, Mat &desc)
{
	Ptr<AKAZE> 					feature 	= AKAZE::create();
	kp.clear();
	feature->detect(rawImg, kp);
	feature->compute(rawImg, kp, desc);
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

void Vision::matching(Mat &img, vector<KeyPoint> &kp, Mat &desc, vector<pair<int,int>> &match_idx)
{	
	Ptr<DescriptorMatcher> matcher 	= DescriptorMatcher::create("BruteForce-Hamming");
	
	if (!ref_kp.empty())
	{
		vector<vector<DMatch>> all_matches;
		vector<Point2f> src, dst;
		vector<int> ref_kp_idx, kp_idx;
		
		matcher->knnMatch(ref_desc, desc, all_matches, 2);
		for (auto &m : all_matches)
		{
			if(m[0].distance < nn_match_ratio * m[1].distance) 
			{
				src.push_back(ref_kp[m[0].queryIdx].pt);
				dst.push_back(kp[m[0].trainIdx].pt);
				
				ref_kp_idx.push_back(m[0].queryIdx);
				kp_idx.push_back(m[0].trainIdx);
			}
		}
		essential_matrix_inliers(src, dst, ref_kp_idx, kp_idx, all_matches, match_idx);
		//homography_matrix_inliers(src, dst, ref_kp_idx, kp_idx, all_matches, match_idx);
		//fundamental_matrix_inliers(src, dst, ref_kp_idx, kp_idx, all_matches, match_idx);	
	}
}

void Vision::homography_matrix_inliers(vector<Point2f> &src, vector<Point2f> &dst, 
										vector<int> &ref_kp_idx, vector<int> &kp_idx, 
										vector<vector<DMatch>> &all_matches,
										vector<pair<int,int>> &match_idx)
{
	const double ransac_thresh = 1.0f; 
	vector<uchar> inlier_mask;
	vector<Point2f> srcInlier, dstInlier;
	
	Homography_Matrix = findHomography(dst, src, RANSAC, ransac_thresh, inlier_mask);
	
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
		PoseFromHomographyMatrix(srcInlier, dstInlier);
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
	
	//double minDis2EpipolarLine = 0.006 * maxVal; // in pixel, greater values -> outlier.
	double minDis2EpipolarLine = .5; // in pixel, greater values -> outlier.
	double confidence = .999;
	vector<uchar> mask;
	Mat E12, E21;
	
	Essential_Matrix = findEssentialMat(dst, src, FOCAL_LENGTH, pp, RANSAC, 
											confidence, minDis2EpipolarLine, mask);

    //cout << "\n\nEssential Matrix =\n " << Essential_Matrix << endl;
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
	cout << '\t' << all_matches.size() << '\t' << mask.size() << '\t' << inliers_num << '\n';
		
	if (!srcInlier.empty() && !dstInlier.empty() && match_idx.size() >= vMIN_NUM_FEAT)
	{
		PoseFromEssentialMatrix(srcInlier, dstInlier);
		cout << "----------------------------------------------------------------" << endl;
	}
}

void Vision::fundamental_matrix_inliers(vector<Point2f> &src, vector<Point2f> &dst, 
										vector<int> &ref_kp_idx, vector<int> &kp_idx, 
										vector<vector<DMatch>> &all_matches,
										vector<pair<int,int>> &match_idx)
{
	double minVal,maxVal;
	cv::minMaxIdx(src, &minVal, &maxVal);
	vector<Point2f> srcInlier, dstInlier;
	
	//double minDis2EpipolarLine = 0.006 * maxVal; // in pixel, greater values -> outlier.
	double minDis2EpipolarLine = .5; // in pixel, greater values -> outlier.
	double confidence = .999;
	vector<uchar> mask;
	
	Fundamental_Matrix = findFundamentalMat(src, dst, FM_RANSAC, 
											minDis2EpipolarLine, confidence, mask);
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
	cout << '\t' << all_matches.size() << '\t' << mask.size() << '\t' << inliers_num << '\n';
		
	if (!srcInlier.empty() && !dstInlier.empty() && match_idx.size() >= vMIN_NUM_FEAT)
	{
		PoseFromFundamentalMatrix(srcInlier, dstInlier);
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

void Vision::PoseFromHomographyMatrix(vector<Point2f> &src, vector<Point2f> &dst)
{
	vector<Mat> R_local, t_local, n_local;
	int solutions = decomposeHomographyMat(Homography_Matrix, mK, R_local, t_local, n_local);
	
	// TODO: put 4 matrices in std::vector<cv::Mat>
	/*for(int i = 0; i < solutions; i++)
	{
		//cout << "R_loc[" <<i<<"] =\n"<< R_local[i]<< endl;
		cout << "t_loc[" <<i<<"] =\t"<< t_local[i].t()<< endl;
	
		if (t_local[i].at<double>(2,0) > t_local[i].at<double>(0,0) &&
			t_local[i].at<double>(2,0) > t_local[i].at<double>(1,0))
		{
			t_f[i] = t_f_prev[i] + sc*(R_f[i]*t_local[i]);
			R_f[i] = R_f_prev[i] * R_local[i];
			//Rodrigues(R_f_0, rvec_0);
			
			setCurrentPose(R_f[i], t_f[i], T_f[i]);
			t_f_prev[i] = t_f[i];
			R_f_prev[i] = R_f[i];
		}
		cout << "-------------------------------------------------"<< endl;
	}
	
	for(size_t i = 0; i < T_f.size(); i++)
	{
		cout << "R_f[" <<i<<"] =\n"<< R_f[i]<< endl;
		cout << "t_f[" <<i<<"] =\t"<< t_f[i].t()<< endl;
		//cout << "T_f[" <<i << "] =\n"<< T_f[i]<< endl;
	}
	cout << "-----------------------------------------------------" << endl;*/
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
		
		if (sc > vMIN_SCALE &&	t_local[0].at<double>(2,0) > t_local[0].at<double>(0,0) &&
						t_local[0].at<double>(2,0) > t_local[0].at<double>(1,0))
		{
			t_f_0 = t_f_prev_0 + sc*(R_f_0*t_local[0]);
			//R_f_0 = R_local[0] * R_f_prev_0;
			R_f_0 = R_f_prev_0 * R_local[0];
			Rodrigues(R_f_0, rvec_0);
		}
		if (sc > vMIN_SCALE &&	t_local[1].at<double>(2,0) > t_local[1].at<double>(0,0) &&
								t_local[1].at<double>(2,0) > t_local[1].at<double>(1,0))
		{			
			t_f_1 = t_f_prev_1 + sc*(R_f_1*t_local[1]);
			R_f_1 = R_f_prev_1 * R_local[1];
			//R_f_1 = R_local[1] * R_f_prev_1;
			Rodrigues(R_f_1, rvec_1);
		}
		if (sc > vMIN_SCALE &&	t_local[2].at<double>(2,0) > t_local[2].at<double>(0,0) &&
								t_local[2].at<double>(2,0) > t_local[2].at<double>(1,0))
		{			
			t_f_2 = t_f_prev_2 + sc*(R_f_2*t_local[2]);
			R_f_2 = R_f_prev_2 * R_local[2];
			//R_f_2 = R_local[2] * R_f_prev_2;
			Rodrigues(R_f_2, rvec_2);
		}
		if (sc > vMIN_SCALE &&	t_local[3].at<double>(2,0) > t_local[3].at<double>(0,0) &&
								t_local[3].at<double>(2,0) > t_local[3].at<double>(1,0))
		{
			t_f_3 = t_f_prev_3 + sc*(R_f_3*t_local[3]);
			R_f_3 = R_f_prev_3 * R_local[3];
			//R_f_3 = R_local[3] * R_f_prev_3;
			Rodrigues(R_f_3, rvec_3);
		}
		
			/*//### NO CONSTRAINTS ###
			t_f_0 = t_f_prev_0 + sc*(R_f_0*t_local[0]);
			R_f_0 = R_local[0] * R_f_prev_0;	
			Rodrigues(R_f_0, rvec_0);
		
			t_f_1 = t_f_prev_1 + sc*(R_f_1*t_local[1]);
			R_f_1 = R_local[1] * R_f_prev_1;
			Rodrigues(R_f_1, rvec_1);
		
			t_f_2 = t_f_prev_2 + sc*(R_f_2*t_local[2]);
			R_f_2 = R_local[2] * R_f_prev_2;
			Rodrigues(R_f_2, rvec_2);
		
			t_f_3 = t_f_prev_3 + sc*(R_f_3*t_local[3]);
			R_f_3 = R_local[3] * R_f_prev_3;	
			Rodrigues(R_f_3, rvec_3);
			//### NO CONSTRAINTS ###*/
			
	}
	setCurrentPose(R_f_0, t_f_0, T_cam_0);
	//cout << "\nT_0\n = "<< T_cam_0 << endl;
	//cout << "\nT_loc_0\n = "<< T_loc_0 << endl;

	setCurrentPose(R_f_1, t_f_1, T_cam_1);
	//cout << "\nT_1\n = "<< T_cam_1 << endl;
	//cout << "\nT_loc_1\n = "<< T_loc_1 << endl;
	
	setCurrentPose(R_f_2, t_f_2, T_cam_2);
	//cout 	<< "\nT_loc_2\n = "<< T_loc_2 
	//		<< "\nT_2\n = "<< T_cam_2 << endl;
	
	setCurrentPose(R_f_3, t_f_3, T_cam_3);
	//cout << "\nT_3\n = "<< T_cam_3 << endl;
	//cout << "\nT_loc_3\n = "<< T_loc_3 << endl;
	
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
}

void Vision::PoseFromEssentialMatrix(vector<Point2f> &src, vector<Point2f> &dst)
{
	Mat R_local, t_local, mask;
	//double err = 0;
	vector<Point3f> pt3D_local_vec;
	Mat pt3D_matrix(3, 1, CV_64F);
	
	recoverPose(Essential_Matrix, dst, src, R_local, t_local, FOCAL_LENGTH, pp, mask);
	setCurrentPose(R_local, t_local, T_loc_E);
	R_local.copyTo(P.rowRange(0,3).colRange(0,3));
	t_local.copyTo(P.col(3));
	P = mK * P;
	//cout << "sc (Essential Matrix)\t = "<< sc  << endl;
	if (sc > vMIN_SCALE &&	t_local.at<double>(2,0) > t_local.at<double>(0,0) &&
							t_local.at<double>(2,0) > t_local.at<double>(1,0))
	{
		t_f_E = t_f_prev_E + sc*(R_f_E * t_local);
		R_f_E = R_f_prev_E * R_local;
		//R_f_E = R_local * R_f_prev_E; 
		Rodrigues(R_f_0, rvec_E);
		setCurrentPose(R_f_E, t_f_E, T_cam_E);
		Reconstruction(src, dst, P_prev, P, pt3D_local_vec);
		for(size_t i = 0; i < pt3D_local_vec.size(); i++)
		{
			Mat pt3D_loc_matrix(3, 1, CV_64F);
			Point3D_2_Mat(pt3D_local_vec[i], pt3D_loc_matrix);
			pt3D_matrix = (R_f_E * pt3D_loc_matrix) + t_f_E;
			map_3D.push_back(Mat_2_Point3D(pt3D_matrix));
		}
		cout << "3D Points [Map]:\t" << map_3D.size() << endl;
		//triangulateFcn(src, dst, P_prev, P, err);
	}
	R_f_prev_E 		= R_f_E.clone();
	rvec_prev_E 	= rvec_E.clone();	
	t_f_prev_E 		= t_f_E.clone();
	P_prev			= P.clone();
}

void Vision::Point3D_2_Mat(Point3f &pt, Mat &mat)
{
	mat.at<double>(0,0) = pt.x;
	mat.at<double>(1,0) = pt.y;
	mat.at<double>(2,0) = pt.z;
}

Point3f Vision::Mat_2_Point3D(Mat &mat)
{
	return (Point3f(mat.at<double>(0,0), mat.at<double>(1,0), mat.at<double>(2,0)));
}

void Vision::Reconstruction(vector<Point2f> &src, 	
							vector<Point2f> &dst, 
							Mat &P_prev, Mat &P, 
							vector<Point3f> &pt3D_loc)
{
	//cout << "\n\nP_prv =\n" << P_prev<< "\nP = \n"<< P << endl;
	Point3f pt3D_cam0, pt3D_cam1;
	Mat pt4D = Mat::zeros(4, src.size(), CV_64F);
	for (size_t i = 0; i < src.size(); i++)
	{
		Mat A(6, 4, CV_64F);
		Mat src_skew(3, 3, CV_64F);
		Mat dst_skew(3, 3, CV_64F);
		
		Mat src_skewXP_prev(3, 4, CV_64F);
		Mat dst_skewXP(3, 4, CV_64F);
		
		getSkewMatrix(src[i], src_skew);
		getSkewMatrix(dst[i], dst_skew);
		
		getExtrinsic(src_skew, P_prev, src_skewXP_prev);
		getExtrinsic(dst_skew, P, dst_skewXP);
		
		// fill out A matrix
		A.at<double>(0,0) = src_skewXP_prev.at<double>(0,0);
		A.at<double>(1,0) = src_skewXP_prev.at<double>(1,0);
		A.at<double>(2,0) = src_skewXP_prev.at<double>(2,0);
		A.at<double>(3,0) = dst_skewXP.at<double>(0,0);
		A.at<double>(4,0) = dst_skewXP.at<double>(1,0);
		A.at<double>(5,0) = dst_skewXP.at<double>(2,0);
		
		A.at<double>(0,1) = src_skewXP_prev.at<double>(0,1);
		A.at<double>(1,1) = src_skewXP_prev.at<double>(1,1);
		A.at<double>(2,1) = src_skewXP_prev.at<double>(2,1);
		A.at<double>(3,1) = dst_skewXP.at<double>(0,1);
		A.at<double>(4,1) = dst_skewXP.at<double>(1,1);
		A.at<double>(5,1) = dst_skewXP.at<double>(2,1);

		A.at<double>(0,2) = src_skewXP_prev.at<double>(0,2);
		A.at<double>(1,2) = src_skewXP_prev.at<double>(1,2);
		A.at<double>(2,2) = src_skewXP_prev.at<double>(2,2);
		A.at<double>(3,2) = dst_skewXP.at<double>(0,2);
		A.at<double>(4,2) = dst_skewXP.at<double>(1,2);
		A.at<double>(5,2) = dst_skewXP.at<double>(2,2);

		A.at<double>(0,3) = src_skewXP_prev.at<double>(0,3);
		A.at<double>(1,3) = src_skewXP_prev.at<double>(1,3);
		A.at<double>(2,3) = src_skewXP_prev.at<double>(2,3);
		A.at<double>(3,3) = dst_skewXP.at<double>(0,3);
		A.at<double>(4,3) = dst_skewXP.at<double>(1,3);
		A.at<double>(5,3) = dst_skewXP.at<double>(2,3);

		Mat u,w,vt;
		cv::SVDecomp(A,w,u,vt, cv::SVD::FULL_UV);
		vt.row(3).reshape(0,4).copyTo(pt4D.col(i));
		
		//cout << "A =\n" << A << "\nw = \n"<< w << "\nu =\n"<< u << "\nvt =\n" << vt << endl;
		
		pt3D_cam0.x = pt4D.at<double>(0,i) / pt4D.at<double>(3,i);
		pt3D_cam0.y = pt4D.at<double>(1,i) / pt4D.at<double>(3,i);
		pt3D_cam0.z = pt4D.at<double>(2,i) / pt4D.at<double>(3,i);
		
		if(pt3D_cam0.z > 0 /* TODO: cam1 also*/)
		{		
			pt3D_loc.push_back(pt3D_cam0);
			/*cout << "\np4D["<<i<<"] = "<<pt4D.col(i).t() << "\np3D =\t"<< pt3D_cam0 << endl;
			cout << "-----------------------------------------------------------------" << endl;*/
		}
	}	
	cout << "\n3D Points[local Coord]: \t" << pt3D_loc.size() << endl;
}
void Vision::getExtrinsic(Mat &skew, Mat &P, Mat &skewXP)
{
	skewXP = skew * P;
}

void Vision::getSkewMatrix(Point2f &pt2D, Mat &skew)
{
	skew.at<double>(0,0) = 0;
	skew.at<double>(0,1) = -1;
	skew.at<double>(0,2) = pt2D.y;
		
	skew.at<double>(1,0) = 1;
	skew.at<double>(1,1) = 0;
	skew.at<double>(1,2) = -pt2D.x;

	skew.at<double>(2,0) = -pt2D.y;
	skew.at<double>(2,1) = pt2D.x;
	skew.at<double>(2,2) = 0;
}


void Vision::triangulateFcn(vector<Point2f> &src, vector<Point2f> &dst, 
							Mat &P_prev, Mat &P, double reprojErr)
{
	//cout << "\n\nP_prev =\n" << P_prev<< "\nP = \n"<< P << endl;
	Mat pt4D;
	Point3f pt3D;
	vector<Point2f> normalized_src, normalized_dst;
	vector<Point3f> pt3D_vec;

	// undistort:
	undistortPoints(src, normalized_src, mK, noArray());
	undistortPoints(dst, normalized_dst, mK, noArray());
	
	triangulatePoints(P_prev, P, normalized_src, normalized_dst, pt4D);
	for (int i = 0; i < pt4D.cols; i++)
	{
		pt3D.x = pt4D.at<float>(0,i) / pt4D.at<float>(3,i);
		pt3D.y = pt4D.at<float>(1,i) / pt4D.at<float>(3,i);
		pt3D.z = pt4D.at<float>(2,i) / pt4D.at<float>(3,i);
		
		pt3D_vec.push_back(pt3D);
	}

	bool triangualtion_succeeded = true;
	/*vector<uchar> status(pt3D_vec.size());
	for (size_t i = 0; i < status.size(); i++)
	{
		status[i] = (pt3D_vec[i].z > 0) ? 1 : 0;
	}
	int nonZero = countNonZero(status);
	
	double percentage = ((double)nonZero / (double)status.size());
	
	if(percentage < 0.5)
	{
		triangualtion_succeeded = false;
		cout 	<< "\n3D Points:\n";
		cout 	<< "\tINLIERS\tNONZERO\t%\tSTATUS\n";
		cout 	<< '\t' << status.size() 	<< '\t' << nonZero 
				<< '\t' << setprecision(4) 	<< percentage*100.0 		
				<< '\t'	<< "REJECTED!" 	
				<< '\n';
	} else
	{
		cout 	<< "\n3D Points:\n";
		cout 	<< "\tINLIERS\tNONZERO\t%\tSTATUS\n";
		cout 	<< '\t' << status.size() 	<< '\t' << nonZero 
				<< '\t' << setprecision(4) 	<< percentage*100.0 		
				<< '\t'	<< "ACCEPTED!"
				<< '\n';
	}*/
	
	if (triangualtion_succeeded)
	{
		// reporjection:
		Mat R(3, 3, CV_64F);
		Mat rvec, tvec;
		
		Mat Rt_prev = mK.inv() * P_prev;
		Mat Rt		= mK.inv() * P;
		
		//cout << "\n\nRt_prev =\n" << Rt_prev << "\nRt = \n"<< Rt << endl;
		
		Rt_prev.rowRange(0,3).colRange(0,3).copyTo(R);
		Rt_prev.col(3).copyTo(tvec);
		
		Rodrigues(R, rvec);
		//cout << "\nrvec =\t" << rvec.t() << "\t tvec =\t"<< tvec.t() << endl;
		vector<Point2f> pt2D_projected;
		projectPoints(pt3D_vec, rvec, tvec, mK, mDistCoef, pt2D_projected);
		
		/*for(size_t i = 0; i < src.size(); i++)
		{
			cout 	<< "src =\t" 		<< src[i] 
					<< "\tnorm_Src =\t" << normalized_src[i] 
					<< "\tprojPt2D =\t" << pt2D_projected[i] 
					<< endl;
		}*/
		//reprojErr = norm(pt2D_projected, src, NORM_L2)/(double)src.size();
		reprojErr = norm(pt2D_projected, normalized_src, NORM_L2)/(double)normalized_src.size();
		
		cout << "\nreprojection Error =\t " << reprojErr << endl;

		double rep_Err_th = 5.0;
		if (reprojErr < rep_Err_th)
		{
			vector<uchar> status(src.size());
			for (size_t i = 0; i < status.size(); i++)
			{
				//status[i] = (norm(src[i]- pt2D_projected[i]) < 20.0);
				status[i] = (norm(normalized_src[i]- pt2D_projected[i]) < 20.0);
			}
			cout << "Keeping \t" << countNonZero(status) << " / "<< status.size() << endl;
			for (size_t i = 0; i < status.size(); i++)
			{
				if(status[i])
				{
					map_3D.push_back(pt3D_vec[i]);
				}
			}
			cout << "3d result sz = \t" << map_3D.size() << endl;
		}
	}
}

void Vision::PoseFromFundamentalMatrix(vector<Point2f> &src, vector<Point2f> &dst)
{
	Mat E;
	try { E = mK.t() * Fundamental_Matrix * mK;}
	catch (cv::Exception const & e) 
	{
		cerr << "\n\nOpenCV exception: \n\n" << e.what() << endl; 
	}
    
    cout << "\nE {from Fund. matrix} =\n " << E << endl;
    
	if (fabsf(determinant(Essential_Matrix))> 1e-07)
	{
		cout << "det(E) != 0 => \t " << determinant(Essential_Matrix)<< endl;
	}
	
	Mat R1(3, 3, CV_64F);
	Mat R2(3, 3, CV_64F);

	Mat t1(3, 1, CV_64F);
	Mat t2(3, 1, CV_64F);
	
	decomposeEToRANDt(Essential_Matrix, R1, R2, t1, t2);
	
	if(determinant(R1) + 1.0 < 1e-09) 
	{
		cout << "det(R) = " << determinant(R1) << "\t flip E's sign..."<< endl;
		Essential_Matrix = -Essential_Matrix;
	}
	
	if(fabsf(determinant(R1)) - 1.0 > 1e-07) 
	{
		cerr << "det(R) != +-1.0, this is not a rotation matrix";
	}
	
	/*Mat P = Mat::eye(3, 4, CV_64F);
	double reprojErr = 0;
	// TODO: P1 has 4 different possibilities... (R1,t1) , (R1,t2) , (R2,t1) , (R2,t2)

	Mat P1_0 = Mat::eye(3, 4, CV_64F);
	R1.copyTo(P1_0.rowRange(0,3).colRange(0,3));
	t1.copyTo(P1_0.rowRange(0,3).col(3));
	triangulateFcn(src, dst, P, P1_0, reprojErr);
	
	Mat P1_1 = Mat::eye(3, 4, CV_64F);
	R1.copyTo(P1_1.rowRange(0,3).colRange(0,3));
	t2.copyTo(P1_1.rowRange(0,3).col(3));
	triangulateFcn(src, dst, P, P1_1, reprojErr);
	
	Mat P1_2 = Mat::eye(3, 4, CV_64F);
	R2.copyTo(P1_2.rowRange(0,3).colRange(0,3));
	t2.copyTo(P1_2.rowRange(0,3).col(3));
	triangulateFcn(src, dst, P, P1_2, reprojErr);
	
	Mat P1_3 = Mat::eye(3, 4, CV_64F);
	R2.copyTo(P1_3.rowRange(0,3).colRange(0,3));
	t1.copyTo(P1_3.rowRange(0,3).col(3));
	triangulateFcn(src, dst, P, P1_3, reprojErr);
	
	// TODO: the judgment is based on reprojection error...
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	Mat R_local_0, t_local_0,
		R_local_1, t_local_1,
		R_local_2, t_local_2,
		R_local_3, t_local_3;
	
	cout << "sc (Fundamental)\t= "<< sc  << endl;
	Mat u,w,vt;
    cv::SVDecomp(Essential_Matrix,w,u,vt, cv::SVD::FULL_UV);
    
    
	Mat W = Mat::zeros(3, 3, CV_64F);
	
	W.at<double>(0,1) = -1;
	W.at<double>(1,0) = 1;
	W.at<double>(2,2) = 1;

	R_local_0 = u * W * vt;
	t_local_0 = u.rowRange(0,3).col(2);
	
	R_local_1 = u * W * vt;
	t_local_1 = -1.0 * u.rowRange(0,3).col(2);

	R_local_2 = u * W.t() * vt;
	t_local_2 = u.rowRange(0,3).col(2);

	R_local_3 = u * W.t() * vt;
	t_local_3 = -1.0 * u.rowRange(0,3).col(2);

	
	// ensure rotation matrix is right handed!
	if (determinant(R_local_0) < 0)
	{
		cout << "det(R_loc_0) = " << determinant(R_local_0) << endl;
		R_local_0 = -1.0 * R_local_0;
		t_local_0 = -1.0 * t_local_0;
	}
	// ensure rotation matrix is right handed!
	if (determinant(R_local_1) < 0)
	{
		cout << "det(R_loc_1) = " << determinant(R_local_1) << endl;
		R_local_1 = -1.0 * R_local_1;
		t_local_1 = -1.0 * t_local_1;
	}

	// ensure rotation matrix is right handed!
	if (determinant(R_local_2) < 0)
	{
		cout << "det(R_loc_2) = " << determinant(R_local_2) << endl;
		R_local_2 = -1.0 * R_local_2;
		t_local_2 = -1.0 * t_local_2;
	}

	// ensure rotation matrix is right handed!
	if (determinant(R_local_3) < 0)
	{
		cout << "det(R_loc_3) = " << determinant(R_local_3) << endl;
		R_local_3 = -1.0 * R_local_3;
		t_local_3 = -1.0 * t_local_3;
	}

	setCurrentPose(R_local_0, t_local_0, T_loc_0);
	Rodrigues(R_local_0, rvec_loc_0);
		
	setCurrentPose(R_local_1, t_local_1, T_loc_1);
	Rodrigues(R_local_1, rvec_loc_1);
		
	setCurrentPose(R_local_2, t_local_2, T_loc_2);
	Rodrigues(R_local_2, rvec_loc_2);
		
	setCurrentPose(R_local_3, t_local_3, T_loc_3);
	Rodrigues(R_local_3, rvec_loc_3);
	
	Mat T_loc_0_inv = Mat::eye(4,4,CV_64F);
	Mat P_loc_0 = Mat::zeros(3,4,CV_64F);
	
	T_loc_0_inv = T_loc_0.inv();
	
	T_loc_0_inv.rowRange(0,3).colRange(0,3).copyTo(P_loc_0.rowRange(0,3).colRange(0,3));
	T_loc_0_inv.rowRange(0,3).col(3).copyTo(P_loc_0.rowRange(0,3).col(3));
	
	
	
	
		if (sc > vMIN_SCALE &&	t_local_0.at<double>(2,0) > t_local_0.at<double>(0,0) &&
								t_local_0.at<double>(2,0) > t_local_0.at<double>(1,0))
		{
			t_f_0 = t_f_prev_0 + sc*(R_f_0*t_local_0);
			//R_f_0 = R_local_0 * R_f_prev_0;
			R_f_0 = R_f_prev_0 * R_local_0;
			Rodrigues(R_f_0, rvec_0);
		}
		if (sc > vMIN_SCALE &&	t_local_1.at<double>(2,0) > t_local_1.at<double>(0,0) &&
								t_local_1.at<double>(2,0) > t_local_1.at<double>(1,0))
		{			
			t_f_1 = t_f_prev_1 + sc*(R_f_1*t_local_1);
			R_f_1 = R_f_prev_1 * R_local_1;
			//R_f_1 = R_local_1 * R_f_prev_1;
			Rodrigues(R_f_1, rvec_1);
		}
		if (sc > vMIN_SCALE &&	t_local_2.at<double>(2,0) > t_local_2.at<double>(0,0) &&
								t_local_2.at<double>(2,0) > t_local_2.at<double>(1,0))
		{			
			t_f_2 = t_f_prev_2 + sc*(R_f_2*t_local_2);
			R_f_2 = R_f_prev_2 * R_local_2;
			//R_f_2 = R_local_2 * R_f_prev_2;
			Rodrigues(R_f_2, rvec_2);
		}
		if (sc > vMIN_SCALE &&	t_local_3.at<double>(2,0) > t_local_3.at<double>(0,0) &&
								t_local_3.at<double>(2,0) > t_local_3.at<double>(1,0))
		{
			t_f_3 = t_f_prev_3 + sc*(R_f_3*t_local_3);
			R_f_3 = R_f_prev_3 * R_local_3;
			//R_f_3 = R_local_3 * R_f_prev_3;
			Rodrigues(R_f_3, rvec_3);
		}*/
		
			/*//### NO CONSTRAINTS ###
			t_f_0 = t_f_prev_0 + sc*(R_f_0*t_local_0);
			//R_f_0 = R_local_0 * R_f_prev_0;	
			R_f_0 = R_f_prev_0 * R_local_0;	
			
			Rodrigues(R_f_0, rvec_0);
		
			t_f_1 = t_f_prev_1 + sc*(R_f_1*t_local_1);
			//R_f_1 = R_local_1 * R_f_prev_1;
			R_f_1 = R_f_prev_1 * R_local_1;	
			Rodrigues(R_f_1, rvec_1);
		
			t_f_2 = t_f_prev_2 + sc*(R_f_2*t_local_2);
			//R_f_2 = R_local_2 * R_f_prev_2;
			R_f_2 = R_f_prev_2 * R_local_2;	
			Rodrigues(R_f_2, rvec_2);
		
			t_f_3 = t_f_prev_3 + sc*(R_f_3*t_local_3);
			//R_f_3 = R_local_3 * R_f_prev_3;
			R_f_3 = R_f_prev_3 * R_local_3;	
				
			Rodrigues(R_f_3, rvec_3);
			//### NO CONSTRAINTS ###*/

	setCurrentPose(R_f_0, t_f_0, T_cam_0);
	//cout << "\nT_0\n = "<< T_cam_0 << endl;
	//cout << "\nT_loc_0\n = "<< T_loc_0 << endl;

	setCurrentPose(R_f_1, t_f_1, T_cam_1);
	//cout << "\nT_1\n = "<< T_cam_1 << endl;
	//cout << "\nT_loc_1\n = "<< T_loc_1 << endl;
	
	setCurrentPose(R_f_2, t_f_2, T_cam_2);
	//cout 	<< "\nT_loc_2\n = "<< T_loc_2 
	//		<< "\nT_2\n = "<< T_cam_2 << endl;
	
	setCurrentPose(R_f_3, t_f_3, T_cam_3);
	//cout << "\nT_3\n = "<< T_cam_3 << endl;
	//cout << "\nT_loc_3\n = "<< T_loc_3 << endl;
	
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
}

void Vision::decomposeEToRANDt(Mat &E, Mat &R1, Mat &R2, Mat &t1, Mat &t2)
{
	Mat u,w,vt;
	cv::SVDecomp(E,w,u,vt, cv::SVD::MODIFY_A);
	
	Mat W = Mat::zeros(3, 3, CV_64F);
	
	W.at<double>(0,1) = -1;
	W.at<double>(1,0) = 1;
	W.at<double>(2,2) = 1;

	R1 = u * W 		* vt;
	R2 = u * W.t() 	* vt;
	
	t1 = u.col(2);
	t2 = -u.col(2);
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

}//namespace ORB_VISLAM
