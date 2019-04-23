#include <opencv2/opencv.hpp>
#include "opencv2/opencv_modules.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdlib.h>     /* exit, EXIT_FAILURE */
#include <stdio.h>
#include <iomanip>
#include <iostream>
#include <string>
#include <limits>
#include <fstream>
#include <vector>
#include <math.h> 

using namespace std;
using namespace cv;
float d_inv1 = 1;

void decomposeHomography(Mat inpHomography, Mat inp_camera_matrix)
{
	vector<Mat> Rs_decomp, ts_decomp, normals_decomp;
	
	int solutions = decomposeHomographyMat(inpHomography, inp_camera_matrix, 
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
		
		cout 	<< "tvec decom \t =" << ts_decomp[i].t() 
      			<< "\nscaled by d \t =" << factor_d1 * ts_decomp[i].t() 
      			<< endl;
		
		cout << "plane normal decom \t =" << normals_decomp[i].t() << endl;
		cout << "------------------------------------------------------" << endl;
	}
}

void visualizeKeyPoints(Mat &output_image, 	vector<KeyPoint> keyP1, vector<KeyPoint> keyP2, float sc)
{
	for (size_t i = 0; i < keyP1.size(); i++)
	{
		Point2f pt_ref(sc*keyP1[i].pt.x,sc*keyP1[i].pt.y);
		cv::circle(output_image,pt_ref,1,Scalar(1,200,220),FILLED);
	}
	
	for (size_t i = 0; i < keyP2.size(); i++)
	{
		Point2f pt_curr(.5*output_image.cols+sc*keyP2[i].pt.x,sc*keyP2[i].pt.y);
		cv::circle(output_image,pt_curr,1,Scalar(199,199,20),FILLED);
	}
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

Mat getclock(Mat img, Point2f point, int window_size)
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
		block_2 = getclock(img_2, kp2_vec[i], window_size);
		
		//cout << "block_current =\n"<< block_2<< endl;
		vector<int> ssd_vec;
		vector<int> ssd_tmp_vec;
		vector<Point2f> kp1_vec;
		
		//for (size_t j = 0; j < 4; j++)
		for (size_t j = 0; j < keyP1.size(); j++)
		{
			kp1_vec.push_back(keyP1[j].pt);
			block_1 = getclock(img_1, kp1_vec[j], window_size);
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

Mat getHomography(const vector<Point2f> &p_ref, const vector<Point2f> &p_mtch)
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

void printHelp(char ** argv)
{
	cout	<< "\n\nNOT ENOUGH ARGUMENT PROVIDED!!\n\nSyntax:"		
			<< argv[0]	
			<< " [/path/2/image Left] [/path/2/image Right] [/path/2/setting file]"
			<< "\n\nEXAMPLE:\n\n" 	
			<< argv[0] 
			<< " /home/xenial/Datasets/CIVIT/Dec_14/VideoFiles/seq1_short/frames/rgb/frame_2000.jpg /home/xenial/Datasets/CIVIT/Dec_14/VideoFiles/seq1_short/frames/rgb/frame_2025.jpg /home/xenial/WS_Farid/orb_slam2_TUT/settingFiles/civit.yaml\n\n"
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
    
    cout << endl << "Camera Parameters: " << endl;
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

int main( int argc, char** argv )
{
	if( argc != 4 )
	{
		printHelp(argv);
		return -1; 
	}
	Mat imgR = imread( argv[1], CV_LOAD_IMAGE_GRAYSCALE );
	Mat imgC = imread( argv[2], CV_LOAD_IMAGE_GRAYSCALE );

	if( !imgR.data || !imgC.data )
	{
		printf(" --(!) Error reading images \n"); 
		return -1;
	}

	if(imgR.channels()<3) //this should be always true
	{
		cvtColor(imgR,imgR,CV_GRAY2BGR);
	}
		
	if(imgC.channels()<3) //this should be always true
	{
		cvtColor(imgC,imgC,CV_GRAY2BGR);
	}

	Mat camera_matrix = Mat::eye(3,3,CV_32F);
	Mat DistCoef(4,1,CV_32F);

	getCameraParamerters(argv[3], camera_matrix, DistCoef);
	
	float scale = .48;
	// imgR:
	int w1,h1;
	int w1_scaled, h1_scaled;
	w1 = imgR.cols;
	h1 = imgR.rows;
	
	w1_scaled = w1*scale;
	h1_scaled = h1*scale;
	
	
	// imgC:
	int w2,h2;
	int w2_scaled, h2_scaled;
	w2 = imgC.cols;
	h2 = imgC.rows;
	
	w2_scaled = w2*scale;
	h2_scaled = h2*scale;

	Mat temp1;
    Mat temp2;

	Mat output_image = Mat::zeros(cv::Size(w1_scaled+w2_scaled, h1_scaled), CV_8UC3);

	resize(imgR,temp1, Size(w1_scaled, h1_scaled));
    temp1.copyTo(output_image(Rect(0, 0, w1_scaled, h1_scaled)));
	
	resize(imgC,temp2, Size(w2_scaled, h2_scaled));
	temp2.copyTo(output_image(Rect(w1_scaled, 0, w2_scaled, h2_scaled)));

	const string &frameWinName = "ORB_VISLAM | Frames";
	namedWindow(frameWinName);
	
	Ptr<FeatureDetector> 		detector 	= ORB::create();
    Ptr<DescriptorExtractor> 	extractor 	= ORB::create();
	vector<KeyPoint> keyR, keyC;
	
	Mat desR, desC;

	// ref img:
	detector->detect(imgR, keyR);
	extractor->compute(imgR, keyR, desR);
    
    // curr img:
	detector->detect(imgC, keyC);
	extractor->compute(imgC, keyC, desC);

	visualizeKeyPoints(output_image, keyR, keyC, scale);
	
	// 1. matched c2r
	// current is bigger loop
	cout << "\n\nReference \t\t<<<---\t\t Current\n" << endl;
	vector <pair<int,int>> matchedC2R;
	matchedC2R = getMatches(imgR, imgC, keyR, keyC);
	//cout << "matches C2R =\t"<<matchedC2R.size()<< endl;
	
	for (size_t k = 0; k < matchedC2R.size(); k++)
	{
		int parent 	= matchedC2R[k].first;
		int match 	= matchedC2R[k].second;
		/*cout 	<< "kp_p[" 
				<< setw(5) << parent 	
				<< setw(5) << "] = (" 	
				<< keyC[parent].pt
				<< "), " 	
				<< right 	<< setw(15) << "kp_m["
				<< setw(5)	<< match	
				<< setw(5) << "] = ("		
				<< right 	<< setw(10) << keyR[match].pt << ")"
				<< endl;*/
	}
	
//	cout<<setfill('-')<<setw(80)<<"-"<<endl;

	// 2. matched r2c
	// ref is bigger loop
	cout << "\n\nReference \t\t--->>>\t\t Current\n" << endl;
	vector <pair<int,int>> matchedR2C;
	matchedR2C = getMatches(imgC, imgR, keyC, keyR);
	//cout << "matches R2C =\t"<<matchedR2C.size()<< endl;
	
	for (size_t k = 0; k < matchedR2C.size(); k++)
	{
		int parent 	= matchedR2C[k].first;
		int match 	= matchedR2C[k].second;
		/*cout 	<< "kp_p[" 
				<< setw(5) << parent 	
				<< setw(5) << "] = (" 	
				<< keyR[parent].pt
				<< "), " 	
				<< right 	<< setw(15) << "kp_m["
				<< setw(5)	<< match	
				<< setw(5) 	<< "] = ("		
				<< right 	<< setw(10) << keyC[match].pt << ")"
				<< endl;*/
	}
	
	cout<<setw(80)<<"-"<<endl;

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
		pt_ref.push_back(keyR[parent].pt);
		pt_matched.push_back(keyC[match].pt);
		
		/*cout 	<< "p[" 
				<< setw(5) 	<< parent 	
				<< "]=" 	<< keyR[parent].pt << " , m[" 
				<< setw(5)	<< match 	<< "] = "
				<< keyC[match].pt
				<< endl;*/
		visualizeMatches(output_image, keyR[parent].pt, keyC[match].pt, scale);
	}
	cout << "pt_ref sz =\t "<<pt_ref.size() << "\t pt_matched sz = \t"<< pt_matched.size()<< endl;
	Mat H_opencv, H_opencv_R;
	Mat H;

	
	//H_opencv_R 	= findHomography(pt_ref, pt_matched, RANSAC);
	//cout << "\nH_openCV (RANSAC) =\n"	<< H_opencv_R 	<< endl;

	cout << "\n\n" << endl;
	cout << "#########################################################################" << endl;
	cout << "\t\t\tHOMOGRAPHY DECOMPOSITION"											<< endl;
	cout << "#########################################################################" << endl;
	

	//! [OPENCV-HOMOGRAPHY]
	cout << "\n\n1. Estimated by opencv H:" << endl;
	H_opencv 	= findHomography(pt_ref, pt_matched);
	cout << "\nH_openCV (regular) =\n"	<< H_opencv		<< endl;
	
	decomposeHomography(H_opencv, camera_matrix);
	//! [OPENCV-HOMOGRAPHY]

    
    //! [user-defined-homography]
	cout << "\n\n2. Estimated by user-function Homography:" << endl;
    //! [estimate-homography]
    H = getHomography(pt_ref,pt_matched);
	cout << "\nH [own] = \n" << H << endl;

	decomposeHomography(H, camera_matrix);
    //! [user-defined-homography]
	

	
	stringstream s_i1;
	int f_k = 1;
	int f_k_plus_1 = 2;
	s_i1 << "Reference Frame = "+to_string(f_k);
		
	cv::putText(output_image,s_i1.str(),cv::Point(.01*output_image.cols,.1*output_image.rows),
    				cv::FONT_HERSHEY_PLAIN, 1, Scalar::all(255),2,LINE_4);
		
	stringstream s_i2;
	s_i2 << "Current Frame = "+to_string(f_k_plus_1);
		
	cv::putText(output_image,s_i2.str(),cv::Point(.01*output_image.cols+.5*output_image.cols,
												.1*output_image.rows),
    				cv::FONT_HERSHEY_PLAIN, 1, Scalar::all(255),2,LINE_4);

	imshow(frameWinName,output_image);
	waitKey();
	return 0;	
}
