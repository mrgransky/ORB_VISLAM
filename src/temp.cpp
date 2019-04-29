#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry> 

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <opencv2/opencv.hpp>

#include <iomanip>

#include <iostream>

#include <string>

#include <limits>

#include <fstream>
#include <vector>
#include <set>
#include <map>
#include <cmath>

#include <algorithm>
#include <thread>
#include <chrono>
#include <mutex>
#include "temp.h"

using namespace std;
using namespace cv;
using namespace Eigen;



A::A(double in_a, double in_b)
{
	cout << "constructor called to initialize!"<< endl;
	a = in_a;
	b = in_b;
}

void A::func1()
{
	c = a + b;
	cout << "func1 result =\t" <<c<< endl;
}

void A::func2(double x)
{
	double z = c;
	cout << "visualizing  result of func1\t" << z << endl;
}

void A::run()
{
	thread t(&A::func1, this);
	thread r(&A::func2, this, c);
	t.join();
	r.join();
}
