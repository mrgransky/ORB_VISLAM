#include <GL/glew.h>
#include <GL/glut.h>
#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/glext.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>

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


class A
{
	public:
		A(double in_a, double in_b );
		double c;
		void func1();
		void func2(double x);
    	void run();
    	
   	private:
   		double a, b;
};

