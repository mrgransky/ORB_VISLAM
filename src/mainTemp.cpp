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
#include "temp.h"

using namespace std;
using namespace cv;
using namespace Eigen;


int main(int argc, char** argv )
{
	A myClass(1.5,2.5);
	
	cout << "Started 2 threads. Waiting for them to finish..." << endl;
	for (int i = 0; i < 10; i++)
	{
	
		myClass.run();
		cout << "-----------------------------------------" << endl;
    }
    cout << "Threads finished." << endl;
    return 0;
}


