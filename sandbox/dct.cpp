/*
This program detects faces in continuous video stream from webcam and shows the cropped faces,

also storing them in consecutively numbered files

Copyright (C) <2014> <Group 02: Kumar Ayush, Reebhu Bhattacharyya, Kshitij Bajaj, Keshav Srinivasan>

This program is free software: you can redistribute it and/or modify

it under the terms of the GNU General Public License as published by

the Free Software Foundation, either version 3 of the License, or

(at your option) any later version.

This program is distributed in the hope that it will be useful,

but WITHOUT ANY WARRANTY; without even the implied warranty of

MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the

GNU General Public License for more details.

You should have received a copy of the GNU General Public License

along with this program. If not, see <http://www.gnu.org/licenses/>.

*/
#include "FaceDetection.h"
#include <opencv2\core\core.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\opencv.hpp>
#include<iostream>
#include <math.h>
using namespace std;

#define INVSQRT2 1.414213562373095048801688724209 * 0.5
#define INV16 (1.0)/(16.0)
#define PI 3.14159

static double alpha(int i)
{
	if (i == 0)
		return INVSQRT2 * 0.5;
	return 0.5;
}

void foo(){};	//to stay

int main(char* argv[], int argc)
{

	cout << "videoFaceDetect.cpp Copyright(C) <2014> <Group 02: Kumar Ayush, Reebhu Bhattacharyya, Kshitij Bajaj, Keshav Srinivasan>\n\n";
	cout << "This program comes with ABSOLUTELY NO WARRANTY.\n\n";
	cout << "This is free software, and you are welcome to redistribute it\n\n";
	cout << "under certain conditions.";
	cout << "Press Esc to stop streaming." <<  endl;
	//videoFaceDetect();//call a function to open videostream(webcam), detect faces in it and show cropped

	cv::Mat image;
	image = cv::imread("lena.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	double result[64];

	//The cosine in window
	double cosine[8][8];
	for (int i = 0; i < 8; i++)
		for (int j = 0; j < 8; j++)
			cosine[i][j] = cos(PI*j*(2.0*i + 1)*INV16);

	//the window as 8x8 is implemented below. We need to slide this window
	for (int y = 0; y < 8; y++)
	{
		for (int x = 0; x < 8; x++)
		{
			result[y * 8 + x] = 0;
			for (int u = 0; u < 8; u++)
				for (int v = 0; v < 8; v++)
					result[y * 8 + x] = alpha(u)*alpha(v)*(int)image.at<char>(u, v) * cosine[u][x] * cosine[v][y];
		}
	}

	for (int i = 0; i < 64; i++)
		cout << result[i] << " ";

	//cv::imshow("output", image);
	foo();
	return 0;
}

