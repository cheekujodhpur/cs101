#include <iostream>
#include <fstream>
#include <cv.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <stdlib.h>
using namespace std;

cv::Mat normalisedmat (int row,int col,int cond)
{ cv::Mat S(row,col,CV_64F);
  for (int i=0;i<col;i++)
{
 float sum=0.0;

 for(int j=0;j<row;j++)
 {
 S.at<float>(j,i)=rand()%100;
 sum+=S.at<float>(j,i);
 }
 
 for(int j=0;j<row;j++)
 {
 S.at<float>(j,i)/=sum;
 }
}
return S;
}
int main(int argc,char**argv)
{
	//Step 1:Load features file
	fstream fin;
	fin.open("features.csv",ios::in);
	if(!fin)
	{
		cout << "Dieeee!" << endl;
		return -1;
	}
	char c;
	while(fin.get(c))
	{
		cout << c;
	}
	cv::waitKey(0);
	//Step 2:Train HMM
		//Step 2.1:Write a HMM class for ease
	return 0;
}
