#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>
#include <cstdio>
using namespace std;

#define INVSQRT2 1.414213562373095048801688724209 * 0.5 //reciprocal of square root of 2
#define PI 3.1415926535897932384626433832795	//value of pi
#define SW_SIZE 8	//size of sliding window
#define OVERLAP 0.5	//extent of overlap
#define APP_SIZE (int)((1-OVERLAP)*SW_SIZE)	//extent by which to slide the window
#define INV2SW_SIZE (1.0)/(SW_SIZE)	//reciprocal of window size
#define IMG_ROW 512	//definition of row size of image
#define IMG_COL 512//definition of column size of image
#define DCT_NUM1 (int)((IMG_ROW-SW_SIZE+APP_SIZE)/APP_SIZE)	//total number of DCT matrices the image will produce
#define DCT_NUM2 (int)((IMG_COL-SW_SIZE+APP_SIZE)/APP_SIZE)	//total number of DCT matrices the image will produce
double DCT[DCT_NUM1*DCT_NUM2][SW_SIZE][SW_SIZE];	//declaration of matrix of matrices of DCT coefficients
int main(char* argv[], int argc)
{

	FILE *fp = fopen("pixels.txt", "w+");	//file to store pixels of output image
	FILE *fp2 = fopen("Original_pixels.txt", "w+");	//file to store pixels of original image
	if (fp == NULL||fp2==NULL)	//check for null pointer 
	{
		cout << "Error in file opening process." << endl;
		return -1;
	}
	cv::Mat image;	//input image matrix
	image = cv::imread("logo.png", CV_LOAD_IMAGE_GRAYSCALE);	//read input image
	if (!image.data)
	{
		cout << "Could not open image." << endl; 
		return -1;
	}
	cout << "Rows:" << image.rows << "\nColumns:" << image.cols << endl;
	if (image.rows != IMG_ROW || image.cols != IMG_COL)	//check if image size matches with definition
		cv::resize(image, image, cv::Size2d(IMG_ROW, IMG_COL));
	cv::imshow("image", image);			//display input image
	//attempting to find DCT coefficients by separate iterations over rows and columns
	double tmp[SW_SIZE][SW_SIZE];		//an array to store the partial DCT coefficients 
	for (int i = 0; i < DCT_NUM1; i++)	//for loop to iterate over the DCT matrices, each of which is a part of a matrix itself of DCT_NUM1*DCT_NUM2 elements
	{
		for (int j = 0; j < DCT_NUM2; j++)	//for loop to iterate over the DCT matrices, each of which is a part of a matrix itself of DCT_NUM1*DCT_NUM2 elements
		{	
			int num = DCT_NUM1 * i + j;		//the number in which order this DCT matrix will appear in the matrix of DCT matrices 
			int start_row = APP_SIZE * i;	//the starting point(row) of the sliding window in the image 
			int start_col = APP_SIZE * j;	//the starting point(column) of the sliding window in the image
			int end_row = start_row + SW_SIZE;	//the opposite corner of sliding window(row) in image
			int end_col = start_col + SW_SIZE;	//the opposite corner of sliding window (column) in image
			double alpha;				
			for (int u = 0; u < SW_SIZE; u++)
			{
				for (int v = 0; v < SW_SIZE; v++)
				{
					tmp[u][v] = 0;
					alpha = (v != 0) ? 1 : INVSQRT2;
					for (int x = start_col, y = start_row + u; x < end_col; x++)
						tmp[u][v] += (alpha*cos(PI *(2 * x + 1)*v / (2 * SW_SIZE))*((int)image.at<uchar>(y, x)));
				}
			}
			for (int u = 0; u<SW_SIZE; u++)
			{
				for (int v = 0; v<SW_SIZE; v++)
				{
					DCT[num][u][v] = 0.0;
					alpha = (v != 0) ? 1 : INVSQRT2;
					for (int y = 0, x = u; y < SW_SIZE; y++)
						DCT[num][u][v] += (alpha*cos(PI *(2 * y + 1)*u / (2 * SW_SIZE))*(tmp[y][x]));
					DCT[num][u][v] = DCT[num][u][v]/ (SW_SIZE*SW_SIZE);
				}
			}
		}
	}
	for (int i = 0; i < DCT_NUM1*DCT_NUM2; i++)
	{
		for (int j = (5 * SW_SIZE) / 64; j < (14 * SW_SIZE) / 64; j++)
		{
			if (j < SW_SIZE)for (int k = 0; k <= j; k++)DCT[i][k][j - k] = 0;
			else for (int k = SW_SIZE - 1; k >= (j - SW_SIZE + 1); k--)DCT[i][k][j - k] = 0;
		}
	}
	cv::Mat output = image.clone();
	for (int i = 0; i < DCT_NUM1; i++)
	{

		for (int j = 0; j < DCT_NUM2; j++)
		{
			int num = DCT_NUM1 * i + j;
			int start_row = APP_SIZE * i;
			int start_col = APP_SIZE * j;
			int end_row = start_row + SW_SIZE;
			int end_col = start_col + SW_SIZE;
			double alpha;
			for (int x = start_row; x<end_row; x++)
			{
				for (int y = start_col; y<end_col; y++)
				{
					double temp = 0;
					tmp[x - start_row][y - start_col] = 0;
					for (int u = 0, v = x - start_row; u < SW_SIZE; u++)
					{
						alpha = (u == 0) ? 0.5 : 1;
						tmp[x - start_row][y - start_col] += (alpha*cos(PI *(2 * u + 1)*(y - start_col) / (2 * SW_SIZE))*DCT[num][v][u]);
					}
				}
			}
			for (int x = start_row; x<end_row; x++)
			{
				for (int y = start_col; y<end_col; y++)
				{
					double temp = 0;
					for (int v = 0, u = y - start_col; v < SW_SIZE; v++)
					{
						alpha = (v == 0) ? 0.5 : 1;
						temp += (alpha*cos(PI *(2 * v + 1)*(x - start_row) / (2 * SW_SIZE))*(tmp[v][u]));
					}
					output.at<uchar>(x, y) = (int)(temp);
				}
			}
		}
	}
	
	cv::imshow("output", output);
	for (int i = 0; i < IMG_ROW; i++)
	{
		for (int j = 0; j < IMG_COL; j++)
		{
			fprintf(fp, "%d ", (int)output.at<uchar>(i, j));
			fprintf(fp2, "%d ", (int)image.at<uchar>(i, j));
		}
		fprintf(fp, "\n");
		fprintf(fp2, "\n");
	}
	fclose(fp);
	fclose(fp2);
	cv::imwrite("output_DCT.jpg", output);
	cv::waitKey(0);
	return 0;
}

