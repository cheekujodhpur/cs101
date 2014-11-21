#include <iostream>
#include <fstream>
#include "CvHMM.h"
#include <opencv2/core/core.hpp>
#include <opencv2/contrib/contrib.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <cmath>
#include <string>
using namespace std;
using namespace cv;
#define PI 3.141592653589793238
#define INVSQRT2 1.414213562373095048801688724209 * 0.5 //reciprocal of square root of 2
#define SW_SIZE 8       //size of sliding window
#define OVERLAP 0.5     //extent of overlap
#define APP_SIZE (int)((1-OVERLAP)*SW_SIZE)     //extent by which to slide the window
#define INV2SW_SIZE (1.0)/(SW_SIZE)     //reciprocal of window size
#define IMG_ROW 512     //definition of row size of image
#define IMG_COL 512//definition of column size of image
#define DCT_NUM1 (int)((IMG_ROW-SW_SIZE+APP_SIZE)/APP_SIZE)     //total number of DCT matrices the image will produce
#define DCT_NUM2 (int)((IMG_COL-SW_SIZE+APP_SIZE)/APP_SIZE)     //total number of DCT matrices the image will produce
#define EPSILON 1E-30

class Person
{
	static unsigned long int num;
	unsigned long int id;
	unsigned int num_of_faces;
public:
	Person();
	Person(unsigned int nof);
	unsigned long int getId(){ return id; }
	bool train(Mat &seq);
};

unsigned long int Person::num = 0;

Person::Person()
{
	id = ++num;
	num_of_faces = 10;
}

Person::Person(unsigned int nof)
{
	id = ++num;
	num_of_faces = nof;
}


double DCT[DCT_NUM1*DCT_NUM2][SW_SIZE][SW_SIZE];        //declaration of matrix of matrices of DCT coefficients

bool Person::train(Mat &seq)
{
	for (int iter = 0; iter<num_of_faces; iter++)
	{
		//declare a filestream which reads images in folder with name same as id and path as specified
		string filename = "";
		filename += to_string(id) + "/" + to_string(iter) + ".jpg";
		Mat image;
		image = cv::imread(filename, CV_LOAD_IMAGE_GRAYSCALE);
		if (!image.data)
		{
			cout << "Yo mama so blind she don't read images" << endl;
			return -1;
		}
		if (image.rows != IMG_ROW || image.cols != IMG_COL)     //check if image size matches with definition
			cv::resize(image, image, cv::Size2d(IMG_ROW, IMG_COL));

		//dct the file, create observation
		double tmp[SW_SIZE][SW_SIZE];           //an array to store the partial DCT coefficients 
		for (int i = 0; i < DCT_NUM1; i++)      //for loop to iterate over the DCT matrices, each of which is a part of a matrix itself of DCT_NUM1*DCT_NUM2 elements
		{
			for (int j = 0; j < DCT_NUM2; j++)      //for loop to iterate over the DCT matrices, each of which is a part of a matrix itself of DCT_NUM1*DCT_NUM2 elements
			{
				int num = DCT_NUM1 * i + j;             //the number in which order this DCT matrix will appear in the matrix of DCT matrices 
				int start_row = APP_SIZE * i;   //the starting point(row) of the sliding window in the image 
				int start_col = APP_SIZE * j;   //the starting point(column) of the sliding window in the image
				int end_row = start_row + SW_SIZE;      //the opposite corner of sliding window(row) in image
				int end_col = start_col + SW_SIZE;      //the opposite corner of sliding window (column) in image
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
						DCT[num][u][v] = DCT[num][u][v] / (SW_SIZE*SW_SIZE);
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
		//store observation sequence
		int counter,kk,ii,jj;
                for (ii = 0; ii < DCT_NUM1*DCT_NUM2; ii++)
                {
                        for (kk = 0, counter = 0; kk < 5; kk++)
                        {
                                for (jj = 0; jj <= kk; jj++, counter++)
                                {
                                        if (kk % 2 == 0) (seq.at<Vec<double,15>>(iter, ii))[counter] = (double)DCT[ii][jj][kk - jj];
                                        else (seq.at<Vec<double,15>>(iter, ii))[counter] = (double)DCT[ii][kk - jj][jj];
                                }
                        }
                }
	}

	//train hmm
	//store hmm, file has header denoting training maximized likelihood
}

int main(int argc, char **argv)
{
	int nof;
	cout << "Number of faces?: " << endl;
	cin >> nof;
	int nos;
	cout << "Number of states?: " << endl;
	cin >> nos;
	Person foo(nof);	//person declared with number of faces
	Mat seq(nof,DCT_NUM1*DCT_NUM2,CV_64FC(15));
	CvHMM hmm;
	Mat trans;
	Mat init;
	Mat mean;
	Mat var;
	hmm.getUniformModel(nos,trans,init,mean,var);	
	
	//print inits, for debug
	cout << "Transition Matrix:" << endl;
	cout << trans << endl;
	cout << "Init Matrix:" << endl;
	cout << init << endl;
	cout << "Mean Matrix:" << endl;
	cout << mean  << endl;
	cout << "Variance Matrix:" << endl;
	cout << var << endl;

	foo.train(seq);
	for(int i = 0;i<seq.rows;i++)
		for(int j = 0;j<5;j++)
			cout << hmm.getProb(mean,var,seq.at<Vec<double,15>>(i,j)) << endl;

	//print sequence, for debug
	/*cout << "Sequence matrix" << endl;
	for(int i = 0;i<seq.rows;i++)
	{
		for(int j = 0;j<seq.cols;j++)
		{
			cout << "(";
			for(int k = 0;k<15;k++)
				cout << seq.at<Vec<double,15>>(i,j)[k] << ",";
			cout << ") ";
		}
		cout << endl << "---------" << endl;
	}*/
	//create a video stream

	//for each frame, run over all hmm we have in the parent folder

	//calculate likelihoods

	//report likelihoods in a table for each person
	return 0;
}
