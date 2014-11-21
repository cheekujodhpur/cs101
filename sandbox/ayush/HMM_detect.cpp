#include <iostream>
#include <fstream>
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
#define SW_SIZE 32       //size of sliding window
#define OVERLAP 0.5     //extent of overlap
#define APP_SIZE (int)((1-OVERLAP)*SW_SIZE)     //extent by which to slide the window
#define INV2SW_SIZE (1.0)/(SW_SIZE)     //reciprocal of window size
#define IMG_ROW 96     //definition of row size of image
#define IMG_COL 96//definition of column size of image
#define DCT_NUM1 (int)((IMG_ROW-SW_SIZE+APP_SIZE)/APP_SIZE)     //total number of DCT matrices the image will produce
#define DCT_NUM2 (int)((IMG_COL-SW_SIZE+APP_SIZE)/APP_SIZE)     //total number of DCT matrices the image will produce
#define EPSILON 1E-200
double DCT[DCT_NUM1*DCT_NUM2][SW_SIZE][SW_SIZE];        //declaration of matrix of matrices of DCT coefficients
class NRowVector
{
	Mat* v;
	int size;
public:
	NRowVector()
	{
		size = 3;
		v = new Mat(1, 3, CV_64F, Scalar(0));
		if (v == NULL)
		{
			cout << "Null pointer returned during initialization of NRowVector." << endl;
		}
	}
	NRowVector(int s, double *arr)
	{
		size = s;
		v = new Mat(1, size, CV_64F);
		if (v == NULL)
		{
			cout<<"Null pointer returned during initialization of NRowVector."<< endl;
		}
		for (int j = 0; j < size; j++)
			v->at<double>(0, j) = arr[j];

	}
	NRowVector(int s)
	{
		size = s;
		v = new Mat(1, s, CV_64F, Scalar(0));
		if (v == NULL)
		{
			cout << "Null pointer returned during initialization of NRowVector." << endl;
		}
	}
	double getElement(int i) const
	{
		return v->at<double>(0, i);
	}
	void setElement(int pos, double val)
	{
		v->at<double>(0, pos) = val;
	}
	int getSize() const
	{
		return size;
	}
	NRowVector(const NRowVector& v1)
	{
		size = v1.size;
		v = new Mat(1, size, CV_64F);
		if (v == NULL)
		{
			cout << "Null pointer returned during initialization of NRowVector." << endl;
		}
		for (int j = 0; j < size; j++)
			setElement(j, v1.getElement(j));
	}
	~NRowVector()
	{
		delete v;
		v = NULL;
	}
	NRowVector operator=(double d)
	{
		double tmp = sqrt(d*d / size);
		for (int j = 0; j < size; j++)setElement(j, tmp);
		return *this;
	}
	NRowVector operator=(const NRowVector &v1)
	{
		size = v1.size;
		v = new Mat(1, size, CV_64F);
		if (v == NULL)
		{
			cout << "Null pointer returned during initialization of NRowVector." << endl;
		}
		for (int j = 0; j < size; j++)setElement(j, v1.getElement(j));
		return *this;
	}
	Mat* convertToMat()
	{
		int r = 1;
		int c = size;
		Mat* m = new Mat(r, c, CV_64F);
		if (m == NULL)
		{
			cout << "Null pointer returned during initialization of Mat." << endl;
		}
		for (int j = 0; j < size; j++)
		{
			m->at<double>(0, j) = getElement(j);
		}
		return m;
	}
};
class NColVector
{
	Mat* v;
	int size;
public:
	NColVector()
	{
		size = 3;
		v = new Mat(3, 1, CV_64F, Scalar(0));
		if (v == NULL)
		{
			cout << "Null pointer returned during initialization of NColVector." << endl;
		}
	}
	NColVector(int s, double *arr)
	{
		size = s;
		v = new Mat(size, 1, CV_64F);
		if (v == NULL)
		{
			cout << "Null pointer returned during initialization of NColVector." << endl;
		}
		for (int j = 0; j < size; j++)
			v->at<double>(j, 0) = arr[j];

	}
	NColVector(int s)
	{
		size = s;
		v = new Mat(s, 1, CV_64F, Scalar(0));
	}
	NColVector(int s, double d)
	{
		size = s;
		v = new Mat(s, 1, CV_64F, Scalar(d));
		if (v == NULL)
		{
			cout << "Null pointer returned during initialization of NColVector." << endl;
		}
	}
	double getElement(int i) const
	{
		return v->at<double>(i, 0);
	}
	void setElement(int pos, double val)
	{
		v->at<double>(pos, 0) = val;
	}
	int getSize() const
	{
		return size;
	}
	double getNorm() const
	{
		double norm = 0.0;
		for (int j = 0; j < size; j++)
		{
			norm += getElement(j)*getElement(j);
		}
		return sqrt(norm);
	}
	NColVector(const NColVector &v1)
	{
		size = v1.size;
		v = new Mat(size, 1, CV_64F);
		if (v == NULL)
		{
			cout << "Null pointer returned during initialization of NColVector." << endl;
		}
		for (int j = 0; j < size; j++)
			setElement(j, v1.getElement(j));
	}
	NColVector(const NRowVector* v1)
	{
		size = v1->getSize();
		v = new Mat(size, 1, CV_64F);
		if (v == NULL)
		{
			cout << "Null pointer returned during initialization of NColVector." << endl;
		}
		for (int j = 0; j < size; j++)
			setElement(j, v1->getElement(j));
	}
	NRowVector transpose()
	{
		NRowVector v2(getSize());
		for (int j = 0; j < getSize(); j++)v2.setElement(j, getElement(j));
		return v2;
	}
	~NColVector()
	{
		delete v;
	}
	NColVector& operator=(double d)
	{
		double tmp = sqrt(d*d / size);
		for (int j = 0; j < size; j++)setElement(j, tmp);
		return *this;
	}
	NColVector& operator=(const NColVector& v1)
	{
		size = v1.size;
		v = new Mat(size, 1, CV_64F);
		if (v == NULL)
		{
			cout << "Null pointer returned during initialization of NColVector." << endl;
		}
		for (int j = 0; j < size; j++)setElement(j, v1.getElement(j));
		return *this;
	}
	Mat convertToMat()
	{
		Mat m(*v);
		return m;
	}

};
class Obs
{
	Mat* obs;
public:
	Obs()
	{
		obs = new Mat(1, 15, CV_64F, Scalar(0));
	}
	Obs(double D[][SW_SIZE][SW_SIZE],int size)
	{
		obs = new Mat(15, size, CV_64F, Scalar(0));
		for (int i = 0; i < size; i++)
		{
			for (int k = 0, counter = 0; k < 5; k++)
			{
				for (int j = 0; j <= k; j++, counter++)
				{
					if (k % 2 == 0) obs->at<double>(counter, i) = (double)(D[i][k-j][j]);
					else obs->at<double>(counter, i) = (double)D[i][j][k-j];
				}
			}
			double max = -DBL_MAX, min = DBL_MAX;
			for (int iter = 0; iter < 15; iter++)
			{
				if (obs->at<double>(iter, i) > max)max = obs->at<double>(iter, i);
				else if (obs->at<double>(iter, i) < min)min = obs->at<double>(iter, i);
			}
			for (int iter = 0; iter < 15; iter++)
			{
				obs->at<double>(iter, i) = obs->at<double>(iter, i) / (max - min);
			}
		}

	}
	void disp()
	{
		cout.setf(ios::fixed);
		for (int c = 0; c < obs->cols; c++)
		{
			for (int r = 0; r < 15; r++)
			{
				cout << obs->at<double>(r, c)<<"\t";
			}
			cout << endl;
		}
		cout.unsetf(ios::fixed);
	}
	Obs(const Obs &o)
	{
		obs = new Mat(o.obs->clone());
	}
	Obs& operator=(const Obs &o)
	{
		obs = new Mat(o.obs->clone());
		return *this;
	}
	~Obs()
	{
		delete obs;
		obs = NULL;
	}
	void setObs(double DCT[][SW_SIZE][SW_SIZE], int size)
	{
		obs = new Mat(15,size, CV_64F);
		for (int i = 0; i < size; i++)
		{
			for (int k = 0, counter = 0; k < 5; k++)
			{
				for (int j = 0; j <= k; j++, counter++)
				{
					if (k % 2 == 0) obs->at<double>(counter,i) = (double)DCT[size][j][k - j];
					else obs->at<double>(counter,i) = (double)DCT[size][k - j][j];
				}
			}
		}

	}
	Mat* getObs()
	{
		return obs;
	}
	NColVector getObs(const int &i)
	{
		double* arr = new double[15];
		for (int j = 0; j < 15; j++)
		{
			arr[j] = obs->at<double>(j, i);
		}
		NColVector v(15, arr);
		delete[] arr;
		return v;
	}

};
NRowVector operator*(const NRowVector &v1, const Mat &op)
{
	
	if (op.rows != v1.getSize())
		return NRowVector(v1.getSize());
	int r = 1;
	int c = op.cols;
	NRowVector prod(c);
	double tmp;
	for (int j = 0; j < c; j++)
	{
		tmp = 0;
		for (int k = 0; k < op.rows; k++)
		{
			tmp += v1.getElement(k)*op.at<double>(k, j);
		}
		prod.setElement(j, tmp);
	}
	return prod;

}
NColVector operator*(const NColVector &v1, const Mat &op)
{
	if (op.cols != v1.getSize())
		return NColVector(v1.getSize());
	int c = 1;
	int r = op.cols;
	NColVector prod = NColVector(c);
	double tmp;
	for (int j = 0; j < r; j++)
	{
		tmp = 0;
		for (int k = 0; k < op.cols; k++)
		{
			tmp += op.at<double>(j, k)*v1.getElement(k);
		}
		prod.setElement(j, tmp);
	}
	return prod;

}
Mat operator*(const NColVector &v1, const NRowVector &v2)
{
	if (v1.getSize() != v2.getSize())return Mat(1, 1, CV_64F, Scalar(0));
	Mat prod(v1.getSize(), v2.getSize(), CV_64F, Scalar(0));
	for (int i = 0; i < v1.getSize(); i++)
		for (int j = 0; j < v2.getSize(); j++)prod.at<double>(i, j) = v1.getElement(i)*v2.getElement(j);
	return prod;
}
NColVector operator+(const NColVector &v1, const NColVector &v2)
{
	int s = v1.getSize() > v2.getSize() ? v1.getSize() : v2.getSize();
	double *arr = new double[s];
	for (int j = 0; j < s; j++)
	{
		arr[j] = 0;
		if (v1.getSize() > j)arr[j] += v1.getElement(j);
		if (v2.getSize() > j)arr[j] += v2.getElement(j);
	}
	NColVector sum = NColVector(s, arr);
	delete[] arr;
	return sum;
}
NColVector operator*(const double &d, const NColVector &v)
{
	NColVector v1(v);
	for (int j = 0; j < v.getSize(); j++)v1.setElement(j, d*v.getElement(j));
	return v1;
}
NColVector operator*(const NColVector &v, const double &d)
{
	NColVector v1(v);
	for (int j = 0; j < v.getSize(); j++)v1.setElement(j, d*v.getElement(j));
	return v1;
}
NRowVector operator+(const NRowVector &v1, const NRowVector &v2)
{
	int s = v1.getSize() > v2.getSize() ? v1.getSize() : v2.getSize();
	double *arr = new double[s];
	for (int j = 0; j < s; j++)
	{
		arr[j] = 0;
		if (v1.getSize() < j)arr[j] += v1.getElement(j);
		if (v2.getSize() < j)arr[j] += v2.getElement(j);
	}
	NRowVector sum = NRowVector(s, arr);
	delete[] arr;
	return sum;
}
NRowVector operator*(const double &d, const NRowVector &v)
{
	NRowVector v1(v);
	for (int j = 0; j < v.getSize(); j++)v1.setElement(j, d*v.getElement(j));
	return v1;
}
NRowVector operator*(const NRowVector &v, const double &d)
{
	NRowVector v1(v);
	for (int j = 0; j < v.getSize(); j++)v1.setElement(j, d*v.getElement(j));
	return v1;
}
double operator*(const NRowVector &v1, const NColVector &v2)
{
	double prod = 0;
	int t = v1.getSize() < v2.getSize() ? v1.getSize() : v2.getSize();
	for (int j = 0; j < t; j++)
	{
		prod = prod + v1.getElement(j)*v2.getElement(j);
	}
	return prod;
}
NColVector operator-(NColVector v1, NColVector v2)
{
	int s = v1.getSize() > v2.getSize() ? v1.getSize() : v2.getSize();
	double *arr = new double[s];
	for (int j = 0; j < s; j++)
	{
		arr[j] = 0;
		if (v1.getSize() > j)arr[j] += v1.getElement(j);
		if (v2.getSize() > j)arr[j] -= v2.getElement(j);
	}
	NColVector sum(s, arr);
	delete[] arr;
	return sum;
}
NRowVector operator-(NRowVector v1, NRowVector v2)
{
	int s = v1.getSize() > v2.getSize() ? v1.getSize() : v2.getSize();
	double *arr = new double[s];
	for (int j = 0; j < s; j++)
	{
		arr[j] = 0;
		if (v1.getSize() > j)arr[j] += v1.getElement(j);
		if (v2.getSize() > j)arr[j] -= v2.getElement(j);
	}
	NRowVector sum = NRowVector(s, arr);
	delete[] arr;
	return sum;
}
Mat operator+=(Mat &m1, const Mat &m2)
{
	return (m1 = m1 + m2);

}
NColVector* convertToColVec(Mat* m)
{
	if (m->cols != 1)return NULL;
	int size = m->rows;
	if (size <= 0)return  NULL;
	double* arr = new double[size];
	if (arr == NULL)return NULL;
	for (int j = 0; j < size; j++)
	{
		arr[j] = m->at<double>(j, 0);
	}
	NColVector* v = new NColVector(size, arr);
	delete[] arr;
	return v;
}
NRowVector* convertToRowVec(Mat* m)
{
	if (m->rows != 1)return NULL;
	int size = m->cols;
	if (size <= 0)return  NULL;
	double* arr = new double[size];
	if (arr == NULL)return NULL;
	for (int j = 0; j < size; j++)
	{
		arr[j] = m->at<double>(0, j);
	}
	NRowVector* v = new NRowVector(size, arr);
	delete[] arr;
	return v;
}
class Gaussian
{
	NColVector mean;
	Mat var;
public:
	Gaussian()
	{
		mean = 0;
		var = 1;
	}
	Gaussian(const NColVector &mu, const Mat &sigma)
	{
		mean = mu;
		var = sigma.clone();
	}
	NColVector getMean() const
	{
		return mean;
	}
	Mat getVar() const
	{
		return var;
	}
	void setMean(const NColVector &mu)
	{
		mean = mu;
	}
	void setVar(const Mat &sigma)
	{
		var = sigma;
	}
	double getProb(NColVector x)
	{
		double det = determinant(var);
		if (det < EPSILON)return 1;
		Mat inv_covar(15, 15, CV_64F);
		invert(var, inv_covar);
		NRowVector mean_t = (x-mean).transpose();
		NRowVector rhs = mean_t*inv_covar;
		double num = exp(-(rhs*(x-mean)) / 2);
		double den = sqrt(2 * pow(PI, var.rows)*abs(determinant(var)));
		double p = num / den;
		return  p;
	}
	Gaussian(const Gaussian &g)
	{
		mean = g.getMean();
		var = g.getVar().clone();
	}
	Gaussian operator=(const Gaussian &g)
	{
		mean = g.getMean();
		var = g.getVar().clone();
		return *this;
	}
	~Gaussian()
	{
	}
};

Obs*O = new Obs;

class HMM
{
private:
	Obs* SEQ;
	Mat* TRANS;
	Mat* INIT;
	NColVector **GAUSS_MEAN;
	Mat **GAUSS_VAR;
	double** GAUSS_PROB;
	int faces;
	int n_obs;
	int states;
	int mixtures;
	Gaussian** GAUSS;
public:
	void write(string outfilename)
	{
		FileStorage fs(outfilename.c_str(),FileStorage::WRITE);
		fs << "n_states" << states;
		fs << "n_mixtures" << mixtures;
		fs << "trans" << *TRANS;
		fs << "init" << *INIT;
		for(int i = 0;i<states;i++)
			for(int j = 0;j<mixtures;j++)
			{
				Mat tmp = GAUSS_MEAN[i][j].convertToMat();
				string tt = "mean_"+to_string(i)+"_"+to_string(j);
				fs << tt << tmp;
			}
		for(int i = 0;i<states;i++)
			for(int j = 0;j<mixtures;j++)
			{
				string tt = "var_"+to_string(i)+"_"+to_string(j);
				fs << tt << GAUSS_VAR[i][j];
			}
		for(int i = 0;i<states;i++)
			for(int j = 0;j<mixtures;j++)
			{
				string tt = "prob_"+to_string(i)+"_"+to_string(j);
                                fs << tt << GAUSS_PROB[i][j];
			}
		fs.release();
	}

	void read(string infilename)
	{
		FileStorage fs(infilename.c_str(),FileStorage::READ);
		fs["n_states"] >> states;
		fs["n_mixtures"] >> mixtures;
		fs["trans"] >> *TRANS;
		fs["init"] >> *INIT;
		for(int i = 0;i<states;i++)
                        for(int j = 0;j<mixtures;j++)
                        {
                                string tt = "var_"+to_string(i)+"_"+to_string(j);
                                fs[tt.c_str()] >> GAUSS_VAR[i][j];
                        }
                for(int i = 0;i<states;i++)
                        for(int j = 0;j<mixtures;j++)
                        {
                                string tt = "prob_"+to_string(i)+"_"+to_string(j);
                                fs[tt.c_str()] >> GAUSS_PROB[i][j];
                        }
		for(int i = 0;i<states;i++)
                        for(int j = 0;j<mixtures;j++)
                        {
                                Mat tmp;
				string tt = "mean_"+to_string(i)+"_"+to_string(j);
                                fs[tt.c_str()] >> tmp;
                                GAUSS_MEAN[i][j] = *convertToColVec(&tmp);
                        }
		fs.release();
		string filename = "0.jpg";
		Mat image;
		image = cv::imread(filename, CV_LOAD_IMAGE_GRAYSCALE);
		if (!image.data)
		{
			cout << "Yo mama so blind she don't read images" << endl;
			return;
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
						//cout << DCT[num][u][v];
					}
				}
			}
		}
		//store observation sequence
		O[0] = Obs(DCT,DCT_NUM1*DCT_NUM2);
		SEQ = O;
		faces = 1;
		n_obs = DCT_NUM1*DCT_NUM2;
		//observation[iter].disp();
	
	}
	HMM(Obs* O, int num_faces, int num_obs, int num_states, int num_mixtures)
	{
		SEQ = O;
		faces = num_faces;
		n_obs = num_obs;
		states = num_states;
		mixtures = num_mixtures;
		INIT_TRANS();
		INIT_INIT();
		INIT_GAUSSIAN();
	}
	HMM(int num_faces,int num_obs, int num_states, int num_mixtures)
	{
		SEQ = NULL;
		faces = num_faces;
		n_obs = num_obs;
		states = num_states;
		mixtures = num_mixtures;
		INIT_TRANS();
		INIT_INIT();
		INIT_GAUSSIAN();
	}
	HMM()
	{
		SEQ = NULL;
		faces = 1;
		n_obs = 0;
		states = 5;
		mixtures = 1;
		INIT_TRANS();
		INIT_INIT();
		INIT_GAUSSIAN();
	}
	void setObs(Obs *O)
	{
		SEQ = O;
		faces = 1;
	}
	void setObs(Obs *o, int num_faces)
	{
		SEQ = o;
		faces = num_faces;
	}
	int getStates()
	{
		return states;
	}
	int getMixtures()
	{
		return mixtures;
	}
	int getNObs()
	{
		return n_obs;
	}
	int getFaces()
	{
		return faces;
	}
	void INIT_TRANS()
	{
		TRANS = new Mat(states, states, CV_64F, Scalar(((double)(1.0)) / states));
		for(int i = 0;i<TRANS->rows;i++)
			for(int j = 0;j<TRANS->cols;j++)
				TRANS->at<double>(i,j) = (rand()%10000)/10000.0;
		double sum;
		for(int i = 0;i<TRANS->rows;i++)
		{
			sum = 0;
			for(int j = 0;j<TRANS->cols;j++)
				sum += TRANS->at<double>(i,j);
			for(int j = 0;j<TRANS->cols;j++)
				TRANS->at<double>(i,j)/=sum;
		}
	}
	void INIT_GAUSSIAN()
	{
		GAUSS = new Gaussian*[states];
		GAUSS_MEAN = new NColVector*[states];
		GAUSS_VAR = new Mat*[states];
		GAUSS_PROB = new double*[states];
		if (GAUSS == NULL) { cout << "Problem" << endl; return; }
		for (int j = 0; j < states; j++)
		{
			GAUSS[j] = new Gaussian[mixtures];
			GAUSS_MEAN[j] = new NColVector[mixtures];
			GAUSS_VAR[j] = new Mat[mixtures];
			GAUSS_PROB[j] = new double[mixtures];
			if (GAUSS[j] == NULL) { cout << "Problem" << endl; return; }
			for (int k = 0; k < mixtures; k++)
			{
				GAUSS_MEAN[j][k] = NColVector(15, 1);
				for (int iter = 0; iter < 15; iter++)GAUSS_MEAN[j][k].setElement(iter, (rand()%10000)/10000.0);
				GAUSS_PROB[j][k] = 1.0 / mixtures;
				GAUSS_VAR[j][k] = Mat(15, 15, CV_64F, Scalar(1));
				for (int r = 0; r < 15; r++)
					for (int c = 0; c < 15; c++)if (r != c)GAUSS_VAR[j][k].at<double>(r, c) = 0;
				GAUSS[j][k] = Gaussian(GAUSS_MEAN[j][k], GAUSS_VAR[j][k]);
			}

		}
	}
	void INIT_GAUSSIAN2()
	{
		GAUSS = new Gaussian*[states];
		GAUSS_MEAN = new NColVector*[states];
		GAUSS_VAR = new Mat*[states];
		GAUSS_PROB = new double*[states];
		if (GAUSS == NULL) { cout << "Problem" << endl; return; }
		int part = n_obs / mixtures;
		NColVector* sum=new NColVector[mixtures];
		Mat *obsClust = new Mat[mixtures];
		for (int i = 0; i < mixtures; i++)sum[i] = NColVector(15,0.0);
		for (int j = 0; j < mixtures; j++)
		{
			obsClust[j] = Mat(15, part*faces, CV_64F);
			for (int i = 0; i < faces; i++)
			{
				for (int k = j*mixtures, count = 0; count < part; k++, count++)
				{
					sum[j] = sum[j] + SEQ[i].getObs(k);
					(obsClust[j]).col(part*i + k) = SEQ[i].getObs()->col(k);
				}
				
			}
			sum[j] = sum[j] * (1 / (faces*part));
		}
		for (int j = 0; j < states; j++)
		{
			GAUSS[j] = new Gaussian[mixtures];
			GAUSS_MEAN[j] = new NColVector[mixtures];
			GAUSS_VAR[j] = new Mat[mixtures];
			GAUSS_PROB[j] = new double[mixtures];
			if (GAUSS[j] == NULL) { cout << "Problem" << endl; return; }
			for (int k = 0; k < mixtures; k++)
			{
				GAUSS_MEAN[j][k] = NColVector(sum[j]);
				GAUSS_PROB[j][k] = 1.0 / mixtures;
				Mat covar(15, 15, CV_64F);
				Mat mean(15, 1, CV_64F, Scalar(0));
				calcCovarMatrix(obsClust[k], covar, mean, CV_COVAR_COLS);
				GAUSS_VAR[j][k] = Mat(covar);
				GAUSS[j][k].setMean(GAUSS_MEAN[j][k]);
				GAUSS[j][k].setVar(GAUSS_VAR[j][k]);
			}

		}
		delete[] sum;
		delete[] obsClust;
	}
	void INIT_INIT()
	{
		INIT = new Mat(1, states, CV_64F, Scalar(((double)(1.0)) / states));
		for(int i = 0;i<INIT->rows;i++)
			for(int j = 0;j<INIT->cols;j++)
				INIT->at<double>(i,j) = (rand()%10000)/10000.0;
		double sum;
		for(int i = 0;i<INIT->rows;i++)
		{
			sum = 0;
			for(int j = 0;j<INIT->cols;j++)
				sum += INIT->at<double>(i,j);
			for(int j = 0;j<INIT->cols;j++)
				INIT->at<double>(i,j)/=sum;
		}

	}
	void correct(Mat* m)
	{
		for (int j = 0; j < m->rows; j++)
		{
			for (int k = 0; k < m->cols; k++)
			{
				if (m->at<double>(j, k) < EPSILON)m->at<double>(j, k) = EPSILON;
			}
		}
	}
	static double getMaxLikelihood(Obs &o, Mat &trans, Mat &init, Gaussian **g, double **gauss_prob,int nstates,int nmix)
	{
		
		Mat DELTA(nstates, o.getObs()->cols, CV_64F, Scalar(0));
		Mat SIGMA(nstates, o.getObs()->cols, CV_32S, Scalar(0));
		double prob = 0;
		double fnToMax;
		NColVector *tmp = NULL;
		for (int j = 0; j < nstates; j++)
		{
			tmp = new NColVector(o.getObs(0));
			if (tmp == NULL)
			{
				cout << "Error in initializing col vector ..." << endl;
				return 0;
			}
			double emis_prob = 0;
			for (int m = 0; m < nmix; m++)emis_prob += gauss_prob[j][m] * g[j][m].getProb(*tmp);
			DELTA.at<double>(j, 0) = init.at<double>(0, j)*emis_prob;
		}
		for (int t = 1; t < o.getObs()->cols; t++)
		{
			for (int j = 0; j < nstates; j++)
			{
				fnToMax = DELTA.at<double>(0, t - 1)*trans.at<double>(0, j);
				SIGMA.at<int>(j, t) = 0;
				tmp = new NColVector(o.getObs(t));
				if (tmp == NULL)
				{
					cout << "Error in initializing col vector ..." << endl;
					return 0;
				}
				double emis_prob = 0;
				for (int m = 0; m < nmix; m++)emis_prob += gauss_prob[j][m] * g[j][m].getProb(*tmp);
				delete tmp;
				tmp = NULL;
				DELTA.at<double>(j, t) = fnToMax*emis_prob;
				for (int k = 1; k < nstates; k++)
				{
					double temp = DELTA.at<double>(0, t - 1)*trans.at<double>(0, j);
					if (temp > fnToMax)
					{
						SIGMA.at<int>(j, t) = k;
						tmp = new NColVector(o.getObs(t));
						if (tmp == NULL)
						{
							cout << "Error in initializing col vector ..." << endl;
							return 0;
						}
						double emis_prob = 0;
						for (int m = 0; m < nmix; m++)emis_prob += gauss_prob[j][m] * g[j][m].getProb(*tmp);
						delete tmp;
						tmp = NULL;
						DELTA.at<double>(j, t) = temp*emis_prob;
						fnToMax = temp;
					}
				}
			}
		}
		int n_obs = o.getObs()->cols;
		int *path_states = new int[n_obs];
		path_states[n_obs - 1] = 0;
		for (int j = 1; j < nstates; j++)
			if (DELTA.at<double>(j, n_obs - 1) - DELTA.at<double>(path_states[n_obs - 1], n_obs - 1) > EPSILON)
				path_states[n_obs - 1] = j;
		for (int t = n_obs - 2; t >= 0; t--)path_states[t] = SIGMA.at<int>(path_states[t + 1], t + 1);
		prob += (DELTA.at<double>(path_states[n_obs - 1], n_obs - 1));
		return prob;
	}
	double ViterbiLikelihood()
	{
		Mat DELTA(states, n_obs, CV_64F,Scalar(0));
		Mat SIGMA(states, n_obs, CV_32S, Scalar(0));
		double prob=0;
		double fnToMax;
		for (int countf = 0; countf < faces; countf++)
		{
			NColVector *tmp = NULL;
			for (int j = 0; j < states; j++)
			{
			tmp = new NColVector(SEQ[countf].getObs(0));
			if (tmp == NULL)
			{
				cout << "Error in initializing col vector ..." << endl;
				return 0;
			}
			double emis_prob = 0;
			for (int m = 0; m < mixtures; m++)emis_prob += GAUSS_PROB[j][m] * GAUSS[j][m].getProb(*tmp);
			DELTA.at<double>(j, 0) = INIT->at<double>(0, j)*emis_prob;
			}
			for (int t = 1; t < n_obs; t++)
			{
				for (int j = 0; j < states; j++)
				{
					fnToMax = DELTA.at<double>(0, t - 1)*TRANS->at<double>(0, j);
					SIGMA.at<int>(j, t) = 0;
					tmp = new NColVector(SEQ[countf].getObs(t));
					if (tmp == NULL)
					{
						cout << "Error in initializing col vector ..." << endl;
						return 0;
					}
					double emis_prob = 0;
					for (int m = 0; m < mixtures; m++)emis_prob += GAUSS_PROB[j][m] * GAUSS[j][m].getProb(*tmp);
					delete tmp;
					tmp = NULL;
					DELTA.at<double>(j, t) = fnToMax*emis_prob;
					for (int k = 1; k < states; k++)
					{
						double temp = DELTA.at<double>(0, t - 1)*TRANS->at<double>(0, j);
						if (temp > fnToMax)
						{
							SIGMA.at<int>(j, t) = k;
							tmp = new NColVector(SEQ[countf].getObs(t));
							if (tmp == NULL)
							{
								cout << "Error in initializing col vector ..." << endl;
								return 0;
							}
							double emis_prob = 0;
							for (int m = 0; m < mixtures; m++)emis_prob += GAUSS_PROB[j][m] * GAUSS[j][m].getProb(*tmp);
							delete tmp;
							tmp = NULL;
							DELTA.at<double>(j, t) = temp*emis_prob;
							fnToMax = temp;
						}					
					}
				}
			}

			int *path_states = new int[n_obs];
			path_states[n_obs - 1] = 0;
			for (int j = 1; j < states; j++)
				if(DELTA.at<double>(j, n_obs - 1) - DELTA.at<double>(path_states[n_obs - 1], n_obs - 1) > EPSILON)
					path_states[n_obs - 1] = j;
			for (int t = n_obs - 2; t >= 0; t--)path_states[t] = SIGMA.at<int>(path_states[t + 1], t + 1);
			prob += (DELTA.at<double>(path_states[n_obs-1], n_obs-1));
			
		}
		prob = prob / faces;
		return prob;
	}
	void train(double probLimit, int maxCount)
	{
		print();
		cout << "Beginning training..." << endl;
		Mat** GAUSS_VAR_;
		NColVector** GAUSS_MEAN_;
		double** GAUSS_PROB_;
		GAUSS_VAR_ = new Mat*[states];
		GAUSS_MEAN_ = new NColVector*[states];
		GAUSS_PROB_ = new double*[states];

		cout << "Declaring Gaussian matrices..." << endl;
		for (int i = 0; i < states; i++)
		{
			GAUSS_PROB_[i] = new double[mixtures];
			GAUSS_MEAN_[i] = new NColVector[mixtures];
			GAUSS_VAR_[i] = new Mat[mixtures];
			for (int j = 0; j < mixtures; j++)
			{
				GAUSS_PROB_[i][j] = GAUSS_PROB[i][j];
				GAUSS_MEAN_[i][j] = GAUSS_MEAN[i][j];
				GAUSS_VAR_[i][j] = GAUSS_VAR[i][j];
			}
		}
		Mat TRANS_ = TRANS->clone();
		Mat INIT_ = INIT->clone();
		int T = SEQ[0].getObs()->cols;
		int N = TRANS->rows;		//=TRANS.rows=TRANS.cols= No. of states
		int M = mixtures;	// No .of mixtures in the Gaussian
		int count = 0;
		int countf = 0;	//counter for iteration over the faces
		Mat *ALPHA = NULL;;
		Mat *BETA = NULL;
		Mat *GAMMA=NULL;
		double coeff[15];
		double emis_prob;
		NColVector* tmp;
		double *Prob, *newProb;
		Prob = new double[faces];
		newProb = new double[faces];
		cout << "Declaring some other variables..." << endl;
		for (int j = 0; j < faces; j++)
		{
			Prob[j] = 0;
			newProb[j] = 0;
		}
		double* diffProb = new double[faces];
		NColVector probDiff;
		//the variables prefixed with num_ and den_ are for storing the total of the updates for all faces so that averaging out takes place in the last step
		Mat num_TRANS = Mat::zeros(TRANS->rows, TRANS->cols, CV_64F);
		Mat num_INIT = Mat::zeros(INIT->rows, INIT->cols, CV_64F);
		NColVector **num_GAUSS_MEAN;
		Mat** num_GAUSS_VAR;
		Mat den_TRANS = Mat::zeros(TRANS->rows, TRANS->cols, CV_64F);
		Mat den_INIT = Mat::zeros(INIT->rows, INIT->cols, CV_64F);
		double **num_GAUSS_PROB, **den_GAUSS_PROB, **den_GAUSS_MEAN, **den_GAUSS_VAR;
		num_GAUSS_PROB = new double*[states];
		num_GAUSS_MEAN = new NColVector*[states];
		num_GAUSS_VAR = new Mat*[states];
		den_GAUSS_PROB = new double*[states];
		den_GAUSS_MEAN = new double*[states];
		den_GAUSS_VAR = new double*[states];
		cout << "Setting up HMM properties..." << endl;
		for (int i = 0; i < states; i++)
		{
			num_GAUSS_PROB[i] = new double[mixtures];
			num_GAUSS_MEAN[i] = new NColVector[mixtures];
			num_GAUSS_VAR[i] = new Mat[mixtures];
			den_GAUSS_PROB[i] = new double[mixtures];
			den_GAUSS_MEAN[i] = new double[mixtures];
			den_GAUSS_VAR[i] = new double[mixtures];
			for (int j = 0; j < mixtures; j++)
			{
				num_GAUSS_PROB[i][j] = 0;
				num_GAUSS_MEAN[i][j] = NColVector(0 * GAUSS_MEAN[i][j]);
				num_GAUSS_VAR[i][j] = Mat::zeros(15, 15, CV_64F);
				den_GAUSS_PROB[i][j] = 0;
				den_GAUSS_MEAN[i][j] = 0;
				den_GAUSS_VAR[i][j] = 0;
			}
			cout << "Initializing HMM Properties for state " << i << " ..." << endl;
		}
		do{
			for (countf = 0; countf < faces; countf++)
			{
				if (countf != 0)
				{
					delete ALPHA;
					delete BETA;
					delete GAMMA;
					ALPHA = NULL;
					BETA = NULL;
					GAMMA = NULL;
				}
				cout << "Processing face " << countf << " ..." << endl;
				ALPHA = new Mat(N, T, CV_64F, Scalar(0));
				BETA = new Mat(N, T, CV_64F, Scalar(0));
				GAMMA = new Mat(N, T, CV_64F, Scalar(0));
				if (ALPHA == NULL || BETA == NULL || GAMMA == NULL)
				{
					cout << "Null pointer returned during initialization of HMM variables." << endl;
					return;
				}
				for (int i = 0; i < N; i++)
				{
					tmp =new NColVector(SEQ[countf].getObs(0));
					if (tmp == NULL)
					{
						cout << "Null pointer returned during initialization of NCOlVector." << endl;
						return;
					}
					emis_prob = 0.0;
					for (int countk = 0; countk < M; countk++)
					{
						/*Gaussian abcd = *GAUSS[i][countk];
						double efgh =  abcd.getProb(*tmp);*/
						emis_prob += GAUSS_PROB_[i][countk] * (GAUSS[i][countk].getProb(*tmp));
					}
					delete tmp;
					tmp = NULL;
					//cout << INIT_.at<double>(0, i) << endl;
					ALPHA->at<double>(i, 0) = INIT_.at<double>(0, i)*emis_prob;
					//cout << "Processing state " << i << " ..." << endl;
				}
				for (int j = 1; j < T; j++)
				{
					for (int i = 0; i < N; i++)
					{
						double temp = 0;
						for (int k = 0; k < N; k++)
						{
							temp += ALPHA->at<double>(k, j - 1)*TRANS_.at<double>(k, i);
						}
						if (temp < EPSILON)temp = EPSILON;
						tmp = new NColVector(SEQ[countf].getObs(j));
						if (tmp == NULL)
						{
							cout << "Null pointer returned during initialization of NCOlVector." << endl;
							return;
						}
						emis_prob = 0.0;
						for (int countk = 0; countk < M; countk++)emis_prob += GAUSS_PROB_[i][countk] * GAUSS[i][countk].getProb(*tmp);
						delete tmp;
						tmp = NULL;
						if (emis_prob < EPSILON)emis_prob = EPSILON;
						ALPHA->at<double>(i, j) = emis_prob*temp;
					}
				}
				Prob[countf] = 0.0;
				for (int j = 0; j < N; j++)
				{
					Prob[countf] += ALPHA->at<double>(j, T - 1);
					if (Prob[countf] < EPSILON)Prob[countf] = EPSILON;
				}
				for (int i = 0; i < N; i++)BETA->at<double>(i, T - 1) = 1;
				for (int j = T - 2; j >= 0; j--)
				{
					for (int i = 0; i < N; i++)
					{
						double temp = 0;
						for (int k = 0; k < N; k++)
						{
							//for (int countj = 0; countj < 15; countj++)coeff[countj] = SEQ[countf]->getObs()->at<double>(countj, j + 1);
							tmp = new NColVector(SEQ[countf].getObs(j + 1));
							if (tmp == NULL)
							{
								cout << "Null pointer returned during initialization of NCOlVector." << endl;
								return;
							}
							emis_prob = 0.0;
							for (int countk = 0; countk < M; countk++)emis_prob += GAUSS_PROB_[i][countk] * GAUSS[i][countk].getProb(*tmp);
							delete tmp;
							tmp = NULL;
							if (emis_prob < EPSILON)emis_prob = EPSILON;
							temp += BETA->at<double>(k, j + 1)*TRANS_.at<double>(i, k)*emis_prob;
						}
						if (temp < EPSILON)temp = EPSILON;
						BETA->at<double>(i, j) = temp;
					}
				}
				cout << "Initializing GAMMA:" << endl;
				for (int i = 0; i < N; i++)
				{
					for (int j = 0; j < T; j++)
					{
						double tmp = 0;
						for (int k = 0; k < N; k++)tmp += ALPHA->at<double>(k, j)*BETA->at<double>(k, j);
						if (tmp < EPSILON)tmp = EPSILON;
						GAMMA->at<double>(i, j) = ALPHA->at<double>(i, j)*BETA->at<double>(i, j) / tmp;
						if (j == 0)
						{
							num_INIT.at<double>(j,i) += ALPHA->at<double>(i,j)*BETA->at<double>(i,j);
							den_INIT.at<double>(j, i) += tmp;
							if (den_INIT.at<double>(j, i) < EPSILON)den_INIT.at<double>(j, i) = EPSILON;
						}
					}
				}
				Vector <Mat> CHI;
				for (int i = 0; i < T - 1; i++)
				{
					Mat temp(N, N, CV_64F);
					for (int j = 0; j < N; j++)
					{
						for (int k = 0; k < N; k++)
						{
							tmp = new NColVector(SEQ[countf].getObs(i + 1));
							if (tmp == NULL)
							{
								cout << "Null pointer returned during initialization of NCOlVector." << endl;
								return;
							}
							emis_prob = 0.0;
							for (int countk = 0; countk < M; countk++)
								emis_prob += GAUSS_PROB_[k][countk] * GAUSS[k][countk].getProb(*tmp);
							delete tmp;
							tmp = NULL;
							if (emis_prob < EPSILON)emis_prob = EPSILON;
							temp.at<double>(j, k) = ALPHA->at<double>(j, i)*TRANS_.at<double>(j, k)*emis_prob*BETA->at<double>(k, i + 1) / (Prob[countf]);
						}
					}
					CHI.push_back(temp);
				}
				for (int i = 0; i < N; i++)
					for (int j = 0; j < N; j++)
					{
					double tmp1, tmp2;
					tmp1 = 0.0;
					tmp2 = 0.0;
					for (int k = 0; k < T - 1; k++)
					{
						tmp1 += CHI[k].at<double>(i, j);
						tmp2 += GAMMA->at<double>(i, k);
					}
					num_TRANS.at<double>(i, j) += tmp1;
					den_TRANS.at<double>(i, j) += tmp2;
					if (den_TRANS.at<double>(i, j) < EPSILON)den_TRANS.at<double>(i, j) = EPSILON;
					}
				for (int i = 0; i < N; i++)
				{
					double temp2 = 0.0;
					for (int m = 0; m < M; m++)
					{
						double temp1;
						for (int t = 1; t < T; t++)
						{
							temp1 = 0.0;
							for (int j = 0; j < N; j++)temp1 += TRANS_.at<double>(j, i)*ALPHA->at<double>(j, t - 1);
							temp1 *= GAUSS[i][m].getProb((SEQ[countf].getObs(t)))*BETA->at<double>(i, t);
							num_GAUSS_PROB[i][m] += GAUSS_PROB[i][m] * (temp1);
						}
						temp2 += GAUSS_PROB[i][m] * temp1;
					}
					for (int m = 0; m < M; m++)
					{
						den_GAUSS_PROB[i][m] += temp2;
						if (den_GAUSS_PROB[i][m] < EPSILON)den_GAUSS_PROB[i][m] = EPSILON;
					}
				}
				for (int i = 0; i < N; i++)
				{
					for (int m = 0; m < M; m++)
					{
						double temp;
						for (int t = 1; t < T; t++)
						{
							temp = 0.0;
							for (int j = 0; j < N; j++)
								temp += TRANS_.at<double>(i, j)*ALPHA->at<double>(j, t - 1);
							temp *= GAUSS[i][m].getProb((SEQ[countf].getObs(t)))*BETA->at<double>(i, t);
							num_GAUSS_MEAN[i][m] = num_GAUSS_MEAN[i][m] + ((SEQ[countf].getObs(t)))*temp;
							NColVector v1 = (SEQ[countf].getObs(t));
							v1 = v1 - GAUSS_MEAN_[i][m];
							num_GAUSS_VAR[i][m] = num_GAUSS_VAR[i][m] + ((v1*(v1.transpose())))*temp;
							if (temp < EPSILON)temp = EPSILON;
							den_GAUSS_MEAN[i][m] += temp;
							den_GAUSS_VAR[i][m] += temp;
						}
					}
				}
				for (int i = 0; i < N; i++)
				{
					tmp = new NColVector(SEQ[countf].getObs(0));
					if (tmp == NULL)
					{
						cout << "Null pointer returned during initialization of NCOlVector." << endl;
						return;
					}
					emis_prob = 0.0;
					for (int countk = 0; countk < M; countk++)emis_prob += GAUSS_PROB_[i][countk] * GAUSS[i][countk].getProb(*tmp);
					delete tmp;
					tmp = NULL;
					ALPHA->at<double>(i, 0) = INIT_.at<double>(0, i)*emis_prob;
				}
				for (int j = 1; j < T; j++)
				{
					for (int i = 0; i < N; i++)
					{
						double temp = 0;
						for (int k = 0; k < N; k++)
						{
							temp += ALPHA->at<double>(k, j - 1)*TRANS_.at<double>(k, i);
						}
						//for (int countj = 0; countj < 15; countj++)coeff[countj] = SEQ[countf]->getObs()->at<double>(countj, j);
						tmp = new NColVector(SEQ[countf].getObs(j));
						if (tmp == NULL)
						{
							cout << "Null pointer returned during initialization of NCOlVector." << endl;
							return;
						}
						emis_prob = 0.0;
						for (int countk = 0; countk < M; countk++)emis_prob += GAUSS_PROB_[i][countk] * (GAUSS[i][countk].getProb(*tmp));
						delete tmp;
						tmp = NULL;
						ALPHA->at<double>(i, j) = emis_prob;
					}
					//cout << "Processing observation no. " << j << " ..." << endl;
				}
				newProb[countf] = 0.0;
				for (int j = 0; j < N; j++)
				{
					newProb[countf] += ALPHA->at<double>(j, T - 1);
				}
				diffProb[countf] = newProb[countf] - Prob[countf];
			}
			probDiff = NColVector(faces, diffProb);
			for (int i = 0; i < INIT_.rows; i++)
				for (int j = 0; j < INIT_.cols; j++)INIT_.at<double>(i, j) = num_INIT.at<double>(i, j) / den_INIT.at<double>(i, j);
			for (int i = 0; i < TRANS_.rows; i++)
				for (int j = 0; j < TRANS_.cols; j++)TRANS_.at<double>(i, j) = num_TRANS.at<double>(i, j) / den_TRANS.at<double>(i, j);
			for (int i = 0; i < N; i++)
				for (int j = 0; j < M; j++)
				{
				GAUSS_PROB_[i][j] = num_GAUSS_PROB[i][j] / den_GAUSS_PROB[i][j];
				GAUSS_MEAN_[i][j] = (1.0 / den_GAUSS_MEAN[i][j])*num_GAUSS_MEAN[i][j];
				GAUSS_VAR_[i][j] = (1.0 / den_GAUSS_VAR[i][j])*num_GAUSS_VAR[i][j];
				}
			for (int countf = 0; countf < faces;countf++)
				Prob[countf] = newProb[countf];
			count++;
		*TRANS = TRANS_.clone();
		*INIT = INIT_.clone();
		for (int i = 0; i < states; i++)
			for (int j = 0; j < mixtures; j++)
			{
			GAUSS_PROB[i][j] = GAUSS_PROB_[i][j];
			GAUSS_MEAN[i][j] = GAUSS_MEAN_[i][j];
			GAUSS_VAR[i][j] = GAUSS_VAR_[i][j];
			GAUSS[i][j].setMean(GAUSS_MEAN[i][j]);
			GAUSS[i][j].setVar(GAUSS_VAR[i][j]);
			}

		print();
		} while (probDiff.getNorm()/(faces) > probLimit && count < maxCount);
		*TRANS = TRANS_.clone();
		*INIT = INIT_.clone();
		for (int i = 0; i < states; i++)
			for (int j = 0; j < mixtures; j++)
			{
			GAUSS_PROB[i][j] = GAUSS_PROB_[i][j];
			GAUSS_MEAN[i][j] = GAUSS_MEAN_[i][j];
			GAUSS_VAR[i][j] = GAUSS_VAR_[i][j];
			GAUSS[i][j].setMean(GAUSS_MEAN[i][j]);
			GAUSS[i][j].setVar(GAUSS_VAR[i][j]);
			}
	}
	void print()
	{
		cout << "Showing init..." << endl;
		for (int i = 0; i<INIT->rows; i++)
		{
			for (int j = 0; j<INIT->cols; j++)
				cout << INIT->at<double>(i, j) << " ";
			cout << endl;
		}
		cout << endl;
		cout << "Showing trans..." << endl;
		for (int i = 0; i<TRANS->rows; i++)
		{
			for (int j = 0; j<TRANS->cols; j++)
				cout << TRANS->at<double>(i, j) << " ";
			cout << endl;
		}
		cout << endl;
		cout << "Showing mean..." << endl;
		for(int i = 0;i<states;i++)
		{
			for(int j = 0;j<mixtures;j++)
			{
				for(int k = 0;k<15;k++)
					cout << GAUSS_MEAN[i][j].getElement(k) << " ";
				cout << "<>";
			}
			cout << endl;
		}
		for(int i = 0;i<states;i++)
			for(int j = 0;j<mixtures;j++)
				cout << GAUSS_VAR[i][j] <<endl;
	}
	void init(Mat &A)
	{
		if (!A.data)
			return;
		int r = A.rows;
		int c = A.cols;
		for (int j = 0; j < r; j++)
			for (int k = 0; k < c; k++)A.at<double>(j, k) = ((double)1) / c;
	}

};
class Person
{
	static unsigned long int num;
	unsigned long int id;
	unsigned int num_of_faces;
public:
	Person();
	Person(unsigned int nof);
	unsigned long int getId(){ return id; }
	bool train(int &max_iter);
	double detect();
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
bool Person::train(int &max_iter)
{
	
	
	Obs *observation = new Obs[num_of_faces];
	if (observation == NULL)
	{
		cout << "Null pointer returned during initialization of observation." << endl;
		return false;
	} 
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
						//cout << DCT[num][u][v];
					}
				}
			}
		}
		//store observation sequence
		observation[iter] = Obs(DCT,DCT_NUM1*DCT_NUM2);
		//observation[iter].disp();
	}

	//train hmm
	HMM model(observation, num_of_faces, DCT_NUM1*DCT_NUM2, 5, 1);
	model.train(1E-40, max_iter);
	//store hmm, file has header denoting training maximized likelihood
	string outfilename = "";
	outfilename += to_string(id) + ".hmm";
	model.write(outfilename);
}

double Person::detect()
{
	string infilename = "";
	infilename += to_string(id) + ".hmm";
	HMM model;
	model.read(infilename);
	return model.ViterbiLikelihood();	
}

int main(int argc, char **argv)
{
	int nof;
	cout << DBL_MAX << endl;
	cout << "Number of faces? " << endl;
	cin >> nof;
	int max_iter;
	cout << "Maximum iterations? " << endl;
	cin >> max_iter;
	Person foo(nof);
	foo.train(max_iter);
	cout << foo.detect()<<endl;

	//create a video stream

	//for each frame, run over all hmm we have in the parent folder

	//calculate likelihoods

	//report likelihoods in a table for each person
	return 0;
}
