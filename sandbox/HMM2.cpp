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
#define SW_SIZE 8       //size of sliding window
#define OVERLAP 0.5     //extent of overlap
#define APP_SIZE (int)((1-OVERLAP)*SW_SIZE)     //extent by which to slide the window
#define INV2SW_SIZE (1.0)/(SW_SIZE)     //reciprocal of window size
#define IMG_ROW 512     //definition of row size of image
#define IMG_COL 512//definition of column size of image
#define DCT_NUM1 (int)((IMG_ROW-SW_SIZE+APP_SIZE)/APP_SIZE)     //total number of DCT matrices the image will produce
#define DCT_NUM2 (int)((IMG_COL-SW_SIZE+APP_SIZE)/APP_SIZE)     //total number of DCT matrices the image will produce

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
			cout << "Null pointer returned during initialization of NRowVector." << endl;
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
	NRowVector* transpose()
	{
		NRowVector *v2 = new NRowVector(getSize());
		if (v2 == NULL)
		{
			cout << "Null pointer returned during initialization of NRowVector." << endl;
		}
		for (int j = 0; j < getSize(); j++)v2->setElement(j, getElement(j));
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
	Mat* convertToMat()
	{
		int r = size;
		int c = 1;
		Mat* m = new Mat(r, c, CV_64F);
		if (v == NULL)
		{
			cout << "Null pointer returned during initialization of Mat." << endl;
		}
		for (int j = 0; j < size; j++)
			m->at<double>(j, 0) = getElement(j);
		return m;
	}
};
class Obs
{
	Mat* obs;
public:
	Obs()
	{
		obs = new Mat(1, 15, CV_64F,Scalar(0));
	}
	Obs(double DCT[][SW_SIZE][SW_SIZE], int size)
	{
		obs=new Mat(15, size, CV_64F);
		for (int i = 0; i < size; i++)
		{			
			for (int k = 0,counter=0; k < 5; k++)
			{
				for (int j = 0; j <= k; j++,counter++)
				{
					if (k % 2 == 0) obs->at<double>(counter,i) = DCT[size][j][k - j];
					else obs->at<double>(counter,i) = DCT[size][k-j][j];
				}
			}
		}
		
	}
	Obs(const Obs &o)
	{
		*obs = (o.obs)->clone();
	}
	Obs& operator=(const Obs &o)
	{
		*obs = (o.obs)->clone();
		return *this;
	}
	~Obs()
	{
		delete obs;
	}
	void setObs(double DCT[][SW_SIZE][SW_SIZE], int size)
	{
		obs = new Mat(size, 15, CV_64F);
		for (int i = 0; i < size; i++)
		{
			for (int k = 0, counter = 0; k < 5; k++)
			{
				for (int j = 0; j <= k; j++, counter++)
				{
					if (k % 2 == 0) obs->at<double>(i, counter) = DCT[size][j][k - j];
					else obs->at<double>(i, counter) = DCT[size][k - j][j];
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
			tmp += op.at<double>(j,k)*v1.getElement(k);
		}
		prod.setElement(j, tmp);
	}
	return prod;

}
Mat operator*(const NColVector &v1, const NRowVector &v2)
{
	if (v1.getSize() != v2.getSize())return Mat(1, 1, CV_64F,Scalar(0));
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
		if (v1.getSize() < j)arr[j] += v1.getElement(j);
		if (v2.getSize() < j)arr[j] += v2.getElement(j);
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
		if (v1.getSize() < j)arr[j] += v1.getElement(j);
		if (v2.getSize() < j)arr[j] -= v2.getElement(j);
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
		if (v1.getSize() < j)arr[j] += v1.getElement(j);
		if (v2.getSize() < j)arr[j] -= v2.getElement(j);
	}
	NRowVector sum = NRowVector(s, arr);
	delete[] arr;
	return sum;
}
Mat operator+=(Mat &m1,const Mat &m2)
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
		arr[j] = m->at<double>(j, 0) ;
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
		Mat inv_covar(var.rows, var.cols, CV_64F, 0);
		invert(var, inv_covar);
		NRowVector mean_t = *(mean.transpose());
		NRowVector rhs = mean_t*inv_covar;
		double num = exp(-(rhs*mean) / 2);
		double den = sqrt(2 *pow(PI,var.rows)*abs(determinant(var)));
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

class HMM
{
private:
	Obs* SEQ;
	Mat* TRANS;
	Mat* INIT;
	NColVector ***GAUSS_MEAN;
	Mat ***GAUSS_VAR;
	double** GAUSS_PROB;
	int faces;
	int n_obs;
	int states;
	int mixtures;
	Gaussian*** GAUSS;
public:

	HMM(Obs* O, int num_faces, int num_obs, int num_states, int num_mixtures)
	{
		SEQ = new Obs[num_faces];
		for (int i = 0; i < num_faces; i++)SEQ[i] = O[i];
		//SEQ = O;
		faces = num_faces;
		n_obs = num_obs;
		states = num_states;
		mixtures = num_mixtures;
		INIT_TRANS();
		INIT_INIT();
		INIT_GAUSSIAN();
	}
	void print()
	{
		cout << "Showing init..." << endl;
		for(int i = 0;i<INIT->rows;i++)
		{
			for(int j = 0;j<INIT->cols;j++)
				cout << INIT->at<double>(i,j) << " ";
			cout << endl;
		}
		cout << endl;
		cout << "Showing trans..." << endl;
		for(int i = 0;i<TRANS->rows;i++)
		{
			for(int j = 0;j<TRANS->cols;j++)
				cout << TRANS->at<double>(i,j) << " ";
			cout << endl;
		}
	}
	void INIT_TRANS()
	{
		TRANS = new Mat(states, states, CV_64F, Scalar(((double)(1.0)) / states));
	}
	void INIT_GAUSSIAN()
	{
		GAUSS = new Gaussian**[states];
		GAUSS_MEAN = new NColVector**[states];
		GAUSS_VAR = new Mat**[states];
		GAUSS_PROB = new double*[states];
		if (GAUSS == NULL) { cout << "Problem" << endl; return; }
		for (int j = 0; j < states; j++)
		{
			GAUSS[j] = new Gaussian*[mixtures];
			GAUSS_MEAN[j] = new NColVector*[mixtures];
			GAUSS_VAR[j] = new Mat*[mixtures];
			GAUSS_PROB[j] = new double[mixtures];
			if (GAUSS[j] == NULL) { cout << "Problem" << endl; return; }
			for (int k = 0; k < mixtures; k++)
			{				
				GAUSS_MEAN[j][k] =new NColVector(15, 1);
				GAUSS_PROB[j][k] = 1.0 / mixtures;
				GAUSS_VAR[j][k] = new Mat(15, 15, CV_64F, 1); 
				for (int r = 0; r < 15; r++)
					for (int c = 0; c < 15; c++)if (r != c)GAUSS_VAR[j][k]->at<double>(r, c) = 0;
				GAUSS[j][k] = new Gaussian(*GAUSS_MEAN[j][k],*GAUSS_VAR[j][k]);
				GAUSS[j][k]->setMean(*GAUSS_MEAN[j][k]);
				GAUSS[j][k]->setVar(*GAUSS_VAR[j][k]);
			
			}
				
		}
	}
	void INIT_INIT()
	{
		INIT = new Mat(1, states, CV_64F, ((double)(1.0)) / states);
	}
	void train(double probLimit, int maxCount)
	{
		cout << "Beginning training..." << endl;
		Mat*** GAUSS_VAR_;
		NColVector*** GAUSS_MEAN_;
		double** GAUSS_PROB_;
		GAUSS_VAR_ = new Mat**[states];
		GAUSS_MEAN_ = new NColVector**[states];
		GAUSS_PROB_ = new double*[states];

		cout << "Declaring Gaussian matrices..." << endl;
		for (int i = 0; i < states; i++)
		{
			GAUSS_PROB_[i] = new double[mixtures];
			GAUSS_MEAN_[i] = new NColVector*[mixtures];
			GAUSS_VAR_[i] = new Mat*[mixtures];
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
		Mat *ALPHA=new Mat(N, T, CV_64F);
		Mat *BETA=new Mat(N, T, CV_64F);
		Mat *GAMMA=new Mat(N, T, CV_64F);
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
				num_GAUSS_MEAN[i][j] = NColVector(0 * *GAUSS_MEAN[i][j]);
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
				/*delete ALPHA;
				delete BETA;
				delete GAMMA;*/

				cout << "Processing face " << countf << " ..." << endl;
				ALPHA =new Mat(N, T, CV_64F,Scalar(0));
				BETA =new Mat(N, T, CV_64F,Scalar(0));
				GAMMA =new Mat(N, T, CV_64F,Scalar(0));
				if (ALPHA == NULL || BETA == NULL || GAMMA == NULL)
				{
					cout << "Null pointer returned during initialization of HMM variables." << endl;
					return;
				}
				for (int i = 0; i < N; i++)
				{
					//for (int countj = 0; countj < 15; countj++)
					//	coeff[countj] = SEQ[countf]->getObs()->at<double>(countj, 0);
					tmp = new NColVector(SEQ[countf].getObs(0));
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
						emis_prob += GAUSS_PROB_[i][countk] * (GAUSS[i][countk]->getProb(*tmp));
					}
					//cout << INIT_.at<double>(0, i) << endl;
					ALPHA->at<double>(i, 0) = INIT_.at<double>(0, i)*emis_prob;
					//cout << "Processing state " << i << " ..." << endl;
					delete tmp;
					tmp = NULL;
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
						for (int countk = 0; countk < M; countk++)emis_prob += GAUSS_PROB_[i][countk] * GAUSS[i][countk]->getProb(*tmp);
						ALPHA->at<double>(i, j) = emis_prob*temp;
						delete tmp;
						tmp = NULL;
					}
					//cout << "Processing observation no. " << j << " ..." << endl;
				}
				Prob[countf] = 0.0;
				for (int j = 0; j < N; j++)
				{
					Prob[countf] += ALPHA->at<double>(j, T - 1);
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
							tmp = new NColVector(SEQ[countf].getObs(j+1));
							if (tmp == NULL)
							{
								cout << "Null pointer returned during initialization of NCOlVector." << endl;
								return;
							}
							emis_prob = 0.0;
							for (int countk = 0; countk < M; countk++)emis_prob += GAUSS_PROB_[i][countk] * GAUSS[i][countk]->getProb(*tmp);
							temp += BETA->at<double>(k, j + 1)*TRANS_.at<double>(i, k)*emis_prob;
						}
						BETA->at<double>(i, j) = temp;
						delete tmp;
						tmp = NULL;
					}
					//cout << "Processing observation no. " << j << " ..." << endl;
				}
				for (int i = 0; i < N; i++)
				{
					for (int j = 0; j < T; j++)
					{
						double tmp = 0;
						for (int k = 0; k < N; k++)tmp += ALPHA->at<double>(k, j)*BETA->at<double>(k, j);
						GAMMA->at<double>(i, j) = ALPHA->at<double>(i, j)*BETA->at<double>(i, j) / tmp;
						if (j == 0)
						{
							num_INIT.at<double>(j, i) += ALPHA->at<double>(i, j)*BETA->at<double>(i, j);
							den_INIT.at<double>(j, i) += tmp;
						}
						//cout << "Processing observation no. " << j << " ..." << endl;
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
							//for (int countj = 0; countj < 15; countj++)coeff[countj] = SEQ[countf]->getObs()->at<double>(countj, i + 1);
							tmp = new NColVector(SEQ[countf].getObs(i+1));
							if (tmp == NULL)
							{
								cout << "Null pointer returned during initialization of NCOlVector." << endl;
								return;
							}
							emis_prob = 0.0;
							for (int countk = 0; countk < M; countk++)
								emis_prob += GAUSS_PROB_[k][countk] * GAUSS[k][countk]->getProb(*tmp);

							temp.at<double>(j, k) = ALPHA->at<double>(j, i)*TRANS_.at<double>(j, k)*emis_prob*BETA->at<double>(k, i + 1) / (Prob[countf]+1e-30);
							delete tmp;
							tmp = NULL;
						}
					}
					CHI.push_back(temp);
					//cout << "Processing observation no. " << i << " ..." << endl;
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
							for (int j = 0; j < N; j++)temp1 += TRANS_.at<double>(j,i)*ALPHA->at<double>(j, t - 1);
							temp1 *= GAUSS[i][m]->getProb((SEQ[countf].getObs(t)))*BETA->at<double>(i, t);
							num_GAUSS_PROB[i][m] += GAUSS_PROB[i][m] * (temp1+1e-30);
						}
						temp2 += GAUSS_PROB[i][m] * temp1;
					}
					for (int m = 0; m < M; m++)
					{
						den_GAUSS_PROB[i][m] += temp2;
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
							temp *= GAUSS[i][m]->getProb((SEQ[countf].getObs(t)))*BETA->at<double>(i, t);
							num_GAUSS_MEAN[i][m] = num_GAUSS_MEAN[i][m] + ((SEQ[countf].getObs(t)))*temp;
							NColVector v1 = (SEQ[countf].getObs(t));
							v1=v1-*GAUSS_MEAN_[i][m];
							num_GAUSS_VAR[i][m] = num_GAUSS_VAR[i][m] + ((v1*(*v1.transpose())))*temp;
							den_GAUSS_MEAN[i][m] += temp;
							den_GAUSS_VAR[i][m] += temp;
						}
					}
				}
				for (int i = 0; i < N; i++)
				{

					//for (int countj = 0; countj < 15; countj++)coeff[countj] = SEQ[countf]->getObs()->at<double>(countj, 0);
					tmp = new NColVector(SEQ[countf].getObs(0));
					if (tmp == NULL)
					{
						cout << "Null pointer returned during initialization of NCOlVector." << endl;
						return;
					}
					emis_prob = 0.0;
					for (int countk = 0; countk < M; countk++)emis_prob += GAUSS_PROB_[i][countk] * GAUSS[i][countk]->getProb(*tmp);
					ALPHA->at<double>(i, 0) = INIT_.at<double>(0, i)*emis_prob;
					delete tmp;
					tmp = NULL;
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
						for (int countk = 0; countk < M; countk++)emis_prob += GAUSS_PROB_[i][countk] * (GAUSS[i][countk]->getProb(*tmp));
						ALPHA->at<double>(i, j) = emis_prob;
						delete tmp;
						tmp = NULL;
					}
					//cout << "Processing observation no. " << j << " ..." << endl;
				}
				newProb[countf] = 0.0;
				for (int j = 0; j < N; j++)
				{
					newProb[countf] += ALPHA->at<double>(j, T - 1);
				}
				diffProb[countf] = newProb[countf] - Prob[countf];
			
				//print size of all
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
				*GAUSS_MEAN_[i][j] = (1 / den_GAUSS_MEAN[i][j])*num_GAUSS_MEAN[i][j];
				*GAUSS_VAR_[i][j] = (1 / den_GAUSS_VAR[i][j])*num_GAUSS_VAR[i][j];
				}
			count++;

		} while (probDiff.getNorm() > probLimit && count << maxCount);
		*TRANS = TRANS_.clone();
		*INIT = INIT_.clone();
		for (int i = 0; i < states; i++)
			for (int j = 0; j < mixtures; j++)
			{
			GAUSS_PROB[i][j] = GAUSS_PROB_[i][j];
			*GAUSS_MEAN[i][j] = *GAUSS_MEAN_[i][j];
			*GAUSS_VAR[i][j] = *GAUSS_VAR_[i][j];
			GAUSS[i][j]->setMean(*GAUSS_MEAN[i][j]);
			GAUSS[i][j]->setVar(*GAUSS_VAR[i][j]);
			}
		print();
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
	static double getMaxLikelihood(Obs &o, Mat &trans, Mat &init, NColVector* gauss_mean, Mat *gauss_var, double *gauss_prob)
	{
		return 0;
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
	unsigned long int getId(){return id;}
	bool train();
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

bool Person::train()
{
	Obs *observation = new Obs[num_of_faces];
	if (observation==NULL)
	{
		cout << "Null pointer returned during initialization of observation." << endl;
		return false;
	}
	for(int iter = 0;iter<num_of_faces;iter++)
	{
	//declare a filestream which reads images in folder with name same as id and path as specified
	string filename = "";
	filename += to_string(id)+"/"+to_string(iter)+".jpg";
	Mat image;
	image = cv::imread(filename,CV_LOAD_IMAGE_GRAYSCALE);
	if(!image.data)
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
	//store observation sequence
		observation[iter] = Obs(DCT,DCT_NUM1*DCT_NUM2);
	}
	
	//train hmm
	HMM model(observation,num_of_faces,DCT_NUM1*DCT_NUM2,5,1);	   
	model.train(0.8,20);
	//store hmm, file has header denoting training maximized likelihood
	string outfilename = "";
	outfilename += to_string(id)+".hmm";
	ofstream outfile;
	outfile.open(outfilename.c_str(),ios::binary);
	outfile.write((char*)&model,sizeof(model));
	outfile.close();
}

int main(int argc,char **argv)
{
	int nof;
	cout << "Enter number of faces you want to take input from. Maximum is 10, but I dare you give more than 2: " << endl;
	cin >> nof;
	Person foo(nof);
	foo.train();

	//create a video stream
	
	//for each frame, run over all hmm we have in the parent folder
	
	//calculate likelihoods

	//report likelihoods in a table for each person
	return 0;
}
