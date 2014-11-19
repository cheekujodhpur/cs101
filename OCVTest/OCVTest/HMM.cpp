#include <iostream>
#include <opencv2\core\core.hpp>
#include <opencv2\contrib\contrib.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <cmath>
#include <cstring>
using namespace std;
using namespace cv;
#define PI 3.141592653589793238
class NRowVector
{
	Mat* v;
	int size;
public:
	NRowVector()
	{
		size = 3;
		v = new Mat(1, 3, CV_64F, Scalar(0));
	}
	NRowVector(int s, double *arr)
	{
		size = s;
		v = new Mat(1, size, CV_64F);
		for (int j = 0; j < size; j++)
			v->at<double>(1, j) = arr[j];

	}
	NRowVector(int s)
	{
		size = s;
		v = new Mat(1, s, CV_64F, Scalar(0));
	}
	double getElement(int i) const
	{
		return v->at<double>(1, i);
	}
	void setElement(int pos, double val)
	{
		v->at<double>(1, pos) = val;
	}
	int getSize() const
	{
		return size;
	}
	NRowVector(const NRowVector* v1)
	{
		size = v1->size;
		v = new Mat(1, size, CV_64F);
		for (int j = 0; j < size; j++)
			setElement(j, v1->getElement(j));
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
		delete v;
		v = new Mat(1, size, CV_64F);
		for (int j = 0; j < size; j++)setElement(j, v1.getElement(j));
	}
	Mat* convertToMat()
	{
		int r = 1;
		int c = size;
		Mat* m = new Mat(r, c, CV_64F);
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
	}
	NColVector(int s, double *arr)
	{
		size = s;
		v = new Mat(size, 1, CV_64F);
		for (int j = 0; j < size; j++)
			v->at<double>(j, 1) = arr[j];

	}
	NColVector(int s)
	{
		size = s;
		v = new Mat(s, 1, CV_64F, Scalar(0));
	}
	double getElement(int i) const
	{
		return v->at<double>(i, 1);
	}
	void setElement(int pos, double val)
	{
		v->at<double>(pos, 1) = val;
	}
	int getSize() const
	{
		return size;
	}
	NColVector(const NColVector* v1)
	{
		size = v1->size;
		v = new Mat(size, 1, CV_64F);
		for (int j = 0; j < size; j++)
			setElement(j, v1->getElement(j));
	}
	NColVector(const NRowVector* v1)
	{
		size = v1->getSize();
		v = new Mat(size, 1, CV_64F);
		for (int j = 0; j < size; j++)
			setElement(j, v1->getElement(j));
	}
	NRowVector* transpose()
	{
		NColVector *v1 = this;
		NRowVector *v2 = new NRowVector(v1->getSize());
		for (int j = 0; j < v1->getSize(); j++)v2->setElement(j, v1->getElement(j));
		delete v1;
		return v2;
	}
	~NColVector()
	{
		delete v;
	}
	NColVector operator=(double d)
	{
		double tmp = sqrt(d*d / size);
		for (int j = 0; j < size; j++)setElement(j, tmp);
		return *this;
	}
	NColVector operator=(const NColVector &v1)
	{
		size = v1.size;
		delete v;
		v = new Mat(size,1, CV_64F);
		for (int j = 0; j < size; j++)setElement(j, v1.getElement(j));
	}
	Mat* convertToMat()
	{
		int r = size;
		int c =  1;
		Mat* m = new Mat(r, c, CV_64F);
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
	Obs(double DCT[][64][64], int size)
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
	void setObs(double DCT[][64][64], int size)
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
	NColVector* getObs(const int &i)
	{
		double* arr = new double[15];
		for (int j = 0; j < 15; j++)
		{
			arr[j] = obs->at<double>(i, j);
		}
		NColVector *v = new NColVector(15, arr);
		delete[] arr;
		return v;
	}
	~Obs()
	{
		delete obs;
	}

};
NRowVector operator*(const NRowVector &v1, const Mat &op)
{
	if (op.rows != v1.getSize())
		return NRowVector(v1.getSize());
	int r = 1;
	int c = op.cols;
	NRowVector prod = NRowVector(c);
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
	if (v1.getSize() != v2.getSize())return Mat::zeros(1, 1, CV_64F);
	Mat prod(v1.getSize(), v2.getSize(), CV_64F, 0);
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
	NColVector *v1 = new NColVector(v);
	for (int j = 0; j < v.getSize(); j++)v1->setElement(j, d*v.getElement(j));
	return v1;
}
NColVector operator*(const NColVector &v, const double &d)
{
	NColVector *v1 = new NColVector(v);
	for (int j = 0; j < v.getSize(); j++)v1->setElement(j, d*v.getElement(j));
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
	NRowVector *v1 = new NRowVector(v);
	for (int j = 0; j < v.getSize(); j++)v1->setElement(j, d*v.getElement(j));
	return v1;
}
NRowVector operator*(const NRowVector &v, const double &d)
{
	NRowVector *v1 = new NRowVector(v);
	for (int j = 0; j < v.getSize(); j++)v1->setElement(j, d*v.getElement(j));
	return v1;
}
double operator*(const NRowVector &v1, const NColVector &v2)
{
	double prod = 0;
	int t = v1.getSize() > v2.getSize() ? v1.getSize() : v2.getSize();
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
	NColVector sum = NColVector(s, arr);
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
		var = sigma;
	}
	NColVector getMean()
	{
		return mean;
	}
	Mat getVar()
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
		NRowVector mean_t = *mean.transpose();
		double num = exp(-((mean_t*inv_covar)*mean) / 2);
		double den = sqrt(2 *pow(PI,var.rows)*determinant(var));
		double p = num / den;
		return  p;
	}

};

class HMM
{
private:
	Obs** SEQ;
	Mat* TRANS;
	Mat* INIT;
	NColVector **GAUSS_MEAN;
	Mat **GAUSS_VAR;
	double** GAUSS_PROB;
	int faces;
	int n_obs;
	int states;
	int mixtures;
	Gaussian*** GAUSS;
public:

	HMM(Obs** O,int num_faces,int num_obs,int num_states,int num_mixtures)
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
	void INIT_TRANS()
	{
		TRANS = new Mat(states, states, CV_64F, ((double)(1.0)) / states);
	}
	void INIT_GAUSSIAN()
	{
		GAUSS = new Gaussian**[states];
		for (int j = 0; j < states; j++)
		{
			GAUSS[j] = new Gaussian*[mixtures];
			for (int k = 0; k < mixtures; k++)
				GAUSS[j][k] = new Gaussian();
		}
		//statements for initializing mean and variance and for setting the mean and variance of the objects
	}
	void INIT_INIT()
	{
		INIT = new Mat(1, states, CV_64F, ((double)(1.0)) / states);
	}
	void train(double probLimit, int maxCount)
	{
		Mat** GAUSS_VAR_;
		NColVector** GAUSS_MEAN_;
		double** GAUSS_PROB_;
		GAUSS_VAR_ = new Mat*[states];
		GAUSS_MEAN_ = new NColVector*[states];
		GAUSS_PROB_ = new double*[states];
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
		int T = SEQ[0]->getObs()->cols;
		int N = TRANS->rows;		//=TRANS.rows=TRANS.cols= No. of states
		int M = mixtures;	// No .of mixtures in the Gaussian
		int count = 0;
		int countf = 0;
		Mat ALPHA(N, T, CV_64F);
		Mat BETA(N, T, CV_64F);
		Mat GAMMA(N, T, CV_64F);
		double coeff[15];
		double emis_prob;
		NColVector* tmp;
		double Prob,newProb;
		//the variables prefixed with num_ and den_ are for storing the total of the updates for all faces so that averaging out takes place in the last step
		Mat num_TRANS=Mat::zeros(TRANS->rows,TRANS->cols,CV_64F);
		Mat num_INIT = Mat::zeros(INIT->rows, INIT->cols, CV_64F);
		NColVector **num_GAUSS_MEAN;
		Mat** num_GAUSS_VAR;
		Mat den_TRANS = Mat::zeros(TRANS->rows, TRANS->cols, CV_64F);
		Mat den_INIT = Mat::zeros(INIT->rows, INIT->cols, CV_64F);
		double **num_GAUSS_PROB,**den_GAUSS_PROB,**den_GAUSS_MEAN,**den_GAUSS_VAR;
		num_GAUSS_PROB = new double*[states];
		num_GAUSS_MEAN = new NColVector*[states];
		num_GAUSS_VAR = new Mat*[states];
		den_GAUSS_PROB = new double*[states];
		den_GAUSS_MEAN = new double*[states];
		den_GAUSS_VAR = new double*[states];
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
				num_GAUSS_MEAN[i][j] = NColVector(0*GAUSS_MEAN[i][j]);
				num_GAUSS_VAR[i][j] = Mat::zeros(15,15,CV_64F);
				den_GAUSS_PROB[i][j] = 0;
				den_GAUSS_MEAN[i][j] = 0;
				den_GAUSS_VAR[i][j] = 0;
			}
		}
		do{
			for (countf = 0; countf < faces; countf++)
			{
				ALPHA=Mat(N, T, CV_64F,0);
				BETA=Mat(N, T, CV_64F,0);
				GAMMA=Mat(N, T, CV_64F,0);
				for (int i = 0; i < N; i++)
				{
					for (int countj = 0; countj < 15; countj++)coeff[countj] = SEQ[countf]->getObs()->at<double>(countj, 0);
					tmp = new NColVector(15, coeff);
					emis_prob = 0.0;
					for (int countk = 0; countk < M; countk++)emis_prob += GAUSS_PROB_[i][countk] * GAUSS[i][countk]->getProb(tmp);
					delete tmp;
					ALPHA.at<double>(i, 0) = INIT_.at<double>(0, i)*emis_prob;
				}
				for (int j = 1; j < T; j++)
				{
					for (int i = 0; i < N; i++)
					{
						double temp = 0;
						for (int k = 0; k < N; k++)
						{
							temp += ALPHA.at<double>(k, j - 1)*TRANS_.at<double>(k, i);
						}
						for (int countj = 0; countj < 15; countj++)coeff[countj] = SEQ[countf]->getObs()->at<double>(countj, j);
						tmp = new NColVector(15, coeff);
						emis_prob = 0.0;
						for (int countk = 0; countk < M; countk++)emis_prob += GAUSS_PROB_[i][countk] * GAUSS[i][countk]->getProb(tmp);
						delete tmp;
						ALPHA.at<double>(i, j) = emis_prob*temp;
					}
				}
				Prob = 0.0;
				for (int j = 0; j < N; j++)
				{
					Prob += ALPHA.at<double>(j, T - 1);
				}
				for (int i = 0; i < N; i++)BETA.at<double>(i, T - 1) = 1;
				for (int j = T - 2; j >= 0; j--)
				{
					for (int i = 0; i < N; i++)
					{
						double temp = 0;
						for (int k = 0; k < N; k++)
						{
							for (int countj = 0; countj < 15; countj++)coeff[countj] = SEQ[countf]->getObs()->at<double>(countj, j + 1);
							tmp = new NColVector(15, coeff);
							emis_prob = 0.0;
							for (int countk = 0; countk < M; countk++)emis_prob += GAUSS_PROB_[i][countk] * GAUSS[i][countk]->getProb(tmp);
							delete tmp;
							temp += BETA.at<double>(k, j + 1)*TRANS_.at<double>(i, k)*emis_prob;
						}
						BETA.at<double>(i, j) = temp;
					}
				}
				for (int i = 0; i < N; i++)
				{
					for (int j = 0; j < T; j++)
					{
						double tmp = 0;
						for (int k = 0; k < N; k++)tmp += ALPHA.at<double>(k, j)*BETA.at<double>(k, j);
						GAMMA.at<double>(i, j) = ALPHA.at<double>(i, j)*BETA.at<double>(i, j) / tmp;
						if (j == 0)
						{
							num_INIT.at<double>(j, i) += ALPHA.at<double>(i, j)*BETA.at<double>(i, j);
							den_INIT.at<double>(j, i) += tmp;
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
							for (int countj = 0; countj < 15; countj++)coeff[countj] = SEQ[countf]->getObs()->at<double>(countj, i + 1);
							tmp = new NColVector(15, coeff);
							emis_prob = 0.0;
							for (int countk = 0; countk < M; countk++)emis_prob += GAUSS_PROB_[i][countk] * GAUSS[i][countk]->getProb(tmp);
							delete tmp;
							temp.at<double>(j, k) = ALPHA.at<double>(j, i)*TRANS_.at<double>(i, j)*emis_prob*BETA.at<double>(k, i + 1) / Prob;
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
						tmp2 += GAMMA.at<double>(i, k);
					}
					num_TRANS.at<double>(i, j) += tmp1;
					den_TRANS.at<double>(i, j) += tmp2;
					}
				for (int i = 0; i < N; i++)
				{
					double temp2=0.0;
					for (int m = 0; m < M; m++)
					{
						double temp1;
						for (int t = 1; t < T; t++)
						{
							temp1 = 0.0;
							for (int j = 0; j < N; j++)temp1 += TRANS_.at<double>(i, j)*ALPHA.at<double>(j, t - 1);
							temp1 *= GAUSS[i][m]->getProb(*(SEQ[countf]->getObs(t)))*BETA.at<double>(i,t);
							num_GAUSS_PROB[i][m] += GAUSS_PROB[i][m] * temp1;
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
							for (int j = 0; j < N; j++)temp += TRANS_.at<double>(i, j)*ALPHA.at<double>(j, t - 1);
							temp *= GAUSS[i][m]->getProb(*(SEQ[countf]->getObs(t)))*BETA.at<double>(i, t);
							num_GAUSS_MEAN[i][m] = num_GAUSS_MEAN[i][m] + temp*(*(SEQ[countf]->getObs(t)));
							num_GAUSS_VAR[i][m] = num_GAUSS_VAR[i][m] + temp*((*SEQ[countf]->getObs(t) - GAUSS_MEAN_[i][m])*(*SEQ[countf]->getObs(t) - GAUSS_MEAN_[i][m]).transpose());
							den_GAUSS_MEAN[i][m] += temp;
							den_GAUSS_VAR[i][m] += temp;
						}
					}
				}

			}
			for (int i = 0; i < INIT_.rows; i++)
				for (int j = 0; j < INIT_.cols; j++)INIT_.at<double>(i, j) = num_INIT.at<double>(i, j) / den_INIT.at<double>(i, j);
			for (int i = 0; i < TRANS_.rows; i++)
				for (int j = 0; j < TRANS_.cols; j++)TRANS_.at<double>(i, j) = num_TRANS.at<double>(i, j) / den_TRANS.at<double>(i, j);
			for (int i = 0; i < N; i++)
				for (int j = 0; j < M; j++)
				{
					GAUSS_PROB_[i][j] = num_GAUSS_PROB[i][j] / den_GAUSS_PROB[i][j];
					GAUSS_MEAN_[i][j] = (1 / den_GAUSS_MEAN[i][j])*num_GAUSS_MEAN[i][j];
					GAUSS_VAR_[i][j] = (1 / den_GAUSS_VAR[i][j])*num_GAUSS_VAR[i][j];
				}
			for (int i = 0; i < N; i++)
			{

				for (int countj = 0; countj < 15; countj++)coeff[countj] = SEQ[countf]->getObs()->at<double>(countj, 0);
				tmp = new NColVector(15, coeff);
				emis_prob = 0.0;
				for (int countk = 0; countk < M; countk++)emis_prob += GAUSS_PROB_[i][countk] * GAUSS[i][countk]->getProb(tmp);
				delete tmp;
				ALPHA.at<double>(i, 0) = INIT_.at<double>(0, i)*emis_prob;
			}
			for (int j = 1; j < T; j++)
			{
				for (int i = 0; i < N; i++)
				{
					double temp = 0;
					for (int k = 0; k < N; k++)
					{
						temp += ALPHA.at<double>(k, j - 1)*TRANS_.at<double>(k, i);
					}
					for (int countj = 0; countj < 15; countj++)coeff[countj] = SEQ[countf]->getObs()->at<double>(countj, j);
					tmp = new NColVector(15, coeff);
					emis_prob = 0.0;
					for (int countk = 0; countk < M; countk++)emis_prob += GAUSS_PROB_[i][countk] * (GAUSS[i][countk]->getProb(tmp));
					delete tmp;
					ALPHA.at<double>(i, j) = emis_prob;
				}
			}
			newProb = 0.0;
			for (int j = 0; j < N; j++)
			{
				newProb += ALPHA.at<double>(j, T - 1);
			}
			count++;
		} while ((newProb - Prob) > probLimit && count<<maxCount);
		*TRANS = TRANS_.clone();
		*INIT = INIT_.clone();
		for (int i = 0; i < states; i++)
			for (int j = 0; j < mixtures; j++)
			{
				GAUSS_PROB[i][j] = GAUSS_PROB_[i][j];
				GAUSS_MEAN[i][j] = GAUSS_MEAN_[i][j];
				GAUSS_VAR[i][j] = GAUSS_VAR_[i][j];
				GAUSS[i][j]->setMean(GAUSS_MEAN[i][j]);
				GAUSS[i][j]->setVar(GAUSS_VAR[i][j]);
			}
	}
	void init( Mat &A)
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

};
void addPerson()
{

}
void addFace(Person p)
{

}