#ifndef CVHMM_H
#define CVHMM_H

#include <iostream>
#include <cmath>
using namespace std;
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

class CvHMM {
public:
	CvHMM(){};
	/* Generates M sequence of states and emissions from a Markov model */
	static void generate(const int &_N,const int &_M, const cv::Mat &_TRANS,const cv::Mat &_EMIS, const cv::Mat &_INIT, cv::Mat &seq, cv::Mat &states)
	{
		seq = cv::Mat(_M,_N,CV_32S);
		states = cv::Mat(_M,_N,CV_32S);
		for (int i=0;i<_M;i++)
		{
			cv::Mat seq_,states_;
			generate(_N,_TRANS,_EMIS,_INIT,seq_,states_);
			for (int t=0;t<_N;t++)
			{
				seq.at<int>(i,t) = seq_.at<int>(0,t);
				states.at<int>(i,t) = states_.at<int>(0,t);
			}
		}

	}
	/* Generates a sequence of states and emissions from a Markov model */
	static void generate(const int &_N,const cv::Mat &_TRANS,const cv::Mat &_EMIS, const cv::Mat &_INIT, cv::Mat &seq, cv::Mat &states)
	{			
		seq = cv::Mat(1,_N,CV_32S);
		states = cv::Mat(1,_N,CV_32S);
		int n_states = _TRANS.rows;
		cv::Mat cumulative_emis(_EMIS.size(),CV_64F); 
		for (int r=0;r<cumulative_emis.rows;r++)
			cumulative_emis.at<double>(r,0) = _EMIS.at<double>(r,0);
		for (int r=0;r<cumulative_emis.rows;r++)
			for (int c=1;c<cumulative_emis.cols;c++)
				cumulative_emis.at<double>(r,c) = cumulative_emis.at<double>(r,c-1) + _EMIS.at<double>(r,c);
		cv::Mat cumulative_trans(_TRANS.size(),CV_64F); 
		for (int r=0;r<cumulative_trans.rows;r++)
			cumulative_trans.at<double>(r,0) = _TRANS.at<double>(r,0);
		for (int r=0;r<cumulative_trans.rows;r++)
			for (int c=1;c<cumulative_trans.cols;c++)
				cumulative_trans.at<double>(r,c) = cumulative_trans.at<double>(r,c-1) + _TRANS.at<double>(r,c);
		cv::Mat cumulative_init(_INIT.size(),CV_64F);
		cumulative_init.at<double>(0,0) = _INIT.at<double>(0,0);
		for (int c=1;c<cumulative_init.cols;c++)
			cumulative_init.at<double>(0,c) = cumulative_init.at<double>(0,c-1) + _INIT.at<double>(0,c);
		double r_init,r_trans,r_emis;
		r_init = (double) rand()/RAND_MAX;
		int last_state;
		for (int c=0;c<cumulative_init.cols;c++)
			if (r_init <= cumulative_init.at<double>(0,c))
			{
				last_state = c;
				break;
			}		
		for (int t=0;t<_N;t++)
		{
			r_trans = (double)rand()/RAND_MAX;			
			for (int i=0;i<cumulative_trans.cols;i++)			
				if (r_trans <= cumulative_trans.at<double>(last_state,i))
				{
					states.at<int>(0,t) = i;
					break;
				}
			r_emis = (double)rand()/RAND_MAX;			
			for (int i=0;i<cumulative_emis.cols;i++)
			{
				if (r_emis <= cumulative_emis.at<double>(states.at<int>(0,t),i))
				{
					seq.at<int>(0,t) = i;
					break;
				}			
			}
			last_state = states.at<int>(0,t);
		}
	}

	static void getUniformModel(const int &n_states,cv::Mat &TRANS,cv::Mat &INIT,cv::Mat &MEAN,cv::Mat &VAR)
	{
		TRANS = cv::Mat(n_states,n_states,CV_64F);
		TRANS = 1.0/n_states;
		INIT = cv::Mat(1,n_states,CV_64F);
		INIT = 1.0/n_states;
		MEAN = cv::Mat::zeros(15,1,CV_64F);
		VAR = cv::Mat::eye(15,15,CV_64F);	
	}


	/*gets multivariate gaussian probability*/
	static double getProb(const cv::Mat &MEAN, const cv::Mat &VAR, const cv::Vec<double,15> &obs)
	{
		double prob = 0;
		double PI = 3.14159;
		double det = cv::determinant(VAR);
		cv::Mat DIFF = cv::Mat(obs)-MEAN;
		cv::Mat DIFF_T;
		cv::transpose(DIFF,DIFF_T);
		cv::Mat IVAR;
		cv::invert(VAR,IVAR);
		cv::Mat rhs = DIFF_T*IVAR;
		cv::Mat e = rhs*DIFF;
		double ee = e.at<double>(0,0);
		double num = exp(-(ee)/2);
		double den = sqrt(2*pow(PI,VAR.rows)*abs(det));
		prob = num/den;
		return prob;
	}

	/* Calculates maximum likelihood estimates of transition and emission probabilities from a sequence of emissions */
	static void train(const cv::Mat &seq, const int max_iter, cv::Mat &TRANS, cv::Mat &MEAN, cv::Mat &VAR, cv::Mat &INIT,bool UseUniformPrior = false)
	{
		/* A Revealing Introduction to Hidden Markov Models, Mark Stamp */
		// 1. Initialization
		int iters = 0;				
		int T = seq.cols; // number of element per sequence
		int C = seq.rows; // number of sequences
		int N = TRANS.rows; // number of states | also N = TRANS.cols | TRANS = A = {aij} - NxN
		correctModel(TRANS,INIT);
		cv::Mat FTRANS,FINIT,FMEAN,FVAR;
		FTRANS = TRANS.clone();
		FMEAN = MEAN.clone();
		FVAR = VAR.clone();
		FINIT = INIT.clone();
		// compute a0		
		cv::Mat a(N,T,CV_64F);
		cv::Mat c(1,T,CV_64F); c.at<double>(0,0) = 0;
		for (int i=0;i<N;i++)
		{
			a.at<double>(i,0) = INIT.at<double>(0,i);//*EMIS.at<double>(i,seq.at<int>(0,0));
			c.at<double>(0,0) += a.at<double>(i,0); 
		}
		// scale the a0(i)
		c.at<double>(0,0) = 1/c.at<double>(0,0);
		for (int i=0;i<N;i++)
			a.at<double>(i,0) *= c.at<double>(0,0);
		double logProb = -DBL_MAX;
		double oldLogProb;
		int data = 0;
		do {
			oldLogProb = logProb;
			// 2. The a-pass
			// compute at(i)
			for (int t=1;t<T;t++)
			{
				c.at<double>(0,t) = 0;
				for (int i=0;i<N;i++)
				{
					a.at<double>(i,t) = 0;
					for (int j=0;j<N;j++)				
						a.at<double>(i,t) += a.at<double>(i,t-1)*TRANS.at<double>(j,i);
					a.at<double>(i,t) = a.at<double>(i,t);// * EMIS.at<double>(i,seq.at<int>(data,t));
					c.at<double>(0,t)+=a.at<double>(i,t);
				}
				// scale at(i)
				c.at<double>(0,t) = 1/c.at<double>(0,t);
				for (int i=0;i<N;i++)
					a.at<double>(i,t)=c.at<double>(0,t)*a.at<double>(i,t);
			}
			// 3. The B-pass
			cv::Mat b(N,T,CV_64F);
			// Let Bt-1(i) = 1 scaled by Ct-1
			for (int i=0;i<N;i++)
				b.at<double>(i,T-1) = c.at<double>(0,T-1);
			// B-pass
			for (int t=T-2;t>-1;t--)
				for (int i=0;i<N;i++)
				{
					b.at<double>(i,t) = 0;
					for (int j=0;j<N;j++)
						b.at<double>(i,t) += TRANS.at<double>(i,j);//*EMIS.at<double>(j,seq.at<int>(data,t+1))*b.at<double>(j,t+1);
					// scale Bt(i) with same scale factor as at(i)
					b.at<double>(i,t) *= c.at<double>(0,t);
				}
			// 4. Compute  Yt(i,j) and Yt(i)
			double denom;
			int index;
			cv::Mat YN(N,T,CV_64F);
			cv::Mat YNN(N*N,T,CV_64F);
			for (int t=0;t<T-1;t++)
			{
				denom = 0;
				for (int i=0;i<N;i++)
					for (int j=0;j<N;j++)
						denom += a.at<double>(i,t)*TRANS.at<double>(i,j);//*EMIS.at<double>(j,seq.at<int>(data,t+1))*b.at<double>(j,t+1);
				index = 0;
				for (int i=0;i<N;i++)
				{
					YN.at<double>(i,t) = 0;
					for (int j=0;j<N;j++)
					{
						YNN.at<double>(index,t) = (a.at<double>(i,t)*TRANS.at<double>(i,j));//*EMIS.at<double>(j,seq.at<int>(data,t+1))*b.at<double>(j,t+1))/denom;
						YN.at<double>(i,t)+=YNN.at<double>(index,t);
						index++;
					}
				}
			}
			// 5. Re-estimate A,B and pi
			// re-estimate pi		
			for (int i=0;i<N;i++)
				INIT.at<double>(0,i) = YN.at<double>(i,0);
			// re-estimate A
			double numer;
			index = 0;
			for (int i=0;i<N;i++)
				for (int j=0;j<N;j++)
				{
					numer = 0;
					denom = 0;
					for (int t=0;t<T-1;t++)
					{
						numer += YNN.at<double>(index,t);
						denom += YN.at<double>(i,t);
					}
					TRANS.at<double>(i,j) = numer/denom;
					index++;
				}
			// re-estimate B
			for (int i=0;i<N;i++)
				for (int j=0;j<1;j++)
				{
					numer = 0;
					denom = 0; 
					for (int t=0;t<T-1;t++)
					{
						if (seq.at<int>(data,t)==j) 
							numer+=YN.at<double>(i,t);
						denom += YN.at<double>(i,t);
					}
				}
			correctModel(TRANS,INIT);
			//FTRANS = (FTRANS*(data+1)+TRANS)/(data+2);
			//FEMIS = (FEMIS*(data+1)+EMIS)/(data+2);
			//FINIT = (FINIT*(data+1)+INIT)/(data+2);
			// 6. Compute log[P(O|y)]
			logProb = 0;
			for (int i=0;i<T;i++)
				logProb += log(c.at<double>(0,i));
			logProb *= -1;
			// 7. To iterate or not
			data++;
			if (data >= C)
			{
				data = 0;
				iters++;
			}
		} while (iters<max_iter && logProb>oldLogProb);
		correctModel(FTRANS,FINIT);
		TRANS = FTRANS.clone();
		INIT = FINIT.clone();
	}
	static void correctModel(cv::Mat &TRANS, cv::Mat &INIT)
	{
		double eps = 1e-30;
		for (int i=0;i<TRANS.rows;i++)
			for (int j=0;j<TRANS.cols;j++)
				if (TRANS.at<double>(i,j)==0)
					TRANS.at<double>(i,j)=eps;
		for (int i=0;i<INIT.cols;i++)
			if (INIT.at<double>(0,i)==0)
				INIT.at<double>(0,i)=eps;
		double sum;
		for (int i=0;i<TRANS.rows;i++)
		{
			sum = 0;
			for (int j=0;j<TRANS.cols;j++)
				sum+=TRANS.at<double>(i,j);
			for (int j=0;j<TRANS.cols;j++)
				TRANS.at<double>(i,j)/=sum;
		}
		sum = 0;
		for (int j=0;j<INIT.cols;j++)
			sum+=INIT.at<double>(0,j);
		for (int j=0;j<INIT.cols;j++)
			INIT.at<double>(0,j)/=sum;
	}
	static void printPaths(const cv::Mat &PATHS,const cv::Mat &P, const int &t)
	{		
		for (int r=0;r<PATHS.rows;r++)
		{			
			for (int c=0;c<=t;c++)
				cout << PATHS.at<int>(r,c);			
			cout << " - " << P.at<double>(r,t) << "\n";
		}
	}
	static void printModel(const cv::Mat &TRANS,const cv::Mat &EMIS,const cv::Mat &INIT)
	{
		cout << "\nTRANS: \n";
		for (int r=0;r<TRANS.rows;r++)
		{
			for (int c=0;c<TRANS.cols;c++)
				cout << TRANS.at<double>(r,c) << " ";
			cout << "\n";
		}
		cout << "\nEMIS: \n";
		for (int r=0;r<EMIS.rows;r++)
		{
			for (int c=0;c<EMIS.cols;c++)
				cout << EMIS.at<double>(r,c) << " ";
			cout << "\n";
		}
		cout << "\nINIT: \n";
		for (int r=0;r<INIT.rows;r++)
		{
			for (int c=0;c<INIT.cols;c++)
				cout << INIT.at<double>(r,c) << " ";
			cout << "\n";
		}
		cout << "\n";
	}
};

#endif
