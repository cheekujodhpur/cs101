#ifndef HEADER_H
#define HEADER_H
#include <highgui.h>
#include <cv.h>
#include <iostream>
#include <conio.h>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/contrib/contrib.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <string>
#include <vector>
void videoFaceDetect();//function to detect faces from continuous video stream from any attached camera device 
void DetectAndCrop(cv::Mat &img, cv::Vector<cv::Mat> &output);//function to detect faces in given image, crop the faces and store them in separate files
int display_caption(cv::Mat &A, cv::Mat &B, char* ch, char* winname, int del);//function which displays a text in  a new window
int display_image(cv::Mat &A, char* winname, int del);//function to display an image in a new window 
int smoothImage(char* img);//performs several linear smoothing operations on an image
#endif