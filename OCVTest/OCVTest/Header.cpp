#include "Header.h"
using namespace std;
using namespace cv;
string face_cascade_name = "E:/opencv/sources/data/haarcascades/haarcascade_frontalface_alt.xml";//location of face detector in haar cascades
CascadeClassifier face_cascade;//specify a new CascadeClassifier which will detect faces
int filenumber;
void videoFaceDetect()
{
	CvCapture* capture = cvCaptureFromCAM(CV_CAP_ANY);//declare a variable to detect and obtain input from any attached camera device
	cvNamedWindow("Webcam", 1);						//create a new window in which to display video streaming from webcam
	cvNamedWindow("Detected", 1);					//create a new window in which to display the detected face
	Mat img;				//store captured frame from the video
	Vector<Mat> crop;		//a vector(similar to a dynamic array) of Mat objects(images) which will store the cropped faces 
	while (true)								//infinite loop -keep capturing video until the user presses Esc
	{
		if (!cvGrabFrame(capture))				//check if it is possible capture a frame from the video stream
		{
			cout << "Could not grab frame from webcam." << endl;
			break;
		}
		img = cvRetrieveFrame(capture);			//get the frame from the video stream;
		imshow("Webcam", img);					//show the captured frame
		for (int i = 0; i < crop.size(); i++)	//for loop to destroy all the windows showing cropped faces from the previously captured frame
		{
			stringstream str1;
			str1 << "Cropped" << i;
			crop.pop_back();
			destroyWindow(str1.str());
		}
		DetectAndCrop(img, crop);		//call function to detect and crop images and store them in assigned vector
		imshow("Detected", img);		//show the captured frame which indicates detected faces by rectangles
		for (int i = 0; i < crop.size(); i++)//for loop to display the cropped images in separate windows
		{
			stringstream str1;
			str1 << "Cropped" << i;
			imshow(str1.str(), crop[i]);
		}
		int ch = waitKey(30);		//wait for user to press a key if he wants to exit
		if ((char)ch == 27)			//check if user pressed Esc key
		{
			cout << "Esc key pressed. Exiting..." << endl;
			break;
		}

	}
	cvReleaseCapture(&capture);			//release the capture stream
	destroyAllWindows();
	return;
}
void DetectAndCrop(Mat &img, Vector<Mat> &output)
{
	vector<Rect> faces;		//Rectangle vector to store rectangles around faces detected
	Mat frame_gray;			//Store greyscale version of original 
	Mat res;				//Store resized cropped image
	Mat gray;				//Store gray cropped image
	string filename;		//filename in which cropped face will be stored
	stringstream ssfn;		//to append number to each filename, depending on image number
	cvtColor(img, frame_gray, COLOR_BGR2GRAY);	//convert original image to gray
	equalizeHist(frame_gray, frame_gray);		//histogram equalization
	if (!face_cascade.load(face_cascade_name))	//see if Cascade has loaded
	{
		cout << "Unable to load." << endl;
		return;
	}
	//detect faces in image and store the surrounding rectangle of each face in a vector
	face_cascade.detectMultiScale(frame_gray, faces, 1.1, 5, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));
	//rectangle object to store each rectangle separately
	Rect roi_c;

	size_t ic = 0; // ic is index of current element
	int ac = 0; // ac is area of current element

	for (ic = 0; ic < faces.size(); ic++) // Iterate through all current elements (detected faces)

	{
		roi_c.x = faces[ic].x;		//store x-coordinate of starting corner of rectangle
		roi_c.y = faces[ic].y;		//store y-coordinate of starting corner of rectangle
		roi_c.width = (faces[ic].width);		//get width of rectangle
		roi_c.height = (faces[ic].height);		//get height of rectangle

		ac = roi_c.width * roi_c.height;// Get the area of current element (detected face)
		output.push_back(img(roi_c));//push the detected and cropped face into the Mat vector
		resize(output[ic], res, Size(128, 128), 0, 0, INTER_LINEAR); // This will be needed later while saving images
		cvtColor(output[ic], gray, CV_BGR2GRAY); // Convert cropped image to Grayscale

		// Form a filename
		ssfn << filenumber << ".jpg";
		//convert filename to string
		filename = ssfn.str();
		//increment filenumber so as to store next image in consecutively numbered file
		filenumber++;
		//store the cropped greyscale image in a file
		imwrite(filename, gray);

		Point pt1(faces[ic].x, faces[ic].y); // Display detected faces on main window - live stream from camera
		Point pt2((faces[ic].x + faces[ic].height), (faces[ic].y + faces[ic].width));
		rectangle(img, pt1, pt2, Scalar(0, 255, 0), 2, 8, 0);	//Display a rectangle around detected face in oiginal image
	}

}
int display_caption(Mat &A, Mat &B, char* ch, char* winname, int del)	//function which displays a text in  a new window
{
	B = Mat::zeros(Size(A.rows, A.cols), A.type());	//create a zero image matrix(black window)
	putText(B, ch, Point(B.cols / 4, B.rows / 4), CV_FONT_HERSHEY_COMPLEX, 1, Scalar(255, 0, 0));
	imshow(winname, B);
	int c = waitKey(del);
	if ((char)c == 27)
		return -1;
	destroyWindow(winname);
	return 0;
}
int display_image(Mat &A, char *winname, int del)		//function to display image 'A' in a window 'winname' for time interval 'del'
{
	imshow(winname, A);		//show image in given window
	int c = waitKey(del);	//wait for user to press Esc for 'del' time interval 
	if ((char)c == 27)	
		return -1;			
	return 0;		//after del time, automatically exit
}
int smoothImage(char *img)
{
	Mat src = imread(img, 1);// read the image specified by the parameter
	Mat dst;	//to store output(destinantion) image
	int kernel_w = 31;		//specify maximum size of kernel
	if (!src.data)return -1;	//if image couldn't be loaded
	dst = src.clone();		//copy the original image and display it
	if (display_image(dst, "Original Image", 1000) != 0) { return 0; }
	/// Applying Homogeneous blur
	if (display_caption(src, dst, "Homogeneous Blur", "Homogeneous Blur", 1000) != 0) { return 0; }
	for (int i = 1; i < kernel_w; i = i + 2)
	{
		blur(src, dst, Size(i, i), Point(-1, -1));
		if (display_image(dst, "Homogenous Blur", 1000) != 0) { return 0; }
	}
	/// Applying Gaussian blur
	if (display_caption(src, dst, "Gaussian Blur", "Gaussian Blur", 1000) != 0) { return 0; }
	for (int i = 1; i < kernel_w; i = i + 2)
	{
		GaussianBlur(src, dst, Size(i, i), 0, 0);
		if (display_image(dst, "Gaussian Blur", 1000) != 0) { return 0; }
	}
	/// Applying Median blur
	if (display_caption(src, dst, "Median Blur", "Median Blur", 1000) != 0) { return 0; }
	for (int i = 1; i < kernel_w; i = i + 2)
	{
		medianBlur(src, dst, i);
		if (display_image(dst, "Median Blur", 1000) != 0) { return 0; }
	}
	///Applying bilateral blur
	if (display_caption(src, dst, "Bilateral Blur", "Bilateral Blur", 1000) != 0) { return 0; }
	for (int i = 1; i < kernel_w; i = i + 2)
	{
		bilateralFilter(src, dst, i, i * 2, i / 2);
		if (display_image(dst, "Bilateral Blur", 1000) != 0) { return 0; }
	}
	waitKey(0);
	destroyAllWindows();//destroy all windows showing the images before exiting
	return 0;
}