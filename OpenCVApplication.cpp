// OpenCVApplication.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "common.h"
#include <math.h>
#include <opencv2/core/utils/logger.hpp>
#include <fstream>
#include "FaceDetection.h"

using namespace cv;
using namespace std;

wchar_t* projectPath;

FaceDetector faceDetector;

struct peak {
	float theta;
	int ro, hval;
	bool operator < (const peak& o) const {
		return hval > o.hval;
	}
};

vector<Point> readWhitePointsFromImage(Mat img) {
	vector<Point> points;

	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			if (img.at<uchar>(i, j) == 255) {
				points.push_back(Point(j, i));
			}
		}
	}

	return points;
}

int getMaxHough(Mat Hough) {
	int max = 0;
	for (int i = 0; i < Hough.rows; i++) {
		for (int j = 0; j < Hough.cols; j++) {
			if (Hough.at<int>(i, j) > max) {
				max = Hough.at<int>(i, j);
			}
		}
	}
	return max;
}

bool isLocalMaxima(Mat Hough, int r, int t, int windowSize) {
	int k = windowSize / 2;
	int max = Hough.at<int>(r, t);
	int maxi = r, maxj = t;

	for (int i = r - k; i <= r + k; i++) {
		for (int j = t - k; j <= t + k; j++) {
			if (i >= 0 && i < Hough.rows && j >= 0 && j < Hough.cols && !(i == r && j == t)) {
				if (Hough.at<int>(i, j) > max) {
					max = Hough.at<int>(i, j);
					maxi = i; maxj = j;
				}
			}
			
		}
	}

	if (maxi == r && maxj == t) 
		return true;

	return false;
}

void houghTransformLine() {
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat img = imread(fname, IMREAD_GRAYSCALE);
		vector<Point> points = readWhitePointsFromImage(img);

		int diagonal = (int)sqrt(img.rows * img.rows + img.cols * img.cols);

		Mat Hough(diagonal + 1, 360, CV_32SC1);
		Hough.setTo(0);

		int n = points.size();
		int dTheta = 1;
		int dRo = 1;
		int k = 10;

		for (int i = 0; i < n; i++) {
			for (int theta = 0; theta < 360; theta += dTheta) {
				Point p = points.at(i);
				float thetaRad = theta * PI / 180.0;

				int ro = p.x * cos(thetaRad) + p.y * sin(thetaRad);

				if (ro >= 0) {
					Hough.at<int>(ro, theta)++;
				}
			}
		}

		Mat houghImg;
		int maxHough = getMaxHough(Hough);
		Hough.convertTo(houghImg, CV_8UC1, 255.f / maxHough);

		imshow("Source image", img);
		imshow("Hough accumulator", houghImg);

		int windowSize = 3;
		vector<peak> peaks;

		for (int i = 0; i < Hough.rows; i++) {
			for (int j = 0; j < Hough.cols; j++) {
				if (isLocalMaxima(Hough, i, j, windowSize)) {
					float thetaRad = j * PI / 180.0;
					peaks.push_back(peak{ thetaRad, i, Hough.at<int>(i, j) });
				}
				
			}
		}

		Mat imgLines = imread(fname, IMREAD_COLOR);
		std::sort(peaks.begin(), peaks.end());

		for (int i = 0; i < k; i++) {
			float theta = peaks.at(i).theta;
			int ro = peaks.at(i).ro;

			float x1, x2;
			if (abs(theta) < 0.1) {
				x1 = ro / cos(theta);
				x2 = (ro - img.cols * sin(theta)) / cos(theta);
				line(imgLines, Point(x1, 0), Point(x2, img.rows), Scalar(0, 255, 0), 1);
			}
			else {
				x1 = ro / sin(theta);
				x2 = (ro - img.rows * cos(theta)) / sin(theta);
				line(imgLines, Point(0, x1), Point(img.cols, x2), Scalar(0, 255, 0), 1);
			}
		}

		imshow("Hough lines", imgLines);
		waitKey();
	}
}

vector<Point2f> readPointsFromFile(const char* path) {
	FILE* f = fopen(path, "r");

	int nr;
	vector<Point2f> points;
	
	fscanf(f, "%d", &nr);
	// printf("%d", nr); 

	for (int i = 0; i < nr; i++) {
		float x, y;
		fscanf(f, "%f %f", &x, &y);
		points.push_back(Point2f(x, y));
	}

	fclose(f);
	return points;
}

void fitLineModel1ClosedForm(Mat img, vector<Point2f> points) {
	int n = points.size();

	float theta1 = 0.0, theta0 = 0.0, a = 0.0, b = 0.0;
	float sumx = 0.0, sumy = 0.0, sumxy = 0.0, sumx2 = 0.0;

	for (int i = 0; i < n; i++) {
		sumx += points[i].x;
		sumy += points[i].y;
		sumxy += points[i].x * points[i].y;
		sumx2 += points[i].x * points[i].x;
	}

	theta1 = (n * sumxy - sumx * sumy) / (n * sumx2 - sumx * sumx);
	theta0 = (1.0 / n) * (sumy - theta1 * sumx);

	Mat img2 = img.clone();
	line(img2, Point(0, theta0), Point(img.cols, theta0 + theta1 * img.cols), Scalar(255, 0, 0), 1);
	imshow("Model 1", img2);
	waitKey();
}

void fitLineModel2ClosedForm(Mat img, vector<Point2f> points) {
	int n = points.size();

	float beta = 0.0, rho =0.0 , sumxy = 0.0, sumx = 0.0, sumy = 0.0, sum2 = 0.0;

	for (int i = 0; i < n; i++) {
		sumx += points[i].x;
		sumy += points[i].y;
		sumxy += points[i].x * points[i].y;
		sum2 += (points[i].y * points[i].y - points[i].x * points[i].x);
	}

	float y = 0.0, x = 0.0;

	y = 2 * sumxy - (2.0 / n) * sumx * sumy;
	x = sum2 + (1.0 / n) * (sumx * sumx - sumy * sumy);

	beta = (- 1.0/2) * atan2(y, x);
	rho = (1.0 / n) * (sumx * cos(beta) + sumy * sin(beta));
	printf("%f %f\n", beta, rho);

	Mat img2 = img.clone();


	if (abs(sin(beta)) > 0.1) {
		float x1 = 0, y1 = (rho - x1 * cos(beta)) / sin(beta);
		float x2 = img.cols, y2 = (rho - x2 * cos(beta)) / sin(beta);
		line(img2, Point(x1, y1), Point(x2, y2), Scalar(255, 255, 0), 1);
	}
	else {
		float x = rho / cos(beta);
		line(img2, Point(x, 0), Point(x, img.rows), Scalar(255, 255, 0), 1);
	}

	imshow("Model 2", img2);
	waitKey();
}

 
void leastMeanSquares() {
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		vector<Point2f> points = readPointsFromFile(fname);

		int height = 500, width = 500;
		Mat img(height, width, CV_8UC3, cv::Scalar(255, 255, 255));
		
		int nr = points.size();
		for (int i = 0; i < nr; i++) {
			Point2f p = points[i];
			/*img.at<Vec3b>(p.y, p.x)[0] = 145; 
			img.at<Vec3b>(p.y, p.x)[1] = 45; 
			img.at<Vec3b>(p.y, p.x)[2] = 129; */
			if (p.x >= 0 && p.x < width && p.y >= 0 && p.y < height)
				circle(img, p, 1, Scalar(145, 45, 129), 1);
		}

		imshow("Points", img);
		waitKey();	

		fitLineModel1ClosedForm(img, points);
		fitLineModel2ClosedForm(img, points);
	}
}

vector<Point> readPointsFromImage(Mat img) {
	vector<Point> points;

	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			if (img.at<uchar>(i, j) == 0) {
				points.push_back(Point(j, i));
			}
		}
	}

	return points;
}

float distance(Point p, float a, float b, float c) {
	float dist = abs(a * p.x + b * p.y + c) / sqrt(a * a + b * b);
	return dist;
}

void ransacLine() {
	char fname[MAX_PATH];

	while (openFileDlg(fname))
	{
		Mat img = imread(fname, IMREAD_GRAYSCALE);
		Mat dst = img.clone();
		vector<Point> points = readPointsFromImage(img);

		float t = 10.0, p = 0.99, q = 0.3;
		int s = 2;

		int n = points.size();
		//printf("Nr of points: %d\n", n);

		int nr_trials = (log(1 - p) / log(1 - pow(q, s)));
		//printf("Nr of trials: %d\n", nr_trials);

		int T = q * n;
		//printf("Threshold T: %d\n", T);

		Point bestp1, bestp2;
		float bestA, bestB, bestC;
		int i = 0, bestModel = 0;

		while (i < nr_trials) {
			i++;

			int r1 = rand() % n;
			int r2 = rand() % n;

			while (r2 == r1) {
				r2 = rand() % n;
			}

			Point p1 = points[r1];
			Point p2 = points[r2];

			float a = p1.y - p2.y;
			float b = p2.x - p1.x;
			float c = p1.x * p2.y - p2.x * p1.y;

			int consensusSet = 0;
			for (int j = 0; j < n; j++) {
				if (distance(points[j], a, b, c) <= t) {
					consensusSet++;
				}
			}

			if (consensusSet > bestModel) {
				bestModel = consensusSet;
				bestA = a; bestB = b; bestC = c;
				bestp1 = p1; bestp2 = p2;
			}

			if (consensusSet > T) break;
		}


		if (abs(bestB) > 5) {
			float x1 = 0, y1 = (-bestC - bestA * x1) / bestB;
			float x2 = img.cols, y2 = (-bestC - bestA * x2) / bestB;
			line(dst, Point(x1, y1), Point(x2, y2), Scalar(0, 255, 0));
		}
		else {
			float y1 = 0, x1 = (-bestC - bestB * y1) / bestA;
			float y2 = img.rows, x2 = (-bestC - bestB * y2) / bestA;
			line(dst, Point(x1, y1), Point(x2, y2), Scalar(0, 255, 0));
		}
		
		imshow("RANSAC", dst);
		waitKey();
	}
}

void testOpenImage()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		imshow("image",src);
		waitKey();
	}
}

void testOpenImagesFld()
{
	char folderName[MAX_PATH];
	if (openFolderDlg(folderName)==0)
		return;
	char fname[MAX_PATH];
	FileGetter fg(folderName,"bmp");
	while(fg.getNextAbsFile(fname))
	{
		Mat src;
		src = imread(fname);
		imshow(fg.getFoundFileName(),src);
		if (waitKey()==27) //ESC pressed
			break;
	}
}

void testImageOpenAndSave()
{
	_wchdir(projectPath);

	Mat src, dst;

	src = imread("Images/Lena_24bits.bmp", IMREAD_COLOR);	// Read the image

	if (!src.data)	// Check for invalid input
	{
		printf("Could not open or find the image\n");
		return;
	}

	// Get the image resolution
	Size src_size = Size(src.cols, src.rows);

	// Display window
	const char* WIN_SRC = "Src"; //window for the source image
	namedWindow(WIN_SRC, WINDOW_AUTOSIZE);
	moveWindow(WIN_SRC, 0, 0);

	const char* WIN_DST = "Dst"; //window for the destination (processed) image
	namedWindow(WIN_DST, WINDOW_AUTOSIZE);
	moveWindow(WIN_DST, src_size.width + 10, 0);

	cvtColor(src, dst, COLOR_BGR2GRAY); //converts the source image to a grayscale one

	imwrite("Images/Lena_24bits_gray.bmp", dst); //writes the destination to file

	imshow(WIN_SRC, src);
	imshow(WIN_DST, dst);

	waitKey(0);
}

void testNegativeImage()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		double t = (double)getTickCount(); // Get the current time [s]
		
		Mat src = imread(fname,IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = Mat(height,width,CV_8UC1);
		// Accessing individual pixels in an 8 bits/pixel image
		// Inefficient way -> slow
		for (int i=0; i<height; i++)
		{
			for (int j=0; j<width; j++)
			{
				uchar val = src.at<uchar>(i,j);
				uchar neg = 255 - val;
				dst.at<uchar>(i,j) = neg;
			}
		}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image",src);
		imshow("negative image",dst);
		waitKey();
	}
}

void testNegativeImageFast()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = src.clone();

		double t = (double)getTickCount(); // Get the current time [s]

		// The fastest approach of accessing the pixels -> using pointers
		uchar *lpSrc = src.data;
		uchar *lpDst = dst.data;
		int w = (int) src.step; // no dword alignment is done !!!
		for (int i = 0; i<height; i++)
			for (int j = 0; j < width; j++) {
				uchar val = lpSrc[i*w + j];
				lpDst[i*w + j] = 255 - val;
			}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image",src);
		imshow("negative image",dst);
		waitKey();
	}
}

void testColor2Gray()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src = imread(fname);

		int height = src.rows;
		int width = src.cols;

		Mat dst = Mat(height,width,CV_8UC1);

		// Accessing individual pixels in a RGB 24 bits/pixel image
		// Inefficient way -> slow
		for (int i=0; i<height; i++)
		{
			for (int j=0; j<width; j++)
			{
				Vec3b v3 = src.at<Vec3b>(i,j);
				uchar b = v3[0];
				uchar g = v3[1];
				uchar r = v3[2];
				dst.at<uchar>(i,j) = (r+g+b)/3;
			}
		}
		
		imshow("input image",src);
		imshow("gray image",dst);
		waitKey();
	}
}

void testBGR2HSV()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname);
		int height = src.rows;
		int width = src.cols;

		// HSV components
		Mat H = Mat(height, width, CV_8UC1);
		Mat S = Mat(height, width, CV_8UC1);
		Mat V = Mat(height, width, CV_8UC1);

		// Defining pointers to each matrix (8 bits/pixels) of the individual components H, S, V 
		uchar* lpH = H.data;
		uchar* lpS = S.data;
		uchar* lpV = V.data;

		Mat hsvImg;
		cvtColor(src, hsvImg, COLOR_BGR2HSV);

		// Defining the pointer to the HSV image matrix (24 bits/pixel)
		uchar* hsvDataPtr = hsvImg.data;

		for (int i = 0; i<height; i++)
		{
			for (int j = 0; j<width; j++)
			{
				int hi = i*width * 3 + j * 3;
				int gi = i*width + j;

				lpH[gi] = hsvDataPtr[hi] * 510 / 360;	// lpH = 0 .. 255
				lpS[gi] = hsvDataPtr[hi + 1];			// lpS = 0 .. 255
				lpV[gi] = hsvDataPtr[hi + 2];			// lpV = 0 .. 255
			}
		}

		imshow("input image", src);
		imshow("H", H);
		imshow("S", S);
		imshow("V", V);

		waitKey();
	}
}

void testResize()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		Mat dst1,dst2;
		//without interpolation
		resizeImg(src,dst1,320,false);
		//with interpolation
		resizeImg(src,dst2,320,true);
		imshow("input image",src);
		imshow("resized image (without interpolation)",dst1);
		imshow("resized image (with interpolation)",dst2);
		waitKey();
	}
}

void testCanny()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src,dst,gauss;
		src = imread(fname,IMREAD_GRAYSCALE);
		double k = 0.4;
		int pH = 50;
		int pL = (int) k*pH;
		GaussianBlur(src, gauss, Size(5, 5), 0.8, 0.8);
		Canny(gauss,dst,pL,pH,3);
		imshow("input image",src);
		imshow("canny",dst);
		waitKey();
	}
}

void testVideoSequence()
{
	_wchdir(projectPath);

	VideoCapture cap("Videos/rubic.avi"); // off-line video from file
	//VideoCapture cap(0);	// live video from web cam
	if (!cap.isOpened()) {
		printf("Cannot open video capture device.\n");
		waitKey(0);
		return;
	}
		
	Mat edges;
	Mat frame;
	char c;

	while (cap.read(frame))
	{
		Mat grayFrame;
		cvtColor(frame, grayFrame, COLOR_BGR2GRAY);
		Canny(grayFrame,edges,40,100,3);
		imshow("source", frame);
		imshow("gray", grayFrame);
		imshow("edges", edges);
		c = waitKey(100);  // waits 100ms and advances to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished\n"); 
			break;  //ESC pressed
		};
	}
}


void testSnap()
{
	_wchdir(projectPath);

	VideoCapture cap(0); // open the deafult camera (i.e. the built in web cam)
	if (!cap.isOpened()) // openenig the video device failed
	{
		printf("Cannot open video capture device.\n");
		return;
	}

	Mat frame;
	char numberStr[256];
	char fileName[256];
	
	// video resolution
	Size capS = Size((int)cap.get(CAP_PROP_FRAME_WIDTH),
		(int)cap.get(CAP_PROP_FRAME_HEIGHT));

	// Display window
	const char* WIN_SRC = "Src"; //window for the source frame
	namedWindow(WIN_SRC, WINDOW_AUTOSIZE);
	moveWindow(WIN_SRC, 0, 0);

	const char* WIN_DST = "Snapped"; //window for showing the snapped frame
	namedWindow(WIN_DST, WINDOW_AUTOSIZE);
	moveWindow(WIN_DST, capS.width + 10, 0);

	char c;
	int frameNum = -1;
	int frameCount = 0;

	for (;;)
	{
		cap >> frame; // get a new frame from camera
		if (frame.empty())
		{
			printf("End of the video file\n");
			break;
		}

		++frameNum;
		
		imshow(WIN_SRC, frame);

		c = waitKey(10);  // waits a key press to advance to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished");
			break;  //ESC pressed
		}
		if (c == 115){ //'s' pressed - snap the image to a file
			frameCount++;
			fileName[0] = NULL;
			sprintf(numberStr, "%d", frameCount);
			strcat(fileName, "Images/A");
			strcat(fileName, numberStr);
			strcat(fileName, ".bmp");
			bool bSuccess = imwrite(fileName, frame);
			if (!bSuccess) 
			{
				printf("Error writing the snapped image\n");
			}
			else
				imshow(WIN_DST, frame);
		}
	}

}

void MyCallBackFunc(int event, int x, int y, int flags, void* param)
{
	//More examples: http://opencvexamples.blogspot.com/2014/01/detect-mouse-clicks-and-moves-on-image.html
	Mat* src = (Mat*)param;
	if (event == EVENT_LBUTTONDOWN)
		{
			printf("Pos(x,y): %d,%d  Color(RGB): %d,%d,%d\n",
				x, y,
				(int)(*src).at<Vec3b>(y, x)[2],
				(int)(*src).at<Vec3b>(y, x)[1],
				(int)(*src).at<Vec3b>(y, x)[0]);
		}
}

void testMouseClick()
{
	Mat src;
	// Read image from file 
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		src = imread(fname);
		//Create a window
		namedWindow("My Window", 1);

		//set the callback function for any mouse event
		setMouseCallback("My Window", MyCallBackFunc, &src);

		//show the image
		imshow("My Window", src);

		// Wait until user press some key
		waitKey(0);
	}
}

/* Histogram display function - display a histogram using bars (simlilar to L3 / Image Processing)
Input:
name - destination (output) window name
hist - pointer to the vector containing the histogram values
hist_cols - no. of bins (elements) in the histogram = histogram image width
hist_height - height of the histogram image
Call example:
showHistogram ("MyHist", hist_dir, 255, 200);
*/
void showHistogram(const std::string& name, int* hist, const int  hist_cols, const int hist_height)
{
	Mat imgHist(hist_height, hist_cols, CV_8UC3, CV_RGB(255, 255, 255)); // constructs a white image

	//computes histogram maximum
	int max_hist = 0;
	for (int i = 0; i<hist_cols; i++)
	if (hist[i] > max_hist)
		max_hist = hist[i];
	double scale = 1.0;
	scale = (double)hist_height / max_hist;
	int baseline = hist_height - 1;

	for (int x = 0; x < hist_cols; x++) {
		Point p1 = Point(x, baseline);
		Point p2 = Point(x, baseline - cvRound(hist[x] * scale));
		line(imgHist, p1, p2, CV_RGB(255, 0, 255)); // histogram bins colored in magenta
	}

	imshow(name, imgHist);
}

Mat distanceTransform(Mat img) {
	Mat dtImg;
	img.copyTo(dtImg);

	int wHV = 2; 
	int wD = 3;   

	// first pass: top-down, left-right
	for (int i = 1; i < dtImg.rows - 1; i++) {
		for (int j = 1; j < dtImg.cols - 1; j++) {
			int minVal = dtImg.at<uchar>(i, j);

			// 5 pixels connectivity
			minVal = min(minVal, dtImg.at<uchar>(i - 1, j - 1) + wD);  // top-left
			minVal = min(minVal, dtImg.at<uchar>(i - 1, j) + wHV);   // top
			minVal = min(minVal, dtImg.at<uchar>(i - 1, j + 1) + wD);  // top-right
			minVal = min(minVal, dtImg.at<uchar>(i, j - 1) + wHV);   // left

			dtImg.at<uchar>(i, j) = minVal;
		}
	}

	// second pass: bottom-up, right-left
	for (int i = dtImg.rows - 2; i >= 1; i--) {
		for (int j = dtImg.cols - 2; j >= 1; j--) {
			int minVal = dtImg.at<uchar>(i, j);

			minVal = min(minVal, dtImg.at<uchar>(i, j + 1) + wHV);   // right
			minVal = min(minVal, dtImg.at<uchar>(i + 1, j - 1) + wD);  // bottom-left
			minVal = min(minVal, dtImg.at<uchar>(i + 1, j) + wHV);   // bottom
			minVal = min(minVal, dtImg.at<uchar>(i + 1, j + 1) + wD);  // bottom-right

			dtImg.at<uchar>(i, j) = minVal;
		}
	}

	return dtImg;
}

Point2d computeCenter(Mat img) {
	double sumX = 0, sumY = 0;
	int count = 0;
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			if (img.at<uchar>(i, j) == 0) { // object pixel
				sumX += j;
				sumY += i;
				count++;
			}
		}
	}
	if (count == 0) return Point2d(img.cols / 2.0, img.rows / 2.0);
	return Point2d(sumX / count, sumY / count);
}

Mat alignToCenter(Mat unknownImg, Mat templateImg)
{
	Point2d templateCenter = computeCenter(templateImg);
	Point2d unknownCenter = computeCenter(unknownImg);

	double dx = templateCenter.x - unknownCenter.x;
	double dy = templateCenter.y - unknownCenter.y;

	Mat aligned = Mat(unknownImg.rows, unknownImg.cols, CV_8UC1, Scalar(255));

	for (int y = 0; y < unknownImg.rows; y++) {
		for (int x = 0; x < unknownImg.cols; x++) {
			if (unknownImg.at<uchar>(y, x) == 0) {
				int newX = round(x + dx);
				int newY = round(y + dy);

				if (newX >= 0 && newX < unknownImg.cols &&
					newY >= 0 && newY < unknownImg.rows) {
					aligned.at<uchar>(newY, newX) = 0;
				}
			}
		}
	}

	return aligned;
}

void patternMatchingDT()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat templateImg = imread(fname, IMREAD_GRAYSCALE);
		imshow("Template", templateImg);

		openFileDlg(fname);
		Mat unknownImg = imread(fname, IMREAD_GRAYSCALE);
		imshow("Unknown object", unknownImg);

		Mat distTransform = distanceTransform(templateImg);
		imshow("Distance Transform", distTransform);

		//unknownImg = alignToCenter(unknownImg, templateImg);
		//imshow("Aligned unknown object", unknownImg);

		float score = 0;
		int count = 0;
		for (int i = 0; i < unknownImg.rows; i++) {
			for (int j = 0; j < unknownImg.cols; j++) {
				if (unknownImg.at<uchar>(i, j) == 0) {  // contour pixel
					score += distTransform.at<uchar>(i, j);
					count++;
				}
			}
		}
		if (count > 0) score = score / count;		
		printf("Pattern matching score: %f\n", score);

		waitKey();
	}
}

Mat correlationChart(int f1, int f2, Mat featureMat) {
	Mat chart(256, 256, CV_8UC1, Scalar(255));

	for (int i = 0; i < featureMat.rows; i++) {
		int val1 = featureMat.at<uchar>(i, f1);
		int val2 = featureMat.at<uchar>(i, f2);
		chart.at<uchar>(val1, val2) = 0;
	}

	return chart;
}

void statisticalDataAnalysis()
{
	char folder[256] = "Images/images_faces";
	char fname[256];

	int p = 400; // nr of images
	int d = 19; // image size
	int N = d * d; // nr of features (pixels)
	Mat featureMat(p, N, CV_8UC1);

	for (int i = 0; i < p; i++) {
		sprintf(fname, "%s/face%05d.bmp", folder, i + 1);
		Mat img = imread(fname,IMREAD_GRAYSCALE);
		for (int u = 0; u < d; u++) {
			for (int v = 0; v < d; v++) {
				int col = u * d + v;
				uchar val = img.at<uchar>(u, v);
				featureMat.at<uchar>(i, col) = val;
			}
		}
	}

	// compute mean value for each feature (pixel)
	vector<float> means;
	for (int i = 0; i < N; i++) {
		int mean = 0;
		for (int k = 0; k < p; k++) {
			mean += featureMat.at<uchar>(k, i);
		}
		mean = mean / p;
		means.push_back((float)mean);
	}

	ofstream covarianceFile;
	covarianceFile.open("covariance.csv");

	// compute the covariance matrix
	Mat covarianceMatrix(N, N, CV_32FC1);
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			float covariance = 0.0;
			for (int k = 0; k < p; k++) {
				covariance += (featureMat.at<uchar>(k, i) - means.at(i)) * (featureMat.at<uchar>(k, j) - means.at(j));
			}
			covariance = covariance / p;
			covarianceMatrix.at<float>(i, j) = covariance;
			covarianceFile << covariance << ",";
		}
		covarianceFile << endl;
	}
	covarianceFile.close();

	// compute the standard deviation for each feature
	vector<float> stdDevs;
	for (int i = 0; i < N; i++) {
		float stdDev = 0;
		for (int k = 0; k < p; k++) {
			stdDev += pow(featureMat.at<uchar>(k, i) - means.at(i), 2);
		}
		stdDev = sqrt(stdDev / p);
		stdDevs.push_back(stdDev);
	}

	ofstream correlationFile;
	correlationFile.open("correlation.csv");

	// compute the correlation matrix
	Mat correlationMatrix(N, N, CV_32FC1);
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			float covariance = covarianceMatrix.at<float>(i, j);
			float correlation = covariance / (stdDevs.at(i) * stdDevs.at(j));
			correlationMatrix.at<float>(i, j) = correlation;
			correlationFile << correlation << ",";
		}
		correlationFile << endl;
	}
	correlationFile.close();

	// correlation charts for some feature pairs
	int i = 5 * d + 4;
	int j = 5 * d + 14;
	printf("Correlation between features %d and %d: %f\n", i, j, correlationMatrix.at<float>(i, j));
	Mat chart1 = correlationChart(i, j, featureMat);
	imshow("Correlation chart a", chart1);

	i = 10 * d + 3;
	j = 9 * d + 15;
	printf("Correlation between features %d and %d: %f\n", i, j, correlationMatrix.at<float>(i, j));
	Mat chart2 = correlationChart(i, j, featureMat);
	imshow("Correlation chart b", chart2);

	i = 5 * d + 4;
	j = 18 * d;
	printf("Correlation between features %d and %d: %f\n", i, j, correlationMatrix.at<float>(i, j));
	Mat chart3 = correlationChart(i, j, featureMat);
	imshow("Correlation chart c", chart3);

	waitKey();
}

void principalComponentAnalysis()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		FILE* f = fopen(fname, "r");
		int n, d;
		fscanf(f, "%d %d", &n, &d);
		Mat FEAT(n, d, CV_64FC1);
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < d; j++) {
				double val;
				fscanf(f, "%lf", &val);
				FEAT.at<double>(i, j) = val;
			}
		}
		fclose(f);

		vector<double> means;
		for (int j = 0; j < d; j++) {
			double mean = 0.0;
			for (int i = 0; i < n; i++) {
				mean += FEAT.at<double>(i, j);
			}
			mean = mean / n;
			means.push_back(mean);
		}

		Mat X(n, d, CV_64FC1);
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < d; j++) {
				X.at<double>(i, j) = FEAT.at<double>(i, j) - means.at(j);
			}
		}

		Mat C = (1.0 / (n - 1)) * X.t() * X; 

		Mat Lambda, Q;
		eigen(C, Lambda, Q);
		Q = Q.t();

		for (int i = 0; i < d; i++) {
			printf("Lambda %d: %f\n", i, Lambda.at<double>(i));
		}

		int k = 2;
		Mat Qk(d, k, CV_64FC1);
		for (int i = 0; i < d; i++) {
			for (int j = 0; j < k; j++) {
				Qk.at<double>(i, j) = Q.at<double>(i, j);
			}
		}

		Mat Xpca;
		Xpca = X * Qk;

		Mat Xkapprox;
		Xkapprox = Xpca * Qk.t();

		float MAD = 0;
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < d; j++) {
				MAD += abs(Xkapprox.at<double>(i, j) - X.at<double>(i, j));
			}
		}
		MAD = MAD / (n * d);
		printf("Mean absolute difference: %f\n", MAD);

		vector<float> mins(k);
		vector<float> maxs(k);
		for (int i = 0; i < k; i++) {
			mins.at(i) = Xpca.at<double>(0, i);
			maxs.at(i) = Xpca.at<double>(0, i);
		}
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < k; j++) {
				if (Xpca.at<double>(i, j) < mins.at(j)) {
					mins.at(j) = Xpca.at<double>(i, j);
				}

				if (Xpca.at<double>(i, j) > maxs.at(j)) {
					maxs.at(j) = Xpca.at<double>(i, j);
				}
			}
		}

		Mat img((int)(maxs.at(0) - mins.at(0) + 1),(int)(maxs.at(1) - mins.at(1) + 1), CV_8UC1, Scalar(255));
		for (int i = 0; i < n; i++) {
			if (k == 2) {
				img.at<uchar>((int)(Xpca.at<double>(i, 0) - mins.at(0)), (int)(Xpca.at<double>(i, 1) - mins.at(1))) = 0;
			}
			else if (k == 3) {
				uchar val = (255 / (maxs.at(2) - mins.at(2))) * (Xpca.at<double>(i, 2) - mins.at(2));
				img.at<uchar>((int)(Xpca.at<double>(i, 0) - mins.at(0)), (int)(Xpca.at<double>(i, 1) - mins.at(1))) = 255 - val;
			}
		}

		imshow("After PCA", img);
		waitKey();
	}
}

float pointsDistance(Point3i p, Point3i center) {
	return sqrt(pow(p.x - center.x, 2) + pow(p.y - center.y, 2));
}


void kmeansClustering()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		Mat img = imread(fname, IMREAD_GRAYSCALE);
		int d = 2;

		vector<Point3i> points;
		for (int i = 0; i < img.rows; i++) {
			for (int j = 0; j < img.cols; j++) {
				if (img.at<uchar>(i, j) == 0) {
					points.push_back(Point3i(j, i, -1));
				}
			}
		}
		int n = points.size();

		srand(time(NULL));
		int c1, c2, c3;
		c1 = rand() % n;
		c2 = rand() % n;
		while (c1 == c2) {
			c2 = rand() % n;
		}
		c3 = rand() % n;
		while (c1 == c3 || c2 == c3) {
			c3 = rand() % n;
		}

		printf("c1 %d c2 %d c3 %d\n", c1, c2, c3);

		vector<int> randomIndex;
		randomIndex.push_back(c1);
		randomIndex.push_back(c2);
		randomIndex.push_back(c3);

		int K = 3;
		vector<Point3i> centers;

		for (int i = 0; i < randomIndex.size(); i++) {
			points.at(randomIndex.at(i)).z = i;
			centers.push_back(points.at(randomIndex.at(i)));
		}

		// Assignment
		boolean changed = false;
		int cycle = 0;
		do {
			printf("cycle: %d\n", cycle);
			for (int i = 0; i < n; i++) {
				float min;
				if (points.at(i).z == -1) {
					min = img.rows * img.rows + img.cols * img.cols;
				}
				else {
					min = pointsDistance(points.at(i), centers.at(points.at(i).z));
				}
				Point3i kMin;
				changed = false;
				for (int k = 0; k < K; k++) {
					if (points.at(i) != centers.at(k)) {
						float dist = pointsDistance(points.at(i), centers.at(k));
						if (dist < min) {
							changed = true;
							min = dist;
							kMin = centers.at(k);
						}
					}
				}
				if (changed) {
					points.at(i).z = kMin.z;
				}
			}

			// Update centers
			for (int k = 0; k < K; k++) {
				int x = 0;
				int y = 0;
				int nr = 0;
				for (int i = 0; i < n; i++) {
					if (points.at(i).z == k) {
						x += points.at(i).x;
						y += points.at(i).y;
						nr++;
					}
				}
				x = x / nr;
				y = y / nr;
				centers.at(k).x = x;
				centers.at(k).y = y;
			}
			cycle++;
		} while (changed);

		// Assign colors to clusters
		vector<Vec3b> colors;
		for (int i = 0; i < K; i++) {
			//uchar col1 = rand() % 255;
			colors.push_back(Vec3b(rand() % 255, rand() % 255, rand() % 255));
		}
		/*colors.push_back({ 255, 0, 0 });
		colors.push_back({ 0, 255, 0 });
		colors.push_back({ 0, 0, 255 });*/

		Mat clustersImg(img.rows, img.cols, CV_8UC3, Scalar(255, 255, 255));

		for (int i = 0; i < points.size(); i++) {
			Point3i p = points.at(i);
			clustersImg.at<Vec3b>(p.x, p.y) = colors.at(p.z);
		}

		imshow("Clusters", clustersImg);

		Mat voronoi(img.rows, img.cols, CV_8UC3, Scalar(255, 255, 255));
		for (int i = 0; i < voronoi.rows; i++) {
			for (int j = 0; j < voronoi.cols; j++) {
				float min = img.rows * img.rows + img.cols * img.cols;
				Point3i kMin;
				for (int k = 0; k < K; k++) {
					float dist = pointsDistance(Point3i(i, j, -1), centers.at(k));
					if (dist < min) {
						min = dist;
						kMin = centers.at(k);
					}
				}
				voronoi.at<Vec3b>(i, j) = colors.at(kMin.z);
			}
		}
		imshow("Voronoi", voronoi);

		waitKey(0);
	}
}	

int* calcHist(Mat img, int nr_bins)
{
	int histSize = 3 * nr_bins;
	int binSize = 256 / nr_bins;
	int* hist = (int*)calloc(histSize, sizeof(int));

	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			int R = img.at<Vec3b>(i, j)[2];
			int G = img.at<Vec3b>(i, j)[1];
			int B = img.at<Vec3b>(i, j)[0];
			hist[(int)(R / binSize)]++;
			hist[(int)(nr_bins + G / binSize)]++;
			hist[(int)(2 * nr_bins + B / binSize)]++;
		}
	}
	return hist;
}

int predict(Mat featureVec, Mat labels, Mat img, int histSize, int k, int nrClasses) 
{
	int* hist = calcHist(img, histSize / 3);

	// compute distance
	vector<float> distances;
	vector<int> indices;

	for (int i = 0; i < featureVec.rows; i++) {
		float dist = 0;
		for (int t = 0; t < histSize; t++) {
			dist += pow(featureVec.at<float>(i, t) - hist[t], 2);
		}
		dist = sqrt(dist);
		distances.push_back(dist);
		indices.push_back(i);
	}

	// sort indices based on distances
	for (int i = 0; i < distances.size() - 1; i++) {
		for (int j = i + 1; j < distances.size(); j++) {
			if (distances[j] < distances[i]) {
				swap(distances[i], distances[j]);
				swap(indices[i], indices[j]);
			}
		}
	}

	// determine k-nearest neighbors
	int* classesVotes = (int*)calloc(nrClasses, sizeof(int));
	for (int i = 0; i < k; i++) {
		classesVotes[labels.at<uchar>(indices[i], 0)]++;
	}

	int maxVotes = 0;
	int maxClass = 0;
	for (int i = 0; i < nrClasses; i++) {
		if (classesVotes[i] > maxVotes) {
			maxVotes = classesVotes[i];
			maxClass = i;
		}
	}
	free(classesVotes);

	return maxClass;
}

void knnClassifier()
{
	const int nrclasses = 6;
	char classes[nrclasses][10] = { "beach", "city", "desert", "forest", "landscape", "snow" };
	
	int nrInst = 672, m = 8, rowX = 0, k = 5;
	int histSize = 3 * m;
	int featureDim = histSize;

	Mat X(nrInst, featureDim, CV_32FC1);
	Mat y(nrInst, 1, CV_8UC1);

	char fname[50];
	for (int c = 0; c < nrclasses; c++) {
		int fileNr = 0;
		while (1) {
			sprintf(fname, "Images/images_KNN/train/%s/%06d.jpeg", classes[c], fileNr++);
			Mat img = imread(fname, IMREAD_COLOR);
			if (img.cols == 0) break;

			// compute histogram
			int* hist = calcHist(img, m);
			for (int d = 0; d < histSize; d++)
				X.at<float>(rowX, d) = hist[d];

			y.at<uchar>(rowX) = c;
			rowX++;
		}
	}

	Mat C(nrclasses, nrclasses, CV_32FC1, Scalar(0));

	int testInstances = 85;
	for (int c = 0; c < nrclasses; c++) {
		int fileNr = 0;
		while (1) {
			sprintf(fname, "Images/images_KNN/test/%s/%06d.jpeg", classes[c], fileNr++);
			Mat img = imread(fname, IMREAD_COLOR);
			if (img.cols == 0) break;
			int predictedClass = predict(X, y, img, histSize, k, nrclasses);
			C.at<float>(c, predictedClass) += 1.0f;
		}
	}

	imshow("Confusion Matrix", C);
	printf("Confusion matrix:\n");
	for (int i = 0; i < C.rows; i++) {
		for (int j = 0; j < C.cols; j++) {
			printf("%.1f ", C.at<float>(i, j));
		}
		printf("\n");
	}

	float accuracy = 0.0;
	for (int i = 0; i < nrclasses; i++) {
		accuracy += C.at<float>(i, i);
	}
	accuracy /= (float)testInstances;
	printf("Accuracy: %f\n", accuracy);

	waitKey();
}

void naiveBayes() {
	const int nrClasses = 10;
	char classes[nrClasses][2] = { "0", "1", "2", "3", "4", "5", "6", "7", "8", "9" };
	int nrinstancesPerClass[nrClasses];

	// Allocate feature matrix X and label vector y
	//int instancesPerClass = 100;
	int nrTraining = 60000;
	float cProb = 0.1;
	int nrFeatures = 28 * 28;
	uchar threshold = 128;
	Mat X(nrTraining, nrFeatures, CV_8UC1);
	Mat y(nrTraining, 1, CV_8UC1);
	Mat L255(nrClasses, nrFeatures, CV_32FC1, Scalar(0));
	Mat apriori(nrClasses, 1, CV_32FC1);

	int rowX = 0;
	char fname[50];
	for (int c = 0; c < nrClasses; c++) {
		int fileNr = 0;
		while (1) {
			sprintf(fname, "Images/images_Bayes/train/%s/%06d.png", classes[c], fileNr++);
			Mat img = imread(fname, IMREAD_GRAYSCALE);
			//if (img.cols == 0) break;
			if (img.cols == 0) {
				apriori.at<float>(c) = (float)fileNr / nrTraining;
				nrinstancesPerClass[c] = fileNr;
				break;
			}

			// Thresholding
			for (int i = 0; i < img.rows; i++) {
				for (int j = 0; j < img.cols; j++) {
					if (img.at<uchar>(i, j) < threshold) {
						X.at<uchar>(rowX, i * img.cols + j) = 0;
					}
					else {
						X.at<uchar>(rowX, i * img.cols + j) = 255;
						L255.at<float>(c, i * img.cols + j)++;
					}
				}
			}
			y.at<uchar>(rowX) = c;
			rowX++;
		}
	}

	// Performance of model - compute accuracy
	int totalNrTestInstances = 0;
	int correct = 0;
	for (int cl = 0; cl < nrClasses; cl++) {
		int fileNr = 0;
		while (1) {
			sprintf(fname, "Images/images_Bayes/test/%s/%06d.png", classes[cl], fileNr++);
			Mat img = imread(fname, IMREAD_GRAYSCALE);
			//if (img.cols == 0) break;
			if (img.cols == 0) {

				break;
			}
			totalNrTestInstances++;

			// Thresholding
			for (int i = 0; i < img.rows; i++) {
				for (int j = 0; j < img.cols; j++) {
					if (img.at<uchar>(i, j) < threshold) {
						img.at<uchar>(i, j) = 0;
					}
					else {
						img.at<uchar>(i, j) = 255;
					}
				}
			}

			float maxProb = INT_MIN;
			int finalClass = 0;
			for (int c = 0; c < nrClasses; c++) {
				float prob = 0;
				for (int i = 0; i < img.rows; i++) {
					for (int j = 0; j < img.cols; j++) {
						float p = L255.at<float>(c, i * img.cols + j) / nrinstancesPerClass[c];
						if (img.at<uchar>(i, j) == 0) {
							p = 1 - p;
						}
						if (p == 0) {
							p = pow(10, -5);
						}
						prob += log(p);
					}
				}
				prob += log(apriori.at<float>(c));
				if (prob > maxProb) {
					maxProb = prob;
					finalClass = c;
				}
			}
			if (finalClass == cl) {
				correct++;
			}
		}
	}
	float accuracy = (float)correct / totalNrTestInstances;
	printf("Accuracy is %f\n", accuracy);

	char imgname[MAX_PATH];
	while (openFileDlg(imgname)) {
		Mat img = imread(imgname, IMREAD_GRAYSCALE);

		// Thresholding
		for (int i = 0; i < img.rows; i++) {
			for (int j = 0; j < img.cols; j++) {
				if (img.at<uchar>(i, j) < threshold) {
					img.at<uchar>(i, j) = 0;
				}
				else {
					img.at<uchar>(i, j) = 255;
				}
			}
		}

		float maxProb = INT_MIN;
		int finalClass = 0;
		for (int c = 0; c < nrClasses; c++) {
			float prob = 0;
			for (int i = 0; i < img.rows; i++) {
				for (int j = 0; j < img.cols; j++) {
					float p = L255.at<float>(c, i * img.cols + j) / nrinstancesPerClass[c];
					if (img.at<uchar>(i, j) == 0) {
						p = 1 - p;
					}
					if (p == 0) {
						p = pow(10, -5);
					}
					prob += log(p);
				}
			}
			prob += log(apriori.at<float>(c));
			printf("Prob for class %d is %f\n", c, prob);
			if (prob > maxProb) {
				maxProb = prob;
				finalClass = c;
			}
		}

		printf("Predicted class is: %d\n", finalClass);
		imshow("Image", img);
		waitKey();
	}
}

void perceptronClassifier() {
	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		Mat img = imread(fname, IMREAD_COLOR);

		int nrFeatures = 3;
		Mat X(0, nrFeatures, CV_32FC1);
		Mat y(0, 1, CV_32FC1);

		for (int i = 0; i < img.rows; i++) {
			for (int j = 0; j < img.cols; j++) {
				Vec3b p = img.at<Vec3b>(i, j);
				float coords[3] = { 1, j, i };
				if (p == Vec3b(255, 0, 0)) {
					// blue point
					X.push_back(Mat(1, 3, CV_32FC1, coords));
					y.push_back(Mat(1, 1, CV_32FC1, Scalar(1)));
				}
				else if (p == Vec3b(0, 0, 255)) {
					// red point
					X.push_back(Mat(1, 3, CV_32FC1, coords));
					y.push_back(Mat(1, 1, CV_32FC1, Scalar(-1)));
				}
			}
		}

		Mat W(1, nrFeatures, CV_32FC1, { 0.1, 0.1, 0.1 });
		int maxIterations = pow(10, 5);
		float Elimit = pow(10, -5);
		int n = X.rows;
		float learningRate = pow(10, -4);
		Mat partialImg;
		img.copyTo(partialImg);

		// batch perceptron
		/*for (int iter = 0; iter < maxIterations; iter++) {
			float E = 0;
			Mat deriv(1, nrFeatures, CV_32FC1, { 0, 0, 0 });
			for (int i = 0; i < n; i++) {
				float z = 0;
				for (int j = 0; j < W.cols; j++) {
					z += W.at<float>(j) * X.at<float>(i, j);
				}
				if (z * y.at<float>(i) < 0) {
					for (int j = 0; j < X.cols; j++) {
						deriv.at<float>(j) -= y.at<float>(i) * X.at<float>(i, j);
					}
					E++;
				}
			}
			E = E / n;
			for (int j = 0; j < nrFeatures; j++) {
				deriv.at<float>(j) /= n;
			}
			if (E < Elimit)
				break;
			for (int j = 0; j < W.cols; j++) {
				W.at<float>(j) -= learningRate * deriv.at<float>(j);
			}
		}*/

		// online perceptron
		for (int iter = 0; iter < maxIterations; iter++) {
			float E = 0;

			for (int i = 0; i < n; i++) {
				float z = 0;

				for (int j = 0; j < W.cols; j++)
					z += W.at<float>(j) * X.at<float>(i, j);

				if (z * y.at<float>(i) < 0)
				{
					for (int j = 0; j < W.cols; j++)
						W.at<float>(j) += learningRate * y.at<float>(i) * X.at<float>(i, j);
					E++;
				}
			}

			E /= n;
			if (E < Elimit) break;
		}
		
		Mat resultImg;
		img.copyTo(resultImg);
		int y1 = (int)(-W.at<float>(0) / W.at<float>(2));
		int y2 = (int)(-(W.at<float>(0) + img.cols * W.at<float>(1)) / W.at<float>(2));
		line(resultImg, Point(0, y1), Point(img.cols, y2), Scalar(0, 255, 0), 2);
		imshow("Perceptron classification", resultImg);
		waitKey();
	}
}

struct weaklearner {
	int feature_i;
	int threshold;
	int class_label;
	float error;
	int classify(Mat X) {
		if (X.at<float>(feature_i) < threshold)
			return class_label;
		else
			return -class_label;
	}
};

int const MAXT = 100;

struct classifier {
	int T;
	float alphas[MAXT];
	weaklearner hs[MAXT];
	int classify(Mat X) {
		float x = 0;
		for (int t = 0; t < T; t++) {
			x += alphas[t] * hs[t].classify(X);
		}
		if (x < 0) {
			return -1;
		}
		else {
			return 1;
		}
	}
};

weaklearner findWeakLearner(Mat X, Mat y, Mat w, int imgSize) {
	weaklearner bestH = { 0, 0, 0, 0 };
	float bestErr = INT_MAX;
	for (int j = 0; j < X.cols; j++) {
		for (int t = 0; t < imgSize; t++) {
			for (int class_label = -1; class_label < 2; class_label += 2) {
				float err = 0;
				for (int i = 0; i < X.rows; i++) {
					float zi;
					if (X.at<float>(i, j) < t) {
						zi = class_label;
					}
					else {
						zi = -class_label;
					}
					if (zi * y.at<float>(i) < 0) {
						err += w.at<float>(i);
					}
				}
				if (err < bestErr) {
					bestErr = err;
					bestH = { j, t, class_label, err };
				}
			}
		}
	}
	return bestH;
}

void drawBoundary(Mat img, classifier clf) {
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			if (img.at<Vec3b>(i, j) == Vec3b(255, 255, 255)) {
				float v[] = { (float)j ,(float)i };
				Mat X(1, 2, CV_32FC1, v);
				int res = clf.classify(X);
				if (res < 0) {
					img.at<Vec3b>(i, j) = { 255, 153, 153 };
				}
				else {
					img.at<Vec3b>(i, j) = { 153, 255, 255 };
				}
			}
		}
	}
	imshow("Boundary", img);
	waitKey();
}

void adaBoost() {
	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		Mat img = imread(fname, IMREAD_COLOR);
		int imgSize = img.rows;

		int nrFeatures = 2;
		int nrExamples = 0;
		Mat X(0, nrFeatures, CV_32FC1);
		Mat y(0, 1, CV_32FC1);

		for (int i = 0; i < img.rows; i++) {
			for (int j = 0; j < img.cols; j++) {
				Vec3b p = img.at<Vec3b>(i, j);
				float coords[2] = { j, i };
				if (p == Vec3b(255, 0, 0)) {
					// blue point
					X.push_back(Mat(1, nrFeatures, CV_32FC1, coords));
					y.push_back(Mat(1, 1, CV_32FC1, Scalar(1)));
					nrExamples++;
				}
				else if (p == Vec3b(0, 0, 255)) {
					// red point
					X.push_back(Mat(1, nrFeatures, CV_32FC1, coords));
					y.push_back(Mat(1, 1, CV_32FC1, Scalar(-1)));
					nrExamples++;
				}
			}
		}

		float initialWeight = 1 / (float)nrExamples;
		Mat W(1, nrExamples, CV_32FC1, Scalar(initialWeight));
		classifier c;
		c.T = 10;
		for (int t = 0; t < c.T; t++) {
			weaklearner learner = findWeakLearner(X, y, W, imgSize);
			float alpha = 0.5 * log((1 - learner.error) / learner.error);
			c.alphas[t] = alpha;
			c.hs[t] = learner;
			float s = 0;
			for (int i = 0; i < nrExamples; i++) {
				W.at<float>(i) *= exp(-alpha * y.at<float>(i) * learner.classify(X.row(i)));
				s += W.at<float>(i);
			}
			for (int i = 0; i < nrExamples; i++) {
				W.at<float>(i) /= s;
			}
		}

		Mat newImg;
		img.copyTo(newImg);
		drawBoundary(newImg, c);
	}
}

bool initializeFaceDetector() {
	if (!faceDetector.isReady()) {
		printf("Loading face detection cascades...\n");
		bool loaded = faceDetector.loadCascades(
			"resources/haarcascade_frontalface_default.xml",
			"resources/haarcascade_eye.xml",
			"resources/haarcascade_mcs_nose.xml",
			"resources/haarcascade_mcs_mouth.xml"
		);

		if (!loaded) {
			printf("ERROR: Failed to load cascades!\n");
			printf("Make sure the cascade files are in the 'resources' folder.\n");
			printf("\nPress any key to continue...\n");
			waitKey(0);
			return false;
		}

		printf("Face detection cascades loaded successfully!\n");
		faceDetector.setParameters(1.1, 3, Size(30, 30));
	}
	return true;
}


int main() 
{
	cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_FATAL);
    projectPath = _wgetcwd(0, 0);

	int op;
	do
	{
		system("cls");
		destroyAllWindows();
		printf("Menu:\n");
		printf(" 1 - Open image\n");
		printf(" 2 - Open BMP images from folder\n");
		printf(" 3 - Image negative\n");
		printf(" 4 - Image negative (fast)\n");
		printf(" 5 - BGR->Gray\n");
		printf(" 6 - BGR->Gray (fast, save result to disk) \n");
		printf(" 7 - BGR->HSV\n");
		printf(" 8 - Resize image\n");
		printf(" 9 - Canny edge detection\n");
		printf(" 10 - Edges in a video sequence\n");
		printf(" 11 - Snap frame from live video\n");
		printf(" 12 - Mouse callback demo\n");
		printf(" 13 - Least Mean Squares\n");
		printf(" 14 - RANSAC line\n");
		printf(" 15 - Hough Transform for line detection\n");
		printf(" 16 - Distance Transform (DT). Pattern Matching using DT\n");
		printf(" 17 - Statistical Data Analysis\n");
		printf(" 18 - Principal Component Analysis (PCA)\n");
		printf(" 19 - K-means Clustering\n");
		printf(" 20 - K-Nearest Neighbor Classifier\n");
		printf(" 21 - Naive Bayes Classifier\n");
		printf(" 22 - Perceptron Classifier\n");
		printf(" 23 - Face Detection in Images\n");
		printf(" 24 - Face Detection in Video\n");
		printf(" 25 - Face Detection from Webcam\n");
		printf(" 26 - Face Detection with Details (face + eyes, nose, mouth)\n");
		printf(" 27 - AdaBoost\n"); 

		printf(" 0 - Exit\n\n");
		printf("Option: ");
		scanf("%d",&op);
		switch (op)
		{
			case 1:
				testOpenImage();
				break;
			case 2:
				testOpenImagesFld();
				break;
			case 3:
				testNegativeImage();
				break;
			case 4:
				testNegativeImageFast();
				break;
			case 5:
				testColor2Gray();
				break;
			case 6:
				testImageOpenAndSave();
				break;
			case 7:
				testBGR2HSV();
				break;
			case 8:
				testResize();
				break;
			case 9:
				testCanny();
				break;
			case 10:
				testVideoSequence();
				break;
			case 11:
				testSnap();
				break;
			case 12:
				testMouseClick();
				break;
			case 13: 
				leastMeanSquares();
				break;
			case 14:
				ransacLine();
				break;
			case 15:
				houghTransformLine();
				break;
			case 16:
				patternMatchingDT();
				break;
			case 17:
				statisticalDataAnalysis();
				break;
			case 18:
				principalComponentAnalysis();
				break;
			case 19:
				kmeansClustering();
				break;
			case 20:
				knnClassifier();
				break;
			case 21:
				naiveBayes();
				break;
			case 22:
				perceptronClassifier();
				break;
			case 23:
				if (initializeFaceDetector()) {
					faceDetector.detectFacesInImage();
				}
				break;

			case 24:
				if (initializeFaceDetector()) {
					faceDetector.detectFacesInVideo();
				}
				break;

			case 25:
				if (initializeFaceDetector()) {
					faceDetector.detectFacesFromWebcam();
				}
				break;
			case 26:
				if (initializeFaceDetector()) {
					faceDetector.detectFacesWithDetailsInImage();
				}
				break;
			case 27:
				adaBoost();
				break;

		}
	}
	while (op!=0);
	return 0;
}
