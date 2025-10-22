// OpenCVApplication.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "common.h"
#include <math.h>
#include <opencv2/core/utils/logger.hpp>

using namespace cv;
using namespace std;

wchar_t* projectPath;

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
		}
	}
	while (op!=0);
	return 0;
}
