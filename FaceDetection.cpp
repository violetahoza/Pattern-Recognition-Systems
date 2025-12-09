#include "stdafx.h"
#include "common.h"
#include "FaceDetection.h"
#include <iostream>
#include <algorithm>
#include <cmath>


FaceDetector::FaceDetector() {
    cascadesLoaded = false;
    scaleFactor = 1.1;
    minNeighbors = 3;
    minFaceSize = Size(30, 30);
    maxFaceSize = Size();
}

bool FaceDetector::loadCascades(const string& faceCascadePath,
    const string& eyeCascadePath,
    const string& noseCascadePath,
    const string& mouthCascadePath) {

    if (!faceCascade.load(faceCascadePath)) {
        printf("ERROR: Could not load face cascade from: %s\n", faceCascadePath.c_str());
        cascadesLoaded = false;
        return false;
    }
    printf("Face cascade loaded successfully\n");

    if (!eyeCascadePath.empty()) {
        if (eyeCascade.load(eyeCascadePath)) {
            printf("Eye cascade loaded successfully\n");
        }
        else {
            printf("Warning: Could not load eye cascade from: %s\n", eyeCascadePath.c_str());
        }
    }

    if (!noseCascadePath.empty()) {
        if (noseCascade.load(noseCascadePath)) {
            printf("Nose cascade loaded successfully\n");
        }
        else {
            printf("Warning: Could not load nose cascade\n");
        }
    }

    if (!mouthCascadePath.empty()) {
        if (mouthCascade.load(mouthCascadePath)) {
            printf("Mouth cascade loaded successfully\n");
        }
        else {
            printf("Warning: Could not load mouth cascade\n");
        }
    }

    cascadesLoaded = true;
    return true;
}

void FaceDetector::setParameters(double scale, int neighbors, Size minSize, Size maxSize) {
    scaleFactor = scale;
    minNeighbors = neighbors;
    minFaceSize = minSize;
    maxFaceSize = maxSize;
}


void FaceDetector::computeIntegralImage(const Mat& input, Mat& integral) {
    integral = Mat(input.rows + 1, input.cols + 1, CV_32S, Scalar(0));

    // compute integral image: each pixel contains sum of all pixels above and to the left
    for (int i = 1; i <= input.rows; i++) {
        int rowSum = 0;
        for (int j = 1; j <= input.cols; j++) {
            rowSum += input.at<uchar>(i - 1, j - 1);
            integral.at<int>(i, j) = rowSum + integral.at<int>(i - 1, j);
        }
    }
}

int FaceDetector::computeRectangleSum(const Mat& integral, Rect rect) {
    int x = rect.x;
    int y = rect.y;
    int w = rect.width;
    int h = rect.height;

    // validate bounds
    if (x < 0 || y < 0 || x + w > integral.cols - 1 || y + h > integral.rows - 1) {
        return 0;
    }

    int A = integral.at<int>(y, x);
    int B = integral.at<int>(y, x + w);
    int C = integral.at<int>(y + h, x);
    int D = integral.at<int>(y + h, x + w);

    return D - B - C + A;
}

Mat FaceDetector::histogramEqualization(const Mat& input) {
    Mat output = input.clone();

    int hist[256] = { 0 };
    for (int i = 0; i < input.rows; i++) {
        for (int j = 0; j < input.cols; j++) {
            hist[input.at<uchar>(i, j)]++;
        }
    }

    int cdf[256] = { 0 };
    cdf[0] = hist[0];
    for (int i = 1; i < 256; i++) {
        cdf[i] = cdf[i - 1] + hist[i];
    }

    int cdfMin = 0;
    for (int i = 0; i < 256; i++) {
        if (cdf[i] > 0) {
            cdfMin = cdf[i];
            break;
        }
    }

    int totalPixels = input.rows * input.cols;
    uchar lut[256];
    for (int i = 0; i < 256; i++) {
        lut[i] = (uchar)(((cdf[i] - cdfMin) * 255.0) / (totalPixels - cdfMin) + 0.5);
    }

    for (int i = 0; i < input.rows; i++) {
        for (int j = 0; j < input.cols; j++) {
            output.at<uchar>(i, j) = lut[input.at<uchar>(i, j)];
        }
    }

    return output;
}

Mat FaceDetector::gaussianBlur(const Mat& input, int kernelSize, double sigma) {
    Mat output = input.clone();
    int k = kernelSize / 2;

    double kernel[9][9];
    double sum = 0.0;

    for (int i = -k; i <= k; i++) {
        for (int j = -k; j <= k; j++) {
            double value = exp(-(i * i + j * j) / (2.0 * sigma * sigma));
            kernel[i + k][j + k] = value;
            sum += value;
        }
    }

    // normalize kernel
    for (int i = 0; i < kernelSize; i++) {
        for (int j = 0; j < kernelSize; j++) {
            kernel[i][j] /= sum;
        }
    }

    // apply convolution
    for (int i = k; i < input.rows - k; i++) {
        for (int j = k; j < input.cols - k; j++) {
            double pixelValue = 0.0;

            for (int ki = -k; ki <= k; ki++) {
                for (int kj = -k; kj <= k; kj++) {
                    pixelValue += input.at<uchar>(i + ki, j + kj) * kernel[ki + k][kj + k];
                }
            }

            output.at<uchar>(i, j) = (uchar)(pixelValue + 0.5);
        }
    }

    return output;
}

// compute Intersection over Union for non-maximum suppression
float FaceDetector::computeIOU(const Rect& box1, const Rect& box2) {
    int x1 = max(box1.x, box2.x);
    int y1 = max(box1.y, box2.y);
    int x2 = min(box1.x + box1.width, box2.x + box2.width);
    int y2 = min(box1.y + box1.height, box2.y + box2.height);

    int intersectionArea = max(0, x2 - x1) * max(0, y2 - y1);
    int box1Area = box1.width * box1.height;
    int box2Area = box2.width * box2.height;
    int unionArea = box1Area + box2Area - intersectionArea;

    return (float)intersectionArea / unionArea;
}

// non-maximum suppression to remove overlapping detections
vector<Rect> FaceDetector::nonMaximumSuppression(vector<Rect>& boxes, vector<int>& scores, float overlapThresh) {
    if (boxes.empty()) {
        return vector<Rect>();
    }

    vector<Rect> result;
    vector<bool> suppressed(boxes.size(), false);

    // sort by scores (descending)
    vector<int> indices(boxes.size());
    for (size_t i = 0; i < indices.size(); i++) {
        indices[i] = i;
    }

    sort(indices.begin(), indices.end(), [&scores](int i1, int i2) {
        return scores[i1] > scores[i2];
        });

    for (size_t i = 0; i < indices.size(); i++) {
        int idx = indices[i];
        if (suppressed[idx]) continue;

        result.push_back(boxes[idx]);

        for (size_t j = i + 1; j < indices.size(); j++) {
            int idx2 = indices[j];
            if (suppressed[idx2]) continue;

            float iou = computeIOU(boxes[idx], boxes[idx2]);
            if (iou > overlapThresh) {
                suppressed[idx2] = true;
            }
        }
    }

    return result;
}


Mat FaceDetector::preprocessImage(const Mat& image) {
    Mat processed;

    // convert to grayscale if needed
    if (image.channels() == 3) {
        cvtColor(image, processed, COLOR_BGR2GRAY);
    }
    else {
        processed = image.clone();
    }

    // apply histogram equalization for better contrast
    processed = histogramEqualization(processed);

    // apply Gaussian blur to reduce noise
    processed = gaussianBlur(processed, 3, 0.8);

    return processed;
}

vector<Rect> FaceDetector::detectFaces(Mat& image, bool preprocess) {
    if (!cascadesLoaded) {
        printf("ERROR: Cascades not loaded!\n");
        return vector<Rect>();
    }

    Mat grayImage;
    if (preprocess) {
        grayImage = preprocessImage(image);
    }
    else {
        if (image.channels() == 3) {
            cvtColor(image, grayImage, COLOR_BGR2GRAY);
        }
        else {
            grayImage = image.clone();
        }
    }

    computeIntegralImage(grayImage, integralImage);

    vector<Rect> faces;
    faceCascade.detectMultiScale(grayImage, faces, scaleFactor, minNeighbors, 0, minFaceSize, maxFaceSize);

    return faces;
}

void FaceDetector::detectEyesInFace(const Mat& faceROI, vector<Rect>& eyes) {
    if (eyeCascade.empty()) {
        return;
    }

    // eyes are typically in upper half of face
    int eyeRegionHeight = (int)(faceROI.rows * 0.6);
    Mat eyeRegion = faceROI(Rect(0, 0, faceROI.cols, eyeRegionHeight));

    eyeCascade.detectMultiScale(eyeRegion, eyes, 1.1, 3, 0, Size(20, 20));
}

void FaceDetector::detectFacialFeatures(const Mat& faceROI, Rect& nose, Rect& mouth) {
    // detect nose (middle region)
    if (!noseCascade.empty()) {
        vector<Rect> noses;
        int noseY = (int)(faceROI.rows * 0.3);
        int noseHeight = (int)(faceROI.rows * 0.4);
        Mat noseRegion = faceROI(Rect(0, noseY, faceROI.cols, noseHeight));

        noseCascade.detectMultiScale(noseRegion, noses, 1.1, 3, 0, Size(15, 15));
        if (!noses.empty()) {
            nose = noses[0];
            nose.y += noseY; // adjust to face coordinates
        }
    }

    // detect mouth (lower region)
    if (!mouthCascade.empty()) {
        vector<Rect> mouths;
        int mouthY = (int)(faceROI.rows * 0.5);
        int mouthHeight = (int)(faceROI.rows * 0.5);
        Mat mouthRegion = faceROI(Rect(0, mouthY, faceROI.cols, mouthHeight));

        mouthCascade.detectMultiScale(mouthRegion, mouths, 1.1, 5, 0, Size(20, 10));
        if (!mouths.empty()) {
            mouth = mouths[0];
            mouth.y += mouthY; // adjust to face coordinates
        }
    }
}

void FaceDetector::detectFacesAndFeatures(Mat& image, bool drawResults) {
    double startTime = (double)getTickCount();

    vector<Rect> faces = detectFaces(image, true);

    vector<vector<Rect>> allEyes;
    vector<Rect> allNoses;
    vector<Rect> allMouths;

    // for each face, detect features
    Mat grayImage;
    if (image.channels() == 3) {
        cvtColor(image, grayImage, COLOR_BGR2GRAY);
    }
    else {
        grayImage = image.clone();
    }

    for (size_t i = 0; i < faces.size(); i++) {
        Rect face = faces[i];
        Mat faceROI = grayImage(face);

        vector<Rect> eyes;
        detectEyesInFace(faceROI, eyes);

        // adjust eye coordinates to full image
        for (size_t j = 0; j < eyes.size(); j++) {
            eyes[j].x += face.x;
            eyes[j].y += face.y;
        }
        allEyes.push_back(eyes);

        Rect nose, mouth;
        detectFacialFeatures(faceROI, nose, mouth);

        if (nose.width > 0) {
            nose.x += face.x;
            nose.y += face.y;
            allNoses.push_back(nose);
        }
        else {
            allNoses.push_back(Rect());
        }

        if (mouth.width > 0) {
            mouth.x += face.x;
            mouth.y += face.y;
            allMouths.push_back(mouth);
        }
        else {
            allMouths.push_back(Rect());
        }
    }

    double endTime = (double)getTickCount();
    double processingTime = (endTime - startTime) / getTickFrequency() * 1000.0;

    if (drawResults) {
        drawDetections(image, faces, allEyes, allNoses, allMouths);
    }

    displayDetectionStats(faces.size(), processingTime);
}

void FaceDetector::drawFacesOnly(Mat& image, const vector<Rect>& faces) {
    for (size_t i = 0; i < faces.size(); i++) {
        rectangle(image, faces[i], Scalar(0, 255, 0), 2);
        string faceLabel = "Face " + to_string(i + 1);
        putText(image, faceLabel, Point(faces[i].x, faces[i].y - 10), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 2);
    }
}

void FaceDetector::drawDetections(Mat& image, const vector<Rect>& faces,
    const vector<vector<Rect>>& eyes,
    const vector<Rect>& noses,
    const vector<Rect>& mouths) {

    for (size_t i = 0; i < faces.size(); i++) {
        rectangle(image, faces[i], Scalar(0, 255, 0), 2);

        string faceLabel = "Face " + to_string(i + 1);
        putText(image, faceLabel, Point(faces[i].x, faces[i].y - 10), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 2);

        if (i < eyes.size()) {
            for (size_t j = 0; j < eyes[i].size(); j++) {
                rectangle(image, eyes[i][j], Scalar(255, 0, 0), 2);
                circle(image, Point(eyes[i][j].x + eyes[i][j].width / 2, eyes[i][j].y + eyes[i][j].height / 2), 3, Scalar(255, 0, 0), -1);
            }
        }

        if (i < noses.size() && noses[i].width > 0) {
            rectangle(image, noses[i], Scalar(0, 255, 255), 2);
        }

        if (i < mouths.size() && mouths[i].width > 0) {
            rectangle(image, mouths[i], Scalar(0, 0, 255), 2);
        }
    }
}

void FaceDetector::displayDetectionStats(int numFaces, double processingTime) {
    printf("Detected %d face(s) in %.2f ms\n", numFaces, processingTime);
}

void FaceDetector::detectFacesInImage() {
    char fname[MAX_PATH];
    while (openFileDlg(fname)) {
        Mat image = imread(fname);
        if (image.empty()) {
            printf("ERROR: Could not load image: %s\n", fname);
            continue;
        }

        double startTime = (double)getTickCount();

        Mat displayImage = image.clone();
        vector<Rect> faces = detectFaces(displayImage, true);

        drawFacesOnly(displayImage, faces);  

        double endTime = (double)getTickCount();
        double processingTime = (endTime - startTime) / getTickFrequency() * 1000.0;

        displayDetectionStats(faces.size(), processingTime);

        imshow("Face Detection - Original", image);
        imshow("Face Detection - Results", displayImage);

        printf("Press any key to continue, ESC to exit...\n");
        int key = waitKey(0);
        if (key == 27) break;
    }
}

void FaceDetector::detectFacesInVideo() {
    char fname[MAX_PATH];
    if (!openFileDlg(fname)) {
        return;
    }

    VideoCapture cap(fname);
    if (!cap.isOpened()) {
        printf("ERROR: Could not open video file: %s\n", fname);
        return;
    }

    Mat frame;
    int frameCount = 0;

    printf("Processing video... Press ESC to exit\n");

    while (cap.read(frame)) {
        if (frame.empty()) break;

        frameCount++;
        double startTime = (double)getTickCount();

        Mat displayFrame = frame.clone();
        vector<Rect> faces = detectFaces(displayFrame, true);

        drawFacesOnly(displayFrame, faces);  

        double endTime = (double)getTickCount();
        double processingTime = (endTime - startTime) / getTickFrequency() * 1000.0;

        string frameLabel = "Frame: " + to_string(frameCount);
        putText(displayFrame, frameLabel, Point(10, 30),
            FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 255, 255), 2);

        string fpsLabel = "FPS: " + to_string((int)(1000.0 / processingTime));
        putText(displayFrame, fpsLabel, Point(10, 60),
            FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 255, 255), 2);

        imshow("Face Detection - Video", displayFrame);

        int key = waitKey(30);
        if (key == 27) break;
    }

    cap.release();
    destroyWindow("Face Detection - Video");
}

void FaceDetector::detectFacesFromWebcam() {
    VideoCapture cap(0);
    if (!cap.isOpened()) {
        printf("ERROR: Could not open webcam!\n");
        return;
    }

    printf("Webcam opened successfully. Press ESC to exit\n");

    Mat frame;
    while (true) {
        cap >> frame;
        if (frame.empty()) {
            printf("ERROR: Could not capture frame\n");
            break;
        }

        double startTime = (double)getTickCount();

        Mat displayFrame = frame.clone();
        vector<Rect> faces = detectFaces(displayFrame, true);

        drawFacesOnly(displayFrame, faces);  

        double endTime = (double)getTickCount();
        double processingTime = (endTime - startTime) / getTickFrequency() * 1000.0;

        putText(displayFrame, "Press ESC to exit", Point(10, 30),
            FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 255, 255), 2);

        string fpsLabel = "FPS: " + to_string((int)(1000.0 / processingTime));
        putText(displayFrame, fpsLabel, Point(10, 60),
            FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 255, 255), 2);

        imshow("Face Detection - Webcam", displayFrame);

        int key = waitKey(10);
        if (key == 27) break;
    }

    cap.release();
    destroyWindow("Face Detection - Webcam");
}

void FaceDetector::detectFacesWithDetailsInImage() {
    char fname[MAX_PATH];
    while (openFileDlg(fname)) {
        Mat image = imread(fname);
        if (image.empty()) {
            printf("ERROR: Could not load image: %s\n", fname);
            continue;
        }

        Mat displayImage = image.clone();
        detectFacesAndFeatures(displayImage, true);  

        imshow("Face Detection with Details - Original", image);
        imshow("Face Detection with Details - Results", displayImage);

        printf("Green = Face | Blue = Eyes | Yellow = Nose | Red = Mouth\n");
        printf("Press any key to continue, ESC to exit...\n");
        int key = waitKey(0);
        if (key == 27) break;
    }
}


namespace FaceDetectionUtils {

    Mat convertToGrayscale(const Mat& image) {
        Mat gray;
        if (image.channels() == 3) {
            cvtColor(image, gray, COLOR_BGR2GRAY);
        }
        else {
            gray = image.clone();
        }
        return gray;
    }

    Mat enhanceContrast(const Mat& image) {
        Mat enhanced;
        equalizeHist(image, enhanced);
        return enhanced;
    }

    bool validateFaceRegion(const Rect& face, const Size& imageSize) {
        // check if face is within image bounds
        if (face.x < 0 || face.y < 0 ||
            face.x + face.width > imageSize.width ||
            face.y + face.height > imageSize.height) {
            return false;
        }

        // check aspect ratio (faces are roughly square)
        float aspectRatio = (float)face.width / face.height;
        if (aspectRatio < 0.5 || aspectRatio > 2.0) {
            return false;
        }

        return true;
    }

    double computeFaceConfidence(const Mat& faceROI) {
        // confidence based on variance
        Scalar mean, stddev;
        meanStdDev(faceROI, mean, stddev);
        return stddev[0];
    }

    void printDetectionInfo(const vector<Rect>& faces, double time) {
        printf("\n=== Detection Results ===\n");
        printf("Number of faces: %zu\n", faces.size());
        printf("Processing time: %.2f ms\n", time);

        for (size_t i = 0; i < faces.size(); i++) {
            printf("Face %zu: Position(%d, %d), Size(%dx%d)\n",
                i + 1, faces[i].x, faces[i].y, faces[i].width, faces[i].height);
        }
        printf("========================\n\n");
    }
}