#ifndef FACE_DETECTION_H
#define FACE_DETECTION_H

#include <opencv2/opencv.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <vector>
#include <string>

using namespace cv;
using namespace std;

class FaceDetector {

private:
    CascadeClassifier faceCascade;
    CascadeClassifier eyeCascade;
    CascadeClassifier noseCascade;
    CascadeClassifier mouthCascade;
    bool cascadesLoaded;

    // detection parameters
    double scaleFactor;
    int minNeighbors;
    Size minFaceSize;
    Size maxFaceSize;

    Mat integralImage;

    // helper methods
    void computeIntegralImage(const Mat& input, Mat& integral);
    int computeRectangleSum(const Mat& integral, Rect rect);
    vector<Rect> nonMaximumSuppression(vector<Rect>& boxes, vector<int>& scores, float overlapThresh = 0.3f);
    float computeIOU(const Rect& box1, const Rect& box2);
    Mat histogramEqualization(const Mat& input);
    Mat gaussianBlur(const Mat& input, int kernelSize, double sigma);

public:
    FaceDetector();
    bool loadCascades(const string& faceCascadePath, const string& eyeCascadePath = "", const string& noseCascadePath = "", const string& mouthCascadePath = "");
    void setParameters(double scale = 1.1, int neighbors = 3, Size minSize = Size(30, 30), Size maxSize = Size());
    vector<Rect> detectFaces(Mat& image, bool preprocess = true);
    void detectFacesAndFeatures(Mat& image, bool drawResults = true);

    void detectFacesInImage();                     
    void detectFacesInVideo();                      
    void detectFacesFromWebcam();                
    void detectFacesWithDetailsInImage();           

    Mat preprocessImage(const Mat& image);
    void detectEyesInFace(const Mat& faceROI, vector<Rect>& eyes);
    void detectFacialFeatures(const Mat& faceROI, Rect& nose, Rect& mouth);

    void drawDetections(Mat& image, const vector<Rect>& faces, const vector<vector<Rect>>& eyes = vector<vector<Rect>>(), const vector<Rect>& noses = vector<Rect>(), const vector<Rect>& mouths = vector<Rect>());
    void drawFacesOnly(Mat& image, const vector<Rect>& faces); 

    bool isReady() const { return cascadesLoaded; }
    void displayDetectionStats(int numFaces, double processingTime);
};
namespace FaceDetectionUtils {
    // image preprocessing
    Mat convertToGrayscale(const Mat& image);
    Mat enhanceContrast(const Mat& image);

    // feature analysis
    bool validateFaceRegion(const Rect& face, const Size& imageSize);
    double computeFaceConfidence(const Mat& faceROI);

    // performance and statistics
    void printDetectionInfo(const vector<Rect>& faces, double time);
}
#endif