#ifndef FACE_DETECTOR_H
#define FACE_DETECTOR_H

#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>
#include <vector>
#include <string>

using namespace cv;
using namespace cv::ml;
using namespace std;

struct HOGParams {
    Size windowSize;     
    Size blockSize;         
    Size blockStride;       
    Size cellSize;          
    int numBins;           

    HOGParams() {
        windowSize = Size(64, 64);
        blockSize = Size(16, 16);
        blockStride = Size(8, 8);
        cellSize = Size(8, 8);
        numBins = 9;
    }
};

struct LBPParams {
    int radius;
    int neighbors;

    LBPParams() {
        radius = 1;
        neighbors = 8;
    }
};


class FaceDetector {
private:
    HOGParams hogParams;
    LBPParams lbpParams;
    Ptr<SVM> svmClassifier;
    bool isTrained;

    vector<float> computeHOGFeatures(const Mat& image);
    vector<float> computeLBPFeatures(const Mat& image);
    vector<float> computeCombinedFeatures(const Mat& image);

    Mat computeGradients(const Mat& image, Mat& magnitudes, Mat& angles);
    vector<float> computeCellHistogram(const Mat& magnitudes, const Mat& angles, int cellX, int cellY, int cellWidth, int cellHeight);
    vector<float> computeBlockDescriptor(const vector<vector<vector<float>>>& cellHistograms, int blockX, int blockY, int blocksPerRow, int blocksPerCol);
    vector<float> normalizeBlock(const vector<float>& blockHist);

    int computeLBPValue(const Mat& image, int x, int y);
    vector<float> computeLBPHistogram(const Mat& image);

    Mat preprocessImage(const Mat& image);
    Mat histogramEqualization(const Mat& input);
    Mat gaussianBlur(const Mat& input, int kernelSize, double sigma);

    vector<Rect> generateNegativeSamples(const Mat& image, const vector<Rect>& faces, int numSamples);

    vector<Rect> nonMaximumSuppression(vector<Rect>& boxes, vector<float>& scores, float overlapThresh);
    float computeIOU(const Rect& box1, const Rect& box2);

public:
    FaceDetector();
    FaceDetector(const HOGParams& hog, const LBPParams& lbp);

    void trainClassifier(const string& positivePath, const string& negativePath, int numPositive, int numNegative);
    void trainFromAnnotatedImages(const vector<Mat>& images, const vector<vector<Rect>>& faceAnnotations, int negativeSamplesPerImage = 10);
    bool saveModel(const string& modelPath);
    bool loadModel(const string& modelPath);

    vector<Rect> detectFaces(Mat& image, float threshold = 0.0, bool useMultiScale = true, bool preprocess = true);
    vector<pair<Rect, float>> detectFacesWithConfidence(Mat& image, float threshold = 0.0, bool useMultiScale = true);

    void drawDetections(Mat& image, const vector<Rect>& faces, const Scalar& color = Scalar(0, 255, 0));
    void displayTrainingInfo();
    void evaluateModel(const vector<Mat>& testImages, const vector<vector<Rect>>& groundTruth);

    void setHOGParams(const HOGParams& params) { hogParams = params; }
    void setLBPParams(const LBPParams& params) { lbpParams = params; }
    HOGParams getHOGParams() const { return hogParams; }
    LBPParams getLBPParams() const { return lbpParams; }
    bool isModelTrained() const { return isTrained; }

    int getFeatureDimension() const;
    int getHOGFeatureDimension() const;
    int getLBPFeatureDimension() const;

    void commandTrain();
    void commandLoadModel();
    void commandTestOnImage();
    void commandEvaluate();


private:
    vector<Rect> parseAnnotationFile(const string& annotPath, int imageWidth, int imageHeight);
    string getAnnotationFilename(const string& imageFilename);
    bool loadDatasetFromFolders(const string& imagesFolder, const string& annotationsFolder, vector<Mat>& images, vector<vector<Rect>>& annotations, const string& extension);
};


namespace Utils {
    vector<Mat> loadImagesFromFolder(const string& folderPath, int maxImages = -1);
    vector<Rect> generateSlidingWindows(const Size& imageSize, const Size& windowSize, const Size& stride);
    Mat resizeWithAspectRatio(const Mat& image, int targetSize);
    vector<Mat> augmentImage(const Mat& image, bool flipHorizontal = true, bool adjustBrightness = true, int numVariations = 3);

    struct DetectionMetrics {
        int truePositives;
        int falsePositives;
        int falseNegatives;
        float precision;
        float recall;
        float f1Score;

        void calculate() {
            precision = (truePositives + falsePositives > 0) ?
                (float)truePositives / (truePositives + falsePositives) : 0;
            recall = (truePositives + falseNegatives > 0) ?
                (float)truePositives / (truePositives + falseNegatives) : 0;
            f1Score = (precision + recall > 0) ?
                2 * precision * recall / (precision + recall) : 0;
        }

        void print() const {
            printf("\n=== Detection Performance ===\n");
            printf("True Positives: %d\n", truePositives);
            printf("False Positives: %d\n", falsePositives);
            printf("False Negatives: %d\n", falseNegatives);
            printf("Precision: %.3f\n", precision);
            printf("Recall: %.3f\n", recall);
            printf("F1-Score: %.3f\n", f1Score);
            printf("===========================\n\n");
        }
    };

    DetectionMetrics computeMetrics(const vector<Rect>& detections,
        const vector<Rect>& groundTruth,
        float iouThreshold = 0.5);
}

#endif 
