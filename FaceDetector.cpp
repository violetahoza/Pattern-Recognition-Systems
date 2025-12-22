#include "stdafx.h"
#include "common.h"
#include "FaceDetector.h"
#include <iostream>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <cstring> 

const char* TRAIN_IMAGES_PATH = "C:\\Users\\hozas\\Desktop\\facultate\\an4\\Sem1\\PRS\\OpenCVApplication-VS2022_OCV490_basic\\FaceDataset\\train\\images";
const char* TRAIN_ANNOTATIONS_PATH = "C:\\Users\\hozas\\Desktop\\facultate\\an4\\Sem1\\PRS\\OpenCVApplication-VS2022_OCV490_basic\\FaceDataset\\train\\annotations";
const char* TEST_IMAGES_PATH = "C:\\Users\\hozas\\Desktop\\facultate\\an4\\Sem1\\PRS\\OpenCVApplication-VS2022_OCV490_basic\\FaceDataset\\test\\images";
const char* TEST_ANNOTATIONS_PATH = "C:\\Users\\hozas\\Desktop\\facultate\\an4\\Sem1\\PRS\\OpenCVApplication-VS2022_OCV490_basic\\FaceDataset\\test\\annotations";

FaceDetector::FaceDetector() {
    isTrained = false;
    svmClassifier = SVM::create();
    svmClassifier->setType(SVM::C_SVC);
    svmClassifier->setKernel(SVM::LINEAR);
    svmClassifier->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 1000, 1e-6));
}

FaceDetector::FaceDetector(const HOGParams& hog, const LBPParams& lbp) : hogParams(hog), lbpParams(lbp) {
    isTrained = false;
    svmClassifier = SVM::create();
    svmClassifier->setType(SVM::C_SVC);
    svmClassifier->setKernel(SVM::LINEAR);
    svmClassifier->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 1000, 1e-6));
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
    CV_Assert(kernelSize <= 9 && kernelSize % 2 == 1);

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

Mat FaceDetector::computeGradients(const Mat& image, Mat& magnitudes, Mat& angles) {
    magnitudes = Mat(image.rows, image.cols, CV_32F);
    angles = Mat(image.rows, image.cols, CV_32F);

    // compute gradients using Sobel-like operators
    for (int i = 1; i < image.rows - 1; i++) {
        for (int j = 1; j < image.cols - 1; j++) {
            float gx = (float)image.at<uchar>(i, j + 1) - (float)image.at<uchar>(i, j - 1);
            float gy = (float)image.at<uchar>(i + 1, j) - (float)image.at<uchar>(i - 1, j);

            magnitudes.at<float>(i, j) = sqrt(gx * gx + gy * gy);

            float angle = atan2(gy, gx) * 180.0 / CV_PI;
            if (angle < 0) angle += 180;
            angles.at<float>(i, j) = angle;
        }
    }

    return magnitudes;
}

vector<float> FaceDetector::computeCellHistogram(const Mat& magnitudes, const Mat& angles, int cellX, int cellY, int cellWidth, int cellHeight) {
    vector<float> histogram(hogParams.numBins, 0.0f);
    float binWidth = 180.0f / hogParams.numBins;

    for (int i = cellY; i < cellY + cellHeight && i < magnitudes.rows; i++) {
        for (int j = cellX; j < cellX + cellWidth && j < magnitudes.cols; j++) {
            float magnitude = magnitudes.at<float>(i, j);
            float angle = angles.at<float>(i, j);

            // linear interpolation between adjacent bins
            float binIndex = angle / binWidth;
            int bin1 = (int)floor(binIndex);
            int bin2 = (bin1 + 1) % hogParams.numBins;

            float weight2 = binIndex - bin1;
            float weight1 = 1.0f - weight2;

            histogram[bin1] += magnitude * weight1;
            histogram[bin2] += magnitude * weight2;
        }
    }

    return histogram;
}

vector<float> FaceDetector::normalizeBlock(const vector<float>& blockHist) {
    vector<float> normalized = blockHist;

    // L2-norm with small epsilon to avoid division by zero
    float epsilon = 1e-5;
    float sum = 0.0f;

    for (float val : blockHist) {
        sum += val * val;
    }

    float norm = sqrt(sum + epsilon * epsilon);

    for (float& val : normalized) {
        val /= norm;
    }

    return normalized;
}

vector<float> FaceDetector::computeBlockDescriptor(const vector<vector<vector<float>>>& cellHistograms, int blockX, int blockY, int blocksPerRow, int blocksPerCol) {
    vector<float> blockDescriptor;

    int cellsPerBlockX = hogParams.blockSize.width / hogParams.cellSize.width;
    int cellsPerBlockY = hogParams.blockSize.height / hogParams.cellSize.height;
    int expectedSize = cellsPerBlockX * cellsPerBlockY * hogParams.numBins;

    if (cellHistograms.empty()) {
        return vector<float>(expectedSize, 0.0f);
    }

    int totalCellsY = (int)cellHistograms.size();
    int totalCellsX = 0;

    // find the maximum number of cells in X direction
    for (int i = 0; i < totalCellsY; i++) {
        if (!cellHistograms[i].empty()) {
            totalCellsX = max(totalCellsX, (int)cellHistograms[i].size());
        }
    }

    if (totalCellsX == 0) {
        return vector<float>(expectedSize, 0.0f);
    }

    // concatenate histograms from all cells in the block
    for (int i = 0; i < cellsPerBlockY; i++) {
        for (int j = 0; j < cellsPerBlockX; j++) {
            int cellY = blockY + i;
            int cellX = blockX + j;

            if (cellY < 0 || cellY >= totalCellsY) {
                for (int k = 0; k < hogParams.numBins; k++) {
                    blockDescriptor.push_back(0.0f);
                }
                continue;
            }

            if (cellHistograms[cellY].empty() || cellX < 0 || cellX >= (int)cellHistograms[cellY].size()) {
                for (int k = 0; k < hogParams.numBins; k++) {
                    blockDescriptor.push_back(0.0f);
                }
                continue;
            }

            const vector<float>& cellHist = cellHistograms[cellY][cellX];

            if (cellHist.empty() || (int)cellHist.size() != hogParams.numBins) {
                for (int k = 0; k < hogParams.numBins; k++) {
                    blockDescriptor.push_back(0.0f);
                }
                continue;
            }

            blockDescriptor.insert(blockDescriptor.end(), cellHist.begin(), cellHist.end());
        }
    }

    if ((int)blockDescriptor.size() != expectedSize) {
        blockDescriptor.resize(expectedSize, 0.0f);
    }

    return normalizeBlock(blockDescriptor);
}


vector<float> FaceDetector::computeHOGFeatures(const Mat& image) {
    if (image.empty()) {
        printf("ERROR: Empty image in computeHOGFeatures\n");
        return vector<float>(getHOGFeatureDimension(), 0.0f);
    }

    Mat resized;
    if (image.size() != hogParams.windowSize) {
        resize(image, resized, hogParams.windowSize);
    }
    else {
        resized = image.clone();
    }

    if (resized.rows != hogParams.windowSize.height || resized.cols != hogParams.windowSize.width) {
        printf("ERROR: Failed to resize to window size\n");
        return vector<float>(getHOGFeatureDimension(), 0.0f);
    }

    Mat magnitudes, angles;
    computeGradients(resized, magnitudes, angles);

    int cellsX = resized.cols / hogParams.cellSize.width;
    int cellsY = resized.rows / hogParams.cellSize.height;

    if (cellsX <= 0 || cellsY <= 0) {
        printf("ERROR: Invalid cell dimensions\n");
        return vector<float>(getHOGFeatureDimension(), 0.0f);
    }

    vector<vector<vector<float>>> cellHistograms(cellsY);
    for (int i = 0; i < cellsY; i++) {
        cellHistograms[i].resize(cellsX);
        for (int j = 0; j < cellsX; j++) {
            cellHistograms[i][j].resize(hogParams.numBins, 0.0f);
        }
    }

    // compute cell histograms
    for (int i = 0; i < cellsY; i++) {
        for (int j = 0; j < cellsX; j++) {
            int cellStartY = i * hogParams.cellSize.height;
            int cellStartX = j * hogParams.cellSize.width;
            cellHistograms[i][j] = computeCellHistogram(magnitudes, angles, cellStartX, cellStartY, hogParams.cellSize.width, hogParams.cellSize.height);
        }
    }

    vector<float> hogDescriptor;

    int blocksX = (resized.cols - hogParams.blockSize.width) / hogParams.blockStride.width + 1;
    int blocksY = (resized.rows - hogParams.blockSize.height) / hogParams.blockStride.height + 1;

    if (blocksX <= 0 || blocksY <= 0) {
        printf("ERROR: Invalid block dimensions\n");
        return vector<float>(getHOGFeatureDimension(), 0.0f);
    }

    // compute block descriptors
    for (int i = 0; i < blocksY; i++) {
        for (int j = 0; j < blocksX; j++) {
            int blockCellY = i * hogParams.blockStride.height / hogParams.cellSize.height;
            int blockCellX = j * hogParams.blockStride.width / hogParams.cellSize.width;
            vector<float> blockDesc = computeBlockDescriptor( cellHistograms, blockCellX, blockCellY, blocksX, blocksY);
            hogDescriptor.insert(hogDescriptor.end(), blockDesc.begin(), blockDesc.end());
        }
    }

    int expectedSize = getHOGFeatureDimension();
    if ((int)hogDescriptor.size() != expectedSize) {
        printf("WARNING: HOG descriptor size mismatch (got %d, expected %d), resizing\n", (int)hogDescriptor.size(), expectedSize);
        hogDescriptor.resize(expectedSize, 0.0f);
    }

    return hogDescriptor;
}


int FaceDetector::computeLBPValue(const Mat& image, int x, int y) {
    uchar center = image.at<uchar>(y, x);
    int lbpValue = 0;

    // 8-neighbor circular pattern
    int dx[8] = { 1, 1, 0, -1, -1, -1, 0, 1 };
    int dy[8] = { 0, 1, 1, 1, 0, -1, -1, -1 };

    for (int i = 0; i < 8; i++) {
        int nx = x + dx[i] * lbpParams.radius;
        int ny = y + dy[i] * lbpParams.radius;

        if (nx >= 0 && nx < image.cols && ny >= 0 && ny < image.rows) {
            if (image.at<uchar>(ny, nx) >= center) {
                lbpValue |= (1 << i);
            }
        }
    }

    return lbpValue;
}

vector<float> FaceDetector::computeLBPHistogram(const Mat& image) {
    vector<float> histogram(256, 0.0f);

    int radius = lbpParams.radius;

    for (int i = radius; i < image.rows - radius; i++) {
        for (int j = radius; j < image.cols - radius; j++) {
            int lbpValue = computeLBPValue(image, j, i);
            histogram[lbpValue]++;
        }
    }

    float sum = 0.0f;
    for (float val : histogram) {
        sum += val;
    }

    if (sum > 0) {
        for (float& val : histogram) {
            val /= sum;
        }
    }

    return histogram;
}

vector<float> FaceDetector::computeLBPFeatures(const Mat& image) {
    Mat resized;
    if (image.size() != hogParams.windowSize) {
        resize(image, resized, hogParams.windowSize);
    }
    else {
        resized = image.clone();
    }

    return computeLBPHistogram(resized);
}

vector<float> FaceDetector::computeCombinedFeatures(const Mat& image) {
    vector<float> hogFeatures = computeHOGFeatures(image);
    vector<float> lbpFeatures = computeLBPFeatures(image);

    vector<float> combined;
    combined.reserve(hogFeatures.size() + lbpFeatures.size());
    combined.insert(combined.end(), hogFeatures.begin(), hogFeatures.end());
    combined.insert(combined.end(), lbpFeatures.begin(), lbpFeatures.end());

    return combined;
}

vector<Rect> FaceDetector::generateNegativeSamples(const Mat& image,
    const vector<Rect>& faces,
    int numSamples) {
    vector<Rect> negatives;
    int attempts = 0;
    int maxAttempts = numSamples * 10;

    while ((int)negatives.size() < numSamples && attempts < maxAttempts) {
        attempts++;

        // random position
        int x = rand() % max(1, (image.cols - hogParams.windowSize.width));
        int y = rand() % max(1, (image.rows - hogParams.windowSize.height));
        Rect candidate(x, y, hogParams.windowSize.width, hogParams.windowSize.height);

        // check if it overlaps with any face
        bool overlaps = false;
        for (const Rect& face : faces) {
            float iou = computeIOU(candidate, face);
            if (iou > 0.3) {
                overlaps = true;
                break;
            }
        }

        if (!overlaps) {
            negatives.push_back(candidate);
        }
    }

    return negatives;
}

void FaceDetector::trainFromAnnotatedImages(const vector<Mat>& images, const vector<vector<Rect>>& faceAnnotations, int negativeSamplesPerImage) {
    printf("\n=== Training HOG+SVM Face Detector ===\n");
    printf("Window Size: %dx%d\n", hogParams.windowSize.width, hogParams.windowSize.height);
    printf("Block Size: %dx%d\n", hogParams.blockSize.width, hogParams.blockSize.height);
    printf("Cell Size: %dx%d\n", hogParams.cellSize.width, hogParams.cellSize.height);
    printf("HOG Bins: %d\n", hogParams.numBins);
    printf("LBP Radius: %d, Neighbors: %d\n", lbpParams.radius, lbpParams.neighbors);

    int featureDim = getFeatureDimension();
    printf("Feature Dimension: %d\n", featureDim);
    printf("\nExtracting features from training images...\n");

    vector<vector<float>> positiveFeatures;
    vector<vector<float>> negativeFeatures;

    int totalFaces = 0;
    for (const auto& faces : faceAnnotations) {
        totalFaces += faces.size();
    }

    printf("Processing %d images with %d total face annotations...\n", (int)images.size(), totalFaces);

    int maxNegativesPerImage = min(negativeSamplesPerImage, 1);
    printf("Limiting negative samples to %d per image\n", maxNegativesPerImage);

    int processedFaces = 0;
    int skippedFaces = 0;

    for (size_t i = 0; i < images.size(); i++) {
        if (images[i].empty() || images[i].cols < hogParams.windowSize.width || images[i].rows < hogParams.windowSize.height) {
            printf("  Skipping image %d: too small\n", (int)i);
            continue;
        }

        Mat gray;
        try {
            gray = preprocessImage(images[i]);
        }
        catch (const Exception& e) {
            printf("  Error preprocessing image %d: %s\n", (int)i, e.what());
            continue;
        }

        if (gray.empty()) {
            printf("  Skipping image %d: preprocessing failed\n", (int)i);
            continue;
        }

        for (const Rect& face : faceAnnotations[i]) {
            Rect safeFace = face & Rect(0, 0, gray.cols, gray.rows);

            if (safeFace.width < hogParams.windowSize.width / 2 ||
                safeFace.height < hogParams.windowSize.height / 2) {
                skippedFaces++;
                continue;
            }

            try {
                Mat faceROI = gray(safeFace);

                if (faceROI.empty()) {
                    skippedFaces++;
                    continue;
                }

                vector<float> features = computeCombinedFeatures(faceROI);

                if ((int)features.size() != featureDim) {
                    printf("  Warning: Feature size mismatch at image %d (got %d, expected %d)\n", (int)i, (int)features.size(), featureDim);
                    skippedFaces++;
                    continue;
                }

                bool valid = true;
                for (float f : features) {
                    if (!isfinite(f)) {
                        valid = false;
                        break;
                    }
                }

                if (!valid) {
                    printf("  Warning: Invalid feature values at image %d\n", (int)i);
                    skippedFaces++;
                    continue;
                }

                positiveFeatures.push_back(features);
                processedFaces++;

                if (processedFaces % 100 == 0) {
                    printf("  Positive samples: %d/%d (skipped: %d)\n",
                        processedFaces, totalFaces, skippedFaces);
                }
            }
            catch (const Exception& e) {
                printf("  Error extracting face features from image %d: %s\n", (int)i, e.what());
                skippedFaces++;
            }
        }

        try {
            vector<Rect> negSamples = generateNegativeSamples(gray, faceAnnotations[i], maxNegativesPerImage);

            for (const Rect& neg : negSamples) {
                try {
                    Mat negROI = gray(neg);

                    if (negROI.empty()) continue;

                    vector<float> features = computeCombinedFeatures(negROI);

                    if ((int)features.size() != featureDim) {
                        continue;
                    }

                    bool valid = true;
                    for (float f : features) {
                        if (!isfinite(f)) {
                            valid = false;
                            break;
                        }
                    }

                    if (valid) {
                        negativeFeatures.push_back(features);
                    }
                }
                catch (const Exception& e) {
                    // skip this negative sample
                }
            }
        }
        catch (const Exception& e) {
            printf("  Error generating negative samples from image %d: %s\n", (int)i, e.what());
        }

        if ((i + 1) % 50 == 0) {
            printf("  Processed %d/%d images (Pos: %d, Neg: %d, Skipped: %d)\n", (int)(i + 1), (int)images.size(), (int)positiveFeatures.size(), (int)negativeFeatures.size(), skippedFaces);
        }
    }

    printf("\nPositive samples: %d\n", (int)positiveFeatures.size());
    printf("Negative samples: %d\n", (int)negativeFeatures.size());
    printf("Skipped faces: %d\n", skippedFaces);

    if (positiveFeatures.empty()) {
        printf("\nERROR: No valid positive samples extracted!\n");
        return;
    }

    if (negativeFeatures.empty()) {
        printf("\nWARNING: No negative samples extracted, training with only positives\n");
    }

    int totalSamples = positiveFeatures.size() + negativeFeatures.size();
    Mat trainingData(totalSamples, featureDim, CV_32F, Scalar(0));
    Mat labels(totalSamples, 1, CV_32S);

    printf("\nPreparing training matrix...\n");

    for (size_t i = 0; i < positiveFeatures.size(); i++) {
        for (int j = 0; j < featureDim; j++) {
            trainingData.at<float>((int)i, j) = positiveFeatures[i][j];
        }
        labels.at<int>((int)i) = 1;
    }

    for (size_t i = 0; i < negativeFeatures.size(); i++) {
        int row = (int)positiveFeatures.size() + (int)i;
        for (int j = 0; j < featureDim; j++) {
            trainingData.at<float>(row, j) = negativeFeatures[i][j];
        }
        labels.at<int>(row) = -1;
    }

    printf("Training SVM classifier...\n");
    printf("  Samples: %d\n", totalSamples);
    printf("  Features: %d\n", featureDim);

    try {
        svmClassifier->train(trainingData, ROW_SAMPLE, labels);
        isTrained = true;
        printf("\n✓ Training completed successfully!\n");
    }
    catch (const Exception& e) {
        printf("\nERROR during SVM training: %s\n", e.what());
        return;
    }

    printf("=====================================\n\n");
}

void FaceDetector::trainClassifier(const string& positivePath, const string& negativePath, int numPositive, int numNegative) {
    printf("\n=== Training HOG+SVM Face Detector ===\n");
    printf("Loading training images...\n");

    vector<Mat> positiveImages = Utils::loadImagesFromFolder(positivePath, numPositive);
    vector<Mat> negativeImages = Utils::loadImagesFromFolder(negativePath, numNegative);

    printf("Positive images: %d\n", (int)positiveImages.size());
    printf("Negative images: %d\n", (int)negativeImages.size());

    int featureDim = getFeatureDimension();
    int totalSamples = positiveImages.size() + negativeImages.size();

    Mat trainingData(totalSamples, featureDim, CV_32F);
    Mat labels(totalSamples, 1, CV_32S);

    printf("Extracting features...\n");

    // process positive samples
    for (size_t i = 0; i < positiveImages.size(); i++) {
        Mat gray = preprocessImage(positiveImages[i]);
        vector<float> features = computeCombinedFeatures(gray);

        for (int j = 0; j < featureDim; j++) {
            trainingData.at<float>(i, j) = features[j];
        }
        labels.at<int>(i) = 1;

        if ((i + 1) % 100 == 0) {
            printf("  Positive: %d/%d\n", (int)(i + 1), (int)positiveImages.size());
        }
    }

    // process negative samples
    for (size_t i = 0; i < negativeImages.size(); i++) {
        int row = positiveImages.size() + i;
        Mat gray = preprocessImage(negativeImages[i]);
        vector<float> features = computeCombinedFeatures(gray);

        for (int j = 0; j < featureDim; j++) {
            trainingData.at<float>(row, j) = features[j];
        }
        labels.at<int>(row) = -1;

        if ((i + 1) % 100 == 0) {
            printf("  Negative: %d/%d\n", (int)(i + 1), (int)negativeImages.size());
        }
    }

    printf("Training SVM...\n");
    svmClassifier->train(trainingData, ROW_SAMPLE, labels);
    isTrained = true;

    printf("✓ Training completed!\n");
    printf("====================================\n\n");
}

bool FaceDetector::saveModel(const string& modelPath) {
    if (!isTrained) {
        printf("ERROR: Model not trained yet!\n");
        return false;
    }

    svmClassifier->save(modelPath);
    printf("Model saved to: %s\n", modelPath.c_str());
    return true;
}

bool FaceDetector::loadModel(const string& modelPath) {
    try {
        svmClassifier = SVM::load(modelPath);
        isTrained = true;
        printf("Model loaded from: %s\n", modelPath.c_str());
        return true;
    }
    catch (const Exception& e) {
        printf("ERROR loading model: %s\n", e.what());
        return false;
    }
}

float FaceDetector::computeIOU(const Rect& box1, const Rect& box2) {
    int x1 = max(box1.x, box2.x);
    int y1 = max(box1.y, box2.y);
    int x2 = min(box1.x + box1.width, box2.x + box2.width);
    int y2 = min(box1.y + box1.height, box2.y + box2.height);

    int intersectionArea = max(0, x2 - x1) * max(0, y2 - y1);
    int box1Area = box1.width * box1.height;
    int box2Area = box2.width * box2.height;
    int unionArea = box1Area + box2Area - intersectionArea;

    if (unionArea <= 0)
        return 0.0f;

    return (float)intersectionArea / unionArea;
}

vector<Rect> FaceDetector::nonMaximumSuppression(vector<Rect>& boxes, vector<float>& scores, float overlapThresh) {
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

vector<pair<Rect, float>> FaceDetector::detectFacesWithConfidence(Mat& image, float threshold, bool useMultiScale) {
    if (!isTrained) {
        printf("ERROR: Model not trained! Train or load a model first.\n");
        return vector<pair<Rect, float>>();
    }

    Mat gray = preprocessImage(image);
    vector<pair<Rect, float>> detections;

    // multi-scale detection
    vector<float> scales = { 1.0 };
    if (useMultiScale) {
        scales = { 0.5, 0.75, 1.0, 1.25, 1.5, 2.0 };
    }

    for (float scale : scales) {
        Mat scaledImage;
        resize(gray, scaledImage, Size(), scale, scale);

        if (scaledImage.cols < hogParams.windowSize.width ||
            scaledImage.rows < hogParams.windowSize.height) {
            continue;
        }

        // sliding window
        int strideX = hogParams.blockStride.width;
        int strideY = hogParams.blockStride.height;

        for (int y = 0; y <= scaledImage.rows - hogParams.windowSize.height; y += strideY) {
            for (int x = 0; x <= scaledImage.cols - hogParams.windowSize.width; x += strideX) {
                Rect window(x, y, hogParams.windowSize.width, hogParams.windowSize.height);
                Mat windowROI = scaledImage(window);

                vector<float> features = computeCombinedFeatures(windowROI);
                int featureDim = getFeatureDimension();
                Mat featureMat(1, featureDim, CV_32F, Scalar(0));

                for (int i = 0; i < min((int)features.size(), featureDim); i++) {
                    featureMat.at<float>(0, i) = features[i];
                }

                float response = svmClassifier->predict(featureMat);
                if (response == 1) {
                    // convert back to original image coordinates
                    Rect originalRect((int)(x / scale), (int)(y / scale), (int)(hogParams.windowSize.width / scale), (int)(hogParams.windowSize.height / scale));
                    detections.push_back({ originalRect, response });
                }
            }
        }
    }

    return detections;
}

vector<Rect> FaceDetector::detectFaces(Mat& image, float threshold,
    bool useMultiScale, bool preprocess) {
    vector<pair<Rect, float>> detectedWithConf = detectFacesWithConfidence(image, threshold, useMultiScale);

    vector<Rect> boxes;
    vector<float> scores;

    for (const auto& det : detectedWithConf) {
        boxes.push_back(det.first);
        scores.push_back(det.second);
    }

    vector<Rect> finalDetections = nonMaximumSuppression(boxes, scores, 0.3f);

    return finalDetections;
}

void FaceDetector::drawDetections(Mat& image, const vector<Rect>& faces,
    const Scalar& color) {
    for (size_t i = 0; i < faces.size(); i++) {
        rectangle(image, faces[i], color, 2);
        string label = "Face " + to_string(i + 1);
        putText(image, label, Point(faces[i].x, faces[i].y - 10), FONT_HERSHEY_SIMPLEX, 0.5, color, 2);
    }
}

void FaceDetector::displayTrainingInfo() {
    printf("\n=== HOG+SVM Face Detector Info ===\n");
    printf("Status: %s\n", isTrained ? "Trained" : "Not Trained");
    printf("Window Size: %dx%d\n", hogParams.windowSize.width, hogParams.windowSize.height);
    printf("HOG Feature Dimension: %d\n", getHOGFeatureDimension());
    printf("LBP Feature Dimension: %d\n", getLBPFeatureDimension());
    printf("Total Feature Dimension: %d\n", getFeatureDimension());
    printf("==================================\n\n");
}

void FaceDetector::evaluateModel(const vector<Mat>& testImages,
    const vector<vector<Rect>>& groundTruth) {
    if (!isTrained) {
        printf("ERROR: Model not trained!\n");
        return;
    }

    printf("\n=== Evaluating Model ===\n");

    int totalTP = 0, totalFP = 0, totalFN = 0;

    for (size_t i = 0; i < testImages.size(); i++) {
        Mat image = testImages[i].clone();
        vector<Rect> detections = detectFaces(image, 0.0, true, true);

        Utils::DetectionMetrics metrics = Utils::computeMetrics(detections, groundTruth[i], 0.5);

        totalTP += metrics.truePositives;
        totalFP += metrics.falsePositives;
        totalFN += metrics.falseNegatives;

        if ((i + 1) % 10 == 0) {
            printf("  Evaluated %d/%d images\n", (int)(i + 1), (int)testImages.size());
        }
    }

    Utils::DetectionMetrics overallMetrics;
    overallMetrics.truePositives = totalTP;
    overallMetrics.falsePositives = totalFP;
    overallMetrics.falseNegatives = totalFN;
    overallMetrics.calculate();
    overallMetrics.print();
}

int FaceDetector::getHOGFeatureDimension() const {
    int cellsPerBlockX = hogParams.blockSize.width / hogParams.cellSize.width;
    int cellsPerBlockY = hogParams.blockSize.height / hogParams.cellSize.height;
    int cellsPerBlock = cellsPerBlockX * cellsPerBlockY;

    int blocksX = (hogParams.windowSize.width - hogParams.blockSize.width) / hogParams.blockStride.width + 1;
    int blocksY = (hogParams.windowSize.height - hogParams.blockSize.height) / hogParams.blockStride.height + 1;
    int totalBlocks = blocksX * blocksY;

    return totalBlocks * cellsPerBlock * hogParams.numBins;
}

int FaceDetector::getLBPFeatureDimension() const {
    return 256; // LBP histogram size
}

int FaceDetector::getFeatureDimension() const {
    return getHOGFeatureDimension() + getLBPFeatureDimension();
}

namespace Utils {
    vector<Mat> loadImagesFromFolder(const string& folderPath, int maxImages) {
        vector<Mat> images;
        if (folderPath.empty()) return images;
        return images;
    }

    vector<Rect> generateSlidingWindows(const Size& imageSize, const Size& windowSize,
        const Size& stride) {
        vector<Rect> windows;

        for (int y = 0; y <= imageSize.height - windowSize.height; y += stride.height) {
            for (int x = 0; x <= imageSize.width - windowSize.width; x += stride.width) {
                windows.push_back(Rect(x, y, windowSize.width, windowSize.height));
            }
        }

        return windows;
    }

    Mat resizeWithAspectRatio(const Mat& image, int targetSize) {
        float aspectRatio = (float)image.cols / image.rows;
        int newWidth, newHeight;

        if (image.cols > image.rows) {
            newWidth = targetSize;
            newHeight = (int)(targetSize / aspectRatio);
        }
        else {
            newHeight = targetSize;
            newWidth = (int)(targetSize * aspectRatio);
        }

        Mat resized;
        resize(image, resized, Size(newWidth, newHeight));
        return resized;
    }

    vector<Mat> augmentImage(const Mat& image, bool flipHorizontal,
        bool adjustBrightness, int numVariations) {
        vector<Mat> augmented;
        augmented.push_back(image.clone());

        if (flipHorizontal) {
            Mat flipped;
            flip(image, flipped, 1);
            augmented.push_back(flipped);
        }

        return augmented;
    }

    DetectionMetrics computeMetrics(const vector<Rect>& detections,
        const vector<Rect>& groundTruth,
        float iouThreshold) {
        DetectionMetrics metrics;
        metrics.truePositives = 0;
        metrics.falsePositives = 0;
        metrics.falseNegatives = 0;

        vector<bool> gtMatched(groundTruth.size(), false);

        for (const Rect& det : detections) {
            bool matched = false;

            for (size_t i = 0; i < groundTruth.size(); i++) {
                if (gtMatched[i]) continue;

                // compute IOU
                int x1 = max(det.x, groundTruth[i].x);
                int y1 = max(det.y, groundTruth[i].y);
                int x2 = min(det.x + det.width, groundTruth[i].x + groundTruth[i].width);
                int y2 = min(det.y + det.height, groundTruth[i].y + groundTruth[i].height);

                int intersection = max(0, x2 - x1) * max(0, y2 - y1);
                int area1 = det.width * det.height;
                int area2 = groundTruth[i].width * groundTruth[i].height;
                float iou = (float)intersection / (area1 + area2 - intersection);

                if (iou >= iouThreshold) {
                    metrics.truePositives++;
                    gtMatched[i] = true;
                    matched = true;
                    break;
                }
            }

            if (!matched) {
                metrics.falsePositives++;
            }
        }

        // count unmatched ground truth boxes as false negatives
        for (bool matched : gtMatched) {
            if (!matched) {
                metrics.falseNegatives++;
            }
        }

        metrics.calculate();
        return metrics;
    }
}

vector<Rect> FaceDetector::parseAnnotationFile(const string& annotPath, int imageWidth, int imageHeight) {
    vector<Rect> faces;
    ifstream annotFile(annotPath);

    if (!annotFile.is_open()) {
        return faces;
    }

    string line;
    int lineNumber = 0;
    while (getline(annotFile, line)) {
        lineNumber++;

        if (line.empty() || line[0] == '#') {
            continue;
        }

        istringstream iss(line);
        vector<float> values;
        float val;

        while (iss >> val) {
            values.push_back(val);
        }

        if (values.size() < 4) {
            printf("  Warning: Line %d in %s has insufficient values (%d), skipping\n", lineNumber, annotPath.c_str(), (int)values.size());
            continue;
        }

        float x_norm, y_norm, w_norm, h_norm;

        x_norm = values[0];
        y_norm = values[1];
        w_norm = values[2];
        h_norm = values[3];

        // convert normalized coordinates to absolute pixel coordinates
        // format: top-left x, top-left y, width, height
        int x = (int)(x_norm * imageWidth);
        int y = (int)(y_norm * imageHeight);
        int w = (int)(w_norm * imageWidth);
        int h = (int)(h_norm * imageHeight);

        if (x < 0 || y < 0 || w <= 0 || h <= 0) {
            printf("  Warning: Line %d has invalid bbox values (x=%d, y=%d, w=%d, h=%d), skipping\n", lineNumber, x, y, w, h);
            continue;
        }

        // clamp to image boundaries
        x = max(0, min(x, imageWidth - 1));
        y = max(0, min(y, imageHeight - 1));

        // adjust width and height if they exceed image boundaries
        if (x + w > imageWidth) {
            w = imageWidth - x;
        }
        if (y + h > imageHeight) {
            h = imageHeight - y;
        }

        // ensure minimum size
        w = max(1, w);
        h = max(1, h);

        faces.push_back(Rect(x, y, w, h));
    }

    annotFile.close();
    return faces;
}

string FaceDetector::getAnnotationFilename(const string& imageFilename) {
    size_t lastDot = imageFilename.find_last_of('.');
    if (lastDot != string::npos) {
        return imageFilename.substr(0, lastDot) + ".txt";
    }
    return imageFilename + ".txt";
}

bool FaceDetector::loadDatasetFromFolders(const string& imagesFolder, const string& annotationsFolder, vector<Mat>& images, vector<vector<Rect>>& annotations, const string& extension) {
    images.clear();
    annotations.clear();

    printf("Loading dataset...\n");
    printf("  Images folder: %s\n", imagesFolder.c_str());
    printf("  Annotations folder: %s\n", annotationsFolder.c_str());

    vector<string> imageExtensions = { "jpg", "jpeg", "png", "bmp" };

    int imageCount = 0;
    int skippedCount = 0;
    int totalFaces = 0;

    for (const string& ext : imageExtensions) {
        char imagesFolderCopy[MAX_PATH];
        char extensionCopy[MAX_PATH];
        strcpy(imagesFolderCopy, imagesFolder.c_str());
        strcpy(extensionCopy, ext.c_str());

        char fname[MAX_PATH];
        FileGetter fg(imagesFolderCopy, extensionCopy);

        while (fg.getNextAbsFile(fname)) {
            Mat img = imread(fname, IMREAD_COLOR);
            if (img.empty()) {
                printf("  Warning: Could not load image: %s\n", fname);
                skippedCount++;
                continue;
            }

            int imageWidth = img.cols;
            int imageHeight = img.rows;

            string fullPath(fname);
            size_t lastSlash = fullPath.find_last_of("/\\");
            string filename = (lastSlash != string::npos) ? fullPath.substr(lastSlash + 1) : fullPath;

            string annotFilename = getAnnotationFilename(filename);
            string annotPath = string(annotationsFolder) + "/" + annotFilename;

            vector<Rect> faces = parseAnnotationFile(annotPath, imageWidth, imageHeight);

            if (faces.empty()) {
                printf("  Warning: No faces found in annotation: %s\n", annotFilename.c_str());
                skippedCount++;
                continue;
            }

            images.push_back(img);
            annotations.push_back(faces);
            imageCount++;
            totalFaces += faces.size();

            if (imageCount % 50 == 0) {
                printf("  Loaded %d images (%d faces)...\n", imageCount, totalFaces);
            }
        }
    }

    printf("\nDataset loading complete:\n");
    printf("  Successfully loaded: %d images\n", imageCount);
    printf("  Total faces: %d\n", totalFaces);
    if (imageCount > 0) {
        printf("  Average faces per image: %.2f\n", (float)totalFaces / imageCount);
    }
    printf("  Skipped: %d images\n", skippedCount);

    return imageCount > 0;
}

void FaceDetector::commandTrain() {
    printf("\n========================================\n");
    printf("   FACE DETECTOR TRAINING\n");
    printf("========================================\n\n");

    vector<Mat> trainingImages;
    vector<vector<Rect>> faceAnnotations;

    if (!loadDatasetFromFolders(TRAIN_IMAGES_PATH, TRAIN_ANNOTATIONS_PATH, trainingImages, faceAnnotations, "")) {
        printf("\n✗ ERROR: Failed to load training data!\n");
        printf("Please check:\n");
        printf("  1. Images folder exists and contains image files (.jpg, .png, .bmp)\n");
        printf("  2. Annotations folder exists and contains corresponding .txt files\n");
        printf("  3. Annotation format: x y width height (one face per line)\n");
        printf("  4. Paths are correct in the source code\n");
        return;
    }

    int totalFaces = 0;
    for (const auto& faces : faceAnnotations) {
        totalFaces += faces.size();
    }

    printf("\nDataset statistics:\n");
    printf("  Total images: %d\n", (int)trainingImages.size());
    printf("  Total faces: %d\n", totalFaces);
    printf("  Average faces per image: %.2f\n", (float)totalFaces / trainingImages.size());

    printf("\nProceed with training? (y/n): ");
    char confirm;
    scanf(" %c", &confirm);

    if (confirm != 'y' && confirm != 'Y') {
        printf("Training cancelled.\n");
        return;
    }

    printf("\n========================================\n");
    printf("Starting training... This may take several minutes.\n");
    printf("========================================\n\n");

    double startTime = (double)getTickCount();

    trainFromAnnotatedImages(trainingImages, faceAnnotations, 1);

    double endTime = (double)getTickCount();
    double trainingTime = (endTime - startTime) / getTickFrequency();

    printf("\n========================================\n");
    printf("Training completed in %.2f seconds (%.2f minutes)\n", trainingTime, trainingTime / 60.0);
    printf("========================================\n\n");

    displayTrainingInfo();

    printf("\nSave trained model? (y/n): ");
    char saveResponse;
    scanf(" %c", &saveResponse);

    if (saveResponse == 'y' || saveResponse == 'Y') {
        printf("Enter model filename (e.g., face_detector.xml): ");
        char modelName[256];
        scanf("%s", modelName);

        if (saveModel(modelName)) {
            printf("\n✓ Model saved successfully to: %s\n", modelName);
        }
        else {
            printf("\n✗ ERROR: Failed to save model.\n");
        }
    }

    printf("\n========================================\n");
    printf("   TRAINING COMPLETE!\n");
    printf("========================================\n");
}

void FaceDetector::commandLoadModel() {
    printf("\n========================================\n");
    printf("   LOAD MODEL\n");
    printf("========================================\n\n");

    char fname[MAX_PATH];
    printf("Select the model file (.xml or .yml)...\n");

    if (openFileDlg(fname)) {
        printf("Loading model from: %s\n", fname);

        if (loadModel(fname)) {
            printf("\n✓ Model loaded successfully!\n\n");
            displayTrainingInfo();
        }
        else {
            printf("\n✗ ERROR: Failed to load model from: %s\n", fname);
            printf("Please check that the file is a valid trained model.\n");
        }
    }
    else {
        printf("No file selected.\n");
    }
}

void FaceDetector::commandTestOnImage() {
    if (!isTrained) {
        printf("\n✗ ERROR: Model not trained!\n");
        printf("Please train a model or load an existing model first.\n");
        return;
    }

    printf("\n========================================\n");
    printf("   FACE DETECTION\n");
    printf("========================================\n\n");

    displayTrainingInfo();

    printf("Select images to process. Press ESC to exit.\n\n");

    char fname[MAX_PATH];
    int imageCount = 0;

    while (openFileDlg(fname)) {
        Mat image = imread(fname);
        if (image.empty()) {
            printf("✗ ERROR: Could not load image: %s\n", fname);
            continue;
        }

        imageCount++;
        printf("\n[Image %d] Processing: %s\n", imageCount, fname);
        printf("  Resolution: %dx%d\n", image.cols, image.rows);

        double startTime = (double)getTickCount();

        Mat displayImage = image.clone();
        vector<Rect> faces = detectFaces(displayImage, 0.0, true, true);

        double endTime = (double)getTickCount();
        double processingTime = (endTime - startTime) / getTickFrequency() * 1000.0;

        printf("  Detected faces: %d\n", (int)faces.size());
        printf("  Processing time: %.2f ms\n", processingTime);

        for (size_t i = 0; i < faces.size(); i++) {
            printf("    Face %d: [%d, %d, %dx%d]\n", (int)(i + 1), faces[i].x, faces[i].y, faces[i].width, faces[i].height);
        }

        drawDetections(displayImage, faces, Scalar(0, 255, 0));

        string infoText = "Faces: " + to_string(faces.size()) + " | Time: " + to_string((int)processingTime) + "ms";
        putText(displayImage, infoText, Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 255, 255), 2);
        putText(displayImage, infoText, Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 255, 0), 1);

        imshow("Original Image", image);
        imshow("Face Detection", displayImage);

        printf("\nPress any key to continue, ESC to exit...\n");
        int key = waitKey(0);
        if (key == 27) {
            printf("\nExiting image detection.\n");
            break;
        }
    }

    printf("\n========================================\n");
    printf("Processed %d image(s)\n", imageCount);
    printf("========================================\n");
}

void FaceDetector::commandEvaluate() {
    if (!isTrained) {
        printf("\n✗ ERROR: Model not trained!\n");
        printf("Please train a model or load an existing model first.\n");
        return;
    }

    printf("\n========================================\n");
    printf("   EVALUATE MODEL PERFORMANCE\n");
    printf("========================================\n\n");

    printf("Using hardcoded test paths:\n");
    printf("  Images: %s\n", TEST_IMAGES_PATH);
    printf("  Annotations: %s\n\n", TEST_ANNOTATIONS_PATH);

    vector<Mat> testImages;
    vector<vector<Rect>> groundTruth;

    if (!loadDatasetFromFolders(TEST_IMAGES_PATH, TEST_ANNOTATIONS_PATH,
        testImages, groundTruth, "")) {
        printf("\n✗ ERROR: Failed to load test data!\n");
        printf("Please check:\n");
        printf("  1. Test images folder exists and contains images\n");
        printf("  2. Test annotations folder exists with corresponding .txt files\n");
        printf("  3. Paths are correct in the source code\n");
        return;
    }

    printf("Proceed with evaluation? (y/n): ");
    char confirm;
    scanf(" %c", &confirm);

    if (confirm != 'y' && confirm != 'Y') {
        printf("Evaluation cancelled.\n");
        return;
    }

    printf("\n========================================\n");
    printf("Starting evaluation...\n");
    printf("========================================\n\n");

    evaluateModel(testImages, groundTruth);

    printf("\n========================================\n");
    printf("   EVALUATION COMPLETE\n");
    printf("========================================\n");
}