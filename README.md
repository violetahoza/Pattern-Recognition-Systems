# Pattern Recognition Systems Lab - OpenCV Application

## Project Description

This repository contains my work for the **Pattern Recognition Systems** laboratory, implemented in C++ using OpenCV. The main goal of the project is to demonstrate practical algorithms and techniques commonly used in pattern recognition and computer vision. The application provides a menu-driven console interface to experiment with various image processing, feature extraction, and model fitting methods.

## How to Run

1. **Requirements**
   - OpenCV
   - C++ compiler 

2. **Build Instructions**
   - Open the project in your C++ IDE.
   - Make sure to set up OpenCV include and library paths.
   - Build and run the application.

3. **Usage**
   - On launch, a menu will appear in the console.
   - Enter the number corresponding to the experiment you wish to run.
   - For image operations, dialog boxes will appear to select files or folders.
   - Results are displayed in OpenCV windows; press any key to continue or ESC to exit certain modes.


## Algorithms Implemented

- **Least Mean Squares (LMS) Line Fitting**: Computes the best-fit line for a set of 2D points using two closed-form models.
- **RANSAC Line Fitting**: Fits a robust line to noisy data by repeatedly sampling minimal subsets and maximizing consensus.

