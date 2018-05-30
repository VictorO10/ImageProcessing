// OpenCVApplication.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "common.h"
#include <stdint.h>
#include <limits.h>

#define INTENSITY 100
#define CENSUS 101
#define CENSUS_KERNEL_H 5
#define CENSUS_KERNEL_W 5
#define DISPARITY_LEVEL 140
//#define PENALTY1 20
//#define PENALTY2 3000 
#define PENALTY1 1
#define PENALTY2 15  

#define NB_PATHS 8


bool inBounds(Point x, int rows, int cols) {
	return (x.x < rows) && (x.y < cols) && (x.x >= 0) && (x.y >= 0);
}

Mat_<int> censusTransorm(Mat_<uchar> img) {

	Mat_<int> census = Mat(img.rows, img.cols, CV_32SC1);

	int kH = CENSUS_KERNEL_H / 2;
	int kW = CENSUS_KERNEL_W / 2;

	for (int i = 0; i < img.rows; ++i) {
		for (int j = 0; j < img.cols; ++j) {
			int encoded = 0;
			int pos = 0;
			int one = 1;
			for (int dH = -kH; dH <= kH; ++dH) {
				for (int dW = -kW; dW <= kW; ++dW) {
					int ii = i + dH;
					int jj = j + dW;
					if (inBounds(Point(ii, jj), img.rows, img.cols)) {
						if (img[ii][jj] > img[i][j]) {
							encoded = encoded | (one << pos);
						}
					}
					pos++;
				}
			}
			census[i][j] = encoded;
		}
	}

	return census;
}


int computeCost(int leftValue, int rightValue, int type) {

	int n1, n2, distance, val;

	switch (type) {
	case INTENSITY: //return square difference of the points
		return (leftValue - rightValue)*(leftValue - rightValue);
	case CENSUS: //calculate Hamming distance
		n1 = leftValue;
		n2 = rightValue;

		distance = 0;
		val = n1 ^ n2;
		while (val) {
			++distance;
			val &= val - 1;
		}

		return distance;
	default:
		return 0;
	}

	return 0;
}

//calculate cost for each disparity
Mat getCostForEachDisparity(Mat_<int> left, Mat_<int> right, int type) {
	const int ROWS = left.rows;
	const int COLS = left.cols;
	const int MAX_DISPARITY = DISPARITY_LEVEL;
	int dims[3] = { ROWS, COLS, MAX_DISPARITY };

	Mat costs = Mat::zeros(3, dims, CV_32S);

	for (int i = 0; i < ROWS; ++i) {
		std::cout << ((float)i / left.rows) * 100 << "%\n";
		for (int j = 0; j < COLS; ++j) {
			for (int k = 0; k < MAX_DISPARITY; ++k) {
				Point rightPoint(i, j + k);
				if (inBounds(rightPoint, left.rows, left.cols)) {
					costs.at<int>(i, j, k) =
						computeCost(left[i][j + k], right[i][j], type);
				}
				else {
					if (type == CENSUS) {
						costs.at<int>(i, j, k) = 30;
					}
					if (type == INTENSITY) {
						costs.at<int>(i, j, k) = 255;
					}
				}
			}
		}
	}
	return costs;
}

int minimum(int a, int b) {
	if (a < b)
		return a;
	return b;
}

int minimum(int a, int b, int c, int d) {
	return minimum(a, minimum(b, minimum(c, d))); //must be tested
}

Mat dynamicProgrammingCosts(Mat local_cost, const int ROWS, const int COLS) {

	const int MAX_DISPARITY = DISPARITY_LEVEL;
	const int PATHS = NB_PATHS;
	int dims[4] = { ROWS, COLS, MAX_DISPARITY, PATHS };

	Mat L = Mat::zeros(4, dims, CV_32S);

	int di[] = { 0, -1, -1, -1, 0, +1, +1, +1 };
	int dj[] = { -1, -1, 0, +1, +1, +1, 0, -1 };

	//iterate from top left to bottom right
	for (int i = 0; i < ROWS; ++i) {
		std::cout << ((float)i / ROWS) * 100 / 2 << "%\n";
		for (int j = 0; j < COLS; ++j) {
			for (int d = 0; d < MAX_DISPARITY; ++d) {
				for (int r = 0; r < PATHS / 2; ++r) {
					L.at<int>(i, j, d, r) = local_cost.at<int>(i, j, d);

					int ii = i + di[r];
					int jj = j + dj[r];

					if (inBounds(Point(ii, jj), ROWS, COLS)) {
						int L1 = L.at<int>(ii, jj, d, r);
						int L2 = INT_MAX;
						if (d - 1 >= 0) {
							L2 = L.at<int>(ii, jj, d - 1, r) + PENALTY1;
						}
						int L3 = INT_MAX;
						if (d + 1 < MAX_DISPARITY) {
							L3 = L.at<int>(ii, jj, d + 1, r) + PENALTY1;
						}
						int L4 = INT_MAX;
						int minn = INT_MAX; //compute the minimum to subtract it later
						for (int k = 0; k < MAX_DISPARITY; ++k) {
							int val = L.at<int>(ii, jj, k, r) + PENALTY2;
							if (val < L4) {
								L4 = val;
							}
							val = L.at<int>(ii, jj, k, r);
							if (val < minn) {
								minn = val;
							}
						}

						L.at<int>(i, j, d, r) =
							L.at<int>(i, j, d, r) + minimum(L1, L2, L3, L4) - minn; //formula (13)
					}
				}
			}
		}
	}

	//iterate from bottom right to top left
	for (int i = ROWS - 1; i >= 0; --i) {
		std::cout << ((float)(ROWS - i) / ROWS) * 100 / 2 + 50 << "%\n";
		for (int j = COLS - 1; j >= 0; --j) {
			for (int d = 0; d < MAX_DISPARITY; ++d) {
				for (int r = PATHS / 2; r < PATHS; ++r) {
					L.at<int>(i, j, d, r) = local_cost.at<int>(i, j, d);

					int ii = i + di[r];
					int jj = j + dj[r];

					if (inBounds(Point(ii, jj), ROWS, COLS)) {
						int L1 = L.at<int>(ii, jj, d, r);
						int L2 = INT_MAX;
						if (d - 1 >= 0) {
							L2 = L.at<int>(ii, jj, d - 1, r) + PENALTY1;
						}
						int L3 = INT_MAX;
						if (d + 1 < MAX_DISPARITY) {
							L3 = L.at<int>(ii, jj, d + 1, r) + PENALTY1;
						}
						int L4 = INT_MAX;
						int minn = INT_MAX; //compute the minimum to subtract it later
						for (int k = 0; k < MAX_DISPARITY; ++k) {
							int val = L.at<int>(ii, jj, k, r) + PENALTY2;
							if (val < L4) {
								L4 = val;
							}
							val = L.at<int>(ii, jj, k, r);
							if (val < minn) {
								minn = val;
							}
						}

						L.at<int>(i, j, d, r) =
							L.at<int>(i, j, d, r) + minimum(L1, L2, L3, L4) - minn; //formula (13)
					}
				}
			}
		}
	}

	return L;
}

Mat computeAggregatedCosts(Mat L, const int ROWS, const int COLS) {

	const int MAX_DISPARITY = DISPARITY_LEVEL;

	int dims[3] = { ROWS, COLS, MAX_DISPARITY };

	Mat aggregated = Mat::zeros(3, dims, CV_32S);

	for (int i = 0; i < ROWS; ++i) {
		std::cout << ((float)i / ROWS) * 100 << "%\n";
		for (int j = 0; j < COLS; ++j) {
			for (int d = 0; d < MAX_DISPARITY; ++d) {
				for (int r = 0; r < NB_PATHS; ++r) {
					aggregated.at<int>(i, j, d) += L.at<int>(i, j, d, r); //formula L14
				}
			}
		}
	}

	return aggregated;
}

//3.4
Mat_<uchar> computeDisparityMap(Mat aggregated, const int ROWS, const int COLS) {
	Mat_<uchar> disparityMap = Mat(ROWS, COLS, CV_8UC1);

	for (int i = 0; i < ROWS; ++i) {
		std::cout << ((float)i / ROWS) * 100 << "%\n";
		for (int j = 0; j < COLS; ++j) {
			int argmin = -1;
			int minDisp = INT_MAX;
			for (int d = 0; d < DISPARITY_LEVEL; ++d) {
				if (aggregated.at<int>(i, j, d) < minDisp) {
					argmin = d;
					minDisp = aggregated.at<int>(i, j, d);
				}
			}
			disparityMap.at<uchar>(i, j) = argmin;
		}
	}

	return disparityMap;
}

int main()
{
	Mat_<uchar> left, right;

	//left = imread("images/Flowers-perfect/im0_resized1.png", CV_LOAD_IMAGE_GRAYSCALE);
	//right = imread("images/Flowers-perfect/im1_resized1.png", CV_LOAD_IMAGE_GRAYSCALE);
	left = imread("images/Storage-perfect/im0_resized1.png", CV_LOAD_IMAGE_GRAYSCALE);
	right = imread("images/Storage-perfect/im1_resized1.png", CV_LOAD_IMAGE_GRAYSCALE);
	//left = imread("images/Backpack-perfect/im0_resized1.png", CV_LOAD_IMAGE_GRAYSCALE);
	//right = imread("images/Backpack-perfect/im1_resized1.png", CV_LOAD_IMAGE_GRAYSCALE);
	//left = imread("images/Playroom-perfect/im0_resized1.png", CV_LOAD_IMAGE_GRAYSCALE);
	//right = imread("images/Playroom-perfect/im1_resized1.png", CV_LOAD_IMAGE_GRAYSCALE);
	//left = imread("images/Motorcycle-perfect/im0_resized1.png", CV_LOAD_IMAGE_GRAYSCALE);
	//right = imread("images/Motorcycle-perfect/im1_resized1.png", CV_LOAD_IMAGE_GRAYSCALE);



	Mat_<int> censusLeft;
	Mat_<int> censusRight;

	censusLeft = censusTransorm(left);
	censusRight = censusTransorm(right);

	std::cout << "Calculating local costs...\n";

	//LOCAL COST BY INTENSITY
	//Mat local_cost = getCostForEachDisparity(left, right, INTENSITY);

	//LOCAL COST BY CENSUS
	Mat local_cost = getCostForEachDisparity(censusLeft, censusRight, CENSUS);

	std::cout << "\nCalculating global costs...\n";
	Mat L = dynamicProgrammingCosts(local_cost, left.rows, left.cols);

	std::cout << "\nCalculating aggregated costs...\n";
	Mat aggregated = computeAggregatedCosts(L, left.rows, left.cols);

	std::cout << "\nCalculating disparity map...\n";
	Mat_<uchar> disparityMap = computeDisparityMap(aggregated, left.rows, left.cols);

	std::cout << "\nNormalizing disparity map...\n";
	Mat_<uchar> disparityMapNormalized = Mat(left.rows, left.cols, CV_8UC1);
	cv::normalize(disparityMap, disparityMapNormalized, 0, 255, NORM_MINMAX, CV_8UC1);

	imshow("Depths", disparityMapNormalized);

	Mat_<uchar> disparityMapNormalizedBlurred = Mat(left.rows, left.cols, CV_8UC1);
	cv::medianBlur(disparityMapNormalized, disparityMapNormalizedBlurred, 3);

	imshow("Blurred", disparityMapNormalizedBlurred);
	waitKey(0);

	return 0;
}