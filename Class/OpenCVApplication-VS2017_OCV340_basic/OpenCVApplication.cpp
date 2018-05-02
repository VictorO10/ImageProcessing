// OpenCVApplication.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "common.h"
#include <queue>"
#include <random>
#include <vector>
#include <fstream>


void testOpenImage()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		imshow("image", src);
		waitKey();
	}
}

void testOpenImagesFld()
{
	char folderName[MAX_PATH];
	if (openFolderDlg(folderName) == 0)
		return;
	char fname[MAX_PATH];
	FileGetter fg(folderName, "bmp");
	while (fg.getNextAbsFile(fname))
	{
		Mat src;
		src = imread(fname);
		imshow(fg.getFoundFileName(), src);
		if (waitKey() == 27) //ESC pressed
			break;
	}
}


void testResize()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		Mat dst1, dst2;
		//without interpolation
		resizeImg(src, dst1, 320, false);
		//with interpolation
		resizeImg(src, dst2, 320, true);
		imshow("input image", src);
		imshow("resized image (without interpolation)", dst1);
		imshow("resized image (with interpolation)", dst2);
		waitKey();
	}
}


void testVideoSequence()
{
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
		cvtColor(frame, grayFrame, CV_BGR2GRAY);
		imshow("source", frame);
		imshow("gray", grayFrame);
		c = cvWaitKey(0);  // waits a key press to advance to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished\n");
			break;  //ESC pressed
		};
	}
}


void testSnap()
{
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
	Size capS = Size((int)cap.get(CV_CAP_PROP_FRAME_WIDTH),
		(int)cap.get(CV_CAP_PROP_FRAME_HEIGHT));

	// Display window
	const char* WIN_SRC = "Src"; //window for the source frame
	namedWindow(WIN_SRC, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_SRC, 0, 0);

	const char* WIN_DST = "Snapped"; //window for showing the snapped frame
	namedWindow(WIN_DST, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_DST, capS.width + 10, 0);

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

		c = cvWaitKey(10);  // waits a key press to advance to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished");
			break;  //ESC pressed
		}
		if (c == 115) { //'s' pressed - snapp the image to a file
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
	//More examples: http://o...content-available-to-author-only...t.com/2014/01/detect-mouse-clicks-and-moves-on-image.html
	Mat* src = (Mat*)param;
	if (event == CV_EVENT_LBUTTONDOWN)
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

/* Histogram display function - display a histogram using bars (simlilar to L3 / PI)
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

void negative_image() {
	Mat img = imread("Images/cameraman.bmp",
		CV_LOAD_IMAGE_GRAYSCALE);
	for (int i = 0; i<img.rows; i++) {
		for (int j = 0; j<img.cols; j++) {
			img.at<uchar>(i, j) = 255 - img.at<uchar>(i, j);
		}
	}
	imshow("negative image", img);
	waitKey(0);
}

void change_gray_levels(int factor) {
	Mat img = imread("Images/Lena_24bits_gray.bmp",
		CV_LOAD_IMAGE_GRAYSCALE);

	for (int i = 0; i < img.rows; ++i) {
		for (int j = 0; j < img.cols; ++j) {
			uchar pixel = img.at<uchar>(i, j);
			int newPixel = pixel + factor;
			if (newPixel < 0) {
				newPixel = 0;
			}
			else if (newPixel > 255) {
				newPixel = 255;
			}
			img.at<uchar>(i, j) = newPixel;
		}
	}
	imshow("change gray levels", img);
	waitKey(0);
}

void change_gray_levels_multiplic(int factor) {
	Mat img = imread("Images/Lena_24bits_gray.bmp",
		CV_LOAD_IMAGE_GRAYSCALE);

	for (int i = 0; i < img.rows; ++i) {
		for (int j = 0; j < img.cols; ++j) {
			uchar pixel = img.at<uchar>(i, j);
			int newPixel = pixel * factor;
			if (newPixel < 0) {
				newPixel = 0;
			}
			else if (newPixel > 255) {
				newPixel = 255;
			}
			img.at<uchar>(i, j) = newPixel;
		}
	}
	imshow("change gray levels", img);
	imwrite("Images/lena_changeGL.bmp", img);
	waitKey(0);
}

void create_squares() {
	Mat img(256, 256, CV_8UC3);
	int halfRow = img.rows / 2;
	int halfCol = img.cols / 2;

	for (int i = 0; i < img.rows; ++i) {
		for (int j = 0; j < img.cols; ++j) {
			Vec3b pixel;
			if (i < halfRow && j < halfCol) {
				pixel[2] = pixel[1] = pixel[0] = 255;
			}
			if (i < halfRow && j >= halfCol) {
				pixel[2] = 255;
				pixel[1] = pixel[0] = 0;
			}
			if (i >= halfRow && j < halfCol) {
				pixel[2] = pixel[0] = 0;
				pixel[1] = 255;
			}
			if (i >= halfRow && j >= halfCol) {
				pixel[2] = pixel[1] = 255;
				pixel[0] = 0;
			}

			img.at<Vec3b>(i, j) = pixel;
		}
	}

	imshow("squares", img);
	waitKey(0);
}

void flip_horizontally() {
	Mat img = imread("Images/Lena_24bits_gray.bmp",
		CV_LOAD_IMAGE_GRAYSCALE);

	int halfCols = img.cols / 2;

	for (int i = 0; i < img.rows; ++i) {
		for (int j = 0; j < halfCols; ++j) {
			int flipCol = img.cols - (j + 1);

			uchar aux = img.at<uchar>(i, j);
			img.at<uchar>(i, j) = img.at<uchar>(i, flipCol);
			img.at<uchar>(i, flipCol) = aux;

			//			swap(img.at<uchar>(i, j), img.at<uchar>(i, flipCol));
		}
	}

	imshow("flipped horizontally", img);
	waitKey(0);
}

void flip_vertically() {
	Mat img = imread("Images/Lena_24bits.bmp",
		CV_LOAD_IMAGE_COLOR);

	int halfRows = img.rows / 2;

	for (int i = 0; i < halfRows; ++i) {
		for (int j = 0; j < img.cols; ++j) {
			int flipRow = img.rows - (i + 1);

			Vec3b aux = img.at<Vec3b>(i, j);
			img.at<Vec3b>(i, j) = img.at<Vec3b>(flipRow, j);
			img.at<Vec3b>(flipRow, j) = aux;

			//			swap(img.at<uchar>(i, j), img.at<uchar>(i, flipCol));
		}
	}

	imshow("flipped vertically", img);
	waitKey(0);
}

void crop_center() {
	Mat imgOrig = imread("Images/Lena_24bits.bmp",
		CV_LOAD_IMAGE_COLOR);

	int croppedRows = imgOrig.rows / 4;
	int croppedCols = imgOrig.cols / 4;

	Mat imgCrop(croppedRows, croppedCols, CV_8UC3);
	for (int i = 0; i < croppedRows; ++i) {
		for (int j = 0; j < croppedCols; ++j) {
			imgCrop.at<Vec3b>(i, j) = imgOrig.at<Vec3b>(i + croppedRows, j + croppedCols);
		}
	}

	imshow("cropped image", imgCrop);
	imwrite("Images/lena_cropped.bmp", imgCrop);
	waitKey(0);
}

void image_resize(int newHeight, int newWidth) {
	Mat imgOrig = imread("Images/Lena_24bits.bmp");

	float heightRatio = imgOrig.rows / ((float)newHeight);
	float widthRatio = imgOrig.cols / ((float)newWidth);
	printf("%f %f", heightRatio, widthRatio);

	Mat imgRes(newHeight, newWidth, CV_8UC3);
	for (int i = 0; i < newHeight; ++i) {
		for (int j = 0; j < newWidth; ++j) {
			int oldi = (int)round(i * heightRatio), oldj = (int)round(j * widthRatio);
			if (oldi >= imgOrig.rows) oldi = imgOrig.rows - 1;
			if (oldj >= imgOrig.cols) oldj = imgOrig.cols - 1;

			imgRes.at<Vec3b>(i, j) = imgOrig.at<Vec3b>(oldi, oldj);
		}
	}

	imshow("resized image", imgRes);
	waitKey(0);
}

void splitRGB() {
	Mat imgOrig = imread("Images/Lena_24bits.bmp");
	int rows, cols;
	rows = imgOrig.rows;
	cols = imgOrig.cols;
	Mat imgRed(rows, cols, CV_8UC1);
	Mat imgGreen(rows, cols, CV_8UC1);
	Mat imgBlue(rows, cols, CV_8UC1);

	for (int i = 0; i < rows; ++i) {
		for (int j = 0; j < cols; ++j) {
			imgRed.at<uchar>(i, j) = imgOrig.at<Vec3b>(i, j)[2];
			imgGreen.at<uchar>(i, j) = imgOrig.at<Vec3b>(i, j)[1];
			imgBlue.at<uchar>(i, j) = imgOrig.at<Vec3b>(i, j)[0];
		}
	}

	imshow("Red channel", imgRed);
	imshow("Green channel", imgGreen);
	imshow("Blue channel", imgBlue);

	waitKey(0);
}

void RGBToGrayscale() {
	Mat imgOrig = imread("Images/Lena_24bits.bmp");
	int rows, cols;
	rows = imgOrig.rows;
	cols = imgOrig.cols;

	Mat imgGrayscale(rows, cols, CV_8UC1);

	for (int i = 0; i < rows; ++i) {
		for (int j = 0; j < cols; ++j) {
			Vec3b colors = imgOrig.at<Vec3b>(i, j);
			imgGrayscale.at<uchar>(i, j) =
				(
					colors[0] +
					colors[1] +
					colors[2]
					) / 3;
		}
	}

	imshow("Grayscale", imgGrayscale);

	waitKey(0);
}

void GrayscaleToBW(uchar treshold) {
	//Mat imgOrig = imread("Images/Lena_24bits.bmp", 
	//	CV_LOAD_IMAGE_GRAYSCALE);
	Mat imgOrig = imread("Images/kids.bmp",
		CV_LOAD_IMAGE_GRAYSCALE);

	for (int i = 0; i < imgOrig.rows; ++i) {
		for (int j = 0; j < imgOrig.cols; ++j) {
			if (imgOrig.at<uchar>(i, j) < treshold) {
				imgOrig.at<uchar>(i, j) = 0;
			}
			else {
				imgOrig.at<uchar>(i, j) = 255;
			}
		}
	}

	imshow("BW", imgOrig);

	waitKey(0);
}

void RGBToHSV() {
	Mat imgOrig = imread("Images/Lena_24bits.bmp",
		CV_LOAD_IMAGE_COLOR);

	Mat imgNormH(imgOrig.rows, imgOrig.cols, CV_8UC1);
	Mat imgNormS(imgOrig.rows, imgOrig.cols, CV_8UC1);
	Mat imgNormV(imgOrig.rows, imgOrig.cols, CV_8UC1);

	for (int i = 0; i < imgOrig.rows; ++i) {
		for (int j = 0; j < imgOrig.cols; ++j) {
			float r = ((float)imgOrig.at<Vec3b>(i, j)[2]) / 255;
			float g = ((float)imgOrig.at<Vec3b>(i, j)[1]) / 255;
			float b = ((float)imgOrig.at<Vec3b>(i, j)[0]) / 255;

			float M = max(max(r, g), b);
			float m = min(min(r, g), b);
			float C = M - m;

			float V, S, H;

			//Value
			V = M;

			//Saturation
			if (V != 0) {
				S = C / V;
			}
			else { //grayscale
				S = 0;
			}

			//Hue
			if (C != 0) {
				if (M == r) {
					H = 60 * (g - b) / C;
				}
				if (M == g) {
					H = 120 + 60 * (b - r) / C;
				}
				if (M == b) {
					H = 240 + 60 * (r - g) / C;
				}
			}
			else { //grayscale
				H = 0;
			}
			if (H < 360) {
				H += 360;
			}

			imgNormH.at<uchar>(i, j) = (uchar)((H * 255) / 360);
			imgNormS.at<uchar>(i, j) = (uchar)(S * 255);
			imgNormV.at<uchar>(i, j) = (uchar)(V * 255);
		}
	}

	imshow("Hue", imgNormH);
	imshow("Saturation", imgNormS);
	imshow("Value", imgNormV);

	waitKey(0);
}

void colorDetect(int th_red_low, int th_red_high) {
	Mat bgrImg = imread("Images/traffic_sign.png", CV_LOAD_IMAGE_COLOR);
	Mat hsvImg;

cv:cvtColor(bgrImg, hsvImg, CV_BGR2HSV);
	Mat hsv_channels[3];
	cv::split(hsvImg, hsv_channels);

	Mat mask;
	cv::inRange(hsv_channels[0], th_red_low, th_red_high, mask);

	imshow("Orgin Stop Sign", bgrImg);
	imshow("Stop sign", mask);

	waitKey(0);
}

/// Global Variables
const int alpha_slider_max = 180;
int th_blue_low, th_blue_high;
int th_red_low, th_red_high;
int th_green_low, th_green_high;
double alpha;
double beta;
Mat maskRed;
Mat maskBlue;
Mat maskGreen;
Mat hsv_channels[3];

void on_trackbar(int, void*)
{
	cv::inRange(hsv_channels[0], th_red_low, th_red_high, maskRed);
	cv::inRange(hsv_channels[1], th_green_low, th_green_high, maskGreen);
	cv::inRange(hsv_channels[2], th_blue_low, th_blue_high, maskBlue);

	imshow("Red", maskRed);
	imshow("Green", maskGreen);
	imshow("Blue", maskBlue);
}

void colorSelect() {
	Mat bgrImg = imread("Images/traffic_sign.png", CV_LOAD_IMAGE_COLOR);
	Mat hsvImg;

cv:cvtColor(bgrImg, hsvImg, CV_BGR2HSV);

	cv::split(hsvImg, hsv_channels);

	const int alpha_slider_max = 180;

	namedWindow("Threshold Trackbars", 1);
	createTrackbar("Red Threshold Low", "Threshold Trackbars", &th_red_low, alpha_slider_max, on_trackbar);
	createTrackbar("Red Threshold High", "Threshold Trackbars", &th_red_high, alpha_slider_max, on_trackbar);

	createTrackbar("Blue Threshold Low", "Threshold Trackbars", &th_blue_low, alpha_slider_max, on_trackbar);
	createTrackbar("Blue Threshold High", "Threshold Trackbars", &th_blue_high, alpha_slider_max, on_trackbar);

	createTrackbar("Green Threshold Low", "Threshold Trackbars", &th_green_low, alpha_slider_max, on_trackbar);
	createTrackbar("Green Threshold High", "Threshold Trackbars", &th_green_high, alpha_slider_max, on_trackbar);

	//Mat maskGreen;
	//
	//Mat maskBlue;
	//

	//imshow("Orgin Stop Sign", bgrImg);
	//imshow("Stop sign", mask);

	waitKey(0);
}

int computeArea(Mat* img, Vec3b colors) { //compute area for a specific color

	int area = 0;

	for (int i = 0; i < img->rows; ++i) {
		for (int j = 0; j < img->cols; ++j) {
			if (colors == img->at<Vec3b>(i, j)) {
				area++;
			}
		}
	}

	return area;
}

Point computeCoM(Mat* img, Vec3b colors, int area) { //Center of Mass

	double rowCenter = 0, colCenter = 0;

	for (int i = 0; i < img->rows; ++i) {
		for (int j = 0; j < img->cols; ++j) {
			if (colors == img->at<Vec3b>(i, j)) {
				rowCenter += i;
				colCenter += j;
			}
		}
	}

	rowCenter /= area;
	colCenter /= area;

	Point coM((int)colCenter, (int)rowCenter);

	return coM;
}

int computeAoE(Mat *img, Vec3b colors, Point coM) { //Angle of Elongation

	float nominator = 0;
	float denominator = 0;

	for (int i = 0; i < img->rows; ++i) {
		for (int j = 0; j < img->cols; ++j) {
			if (colors == img->at<Vec3b>(i, j)) {
				nominator += (i - coM.y) * (j - coM.x);
				denominator += (j - coM.x)*(j - coM.x) - (i - coM.y)*(i - coM.y);
			}
		}
	}

	nominator *= 2;

	float twoAlpha = atan2(nominator, denominator);

	float aoE = (twoAlpha / 2) * (180 / CV_PI);

	if (aoE < 0) aoE += 180;

	return (int)aoE;
}

int computePerimeter(Mat *img, Vec3b colors) {

	int perimeter = 0;

	for (int i = 0; i < img->rows; ++i) {
		for (int j = 0; j < img->cols; ++j) {
			if (colors == img->at<Vec3b>(i, j)) {
				bool edge = false;

				for (int ki = -1; ki <= 1; ++ki) {
					for (int kj = -1; kj <= 1; ++kj) {
						int ii = i + ki, jj = j + kj;
						if (img->at<Vec3b>(ii, jj) != colors) {
							edge = true;
						}
					}
				}

				if (edge) {
					perimeter++;
				}
			}
		}
	}

	return (int)(perimeter * (CV_PI / 4)); //multiplying with pi/4 is a correction
}

float computeThinnessRatio(int perimeter, int area) {
	return (4 * CV_PI) * ((float)(area) / (perimeter*perimeter));
}

float computeAspectRatio(Mat *img, Vec3b colors) {
	int cmax = -1, rmax = -1, cmin = POS_INFINITY, rmin = POS_INFINITY;

	for (int i = 0; i < img->rows; ++i) {
		for (int j = 0; j < img->cols; ++j) {
			if (colors == img->at<Vec3b>(i, j)) {
				if (i > rmax) rmax = i;
				if (i < rmin) rmin = i;

				if (j > cmax) cmax = j;
				if (j < cmin) cmin = j;
			}
		}
	}

	return ((float)cmax - cmin + 1) / (rmax - rmin + 1);
}

void drawStuff(Mat *img, Vec3b colors, Point coM, int aoE) {

	Mat imgC; //img Countour
	img->copyTo(imgC);
	//draw the countour points
	for (int i = 0; i < imgC.rows; ++i) {
		for (int j = 0; j < imgC.cols; ++j) {
			if (colors == img->at<Vec3b>(i, j)) {
				bool edge = false;

				for (int ki = -1; ki <= 1; ++ki) {
					for (int kj = -1; kj <= 1; ++kj) {
						int ii = i + ki, jj = j + kj;
						if (img->at<Vec3b>(ii, jj) != colors) {
							imgC.at<Vec3b>(ii, jj) = Vec3b(0, 0, 0);
						}
					}
				}

			}
		}
	}

	//center of mass
	circle(imgC, coM, 8, Scalar(0, 0, 0), 2);
	imgC.at<Vec3b>(coM.y, coM.x) = Vec3b(0, 0, 0);

	//line through center of mass with angle of elongation
	float aoERad = ((aoE)* CV_PI) / 180;
	float slope = tan(aoERad);
	float n = (float)(coM.y - slope*coM.x);

	int pStartY = coM.y - 25;
	int pStartX = (int)((slope * (pStartY)) + n);
	int pEndY = coM.y + 25;
	int pEndX = (int)((slope * (pEndY)) + n);

	Point pEnd(pEndY, pEndX), pStart(pStartY, pStartX);
	line(imgC, coM, pEnd, Scalar(0, 0, 0), 2);


	imshow("Selected Object", imgC);

	waitKey(0);
}

void computeProperties(int event, int x, int y, int flags, void* param) {

	Mat* imgRead = (Mat*)param;
	int R = imgRead->at<Vec3b>(y, x)[2];
	int G = imgRead->at<Vec3b>(y, x)[1];
	int B = imgRead->at<Vec3b>(y, x)[0];

	if (event == EVENT_LBUTTONDOWN) {
		printf("%d %d, color: (%d, %d, %d)\n", y, x, R, G, B);

		int area = computeArea(imgRead, imgRead->at<Vec3b>(y, x));
		printf("Area: %d\n", area);

		Point coM = computeCoM(imgRead, imgRead->at<Vec3b>(y, x), area); //center of Mass
		printf("Center of mass: %d %d\n", coM.y, coM.x);

		int aoE = computeAoE(imgRead, imgRead->at<Vec3b>(y, x), coM); //angle of elongation
		printf("angle of elongation: %d\n", aoE);

		int perimeter = computePerimeter(imgRead, imgRead->at<Vec3b>(y, x));
		printf("perimeter: %d\n", perimeter);

		float thinnessRatio = computeThinnessRatio(perimeter, area);
		printf("thinness Ratio: %.2f\n", thinnessRatio);

		float aspectRatio = computeAspectRatio(imgRead, imgRead->at<Vec3b>(y, x));
		printf("aspect Ratio: %.2f\n", aspectRatio);

		drawStuff(imgRead, imgRead->at<Vec3b>(y, x), coM, aoE);
	}

}

void objectProperties() {
	Mat imgRead = imread("Images/MultipleObjects/geometrical_features.bmp");

	//Create a window
	namedWindow("Objects", 1);

	//set the callback function for any mouse event
	setMouseCallback("Objects", computeProperties, &imgRead);

	//show the image
	imshow("Objects", imgRead);

	// Wait until user press some key
	waitKey(0);
}

void genColImg(Mat labels) {
	Mat colors(labels.rows, labels.cols, CV_8UC3);
	std::default_random_engine gen;
	std::uniform_int_distribution<int> d(0, 255);

	//get the max label
	int labelsMax = -1;
	for (int i = 0; i < labels.rows; ++i) {
		for (int j = 0; j < labels.cols; ++j) {
			if (labels.at<int>(i, j) > labelsMax) {
				labelsMax = labels.at<int>(i, j);
			}
		}
	}

	std::vector<Vec3b> labelColors;
	labelColors.push_back(Vec3b(255, 255, 255));
	for (int i = 1; i <= labelsMax; ++i) {
		labelColors.push_back(Vec3b(d(gen), d(gen), d(gen))); //random color for each label
	}

	for (int i = 0; i < labels.rows; ++i) {
		for (int j = 0; j < labels.cols; ++j) {
			colors.at<Vec3b>(i, j) = labelColors[labels.at<int>(i, j)];
		}
	}

	imshow("Labeled Objects", colors);
	waitKey(0);
}

bool inBounds(int i, int j, int rows, int cols) {
	return (i >= 0 && j >= 0 && i < rows && j < cols);
}

void BFS() {
	Mat imgRead = imread("Images/labeling/labeling2.bmp", CV_LOAD_IMAGE_GRAYSCALE);

	int label = 0;
	Mat labels = Mat::zeros(imgRead.rows, imgRead.cols, CV_32SC1);
	//printf("rows = %d cols = %d\nimgRead: rows = %d cols = %d", labels.rows, labels.cols, imgRead.rows, imgRead.cols);
	//scanf("%d", &label);
	std::queue<Point2i> Q;

	int kernel = 1;
	for (int i = 0; i < imgRead.rows; ++i) {
		for (int j = 0; j < imgRead.cols; ++j) {
			if (imgRead.at<uchar>(i, j) == 0 && labels.at<int>(i, j) == 0) {
				label++;
				labels.at<int>(i, j) = label;
				Q.push( Point2i(i, j) );

				while (!Q.empty()) {
					Point2i q = Q.front();
					Q.pop();
					for (int ii = -kernel; ii <= kernel; ++ii) {
						for (int jj = -kernel; jj <= kernel; ++jj) {
							int iii = q.x + ii, jjj = q.y + jj;
							if (inBounds(iii, jjj, imgRead.rows, imgRead.cols)) {
								if (imgRead.at<uchar>(iii, jjj) == 0 && labels.at<int>(iii, jjj) == 0) {
									labels.at<int>(iii, jjj) = label;
									Q.push(Point2i(iii, jjj));
								}
							}
						}
					}
				}

			}
		}
	}

	genColImg(labels);

}

void twoPass() {
	Mat imgRead = imread("Images/labeling/labeling1.bmp", CV_LOAD_IMAGE_GRAYSCALE);

	int label = 0;
	Mat labels = Mat::zeros(imgRead.rows, imgRead.cols, CV_32SC1);
	std::vector<std::vector<int>> edges;

	int di[] = { -1, -1, -1, 0 };
	int dj[] = { -1, 0, +1, -1 };

	for (int i = 0; i < imgRead.rows; ++i) {
		for (int j = 0; j < imgRead.cols; ++j) {
			if (imgRead.at<uchar>(i, j) == 0 && labels.at<int>(i, j) == 0) {
				std::vector<int> L;
				for (int d = 0; d < 4; ++d) {
					int ii = i + di[d], jj = j + dj[d];
					if (inBounds(ii, jj, imgRead.rows, imgRead.cols)) {
						if (labels.at<int>(ii, jj) > 0) {
							L.push_back(labels.at<int>(ii, jj));
						}
					}
				}
				if (L.size() == 0) {
					label++;
					labels.at<int>(i, j) = label;
					edges.resize(label + 1);
				}
				else {
					int x = *std::min_element(std::begin(L), std::end(L));
					labels.at<int>(i, j) = x;
					for (int k = 0; k < L.size(); ++k) {
						int y = L[k];
						if (y != x) {
							edges[x].push_back(y);
							edges[y].push_back(x);
						}
					}
				}

			}
		}
	}

	std::cout << "AM AJUNS\n";

	int newLabel = 0;
	std::vector<int> newLabels(label + 1, 0);
	for (int i = 1; i <= label; ++i) {
		if (newLabels[i] == 0) {
			newLabel++;
			std::queue<int> Q;
			newLabels[i] = newLabel;
			Q.push(i);
			while (!Q.empty()) {
				int x = Q.front();
				Q.pop();
				for (int j = 0; j < edges[x].size(); ++j) {
					int y = edges[x][j];
					if (newLabels[y] == 0) {
						newLabels[y] = newLabel;
						Q.push(y);
					}
				}
			}
		}
	}

	for (int i = 0; i < imgRead.rows; ++i) {
		for (int j = 0; j < imgRead.cols; ++j) {
			labels.at<int>(i, j) = newLabels[labels.at<int>(i, j)];
		}
	}

	genColImg(labels);
}

void drawCountour() {
	Mat img = imread("Images/countour/triangle_up.bmp", CV_LOAD_IMAGE_GRAYSCALE);
	std::vector<Point2d> border;
	std::vector<int> chainCode;
	std::vector<int> derivCode;

	int di[] = { 0, -1, -1, -1, 0, +1, +1, +1 };
	int dj[] = { +1, +1, 0, -1, -1, -1, 0, +1 };

	bool done = false;
	for (int i = 0; i < img.rows && !done; ++i) {
		for (int j = 0; j < img.cols && !done; ++j) {
			if (img.at<uchar>(i, j) == 0) { //black pixel
				border.push_back(Point2d(i, j));
				int dir = 7;
				do {
					int ii = i + di[dir];
					int jj = j + dj[dir];
					if (img.at<uchar>(ii, jj) == 0) {
						chainCode.push_back(dir);
						border.push_back(Point2d(ii, jj));
						i = ii;
						j = jj;

						if (dir % 2 == 0) dir = (dir + 7) % 8;
						else if (dir % 2 == 1) dir = (dir + 6) % 8;
					}
					else {
						dir = (dir + 1) % 8;
					}

					if (border.size() >= 4) {
						std::cout << border.at(border.size() - 1) << " " << border.at(1) << "\n";
						std::cout << border.at(border.size() - 2) << " " << border.at(0) << "\n";
					}
				} while (border.size() < 4 ||
					(border.at(border.size() - 1) != border.at(1)) ||
					(border.at(border.size() - 2) != border.at(0)));
				done = true;
			}
		}
	}
	//remove last 2, which are actually the first 2
	border.pop_back();
	border.pop_back();

	chainCode.pop_back();
	chainCode.pop_back();

	Mat contour = Mat::zeros(img.rows, img.cols, CV_8UC1);

	for (int i = 0; i < border.size(); ++i) {
		contour.at<uchar>(border.at(i).x, border.at(i).y) = 255;
	}

	std::cout << "START POSITION: (" << border.at(0).y << ", " << border.at(0).x << ")\n";
	std::cout << "CHAIN CODE(" << chainCode.size() << "):\n";
	for (int i = 0; i < chainCode.size(); ++i) {
		std::cout << chainCode.at(i) << " ";
		if (i > 0) {
			derivCode.push_back((chainCode.at(i) - chainCode.at(i - 1) + 8) % 8);
		}
	}
	std::cout << "\n";

	std::cout << "DERIV CODE:\n";
	for (int i = 0; i < derivCode.size(); ++i) {
		std::cout << derivCode.at(i) << " ";
	}


	imshow("imread", img);
	imshow("contour", contour);
	waitKey(0);

}

void reconstructContour() {
	std::ifstream f("Images/countour/reconstruct.txt");
	if (!f.is_open()) {
		std::cout << "File not found!\n";
		waitKey(0);
		return;
	}

	Mat contour = Mat::zeros(700, 700, CV_8UC1);

	int crtx, crty;
	f >> crtx;
	f >> crty;
	contour.at<uchar>(crtx, crty) = 255;

	int nbCodes;
	f >> nbCodes;

	int di[] = { 0, -1, -1, -1, 0, +1, +1, +1 };
	int dj[] = { +1, +1, 0, -1, -1, -1, 0, +1 };

	for (int i = 0; i < nbCodes; ++i) {
		int dir;
		f >> dir;
		int ii = crtx + di[dir];
		int jj = crty + dj[dir];

		contour.at<uchar>(ii, jj) = 255;

		crtx = ii;
		crty = jj;
	}

	f.close();
	imshow("contour", contour);
	waitKey(0);	
}

Mat dilate(Mat img) {
	Mat dilated = Mat::zeros(img.rows, img.cols, CV_8UC1);
	dilated = 255 - dilated; //make image white (255 - white, 0 - black)

	for (int i = 0; i < img.rows; ++i) {
		for (int j = 0; j < img.cols; ++j) {
			if (img.at<uchar>(i, j) == 0) { //make neighbours black, including itself
				for (int di = -1; di <= 1; ++di) {
					for (int dj = -1; dj <= 1; ++dj) {
						int ii = i + di;
						int jj = j + dj;
						if (inBounds(ii, jj, img.rows, img.cols)) {
							dilated.at<uchar>(ii, jj) = 0;
						}
					}
				}
			}
		}
	}

	return dilated;
}

Mat erode(Mat img) {
	Mat eroded = Mat::zeros(img.rows, img.cols, CV_8UC1);
	eroded = 255 - eroded; //make image white (255 - white, 0 - black)

	for (int i = 0; i < img.rows; ++i) {
		for (int j = 0; j < img.cols; ++j) {
			if (img.at<uchar>(i, j) == 0) { //make it white if one of the neighs is white
				boolean isBoundary = false;
				for (int di = -1; di <= 1; ++di) {
					for (int dj = -1; dj <= 1; ++dj) {
						int ii = i + di;
						int jj = j + dj;
						if (inBounds(ii, jj, img.rows, img.cols)) {
							if (img.at<uchar>(ii, jj) == 255) {
								isBoundary = true;
							}
						}
					}
				}
				if (!isBoundary) {
					eroded.at<uchar>(i, j) = 0;
				}
			}
		}
	}

	return eroded;
}

Mat open(Mat img) {
	return dilate(erode(img));
}

Mat close(Mat img) {
	return erode(dilate(img));
}

Mat bound(Mat img) {
	Mat eroded = erode(img);
	return (255 - eroded) + img;
}

Mat dilate4Neigh(Mat img) {
	Mat dilated = Mat::zeros(img.rows, img.cols, CV_8UC1);
	dilated = 255 - dilated; //make image white (255 - white, 0 - black)

	for (int i = 0; i < img.rows; ++i) {
		for (int j = 0; j < img.cols; ++j) {
			if (img.at<uchar>(i, j) == 0) { //make neighbours black, including itself
				for (int di = -1; di <= 1; ++di) {
					for (int dj = -1; dj <= 1; ++dj) {
						if (abs(di) + abs(dj) == 1) {
							int ii = i + di;
							int jj = j + dj;
							if (inBounds(ii, jj, img.rows, img.cols)) {
								dilated.at<uchar>(ii, jj) = 0;
							}
						}
					}
				}
			}
		}
	}

	return dilated;
}

Mat reunion(Mat a, Mat b) {

	Mat reun = Mat::zeros(a.rows, a.cols, CV_8UC1);
	reun = 255 - reun; //all white

	for (int i = 0; i < a.rows; ++i) {
		for (int j = 0; j < a.cols; ++j) {
			if (a.at<uchar>(i, j) == 0 || b.at<uchar>(i, j) == 0) {
				reun.at<uchar>(i, j) = 0;
			}
		}
	}

	return reun;
}

Mat fill(Mat img) {
	Mat filled = Mat::zeros(img.rows, img.cols, CV_8UC1);
	filled = 255 - filled;
	boolean finished = false;
	//find starting point
	for (int i = 0; i < img.rows && !finished; ++i) {
		for (int j = 0; j < img.rows && !finished; ++j) {
			int ii = i - 1;
			int jj = j;
			if (inBounds(ii, jj, img.rows, img.cols)) {
				if (img.at<uchar>(ii, jj) == 0) {
					ii = i;
					jj = j - 1;
					if (inBounds(ii, jj, img.rows, img.cols)) {
						if (img.at<uchar>(ii, jj) == 0) {
							ii = i;
							jj = j + 1;
							if (inBounds(ii, jj, img.rows, img.cols)) {
								if (img.at<uchar>(ii, jj) == 255) {
									ii = i + 1;
									jj = j;

									if (inBounds(ii, jj, img.rows, img.cols)) {
										if (img.at<uchar>(ii, jj) == 255) {
											ii = i;
											jj = j + 1;

											filled.at<uchar>(i, j) = 0; //starting point
											finished = true;
										}
									}
								}
							}
						}
					}
				}
			}
		}
	}

	Mat imgC = 255 - img; //complement of img
	Mat prevFilled = filled;
	do{
		prevFilled = filled;
		filled = dilate(filled);
		filled = filled + imgC;	
	} while (! cv::countNonZero(filled!=prevFilled) == 0);

	filled = reunion(img, filled);

	return filled;
}

int* computeHistogram(Mat img) {

	int *hist = new int[256];
	memset(hist, 0, sizeof(int) * 256);
	int maxVal = -1;

	for (int i = 0; i < img.rows; ++i) {
		for (int j = 0; j < img.cols; ++j) {
			hist[img.at<uchar>(i, j)] ++;
			if (hist[img.at<uchar>(i, j)] > maxVal) {
				maxVal = hist[img.at<uchar>(i, j)];
			}
		}
	}

	return hist;
}

float computeMean(Mat img) {

	int sum = 0;

	for (int i = 0; i < img.rows; ++i) {
		for (int j = 0; j < img.cols; ++j) {
			sum += img.at<uchar>(i, j);
		}
	}

	float mean = ((float)sum) / (img.rows * img.cols);

	std::cout << "Mean: " << mean << std::endl;

	return mean;
}

//standard deviation
float computeSD() {
	Mat img = imread("Images/statistical_prop/balloons.bmp", CV_LOAD_IMAGE_GRAYSCALE);

	float mean = computeMean(img);

	float sum = 0;

	for (int i = 0; i < img.rows; ++i) {
		for (int j = 0; j < img.cols; ++j) {
			sum += (img.at<uchar>(i, j) - mean) * (img.at<uchar>(i, j) - mean);
		}
	}

	float sd = sqrt(((float)sum) / (img.rows * img.cols));

	std::cout << "Standard Deviation: " << sd << std::endl;
	getchar();
	getchar();
	

	return sd;
}

void thAlgo() {
	Mat img = imread("Images/statistical_prop/eight.bmp", CV_LOAD_IMAGE_GRAYSCALE);

	int *histo = computeHistogram(img);

	int thC, iMax = -1, iMin = 99999;

	for (int i = 0; i < img.rows; ++i) {
		for (int j = 0; j < img.cols; ++j) {
			if (img.at<uchar>(i, j) > iMax) {
				iMax = img.at<uchar>(i, j);
			}
			if (img.at<uchar>(i, j) < iMin) {
				iMin = img.at<uchar>(i, j);
			}
		}
	}

	thC = (iMax + iMin) / 2; //current threshold
	int thL; //last Threshold

	float err = 5;
	float mean1, mean2;
	int N; //nb of pixels below/higher than the th

	do {
		mean1 = 0; mean2 = 0; N = 0;
		for (int i = 0; i <= thC; ++i) {
			mean1 += (i * histo[i]);
			N += histo[i];
		}
		mean1 /= N;

		N = 0;
		for (int i = thC + 1; i <= iMax; ++i) {
			if (i <= iMax) {
				mean2 += (i * histo[i]);
				N+=histo[i];
			}
		}
		mean2 /= N;

		thL = thC;
		thC = (mean1 + mean2) / 2;

	} while (thC - thL > err);

	Mat imgTh(img);

	for (int i = 0; i < imgTh.rows; ++i) {
		for (int j = 0; j < imgTh.cols; ++j) {
			if (img.at<uchar>(i, j) <= thC) {
				imgTh.at<uchar>(i, j) = 0;
			}
			else {
				imgTh.at<uchar>(i, j) = 255;
			}
		}
	}

	imshow("Th img", imgTh);

	waitKey(0);
}

void histoStrShr(int outMin, int outMax) {
	Mat img = imread("Images/statistical_prop/Hawkes_Bay_NZ.bmp", CV_LOAD_IMAGE_GRAYSCALE);
	showHistogram("histoBefore", computeHistogram(img), 255, 500);

	int inMin = 9999, inMax = -1;

	for (int i = 0; i < img.rows; ++i) {
		for (int j = 0; j < img.cols; ++j) {
			if (img.at<uchar>(i, j) > inMax) {
				inMax = img.at<uchar>(i, j);
			}
			if (img.at<uchar>(i, j) < inMin) {
				inMin = img.at<uchar>(i, j);
			}
		}
	}

	float factor = (outMax - outMin) / (inMax - inMin);


	img = outMin + (img - inMin) * factor;

	imshow("histo str/shr", img);
	
	showHistogram("histoAfter", computeHistogram(img), 255, 500);

	waitKey(0);
}

void gammaCorrection(float gamma) {
	Mat img = imread("Images/statistical_prop/wilderness.bmp", CV_LOAD_IMAGE_GRAYSCALE);
	showHistogram("histoBefore", computeHistogram(img), 255, 500);

	for (int i = 0; i < img.rows; ++i) {
		for (int j = 0; j < img.cols; ++j) {
			int val = 255 * pow((img.at<uchar>(i, j) / 255.0), gamma);
			if (val > 255) {
				img.at<uchar>(i, j) = 255;
			}
			else if (val < 0) {
				img.at<uchar>(i, j) = 0;
			}
			else {
				img.at<uchar>(i, j) = val;
			}
		}
	}

	showHistogram("histoAfter", computeHistogram(img), 255, 500);
	imshow("gammaCorrection", img);

	waitKey(0);
}

void histoSlide(int offset) {
	Mat img = imread("Images/statistical_prop/wilderness.bmp", CV_LOAD_IMAGE_GRAYSCALE);
	showHistogram("histoBefore", computeHistogram(img), 255, 500);

	for (int i = 0; i < img.rows; ++i) {
		for (int j = 0; j < img.cols; ++j) {
			int val = (img.at<uchar>(i, j) + offset);
			if (val > 255) {
				img.at<uchar>(i, j) = 255;
			}
			else if (val < 0) {
				img.at<uchar>(i, j) = 0;
			}
			else {
				img.at<uchar>(i, j) = val;
			}
		}
	}

	showHistogram("histoAfter", computeHistogram(img), 255, 500);
	imshow("histoSlide", img);

	waitKey(0);
}

float* computePDF(Mat img) {
	int* histoInt = computeHistogram(img);
	float* histoFloat = new float[256];

	int M = img.rows * img.cols; //nb of pixels

	for (int i = 0; i < 256; ++i) {
		histoFloat[i] = (float)histoInt[i] / (float)M;
	}

//	showHistogram("PDF", histo, 256, 500);

	return histoFloat;
}

float* computeCPDF(Mat img) {

	float* histo = computePDF(img);

	for (int i = 1; i < 256; ++i) {
		histo[i] = histo[i - 1] + histo[i];
	}

	return histo;
}

void histoEq() {
	Mat img = imread("Images/statistical_prop/Hawkes_Bay_NZ.bmp", CV_LOAD_IMAGE_GRAYSCALE);

	float* histo = computeCPDF(img);

	for (int i = 0; i < img.rows; ++i) {
		for (int j = 0; j < img.cols; ++j) {
			img.at<uchar>(i, j) = (int)(255 * histo[img.at<uchar>(i, j)]);
		}
	}

	imshow("Equalization", img);

	waitKey(0);
}

void convolve(Mat src, Mat dest, Mat filter, Point center) {

}

void convolution(Mat_<float> &filter, Mat_<uchar> &img, Mat_<uchar> &output) {

	output.create(img.size());
	memcpy(output.data, img.data, img.rows * img.cols * sizeof(uchar));

	int scalingCoeff = 1;
	int additionFactor = 0;
	bool lowPass = true;
	int sPos = 0;
	int sNeg = 0;


	//TODO: decide if the filter is low pass or high pass and compute the scaling coefficient and the addition factor
	// low pass if all elements >= 0
	// high pass has elements < 0
	for (int i = 0; i < filter.rows; ++i) {
		for (int j = 0; j < filter.cols; ++j) {
			if (filter.at<int>(i, j) < 0) {
				lowPass = false;
			}
		}
	}
	

	// compute scaling coefficient and addition factor for low pass and high pass
	// low pass: additionFactor = 0, scalingCoeff = sum of all elements
	// high pass: formula 9.20
	if (lowPass) {
		scalingCoeff = 0;
		additionFactor = 0;
		for (int i = 0; i < filter.rows; ++i) {
			for (int j = 0; j < filter.cols; ++j) {
				scalingCoeff += filter.at<int>(i, j);
			}
		}
	}
	else {
		additionFactor = 255 / 2;
		for (int i = 0; i < filter.rows; ++i) {
			for (int j = 0; j < filter.cols; ++j) {
				if (filter.at<int>(i, j) > 0) {
					sPos += filter.at<int>(i, j);
				}
				else {
					sNeg -= filter.at<int>(i, j);
				}
			}
		}
		scalingCoeff = 2 * (max(sPos, sNeg));
	}


	// TODO: implement convolution operation (formula 9.2)
	// do not forget to divide with the scaling factor and add the addition factor in order to have values between [0, 255]
	//w - size of filter(w rows, cols). 2k+1 = w  => k = (w-1) / 2
	int w = filter.cols;
	int k = (w - 1) / 2;

	for (int i = k; i < img.rows - k; ++i) {
		for (int j = k; j < img.cols - k; ++j) {
			int sum = 0;
			for (int ii = -k; ii <= k; ++ii) {
				for (int jj = -k; jj <= k; ++jj) {
					sum += (filter.at<int>(ii + k, jj + k) * img.at<uchar>(i + ii, j + jj));
				}
			}
			output.at<uchar>(i - k, j - k) = ((float)sum / scalingCoeff) + additionFactor;
		}
	}

	imshow("orig", img);
	imshow("filtered", output);
	waitKey(0);
}


/*  in the frequency domain, the process of convolution simplifies to multiplication => faster than in the spatial domain
the output is simply given by F(u,v)ĂG(u,v) where F(u,v) and G(u,v) are the Fourier transforms of their respective functions
The frequency-domain representation of a signal carries information about the signal's magnitude and phase at each frequency*/

/*
The algorithm for filtering in the frequency domain is:
a) Perform the image centering transform on the original image (9.15)
b) Perform the DFT transform
c) Alter the Fourier coefficients according to the required filtering
d) Perform the IDFT transform
e) Perform the image centering transform again (this undoes the first centering transform)
*/

void centering_transform(Mat img) {
	//expects floating point image
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			img.at<float>(i, j) = ((i + j) & 1) ? -img.at<float>(i, j) : img.at<float>(i, j);
		}
	}
}

Mat generic_frequency_domain_filter(Mat src)
{
	int height = src.rows;
	int width = src.cols;

	Mat srcf;
	src.convertTo(srcf, CV_32FC1);
	// Centering transformation
	centering_transform(srcf);

	//perform forward transform with complex image output
	Mat fourier;
	dft(srcf, fourier, DFT_COMPLEX_OUTPUT);

	//split into real and imaginary channels fourier(i, j) = Re(i, j) + i * Im(i, j)
	Mat channels[] = { Mat::zeros(src.size(), CV_32F), Mat::zeros(src.size(), CV_32F) };
	split(fourier, channels);  // channels[0] = Re (real part), channels[1] = Im (imaginary part)

							   //calculate magnitude and phase in floating point images mag and phi
							   // http://www3.ncc.edu/faculty/ens/schoenf/elt115/complex.html
							   // from cartesian to polar coordinates

	Mat mag, phi;
	magnitude(channels[0], channels[1], mag);
	phase(channels[0], channels[1], phi);


	// TODO: Display here the log of magnitude (Add 1 to the magnitude to avoid log(0)) (see image 9.4e))
	// do not forget to normalize

	// TODO: Insert filtering operations here ( channels[0] = Re(DFT(I), channels[1] = Im(DFT(I) )


	//perform inverse transform and put results in dstf
	Mat dst, dstf;
	merge(channels, 2, fourier);
	dft(fourier, dstf, DFT_INVERSE | DFT_REAL_OUTPUT);

	// Inverse Centering transformation
	centering_transform(dstf);

	//normalize the result and put in the destination image
	normalize(dstf, dst, 0, 255, NORM_MINMAX, CV_8UC1);

	return dst;
}

void medianFilter(Mat img, int w) {

	int k = (w - 1) / 2;
	std::cout << k << std::endl;

	std::cout << "---MEDIAN FILTER---\n";

	Mat output(img.rows - 2*k, img.cols - 2*k, CV_8UC1);

	for (int i = k; i < img.rows - k; i++) {
		for (int j = k; j < img.cols - k; j++) {
			std::vector<uchar> orderedV;
			for (int u = 0; u < w; u++) {
				for (int v = 0; v < w; v++) {
					uchar val = img.at<uchar>(i + u - k, j + v - k);
					orderedV.push_back(val);
				}
			}
			std::sort(orderedV.begin(), orderedV.end());
			output.at<uchar>(i - k, j - k) = orderedV[(w * w) / 2];
			orderedV.clear();
		}
	}

	imshow("orig", img);
	imshow("medianFiltered", output);
	waitKey(0);
}

Mat generateGaussFilter2d(int w) {

	float sigma = (float)w / 6;

	Mat_<float> filter(w, w, CV_32FC1);

	float scale = (2 * CV_PI*sigma*sigma);
	int middle = w / 2;

	for (int i = 0; i < w; ++i) {
		for (int j = 0; j < w; ++j) {
			filter.at<float>(i, j) = 
				exp(
					-((i - middle) * (i - middle) + 
					(j - middle) * (j - middle)) /
					(2 * sigma*sigma)
				) /
				scale;
		}
	}

	return filter;
}

void applyGaussFilter2d(Mat_<uchar> img, int w) {

	Mat_<float> filter = generateGaussFilter2d(w);
	Mat_<uchar> output;

	filter2D(img, output, 8, filter);
	//convolution(filter, img, output);

	imshow("orig", img);
	imshow("gaussFiltered", output);
	waitKey(0);
}

std::vector<float> generateGaussFilter1d(int w) {
	float sigma = (float)w / 6;

	std::vector<float> filter;

	float scale = (std::sqrt(2 * CV_PI)*sigma);
	int middle = w / 2;

	for (int i = 0; i < w; ++i) {
		filter.push_back(
			exp(
				-((i - middle) * (i - middle)) /
				(2 * sigma*sigma)
			) /
			scale
		);
	}

	return filter;
}

void applyGaussFilter1d(Mat_<uchar> img, int w) {

	std::vector<float> filter = generateGaussFilter1d(w);

	int k = (w - 1 / 2);
	Mat output(img.rows - 2*k, img.cols - 2*k, CV_8UC1);
	float filter_sum = cv::sum(filter)[0];

//	std::cout << img.cols << std::endl;

	//for (int i = k; i < img.rows - k; ++i) {
	//	for (int j = k; j < img.cols - k; ++j) {
	//		float intensity = 0;
	//		for (int u = 0; u < w; ++u) {
	//		//	std::cout << i + u - k << " " << j << std::endl;
	//			intensity +=
	//				filter[u] * img.at<uchar>(i + u - k, j);
	//		}
	//		output.at<uchar>(i, j) = intensity / filter_sum;
	//		for (int u = 0; u < w; ++u) {
	//			//	std::cout << i + u - k << " " << j << std::endl;
	//			intensity +=
	//				filter[u] * img.at<uchar>(i, j + u - k);
	//		}
	//	}
	//}

	
	
	
	
	
	sepFilter2D(img, output, 8, filter, filter);

	



	imshow("orig", img);
	imshow("gaussFiltered", output);
	waitKey(0);
}

	



int main()
{
	int op;
	do
	{
		system("cls");
		destroyAllWindows();
		printf("Menu:\n");
		printf(" 1 - Open image\n");
		printf(" 2 - Open BMP images from folder\n");
		printf(" 3 - Resize image\n");
		printf(" 4 - Process video\n");
		printf(" 5 - Snap frame from live video\n");
		printf(" 6 - Mouse callback demo\n");
		printf(" 7 - Test negative image\n");
		printf(" 8 - Change gray levels addition\n");
		printf(" 9 - Change gray levels multiplicative\n");
		printf(" 10 - Create squares\n");
		printf(" 11 - Horizontal flip\n");
		printf(" 12 - Vertical flip\n");
		printf(" 13 - Crop center\n");
		printf(" 14 - Image resize\n");
		printf(" 15 - Split RGB\n");
		printf(" 16 - RGB to Grayscale\n");
		printf(" 17 - Grayscale to Black and White\n");
		printf(" 18 - RGB to HSV\n");
		printf(" 19 - Detect based on color\n");
		printf(" 20 - Select color\n");
		printf(" 21 - Object properties\n");
		printf(" 22 - Label BFS\n");
		printf(" 23 - Label two pass\n");
		printf(" 24 - Draw Countour\n");
		printf(" 25 - Reconstruct Countour\n");
		printf(" 26 - Dilate\n");
		printf(" 27 - Erode\n");
		printf(" 28 - Open\n");
		printf(" 29 - Close\n");
		printf(" 30 - Boundary\n");
		printf(" 31 - Fill\n");
		printf(" 32 - Compute Histogram\n");
		printf(" 33 - Mean and Standard Deviation\n");
		printf(" 34 - Image Threshold\n");
		printf(" 35 - Histo Strech/Shrink\n");
		printf(" 36 - Gamma Correction\n");
		printf(" 37 - Histo Slide\n");
		printf(" 38 - Histo Equalization\n");
		printf(" 39 - Convolution\n");
		printf(" 40 - Median Filter\n");
		printf(" 41 - Apply Gauss Filter 2d\n");
		printf(" 42 - Apply Gauss Filter 1d\n");
		printf(" 0 - Exit\n\n");
		printf("Option: ");
		scanf("%d", &op);

		Mat img, eroded, dilated, opened, closed, boundary, filled;
		int outMin, outMax;
		switch (op)
		{
		case 1:
			testOpenImage();
			break;
		case 2:
			testOpenImagesFld();
			break;
		case 3:
			testResize();
			break;
		case 4:
			testVideoSequence();
			break;
		case 5:
			testSnap();
			break;
		case 6:
			testMouseClick();
			break;
		case 7:
			negative_image();
			break;
		case 8:
			change_gray_levels(100);
			break;
		case 9:
			change_gray_levels_multiplic(5);
			break;
		case 10:
			create_squares();
			break;
		case 11:
			flip_horizontally();
			break;
		case 12:
			flip_vertically();
			break;
		case 13:
			crop_center();
			break;
		case 14:
			int newHeight, newWidth;
			printf("newHeight newWidth: ");
			scanf("%d %d", &newHeight, &newWidth);
			image_resize(newHeight, newWidth);
			break;
		case 15:
			splitRGB();
			break;
		case 16:
			RGBToGrayscale();
			break;
		case 17:
			uchar treshold;
			printf("Please introduce the treshold value(0 - 255): ");
			scanf("%hhu", &treshold);
			GrayscaleToBW(treshold);
			break;
		case 18:
			RGBToHSV();
			break;
		case 19:
			int th_red_low, th_red_high;
			printf("Threshold for red, low and high: ");
			scanf("%d %d", &th_red_low, &th_red_high);
			colorDetect(th_red_low, th_red_high);
			break;
		case 20:
			colorSelect();
			break;
		case 21:
			objectProperties();
			break;
		case 22:
			BFS();
			break;
		case 23:
			twoPass();
			break;
		case 24:
			drawCountour();
			break;
		case 25:
			reconstructContour();
			break;
		case 26:
			img = imread("Images/morph/1_Dilate/wdg2ded1_bw.bmp", CV_LOAD_IMAGE_GRAYSCALE);
			dilated = img;
			for (int i = 0; i < 10; ++i)
				dilated = dilate(dilated);
			imshow("orig", img);
			imshow("dilated", dilated);
			waitKey(0);
			break;
		case 27:
			img = imread("Images/morph/1_Dilate/wdg2thr3_bw.bmp", CV_LOAD_IMAGE_GRAYSCALE);
			eroded = img;
			for (int i = 0; i < 10; ++i)
				eroded = erode(eroded);
			imshow("orig", img);
			imshow("eroded", eroded);
			waitKey(0);
			break;
		case 28:
			img = imread("Images/morph/3_Open/cel4thr3_bw.bmp", CV_LOAD_IMAGE_GRAYSCALE);
			opened = open(img);
			imshow("orig", img);
			imshow("opened", opened);
			waitKey(0);
			break;
		case 29:
			img = imread("Images/morph/4_Close/phn1thr1_bw.bmp", CV_LOAD_IMAGE_GRAYSCALE);
			closed = close(img);
			imshow("orig", img);
			imshow("closed", closed);
			waitKey(0);
			break;
		case 30:
			img = imread("Images/morph/5_BoundaryExtraction/reg1neg1_bw.bmp", CV_LOAD_IMAGE_GRAYSCALE);
			boundary = bound(img);
			imshow("orig", img);
			imshow("boundary", boundary);
			waitKey(0);
			break;
		case 31:
			img = imread("Images/morph/6_RegionFilling/wdg2ded1_bw.bmp", CV_LOAD_IMAGE_GRAYSCALE);
			filled = fill(img);
			imshow("orig", img);
			imshow("filled", filled);
			waitKey(0);
			break;
		case 32:
			img = imread("Images/statistical_prop/balloons.bmp", CV_LOAD_IMAGE_GRAYSCALE);
			showHistogram("Histogram", computeHistogram(img), 255, 500);
			break;
		case 33:
			computeSD();
			break;
		case 34:
			thAlgo();
			break;
		case 35:
			std::cout << "outMin, outMax: \n";
			std::cin >> outMin >> outMax;
			histoStrShr(outMin, outMax);
			break;
		case 36:
			int gamma;
			std::cout << "gamma: \n";
			std::cin >> gamma;
			gammaCorrection(gamma);
			break;
		case 37:
			int offset;
			std::cout << "offset: \n";
			std::cin >> offset;
			histoSlide(offset);
			break;
		case 38:
			histoEq();
			break;
		case 39:
			// PART 1: convolution in the spatial domain
			//Mat_<uchar> img = imread("cameraman.bmp", CV_LOAD_IMAGE_GRAYSCALE);
			//Mat_<uchar> outputImage;

			//// LOW PASS	
			//// mean filter 5x5
			//int meanFilterData5x5[25];
			////fill_n(meanFilterData5x5, 25, 1);
			//Mat_<int> meanFilter5x5(5, 5, meanFilterData5x5);

			//// mean filter 3x3
			//Mat_<int> meanFilter3x3(3, 3, meanFilterData5x5);

			//// gaussian filter
			//int gaussianFilterData[9] = { 1, 2, 1, 2, 4, 2, 1, 2, 1 };
			//Mat_<int> gaussianFilter(3, 3, gaussianFilterData);

			//// HIGH PASS
			//// laplace filter 3x3
			//int laplaceFilterData[9] = { -1, -1, -1, -1, 8, -1, -1, -1, -1 };
			//Mat_<int> laplaceFilter(3, 3, laplaceFilterData);

			//int highpassFilterData[9] = { -1, -1, -1, -1, 9, -1, -1, -1, -1 };
			//Mat_<int> highpassFilter(3, 3, highpassFilterData);

			////TODO: convolution with the mean filter 5 x 5
			////TODO: convolution with the mean filter 3 x 3
			////TODO: convolution with the gaussian filter
			////TODO: convolution with the laplacian filter
			////TODO: convolution with the highpass filter


			//// PART 2: convolution in the frequency domain
			//// use the generic_frequency_domain_filter() function

			//// TODO: convolution with the ideal low pass filter (formula 9.16) take R^2 = 20
			//// TODO: convolution with the ideal high pass filter (formula 9.17) take R^2 = 20
			//// TODO: convolution with the Gaussian low pass filter (formula 9.18) take A = 10
			//// TODO: convolution with the Gaussian high pass filter (formula 9.19) take A = 10

			break;

		case 40:
			img = imread("Images/noise_images/balloons_Salt&Pepper.bmp", CV_LOAD_IMAGE_GRAYSCALE);
			medianFilter(img, 3);
			break;
		case 41:
			img = imread("Images/noise_images/balloons_Gauss.bmp", CV_LOAD_IMAGE_GRAYSCALE);
			applyGaussFilter2d(img, 5);
			break;
		case 42:
			img = imread("Images/noise_images/portrait_Gauss2.bmp", CV_LOAD_IMAGE_GRAYSCALE);
			applyGaussFilter1d(img, 5);
			break;
		}	
	} while (op != 0);

	return 0;
}