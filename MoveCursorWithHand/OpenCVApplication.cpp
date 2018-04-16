// OpenCVApplication.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "common.h"
#include <random>
#include <windows.h>


void testBGR2HSV()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname);
		int height = src.rows;
		int width = src.cols;

		// Componentele d eculoare ale modelului HSV
		Mat H = Mat(height, width, CV_8UC1);
		Mat S = Mat(height, width, CV_8UC1);
		Mat V = Mat(height, width, CV_8UC1);

		// definire pointeri la matricele (8 biti/pixeli) folosite la afisarea componentelor individuale H,S,V
		uchar* lpH = H.data;
		uchar* lpS = S.data;
		uchar* lpV = V.data;

		Mat hsvImg;
		cvtColor(src, hsvImg, CV_BGR2HSV);

		// definire pointer la matricea (24 biti/pixeli) a imaginii HSV
		uchar* hsvDataPtr = hsvImg.data;

		for (int i = 0; i<height; i++)
		{
			for (int j = 0; j<width; j++)
			{
				int hi = i*width * 3 + j * 3;
				int gi = i*width + j;

				lpH[gi] = hsvDataPtr[hi] * 510 / 360;		// lpH = 0 .. 255
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

Mat computeHue(Mat src)
{

	//Mat src = imread(fname);
	int height = src.rows;
	int width = src.cols;

	// Componentele d eculoare ale modelului HSV
	Mat H = Mat(height, width, CV_8UC1);
	Mat S = Mat(height, width, CV_8UC1);
	Mat V = Mat(height, width, CV_8UC1);

	// definire pointeri la matricele (8 biti/pixeli) folosite la afisarea componentelor individuale H,S,V
	uchar* lpH = H.data;
	uchar* lpS = S.data;
	uchar* lpV = V.data;

	Mat hsvImg;
	cvtColor(src, hsvImg, CV_BGR2HSV);

	// definire pointer la matricea (24 biti/pixeli) a imaginii HSV
	uchar* hsvDataPtr = hsvImg.data;

	for (int i = 0; i<height; i++)
	{
		for (int j = 0; j<width; j++)
		{
			int hi = i*width * 3 + j * 3;
			int gi = i*width + j;

			lpH[gi] = hsvDataPtr[hi] * 510 / 360;		// lpH = 0 .. 255
			lpS[gi] = hsvDataPtr[hi + 1];			// lpS = 0 .. 255
			lpV[gi] = hsvDataPtr[hi + 2];			// lpV = 0 .. 255
		}
	}

	return H;
}

Mat applyTh(Mat img, int lo, int hi) {

	Mat res = Mat(img.rows, img.cols, CV_8UC1);


	for (int i = 0; i < img.rows; ++i) {
		for (int j = 0; j < img.cols; ++j) {
			if (img.at<uchar>(i, j) < lo || img.at<uchar>(i, j) > hi) {
				res.at<uchar>(i, j) = 255;
			}	
			else
			{
				res.at<uchar>(i, j) = 0;
			}
		}
	}

	return res;
}

bool inBounds(int i, int j, int rows, int cols) {
	return (i >= 0 && j >= 0 && i < rows && j < cols);
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

}

Mat BFSLabel(Mat imgRead, int &nbLabels) {

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
				Q.push(Point2i(i, j));

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

	nbLabels = label;

	return labels;

}

int computeArea(Mat img) { //compute area for a specific color

	int area = 0;

	for (int i = 0; i < img.rows; ++i) {
		for (int j = 0; j < img.cols; ++j) {
			if (img.at<uchar>(i, j) == 0) {
				area++;
			}
		}
	}

	return area;
}


//calculate the Area for each individual label, and return the label with the highest area :)

int getHighestAreaLabel(Mat labels, int nbLabels, int &highestArea) { 

	int* areas = new int[nbLabels + 1](); //new int array initialized to 0
	int labelHighestArea = 0;
	highestArea = -1;

	std::cout << nbLabels << std::endl;

	for (int i = 0; i < labels.rows; ++i) {
		for (int j = 0; j < labels.cols; ++j) {
			int crtLbl = labels.at<int>(i, j);
			if (crtLbl > 0) { //the label is not background
				++areas[crtLbl];
				if (areas[crtLbl] > highestArea) {
					highestArea = areas[crtLbl];
					labelHighestArea = crtLbl;
				}
			}
		}
	}

	return labelHighestArea;
}

//only one object with the label label will be made black, everything else is set to white
Mat binaryOneLabel(Mat labels, int label) {
	Mat matOneLabel = Mat(labels.rows, labels.cols, CV_8UC1);

	for (int i = 0; i < labels.rows; ++i) {
		for (int j = 0; j < labels.cols; ++j) {
			if (labels.at<int>(i, j) == label) {
				matOneLabel.at<uchar>(i, j) = 0;
			}
			else {
				matOneLabel.at<uchar>(i, j) = 255;
			}
		}
	}

	return matOneLabel;
}

int computePerimeter(Mat img) {

	int perimeter = 0;

	for (int i = 0; i < img.rows; ++i) {
		for (int j = 0; j < img.cols; ++j) {
			if (img.at<uchar>(i, j) == 0) {
				bool edge = false;

				for (int ki = -1; ki <= 1; ++ki) {
					for (int kj = -1; kj <= 1; ++kj) {
						int ii = i + ki, jj = j + kj;
						if (inBounds(ii, jj, img.rows, img.cols)) {
							if (img.at<uchar>(ii, jj) == 255) {
								edge = true;
							}
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

//thinness(circularity) = 1 => circle
float computeThinnessRatio(int perimeter, int area) {
	return (4 * CV_PI) * ((float)(area) / (perimeter*perimeter));
}

//aspect ratio = width/height
float computeAspectRatio(Mat img) {
	int cmax = -1, rmax = -1, cmin = POS_INFINITY, rmin = POS_INFINITY;

	for (int i = 0; i < img.rows; ++i) {
		for (int j = 0; j < img.cols; ++j) {
			if (img.at<uchar>(i, j) == 0) {
				if (i > rmax) rmax = i;
				if (i < rmin) rmin = i;

				if (j > cmax) cmax = j;
				if (j < cmin) cmin = j;
			}
		}
	}

	return ((float)cmax - cmin + 1) / (rmax - rmin + 1);
}

//hugeClose
Mat hugeClose(Mat img) {
	Mat res = img;
	for (int i = 0; i < 2; ++i) {
		res = dilate(res);
	}

	for (int i = 0; i < 2; ++i) {
		res = erode(res);
	}

	return res;
}


Mat detectHand(Mat src, boolean &fist) { //compute HUE histogram

	Mat hand;

	Mat rsz; //resized
	resizeImg(src, rsz, 1080, true);
		
	Mat hue = computeHue(rsz);

//	imshow("hue", hue);

	int *histo = computeHistogram(hue);

//	showHistogram("HueHisto", histo, 256, 300);

	Mat imgTh = applyTh(hue, 1, 35);
	//imshow("threshold", imgTh);

	//erode x times
	Mat eroded = imgTh;

	for (int i = 0; i < 1; ++i) {
		eroded = erode(eroded);
	}

	//close y times
	Mat closed = eroded;
		
	for (int i = 0; i < 5; ++i) {
		closed = close(closed);
	}

//		imshow("closed", closed);

	int nbOfLabels;
	Mat labels = BFSLabel(closed, nbOfLabels);

	if (nbOfLabels == 0) { //no object detected
		return Mat(0, 0, CV_8UC1);
	}

//	genColImg(labels); //show labels

	int handArea; //the area of the hand
	//generate matrix with the object with the highest area, in our case the hand.
	hand = binaryOneLabel(labels, getHighestAreaLabel(labels, nbOfLabels, handArea));

//	imshow("hand", hand);

	hand = hugeClose(hand);
	//area has changed, compute it again
	handArea = computeArea(hand);

//	imshow("handHugeClose", hand);


	//compute Aspect Ratio to determine if it's an open fist or not
	//float aspectRatio = computeAspectRatio(hand);
	//std::cout << "aspect ratio: " << aspectRatio << std::endl;

	//compute circularity to determine if it's an open fist or not
	float circularity = computeThinnessRatio(computePerimeter(hand), handArea);
	std::cout << "circularity: " << circularity << std::endl;

	if (circularity > 0.45) {
		fist = true;
		std::cout << "FIST\n";
	}
	else {
		fist = false;
		std::cout << "OPEN HAND\n";
	}

	return hand;
}

Point computeCoM(Mat img, int area) { //Center of Mass

	double rowCenter = 0, colCenter = 0;

	for (int i = 0; i < img.rows; ++i) {
		for (int j = 0; j < img.cols; ++j) {
			if (img.at<uchar>(i, j) == 0) {
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

Point computeHandCoM(Mat hand) {

	for (int i = 0; i < 2; ++i) {
		hand = dilate(hand);
	}

	int area = computeArea(hand);

	return computeCoM(hand, area);
}

// Get the horizontal and vertical screen sizes in pixel
void GetDesktopResolution(int& horizontal, int& vertical)
{
	RECT desktop;
	// Get a handle to the desktop window
	const HWND hDesktop = GetDesktopWindow();
	// Get the size of screen to the variable desktop
	GetWindowRect(hDesktop, &desktop);
	// The top left corner will have coordinates (0,0)
	// and the bottom right corner will have coordinates
	// (horizontal, vertical)
	horizontal = desktop.right;
	vertical = desktop.bottom;
}

// left clicks at the current mouse possition
void leftClick() { 
	INPUT Inputs[1] = { 0 };

	Inputs[0].type = INPUT_MOUSE;
	Inputs[0].mi.dwFlags = MOUSEEVENTF_LEFTDOWN;

	SendInput(1, Inputs, sizeof(INPUT));
}

//releases left clcik
void releaseLeftClick() {
	INPUT Inputs[1] = { 0 };

	Inputs[0].type = INPUT_MOUSE;
	Inputs[0].mi.dwFlags = MOUSEEVENTF_LEFTUP;

	SendInput(1, Inputs, sizeof(INPUT));
}



void moveCursorWithHand()
{
	//VideoCapture cap("Videos/rubic.avi"); // off-line video from file
	VideoCapture cap(0);	// live video from web cam
	if (!cap.isOpened()) {
		printf("Cannot open video capture device.\n");
		waitKey(0);
		return;
	}

	Mat edges;
	Mat frame;
	Point cOm; //center of mass
	boolean fist, wasFist = false; //true if the shape of the hand is a fist
	int i = 0;
	char c;

	int widthScreen, heightScreen;
	GetDesktopResolution(widthScreen, heightScreen);

	//namedWindow("source");

	while (cap.read(frame))
	{
		//erase mirror effect, flip horizontally
		flip(frame, frame, 1);

		Mat hand = detectHand(frame, fist);
		if (hand.rows == 0) {
			std::cout << "No hand detected\n";
		}
		else {

			imshow("source", frame);
			

			//get the center of mass after some dilation
			cOm = computeHandCoM(hand);
			
			Point cOmOld = cOm; //save the position on the window in order to print the cOm

			//move according to screen resolution
			cOm.x = ((float)widthScreen / hand.cols) * cOm.x;
			cOm.y = ((float)heightScreen / hand.rows) * cOm.y;
			
			SetCursorPos(cOm.x, cOm.y);

			if (fist) {
				leftClick();
				releaseLeftClick();
				wasFist = true;
				circle(hand, cOmOld, 4, Scalar(255, 0, 0), 2); //color center of mass differently if fist
			}
			else {
				//if (wasFist) {
				//	releaseLeftClick();
				//}
				wasFist = false;
				circle(hand, cOmOld, 4, Scalar(125, 125, 125), 2); //color center of mass differently if open hand
			}

			imshow("hand", hand);
				
		}
		waitKey(1);
	}
}

int main()
{
	int op;
	do
	{
		system("cls");
		destroyAllWindows();
		moveCursorWithHand();
		
	}
	while (true);
	return 0;
}