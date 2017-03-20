//#include <iostream>
//#include <fstream>
//#include <strstream>
//#include <opencv2/core/core.hpp>
//#include <opencv2/highgui/highgui.hpp>
//#include <opencv2/imgproc/imgproc.hpp>
//#include <opencv2/objdetect/objdetect.hpp>
//#include <opencv2/ml/ml.hpp>
//
//
//using namespace std;
//using namespace cv;
//
////-----------------------宏定义----------------------------
////---------------------------------------------------------
//#define PosSamNO 25		//正样本个数
//#define NegSamNO 94		//负样本个数
//#define HardExampleNO 0	//HardExample：负样本个数。如果HardExampleNO大于0，表示处理完初始负样本集后，继续处理HardExample负样本集。  
////不使用HardExample时必须设置为0，因为特征向量矩阵和特征类别矩阵的维数初始化时用到这个值  
//
//
//#define BIAS -0.5	//超平面偏移
//
//#define TRAIN true			//是否进行训练,true表示训练，false表示读取xml文件中的SVM模型
//#define SamWidth 32			//样本宽度
//#define SamHight 16			//样本高度
//#define VideoOutputHight 128//输出视频图像的高度
//
//#define TestImage "../Data/TestImage/5.jpg"			//用于检测的测试图像
//#define ResultImage "../Data/Result/5.jpg"			//测试图像的检测结果
//#define TestVideo "../Data/TestVideo/bh_20160923_celue.avi"	//用于检测的测试视频
//#define ResultVideo "../Data/Result/bh_20160923_celue.avi"		//测试视频的检测结果
//
//#define PosSetFile "../Data/PosSetResized/"			//正样本图片的文件夹
//#define NegSetFile "../Data/NegSetResized/"			//负样本图片的文件夹
//#define HardSetFile "../Data/HardSetResized/"		//难例负样本图片的文件夹
//#define SetName "0SetName.txt"						//样本图片的文件名列表txt
//
//#define SaveSvmName "../Data/Result/SVM_HOG.xml"	//训练保存的模型文件名称
//#define LoadSvmName "../Data/Result/SVM_HOG.xml"	//载入已有的模型文件名称
//
////---------------------------------------------------------
//
//int main()  
//{
//
//	////-----------------------------进行图像分割---------------------------------
//	////变量定义
//	//Mat src = imread(TestImage); //读取被测图像
//	//Mat dst = Mat::zeros(src.rows,src.cols,CV_8UC1);//输出图像
//
//	////进行图像分割（单像素操作法，速度较慢）
//	//for (int r = 0;r<src.rows;r++)
//	//{
//	//	for (int c = 0;c<src.cols;c++)
//	//	{
//	//		//cout<<"a"<<float(src.at<Vec3b>(r,c)[0])<<"b"<<float(src.at<Vec3b>(r,c)[1])<<"c"<<float(src.at<Vec3b>(r,c)[2]);
//	//		Mat tempMat = (Mat_<float>(1,3) << src.at<Vec3b>(r,c)[0],src.at<Vec3b>(r,c)[1],src.at<Vec3b>(r,c)[2]);
//	//		float response = svm.predict(tempMat);  
//
//	//		if (response > 0)  
//	//			dst.at<uchar>(r,c) = 255;
//	//		//else if (response == -1)
//	//		//	src.at<uchar>(r, c)  = 0;
//	//	}
//	//}
//	////储存检测图像结果
//	//imwrite(ResultImage,dst);  
//	//namedWindow("dst");  
//	//imshow("dst",dst);  
//	//waitKey(0);//注意：imshow之后必须加waitKey，否则无法显示图像  
//
//	////-----------------------------end---------------------------------
//
//	//-----------------------------进行视频图像分割---------------------------------
//	//变量定义
//	VideoCapture myVideo(TestVideo);//读取视频  
//	Mat src_BGR,src_HSV,src_BGR_32FC3,src_HSV_32FC3,gray;
//	Mat src_BGR_Channels[3],src_HSV_Channels[3];
//	Mat R,G,B;//RGB单通道
//	Mat nR,nG,nB;//nRGB单通道
//	Mat H,S,V;//HSV单通道
//	Mat rThr,gThr,bThr,wThr,threshold1,threshold2,threshold;//二值图像
//
//	//打开视频
//	if(!myVideo.isOpened()){cout<<"视频读取错误"<<endl;system("puase");return -1;}
//
//	//设置生成的视频
//	double videoRate=myVideo.get(CV_CAP_PROP_FPS);//获取帧率
//	int videoWidth=myVideo.get(CV_CAP_PROP_FRAME_WIDTH);//获取视频图像宽度
//	int videoHight=myVideo.get(CV_CAP_PROP_FRAME_HEIGHT);//获取视频图像高度
//	int videoDelay=1000/videoRate;//每帧之间的延迟与视频的帧率相对应（设置跑程序的时候播放视频的速率）
//	VideoWriter outputVideo(ResultVideo, CV_FOURCC('M', 'J', 'P', 'G'), videoRate,Size(VideoOutputHight*videoWidth/videoHight,VideoOutputHight));//设置视频类
//	Mat videoImage = Mat::zeros(Size(videoWidth,videoHight),CV_8UC3);
//
//	//开始视频处理
//	bool stop = false;
//	while (!stop)
//	{
//		//变量定义
//		if (!myVideo.read(src_BGR)){cout<<"视频结束"<<endl;waitKey(0); break;}//获取视频帧
//
//		//改变视频图像尺寸
//		
//		//cvtColor(src_BGR,src_HSV,CV_BGR2HSV);//转换为HSV
//		//cvtColor(src_BGR,gray,CV_BGR2GRAY);//转换为灰度图
//		src_BGR.convertTo(src_BGR_32FC3,CV_32FC3);//转换图像数据类型
//		//src_HSV.convertTo(src_HSV_32FC3,CV_32FC3);//转换图像数据类型
//		split(src_BGR_32FC3,src_BGR_Channels);//分离RGB通道
//		//split(src_HSV_32FC3,src_HSV_Channels);//分离HSV通道
//		B = src_BGR_Channels[0];G = src_BGR_Channels[1];R = src_BGR_Channels[2];
//		//nB = B/(R+G+B);nG = G/(R+G+B);nR = R/(R+G+B);
//		//H = src_HSV_Channels[0];S = src_HSV_Channels[1];V = src_HSV_Channels[2];
//
//		//进行图像分割（色彩空间SVM方法）（矩阵操作方法，速度快）
//		//threshold = resultMat.at<float>(0)*R + resultMat.at<float>(1)*G + resultMat.at<float>(2)*B + rho > BIAS;
//		//-2.66061854e-002 1.34229800e-002 2.48698164e-002 -9.9427669280579056e-001;
//		//5.06762154e-002 -2.20872685e-001 2.25031987e-001 2.3481620039919280e+000;
//		threshold1 = -(-0.02661)*R-(0.01342)*G-(0.02487)*B+(-0.9943)>-0.5;//r
//		//threshold2 = -(0.03109)*R-(-0.05308)*G-(0.03566)*B+(0.3493)>-0.1;//g
//		threshold2 = -(0.050676)*R-(-0.2209)*G-(0.2250)*B+(2.34816)>0.5;//g
//		threshold = threshold1 | threshold2;
//		morphologyEx(threshold,threshold,MORPH_OPEN,getStructuringElement(MORPH_ELLIPSE, Size(3,3)));//形态学滤波：场内去噪//#9.23
//		morphologyEx(threshold,threshold,MORPH_CLOSE,getStructuringElement(MORPH_ELLIPSE, Size(3,3)));//形态学滤波：场外去噪//#9.23
//		//////////////////////////threshold2 = -(0.050676)*R-(-0.2209)*G-(0.2250)*B+(2.34816)>0.5;//g
//		//medianBlur(threshold,threshold,5);
//
//		//threshold = - (0.273370087)*R - (-0.377990603)*G - (0.350543886)*B + (32.809016951026422) > BIAS;//0.273370087等为xml中的数据
//
//		////进行图像分割（阈值法）
//		//rThr = (R/(G+B))>0.7;
//		//gThr = (R<150)&(B<150)&(G>R+5)&(G>B+5);
//		//bThr = (R>110)&(G>110)&(B>110)&(B>R-5)&(B>G-5);
//		//wThr = gray>200;
//		//threshold = ~(bThr|wThr|rThr);
//
//		////仅供生成视频！！！！！！！！！！！！！
//		//for (int r = 0;r<src_BGR.rows;r++)
//		//{
//		//	for (int c = 0;c<src_BGR.cols;c++)
//		//	{
//		//		if (threshold.at<uchar>(r,c)>0)
//		//		{
//		//			videoImage.at<Vec3b>(r,c) = Vec3b(255,255,255);
//		//		} 
//		//		else
//		//		{
//		//			videoImage.at<Vec3b>(r,c) = Vec3b(0,0,0);
//		//		}
//
//		//	}
//		//}
//
//		////进行图像分割（单像素操作法，速度较慢）
//		//Mat dst = Mat::zeros(VideoOutputHight,VideoOutputHight*videoWidth/videoHight,CV_8U);//输出图像
//		//for (int r = 0;r<src.rows;r++)
//		//{
//		//	for (int c = 0;c<src.cols;c++)
//		//	{
//		//		//cout<<"a"<<float(src.at<Vec3b>(r,c)[0])<<"b"<<float(src.at<Vec3b>(r,c)[1])<<"c"<<float(src.at<Vec3b>(r,c)[2]);
//		//		Mat tempMat = (Mat_<float>(1,3) << src.at<Vec3b>(r,c)[0],src.at<Vec3b>(r,c)[1],src.at<Vec3b>(r,c)[2]);
//		//		float response = svm.predict(tempMat);  
//		//		if (response == 1)  
//		//			dst.at<uchar>(r,c) = 255;
//		//		else if (response == -1)
//		//			dst.at<uchar>(r, c)  = 0;
//
//		//		//仅供生成视频！！！！！！！！！！！！！
//		//		if (response == 1)  
//		//			src.at<Vec3b>(r,c) = Vec3b(255,255,255);
//		//		else if (response == -1)
//		//			src.at<Vec3b>(r,c) = Vec3b(0,0,0);
//		//	}
//		//}
//
//		//储存视频图像
//		//outputVideo<<videoImage;
//		namedWindow("threshold",WINDOW_NORMAL);
//		imshow("threshold",threshold);
//		namedWindow("src",WINDOW_NORMAL);
//		imshow("src",src_BGR);
//
//		//dst、threshold才是该得到的二值图像，但是二值图不能写入视频
//
//
//		if(waitKey(1)>=0)stop = true;//通过按键停止视频
//	}
//	//-----------------------------end---------------------------------
//
//}  