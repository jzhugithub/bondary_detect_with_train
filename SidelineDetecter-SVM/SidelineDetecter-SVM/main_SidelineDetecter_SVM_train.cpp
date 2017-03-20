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
//#define PosSamNO 22		//正样本个数
//#define NegSamNO 56		//负样本个数
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
//#define TestVideo "../Data/TestVideo/BH_20160924.avi"	//用于检测的测试视频
//#define ResultVideo "../Data/Result/BH_20160924.avi"		//测试视频的检测结果
//
//#define PosSetFile "../Data/PosSetResized/"			//正样本图片的文件夹
//#define NegSetFile "../Data/NegSetResized/"			//负样本图片的文件夹
//#define HardSetFile "../Data/HardSetResized/"		//难例负样本图片的文件夹
//#define SetName "0SetName.txt"						//样本图片的文件名列表txt
//
//#define SaveSvmName "../Data/Result/SVM_HOG.xml"	//训练保存的模型文件名称
//#define LoadSvmName "../Data/Result/SVM_HOG.xml"	//载入已有的模型文件名称
//
////-----------------------继承类----------------------------
////---------------------------------------------------------
////继承自CvSVM的类，因为生成setSVMDetector()中用到的检测子参数时，需要用到训练好的SVM的decision_func参数，  
////但通过查看CvSVM源码可知decision_func参数是protected类型变量，无法直接访问到，只能继承之后通过函数访问  
//class MySVM : public CvSVM  
//{  
//public:  
//	//获得SVM的决策函数中的alpha数组  
//	double * get_alpha_vector()  
//	{  
//		return this->decision_func->alpha;  
//	}  
//
//	//获得SVM的决策函数中的rho参数,即偏移量  
//	float get_rho()  
//	{  
//		return this->decision_func->rho;  
//	}  
//};  
////-----------------------主函数----------------------------
////---------------------------------------------------------
//
//int main()  
//{
//	//变量定义
//	int descriptorDim = 3;//特征向量维数，色彩空间为三维
//	MySVM svm;//SVM分类器
//
//	//----------------训练分类器or直接读取分类器---------------------
//	//---------------------------------------------------------------
//	if(TRAIN) //训练分类器，并保存XML文件
//	{
//		//0.训练变量定义
//		string ImgName;//图片名
//		ifstream finPos((string)PosSetFile+SetName);//正样本图片的文件名列表  
//		ifstream finNeg((string)NegSetFile+SetName);//负样本图片的文件名列表
//		ifstream finHard((string)HardSetFile+SetName);//难例负样本图片的文件名列表 
//		Mat sampleFeatureMat = Mat::zeros((PosSamNO+NegSamNO+HardExampleNO)*SamHight*SamWidth, descriptorDim, CV_32FC1);
//		//所有训练样本的特征向量组成的矩阵，行数等于所有样本的个数，列数等于特征向量维数
//		Mat sampleLabelMat = Mat::zeros((PosSamNO+NegSamNO+HardExampleNO)*SamHight*SamWidth, 1, CV_32FC1);
//		//训练样本的类别向量，行数等于所有样本的个数，列数等于1；1表示有机器人，-1表示无机器人  
//
//		//1.依次读取正样本图片，生成训练数据矩阵
//		for(int num=0; num<PosSamNO && getline(finPos,ImgName); num++)  
//		{
//			ImgName = PosSetFile + ImgName;//加上正样本的路径名  
//			cout<<"处理："<<ImgName<<endl;  
//			Mat src_BGR = imread(ImgName);//读取图片
//			Mat src_HSV;
//			resize(src_BGR,src_BGR,Size(SamWidth,SamHight));//将训练样本归一化为统一大小
//			cvtColor(src_BGR,src_HSV,CV_BGR2HSV);//转换为HSV
//			//将色彩空间特征向量复制到样本特征矩阵sampleFeatureMat
//			for (int w = 0;w<SamWidth;w++)
//			{
//				for (int h = 0;h<SamHight;h++)
//				{
//					//BGR
//					float B = src_BGR.at<Vec3b>(h,w)[0];
//					float G = src_BGR.at<Vec3b>(h,w)[1];
//					float R = src_BGR.at<Vec3b>(h,w)[2];
//					//nBGR
//					float nB = B/(R+G+B);
//					float nG = G/(R+G+B);
//					float nR = R/(R+G+B);
//					//HSV
//					float H = src_HSV.at<Vec3b>(h,w)[0];
//					float S = src_HSV.at<Vec3b>(h,w)[1];
//					float V = src_HSV.at<Vec3b>(h,w)[2];
//					//特征矩阵赋值
//					sampleFeatureMat.at<float>(num*SamHight*SamWidth+w*SamHight+h,0) = R;
//					sampleFeatureMat.at<float>(num*SamHight*SamWidth+w*SamHight+h,1) = G;
//					sampleFeatureMat.at<float>(num*SamHight*SamWidth+w*SamHight+h,2) = B;
//					//sampleFeatureMat.at<float>(num*SamHight*SamWidth+w*SamHight+h,3) = R/(R+G+B);
//					//sampleFeatureMat.at<float>(num*SamHight*SamWidth+w*SamHight+h,4) = G/(R+G+B);
//					//sampleFeatureMat.at<float>(num*SamHight*SamWidth+w*SamHight+h,5) = B/(R+G+B);
//
//					//sampleLabelMat赋值
//					sampleLabelMat.at<float>(num*SamHight*SamWidth+w*SamHight+h,0) = 1;//正样本类别为1
//				}
//			}
//		}  
//
//		//2.依次读取负样本图片，生成训练数据矩阵
//		for(int num=0; num<NegSamNO && getline(finNeg,ImgName); num++)  
//		{  
//			ImgName = NegSetFile + ImgName;//加上正样本的路径名
//			cout<<"处理："<<ImgName<<endl;  
//			Mat src_BGR = imread(ImgName);//读取图片
//			Mat src_HSV;
//			resize(src_BGR,src_BGR,Size(SamWidth,SamHight));//将训练样本归一化为统一大小
//			cvtColor(src_BGR,src_HSV,CV_BGR2HSV);//转换为HSV
//
//			//将色彩空间特征向量复制到样本特征矩阵sampleFeatureMat
//			for (int w = 0;w<SamWidth;w++)
//			{
//				for (int h = 0;h<SamHight;h++)
//				{
//					//BGR
//					float B = src_BGR.at<Vec3b>(h,w)[0];
//					float G = src_BGR.at<Vec3b>(h,w)[1];
//					float R = src_BGR.at<Vec3b>(h,w)[2];
//					//nBGR
//					float nB = B/(R+G+B);
//					float nG = G/(R+G+B);
//					float nR = R/(R+G+B);
//					//HSV
//					float H = src_HSV.at<Vec3b>(h,w)[0];
//					float S = src_HSV.at<Vec3b>(h,w)[1];
//					float V = src_HSV.at<Vec3b>(h,w)[2];
//					//特征矩阵赋值
//					sampleFeatureMat.at<float>((num+PosSamNO)*SamHight*SamWidth+w*SamHight+h,0) = R;
//					sampleFeatureMat.at<float>((num+PosSamNO)*SamHight*SamWidth+w*SamHight+h,1) = G;
//					sampleFeatureMat.at<float>((num+PosSamNO)*SamHight*SamWidth+w*SamHight+h,2) = B;
//					//sampleFeatureMat.at<float>((num+PosSamNO)*SamHight*SamWidth+w*SamHight+h,3) = R/(R+G+B);
//					//sampleFeatureMat.at<float>((num+PosSamNO)*SamHight*SamWidth+w*SamHight+h,4) = G/(R+G+B);
//					//sampleFeatureMat.at<float>((num+PosSamNO)*SamHight*SamWidth+w*SamHight+h,5) = B/(R+G+B);
//
//					//sampleLabelMat赋值
//					sampleLabelMat.at<float>((num+PosSamNO)*SamHight*SamWidth+w*SamHight+h,0) = -1;//负样本类别为-1
//				}
//			}
//		}  
//
//		//3.依次读取HardExample负样本图片，生成训练数据矩阵
//		for(int num=0; num<HardExampleNO && getline(finHard,ImgName); num++)  
//		{
//			ImgName = HardSetFile + ImgName;//加上正样本的路径名
//			cout<<"处理："<<ImgName<<endl;  
//			Mat src_BGR = imread(ImgName);//读取图片
//			Mat src_HSV;
//			resize(src_BGR,src_BGR,Size(SamWidth,SamHight));//将训练样本归一化为统一大小
//			cvtColor(src_BGR,src_HSV,CV_BGR2HSV);//转换为HSV
//
//			//将色彩空间特征向量复制到样本特征矩阵sampleFeatureMat
//			for (int w = 0;w<SamWidth;w++)
//			{
//				for (int h = 0;h<SamHight;h++)
//				{
//					//BGR
//					float B = src_BGR.at<Vec3b>(h,w)[0];
//					float G = src_BGR.at<Vec3b>(h,w)[1];
//					float R = src_BGR.at<Vec3b>(h,w)[2];
//					//nBGR
//					float nB = B/(R+G+B);
//					float nG = G/(R+G+B);
//					float nR = R/(R+G+B);
//					//HSV
//					float H = src_HSV.at<Vec3b>(h,w)[0];
//					float S = src_HSV.at<Vec3b>(h,w)[1];
//					float V = src_HSV.at<Vec3b>(h,w)[2];
//					//特征矩阵赋值
//					sampleFeatureMat.at<float>((num+PosSamNO+NegSamNO)*SamHight*SamWidth+w*SamHight+h,0) = R;
//					sampleFeatureMat.at<float>((num+PosSamNO+NegSamNO)*SamHight*SamWidth+w*SamHight+h,1) = G;
//					sampleFeatureMat.at<float>((num+PosSamNO+NegSamNO)*SamHight*SamWidth+w*SamHight+h,2) = B;
//					//sampleFeatureMat.at<float>((num+PosSamNO+NegSamNO)*SamHight*SamWidth+w*SamHight+h,3) = R/(R+G+B);
//					//sampleFeatureMat.at<float>((num+PosSamNO+NegSamNO)*SamHight*SamWidth+w*SamHight+h,4) = G/(R+G+B);
//					//sampleFeatureMat.at<float>((num+PosSamNO+NegSamNO)*SamHight*SamWidth+w*SamHight+h,5) = B/(R+G+B);
//
//					//sampleLabelMat赋值
//					sampleLabelMat.at<float>((num+PosSamNO+NegSamNO)*SamHight*SamWidth+w*SamHight+h,0) = -1;//负样本类别为-1
//				}
//			}
//		}  
//
//		//4.训练SVM分类器  
//		//迭代终止条件，当迭代满1000次或误差小于FLT_EPSILON时停止迭代
//		CvTermCriteria criteria = cvTermCriteria(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, 1000, FLT_EPSILON);  
//		//SVM参数：SVM类型为C_SVC；线性核函数；松弛因子C=0.01
//		CvSVMParams param(CvSVM::C_SVC, CvSVM::LINEAR, 0, 1, 0, 0.01, 0, 0, 0, criteria);  
//		cout<<"开始训练SVM分类器"<<endl;  
//		svm.train(sampleFeatureMat, sampleLabelMat, Mat(), Mat(), param);//训练分类器
//		cout<<"训练完成"<<endl;  
//		svm.save(SaveSvmName);//将训练好的SVM模型保存为xml文件  
//	}  
//	else //若TRAIN为false，从XML文件读取训练好的分类器  
//	{  
//		svm.load(LoadSvmName);//从XML文件读取训练好的SVM模型  
//	}  
//
//
//	//----------------读取模型数据生成w+b---------------------
//	descriptorDim = svm.get_var_count();//特征向量的维数（和前面训练时的大小一样，添加此句是为了在不训练时也能拿到维数）
//	int supportVectorNum = svm.get_support_vector_count();//支持向量的个数  
//	//cout<<"支持向量个数："<<supportVectorNum<<endl;  
//
//	Mat alphaMat = Mat::zeros(1, supportVectorNum, CV_32FC1);//alpha向量，长度等于支持向量个数  
//	Mat supportVectorMat = Mat::zeros(supportVectorNum, descriptorDim, CV_32FC1);//支持向量矩阵  
//	Mat resultMat = Mat::zeros(1, descriptorDim, CV_32FC1);//alpha向量乘以支持向量矩阵的结果
//	float rho;//b偏移的结果
//
//	//将支持向量的数据复制到supportVectorMat矩阵中  
//	for(int i=0; i<supportVectorNum; i++)  
//	{  
//		const float * pSVData = svm.get_support_vector(i);//返回第i个支持向量的数据指针  
//		for(int j=0; j<descriptorDim; j++)  
//		{  
//			//cout<<pData[j]<<" ";  
//			supportVectorMat.at<float>(i,j) = pSVData[j];  
//		}  
//	}  
//
//	//将alpha向量的数据复制到alphaMat中  
//	double * pAlphaData = svm.get_alpha_vector();//返回SVM的决策函数中的alpha向量  
//	for(int i=0; i<supportVectorNum; i++)  
//	{  
//		alphaMat.at<float>(0,i) = pAlphaData[i];  
//	}  
//
//	//计算-(alphaMat * supportVectorMat),结果放到resultMat中  
//	//gemm(alphaMat, supportVectorMat, -1, 0, 1, resultMat);//不知道为什么加负号？  
//	//注意因为svm.predict使用的是alpha*sv*another-rho，如果为负的话则认为是正样本，
//	//在HOG的检测函数中，使用rho+alpha*sv*another(another为-1)
//	resultMat = -1 * alphaMat * supportVectorMat;  
//
//	//b偏移结果
//	rho = svm.get_rho();
//
//
//	////定义w+b的向量
//	//vector<float> myDetector;
//
//	////将resultMat中的数据复制到数组myDetector中  
//	//for(int i=0; i<descriptorDim; i++)  
//	//{  
//	//	myDetector.push_back(resultMat.at<float>(0,i));  
//	//}
//	////最后添加偏移量rho，得到w+b的向量
//	//myDetector.push_back(svm.get_rho());  
//	////cout<<"检测子维数(w+b)："<<myDetector.size()<<endl;
//	////cout<<myDetector.at(0)<<endl<<myDetector.at(1)<<endl<<myDetector.at(2)<<endl<<myDetector.at(3)<<endl;
//	//float w1 = myDetector.at(0),w2 = myDetector.at(1),w3 = myDetector.at(2),b = myDetector.at(3);
//
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
//	Mat rThr,gThr,bThr,wThr,threshold;//二值图像
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
//	Mat videoImage = Mat::zeros(Size(VideoOutputHight*videoWidth/videoHight,VideoOutputHight),CV_8UC3);
//
//
//	//开始视频处理
//	bool stop = false;
//	while (!stop)
//	{
//		//变量定义
//		if (!myVideo.read(src_BGR)){cout<<"视频结束"<<endl;waitKey(0); break;}//获取视频帧
//
//		//改变视频图像尺寸
//		resize(src_BGR,src_BGR,Size(VideoOutputHight*videoWidth/videoHight,VideoOutputHight));
//		cvtColor(src_BGR,src_HSV,CV_BGR2HSV);//转换为HSV
//		cvtColor(src_BGR,gray,CV_BGR2GRAY);//转换为灰度图
//		src_BGR.convertTo(src_BGR_32FC3,CV_32FC3);//转换图像数据类型
//		src_HSV.convertTo(src_HSV_32FC3,CV_32FC3);//转换图像数据类型
//		split(src_BGR_32FC3,src_BGR_Channels);//分离RGB通道
//		split(src_HSV_32FC3,src_HSV_Channels);//分离HSV通道
//		B = src_BGR_Channels[0];G = src_BGR_Channels[1];R = src_BGR_Channels[2];
//		nB = B/(R+G+B);nG = G/(R+G+B);nR = R/(R+G+B);
//		H = src_HSV_Channels[0];S = src_HSV_Channels[1];V = src_HSV_Channels[2];
//
//		//进行图像分割（色彩空间SVM方法）（矩阵操作方法，速度快）
//		threshold = resultMat.at<float>(0)*R + resultMat.at<float>(1)*G + resultMat.at<float>(2)*B + rho > BIAS;
//		//threshold = - (0.273370087)*R - (-0.377990603)*G - (0.350543886)*B + (32.809016951026422) > BIAS;//0.273370087等为xml中的数据
//
//		////进行图像分割（阈值法）
//		//rThr = (R/(G+B))>0.7;
//		//gThr = (R<150)&(B<150)&(G>R+5)&(G>B+5);
//		//bThr = (R>110)&(G>110)&(B>110)&(B>R-5)&(B>G-5);
//		//wThr = gray>200;
//		//threshold = ~(bThr|wThr|rThr);
//
//		//仅供生成视频！！！！！！！！！！！！！
//		for (int r = 0;r<src_BGR.rows;r++)
//		{
//			for (int c = 0;c<src_BGR.cols;c++)
//			{
//				if (threshold.at<uchar>(r,c)>0)
//				{
//					videoImage.at<Vec3b>(r,c) = Vec3b(255,255,255);
//				} 
//				else
//				{
//					videoImage.at<Vec3b>(r,c) = Vec3b(0,0,0);
//				}
//
//			}
//		}
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
//		outputVideo<<videoImage;
//		imshow("videoImage",videoImage);
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