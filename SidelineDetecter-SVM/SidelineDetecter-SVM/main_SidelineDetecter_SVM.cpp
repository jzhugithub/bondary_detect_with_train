//#include <iostream>
//#include <fstream>
//#include <strstream>
//#include <opencv2/core/core.hpp>
//#include <opencv2/highgui/highgui.hpp>
//#include <opencv2/imgproc/imgproc.hpp>
//#include <opencv2/objdetect/objdetect.hpp>
//#include <opencv2/ml/ml.hpp>
//
//using namespace std; 
//using namespace cv;
////-----------------------宏定义----------------------------
//#define VideoOutputHight 128		//输出视频图像的高度
//#define TestVideo "../Data/TestVideo/output.avi"	//用于检测的测试视频
//#define ResultVideo "../Data/Result/output.avi"		//测试视频的检测结果
//#define LoadSvmName "../Data/Result/SVM_HOG.xml"	//载入已有的模型文件名称
//
////-----------------------继承类----------------------------
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
//
//enum xSideType{noxSide,leftSide,rightSide};
//enum ySideType{noySide,topSide,bottomSide};
//
////*******************************输入输出********************************
////输入
//Mat src;
//double angleBias = 0.0;//线角度矫正，即竖线在图像中应有的角度(逆时针为正)
////输出
//xSideType xSide;//竖线结果
//ySideType ySide;//横线结果
//int xValue;//竖线x坐标（哪一列）（没找到则返回-1）
//int yValue;//横线y坐标（哪一行）（没找到则返回-1）
//Point2i Px1,Px2;//竖线段端点（没找到则返回-1,-1）
//Point2i Py1,Py2;//横线段端点（没找到则返回-1,-1）
////*********************************end***********************************
//int main()
//{
//	//----------------读取模型数据生成w+b---------------------
//	//变量定义
//	MySVM svm;//SVM分类器
//	svm.load(LoadSvmName);//从XML文件读取训练好的SVM模型
//	int descriptorDim = 3;//特征向量维数，色彩空间为三维
//	int supportVectorNum = svm.get_support_vector_count();//支持向量的个数  
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
//			supportVectorMat.at<float>(i,j) = pSVData[j];  
//		}  
//	}  
//	//将alpha向量的数据复制到alphaMat中  
//	double * pAlphaData = svm.get_alpha_vector();//返回SVM的决策函数中的alpha向量  
//	for(int i=0; i<supportVectorNum; i++)  
//	{  
//		alphaMat.at<float>(0,i) = pAlphaData[i];  
//	}  
//	//计算-(alphaMat * supportVectorMat),结果放到resultMat中  
//	resultMat = -1 * alphaMat * supportVectorMat;  
//	//b偏移结果
//	rho = svm.get_rho();
//
//	//-----------------------视频设置--------------------------
//	//打开视频
//	VideoCapture cap(TestVideo);
//	if(!cap.isOpened()){cout<<"视频读取错误"<<endl;system("puase");return -1;}
//	//获取视频信息
//	double rate=cap.get(CV_CAP_PROP_FPS);//获取帧率
//	int delay=1000/rate;//每帧之间的延迟与视频的帧率相对应
//	double width=cap.get(CV_CAP_PROP_FRAME_WIDTH);//获取图像宽度
//	double height=cap.get(CV_CAP_PROP_FRAME_HEIGHT);//获取图像高度
//	//设置输出视频
//	VideoWriter outputVideo(ResultVideo, CV_FOURCC('M', 'J', 'P', 'G'), rate, Size(VideoOutputHight*width/height,VideoOutputHight));//写视频类
//
//	//变量定义
//	Mat src32FC3,threshold,edges;//定义帧：转换数据类型，灰度图，二值图，边缘图
//	Mat srcChannels[3];//RBG
//	Mat R,G,B;//RGB单通道
//	vector<Vec4i> lines,vlines,hlines;//lines线段矢量集合(列，行，列，行)，vlines：竖线集合，hlines：横线集合
//
//	//开始视频处理
//	bool stop = false;
//	while(!stop)
//	{
//		//获取图像
//		if(!cap.read(src)){cout<<"视频结束"<<endl;waitKey(0);break;}
//
//		//图像预处理
//		resize(src,src,Size(VideoOutputHight*width/height,VideoOutputHight));//调整大小
//		//GaussianBlur(src,src,Size(7,7),1.5);//高斯滤波
//		src.convertTo(src32FC3,CV_32FC3);//转换图像数据类型
//		split(src32FC3,srcChannels);//分离RGB通道
//		B = srcChannels[0];G = srcChannels[1];R = srcChannels[2];
//
//		//进行图像分割（色彩空间SVM方法）（矩阵操作方法，速度快）
//		threshold = resultMat.at<float>(0)*R + resultMat.at<float>(1)*G + resultMat.at<float>(2)*B + rho > -0.5;
//		//threshold = - 0.273370087*B - (-0.377990603)*G - 0.350543886*R + 32.809016951026422 > 0;//0.273370087等为xml中的数据
//
//		//形态学滤波
//		morphologyEx(threshold,threshold,MORPH_OPEN,getStructuringElement(MORPH_RECT, Size(9,9)));//形态学滤波：去噪
//		imshow("s22",threshold);
//		waitKey(1);
//
//		//形态学滤波：边缘检测
//		morphologyEx(threshold,edges,MORPH_GRADIENT ,getStructuringElement(MORPH_RECT, Size(5,5)));
//
//		//进行霍夫变换
//		HoughLinesP(edges,lines,1,CV_PI/360,48,48,0);
//		/*	第五个参数，Accumulator threshold parameter. Only those lines are returned that get enough votes (  ).
//			第六个参数，double类型的minLineLength，有默认值0，表示最低线段的长度，比这个设定参数短的线段就不能被显现出来。
//			第七个参数，double类型的maxLineGap，有默认值0，允许将同一行点与点之间连接起来的最大的距离*/
//
//		//找出竖线集合
//		vlines.clear();
//		vector<double> anglex;//竖线集合的角度
//		for( size_t i = 0; i < lines.size(); i++ )//找出竖线集合
//		{
//			Vec4i l = lines[i];
//			double angle = atan(double(l[2]-l[0])/double(l[3]-l[1]))/CV_PI*180;//此线与y轴所呈角度
//			if (abs(angle-angleBias)<15)
//			{
//				vlines.push_back(l);
//				anglex.push_back(abs(angle-angleBias));
//				line( src, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0,255,255), 1, CV_AA);//画线，point:(列，行)
//			}
//		}
//
//		//找出竖线
//		if (vlines.size()>0)
//		{
//			vector<double> anglexS(anglex);//角度从大到小排序
//			sort(anglexS.begin(),anglexS.end());
//
//			for (size_t i = 0;i < anglex.size();i++)
//			{
//				//cout<<anglexS[i]<<" ";
//				if (anglex[i] == anglexS[anglexS.size()-1])//确定竖线
//				{
//					//cout<<"minx"<<anglex[i];
//					Px1 = Point2i(vlines[i][0],vlines[i][1]);
//					Px2 = Point2i(vlines[i][2],vlines[i][3]);
//					xValue = (Px1.x+Px2.x)/2;
//
//					//判断竖线类型
//					int countLeft = 0,countRight = 0;//积分值
//					int areaLeft = 0,areaRight = 0;//面积
//					int f2a = (Px1.y-Px2.y)*(Px1.x-Px2.x);//分类函数系数
//					int f2b = -(Px1.y-Px2.y)*(Px1.y-Px2.y);
//					int f2c = (Px1.y-Px2.y)*(Px1.y*Px2.x-Px2.y*Px1.x);
//					for (int xi = 0;xi<threshold.cols;xi++)
//					{
//						for (int yi = 0;yi<threshold.rows;yi++)
//						{
//							if (f2a*yi+f2b*xi+f2c>0)//左
//							{
//								areaLeft++;
//								if (threshold.at<uchar>(yi,xi)!=0)
//								{
//									countLeft++;
//								} 
//							} 
//							else//右
//							{
//								areaRight++;
//								if (threshold.at<uchar>(yi,xi)!=0)
//								{
//									countRight++;
//								}
//							}
//
//						}
//					}
//					if (1.0*countLeft/areaLeft>0.6)//判断类型
//					{
//						cout<<"left"<<1.0*countLeft/areaLeft<<endl;
//						xSide = leftSide;
//						line( src, Px1, Px2, Scalar(255,0,0), 1, CV_AA);//画线，point:(列，行)
//
//					} 
//					else if (1.0*countRight/areaRight>0.6)
//					{
//						cout<<"right"<<1.0*countRight/areaRight<<endl;
//						xSide = rightSide;
//						line( src, Px1, Px2, Scalar(0,255,0), 1, CV_AA);//画线，point:(列，行)
//
//					}
//					else
//					{
//						cout<<"xfalse"<<endl;
//						xSide = noxSide;
//					}
//					break;
//				}
//			}
//		}
//		else
//		{
//			xValue = -1;
//			xSide = noxSide;
//			Px1 = Point2i(-1,-1);
//			Px2 = Point2i(-1,-1);
//		}
//		cout<<"xValue"<<xValue<<endl;
//		//找出横线集合
//		hlines.clear();
//		vector<double> angley;//横线集合的角度
//		for( size_t i = 0; i < lines.size(); i++ )//找出横线集合
//		{
//			Vec4i l = lines[i];
//			double angle = atan(double(l[2]-l[0])/double(l[3]-l[1]))/CV_PI*180;//此线与y轴所呈角度
//			if (abs(angle-90-angleBias)<15)
//			{
//				hlines.push_back(l);
//				angley.push_back(abs(angle-angleBias));
//				line( src, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0,255,255), 1, CV_AA);//画线，point:(列，行)
//			}
//		}
//
//		//找出横线
//		if (hlines.size()>0)
//		{
//			vector<double> angleyS(angley);//角度从大到小排序
//			sort(angleyS.begin(),angleyS.end());
//			for (size_t i = 0;i < angley.size();i++)
//			{
//				//cout<<angleyS[i]<<" ";
//				if (angley[i] == angleyS[angleyS.size()-1])//确定横线
//				{
//					//cout<<"miny"<<angley[i];
//					Py1 = Point2i(hlines[i][0],hlines[i][1]);
//					Py2 = Point2i(hlines[i][2],hlines[i][3]);
//					yValue = (Py1.y+Py2.y)/2;
//
//
//					//判断竖线类型
//					int countBottom = 0,countTop = 0;//积分值
//					int areaBottom = 0,areaTop = 0;//面积
//					int f1a = (Py1.x-Py2.x)*(Py1.y-Py2.y);//分类函数系数
//					int f1b = -(Py1.x-Py2.x)*(Py1.x-Py2.x);
//					int f1c = (Py1.x-Py2.x)*(Py1.x*Py2.y-Py2.x*Py1.y);
//					for (int xi = 0;xi<threshold.cols;xi++)
//					{
//						for (int yi = 0;yi<threshold.rows;yi++)
//						{
//							if (f1a*xi+f1b*yi+f1c<0)//下
//							{
//								areaBottom++;
//								if (threshold.at<uchar>(yi,xi)!=0)
//								{
//									countBottom++;
//								} 
//							} 
//							else//上
//							{
//								areaTop++;
//								if (threshold.at<uchar>(yi,xi)!=0)
//								{
//									countTop++;
//								}
//							}
//
//						}
//					}
//					if (1.0*countBottom/areaBottom>0.6)//判断类型
//					{
//						cout<<"Bottom"<<1.0*countBottom/areaBottom<<endl;
//						ySide = bottomSide;
//						line( src, Py1, Py2, Scalar(0,0,255), 1, CV_AA);//画线，point:(列，行)
//
//					} 
//					else if (1.0*countTop/areaTop>0.6)
//					{
//						cout<<"Top"<<1.0*countTop/areaTop<<endl;
//						ySide = topSide;
//						line( src, Py1, Py2, Scalar(255,255,0), 1, CV_AA);//画线，point:(列，行)
//
//					}
//					else
//					{
//						cout<<"yfalse"<<endl;
//						ySide = noySide;
//					}
//					break;
//				}
//			}
//		}
//		else
//		{
//			yValue = -1;
//			ySide = noySide;
//			Py1 = Point2i(-1,-1);
//			Py2 = Point2i(-1,-1);
//		}
//		cout<<"yValue"<<yValue<<endl;
//
//		//显示
//		namedWindow("threshold",WINDOW_NORMAL);
//		imshow("threshold",threshold);
//		namedWindow("edges",WINDOW_NORMAL);
//		imshow("edges",edges);
//		namedWindow("src",WINDOW_NORMAL);
//		imshow("src",src);
//
//
//		outputVideo<<src;
//
//		if(waitKey(100)>=0)//引入延迟，通过按键停止视频
//			stop = true;
//	}
//	return 0;
//}
