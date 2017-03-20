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
//#define xMin 0						//场地范围#8.27
//#define xMax 20						//场地范围#8.27
//#define yMin 0						//场地范围#8.27
//#define yMax 20						//场地范围#8.27
//#define BIAS -0.5					//超平面偏移
//#define TestVideo "../Data/TestVideo/out9.15.avi"	//用于检测的测试视频
//#define ResultVideo "../Data/Result/out9.15.avi"		//测试视频的检测结果
//#define LoadSvmName "../Data/Result/SVM_HOG.xml"	//载入已有的模型文件名称
//
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
////计算点P0指向P1和P2所在直线的向量像素长度
//double computeDistance(Point2d P0,Point2d P1,Point2d P2);
////将像素长度转换为实际长度
//double imgDis2realDis(double h, double f,double imgDis);
//
//
////*******************************输入输出********************************
////输入
//Mat src;
//double quadHeight = 1.4;//四旋翼高度
//double camF = 419.0;//焦距的像素长度
//double angleBias = 0.0;//线角度矫正，即竖线在图像中应有的角度(逆时针为正)
////输出
//xSideType xSide;//竖线结果
//ySideType ySide;//横线结果
//Point2d Px1,Px2;//竖线段端点（没找到则返回-1,-1）
//Point2d Py1,Py2;//横线段端点（没找到则返回-1,-1）
//double xDis;//竖线与四旋翼距离（竖线在图像左边时为负）（没找到则返回10000）
//double yDis;//横线与四旋翼距离（横线在图像下边时为负）（没找到则返回10000）
//Point3d outputResult = Point3d(999,999,0);//输出坐标：x，y，标志位#8.27
////x：向上为正方向，y：向右为正方向(没有找到则为999)，标志位：(0-无边界，1-y更新，2-x更新，3-xy更新)#8.27
//
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
//	Point2d Pcenter = Point2d(VideoOutputHight*width/(2*height),VideoOutputHight/2);//图像中心点坐标
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
//		threshold = resultMat.at<float>(0)*R + resultMat.at<float>(1)*G + resultMat.at<float>(2)*B + rho > BIAS;
//		//threshold = - 0.273370087*B - (-0.377990603)*G - 0.350543886*R + 32.809016951026422 > 0;//0.273370087等为xml中的数据
//
//		imshow("s22",threshold);
//		waitKey(1);
//
//		//形态学滤波
//		morphologyEx(threshold,threshold,MORPH_CLOSE,getStructuringElement(MORPH_ELLIPSE, Size(9,9)));//形态学滤波：场外去噪
//		morphologyEx(threshold,threshold,MORPH_OPEN,getStructuringElement(MORPH_ELLIPSE, Size(11,11)));//形态学滤波：场内去噪
//		
//
//
//		//形态学滤波：边缘检测
//		morphologyEx(threshold,edges,MORPH_GRADIENT ,getStructuringElement(MORPH_RECT, Size(7,7)));
//
//		//进行霍夫变换
//		HoughLinesP(edges,lines,1,CV_PI/360,48,48,0);
//		/*	第五个参数，Accumulator threshold parameter. Only those lines are returned that get enough votes (  ).
//			第六个参数，double类型的minLineLength，有默认值0，表示最低线段的长度，比这个设定参数短的线段就不能被显现出来。
//			第七个参数，double类型的maxLineGap，有默认值0，允许将同一行点与点之间连接起来的最大的距离*/
//
//		//找出竖线集合
//		vlines.clear();
//		vector<double> anglex;//竖线集合与边界的夹角绝对值
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
//					Px1 = Point2d(vlines[i][0],vlines[i][1]);
//					Px2 = Point2d(vlines[i][2],vlines[i][3]);
//
//					//分类函数系数
//					int f2a = (Px1.y-Px2.y)*(Px1.x-Px2.x);
//					int f2b = -(Px1.y-Px2.y)*(Px1.y-Px2.y);
//					int f2c = (Px1.y-Px2.y)*(Px1.y*Px2.x-Px2.y*Px1.x);
//
//					//计算长度
//					double xImgDis = computeDistance(Pcenter,Px1,Px2);//计算中心点到竖线的像素长度
//					if (f2a*Pcenter.y+f2b*Pcenter.x+f2c<0)//距离赋符号：点在线右边
//						xImgDis = -xImgDis;
//					xDis = imgDis2realDis(quadHeight,camF,xImgDis * height/VideoOutputHight);//从像素距离转换到实际距离
//					cout<<"xImgDis:"<<xImgDis<<endl;
//					cout<<"xDis:"<<xDis<<endl;
//
//					//判断竖线类型
//					int countLeft = 0,countRight = 0;//积分值
//					int areaLeft = 0,areaRight = 0;//面积
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
//			xSide = noxSide;
//			Px1 = Point2d(-1,-1);
//			Px2 = Point2d(-1,-1);
//			xDis = 10000;
//		}
//
//		//找出横线集合
//		hlines.clear();
//		vector<double> angley;//横线集合和边界的夹角绝对值
//		for( size_t i = 0; i < lines.size(); i++ )//找出横线集合
//		{
//			Vec4i l = lines[i];
//			double angle = atan(double(l[2]-l[0])/double(l[3]-l[1]))/CV_PI*180;//此线与y轴所呈角度
//			if (abs(angle-90-angleBias)<15)
//			{
//				hlines.push_back(l);
//				angley.push_back(abs(angle-90-angleBias));
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
//					Py1 = Point2d(hlines[i][0],hlines[i][1]);
//					Py2 = Point2d(hlines[i][2],hlines[i][3]);
//
//					//分类函数系数
//					int f1a = (Py1.x-Py2.x)*(Py1.y-Py2.y);
//					int f1b = -(Py1.x-Py2.x)*(Py1.x-Py2.x);
//					int f1c = (Py1.x-Py2.x)*(Py1.x*Py2.y-Py2.x*Py1.y);
//
//					//计算长度
//					double yImgDis = computeDistance(Pcenter,Py1,Py2);//计算中心点到横线的像素长度
//					if (f1a*Pcenter.x+f1b*Pcenter.y+f1c>0)//距离赋符号：点在线上边
//						yImgDis = -yImgDis;
//					yDis = imgDis2realDis(quadHeight,camF,yImgDis * height/VideoOutputHight);//从像素距离转换到实际距离
//					cout<<"yImgDis:"<<yImgDis<<endl;
//					cout<<"yDis:"<<yDis;
//
//					//判断竖线类型
//					int countBottom = 0,countTop = 0;//积分值
//					int areaBottom = 0,areaTop = 0;//面积
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
//			ySide = noySide;
//			Py1 = Point2d(-1,-1);
//			Py2 = Point2d(-1,-1);
//			yDis = 10000;
//		}
//
//		//计算四旋翼坐标#8.27
//		Point3d detectResult = Point3d(999,999,0);//每帧初始化为无边界
//		//x：图像上向右为正方向，y：图像上向上为正方向(没有找到则为999)，标志位：(0-无边界，1-x更新，2-y更新，3-xy更新)#8.27
//
//		if (xSide != noxSide)//x更新#8.27
//		{
//			if (xSide == leftSide)
//			{
//				detectResult.x = xMin - xDis;
//			} 
//			else
//			{
//				detectResult.x = xMax - xDis;
//			}
//			detectResult.z = 1;
//		}
//		if (ySide != noySide)//y更新#8.27
//		{
//			if (ySide == bottomSide)
//			{
//				detectResult.y = yMin - yDis;
//			} 
//			else
//			{
//				detectResult.y = yMax - yDis;
//			}
//			detectResult.z = 2;
//		}
//		if(xSide != noxSide && ySide != noySide)//xy更新#8.27
//		{
//			detectResult.z = 3;
//		}
//		if(quadHeight<1)//如果高度太低，则视为误检测
//		{
//			detectResult.z = 0;
//		}
//		
//		outputResult = Point3d(detectResult.y,detectResult.x,detectResult.z);//输出坐标赋值
//		cout<<"x"<<outputResult.x<<"  y"<<outputResult.y<<"  flag"<<outputResult.z<<endl;//#8.27
//
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
//		if(waitKey(50)>=0)//引入延迟，通过按键停止视频
//			stop = true;
//	}
//	return 0;
//}
//
//double computeDistance(Point2d P0,Point2d P1,Point2d P2)
//{
//	double c_2 = (P0.x - P1.x)*(P0.x - P1.x) + (P0.y - P1.y)*(P0.y - P1.y);//斜边平方
//	double a = ((P2.x - P1.x)*(P0.x - P1.x) + (P2.y - P1.y)*(P0.y - P1.y))/sqrt((P2.x - P1.x)*(P2.x - P1.x)+(P2.y - P1.y)*(P2.y - P1.y));//直角边a
//
//	return sqrt(c_2-a*a);
//}
//double imgDis2realDis(double h, double f,double imgDis)
//{
//	return h*imgDis/f;
//}
