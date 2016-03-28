//--------------------------------------【程序说明】-------------------------------------------
//		程序描述：肤色模型人脸检测程序
//		开发测试所用操作系统： Windows 7 32bit
//		开发测试所用IDE版本：Visual Studio 2010
//		开发测试所用OpenCV版本：	2.4.9
//------------------------------------------------------------------------------------------------



//---------------------------------【头文件、命名空间包含部分】----------------------------
//		描述：包含程序所使用的头文件和命名空间
//------------------------------------------------------------------------------------------------
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/contrib/contrib.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>

using namespace std;
using namespace cv;


//-----------------------------------【全局变量声明部分】-----------------------------------
//		描述：全局变量声明
//-----------------------------------------------------------------------------------------------
Mat g_srcImage;  
Mat g_dstImage; 
//-----------------------------------【全局函数声明部分】--------------------------------------
//		描述：全局函数声明
//-----------------------------------------------------------------------------------------------


static void on_EllipticalModel();//椭圆肤色模型回调函数
static void on_GaussianModel();//高斯肤色模型回调函数
static void ShowHelpText();


//-----------------------------------【main( )函数】--------------------------------------------
//		描述：控制台应用程序的入口函数，我们的程序从这里开始
//-----------------------------------------------------------------------------------------------
int main( )
{
	//改变console字体颜色
	system("color 2F");  

	ShowHelpText();

	//载入原图
	g_srcImage = imread("face.jpg");
	if( !g_srcImage.data ) {
		printf("读取图片错误~！ \n"); 
		return false;
	}

	//显示原始图
	namedWindow("原始图");
	imshow("原始图", g_srcImage);


	//轮询获取按键信息
	while(1)
	{
		int c;

		

		//获取按键
		c = waitKey(0);

		//按下键盘按键Q或者ESC，程序退出
		if( (char)c == 'q'||(char)c == 27 )
			break;
		//按下键盘按键1，使用椭圆肤色模型进行人脸检测
		else if( (char)c == 49 ){//键盘按键2的ASII码为50
			on_EllipticalModel();
		}
		//按下键盘按键2，使用高斯肤色模型进行人脸检测
		else if( (char)c == 50 ){//键盘按键3的ASII码为51
			on_GaussianModel();
		}	
	}
	destroyAllWindows;
	return 0;
}


//-----------------------------------【on_EllipticalModel( )函数】----------------------------------
//		描述：椭圆肤色模型的回调函数
//-----------------------------------------------------------------------------------------------
static void on_EllipticalModel()
{
	Mat output_mask; 
	Mat output_image;  
/*******************************1.椭圆模型**********************************************/
	Mat skinCrCbHist = Mat::zeros(Size(256, 256), CV_8UC1);  //
	ellipse(skinCrCbHist, Point(113, 155.6), Size(23.4, 15.2), 43.0, 0.0, 360.0, Scalar(255, 255, 255),-1);  
	//imshow("椭圆模型",skinCrCbHist); 

	Mat ycrcb_image;  
	output_mask = Mat::zeros(g_srcImage.size(), CV_8UC1);  
	cvtColor(g_srcImage, ycrcb_image, CV_BGR2YCrCb); //首先转换成到YCrCb空间  

	for(int i = 0; i < g_srcImage.rows; i++) //利用椭圆模型进肤色检测  
	{  
		uchar* p = (uchar*)output_mask.ptr<uchar>(i);  //指针指向一维输出模板的第i行
		Vec3b* ycrcb = (Vec3b*)ycrcb_image.ptr<Vec3b>(i);  //指针指向ycrcb空间内待检图像的第i行
		for(int j = 0; j < g_srcImage.cols; j++)  
		{  
			if(skinCrCbHist.at<uchar>(ycrcb[j][1], ycrcb[j][2]) > 0)  
				p[j] = 255;  //二值化
		}  
	}     

	Mat element = getStructuringElement(MORPH_RECT, Size(3, 3), Point(-1, -1) ); 
	morphologyEx(output_mask,output_mask,MORPH_CLOSE,element);  //对一维输出模板进行形态学闭运算，消除小型黑洞

/*******************************2.检测并绘制轮廓**********************************************/
	vector<vector<Point>> contours;   // 轮廓     
	vector<Vec4i> hierarchy;    // 轮廓的结构信息   
	vector<vector<Point>> filterContours; // 筛选后的轮廓  
	contours.clear();   //
	hierarchy.clear();   //
	filterContours.clear();  //轮廓相关变量初始化为0

	findContours(output_mask,contours,hierarchy,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_SIMPLE);    
	/*去除伪轮廓*/   
	for (size_t i = 0;i<contours.size();i++)   
	{  
		if (fabs(contourArea(Mat(contours[i])))>1000&&fabs(arcLength(Mat(contours[i]),true))<2000)  //判断手进入区域的阈值  
			filterContours.push_back(contours[i]);  
	}  

	output_mask.setTo(0);  
	drawContours(output_mask, filterContours, -1, Scalar(255,0,0), CV_FILLED); //8, hierarchy); 
	//imshow("输出模板",output_mask);
	g_srcImage.copyTo(output_image, output_mask);   
    //imshow("椭圆肤色模型检测人脸", output_image);  
	
	g_srcImage.copyTo(g_dstImage);
	for (size_t i = 0; i < filterContours.size(); i++)   
	{  
		Rect r2= boundingRect(Mat(filterContours[i])); //得到外包络矩形
		 
		if(r2.height<r2.width*4){//排除部分误检的胳膊，不过效果好像不是很好
		rectangle(g_dstImage,r2, Scalar(0,0,255),1.5);//绘制外包络矩形
		}
	}   
	imshow("椭圆模型检测结果",g_dstImage);
	
} 

//-----------------------------------【on_GaussianModel( )函数】----------------------------------
//		描述：椭圆肤色模型的回调函数
//-----------------------------------------------------------------------------------------------
static void on_GaussianModel()
{
	double bmean=117.4361;
    double rmean=156.5599;
    double brcov[2][2]={160.1301,12.1430,12.1430,299.4574};//高斯模型参数


	Mat ycbcrImage;
    Mat gaussImage;
    Mat otsuImage;
	Mat output_image;

	cvtColor(g_srcImage, ycbcrImage,CV_BGR2YCrCb); //转换成到YCrCb空间 
    
/*******************************1.简单高斯模型**********************************************/
	vector<Mat> channels;
    split(ycbcrImage,channels);//通道的拆分 

    Mat imagey,imageCb,imageCr;

	//提取y通道的数据  
    imagey = channels.at(0);  
    //提取Cb通道的数据  
    imageCb = channels.at(1);  
    //提取Cr通道的数据  
    imageCr = channels.at(2); 
    
	double (*m_pLikeliHoodArray)[1000]= new double[1000][1000];//肤色似然概率

	
	for(int i=0; i<g_srcImage.rows; i++)  //基于Ycbcr空间的简单高斯建模_计算肤色似然概率
	{
		for(int j=0; j<g_srcImage.cols; j++)
		{
			double x1,x2,t;    
	
			x1 =imageCr.at<uchar>(i,j)-bmean;
			x2 =imageCb.at<uchar>(i,j)-rmean;

			t = x1*(x1*brcov[1][1]-x2*brcov[1][0])+x2*(-x1*brcov[0][1]+x2*brcov[0][0]);
			t /= (brcov[0][0]*brcov[1][1]-brcov[0][1]*brcov[1][0]);
			t /= (-2);
			m_pLikeliHoodArray[i][j] = exp(t);//计算肤色似然概率
		}
	}

	double max = 0.0;
	for(int i=0; i<g_srcImage.rows; i++)
		for(int j=0; j<g_srcImage.cols; j++)
			if(m_pLikeliHoodArray[i][j] > max) 
				max = m_pLikeliHoodArray[i][j];
	
	for(int i=0; i<g_srcImage.rows; i++)
	{
		for(int j=0; j<g_srcImage.cols; j++)
		{
			m_pLikeliHoodArray[i][j] /= max;
            m_pLikeliHoodArray[i][j]=m_pLikeliHoodArray[i][j]*255;//把图像的像素值范围归一化到0-255之间。
		}
	}
	/*创建并生成高斯图像*/
	Mat imgGauss;
	imgGauss=Mat::zeros(g_srcImage.size(), CV_8UC1);

	for(int a=0;a<g_srcImage.rows;a++)
	{
		 for(int b=0;b<g_srcImage.cols;b++)
		 {
             imgGauss.at<uchar>(a,b)=m_pLikeliHoodArray[a][b];//将肤色似然图像的像素值存到临时图像的B通道
		}
	}
   //imshow("imgGauss",imgGauss);


	/**********************2.otsu算法求阀值****************************/
	int width = imgGauss.cols;
	int height = imgGauss.rows;
	int pixelCount[256];
	float pixelPro[256];
	int i, j, pixelSum = imgGauss.cols*imgGauss.rows, threshold = 0;
	uchar* data = (uchar*)imgGauss.data;

	for(i = 0; i <256; i++)
	{
		pixelCount[i] = 0;
		pixelPro[i] = 0;
	}

	//统计灰度级中每个像素在整幅图像中的个数
	for(i = 0; i < imgGauss.rows; i++)
	{
		for(j = 0;j < imgGauss.cols;j++)
		{
		pixelCount[(int)data[i * imgGauss.step+ j]]++;
		}
	}

	//计算每个像素在整幅图像中的比例
	for(i = 0; i < 256; i++)
	{
		pixelPro[i] = (float)pixelCount[i] / pixelSum;
	}

	//遍历灰度级[0,255]
	float w0, w1, u0tmp, u1tmp, u0, u1, u, 
			deltaTmp, deltaMax = 0;
	for(i = 0; i < 256; i++)
	{
		w0 = w1 = u0tmp = u1tmp = u0 = u1 = u = deltaTmp = 0;
		for(j = 0; j < 256; j++)
		{
			if(j <= i)   //背景部分
			{
				w0 += pixelPro[j];
				u0tmp += j * pixelPro[j];
			}
			else   //前景部分
			{
				w1 += pixelPro[j];
				u1tmp += j * pixelPro[j];
			}
		}
		u0 = u0tmp / w0;
		u1 = u1tmp / w1;
		u = u0tmp + u1tmp;
		deltaTmp = 
			w0 * pow((u0 - u), 2) + w1 * pow((u1 - u), 2);
		if(deltaTmp > deltaMax)
		{
			deltaMax = deltaTmp;
			threshold = i;
		}
	}
	
/****************************************3.OTSU二值化******************************************************/
	 Mat imgOtsu =Mat::zeros(g_srcImage.size(), CV_8UC1);;
    // cvCopy(imgGauss,imgOtsu);
	 imgGauss.copyTo(imgOtsu);
     int threValue;
   
	 CvScalar otsu;
	 int m;
	 
	 for(int i=0;i<imgGauss.rows;i++)//  二值化
	 {
		 for(int j=0;j<imgGauss.cols;j++)
		 {
			// otsu=cvGet2D(imgOtsu,i,j);
			// m=(int)otsu.val[0];
			 m=(int)imgOtsu.at<uchar>(i,j);
			 if(m>=threshold)
				 m=255;
			 else
				 m=0;
			 //otsu.val[0]=m;
			// cvSet2D(imgOtsu,i,j,otsu);  
			 imgOtsu.at<uchar>(i,j)=m;

		 }
	 }
	// imshow("imgOtsu",imgOtsu);

/****************************************3.检测并绘制轮廓******************************************************/
	Mat element = getStructuringElement(MORPH_RECT, Size(3, 3), Point(-1, -1) );

	morphologyEx(imgOtsu,imgOtsu,MORPH_CLOSE,element);  //对一维输出模板进行形态学闭运算，消除小型黑洞


	vector<vector<Point>> contours;   // 轮廓     
	vector<Vec4i> hierarchy;    // 轮廓的结构信息   
	vector<vector<Point>> filterContours; // 筛选后的轮廓  
	contours.clear();   //
	hierarchy.clear();   //
	filterContours.clear();  //轮廓相关变量初始化为0

	findContours(imgOtsu,contours,hierarchy,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_SIMPLE);    
	/*去除伪轮廓*/   
	for (size_t i = 0;i<contours.size();i++)   
	{  
		if (fabs(contourArea(Mat(contours[i])))>1000&&fabs(arcLength(Mat(contours[i]),true))<2000)  //判断手进入区域的阈值  
			filterContours.push_back(contours[i]);  
	}  

	imgOtsu.setTo(0);  
	drawContours(imgOtsu, filterContours, -1, Scalar(255,0,0), CV_FILLED); //8, hierarchy); 
//	imshow("输出模板",imgOtsu);
	g_srcImage.copyTo(output_image, imgOtsu);   
//	imshow("高斯模型检测人脸", output_image);  
	//output_image.setTo(0); //????
	g_srcImage.copyTo(g_dstImage);
	for (size_t i = 0; i < filterContours.size(); i++)   
	{  
		Rect r2= boundingRect(Mat(filterContours[i])); //得到外包络矩形
		 
	//	if(r2.height<r2.width*4){//排除部分误检的胳膊，不过效果好像不是很好
		rectangle(g_dstImage,r2, Scalar(0,0,255),1.5);//绘制外包络矩形
	//	}
	}   
	imshow("简单高斯模型检测结果",g_dstImage);
} 

//-----------------------------------【ShowHelpText( )函数】----------------------------------
//		描述：输出一些帮助信息
//----------------------------------------------------------------------------------------------
static void ShowHelpText()
{
	//输出OpenCV版本
	printf("\n\n\t\t\t   当前使用的OpenCV版本为：" CV_VERSION );
	printf("\n\n  ----------------------------------------------------------------------------\n");

	//输出一些帮助信息
	printf("\n\t肤色模型人脸检测程序\n\n");
	printf( "\n\t按键操作说明: \n\n"
		"\t\t键盘按键【ESC】或者【Q】- 退出程序\n"
		"\t\t键盘按键【1】- 使用椭圆肤色模型\n"
		"\t\t键盘按键【2】- 使用简单高斯肤色模型\n"
			);
}


