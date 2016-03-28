#include <stdio.h>
#include <cv.h>
#include <cxcore.h>
#include <highgui.h>
#include <iostream>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

double bmean=117.4361;//高斯模型参数
double rmean=156.5599;
double brcov[2][2]={160.1301,12.1430,12.1430,299.4574} ;

Mat pImg; //声明IplImage指针，输出图像
Mat imgGauss;
Mat imgOtsu;

struct TCbCr
{
	double Cb;
	double Cr;
} CbCr;

struct TCbCr CalCbCr(int B, int G, int R)
{
	struct TCbCr res;
	res.Cb =( 128 - 37.797 * R/255 - 74.203 * G/255 +   112 * B/255);
	res.Cr =( 128 + 112    * R/255 - 93.786 * G/255 -18.214 * B/255);
	return res;
}


*********************滤波*****************************/

void filter(double source[800][800],int m_nWidth,int m_nHeight)
{
	int x,y;
	double **temp;
	//申请一个临时二维数组
	temp = new  double*[m_nHeight+2];
	for(x=0;x <=m_nHeight+1; x++)
		temp[x] = new double[m_nWidth+2];

	//边界均设为0
	for(x=0; x<=m_nHeight+1; x++)
	{
		temp[x][0] = 0;
		temp[x][m_nWidth+1] = 0;
	}
	for(y=0; y<=m_nWidth+1; y++)
	{
		temp[0][y] = 0;
		temp[m_nHeight+1][y] = 0;
	}

	//将原数组的值赋予临时数组
	for(x=0; x<m_nHeight; x++)
		for(y=0; y<m_nWidth; y++)
			temp[x+1][y+1] = source[x][y];

	//均值滤波
	for(x=0; x<m_nHeight; x++)
	{
		for(y=0; y<m_nWidth; y++)
		{
			source[x][y] = 0;
			for(int k=0;k<=2;k++)
				for(int l=0;l<=2;l++)
					source[x][y] += temp[x+k][y+l];

			source[x][y] /= 9;
		}
	}

	if(temp!=NULL)
	{
		for(int x=0;x<=m_nHeight+1;x++)
			if(temp[x]!=NULL) delete temp[x];
		delete temp;
	}
}
**********************otsu算法求阀值****************************/
int otsuThreshold(IplImage *frame)
{

    int width = frame->width;//图像的宽
	int height = frame->height;//图像的高
	int pixelCount[256];//像素计数器
	float pixelPro[256];//区域占比
	int i, j, pixelSum = width * height, threshold = 0;
	uchar* data = (uchar*)frame->imageData;

	for(i = 0; i <256; i++)
	{
		pixelCount[i] = 0;
		pixelPro[i] = 0;
	}

	//统计灰度级中每个像素在整幅图像中的个数
	for(i = 0; i < height; i++)
	{
		for(j = 0;j < width;j++)
		{
		pixelCount[(int)data[i * frame->widthStep+ j]]++;
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
				w0 += pixelPro[j];//属于背景的像素值占比累加
				u0tmp += j * pixelPro[j];//属于背景的像素均值累加计算（未除w0）
			}
			else   //前景部分
			{
				w1 += pixelPro[j];//属于前景的像素值占比累加
				u1tmp += j * pixelPro[j];//属于前景的像素均值累加计算（未除w1）
			}
		}
		u0 = u0tmp / w0;//背景像素的均值
		u1 = u1tmp / w1;//前景像素的均值
		u = u0tmp + u1tmp;//整幅图像的均值
		deltaTmp = 
			w0 * pow((u0 - u), 2) + w1 * pow((u1 - u), 2);//计算方差
		if(deltaTmp > deltaMax)
		{
			deltaMax = deltaTmp;
			threshold = i;
		}
	}
	return threshold;
}

*******************************简单高斯模型**********************************************/
Mat likeliHood(Mat pImg)
{
    double (*m_pLikeliHoodArray)[800]= new double[800][800];

	int pImgH=pImg.rows;
	int pImgW=pImg.cols;
	CvScalar s;

    for(int i=0; i<pImgH; i++)     //基于Ycbcr空间的简单高斯建模
	{
		for(int j=0; j<pImgW; j++)
		{
			double x1,x2;    
           // s=cvGet2D(pImg,i,j);//取某像素的RGB值
			s=pImg.at<uchar>(i,j);//取某像素的RGB值
			TCbCr temp = CalCbCr(s.val[0],s.val[1],s.val[2]);//计算YCbCr值
			x1 = temp.Cb-bmean;
			x2 = temp.Cr-rmean;
			  double t;
			t = x1*(x1*brcov[1][1]-x2*brcov[1][0])+x2*(-x1*brcov[0][1]+x2*brcov[0][0]);
			t /= (brcov[0][0]*brcov[1][1]-brcov[0][1]*brcov[1][0]);
			t /= (-2);
			m_pLikeliHoodArray[i][j] = exp(t);//计算肤色似然概率
		}
	}

    filter(m_pLikeliHoodArray,pImgW,pImgH);//对肤色似然图像进行均值滤波

	double max = 0.0;
	for(int i=0; i<pImgH; i++)
		for(int j=0; j<pImgW; j++)
			if(m_pLikeliHoodArray[i][j] > max) 
				max = m_pLikeliHoodArray[i][j];
	
	for(int i=0; i<pImgH; i++)
	{
		for(int j=0; j<pImgW; j++)
		{
			m_pLikeliHoodArray[i][j] /= max;
            m_pLikeliHoodArray[i][j]=m_pLikeliHoodArray[i][j]*255;//把图像的像素值范围归一化到0-255之间。
		}
	}

    //IplImage *imgGauss = 0;
	//imgGauss = cvCreateImage(cvGetSize(pImg),IPL_DEPTH_8U,1);//创建高斯图像
	Mat imgGauss;
	imgGauss=Mat::zeros(pImg.size(), CV_8UC1);
    CvScalar imgTemp;
	 
	for(int a=0;a<pImg.rows;a++)
	{
		 for(int b=0;b<pImg.cols;b++)
		 {
			imgTemp = cvGet2D(imgGauss,a,b);
			imgTemp.val[0] = m_pLikeliHoodArray[a][b];//将肤色似然图像的像素值存到临时图像的B通道
            cvSet2D(imgGauss,a,b,imgTemp);
		
		}

	}

	delete[]m_pLikeliHoodArray;
	return imgGauss;
	

}

****************************************OTSU******************************************************/
IplImage* OTSU(Mat imgGauss)
{
	 Mat imgOtsu = cvCreateImage(cvGetSize(imgGauss),IPL_DEPTH_8U,1);
     cvCopy(imgGauss,imgOtsu);
     int threValue;
     threValue = otsuThreshold(imgOtsu);//调用otsu函数求阀值
     cout<<"The Threshold of this Image in Otsu is:"<<threValue<<endl;//输出显示阀值
   
	 CvScalar otsu;
	 int m;
	 
	 for(int i=0;i<imgGauss->height;i++)//  二值化
	 {
		 for(int j=0;j<imgGauss->width;j++)
		 {
			 otsu=cvGet2D(imgOtsu,i,j);
			 m=(int)otsu.val[0];
			 if(m>=threValue)
				 m=255;
			 else
				 m=0;
			 otsu.val[0]=m;
			 cvSet2D(imgOtsu,i,j,otsu);  
		 }
	 }
	 return imgOtsu;
}

*******************************************************************************************/
void operateStart()
{
	char strAddress[20];
	cout<<"                      简单高斯模型的肤色检测"<<endl;
	cout<<"请输入图像路径: 例如E:***.img"<<endl;
	cin>>strAddress;
	pImg = cvLoadImage( strAddress,1);//载入图像
	cout<<"information:"<<endl;//显示基本信息
	cout<<"height："<<pImg.rows<<endl;
	cout<<"width："<<pImg.cols<<endl;
	
}

void operateShow()
{
	namedWindow( "pImg", 1 );//创建窗口  
	imshow( "pImg", pImg );//显示原始图像
    namedWindow( "LikeliHoodArray", 1 );//创建窗口
    imshow( "LikeliHoodArray", imgGauss );//显示相似度图像
    namedWindow("imgOtsu", CV_WINDOW_AUTOSIZE );
    imshow( "imgOtsu", imgOtsu);//显示二值化图像    

    cvWaitKey(0); //等待按键

   // cvDestroyWindow( "pImg" );
   // cvDestroyWindow( "imgOtsu" );
  // cvDestroyWindow( "LikeliHoodArray" );//销毁窗口
   // cvReleaseImage( &imgGauss ); //释放图像
   // cvReleaseImage(&imgOtsu);
   // cvReleaseImage( &pImg );
}

int main() 

{	
	operateStart();

	imgGauss=likeliHood(pImg);

	imgOtsu=OTSU(imgGauss);
	
	operateShow();
    
	return 0;
    
}