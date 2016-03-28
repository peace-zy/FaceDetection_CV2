#include <stdio.h>
#include <cv.h>
#include <cxcore.h>
#include <highgui.h>
#include <iostream>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

double bmean=117.4361;//��˹ģ�Ͳ���
double rmean=156.5599;
double brcov[2][2]={160.1301,12.1430,12.1430,299.4574} ;

Mat pImg; //����IplImageָ�룬���ͼ��
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


*********************�˲�*****************************/

void filter(double source[800][800],int m_nWidth,int m_nHeight)
{
	int x,y;
	double **temp;
	//����һ����ʱ��ά����
	temp = new  double*[m_nHeight+2];
	for(x=0;x <=m_nHeight+1; x++)
		temp[x] = new double[m_nWidth+2];

	//�߽����Ϊ0
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

	//��ԭ�����ֵ������ʱ����
	for(x=0; x<m_nHeight; x++)
		for(y=0; y<m_nWidth; y++)
			temp[x+1][y+1] = source[x][y];

	//��ֵ�˲�
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
**********************otsu�㷨��ֵ****************************/
int otsuThreshold(IplImage *frame)
{

    int width = frame->width;//ͼ��Ŀ�
	int height = frame->height;//ͼ��ĸ�
	int pixelCount[256];//���ؼ�����
	float pixelPro[256];//����ռ��
	int i, j, pixelSum = width * height, threshold = 0;
	uchar* data = (uchar*)frame->imageData;

	for(i = 0; i <256; i++)
	{
		pixelCount[i] = 0;
		pixelPro[i] = 0;
	}

	//ͳ�ƻҶȼ���ÿ������������ͼ���еĸ���
	for(i = 0; i < height; i++)
	{
		for(j = 0;j < width;j++)
		{
		pixelCount[(int)data[i * frame->widthStep+ j]]++;
		}
	}

	//����ÿ������������ͼ���еı���
	for(i = 0; i < 256; i++)
	{
		pixelPro[i] = (float)pixelCount[i] / pixelSum;
	}

	//�����Ҷȼ�[0,255]
	float w0, w1, u0tmp, u1tmp, u0, u1, u, 
			deltaTmp, deltaMax = 0;
	for(i = 0; i < 256; i++)
	{
		w0 = w1 = u0tmp = u1tmp = u0 = u1 = u = deltaTmp = 0;
		for(j = 0; j < 256; j++)
		{
			if(j <= i)   //��������
			{
				w0 += pixelPro[j];//���ڱ���������ֵռ���ۼ�
				u0tmp += j * pixelPro[j];//���ڱ��������ؾ�ֵ�ۼӼ��㣨δ��w0��
			}
			else   //ǰ������
			{
				w1 += pixelPro[j];//����ǰ��������ֵռ���ۼ�
				u1tmp += j * pixelPro[j];//����ǰ�������ؾ�ֵ�ۼӼ��㣨δ��w1��
			}
		}
		u0 = u0tmp / w0;//�������صľ�ֵ
		u1 = u1tmp / w1;//ǰ�����صľ�ֵ
		u = u0tmp + u1tmp;//����ͼ��ľ�ֵ
		deltaTmp = 
			w0 * pow((u0 - u), 2) + w1 * pow((u1 - u), 2);//���㷽��
		if(deltaTmp > deltaMax)
		{
			deltaMax = deltaTmp;
			threshold = i;
		}
	}
	return threshold;
}

*******************************�򵥸�˹ģ��**********************************************/
Mat likeliHood(Mat pImg)
{
    double (*m_pLikeliHoodArray)[800]= new double[800][800];

	int pImgH=pImg.rows;
	int pImgW=pImg.cols;
	CvScalar s;

    for(int i=0; i<pImgH; i++)     //����Ycbcr�ռ�ļ򵥸�˹��ģ
	{
		for(int j=0; j<pImgW; j++)
		{
			double x1,x2;    
           // s=cvGet2D(pImg,i,j);//ȡĳ���ص�RGBֵ
			s=pImg.at<uchar>(i,j);//ȡĳ���ص�RGBֵ
			TCbCr temp = CalCbCr(s.val[0],s.val[1],s.val[2]);//����YCbCrֵ
			x1 = temp.Cb-bmean;
			x2 = temp.Cr-rmean;
			  double t;
			t = x1*(x1*brcov[1][1]-x2*brcov[1][0])+x2*(-x1*brcov[0][1]+x2*brcov[0][0]);
			t /= (brcov[0][0]*brcov[1][1]-brcov[0][1]*brcov[1][0]);
			t /= (-2);
			m_pLikeliHoodArray[i][j] = exp(t);//�����ɫ��Ȼ����
		}
	}

    filter(m_pLikeliHoodArray,pImgW,pImgH);//�Է�ɫ��Ȼͼ����о�ֵ�˲�

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
            m_pLikeliHoodArray[i][j]=m_pLikeliHoodArray[i][j]*255;//��ͼ�������ֵ��Χ��һ����0-255֮�䡣
		}
	}

    //IplImage *imgGauss = 0;
	//imgGauss = cvCreateImage(cvGetSize(pImg),IPL_DEPTH_8U,1);//������˹ͼ��
	Mat imgGauss;
	imgGauss=Mat::zeros(pImg.size(), CV_8UC1);
    CvScalar imgTemp;
	 
	for(int a=0;a<pImg.rows;a++)
	{
		 for(int b=0;b<pImg.cols;b++)
		 {
			imgTemp = cvGet2D(imgGauss,a,b);
			imgTemp.val[0] = m_pLikeliHoodArray[a][b];//����ɫ��Ȼͼ�������ֵ�浽��ʱͼ���Bͨ��
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
     threValue = otsuThreshold(imgOtsu);//����otsu������ֵ
     cout<<"The Threshold of this Image in Otsu is:"<<threValue<<endl;//�����ʾ��ֵ
   
	 CvScalar otsu;
	 int m;
	 
	 for(int i=0;i<imgGauss->height;i++)//  ��ֵ��
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
	cout<<"                      �򵥸�˹ģ�͵ķ�ɫ���"<<endl;
	cout<<"������ͼ��·��: ����E:***.img"<<endl;
	cin>>strAddress;
	pImg = cvLoadImage( strAddress,1);//����ͼ��
	cout<<"information:"<<endl;//��ʾ������Ϣ
	cout<<"height��"<<pImg.rows<<endl;
	cout<<"width��"<<pImg.cols<<endl;
	
}

void operateShow()
{
	namedWindow( "pImg", 1 );//��������  
	imshow( "pImg", pImg );//��ʾԭʼͼ��
    namedWindow( "LikeliHoodArray", 1 );//��������
    imshow( "LikeliHoodArray", imgGauss );//��ʾ���ƶ�ͼ��
    namedWindow("imgOtsu", CV_WINDOW_AUTOSIZE );
    imshow( "imgOtsu", imgOtsu);//��ʾ��ֵ��ͼ��    

    cvWaitKey(0); //�ȴ�����

   // cvDestroyWindow( "pImg" );
   // cvDestroyWindow( "imgOtsu" );
  // cvDestroyWindow( "LikeliHoodArray" );//���ٴ���
   // cvReleaseImage( &imgGauss ); //�ͷ�ͼ��
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