//--------------------------------------������˵����-------------------------------------------
//		������������ɫģ������������
//		�����������ò���ϵͳ�� Windows 7 32bit
//		������������IDE�汾��Visual Studio 2010
//		������������OpenCV�汾��	2.4.9
//------------------------------------------------------------------------------------------------



//---------------------------------��ͷ�ļ��������ռ�������֡�----------------------------
//		����������������ʹ�õ�ͷ�ļ��������ռ�
//------------------------------------------------------------------------------------------------
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/contrib/contrib.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>

using namespace std;
using namespace cv;


//-----------------------------------��ȫ�ֱ����������֡�-----------------------------------
//		������ȫ�ֱ�������
//-----------------------------------------------------------------------------------------------
Mat g_srcImage;  
Mat g_dstImage; 
//-----------------------------------��ȫ�ֺ����������֡�--------------------------------------
//		������ȫ�ֺ�������
//-----------------------------------------------------------------------------------------------


static void on_EllipticalModel();//��Բ��ɫģ�ͻص�����
static void on_GaussianModel();//��˹��ɫģ�ͻص�����
static void ShowHelpText();


//-----------------------------------��main( )������--------------------------------------------
//		����������̨Ӧ�ó������ں��������ǵĳ�������￪ʼ
//-----------------------------------------------------------------------------------------------
int main( )
{
	//�ı�console������ɫ
	system("color 2F");  

	ShowHelpText();

	//����ԭͼ
	g_srcImage = imread("face.jpg");
	if( !g_srcImage.data ) {
		printf("��ȡͼƬ����~�� \n"); 
		return false;
	}

	//��ʾԭʼͼ
	namedWindow("ԭʼͼ");
	imshow("ԭʼͼ", g_srcImage);


	//��ѯ��ȡ������Ϣ
	while(1)
	{
		int c;

		

		//��ȡ����
		c = waitKey(0);

		//���¼��̰���Q����ESC�������˳�
		if( (char)c == 'q'||(char)c == 27 )
			break;
		//���¼��̰���1��ʹ����Բ��ɫģ�ͽ����������
		else if( (char)c == 49 ){//���̰���2��ASII��Ϊ50
			on_EllipticalModel();
		}
		//���¼��̰���2��ʹ�ø�˹��ɫģ�ͽ����������
		else if( (char)c == 50 ){//���̰���3��ASII��Ϊ51
			on_GaussianModel();
		}	
	}
	destroyAllWindows;
	return 0;
}


//-----------------------------------��on_EllipticalModel( )������----------------------------------
//		��������Բ��ɫģ�͵Ļص�����
//-----------------------------------------------------------------------------------------------
static void on_EllipticalModel()
{
	Mat output_mask; 
	Mat output_image;  
/*******************************1.��Բģ��**********************************************/
	Mat skinCrCbHist = Mat::zeros(Size(256, 256), CV_8UC1);  //
	ellipse(skinCrCbHist, Point(113, 155.6), Size(23.4, 15.2), 43.0, 0.0, 360.0, Scalar(255, 255, 255),-1);  
	//imshow("��Բģ��",skinCrCbHist); 

	Mat ycrcb_image;  
	output_mask = Mat::zeros(g_srcImage.size(), CV_8UC1);  
	cvtColor(g_srcImage, ycrcb_image, CV_BGR2YCrCb); //����ת���ɵ�YCrCb�ռ�  

	for(int i = 0; i < g_srcImage.rows; i++) //������Բģ�ͽ���ɫ���  
	{  
		uchar* p = (uchar*)output_mask.ptr<uchar>(i);  //ָ��ָ��һά���ģ��ĵ�i��
		Vec3b* ycrcb = (Vec3b*)ycrcb_image.ptr<Vec3b>(i);  //ָ��ָ��ycrcb�ռ��ڴ���ͼ��ĵ�i��
		for(int j = 0; j < g_srcImage.cols; j++)  
		{  
			if(skinCrCbHist.at<uchar>(ycrcb[j][1], ycrcb[j][2]) > 0)  
				p[j] = 255;  //��ֵ��
		}  
	}     

	Mat element = getStructuringElement(MORPH_RECT, Size(3, 3), Point(-1, -1) ); 
	morphologyEx(output_mask,output_mask,MORPH_CLOSE,element);  //��һά���ģ�������̬ѧ�����㣬����С�ͺڶ�

/*******************************2.��Ⲣ��������**********************************************/
	vector<vector<Point>> contours;   // ����     
	vector<Vec4i> hierarchy;    // �����Ľṹ��Ϣ   
	vector<vector<Point>> filterContours; // ɸѡ�������  
	contours.clear();   //
	hierarchy.clear();   //
	filterContours.clear();  //������ر�����ʼ��Ϊ0

	findContours(output_mask,contours,hierarchy,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_SIMPLE);    
	/*ȥ��α����*/   
	for (size_t i = 0;i<contours.size();i++)   
	{  
		if (fabs(contourArea(Mat(contours[i])))>1000&&fabs(arcLength(Mat(contours[i]),true))<2000)  //�ж��ֽ����������ֵ  
			filterContours.push_back(contours[i]);  
	}  

	output_mask.setTo(0);  
	drawContours(output_mask, filterContours, -1, Scalar(255,0,0), CV_FILLED); //8, hierarchy); 
	//imshow("���ģ��",output_mask);
	g_srcImage.copyTo(output_image, output_mask);   
    //imshow("��Բ��ɫģ�ͼ������", output_image);  
	
	g_srcImage.copyTo(g_dstImage);
	for (size_t i = 0; i < filterContours.size(); i++)   
	{  
		Rect r2= boundingRect(Mat(filterContours[i])); //�õ���������
		 
		if(r2.height<r2.width*4){//�ų��������ĸ첲������Ч�������Ǻܺ�
		rectangle(g_dstImage,r2, Scalar(0,0,255),1.5);//������������
		}
	}   
	imshow("��Բģ�ͼ����",g_dstImage);
	
} 

//-----------------------------------��on_GaussianModel( )������----------------------------------
//		��������Բ��ɫģ�͵Ļص�����
//-----------------------------------------------------------------------------------------------
static void on_GaussianModel()
{
	double bmean=117.4361;
    double rmean=156.5599;
    double brcov[2][2]={160.1301,12.1430,12.1430,299.4574};//��˹ģ�Ͳ���


	Mat ycbcrImage;
    Mat gaussImage;
    Mat otsuImage;
	Mat output_image;

	cvtColor(g_srcImage, ycbcrImage,CV_BGR2YCrCb); //ת���ɵ�YCrCb�ռ� 
    
/*******************************1.�򵥸�˹ģ��**********************************************/
	vector<Mat> channels;
    split(ycbcrImage,channels);//ͨ���Ĳ�� 

    Mat imagey,imageCb,imageCr;

	//��ȡyͨ��������  
    imagey = channels.at(0);  
    //��ȡCbͨ��������  
    imageCb = channels.at(1);  
    //��ȡCrͨ��������  
    imageCr = channels.at(2); 
    
	double (*m_pLikeliHoodArray)[1000]= new double[1000][1000];//��ɫ��Ȼ����

	
	for(int i=0; i<g_srcImage.rows; i++)  //����Ycbcr�ռ�ļ򵥸�˹��ģ_�����ɫ��Ȼ����
	{
		for(int j=0; j<g_srcImage.cols; j++)
		{
			double x1,x2,t;    
	
			x1 =imageCr.at<uchar>(i,j)-bmean;
			x2 =imageCb.at<uchar>(i,j)-rmean;

			t = x1*(x1*brcov[1][1]-x2*brcov[1][0])+x2*(-x1*brcov[0][1]+x2*brcov[0][0]);
			t /= (brcov[0][0]*brcov[1][1]-brcov[0][1]*brcov[1][0]);
			t /= (-2);
			m_pLikeliHoodArray[i][j] = exp(t);//�����ɫ��Ȼ����
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
            m_pLikeliHoodArray[i][j]=m_pLikeliHoodArray[i][j]*255;//��ͼ�������ֵ��Χ��һ����0-255֮�䡣
		}
	}
	/*���������ɸ�˹ͼ��*/
	Mat imgGauss;
	imgGauss=Mat::zeros(g_srcImage.size(), CV_8UC1);

	for(int a=0;a<g_srcImage.rows;a++)
	{
		 for(int b=0;b<g_srcImage.cols;b++)
		 {
             imgGauss.at<uchar>(a,b)=m_pLikeliHoodArray[a][b];//����ɫ��Ȼͼ�������ֵ�浽��ʱͼ���Bͨ��
		}
	}
   //imshow("imgGauss",imgGauss);


	/**********************2.otsu�㷨��ֵ****************************/
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

	//ͳ�ƻҶȼ���ÿ������������ͼ���еĸ���
	for(i = 0; i < imgGauss.rows; i++)
	{
		for(j = 0;j < imgGauss.cols;j++)
		{
		pixelCount[(int)data[i * imgGauss.step+ j]]++;
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
				w0 += pixelPro[j];
				u0tmp += j * pixelPro[j];
			}
			else   //ǰ������
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
	
/****************************************3.OTSU��ֵ��******************************************************/
	 Mat imgOtsu =Mat::zeros(g_srcImage.size(), CV_8UC1);;
    // cvCopy(imgGauss,imgOtsu);
	 imgGauss.copyTo(imgOtsu);
     int threValue;
   
	 CvScalar otsu;
	 int m;
	 
	 for(int i=0;i<imgGauss.rows;i++)//  ��ֵ��
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

/****************************************3.��Ⲣ��������******************************************************/
	Mat element = getStructuringElement(MORPH_RECT, Size(3, 3), Point(-1, -1) );

	morphologyEx(imgOtsu,imgOtsu,MORPH_CLOSE,element);  //��һά���ģ�������̬ѧ�����㣬����С�ͺڶ�


	vector<vector<Point>> contours;   // ����     
	vector<Vec4i> hierarchy;    // �����Ľṹ��Ϣ   
	vector<vector<Point>> filterContours; // ɸѡ�������  
	contours.clear();   //
	hierarchy.clear();   //
	filterContours.clear();  //������ر�����ʼ��Ϊ0

	findContours(imgOtsu,contours,hierarchy,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_SIMPLE);    
	/*ȥ��α����*/   
	for (size_t i = 0;i<contours.size();i++)   
	{  
		if (fabs(contourArea(Mat(contours[i])))>1000&&fabs(arcLength(Mat(contours[i]),true))<2000)  //�ж��ֽ����������ֵ  
			filterContours.push_back(contours[i]);  
	}  

	imgOtsu.setTo(0);  
	drawContours(imgOtsu, filterContours, -1, Scalar(255,0,0), CV_FILLED); //8, hierarchy); 
//	imshow("���ģ��",imgOtsu);
	g_srcImage.copyTo(output_image, imgOtsu);   
//	imshow("��˹ģ�ͼ������", output_image);  
	//output_image.setTo(0); //????
	g_srcImage.copyTo(g_dstImage);
	for (size_t i = 0; i < filterContours.size(); i++)   
	{  
		Rect r2= boundingRect(Mat(filterContours[i])); //�õ���������
		 
	//	if(r2.height<r2.width*4){//�ų��������ĸ첲������Ч�������Ǻܺ�
		rectangle(g_dstImage,r2, Scalar(0,0,255),1.5);//������������
	//	}
	}   
	imshow("�򵥸�˹ģ�ͼ����",g_dstImage);
} 

//-----------------------------------��ShowHelpText( )������----------------------------------
//		���������һЩ������Ϣ
//----------------------------------------------------------------------------------------------
static void ShowHelpText()
{
	//���OpenCV�汾
	printf("\n\n\t\t\t   ��ǰʹ�õ�OpenCV�汾Ϊ��" CV_VERSION );
	printf("\n\n  ----------------------------------------------------------------------------\n");

	//���һЩ������Ϣ
	printf("\n\t��ɫģ������������\n\n");
	printf( "\n\t��������˵��: \n\n"
		"\t\t���̰�����ESC�����ߡ�Q��- �˳�����\n"
		"\t\t���̰�����1��- ʹ����Բ��ɫģ��\n"
		"\t\t���̰�����2��- ʹ�ü򵥸�˹��ɫģ��\n"
			);
}


