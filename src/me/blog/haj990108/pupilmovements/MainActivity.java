package me.blog.haj990108.pupilmovements;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedList;
import java.util.List;
import java.util.Queue;

import me.blog.haj990108.pupilmovements.R;
import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewFrame;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDouble;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.core.Core.MinMaxLocResult;
import org.opencv.highgui.Highgui;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.objdetect.Objdetect;

import android.app.Activity;
import android.content.Context;
import android.content.res.Configuration;
import android.graphics.Color;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.util.Log;
import android.view.Gravity;
import android.view.Menu;
import android.view.MenuItem;
import android.view.View;
import android.view.WindowManager;
import android.widget.Toast;

public class MainActivity extends Activity implements CvCameraViewListener2 {
	 
    private static final String TAG = "OCVSample::Activity";
    private static final Scalar FACE_RECT_COLOR = new Scalar(0, 255, 0, 255);
    public static final int JAVA_DETECTOR = 0;
    private static final int TM_SQDIFF = 0;
    private static final int TM_SQDIFF_NORMED = 1;
    private static final int TM_CCOEFF = 2;
    private static final int TM_CCOEFF_NORMED = 3;
    private static final int TM_CCORR = 4;
    private static final int TM_CCORR_NORMED = 5;
 
    private int learn_frames = 1;
    private Mat teplateR;
    private Mat teplateL;
    int method = 0;//추적방식 0~5
 
    private MenuItem mItemFace50;
    private MenuItem mItemFace40;
    private MenuItem mItemFace30;
    private MenuItem mItemFace20;
    private MenuItem mItemType;
 
    private Mat mRgba;
    private Mat mGray;
    
    
    // matrix for zooming
    private Mat mZoomWindow;
    //private Mat mZoomWindow2;
 
    private File mCascadeFile;
    private CascadeClassifier mJavaDetector;
    private CascadeClassifier mJavaDetectorEyeL;
    private CascadeClassifier mJavaDetectorEyeR;
    
    
    private boolean isLineVisible = true;
    private boolean isZoomwindowVisible = true;
    
    private double rMean = 0;
    
     
    private int mDetectorType = JAVA_DETECTOR;
    private String[] mDetectorName;
 
    private float mRelativeFaceSize = 0.5f;
    private int mAbsoluteFaceSize = 0;
 
    private CameraBridgeViewBase mOpenCvCameraView;
    
    private static boolean isTablet;
    
    private  boolean isClickEyeLeft = true;//클릭인식 기준 눈이 왼쪽인가
    private  boolean isCursorEyeLeft = true;//커서이동 기준 눈이 왼쪽인가
 
    double xCenter = -1;
    double yCenter = -1;
    
    private int isLeft = 0, isUp = 0;
    private boolean isWink = false;
    private ArrayList<Integer> rList = new ArrayList<Integer>();//마이닝되니 데이터 저장
    private static int learn_framesMAX = 30;//이 기간 이상 고정시 확정
    private int highY, lowY;
    private int learn_course = 0;//0:없음, 1:위, 2:아래 얻음
    private boolean isFaceVisible = true;
    private static int cameraWidth = 352;//1280*720
    private static int cameraHeight = 288;//352*288
    
    static {
        if (!OpenCVLoader.initDebug()) {
            // Handle initialization error
        	Log.e("TAG", "omg! - OPENCV initialization error");
        }else{
        	Log.i("TAG", "omg! - OPENCV initialization success");
        }
    }//http://docs.opencv.org/doc/tutorials/introduction/android_binary_package/dev_with_OCV_on_Android.html#application-development-with-static-initialization
 
    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
            case LoaderCallbackInterface.SUCCESS: {
                Log.i(TAG, "OpenCV loaded successfully");
 
 
                try {
                    //==얼굴전면==//
                    InputStream is = getResources().openRawResource(
                            R.raw.lbpcascade_frontalface);
                    File cascadeDir = getDir("cascade", Context.MODE_PRIVATE);
                    mCascadeFile = new File(cascadeDir,
                            "lbpcascade_frontalface.xml");
                	/*InputStream is = getResources().openRawResource(
                            R.raw.haarcascade_eye_tree_eyeglasses);
                    File cascadeDir = getDir("cascade", Context.MODE_PRIVATE);
                    mCascadeFile = new File(cascadeDir,
                            "haarcascade_eye_tree_eyeglasses.xml");*/
                    //안경있는 애들 xml파일인데 안경 없으면 인식 안됨. 안경 있으면 되나?
                    FileOutputStream os = new FileOutputStream(mCascadeFile);
 
                    byte[] buffer = new byte[4096];
                    int bytesRead;
                    while ((bytesRead = is.read(buffer)) != -1) {
                        os.write(buffer, 0, bytesRead);
                    }
                    is.close();
                    os.close();
 
                    //==우측눈==//
                    InputStream iser = getResources().openRawResource(
                            R.raw.haarcascade_righteye_2splits);
                    File cascadeDirER = getDir("cascadeER",
                            Context.MODE_PRIVATE);
                    File cascadeFileER = new File(cascadeDirER,
                            "haarcascade_eye_right.xml");
                    FileOutputStream oser = new FileOutputStream(cascadeFileER);
 
                    byte[] bufferER = new byte[4096];
                    int bytesReadER;
                    while ((bytesReadER = iser.read(bufferER)) != -1) {
                        oser.write(bufferER, 0, bytesReadER);
                    }
                    iser.close();
                    oser.close();
                    
                    //==좌측눈==//
                    InputStream isel = getResources().openRawResource(
                            R.raw.haarcascade_lefteye_2splits);
                    File cascadeDirEL = getDir("cascadeER",
                            Context.MODE_PRIVATE);
                    File cascadeFileEL = new File(cascadeDirEL,
                            "haarcascade_eye_right.xml");
                    FileOutputStream osel = new FileOutputStream(cascadeFileEL);
                     
                    byte[] bufferEL = new byte[4096];
                    int bytesReadEL;
                    while ((bytesReadEL = isel.read(bufferEL)) != -1) {
                    	osel.write(bufferEL, 0, bytesReadEL);
                    }
                    isel.close();
                    osel.close();
                    //-- end --//
 
                    mJavaDetector = new CascadeClassifier(
                            mCascadeFile.getAbsolutePath());
                    if (mJavaDetector.empty()) {
                        Log.e(TAG, "Failed to load cascade classifier");
                        mJavaDetector = null;
                    } else
                        Log.i(TAG, "Loaded cascade classifier from "
                                + mCascadeFile.getAbsolutePath());
                    
                    mJavaDetectorEyeR = new CascadeClassifier(
                            cascadeFileER.getAbsolutePath());
                    if (mJavaDetectorEyeR.empty()) {
                        Log.e(TAG, "Failed to load cascade classifier");
                        mJavaDetectorEyeR = null;
                    } else
                        Log.i(TAG, "Loaded cascade classifier from "
                                + mCascadeFile.getAbsolutePath());
 
                    mJavaDetectorEyeL = new CascadeClassifier(
                            cascadeFileEL.getAbsolutePath());
                    if (mJavaDetectorEyeL.empty()) {
                        Log.e(TAG, "Failed to load cascade classifier");
                        mJavaDetectorEyeL = null;
                    } else
                        Log.i(TAG, "Loaded cascade classifier from "
                                + mCascadeFile.getAbsolutePath());
 
                    cascadeDir.delete();
 
                } catch (IOException e) {
                    e.printStackTrace();
                    Log.e(TAG, "Failed to load cascade. Exception thrown: " + e);
                }
                mOpenCvCameraView.setCameraIndex(1);
                mOpenCvCameraView.setMaxFrameSize(352,288);// 최대 크기 지정 (1280,720)6fps  (768,512)12fps
                //720*720 미만으로 가니까 자꾸 에러가 뜨더라.... -> 카메라 크기가 352*288로 지정이 되는데, 원래 mat 사이즈로 resize를 안하니까!!!! 이거 정사각으로 뜸
                //mOpenCvCameraView.setMaxFrameSize(440, 330);
                mOpenCvCameraView.enableFpsMeter();
                mOpenCvCameraView.enableView();//뷰 활성화
 
            }
                break;
            default: {
                super.onManagerConnected(status);
            }
                break;
            }
        }
    };
 
    public MainActivity() {
        mDetectorName = new String[2];
        mDetectorName[JAVA_DETECTOR] = "Java";
 
        Log.i(TAG, "Instantiated new " + this.getClass());
    }
 
    /** Called when the activity is first created. */
    @Override
    public void onCreate(Bundle savedInstanceState) {
        Log.i(TAG, "called onCreate");
        
        super.onCreate(savedInstanceState);
        
        isTablet = isTablet(getApplicationContext());
        
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        
        setContentView(R.layout.activity_main);
        
        mOpenCvCameraView = (CameraBridgeViewBase) findViewById(R.id.NativeCameraView1);
        mOpenCvCameraView.setCvCameraViewListener(this);
 
        
    }
 
    @Override
    public void onPause() {
        super.onPause();
        if (mOpenCvCameraView != null)  mOpenCvCameraView.disableView();
    }
 
    @Override
    public void onResume() {
        super.onResume();
        /*OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_2_4_9, this,
                mLoaderCallback);*/
        mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
    }
 
    public void onDestroy() {
        super.onDestroy();
        mOpenCvCameraView.disableView();
    }
 
    public void onCameraViewStarted(int width, int height) {
        mGray = new Mat();
        mRgba = new Mat();
    }
 
    public void onCameraViewStopped() {
        mGray.release();
        mRgba.release();
        if(isZoomwindowVisible)
        	mZoomWindow.release();
        
        //mZoomWindow2.release();
    }
    static boolean isTablet (Context context) {
        int xlargeBit = 4; // Configuration.SCREENLAYOUT_SIZE_XLARGE;  // upgrade to HC SDK to get this 
        Configuration config = context.getResources().getConfiguration(); 
        return (config.screenLayout & xlargeBit) == xlargeBit; 
    }
    public Mat onCameraFrame(CvCameraViewFrame inputFrame) {
    	//Winklick의 faceService와 같다. 실험용
    	
        mRgba = inputFrame.rgba();
        mGray = inputFrame.gray();//(352,288)
        
       
        
        Core.flip(mRgba, mRgba, 1);
        Core.flip(mGray, mGray, 1);//(352,288)
        
        if(!isTablet){//스마트폰에 맞게 화면회전
        	mRgba = mRgba.t();
            mGray = mGray.t();//(288,352)
            Core.flip(mRgba, mRgba, 1);
            Core.flip(mGray, mGray, 1);//(288,352)
        }
        
 
        if (mAbsoluteFaceSize == 0) {
            int height = mGray.rows();
            if (Math.round(height * mRelativeFaceSize) > 0) {
                mAbsoluteFaceSize = Math.round(height * mRelativeFaceSize);
            }
        }
        
        MatOfRect faces = new MatOfRect();
 
        if (mJavaDetector != null)
            mJavaDetector.detectMultiScale(mGray, faces, 1.1, 2,
                    2,
                    new Size(mAbsoluteFaceSize, mAbsoluteFaceSize),
                    new Size());
 
        Rect[] facesArray = faces.toArray();
        if(facesArray.length > 0){
        	
        	DETECT_EYE_MOTION(facesArray);
            
        	//mRgba = mGray.submat(originalEyeArea(r,isClickEyeLeft));
            
            /*if ( getClickNum(shrinkArea(r,isClickEyeLeft)) <= 4 && getClickNum(shrinkArea(r,!isClickEyeLeft))>4){
            	Log.e("MOVE", "CLICK = ("+getClickNum(shrinkArea(r,isClickEyeLeft))+", "+getClickNum(shrinkArea(r,!isClickEyeLeft))+")");//잘 작동
            	isWink = true;
            	isUp = isLeft = 0;
            	
            }else{
            	isWink = false;
            	Imgproc.resize(getCursorReturnMat(shrinkArea(r,isCursorEyeLeft)), mRgba, mRgba.size());//커서정보(이동)+머신러닝
            	
            }
            if(isWink){
            	Log.d("WINK", "wink = "+"W");
            }else{
            	Log.d("WINK", "wink = "+"-");
            }
            String s = new String();
            switch(isLeft){
            case 1:
            	s = "L";
            	break;
            case -1:
            	s = "R";
            	break;
            case 0:
            	s = "-";
            	break;
            }
            switch(isUp){
            case 1:
            	s += "U";
            	break;
            case -1:
            	s += "D";
            	break;
            case 0:
            	s += "-";
            	break;
            }
            Log.d("MOVE", s);*/
            
            
        }else{
        	
        	
        	blackLightCompensation(mGray, mGray);// TODO : mGray를 BLC시킴. blackLightCompensation
        	if (mJavaDetector != null)
                mJavaDetector.detectMultiScale(mGray, faces, 1.1, 2,
                        2,
                        new Size(mAbsoluteFaceSize, mAbsoluteFaceSize),
                        new Size());
        	facesArray = faces.toArray();
        	
        	if(facesArray.length > 0){
        		Log.d("TAG", "BLC");
        		DETECT_EYE_MOTION(facesArray);
        	}else{
        		toastShowUI("얼굴을 찾을 수 없습니다!");
        	}
        	//global.setFaceExist(false);
        }
        
        //mGray = null;//onCameraViewStopped의 mGray.release()에서 NullPointerException 오류가 생긴다
        
        if(!isFaceVisible){
        	Imgproc.resize(mRgba, mRgba, new Size(1,1));//리턴값을 투명하게 하는 대신 1,1사이즈로 해서 안보이게함
        }else{
        	Imgproc.resize(mRgba, mRgba, new Size(352,288));//TODO : 아직도 isFaceVisible = false하면 Assertion Failed 뜸
        }
        return mRgba;
    }
    
    private void DETECT_EYE_MOTION (Rect[] facesArray){
    	if (isLineVisible)
    		Core.rectangle(mRgba, facesArray[0].tl(), facesArray[0].br(),
                FACE_RECT_COLOR, 3);//얼굴 면적을 초록 사각형으로 표시
        
        xCenter = (facesArray[0].x + facesArray[0].width + facesArray[0].x) / 2;
        yCenter = (facesArray[0].y + facesArray[0].y + facesArray[0].height) / 2;
        
        if (isLineVisible){
            Point center = new Point(xCenter, yCenter);//얼굴 중점 그리고 중점에 미니원(뽀대용)을 그린다
            
            Core.circle(mRgba, center, 10, new Scalar(255, 0, 0, 255), 3);
            
            Core.putText(mRgba, "[" + center.x + "," + center.y + "]",
                    new Point(center.x + 20, center.y + 20),
                    Core.FONT_HERSHEY_SIMPLEX, 0.7, new Scalar(255, 255, 255,
                            255));//중점 옆에 위치좌표를 보여주는 텍스트 표시
        }
        

        Rect r = facesArray[0];
        Mat smp = mGray.submat(originalEyeArea(r,true));
        blackLightCompensation(smp, smp);
        //Imgproc.dilate(smp, smp, Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(2,2)));//검은색 줄이기
        
        //final int threshHold = 70;//클수록 어두워짐
        //Imgproc.threshold(smp, smp, threshHold, 255, Imgproc.THRESH_BINARY);
        
        //Imgproc.equalizeHist(smp, smp);//이거도 너무 검게나옴
        //blackLightCompensation(smp, smp);//눈이미지에 하는거 안정성 떨어짐
        mRgba = smp;
        
    }
    
    private void blackLightCompensation(Mat src, Mat dst)
    {
        // rate : 0~1의 실수로서 이 rate보다 어두운 빛은 전부다 위로 올려버린다.
		//if(rate >= 1 || rate < 0) Log.e("TAG", "Rate isn't on 0~1 at blackLightCompensation");
    	if(src.empty() || src == null) Log.e("TAG", "Empty Mat at blackLightCompensation");
    	
    	Mat C = src.clone();
    	
    	src.convertTo(src, CvType.CV_64FC1); // New line added. 
    	int size = (int) (src.total() * src.channels());
    	double[] temp = new double[size]; // use double[] instead of byte[]
    	src.get(0, 0, temp);
    	
    	double m = 0;
    	for (int i = 0; i < size; i++) m += temp[i];
    	m /= size;//part A
    	
    	double ratio = 127.5/m;
    	for (int i = 0; i < size; i++){
    		if(temp[i] * ratio >= 255){
    			temp[i] = 255;
    		}else{
    			temp[i] *= ratio;
    		}
    	}//part B
    	
    	/*int divby = 5;//divby당 1번의 픽셀만 더한다
    	for (int i = 0; i < size; i++) if(i % divby == 0) m += temp[i];
    	m /= Math.floor(size/divby);*/
    	
    	//part A, B 둘 다 8.9fps를 만드는데 영향을 줌. 둘다 속도 부하점
    	C.put(0, 0, temp);
    	C.convertTo(src, CvType.CV_8U);
    	dst = C;
    }
    
    private Mat getCursorReturnMat(Rect area)
    {
    	////isCursorNotClick = true
    	Mat C = mGray.submat(area);
    	
    	//code
    	Imgproc.equalizeHist(C, C);
    	upBriCon(C,C, 30, 1);
    	Imgproc.erode(C, C, Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(2,2)));//살붙이기
    	Imgproc.threshold(C, C, 65, 255, Imgproc.THRESH_BINARY);//Imgproc.threshold(C, C, 50, 255, Imgproc.THRESH_BINARY);
    	//값을 크게하면 어둡게 됨
    	
    	//Log.i("TAG", "CTYPE = "+C.type());//0
    	//Log.i("TAG", "colorwhite = "+colorArr[0]);//백색uchar 255.0
    	
    	//=== 머리카락 제거 ===//
    	//색채우기 알고리즘  http://ko.wikipedia.org/wiki/플러드_필 (알고리즘 특성상 mask는 가로세로 크기가 원본c보다 +2씩 커야함)
    	Floodfill : for(int x = 1; x < C.width() + 1; x++){
        	double colorArr[] = C.get(1,x);//배열길이 = 1
    		if (colorArr != null && colorArr[0] < 10){
    			Imgproc.floodFill(C, Mat.zeros(C.rows() + 2, C.cols() + 2, CvType.CV_8U), new Point(x,1), new Scalar(255,255,255));
    			break Floodfill;
    		}
        	
        }
    	
    	//=== 눈이 영역 밖으로 나갔는지 조사 ===//
    	if(isOutStrict(C) != 0){
    		if(isFaceVisible){
    			toastShowUI("테두리 안에 눈을 맞춰주세요!");
    		}else{
    			toastShowUI("눈과 카메라의 높이를 맞춰주세요!");
    		}
    		
    		Imgproc.resize(mGray, C, new Size(cameraWidth, cameraHeight));
    		//네모 영역 차이가 나는 이유 : 눈 mat는 equalizeHist된 상태. 나머지는 그냥 흑백
    		//Area밖으로 나갔을 경우. isFaceVisible==false여도 무조건 Mat가 보이게 해야 함
    		Imgproc.cvtColor(C, C, Imgproc.COLOR_GRAY2RGBA);
    		
    		rList.clear();//머신러닝 한거 다 제거
	    	return C;
    	}
    	
    	
    	//=== 눈 양 끝점, 눈 중앙 ===//
		
		Point E1 = new Point(0,0);
		Point E2 = new Point(0,0);
		Point eyeCenter = new Point(0,0);//눈 중점 (원점)
		
		E1Loop : for(int x = 1; x < C.width() + 1; x++){

            for(int y = C.height(); y > 0; y--){
            	
            	double colorArr[] = C.get(y,x);//배열길이 = 1
            	
            	if (colorArr != null && colorArr[0] < 10){
            		E1 = new Point(x,y);
            		//Core.circle(C, new Point(x, y), 3, new Scalar(100, 100, 100, 255), 1);
            		break E1Loop;
            	}

            }
        }//눈 왼쪽 E1
		
		E2Loop : for(int x = C.width(); x > 0; x--){

            for(int y = C.height(); y > 0; y--){
            	
            	double colorArr[] = C.get(y,x);//배열길이 = 1
            	
            	if (colorArr != null && colorArr[0] < 10){
            		E2 = new Point(x,y);
            		//Core.circle(C, new Point(x, y), 3, new Scalar(100, 100, 100, 255), 1);
            		break E2Loop;
            	} 

            }
        }//눈 오른쪽 E2
        eyeCenter = new Point( 0.5 * (E1.x + E2.x),  0.5 * (E1.y + E2.y));
    	
        
        
        //===== 좌우상하 커서이동 조사 =====//
    	//=== 좌우이동 ===//
        int blackA = 0;//왼쪽
        int blackB = 0;//가운데
        int blackC = 0;//오른쪽
        
        for(int x = (int)E1.x; x < (int)E2.x + 1; x++){
        	for(int y = C.height(); y > 0; y--){
            	double colorArr[] = C.get(y,x);//배열길이 = 1
            	
            	if (colorArr != null && colorArr[0] < 10){
            		//x줄에서 검은색 발견시
            		if(x < (int)((E2.x - E1.x)/3) + E1.x){
            			blackA ++;//왼쪽
            		}else if(x > (int)(2 * (E2.x - E1.x)/3) + E1.x){
            			blackC ++;//오른쪽
            		}else{
            			blackB ++;//가운데
            		}
            	}
            }
        }
        
        if(Math.max(blackA, Math.max(blackB, blackC)) == blackA){
        	isLeft = 1;
        	//Log.i("MOVE", "L");
        }else if(Math.max(blackA, Math.max(blackB, blackC)) == blackC){
        	isLeft = -1;
        	//Log.i("MOVE", "R");
        }else{
        	isLeft = 0;
        	//Log.i("MOVE", "-");
        }
        
        //=== 상하이동 ===//
        //상하를 위한 밝은 스레시
        C = mGray.submat(area);
    	Imgproc.equalizeHist(C, C);
    	upBriCon(C,C, 30, 1);
    	Imgproc.erode(C, C, Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(2,2)));
    	
    	
    	
    	
    	Imgproc.threshold(C, C, 30, 255, Imgproc.THRESH_BINARY);//35
    	Floodfill : for(int x = 1; x < C.width() + 1; x++){
        	double colorArr[] = C.get(1,x);//배열길이 = 1
    		if (colorArr != null && colorArr[0] < 10){
    			Imgproc.floodFill(C, Mat.zeros(C.rows() + 2, C.cols() + 2, CvType.CV_8U), new Point(x,1), new Scalar(255,255,255));
    			break Floodfill;
    		}
        	
        }
    	
    	blackA = 0;//화면 맨 위.y
        blackB = 0;//화면 맨 아래.y//둘 사이에 있으면 고정
    	
    	//약한트레시의 상하 -> 윙크, 아래쳐다봄 구분 불가
    	for(int y = 1; y < C.height() + 1; y++){
        	L : for(int x = (int)E1.x; x < (int)E2.x + 1; x++){
            	double colorArr[] = C.get(y,x);//배열길이 = 1
            	
            	if (colorArr != null && colorArr[0] < 10){
            		//가로줄에서 검은색 발견시
            		if(blackA == 0){
            			blackA = y;//맨윗줄
            		}else{
            			blackB = y;//맨아랫줄
            		}
            		break L;
            	}
            }
        }
        Core.circle(C, E1, 2, new Scalar(100, 100, 100, 255), 1);
    	Core.circle(C, E2, 2, new Scalar(100, 100, 100, 255), 1);
    	Core.circle(C, new Point(0.5*C.width(),blackB), 2, new Scalar(100, 100, 100, 255), 1);
    	Core.circle(C, new Point(0.5*C.width(),0.5*(E1.y+E2.y)), 3, new Scalar(100, 100, 100, 255), 1);//세로원점
        
    	final int blackM = (int)(0.5*(E1.y+E2.y) - blackB);//세로 원점 기준 동공세로위치 
    	//TODO 세로
    	//이거 할때 가운데만 마이닝하고 최대 최소 구한 후 그 이상 및 이하가 나오면 위 아래로 인식하자
    	
    	if(learn_course < 2){//머신러닝 중이면 0:없음, 1:위, 2:아래 얻음(종료)
    		//Log.d("TAG", "Learning  = "+Math.abs(blackM - prevBlackM));//가속도 : 정지시 0~1, 움직이면 2~6
    		
    		Log.d("MOVE", "Learning  = "+blackM);//위치
    		
    		rList.add(blackM);//현재 위치 rList에 추가
    		
	    	if(rList.size() >= learn_framesMAX){//데이터 충분히 모이면
	    		//rList에서 툭 튀어나온 값 정리 (샘플링시 포함된 통계적 오류)
	    		//Log.e("MOVE", "UNsorted rList = "+rList.toString());
	    		for(int k=0; k<rList.size(); k++){
	    			if(Math.abs(rList.get(k) - mode(rList)) > 3){
	    				rList.remove(k);
	    			}
	    		}
	    		highY = Collections.max(rList);
	    		lowY = Collections.min(rList);
	    		Log.e("MOVE", "sorted rList = "+rList.toString());//위
	    		rList.clear();
	    		learn_course = 2;//끝냄
	    	}
    		
	    	isLeft = 0;//머신러닝중에는 화면도 머신러닝. 세로도 이동 불가하므로 가로도 이동 불가하게 하자
    	}else{
    		if(blackM > highY){
    			isUp = 1;
            	//Log.i("MOVE", "U");//위
    		}else if(blackM < lowY){
    			isUp = -1;
            	//Log.i("MOVE", "D");//아래
    		}else{
    			isUp = 0;
            	//Log.i("MOVE", "-");//가운데
    		}
    	}
    	
    	//global.setLearn_course(learn_course);
    	//나머지 isLeft, isUp, isWink의 업로드는 onCameraFrame에서 하고있다
    	Imgproc.cvtColor(C, C, Imgproc.COLOR_GRAY2RGBA);
    	return C;
    }
    
    private int isOutStrict(Mat C){
    	//흑백만 됨 (맨끝만 조사하지 말고 여러행 연속시 아웃처리하자)
    	int isOut = 0;
    	
    	Loop : for(int y = C.height(); y > 0; y--){
    		
        	double colorArr[] = C.get(y,1);//배열길이 = 1
        	if (colorArr != null && colorArr[0] < 10){
        		//x줄에서 검은색 발견시
        		isOut = 1;
        		break Loop;
        	}

        }
    	Loop : for(int y = C.height(); y > 0; y--){
    		
        	double colorArr[] = C.get(y,C.width());//배열길이 = 1
        	if (colorArr != null && colorArr[0] < 10){
        		//x줄에서 검은색 발견시
        		isOut = 2;
        		break Loop;
        	}

        }
        Loop : for(int x = C.width(); x > 0; x--){
    		
        	double colorArr[] = C.get(1,x);//배열길이 = 1
        	if (colorArr != null && colorArr[0] < 10){
        		//x줄에서 검은색 발견시
        		isOut = 3;
        		break Loop;
        	}

        }
        Loop : for(int x = C.width(); x > 0; x--){
    		
        	double colorArr[] = C.get(C.height(),x);//배열길이 = 1
        	if (colorArr != null && colorArr[0] < 10){
        		//x줄에서 검은색 발견시
        		isOut = 4;
        		break Loop;
        	}

        }
    	//0: 안튀어나옴
        //1: 왼쪽
        //2: 오른쪽
        //3: 위쪽
        //4: 아래쪽
    	return isOut;
    }
    
    private int mode(ArrayList<Integer> arr){
    	// arr에서 최빈값을 반환
    	ArrayList<Integer> freqArr = new ArrayList<Integer>();
    	for (int k = 0; k < arr.size(); k++){
    		freqArr.add(Collections.frequency(arr, arr.get(k)));
        }
    	
    	//arr.indexOf(object)
    	//Collections.frequency(arr, 2);
    	return arr.get(freqArr.indexOf(Collections.max(freqArr)));
    }
    
    private int getClickNum(Rect areaClick)
    {
    	//isCursorNotClick = false//areaClick=클릭담당 눈, areaOther=다른 눈을 받고 둘의 차이를 비교하는게 좋을 듯 하다TODO 
    	final int threshHold = 50;//클수록 어두워짐
    	boolean getClick = false;
    	Mat Cclick = mGray.submat(areaClick);
    	
    	
    	
    	
    	Imgproc.equalizeHist(Cclick, Cclick);
    	upBriCon(Cclick,Cclick, 30, 1);
    	Imgproc.erode(Cclick, Cclick, Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(2,2)));
    	Imgproc.threshold(Cclick, Cclick, threshHold, 255, Imgproc.THRESH_BINARY);//밝은 : Imgproc.threshold(C, C, 35, 255, Imgproc.THRESH_BINARY);
    	
    	
    	
    	
    	for(int y = 1; y < 3; y++){
    		Floodfill : for(int x = 1; x < Cclick.width() + 1; x++){//머리칼 제거
            	double colorArr[] = Cclick.get(1,x);//배열길이 = 1
	    		if (colorArr != null && colorArr[0] < 10){
	    			Imgproc.floodFill(Cclick, Mat.zeros(Cclick.rows() + 2, Cclick.cols() + 2, CvType.CV_8U), new Point(x,1), new Scalar(255,255,255));
	    			break Floodfill;
	    		}
	        }
    	}
    	int blackLine = 0;
    	for(int x = 1; x < Cclick.width() + 1; x++){
        	L:for(int y = Cclick.height(); y > 0; y--){
        		double colorArr[] = Cclick.get(y,x);//배열길이 = 1
            	if (colorArr != null && colorArr[0] < 10){
            		blackLine ++;
            		break L;
            	}
            }
        }
    	if(blackLine < (int)(Cclick.width()/3)){
    		//floodfill이후 동공 손상시 isWink = false처리후 종료
    		Imgproc.cvtColor(Cclick, Cclick, Imgproc.COLOR_GRAY2RGBA);
    		getClick = false;
    	}
        
        
    	
    	
    	int clickNum = 0;//세로 덩어리 최대 높이
        for(int x = 1; x < Cclick.width() + 1; x++){
        	int num = 0;
        	for(int y = Cclick.height(); y > 0; y--){
            	double colorArr[] = Cclick.get(y,x);//배열길이 = 1
            	if (colorArr != null && colorArr[0] < 10){//검은색
            		num++;
            	}else{//흰색
            		if(clickNum < num){
    	        		clickNum = num;
    	        	}
            		num = 0;
            	}
            }
        	if(clickNum < num){//최종시
        		clickNum = num;
        	}
        }
        //Log.i("MOVE", "click : other = "+clickNum+" : "+otherNum);
    	

    	return clickNum;
    }
    
    private Rect originalEyeArea(Rect area, boolean isEyeLeft)
    {
    	
    	if(isEyeLeft){
    		area = new Rect(area.x + area.width / 16,
                    (int) (area.y + (area.height / 3.8)),
                    (area.width - 2 * area.width / 16) / 2, (int) (area.height / 3.0));//eyearea_left
    	}else{
    		area = new Rect(area.x + area.width / 16
                    + (area.width - 2 * area.width / 16) / 2,
                    (int) (area.y + (area.height / 3.8)),
                    (area.width - 2 * area.width / 16) / 2, (int) (area.height / 3.0));//eyearea_right
    	}
		
		return area;
    }
    
    private Rect shrinkArea(Rect area, boolean isEyeLeft)
    {
    	
    	if(isEyeLeft){
    		area = new Rect(area.x + area.width / 16,
                    (int) (area.y + (area.height / 3.8)),
                    (area.width - 2 * area.width / 16) / 2, (int) (area.height / 3.0));//eyearea_left
    	}else{
    		area = new Rect(area.x + area.width / 16
                    + (area.width - 2 * area.width / 16) / 2,
                    (int) (area.y + (area.height / 3.8)),
                    (area.width - 2 * area.width / 16) / 2, (int) (area.height / 3.0));//eyearea_right
    	}
        
    	int lx = (int)(area.width * 30 / 164);
		int ly = (int)(area.width * 30 / 125);
		int rx = (int)(area.width * 130 / 164);
		int ry = (int)(area.width * 60 / 125);
		
		if(isEyeLeft){
			lx += 5;
			rx += 5;//5
		}
		
		Rect shrinkArea = new Rect(area.x + lx,area.y + ly,rx-lx,ry-ly);
		
		return shrinkArea;
    }
    
    
    //--안내 메세지 관련--//
    private Toast toast = null;
    private void toastShowUI(final String message){
    	new Handler(Looper.getMainLooper()).post(new Runnable() { // new Handler and Runnable
            @Override
            public void run() {
                toastShow(message);
            }
    	});
    }
    private void toastShow(String message) {
    	if (toast == null) {
    		toast = Toast.makeText(this, message, Toast.LENGTH_SHORT);
    	}else{
    		toast.setText(message);
    	}
    	toast.setGravity(Gravity.CENTER, 0, 0);
    	toast.show();
    }
    //--안내 메세지 관련--//
    
    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        Log.i(TAG, "called onCreateOptionsMenu");
        mItemFace50 = menu.add("Face size 50%");
        mItemFace40 = menu.add("Face size 40%");
        mItemFace30 = menu.add("Face size 30%");
        mItemFace20 = menu.add("Face size 20%");
        mItemType = menu.add(mDetectorName[mDetectorType]);
        return true;
    }
 
    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        Log.i(TAG, "called onOptionsItemSelected; selected item: " + item);
        if (item == mItemFace50)
            setMinFaceSize(0.5f);
        else if (item == mItemFace40)
            setMinFaceSize(0.4f);
        else if (item == mItemFace30)
            setMinFaceSize(0.3f);
        else if (item == mItemFace20)
            setMinFaceSize(0.2f);
        else if (item == mItemType) {
            int tmpDetectorType = (mDetectorType + 1) % mDetectorName.length;
            item.setTitle(mDetectorName[tmpDetectorType]);
        }
        return true;
    }
 
    private void setMinFaceSize(float faceSize) {
        mRelativeFaceSize = faceSize;
        mAbsoluteFaceSize = 0;
    }
    
    private void upBriCon(Mat src, Mat dst, double bri, double con)
    {
        // TODO : fps가 9로 array로 접근하는거 보다 빠르다
    	//mGray.submat(area);
    	if(src.empty() || src == null)
    		Log.e("TAG", "Empty Mat on upBriCon");
    	
    	Mat C = src.clone();
    	
    	src.convertTo(src, CvType.CV_64FC1); // New line added. 
    	int size = (int) (src.total() * src.channels());
    	double[] temp = new double[size]; // use double[] instead of byte[]
    	src.get(0, 0, temp);
    	
    	for (int i = 0; i < size; i++)
    	   temp[i] = (con * temp[i] + bri);  // no more casting required.
    	C.put(0, 0, temp);
    	C.convertTo(src, CvType.CV_8U);
    	//Imgproc.cvtColor(C, C, Imgproc.COLOR_BGR2GRAY);//넣으면 안됨
    	//Imgproc.GaussianBlur(C, C, new Size(3, 3), 0);
    	//Imgproc.threshold(C, C, 80, 255, Imgproc.THRESH_BINARY);
    	dst = C;
    }
    
    private Mat get_template(CascadeClassifier clasificator, Rect area, int size) {
        Mat template = new Mat();
        Mat mROI = mGray.submat(area);
        MatOfRect eyes = new MatOfRect();
        Point iris = new Point();
        Rect eye_template = new Rect();
        clasificator.detectMultiScale(mROI, eyes, 1.15, 2,
                Objdetect.CASCADE_FIND_BIGGEST_OBJECT
                        | Objdetect.CASCADE_SCALE_IMAGE, new Size(30, 30),
                new Size());
 
        Rect[] eyesArray = eyes.toArray();
        for (int i = 0; i < eyesArray.length;) {
            Rect e = eyesArray[i];
            e.x = area.x + e.x;
            e.y = area.y + e.y;
            Rect eye_only_rectangle = new Rect((int) e.tl().x,
                    (int) (e.tl().y + e.height * 0.4), (int) e.width,
                    (int) (e.height * 0.6));
            mROI = mGray.submat(eye_only_rectangle);
            Mat vyrez = mRgba.submat(eye_only_rectangle);
             
             
            Core.MinMaxLocResult mmG = Core.minMaxLoc(mROI);
 
            
            iris.x = mmG.minLoc.x + eye_only_rectangle.x;
            iris.y = mmG.minLoc.y + eye_only_rectangle.y;
            eye_template = new Rect((int) iris.x - size / 2, (int) iris.y
                    - size / 2, size, size);
            if (isLineVisible) {
            	Core.circle(vyrez, mmG.minLoc, 2, new Scalar(255, 255, 255, 255), 2);
            	Core.rectangle(mRgba, eye_template.tl(), eye_template.br(),
                    new Scalar(255, 0, 0, 255), 2);
            }
            
            
            template = (mGray.submat(eye_template)).clone();
            return template;
        }
        return template;
    }
     
    public void onRecreateClick(View v)
    {
        learn_frames = 0;
    }
    
    
    
    
}