package me.blog.haj990108.pupilmovements;

import java.io.File;
import java.io.FileNotFoundException;
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
import org.opencv.android.Utils;
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
import android.graphics.Bitmap;
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

	private int learn_frames = 1;
	private Mat teplateR;
	private Mat teplateL;
	int method = 0;// 추적방식 0~5

	private Mat mRgba;
	private Mat mGray;

	// matrix for zooming
	private Mat mZoomWindow;
	// private Mat mZoomWindow2;

	private File mCascadeFile;
	private File mCascadeFileEL;
	private File mCascadeFileER;

	private CascadeClassifier mJavaDetector;
	private CascadeClassifier mJavaDetectorEL;
	private CascadeClassifier mJavaDetectorER;

	private boolean isLineVisible = true;
	private boolean isZoomwindowVisible = false;

	private double rMean = 0;

	private int mDetectorType = JAVA_DETECTOR;
	private String[] mDetectorName;

	private float mRelativeFaceSize = 0.5f;
	private int mAbsoluteFaceSize = 0;
	private int mAbsoluteEyeSize = 0;
	
	private Point eyeCL = new Point(0,0);
	private Point eyeCR = new Point(1,0); // 각도 돌릴때 에러 방지

	private CameraBridgeViewBase mOpenCvCameraView;

	private static boolean isTablet;

	private boolean isClickEyeLeft = true;// 클릭인식 기준 눈이 왼쪽인가
	private boolean isCursorEyeLeft = true;// 커서이동 기준 눈이 왼쪽인가

	double xCenter = -1;
	double yCenter = -1;

	private int isLeft = 0, isUp = 0;
	private boolean isWink = false;
	private ArrayList<Integer> rList = new ArrayList<Integer>();// 마이닝되니 데이터 저장
	private static int learn_framesMAX = 30;// 이 기간 이상 고정시 확정
	private int highY, lowY;
	private int learn_course = 0;// 0:없음, 1:위, 2:아래 얻음
	private boolean isFaceVisible = true;
	private static int cameraWidth = 352;// 1280*720
	private static int cameraHeight = 288;// 352*288

	static {
		if (!OpenCVLoader.initDebug()) {
			// Handle initialization error
			Log.e("TAG", "omg! - OPENCV initialization error");
		} else {
			Log.i("TAG", "omg! - OPENCV initialization success");
		}
	}// http://docs.opencv.org/doc/tutorials/introduction/android_binary_package/dev_with_OCV_on_Android.html#application-development-with-static-initialization

	private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
		@Override
		public void onManagerConnected(int status) {
			switch (status) {
			case LoaderCallbackInterface.SUCCESS: {
				Log.i(TAG, "OpenCV loaded successfully");

				// 얼굴 찾기 위한 cascade 불러옴

				File cascadeDir = getDir("cascade", Context.MODE_PRIVATE);
				mCascadeFile = new File(cascadeDir,
						"lbpcascade_frontalface.xml");

				File cascadeDirEL = getDir("cascadeEL", Context.MODE_PRIVATE);
				mCascadeFileEL = new File(cascadeDirEL,
						"haarcascade_mcs_lefteye.xml");

				File cascadeDirER = getDir("cascadeER", Context.MODE_PRIVATE);
				mCascadeFileER = new File(cascadeDirER,
						"haarcascade_mcs_righteye.xml");

				int[] raw = new int[] { R.raw.lbpcascade_frontalface,
						R.raw.haarcascade_mcs_lefteye,
						R.raw.haarcascade_mcs_righteye };

				for (int i = 0; i < 3; i++) {
					try {
						InputStream is = getResources().openRawResource(raw[i]);
						FileOutputStream os;

						if (i == 0) {
							os = new FileOutputStream(mCascadeFile);
						} else if (i == 1) {
							os = new FileOutputStream(mCascadeFileEL);
						} else {
							os = new FileOutputStream(mCascadeFileER);
						}

						byte[] buffer = new byte[4096];
						int bytesRead;
						while ((bytesRead = is.read(buffer)) != -1) {
							os.write(buffer, 0, bytesRead);
						}

						is.close();
						os.close();

					} catch (IOException e) {
						e.printStackTrace();
						Log.e(TAG, "Failed to load cascade. Exception thrown: " + e);
					}
				}

				mJavaDetector = new CascadeClassifier(mCascadeFile.getAbsolutePath());
				mJavaDetectorEL = new CascadeClassifier(mCascadeFileEL.getAbsolutePath());
				mJavaDetectorER = new CascadeClassifier(mCascadeFileER.getAbsolutePath());
				
				if (mJavaDetector.empty()) {
					Log.e(TAG, "Failed to load cascade classifier");
					mJavaDetector = null;
				} else {
					Log.i(TAG, "Loaded cascade classifier from " + mCascadeFile.getAbsolutePath());
				}
				if (mJavaDetectorEL.empty()) {
					Log.e(TAG, "Failed to load cascade classifier EL");
					mJavaDetectorEL = null;
				} else {
					Log.i(TAG, "Loaded cascade classifier EL from " + mCascadeFileEL.getAbsolutePath());
				}
				if (mJavaDetectorER.empty()) {
					Log.e(TAG, "Failed to load cascade classifier ER");
					mJavaDetectorER = null;
				} else {
					Log.i(TAG, "Loaded cascade classifier ER from " + mCascadeFileER.getAbsolutePath());
				}

				cascadeDir.delete();
				cascadeDirEL.delete();
				cascadeDirER.delete();

				mOpenCvCameraView.setCameraIndex(1);
				mOpenCvCameraView.setMaxFrameSize(352, 288);// 최대 크기 지정 (1280,720)6fps (768,512)12fps
				// 720*720 미만으로 가니까 자꾸 에러가 뜨더라.... -> 카메라 크기가 352*288로 지정이 되는데,
				// 원래 mat 사이즈로 resize를 안하니까!!!! 이거 정사각으로 뜸
				// mOpenCvCameraView.setMaxFrameSize(440, 330);
				mOpenCvCameraView.enableFpsMeter();
				mOpenCvCameraView.enableView();// 뷰 활성화

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
		if (mOpenCvCameraView != null)
			mOpenCvCameraView.disableView();
	}

	@Override
	public void onResume() {
		super.onResume();
		/*
		 * OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_2_4_9, this,
		 * mLoaderCallback);
		 */
		mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
	}

	public void onDestroy() {
		super.onDestroy();
		finish(); // mOpenCvCameraView.disableView();
	}

	public void onCameraViewStarted(int width, int height) {
		mGray = new Mat();
		mRgba = new Mat();
	}

	public void onCameraViewStopped() {
		mGray.release();
		mRgba.release();
		if (isZoomwindowVisible)
			mZoomWindow.release();
	}

	static boolean isTablet(Context context) {
		int xlargeBit = 4; // Configuration.SCREENLAYOUT_SIZE_XLARGE; 
		Configuration config = context.getResources().getConfiguration();
		return (config.screenLayout & xlargeBit) == xlargeBit;
	}

	public Mat onCameraFrame(CvCameraViewFrame inputFrame) {
		// Winklick의 faceService와 같다. 실험용

		mRgba = inputFrame.rgba();
		mGray = inputFrame.gray();// (352,288)

		Core.flip(mRgba, mRgba, 1);
		Core.flip(mGray, mGray, 1);// (352,288)

		if (!isTablet) {// 스마트폰에 맞게 화면회전
			mRgba = mRgba.t();
			mGray = mGray.t();// (288,352)
			Core.flip(mRgba, mRgba, 1);
			Core.flip(mGray, mGray, 1);// (288,352)
		}

		if (mAbsoluteFaceSize == 0) {
			int height = mGray.rows();
			if (Math.round(height * mRelativeFaceSize) > 0) {
				mAbsoluteFaceSize = Math.round(height * mRelativeFaceSize);
			}
		}

		MatOfRect faces = new MatOfRect();

		if (mJavaDetector != null)
			mJavaDetector.detectMultiScale(mGray, faces, 1.1, 2, 2, new Size(
					mAbsoluteFaceSize, mAbsoluteFaceSize), new Size()); // 얼굴 검색!

		Rect[] facesArray = faces.toArray();

		if (facesArray.length > 0) {
			// Log.d("TAG", "-");
			// Imgproc.equalizeHist(mGray, mGray);
			detectEyeLocation(facesArray); // 얼굴 바로 찾으면 바로 눈 추적!
		} else {

			// Log.d("TAG", "BLC");
			
			Imgproc.equalizeHist(mGray, mGray); // TODO : 역광보정까지 넣으면 진짜 대박
			// 얼굴 바로 못찾으면 히스토그램 균일화 후 눈 추적!
			// 사실 Cascade 사용 전에는 히스토그램 균일화 하는게 맞다.

			if (mJavaDetector != null)
				mJavaDetector.detectMultiScale(mGray, faces, 1.1, 2, 2,
						new Size(mAbsoluteFaceSize, mAbsoluteFaceSize),
						new Size());
			facesArray = faces.toArray();

			if (facesArray.length > 0) {
				detectEyeLocation(facesArray);
			} else {
				toastShowUI("얼굴을 찾을 수 없습니다!"); // 히스토그램 균일화 이후로도 못찾으면 포기하자.
			}
			// global.setFaceExist(false);
		}

		// mGray = null;//onCameraViewStopped의 mGray.release()에서
		// NullPointerException 오류가 생긴다

		if (!isFaceVisible) {
			Imgproc.resize(mRgba, mRgba, new Size(1, 1));// 리턴값을 투명하게 하는 대신 1,1사이즈로 해서 안보이게함
		} else {
			Imgproc.resize(mRgba, mRgba, new Size(352, 288));// TODO : 아직도 isFaceVisible = false하면 Assertion Failed 뜸
		}
		return mRgba;
	}

	private void detectEyeLocation(Rect[] facesArray) {
		if (isLineVisible)
			Core.rectangle(mRgba, facesArray[0].tl(), facesArray[0].br(),
					FACE_RECT_COLOR, 3);// 얼굴 면적을 초록 사각형으로 표시

		xCenter = (facesArray[0].x + facesArray[0].width + facesArray[0].x) / 2;
		yCenter = (facesArray[0].y + facesArray[0].y + facesArray[0].height) / 2;

		if (isLineVisible) {
			Point center = new Point(xCenter, yCenter);// 얼굴 중점 그리고 중점에 미니원(뽀대용)을 그린다

			Core.circle(mRgba, center, 10, new Scalar(255, 0, 0, 255), 3);

			Core.putText(mRgba, "[" + center.x + "," + center.y + "]",
					new Point(center.x + 20, center.y + 20),
					Core.FONT_HERSHEY_SIMPLEX, 0.7, new Scalar(255, 255, 255,
							255));// 중점 옆에 위치좌표를 보여주는 텍스트 표시
		}

		Rect r = facesArray[0];
		Mat smp = mGray.submat(r); // 얼굴 영역만 추림 // mGray.submat(originalEyeArea(r, true));
		Rect eyeML = mcs_eyearea(r, true); Rect eyeMR = mcs_eyearea(r, false); // 눈 검출영역
		
		
		if (mAbsoluteEyeSize == 0) {
			int height = eyeML.height;
			if (Math.round(height * mRelativeFaceSize) > 0) {
				mAbsoluteEyeSize = Math.round(height * mRelativeFaceSize);
			}
		}
		
		MatOfRect eyeLs = new MatOfRect(); Mat eyeMomL = smp.submat(eyeML); //Imgproc.equalizeHist(eyeMomL, eyeMomL);
		if (mJavaDetectorEL != null) {
			mJavaDetectorEL.detectMultiScale(eyeMomL, eyeLs, 1.1, 2, 2, 
					new Size(mAbsoluteEyeSize, mAbsoluteEyeSize), new Size()); // 왼눈 검출
		}
		Rect[] eyeLArray = eyeLs.toArray();
		
		MatOfRect eyeRs = new MatOfRect(); Mat eyeMomR = smp.submat(eyeMR); //Imgproc.equalizeHist(eyeMomR, eyeMomR);
		if (mJavaDetectorER != null) {
			mJavaDetectorER.detectMultiScale(eyeMomR, eyeRs, 1.1, 2, 2, 
					new Size(mAbsoluteEyeSize, mAbsoluteEyeSize), new Size());  // 오른눈 검출
		}
		Rect[] eyeRArray = eyeRs.toArray();
		
		
		if(eyeLArray.length>0){
			Rect eyeL = eyeLArray[eyeLArray.length - 1]; // 눈썹 찾지 않도록 마지막껄로!
			eyeCL = new Point(
					eyeML.x + 0.5*(eyeL.tl().x + eyeL.br().x),
					eyeML.y + 0.5*(eyeL.tl().y + eyeL.br().y));
			/*Core.rectangle(smp, 
					new Point(eyeML.x + eyeL.tl().x, eyeML.y + eyeL.tl().y), 
					new Point(eyeML.x + eyeL.br().x, eyeML.y + eyeL.br().y), FACE_RECT_COLOR, 3);*/ //왼눈 검출 범위
		}else{
			Log.e("TAG", "no left eye detected");
		}
		
		if(eyeRArray.length>0){
			Rect eyeR = eyeRArray[eyeRArray.length - 1]; // 눈썹 찾지 않도록 마지막껄로!
			eyeCR = new Point(
					eyeMR.x + 0.5*(eyeR.tl().x + eyeR.br().x),
					eyeMR.y + 0.5*(eyeR.tl().y + eyeR.br().y));
			/*Core.rectangle(smp, 
					new Point(eyeMR.x + eyeR.tl().x, eyeMR.y + eyeR.tl().y), 
					new Point(eyeMR.x + eyeR.br().x, eyeMR.y + eyeR.br().y), FACE_RECT_COLOR, 3);*/ //오른눈 검출 범위
		}else{
			Log.e("TAG", "no right eye detected");
		}
		// blackLightCompensation(smp, smp);
		// Imgproc.dilate(smp, smp,
		// Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new
		// Size(2,2)));//검은색 줄이기

		// final int threshHold = 70;//클수록 어두워짐
		// Imgproc.threshold(smp, smp, threshHold, 255, Imgproc.THRESH_BINARY);

		// Imgproc.equalizeHist(smp, smp);//이거도 너무 검게나옴
		// blackLightCompensation(smp, smp);//눈이미지에 하는거 안정성 떨어짐
		
		mRgba = adjustedFace(smp);

	}
	
	double prev_angle = 0;
	
	private Mat adjustedFace(Mat faceOnly) { // 얼굴 기울임을 보정함.
		double eyesCenter_x, eyesCenter_y, dx, dy, len, angle;
		eyesCenter_x = 0.5*(eyeCL.x + eyeCR.x); eyesCenter_y = 0.5*(eyeCL.y + eyeCR.y);
		
		dx = eyeCR.x - eyeCL.x; dy = eyeCR.y - eyeCL.y;
		len = Math.sqrt(dx*dx + dy*dy);
		angle = Math.atan2(dy, dx) * 180.0 / Math.PI; // rad -> deg
		if(prev_angle != 0 && angle > 20 ) angle = prev_angle;
		prev_angle = angle;
		
		final double DESIRED_LEFT_EYE_X = 0.16;
		final double DESIRED_RIGHT_EYE_X = 0.84; // 0.84 = 1 - 0.16
		final double DESIRED_LEFT_EYE_Y = 0.14; // right와 일치.
		final int DESIRED_FACE_WIDTH = 150; final int DESIRED_FACE_HEIGHT = 150; // 내가 원하는 표준 얼굴 크기
		
		double desiredLen = DESIRED_RIGHT_EYE_X - DESIRED_LEFT_EYE_X; // 내가 원하는 표준 얼굴 내 눈 비율
		double scale = desiredLen * DESIRED_FACE_WIDTH / len; // 원본 눈 사이 거리 -> 표준 눈 사이 거리
		
		
		double ex = DESIRED_FACE_WIDTH * 0.5 - eyesCenter_x;
		double ey = DESIRED_FACE_HEIGHT * DESIRED_LEFT_EYE_Y - eyesCenter_y;
		
		Mat rot_mat = Imgproc.getRotationMatrix2D(new Point(eyesCenter_x, eyesCenter_y), angle, scale);
		// 원하는 각도, 크기에 대한 변환 행렬을 취득한다.
		
		double colorArr[] = rot_mat.get(0,2); // rot_mat의 (x,y)=(2,0)좌표
		rot_mat.put(0, 2, colorArr[0] + ex);
		colorArr = rot_mat.get(1,2);
		rot_mat.put(1, 2, colorArr[0] + ey); // 원하는 중심으로 눈의 중심을 이동
		
		Mat adjustedFace = new Mat(DESIRED_FACE_WIDTH, DESIRED_FACE_HEIGHT, faceOnly.type(), new Scalar(128));
		// 얼굴 영상을 원하는 각도 & 크기 & 위치로 변환
		Imgproc.warpAffine(faceOnly, adjustedFace, rot_mat, adjustedFace.size()); // 최대 크기로 할려면 Imgproc.INTER_LINEAR
		
		return stabilizated_eye(adjustedFace);
		
		/*앞으로 할거
		 * 1. 얼굴 좌우에 대한 히스토그램 균일화
		 * 2. 안경 착용 보정
		 * 3. 얼굴 중점(eyesCenter)기준으로 안정적인 눈영역 추리기 및 동공만 추리기*/
	}
	
	private Rect mcs_eyearea(Rect face, boolean isEyeLeft) { // face.tl()를 원점으로 한다.
		
		double EYE_SX = 0.10;
		double EYE_SY = 0.19;
		double EYE_SW = 0.40;
		double EYE_SH = 0.39;
		
		return cutted_eye_area(face, isEyeLeft, EYE_SX, EYE_SY, EYE_SW, EYE_SH);
	}
	
	private Mat equalizeHistLR(Mat src){ // 너무 느림(3fps). 좌우의 그림자 정도가 같아지지만, 광원에 따른 변화가 아예 없어지진 않음..
		int w = src.rows(); int h  = src.cols();
		Mat ori = new Mat(); Imgproc.equalizeHist(src, ori);
		
		int midX = w/2;
		
		Mat leftSide = src.submat(new Rect(0,0,midX,h));
		Mat rightSide = src.submat(new Rect(midX,0,w-midX,h));
		
		Imgproc.equalizeHist(leftSide, leftSide);
		Imgproc.equalizeHist(rightSide, rightSide);
		
		
		Mat dst = new Mat(src.rows(), src.cols(), src.type());
		
		for(int y=0; y<h; y++){
			for(int x=0; x<w; x++){
				int v;
				
				if(x < w/4){ // 0~0.25 : 왼쪽 그대로
					v = (int) Math.round(leftSide.get(y,x)[0]);// int로의 강제 형변환은 내림이다.
				}
				else if(x < w*2/4){ // 0.25~0.5 : 왼쪽 + 가운데 그라데이션
					double lv = leftSide.get(y,x)[0];
					double ov = ori.get(y,x)[0];
					double d = ((double)x - w*1/4) / (w/4); //w랑 x가 정수여서 d도 아예 정수만 나왔다 ㅆㅃ
					
					v = (int) Math.round((1 - d) * lv + d * ov); // 색 그라데이션
				}
				else if(x < w*3/4){ // 0.5~0.75 : 오른쪽 + 가운데 그라데이션
					double rv = rightSide.get(y,x-midX)[0];
					double ov = ori.get(y,x)[0];
					double d = ((double)x - w*2/4) / (w/4);
					
					v = (int) Math.round((1 - d) * ov + d * rv); // 색 그라데이션
				}
				else { // 0.75~1 : 오른쪽 그대로
					v = (int) Math.round(rightSide.get(y,x-midX)[0]);
				}
				
				dst.put(y, x, v);
			}
		}
		return dst;
	}
	
	////====여기 아래서부터는 내가 알고리즘 구조구성을 위해 만든 코드들이다===////
	
	private Mat stabilizated_eye(Mat final_face){
		//Imgproc.cvtColor(final_face, final_face, Imgproc.COLOR_GRAY2RGB);
		//Imgproc.bilateralFilter(final_face, final_face, 15, 80, 80);
		// TODO : 가우시안에 비해 특징모서리는 살려둠. 이거 꼭 되게 하자.
		
		Rect eyeL = final_eye_area(new Rect(0,0, final_face.width(), final_face.height()),true);
		Rect eyeR = final_eye_area(new Rect(0,0, final_face.width(), final_face.height()),false);
		//Mat나 Rect에서 tl()은 (1,1)가 아닌 (0,0)이다!!
		
		Mat mat_eyeL = final_face.submat(eyeL); Mat mat_eyeR = final_face.submat(eyeR);
		
		Imgproc.equalizeHist(mat_eyeL, mat_eyeL);
		// submat은 원본 Mat에서 주소만 갖고오는것! submat을 변형하면 원본 출력시 영향을 준다! 이를 막을려면 mat.clone() 쓰자.
		
		Imgproc.threshold(final_face.submat(eyeL), final_face.submat(eyeL), 
				15, 255, Imgproc.THRESH_BINARY_INV ); // 반전시켜서 용량 절약 하자.
		
		/* TODO : 대조군이 binary mask 말고 그냥 erode쓰고 흰 덩어리 중점 찾은거 같음 ㅆㅃ
		 * 공개한 소스코드랑 논문이 제시한 방법이랑 달라. 
		 * getStructuringElement -> equalizeHist -> bilateralFilter -> threshold -> erode -> canny
		 * 근데 Kalman filter를 통해 얼굴 영역 변화를 줄일 수 있다든데.. -> 회전 안정화가 시급. 이걸 써볼까?
		 */
		

		//Core.rectangle(final_face, eyeL.tl(), eyeL.br(), FACE_RECT_COLOR, 1);
		//Core.rectangle(final_face, eyeR.tl(), eyeR.br(), FACE_RECT_COLOR, 1);
		// 이미지 처리에 영향 주므로 그림은 맨 나중에 하자.
		return final_face;
	}
	
	private Rect final_eye_area(Rect final_face, boolean isEyeLeft) { // final_face.tl()를 원점으로 한다.
		
		double EYE_SX = 0;
		double EYE_SY = 0;
		double EYE_SW = 0.40;
		double EYE_SH = 0.39;
		
		return cutted_eye_area(final_face, isEyeLeft, EYE_SX, EYE_SY, EYE_SW, EYE_SH);
	}
	
	private Rect cutted_eye_area(Rect face, boolean isEyeLeft, double EYE_SX, double EYE_SY, double EYE_SW, double EYE_SH) {
		
		int x = isEyeLeft ? (int) (face.width * EYE_SX) : (int) (face.width * (1.0 - EYE_SW - EYE_SX));

		face = new Rect(
				x, // face.x + x
				(int) (face.height * EYE_SY), // face의 부모 Mat를 원점으로 잡는 법 : (int) (face.y + (face.height * EYE_SY))
				(int) (face.width * EYE_SW),
				(int) (face.height * EYE_SH) );

		return face;
	}
	
	
 	private void blackLightCompensation(Mat src, Mat dst) {
		// rate : 0~1의 실수로서 이 rate보다 어두운 빛은 전부다 위로 올려버린다.
		// if(rate >= 1 || rate < 0) Log.e("TAG",
		// "Rate isn't on 0~1 at blackLightCompensation");
		if (src.empty() || src == null)
			Log.e("TAG", "Empty Mat at blackLightCompensation");

		Mat C = src.clone();

		src.convertTo(src, CvType.CV_64FC1); // New line added.
		int size = (int) (src.total() * src.channels());
		double[] temp = new double[size]; // use double[] instead of byte[]
		src.get(0, 0, temp);

		double m = 0;
		for (int i = 0; i < size; i++)
			m += temp[i];
		m /= size;// part A

		double ratio = 127.5 / m;
		for (int i = 0; i < size; i++) {
			if (temp[i] * ratio >= 255) {
				temp[i] = 255;
			} else {
				temp[i] *= ratio;
			}
		}// part B

		/*
		 * int divby = 5;//divby당 1번의 픽셀만 더한다 for (int i = 0; i < size; i++)
		 * if(i % divby == 0) m += temp[i]; m /= Math.floor(size/divby);
		 */

		// part A, B 둘 다 8.9fps를 만드는데 영향을 줌. 둘다 속도 부하점
		C.put(0, 0, temp);
		C.convertTo(src, CvType.CV_8U);
		dst = C;
	}

	private int mode(ArrayList<Integer> arr) {
		// arr에서 최빈값을 반환
		ArrayList<Integer> freqArr = new ArrayList<Integer>();
		for (int k = 0; k < arr.size(); k++) {
			freqArr.add(Collections.frequency(arr, arr.get(k)));
		}

		// arr.indexOf(object)
		// Collections.frequency(arr, 2);
		return arr.get(freqArr.indexOf(Collections.max(freqArr)));
	}

	

	// --안내 메세지 관련--//
	private Toast toast = null;

	private void toastShowUI(final String message) {
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
		} else {
			toast.setText(message);
		}
		toast.setGravity(Gravity.CENTER, 0, 0);
		toast.show();
	}

	// --안내 메세지 관련--//

	private void upBriCon(Mat src, Mat dst, double bri, double con) {
		// TODO : fps가 9로 array로 접근하는거 보다 빠르다
		// mGray.submat(area);
		if (src.empty() || src == null)
			Log.e("TAG", "Empty Mat on upBriCon");

		Mat C = src.clone();

		src.convertTo(src, CvType.CV_64FC1); // New line added.
		int size = (int) (src.total() * src.channels());
		double[] temp = new double[size]; // use double[] instead of byte[]
		src.get(0, 0, temp);

		for (int i = 0; i < size; i++)
			temp[i] = (con * temp[i] + bri); // no more casting required.
		C.put(0, 0, temp);
		C.convertTo(src, CvType.CV_8U);
		// Imgproc.cvtColor(C, C, Imgproc.COLOR_BGR2GRAY);//넣으면 안됨
		// Imgproc.GaussianBlur(C, C, new Size(3, 3), 0);
		// Imgproc.threshold(C, C, 80, 255, Imgproc.THRESH_BINARY);
		dst = C;
	}

}