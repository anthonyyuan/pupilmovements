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
import org.opencv.core.MatOfKeyPoint;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.core.Core.MinMaxLocResult;
import org.opencv.features2d.FeatureDetector;
import org.opencv.features2d.KeyPoint;
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
		} else { // 얼굴찾기 실험1 : 히스토그램 균일화 후 눈 추적!

			Mat test = new Mat();
			Imgproc.equalizeHist(mGray, test); // 사실 Cascade 사용 전에는 히스토그램 균일화 하는게 맞다.
			

			if (mJavaDetector != null)
				mJavaDetector.detectMultiScale(test, faces, 1.1, 2, 2,
						new Size(mAbsoluteFaceSize, mAbsoluteFaceSize),
						new Size());
			facesArray = faces.toArray();

			if (facesArray.length > 0) {
				detectEyeLocation(facesArray);
			} else { // 얼굴찾기 실험2 : BLC
				
				// 안경때문에 나중에 눈영역 histEq할때 동공이 안보임. -> 눈 영역따고 바로 안경 지우고 histEq해야...
				
				test = new Mat();
				blackLightCompensation(mGray,test);
				Imgproc.equalizeHist(test, test);
				

				if (mJavaDetector != null)
					mJavaDetector.detectMultiScale(test, faces, 1.1, 2, 2,
							new Size(mAbsoluteFaceSize, mAbsoluteFaceSize),
							new Size());
				facesArray = faces.toArray();

				if (facesArray.length > 0) {
					detectEyeLocation(facesArray);
				}else{
					//blackLightCompensation(mGray,mGray);
					//mRgba = mGray;
					toastShowUI("얼굴을 찾을 수 없습니다!"); // 히스토그램 균일화 이후로도 못찾으면 포기하자.
				}
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
	
	double prev_angle, prev_scale = 0;
	
	
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
		
		if(prev_scale != 0 && scale - prev_scale > 0.3 ) scale = 0.5 * (scale + prev_scale);
		prev_scale = scale;
		
		double ex = DESIRED_FACE_WIDTH * 0.5 - eyesCenter_x;
		double ey = DESIRED_FACE_HEIGHT * DESIRED_LEFT_EYE_Y - eyesCenter_y;
		
		Mat rot_mat = Imgproc.getRotationMatrix2D(new Point(eyesCenter_x, eyesCenter_y), angle, scale);
		// 원하는 각도, 크기에 대한 변환 행렬을 취득한다.
		
		//Log.d("TAG", "scale = " + scale);
		
		double colorArr[] = rot_mat.get(0,2); // rot_mat의 (x,y)=(2,0)좌표
		rot_mat.put(0, 2, colorArr[0] + ex);
		colorArr = rot_mat.get(1,2);
		rot_mat.put(1, 2, colorArr[0] + ey); // 원하는 중심으로 눈의 중심을 이동
		
		Mat adjustedFace = new Mat(DESIRED_FACE_WIDTH, DESIRED_FACE_HEIGHT, faceOnly.type(), new Scalar(128));
		// 얼굴 영상을 원하는 각도 & 크기 & 위치로 변환
		Imgproc.warpAffine(faceOnly, adjustedFace, rot_mat, adjustedFace.size()); // 최대 크기로 할려면 Imgproc.INTER_LINEAR
		
		return detect_eye_move(adjustedFace);
		
		/*앞으로 할거
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
	
	Mat prev = new Mat();
	
	private Mat detect_eye_move(Mat final_face){
		
		Rect eyeL = final_eye_area(new Rect(0,0, final_face.width(), final_face.height()),true);
		Rect eyeR = final_eye_area(new Rect(0,0, final_face.width(), final_face.height()),false);
		//Mat나 Rect에서 tl()은 (1,1)가 아닌 (0,0)이다!!
		
		
		Rect eyeLR = new Rect(eyeL.x,eyeL.y,(int)(eyeR.br().x),eyeL.height);
		Imgproc.adaptiveBilateralFilter(final_face.submat(eyeLR).clone(), final_face.submat(eyeLR), new Size(3,3), 3); // 가우시안에 비해 모서리를 잘 잡음.
		
		Mat eyeLR_mat = final_face.submat(eyeLR);
		Mat fat_eyeLR_mat = eyeLR_mat.clone();
		Mat rot_mat = get_rotation_mat_of_eye(fat_eyeLR_mat);//get_rotation_mat_of_eye(eyeLR_mat.clone());//TODO : col,row알고 지우자.
		if(rot_mat.cols() < 1) return final_face;
		Imgproc.warpAffine(eyeLR_mat.clone(), eyeLR_mat, rot_mat, eyeLR_mat.size(),
				Imgproc.INTER_LINEAR, Imgproc.BORDER_CONSTANT, new Scalar(128)); // 회전 : 배경색 = 128
		
		//현재는 eyeL에 대해서만 눈 이동 찾기를 진행한다.
		
		get_both_eye_size(fat_eyeLR_mat, LEFT_EYE_AREA_COLOR, RIGHT_EYE_AREA_COLOR); fat_eyeLR_mat.release();
		//회전이 된 이후에 작동이 되어야 하므로 여기에 있어야 한다.
		//여기서는 rot_mat가 아니고 eyeLR_mat임에 유의!
		
		
		
		
		Mat mat_eyeL = final_face.submat(eyeL); Mat mat_eyeR = final_face.submat(eyeR);
		
			
		/*Imgproc.equalizeHist(mat_eyeL, mat_eyeL);
		final int thresh = 10;
		Imgproc.threshold(final_face.submat(eyeL), final_face.submat(eyeL), 
				thresh, 255, Imgproc.THRESH_BINARY_INV );*/ // 반전시켜서 용량 절약 하자.
		/*TODO : col,row알고 주석 지우자. */
		
		/*테스트
		
		Mat test = Mat.ones(new Size( mat_eyeL.cols(), mat_eyeL.rows() ), mat_eyeL.type());
		test.convertTo(test, -1, 100);//회색 Mat
		
		Core.subtract(test, mat_eyeL.clone(), mat_eyeL); // 회색 - 검은 배경 흰 눈 = 회색 배경 검은 눈
		
		Core.subtract(
				Mat.zeros(new Size( mat_eyeL.cols(), mat_eyeL.rows() ), mat_eyeL.type()),
						mat_eyeL.clone(), mat_eyeL); // 검게 나옴 =  if a>b ; a-b ; 0
		
		//Core.absdiff(prev.clone(), mat_eyeL.clone(), motion); // dst=saturate(|a - b|)
		 */
		
		//손상된 동공을 원으로 재건.
		//reconstruct_pupil(mat_eyeL); //TODO : col,row알고 주석 지우자.
		
		//이전 Mat과의 차이를 출력.
		// -> 눈 움직임 완전고정을 해도 잡음이 너무 크다. -> 원형으로 동공을 재건해야 가능! 
		/*Mat motion = new Mat();
		
		if(prev.cols() > 1) {
			
			if(prev.cols() != mat_eyeL.cols() && prev.rows() != mat_eyeL.rows())
				Imgproc.resize(prev, prev, new Size(mat_eyeL.cols(), mat_eyeL.rows()));
			
			Core.subtract(prev.clone(), mat_eyeL.clone(), motion); // dst=saturate(a - b)
			//차집합 구하는 코드. 작동함.
			
			Log.d("TAG", "prev.size = "+prev.size());
			
			mat_eyeL.copyTo(prev);
			motion.copyTo(mat_eyeL);
			
		}else{
			mat_eyeL.copyTo(prev);
			return final_face;
		}*/
		
		
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
	
	private void reconstruct_pupil(Mat eyeMat){
		
		Point avg;
		int addX = 0, addY = 0, pixelNum = 0;
		
		for(int x=0; x<eyeMat.cols(); x++){
			for(int y=0; y<eyeMat.rows(); y++){
	    		if (eyeMat.get(y,x)[0] > 10){
	    			addX += x;
	    			addY += y;
	    			pixelNum ++;
	    		}
	        }
		}
		
		avg = new Point(addX/pixelNum, addY/pixelNum);
		
		Core.circle(eyeMat, avg, 3, new Scalar(100), 1);
		
		//inv 됬으므로 dilate가 살찌기, erode가 살빼기가 된다.
		/*Mat kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(2,2));
		Imgproc.dilate(eyeMat, eyeMat, kernel);//찌기
		Imgproc.erode(eyeMat, eyeMat, kernel);//빼기
		
		Imgproc.Canny(eyeMat, eyeMat, 5, 70);
		Imgproc.GaussianBlur(eyeMat, eyeMat, new Size(3, 3), 0);
		
		Mat circles = new Mat();
		Imgproc.HoughCircles(eyeMat, circles, Imgproc.CV_HOUGH_GRADIENT, 1, eyeMat.cols()/8);
		
		if (circles.cols() > 0) {
		    for (int x = 0; x < circles.cols(); x++) {
		        double vCircle[] = circles.get(0,x);

		        if (vCircle == null)
		            break;

		        Point pt = new Point(Math.round(vCircle[0]), Math.round(vCircle[1]));
		        int radius = (int)Math.round(vCircle[2]);

		        // draw the found circle
		        Core.circle(eyeMat, pt, radius, new Scalar(150), 1);
		        Core.circle(eyeMat, pt, 3, new Scalar(150), 1);
		    }
		}
		Log.d("TAG", "검출된 원 갯수 = "+circles.cols()); //원도 작동 안한다.
*/		
		/*MatOfKeyPoint matOfKeyPoints = new MatOfKeyPoint();
        FeatureDetector blobDetector = FeatureDetector.create(FeatureDetector.SIMPLEBLOB);
        blobDetector.detect(eyeMat, matOfKeyPoints);
        KeyPoint[] blopList = matOfKeyPoints.toArray();
        
        Log.d("TAG", "검출된 Blop 갯수 = "+blopList.length); threshold 늘리고 줄여봐도 작동 안됨*/
	}
		
	private Mat get_rotation_mat_of_eye(Mat src){ //눈 구석에 맞춰서 회전 및 정렬한다.
		
		Imgproc.equalizeHist(src, src);
		//Imgproc.adaptiveBilateralFilter(src.clone(), src, new Size(3,3), 3);
		// 가우시안에 비해 특징모서리는 살려둠.
		
		
		Imgproc.threshold(src, src, 30, 255, Imgproc.THRESH_BINARY_INV ); //둘다 src일때만 실행됨. // 눈과 안경이 분리되어야..
		
		
		boolean isEyeglass = false;
		EyeglassesFloodfill : for(int y = 0; y < src.rows(); y++){
    		if (src.get(y,src.cols()/2)[0] > 10){
    			toastShowUI("안경을 벗어주세요!");
    			isEyeglass = true;
    			break EyeglassesFloodfill;
    			
    		}
        }
		if(isEyeglass) return new Mat();
		
		
		// 1. 가운데서부터 양 옆으로, y는 밑에서부터 2/3 지점까지 검사하여 코에 가까운 눈의 끝점 2개를 찾고 점 p,q라고 놓는다.
		//   (각 눈별로 안하는 이유는, 고개를 옆으로 살짝 돌리는 경우 가운데 쳐다봐도 오른쪽으로 인식될 수 있기 때문임, 또한 먼 쪽은 인풋에서부터 잘렸을 수 있음.)
				
		Point leftCorner = new Point(0,0), rightCorner = new Point(0,0);
		
		
		LeftCorner : for(int x = src.cols()/2; x > src.cols()/4; x--){
			for(int y = src.rows()-1; y > 0.4*src.rows(); y--){
	    		if (src.get(y,x)[0] > 10){
	    			leftCorner = new Point(x,y);
	    			Imgproc.floodFill(src, Mat.zeros(src.rows() + 2, src.cols() + 2, src.type()), 
	    					leftCorner, new Scalar(LEFT_EYE_AREA_COLOR));
	    			break LeftCorner;
	    		}
	        }
		}
		
		RightCorner : for(int x = src.cols()/2 + 1; x < src.cols()*0.75; x++){
			for(int y = src.rows()-1; y > 0.4*src.rows(); y--){
	    		if (src.get(y,x)[0] > 10){
	    			rightCorner = new Point(x,y);
	    			Imgproc.floodFill(src, Mat.zeros(src.rows() + 2, src.cols() + 2, src.type()), 
	    					rightCorner, new Scalar(RIGHT_EYE_AREA_COLOR));
	    			break RightCorner;
	    		}
	        }
		}
		
		if(leftCorner.x == 0 && rightCorner.x == 0) return new Mat();
		
		//Core.circle(src, leftCorner, 2, new Scalar(100), 2);
		//Core.circle(src, rightCorner, 2, new Scalar(100), 2);//잘 됨!
		
		
		// 2. 이전 선분 p'q'와 현재 pq와의 거리비, pq와 p'q' 사이의 각을 구한다.
		//    (어쩌피 3.서 맞출꺼면 그냥 표준 비율을 구하는게 낫지 않을까? 맨 처음꺼에 대한 영향이 너무 큰데..)
		
		double cornerCenter_x, cornerCenter_y, dx, dy, len, angle;
		cornerCenter_x = 0.5*(leftCorner.x + rightCorner.x); cornerCenter_y = 0.5*(leftCorner.y + rightCorner.y);
		
		dx = rightCorner.x - leftCorner.x; dy = rightCorner.y - leftCorner.y;
		len = Math.sqrt(dx*dx + dy*dy);
		angle = Math.atan2(dy, dx) * 180.0 / Math.PI; // rad -> deg
		
		// if(prev_angle != 0 && angle > 20 ) angle = prev_angle;
		// prev_angle = angle;
		
		
		
		final double DESIRED_LEFT_CORNER_X = 0.30;
		final double DESIRED_RIGHT_CORNER_X = 1 - DESIRED_LEFT_CORNER_X; //1-0.33
		final double DESIRED_LEFT_CORNER_Y = 0.44;
		
		final int DESIRED_WIDTH = src.width();
		final int DESIRED_HEIGHT = src.height();//DESIRED가 전체 Mat에서의 비율이므로 src의 col,row가 맞다.
		
		double desiredLen = DESIRED_RIGHT_CORNER_X - DESIRED_LEFT_CORNER_X; // 내가 원하는 표준 얼굴 내 눈 비율
		double scale = desiredLen * DESIRED_WIDTH / len; // 원본 눈 사이 거리 -> 표준 눈 사이 거리
		
		// if(prev_scale != 0 && scale - prev_scale > 0.3 ) scale = 0.5 * (scale + prev_scale);
		// prev_scale = scale;
		
		double ex = DESIRED_WIDTH * 0.5 - cornerCenter_x;
		double ey = DESIRED_HEIGHT * DESIRED_LEFT_CORNER_Y - cornerCenter_y;
		
		Mat rot_mat = Imgproc.getRotationMatrix2D(new Point(cornerCenter_x, cornerCenter_y), angle, scale);
		// 원하는 각도, 크기에 대한 변환 행렬을 취득한다.
		
		double colorArr[] = rot_mat.get(0,2); // rot_mat의 (x,y)=(2,0)좌표
		rot_mat.put(0, 2, colorArr[0] + ex);
		colorArr = rot_mat.get(1,2);
		rot_mat.put(1, 2, colorArr[0] + ey); // 원하는 중심으로 눈의 중심을 이동
		
		// 3. 현재 거리를 이전꺼로 맞춰 회전한다.
		
		
		return rot_mat;
		//Imgproc.warpAffine(src.clone(), src, rot_mat, src.size()); 
		
		// 얼굴 영상을 원하는 각도 & 크기 & 위치로 변환
		
	}

	final int LEFT_EYE_AREA_COLOR = 140;
	final int RIGHT_EYE_AREA_COLOR = 160;
	int eye_width = 0, eye_height = 0;
	
	private void get_both_eye_size(Mat eyeMat, int LEFT_EYE_AREA_COLOR, int RIGHT_EYE_AREA_COLOR){

		int aX, bX, aY, bY; aX = bX = aY = bY = 0;
		int dxL, dyL, dxR, dyR;
		
		for(int x =0; x<eyeMat.cols()/2; x++){
			for(int y=0; y<eyeMat.rows(); y++){
	    		if (eyeMat.get(y,x)[0] == LEFT_EYE_AREA_COLOR){
	    			if(aX == 0){
	    				aX = x;
	    			}
	    			if(bX < x){
	    				bX = x;
	    			}
	    			if(aY == 0 || aY > y){
	    				aY = y;
	    			}
	    			if(bY < y){
	    				bY = y;
	    			}
	    		}
	        }
		}
		
		dxL = bX - aX;
		dyL = bY - aY;
		aX = bX = aY = bY = 0;
		
		RightEye : for(int x = eyeMat.cols()/2 + 1; x < eyeMat.cols(); x++){
			for(int y=0; y<eyeMat.rows(); y++){
	    		if (eyeMat.get(y,x)[0] == RIGHT_EYE_AREA_COLOR){
	    			if(aX == 0){
	    				aX = x;
	    			}
	    			if(bX < x){
	    				bX = x;
	    			}
	    			if(aY == 0 || aY > y){
	    				aY = y;
	    			}
	    			if(bY < y){
	    				bY = y;
	    			}
	    		}
	        }
		}
		
		dxR = bX - aX;
		dyR = bY - aY;
		
		if(dxL * dyL > dxR * dyR){
			eye_width = dxL; eye_height = dyL;
		}else{
			eye_width = dxR; eye_height = dyR;
		}
		
		Log.d("TAG", "eye_width, eye_height = "+eye_width+", "+eye_height);
		
		/*Imgproc.floodFill(eyeMat, Mat.zeros(eyeMat.rows() + 2, eyeMat.cols() + 2, eyeMat.type()), 
				new Point(leftX, bothY), new Scalar(100));*/
	}
	
	/*if (learn_frames < 5) {
	    teplateR = get_template(mJavaDetectorEye, eyearea_right, 24);
	    teplateL = get_template(mJavaDetectorEye, eyearea_left, 24);
	    learn_frames++;
	} else {
	    // Learning finished, use the new templates for template
	    // matching
	    match_eye(eyearea_right, teplateR, method);
	    match_eye(eyearea_left, teplateL, method);

	}*/
	private Mat get_template(CascadeClassifier clasificator, Rect area, int size) {
		/* 이 방법의 문제점 : 가장 어두운 영역을 찾으므로 거의 눈꺼풀과 동공의 교점을 찾는다.
		 * 또한, 원의 기하학적 특성을 고려하지 않으므로 정확한 동공중점을 찾을 수 없다. */
        Mat template = new Mat();
        Mat mROI = mGray.submat(area);
        MatOfRect eyes = new MatOfRect();
        Point iris = new Point();
        Rect eye_template = new Rect();
        clasificator.detectMultiScale(mROI, eyes, 1.15, 2,
            Objdetect.CASCADE_FIND_BIGGEST_OBJECT | Objdetect.CASCADE_SCALE_IMAGE, new Size(30, 30),
            new Size());//Cascade를 사용하여 가장 큰 물체를 찾는다.

        Rect[] eyesArray = eyes.toArray();
        for (int i = 0; i < eyesArray.length;) {
            Rect e = eyesArray[i];
            e.x = area.x + e.x;
            e.y = area.y + e.y;
            Rect eye_only_rectangle = new Rect((int) e.tl().x, (int)(e.tl().y + e.height * 0.4), (int) e.width, (int)(e.height * 0.6));
            mROI = mGray.submat(eye_only_rectangle);
            Mat vyrez = mRgba.submat(eye_only_rectangle);//찾아낸 eye 영역 중 일부를 잘라내어 저장한다.


            Core.MinMaxLocResult mmG = Core.minMaxLoc(mROI);//가장 밝고 어두운 영역을 찾음.

            Core.circle(vyrez, mmG.minLoc, 2, new Scalar(255, 255, 255, 255), 2);
            iris.x = mmG.minLoc.x + eye_only_rectangle.x;//가장 어두운 영역을 눈의 동공중점으로 놓는다.
            iris.y = mmG.minLoc.y + eye_only_rectangle.y;//눈 영역 내 동공중점을 중심으로~~
            eye_template = new Rect((int) iris.x - size / 2, (int) iris.y - size / 2, size, size);
            //~~하는 변 길이가 size인 정사각형을 eye_template로 하고 이를 출력한다.
            
            Core.rectangle(mRgba, eye_template.tl(), eye_template.br(),
                new Scalar(255, 0, 0, 255), 2);//이건 그냥 그림그리기.
            
            template = (mGray.submat(eye_template)).clone();//클론을 통해 원본영상을 손대지 않는다.
            return template;
        }
        return template;
    }
    private void match_eye(Rect area, Mat mTemplate) { //template를 보고 일치하는 것이 있는지 검사.
    	/* 이건 이전에 찾았던 Mat template보고 이와 가장 닮은거 위치 찾는거 같음. 
    	 * 매력적이긴 하지만 이도 역시 기하학적인 특성을 고려 안함. -> Blob 중점을 원 중점이라 놓는것과 같음*/
        Point matchLoc;
        Mat mROI = mGray.submat(area);
        int result_cols = mROI.cols() - mTemplate.cols() + 1;
        int result_rows = mROI.rows() - mTemplate.rows() + 1;
        // Check for bad template size
        if (mTemplate.cols() == 0 || mTemplate.rows() == 0) {
            return;
        }
        Mat mResult = new Mat(result_cols, result_rows, CvType.CV_8U);

        Imgproc.matchTemplate(mROI, mTemplate, mResult, Imgproc.TM_SQDIFF_NORMED);
        // Method가 TM_SQDIFF계열이므로 mTemplate는 일치하는 대상일 %가 높을수록 밝기가 낮다.

        Core.MinMaxLocResult mmres = Core.minMaxLoc(mResult); // 
        matchLoc = mmres.minLoc; // 계열은 일치시 0이며, 다를수록 커진다.

        Point matchLoc_tx = new Point(matchLoc.x + area.x, matchLoc.y + area.y);
        Point matchLoc_ty = new Point(matchLoc.x + mTemplate.cols() + area.x,
            matchLoc.y + mTemplate.rows() + area.y);

        Core.rectangle(mRgba, matchLoc_tx, matchLoc_ty, new Scalar(255, 255, 0,
            255));
        Rect rec = new Rect(matchLoc_tx, matchLoc_ty);

    }
    
    
	
	
	
	private void remove_eyeglasses(Mat src){
		//안경 인식을 해도 너무 불안정해서 막 기울어짐. 안경얼굴인식 따로 만들고 이거 쓰자.
		//clone이랑 copyTo 둘다 hard copy 라는데 왜 그러지... http://answers.opencv.org/question/7682
		
		Mat skin = src.clone(); skin.copyTo(src);
		Mat eye = src.clone(), skinArea = src.clone(); //src.copyTo(eye); src.copyTo(skinArea);
		
		Imgproc.equalizeHist(src, src);
		Imgproc.adaptiveBilateralFilter(src.clone(), src, new Size(3,3), 3);
		// 가우시안에 비해 특징모서리는 살려둠. // 얘는 src랑 dst 달라야 함. http://stackoverflow.com/questions/38460950
		Imgproc.threshold(src, src, 30, 255, Imgproc.THRESH_BINARY_INV ); //둘다 src일때만 실행됨. // 눈과 안경이 분리되어야..
		
		src.copyTo(eye); //작동 됨. thr(src,src)에서 src = skin;로 하면 그냥 threshold된 src 나온다. (영향 x)
		
		//TODO : 어쩌피 블러해야지 눈과 안경영역이 깔끔하게 분리가 됨
		// -> 블러를 영역 나눠서 하면 맨 끝에 테두리 영역은 값이 0처리되므로 끝에 안경코가 안닿는 수가 있음
		// -> 블러 할려면 아예 그냥 하나로 합쳐서 지우는게 낫다. isEyeLeft 제거하자.
		
		
		
		final int EYE_GLASSES_COLOR = 100;
		EyeglassesFloodfill : for(int y = 0; y < eye.rows(); y++){
    		if (eye.get(y,eye.cols()/2)[0] > 10){
    			Imgproc.floodFill(eye, Mat.zeros(eye.rows() + 2, eye.cols() + 2, CvType.CV_8U), 
    					new Point(eye.cols()/2,y), new Scalar(EYE_GLASSES_COLOR));
    			break EyeglassesFloodfill;
    		}
        }
		
		for(int x = 0; x < eye.cols(); x++){
        	for(int y = 0; y < eye.rows(); y++){
        		if (eye.get(y,x)[0] != EYE_GLASSES_COLOR){
        			eye.put(y, x, 0);
        		}
            }
        } // eye : 검은 배경에 EYE_GLASSES_COLOR로 안경만 칠함. (여기까지는 됨.)
		
		int skin_color = 0, skin_size = 0;
		
		for(int x = 0; x < skinArea.cols(); x++){
        	for(int y = 0; y < skinArea.rows(); y++){
        		if (skinArea.get(y,x)[0] < 10) skin_color += skin.get(y,x)[0]; skin_size ++;
            }
        }
		
		skin_color /= skin_size; // 피부색 평균 구함.
		
		for(int x = 0; x < eye.cols(); x++){
        	for(int y = 0; y < eye.rows(); y++){
        		if (eye.get(y,x)[0] == EYE_GLASSES_COLOR){
        			skin.put(y, x, skin_color);
        		}
            }
        }//안경 영역 제거 (아직까지는 안됨.)
		
		skin.copyTo(src);
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