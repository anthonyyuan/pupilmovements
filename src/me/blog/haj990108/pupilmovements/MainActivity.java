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
	private boolean isZoomwindowVisible = true;

	private double rMean = 0;

	private int mDetectorType = JAVA_DETECTOR;
	private String[] mDetectorName;

	private float mRelativeFaceSize = 0.5f;
	private int mAbsoluteFaceSize = 0;

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
		mOpenCvCameraView.disableView();
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
			DETECT_EYE_MOTION(facesArray); // 얼굴 바로 찾으면 바로 눈 추적!
		} else {

			// Log.d("TAG", "BLC");

			Imgproc.equalizeHist(mGray, mGray);// 얼굴 바로 못찾으면 히스토그램 균일화 후 눈 추적!

			if (mJavaDetector != null)
				mJavaDetector.detectMultiScale(mGray, faces, 1.1, 2, 2,
						new Size(mAbsoluteFaceSize, mAbsoluteFaceSize),
						new Size());
			facesArray = faces.toArray();

			if (facesArray.length > 0) {
				DETECT_EYE_MOTION(facesArray);
			} else {
				toastShowUI("얼굴을 찾을 수 없습니다!"); // blc이후로도 못찾으면 포기하자.
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

	private void DETECT_EYE_MOTION(Rect[] facesArray) {
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
		Mat smp = mGray.submat(originalEyeArea(r, true));
		blackLightCompensation(smp, smp);
		// Imgproc.dilate(smp, smp,
		// Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new
		// Size(2,2)));//검은색 줄이기

		// final int threshHold = 70;//클수록 어두워짐
		// Imgproc.threshold(smp, smp, threshHold, 255, Imgproc.THRESH_BINARY);

		// Imgproc.equalizeHist(smp, smp);//이거도 너무 검게나옴
		// blackLightCompensation(smp, smp);//눈이미지에 하는거 안정성 떨어짐
		mRgba = smp;

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

	private Rect originalEyeArea(Rect area, boolean isEyeLeft) {

		if (isEyeLeft) {
			area = new Rect(area.x + area.width / 16,
					(int) (area.y + (area.height / 3.8)),
					(area.width - 2 * area.width / 16) / 2,
					(int) (area.height / 3.0));// eyearea_left
		} else {
			area = new Rect(area.x + area.width / 16
					+ (area.width - 2 * area.width / 16) / 2,
					(int) (area.y + (area.height / 3.8)),
					(area.width - 2 * area.width / 16) / 2,
					(int) (area.height / 3.0));// eyearea_right
		}

		return area;
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