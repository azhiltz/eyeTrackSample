# eyeTrackSample

## 步骤

1. 在phone上安装 opencvManger.apk，下载地址见 [连接](https://jaist.dl.sourceforge.net/project/opencvlibrary/opencv-android/3.0.0/OpenCV-3.0.0-android-sdk-1.zip) ，安装与手机abi对应的库。
2. 编译安装运行。
3. 现在能够读取摄像头数据，调用Dlib检测人脸，但是多线程和识别还未集成进去。

## TODO

1. push 人脸到队列里面；
2. 启动人脸识别线程，进行识别处理；
3. 非检测帧人脸框的显示。