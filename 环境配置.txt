软件开发环境:python3.8 + pycharm
环境配置步骤：
【第一步：安装python3.8】
方法一：直接在python官网下载pyhon3.8的exe文件，安装即可
方法二：
如果有ananconda，可以使用命令"conda create -n env_py python=3.8"创建3.8的虚拟环境
然后激活虚拟环境“conda activate env_py”,然后再进行依赖库的安装

【第二步：安装软件所需的依赖库】
先安装requirements.txt中的第三方库，然后安装libs目录下的两个库
按照顺序运行下面3条命令：（命令行需先进入FaceRecongnition）
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install libs/dlib-19.19.0-cp38-cp38-win_amd64.whl
pip install libs/face_recognition-1.3.0-py2.py3-none-any.whl

Software development environment: Python 3.8+PyCharm
Environment configuration steps:
[Step 1: Install Python 3.8]
Method 1: Download the exe file for Python 3.8 directly from the Python official website and install it
Method 2:
If there is ananconda, you can use the command "conda create - n env_py Python=3.8" to create a 3.8 virtual environment
Then activate the virtual environment "conda activate env_py" and proceed with the installation of dependent libraries
[Step 2: Install the required dependency libraries for the software]
First, install the third-party libraries in requirements. txt, and then install the two libraries in the libs directory
Run the following three commands in order: (The command line needs to first enter FaceReconstion)
Pip install - r requirements. txt - i https://pypi.tuna.tsinghua.edu.cn/simple
Pip install libs/dlib-19.19.0-cp38-cp38-win-amd64.whl
Pip install libs/face recognition 1.3.0 py2 py3 none any whl
