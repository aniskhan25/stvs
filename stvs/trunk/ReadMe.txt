Steps to take for compiling the code for static visual saliency maps.

1) Install nvidia device drivers and cuda toolkit 4.0.

'devdriver_4.0_winvista-win7_64_270.81_general.exe'
'cudatoolkit_4.0.17_win_32.msi'
'gpucomputingsdk_4.0.17_win_32.exe'

2) Install OpenCV 2.3.

'OpenCV-2.3.0rc-win32-vs2008.zip'

3) Create new project in Visual Studio 2008 Express Edition.

4) Right-click project in Solution Explorer >> GOTO Custum Build Rules, select CUDA Runtime API Build rule v4.0.

5) Right-click project in Solution Explorer >> Add existing files >>add all the source and header files.

6) Unzip the source files into your project directory.

7) GOTO Project Property Pages >> C/C++ >> Additional include directories:

8) GOTO Project Property Pages >> C/C++ >> Additional include directories:

"C:\ProgramData\NVIDIA Corporation\NVIDIA GPU Computing SDK 4.0\C\common\inc";C:\OpenCV2.3\include;$(CUDA_PATH_V4_0)\include

Here, I have include directories added for cutil common functions, opencv and cuda sdk.

9) GOTO Project Property Pages >> Linker >> Additional library directories:

"C:/ProgramData/NVIDIA Corporation/NVIDIA GPU Computing SDK 4.0/C/common/lib/$(PlatformName)";C:/OpenCV2.3/lib;$(CUDA_PATH_V4_0)/lib/$(PlatformName);

Here, I have include directories added for cutil common functions, opencv and cuda sdk.

10) GOTO Project Property Pages >> Linker >> Additional dependencies:

cudart.lib cutil32.lib cufft.lib opencv_imgproc230.lib opencv_core230.lib opencv_highgui230.lib

11) GOTO Project Property Pages >> C/C++ >> Code Generation >> Runtime Library >> Multi-threaded (/MT).

12) GOTO Project Property Pages >> Linker >> General >> Output file.

"C:/ProgramData/NVIDIA Corporation/NVIDIA GPU Computing SDK 4.0/C/bin/win32/$(ConfigurationName)/stvs.exe"

13) Compile and execute with all the required .dlls and the .avi file in bin directory.

