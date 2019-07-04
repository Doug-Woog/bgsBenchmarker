This project aims to evaluate the ability of a variety of background subtraction models (found in the bgslibrary project by andrewssobral) to detect humans and then output the resultant data to a file.
Its original purpose was to evaluate video footage from the publically available online PETS2009 dataset and, as such, a folder containing all the groundtruth boxes for a section of video from the PETS2009 dataset has been included. The create_videos.py script can be used to generate this video from the PETS2009 data.

The main script needed to run this project is Model_Testing.py.
Make sure it has access to all required libraries and modules and has access to the _init_paths.py script from the Object-Detection-Metrics project.
It works by pre training each model on a video of the background and then applies it to the full test video to produce a foreground mask. This foreground mask is then enhanced by grouping together pixels to form a more 'blob-like' output and boxes are generated around these pixels cluster. These boxes are evaluated against the groundtruth's boxes to calculate their precision and recall.

The function used to generate the data is the test() function in Model_Testing.py. This function generates a csv file recording the FPS, precision and recall of each model.

Here is an overview of it:

test(models,IOU,WithMOG=True,AP=False)

- 'models' is a string that is the name of a file in the same directory as Model_Testing.py which has a list of the bgs models to be tested seperated by commas and each term is surrounded by apostrophes
- 'IOU' is an array of floats describing which IOU thresholds to test at
- 'WithMOG' is boolean which defaults to True it decides whether the native cv2.createBackgroundSubtractorMOG2() model is also used in addition to the bgslibrary models
- 'AP' is boolean which defaults to False and it decides whether to include a row for the Average Precision in the outputted csv file, however this isn't the real average so it is recommended to leave it as False

There are other parameters and settings in Model_Testing.py that can be set up or changed including:
- The learning rates of the MOG2 model
- Input video dimensions and shape
- Number of Pre-Training Frames
- Whether to display the various videos produced during testing (NOTE: THIS CAN HEAVILY AFFECT FPS)

Notes:
- There is an explicit exeption line for the models beginning with 'LB', 'Multi' and 'VuMeter' as when ran with these models the program produces this error:
RuntimeError: OpenCV(3.4.6) c:\build\3_4_winpack-build-win64-vc15\opencv\modules\core\src\array.cpp:1246: error: (-5:Bad argument) Array should be CvMat or IplImage in function 'cvGetSize'
I am currently unsure as to why it throws this error (Any suggestions would be greatly appreciated)
- A text file called "Available bgs models.txt" is included which can be inputted into test() and lists all the bgs models currently available from bgslibrary (Apart from T2FMRF_UM as that one does not work)
- When evaluating the precision and recall, the first 20 frames are omitted which can be changed in the 'PeopleDetectionMetrics.py' script
- The referenced PETS2009 videos are not included but can be found online
- An example .csv is included
- Make sure there is both a 'groundtruths' and a 'detections' folder which can be accessed by 'PeopleDetectionMetrics.py' and 'Model_Testing.py'
- Make sure to go through the 2 python files and assign the relavent folders to their respective variables (There is only one that should need setting which is bgs_dir in 'Model_Testing.py')
- MixtureOfGaussianV2 model is the cv2.createBackgroundSubtractorMOG2() model but it does not yield the same results as it has been assigned different hyper-parameters such as learning rate