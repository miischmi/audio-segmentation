# Background
This work contributes to the automatic segmentation of opera recordings in audio file format through audio analysis based on segmented reference recordings by synchronization and matching procedures. This segmentation tool was developed for the project "Opera forever" of the Bern University of the Arts (HKB) as part of a paper in the bachelor's degree program in business information technologies (Bsc WI). Detailed information can be found at http://make.opendata.ch/wiki/project:opera_forever

## Data
All data used originates from the Ehrenreich-Collection of the HKB and can be accessed at https://ehrenreich.bfh.science/ (login required).

## Paper
The associated papers
- Automated Segmentation in Audio Analysis in the Ehrenreich Collection (2020)
- Optimizing Feature Representations for Automated Segmentation within the Ehrenreich Collection (2021)
 are in possession of the author (michelle.schmid@students.bfh.ch) and the supervisor (eduard.klein@bfh.ch).  
<br/>
<br/>


# Prerequisites
- Python 3.8<br/>
- pip<br/>

The .wav files used in this project are not published. To run the programm, you must provide your own .wav files.

# Segmentation Tool
All requirements needed to run the program can be found in the file <i>requirements.txt</i>.

*main* - This main is used to match the annotated segments of the reference recording with the complete test recording and requires a correspondingly large amount of computing power.

*main_samples* - This main is used for experimenting. Different short samples of two recordings can be compared with each other.

## Cyclic shift
This function shifts a chroma vector by a certain number of semitones. In other words, the chroma band D♯ becomes the chroma band E, the E becomes F and so on, and the last band B becomes C. At the moment, it still has to be deactivated and adjusted manually depending on the key of the recordings to be compared.

# Issues
**Status May 21:** Librosa still only works with Python version 3.8, see https://discuss.python.org/t/not-able-to-install-package-librosa-using-pip-and-pycharm/5761/2

