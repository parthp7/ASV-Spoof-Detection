# ASV-Spoof-Detection
Spoofing detection system for automatic speaker verification

This project is aimed at implementing a comprehensive spoofing detection system for automatic speaker verification based on research paper presented at ASVSPOOF2015 challenge [[1]].

Abstract: 
	
	Anti-Spoofing System for automatic speaker verification

With the growth of interest in reliable ASV systems, the development of the spoofing techniques has increased tremendously.With the different spoofing methods present such as “Replay Attack”, “Cut and Paste”, “Handkerchief tampering”, “Nasalisation tampering”, etc. Despite the development of new spoofing detection methods, most of ASV spoofing countermeasures presented so far depend on a training dataset related to a specific type of attack, while the nature of the spoofing attack is usually unknown in real life. So as a countermeasure for this the paper discusses methods for correctly identifying spoofed speech data using combinations of different front-end acoustic features.
	
The proposed system consists of three main components: acoustic feature extractor, total variability i-vector extractor, and classifier. The first step implementation is done in three domains: Amplitude spectral features like MFCC and MFPC, Phase-based feature like CosPhasePC and wavelet-based feature like Mel-Wavelet Principal Coefficients (MWPC). In all these feature extractors, first 12 coefficients and their first and second derivatives were combined to produce a 36-length feature vector. The window function used is Hamming window function with window length 256 and 50% overlap for extracting the above features. The next stage uses this derived feature vector and produces i-vector using total variability modelling. I-vector of the Total Variability space is extracted by means of Gaussian factor analyser defined on mean supervectors of the Universal Background Model (UBM) and Total-variability matrix T. UBM is represented by the diagonal covariance Gaussian mixture model of the described features. Using each of the features extracted, an i-vector is produced which can be used for classification, but in this system, some combination of the i-vectors are fused together to produce minimum error rate. The fusion is done by combining all vectors into one common vector which is centred and length normalized. The classifiers used for matching this i-vectors are Support Vector Machine (SVM) and Deep Belief Network (DBN). The speech is trained to produce i-vectors through supervised learning using SVM or DBN. This trained vector is compared with produced fused i-vector from testing data to determine if the speech is spoofed or not.

[1]: http://ieeexplore.ieee.org/xpls/icp.jsp?arnumber=7472724
