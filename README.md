# awesome-speech-enhancement
a list of speech frontend, such as speech enhancement\speech seperation\sound source localization
This repo summarizes the tutorials, datasets, papers, codes and tools for single-/multi-channel speech enhancement/speech seperation task. You are kindly invited to pull requests. 
<!--TODO ...
datasets...
Tutorials...
https://github.com/topics/beamforming
-->

## Table of Contents
- [Speech Enhancement](#SpeechEnhancement)
- [Dereverberation](#Dereverberation)
- [Speech Seperation](#SpeechSeperation(singlechannel))
- [Array Signal Processing](#ArraySignalProcessing)
- [Sound Event Detection](#SoundEventDetection)
- [Tools](#Tools)
- [Resources](#Resources)


## Speech Enhancement
  ### Magnitude spectrogram
  #### IRM 
  * On Training Targets for Supervised Speech Separation, Wang, 2014. [[Paper]](https://ieeexplore.ieee.org/document/6887314) [[IRM-SE-LSTM]](https://github.com/haoxiangsnr/IRM-based-Speech-Enhancement-using-LSTM) [[nn-irm]](https://github.com/zhaoforever/nn-irm) [[rnn-se]](https://github.com/amaas/rnn-speech-denoising) [[DL4SE]](https://github.com/miralv/Deep-Learning-for-Speech-Enhancement)
  * A Hybrid DSP/Deep Learning Approach to Real-Time Full-Band Speech Enhancement, Valin, 2018. [Paper](https://ieeexplore.ieee.org/document/8547084/) [[RNNoise]](https://github.com/xiph/rnnoise)
  * A Perceptually-Motivated Approach for Low-Complexity, Real-Time Enhancement of Fullband Speech, Valin, 2020. [Paper](https://arxiv.org/abs/2008.04259) [[PercepNet]](https://github.com/jzi040941/PercepNet)

  #### Magnitude spectrogram mapping
  * An Experimental Study on Speech Enhancement Based on Deep Neural Networks, Xu, 2014. [[Paper]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6665000)
  * A Regression Approach to Speech Enhancement Based on Deep Neural Networks, Xu, 2014. [[Paper]](https://ieeexplore.ieee.org/document/6932438) [[sednn]](https://github.com/yongxuUSTC/sednn) [[DNN-SE-Xu]](https://github.com/yongxuUSTC/DNN-Speech-enhancement-demo-tool) [[DNN-SE-Li]](https://github.com/hyli666/DNN-SpeechEnhancement) [[SE toolkit]](https://github.com/jtkim-kaist/Speech-enhancement) [[TensorFlow-SE]](https://github.com/linan2/TensorFlow-speech-enhancement-Chinese)
  * Speech enhancement with LSTM recurrent neuralnetworks and its application to noise-robust ASR, Weninger, 2015. [[Paper]](https://hal.inria.fr/hal-01163493/file/weninger_LVA15.pdf)
  * A Fully Convolutional Neural Network for Speech Enhancement, Park, 2016. [[Paper]](https://arxiv.org/abs/1609.07132) [[CNN4SE]](https://github.com/dtx525942103/CNN-for-single-channel-speech-enhancement)
  * A Convolutional Recurrent Neural Network for Real-Time Speech Enhancement, Tan, 2018. [[Paper]](https://web.cse.ohio-state.edu/~wang.77/papers/Tan-Wang1.interspeech18.pdf) [[CRN-Tan]](https://github.com/JupiterEthan/CRN-causal)
  * Convolutional-Recurrent Neural Networks for Speech Enhancement, Zhao, 2018. [[Paper]](https://arxiv.org/pdf/1805.00579.pdf) [[CRN-Hao]](https://github.com/haoxiangsnr/A-Convolutional-Recurrent-Neural-Network-for-Real-Time-Speech-Enhancement)

  ### Complex spectrogram
  * complex spectrogram mapping : [Fu](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8168119&tag=1); [Tan](https://web.cse.ohio-state.edu/~wang.77/papers/Tan-Wang.taslp20.pdf) ([[GCRN]](https://github.com/JupiterEthan/GCRN-complex))
  * complex neural network : [Choi](https://arxiv.org/abs/1903.03107) ([[DC-UNet]](https://github.com/chanil1218/DCUnet.pytorch)); [Hu](https://isca-speech.org/archive/Interspeech_2020/pdfs/2537.pdf) ([[DCCRN]](https://github.com/huyanxin/DeepComplexCRN)); [Kim](https://ieeexplore.ieee.org/document/9053591) 
  * magnitude and phase : [Yin](https://arxiv.org/abs/1911.04697) ([[PHASEN]](https://github.com/huyanxin/phasen))
  * Complex Ratio Masking : [Williamson](https://ieeexplore.ieee.org/abstract/document/7906509); [Chen](http://web.cse.ohio-state.edu/~wang.77/papers/Chen-Wang.jasa17.pdf); [Choi](https://arxiv.org/abs/2006.00687)
  
  ### Time domain
  * [Defossez](https://arxiv.org/abs/2006.12847) ([[facebookDenoiser]](https://github.com/facebookresearch/denoiser)); [Macartney](https://arxiv.org/pdf/1811.11307.pdf) ([[WaveUNet]](https://github.com/YosukeSugiura/Wave-U-Net-for-Speech-Enhancement-NNabla)); [Guimarães](https://www.sciencedirect.com/science/article/pii/S0957417420304061) ([[SEWUNet]](https://github.com/Hguimaraes/SEWUNet)); [Stoller](https://arxiv.org/abs/1806.03185) ([[WaveUNet]](https://github.com/haoxiangsnr/Wave-U-Net-for-Speech-Enhancement)); [Pandey_AECNN](https://ieeexplore.ieee.org/document/8701652); [Ali](https://ieeexplore.ieee.org/document/9211072); [Pandey_TCNN](https://ieeexplore.ieee.org/document/8683634); [Pandey_DCN](https://ieeexplore.ieee.org/document/9372863); [Pandey_DPSARNN](https://arxiv.org/abs/2010.12713)

  ### GAN
  * segan_pytorch [[Code]](https://github.com/santi-pdp/segan_pytorch)
  * relativisticgan [[Code]](https://github.com/deepakbaby/se_relativisticgan)
  * MetricGAN [[Code]](https://github.com/JasonSWFu/MetricGAN)
  * Hifigan-denoiser [[Code]](https://github.com/rishikksh20/hifigan-denoiser)

  ### DNN with traditional SE
  * [Nicolson](https://arxiv.org/abs/1906.07319) ([[DeepXi]](https://github.com/anicolson/DeepXi))
  * [Li](http://staff.ustc.edu.cn/~jundu/Publications/publications/chaili2019trans.pdf) ([[SE-MLC]](https://github.com/LiChaiUSTC/Speech-enhancement-based-on-a-maximum-likelihood-criterion))
  
  ### Subband SE
  * FullSubNet [[Code]](https://github.com/haoxiangsnr/FullSubNet)

  ### NMF
  * Speech_Enhancement_DNN_NMF 
  [[Code]](https://github.com/eesungkim/Speech_Enhancement_DNN_NMF)
  * gcc-nmf:Real-time GCC-NMF Blind Speech Separation and Enhancement 
  [[Code]](https://github.com/seanwood/gcc-nmf)

  ### Multi-stage
  * [Westhausen](https://www.isca-speech.org/archive/Interspeech_2020/pdfs/2631.pdf) ([[DTLN]](https://github.com/breizhn/DTLN))
  * [Xu](http://www.cs.columbia.edu/cg/listen_to_the_silence/paper.pdf) ([[LSS]](https://github.com/henryxrl/Listening-to-Sound-of-Silence-for-Speech-Denoising))
  ### Data collection
  * [Kashyap](https://arxiv.org/pdf/2104.03838.pdf)([[Noise2Noise]](https://github.com/madhavmk/Noise2Noise-audio_denoising_without_clean_training_data))
  
  ### Challenge
  * DNS Challenge [[DNS Interspeech2020]](https://www.microsoft.com/en-us/research/academic-program/deep-noise-suppression-challenge-interspeech-2020/) [[DNS ICASSP2021]](https://www.microsoft.com/en-us/research/academic-program/deep-noise-suppression-challenge-icassp-2021/) [[DNS Interspeech2021]](https://www.microsoft.com/en-us/research/academic-program/deep-noise-suppression-challenge-interspeech-2021/)

  ### Other repositories
  * Collection of papers, datasets and tools on the topic of Speech Dereverberation and Speech Enhancement 
  [[Link]](https://github.com/jonashaag/speech-enhancement)

## Dereverberation
* dereverberation Single-Channel Dereverberation in Matlab 
[[Code]](https://github.com/csd111/dereverberation)
* speech_dereverbaration_using_lp_residual Single Channel Speech Dereverbaration using LP Residual 
[[Code]](https://github.com/shamim-hussain/speech_dereverbaration_using_lp_residual)
* dereverberate 
[[Code]](https://github.com/matangover/dereverberate)
* dereverberation-and-denoising:Supervised Speech Dereverberation in Noisy Environments using Exemplar-based Sparse Representations [[Code]](https://github.com/deepakbaby/dereverberation-and-denoising)
* DNN_WPE [[Code]](https://github.com/nttcslab-sp/dnn_wpe)
* nara_wpe:Different implementations of "Weighted Prediction Error" for speech dereverberation [[Code]](https://github.com/fgnt/nara_wpe)
* Dereverberation-toolkit-for-REVERB-challenge [[Code]](https://github.com/hshi-speech/Dereverberation-toolkit-for-REVERB-challenge)

## Speech Seperation (single channel)
* Tasnet: time-domain audio separation network for real-time, single-channel speech separation
[[Code]](https://github.com/mpariente/asteroid/blob/master/egs/whamr/TasNet)
* Conv-TasNet: Surpassing Ideal Time-Frequency Masking for Speech Separation 
[[Code]](https://github.com/kaituoxu/Conv-TasNet)
* Dual-path RNN: efficient long sequence modeling for time-domain single-channel speech separation
[[Code1]](https://github.com/ShiZiqiang/dual-path-RNNs-DPRNNs-based-speech-separation) 
[[Code2]](https://github.com/JusperLee/Dual-Path-RNN-Pytorch)
* DANet:Deep Attractor Network (DANet) for single-channel speech separation 
[[Code]](https://github.com/naplab/DANet)
* TAC end-to-end microphone permutation and number invariant multi-channel speech separation 
[[Code]](https://github.com/yluo42/TAC)
* uPIT-for-speech-separation:Speech separation with utterance-level PIT 
[[Code]](https://github.com/funcwj/uPIT-for-speech-separation)
* LSTM_PIT_Speech_Separation 
[[Code]](https://github.com/pchao6/LSTM_PIT_Speech_Separation)
* Deep-Clustering
[[Code]](https://github.com/JusperLee/Deep-Clustering-for-Speech-Separation)
[[Code]](https://github.com/simonsuthers/Speech-Separation)
[[Code]](https://github.com/funcwj/deep-clustering)
* sound separation(Google) [[Code]](https://github.com/google-research/sound-separation)
* sound separation: Deep learning based speech source separation using Pytorch [[Code]](https://github.com/AppleHolic/source_separation)
* music-source-separation 
[[Code]](https://github.com/andabi/music-source-separation)
* Singing-Voice-Separation 
[[Code]](https://github.com/Jeongseungwoo/Singing-Voice-Separation)
* Comparison-of-Blind-Source-Separation-techniques[[Code]](https://github.com/TUIlmenauAMS/Comparison-of-Blind-Source-Separation-techniques)
* FastICA[[Code]](https://github.com/ShubhamAgarwal1616/FastICA)
* A localisation- and precedence-based binaural separation algorithm[[Download]](http://iosr.uk/software/downloads/PrecSep_toolbox.zip)
* Convolutive Transfer Function Invariant SDR [[Code]](https://github.com/fgnt/ci_sdr)
* 
## Array Signal Processing
* <font color=red>MASP</font>:Microphone Array Speech Processing [[Code]](https://github.com/ZitengWang/MASP)
* BeamformingSpeechEnhancer 
[[Code]](https://github.com/hkmogul/BeamformingSpeechEnhancer)
* steernet [[Code]](https://github.com/FrancoisGrondin/steernet)
* DNN_Localization_And_Separation 
[[Code]](https://github.com/shaharhoch/DNN_Localization_And_Separation)
* nn-gev:Neural network supported GEV beamformer<font color=red>CHiME3</font> [[Code]](https://github.com/fgnt/nn-gev)
* <font color=red>chime4</font>-nn-mask:Implementation of NN based mask estimator in pytorch（reuse some programming from nn-gev）[[Code]](https://github.com/funcwj/chime4-nn-mask)
* beamformit_matlab:A MATLAB implementation of <font color=red>CHiME4</font> baseline Beamformit  [[Code]](https://github.com/gogyzzz/beamformit_matlab)
* pb_chime5:Speech enhancement system for the <font color=red>CHiME-5</font> dinner party scenario [[Code]](https://github.com/fgnt/pb_chime5)
* beamformit:麦克风阵列算法 [[Code]](https://github.com/592595/beamformit)
* Beamforming-for-speech-enhancement [[Code]](https://github.com/AkojimaSLP/Beamforming-for-speech-enhancement)
* deepBeam [[Code]](https://github.com/auspicious3000/deepbeam)
* NN_MASK [[Code]](https://github.com/ZitengWang/nn_mask)
* Cone-of-Silence [[Code]](https://github.com/vivjay30/Cone-of-Silence)
-----------------------------------------------------------------------
* binauralLocalization 
[[Code]](https://github.com/nicolasobin/binauralLocalization)
* robotaudition_examples:Some Robot Audition simplified examples (sound source localization and separation), coded in Octave/Matlab [[Code]](https://github.com/balkce/robotaudition_examples)
* WSCM-MUSIC
[[Code]](https://github.com/xuchenglin28/WSCM-MUSIC)
* doa-tools
[[Code]](https://github.com/morriswmz/doa-tools)
*  Regression and Classification for Direction-of-Arrival Estimation with Convolutional Recurrent Neural Networks
[[Code]](https://github.com/RoyJames/doa-release) [[PDF]](https://arxiv.org/pdf/1904.08452v3.pdf)
* messl:Model-based EM Source Separation and Localization 
[[Code]](https://github.com/mim/messl)
* messlJsalt15:MESSL wrappers etc for JSALT 2015, including CHiME3 [[Code]](https://github.com/speechLabBcCuny/messlJsalt15)
* fast_sound_source_localization_using_TLSSC:Fast Sound Source Localization Using Two-Level Search Space Clustering
[[Code]](https://github.com/LeeTaewoo/fast_sound_source_localization_using_TLSSC)
* Binaural-Auditory-Localization-System 
[[Code]](https://github.com/r04942117/Binaural-Auditory-Localization-System)
* Binaural_Localization:ITD-based localization of sound sources in complex acoustic environments [[Code]](https://github.com/Hardcorehobel/Binaural_Localization)
* Dual_Channel_Beamformer_and_Postfilter [[Code]](https://github.com/XiaoxiangGao/Dual_Channel_Beamformer_and_Postfilter)
* 麦克风声源定位 [[Code]](https://github.com/xiaoli1368/Microphone-sound-source-localization)
* RTF-based-LCMV-GSC [[Code]](https://github.com/Tungluai/RTF-based-LCMV-GSC)
* DOA [[Code]](https://github.com/wangwei2009/DOA)


## Sound Event Detection
* sed_eval - Evaluation toolbox for Sound Event Detection 
[[Code]](https://github.com/TUT-ARG/sed_eval)
* Benchmark for sound event localization task of DCASE 2019 challenge 
[[Code]](https://github.com/sharathadavanne/seld-dcase2019)
* sed-crnn DCASE 2017 real-life sound event detection winning method. 
[[Code]](https://github.com/sharathadavanne/sed-crnn)
* seld-net 
[[Code]](https://github.com/sharathadavanne/seld-net)

## Tools
* APS:A workspace for single/multi-channel speech recognition & enhancement & separation.  [[Code]](https://github.com/funcwj/aps)
* AKtools:the open software toolbox for signal acquisition, processing, and inspection in acoustics [[SVN Code]](https://svn.ak.tu-berlin.de/svn/AKtools)(username: aktools; password: ak)
* espnet [[Code]](https://github.com/espnet/espnet)
* asteroid:The PyTorch-based audio source separation toolkit for researchers[[PDF]](https://arxiv.org/pdf/2005.04132.pdf)[[Code]](https://github.com/mpariente/asteroid)
* ONSSEN: An Open-source Speech Separation and Enhancement Library 
[[Code]](https://github.com/speechLabBcCuny/onssen)
* separation_data_preparation[[Code]](https://github.com/YongyuG/separation_data_preparation)
* MatlabToolbox [[Code]](https://github.com/IoSR-Surrey/MatlabToolbox)
* athena-signal [[Code]](https://github.com/athena-team/athena-signal）
* python_speech_features [[Code]](https://github.com/jameslyons/python_speech_features)
* speechFeatures:语音处理，声源定位中的一些基本特征 [[Code]](https://github.com/SusannaWull/speechFeatures)
* sap-voicebox [[Code]](https://github.com/ImperialCollegeLondon/sap-voicebox)
* Calculate-SNR-SDR [[Code]](https://github.com/JusperLee/Calculate-SNR-SDR)
* RIR-Generator [[Code]](https://github.com/ehabets/RIR-Generator)
* Python library for Room Impulse Response (RIR) simulation with GPU acceleration [[Code]](https://github.com/DavidDiazGuerra/gpuRIR)
* ROOMSIM:binaural image source simulation [[Code]](https://github.com/Wenzhe-Liu/ROOMSIM)
* binaural-image-source-model [[Code]](https://github.com/iCorv/binaural-image-source-model)
* PESQ [[Code]](https://github.com/vBaiCai/python-pesq)
* SETK: Speech Enhancement Tools integrated with Kaldi 
[[Code]](https://github.com/funcwj/setk)
* pb_chime5:Speech enhancement system for the CHiME-5 dinner party scenario [[Code]](https://github.com/fgnt/pb_chime5)

## Resources
* Speech Signal Processing Course(ZH) [[Link]](https://github.com/veenveenveen/SpeechSignalProcessingCourse)
* Speech Algorithms(ZH) [[Link]](https://github.com/Ryuk17/SpeechAlgorithms)
* CCF语音对话与听觉专业组语音对话与听觉前沿研讨会(ZH) [[Link]](https://www.bilibili.com/video/BV1MV411k7iJ)


