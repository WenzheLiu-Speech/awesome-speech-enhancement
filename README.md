# Awesome Speech Enhancement

This repository summarizes the papers, codes, and tools for single-/multi-channel speech enhancement/speech separation. Welcome to pull requests.
<!--TODO ...
datasets...
Tutorials...
https://github.com/topics/beamforming
-->

## Contents
- [Speech_Enhancement](#Speech_Enhancement)
- [Dereverberation](#Dereverberation)
- [Speech_Seperation](#Speech_Seperation)
- [Array_Signal_Processing](#Array_Signal_Processing)
- [Tools](#Tools)
- [Books](#Books)
- [Resources](#Resources)

## Speech_Enhancement

  ### Magnitude spectrogram
  #### spectral masking
  * 2014, On Training Targets for Supervised Speech Separation, Wang. [[Paper]](https://ieeexplore.ieee.org/document/6887314)  
  * 2018, A Hybrid DSP/Deep Learning Approach to Real-Time Full-Band Speech Enhancement, [Valin](https://github.com/jmvalin). [[Paper]](https://ieeexplore.ieee.org/document/8547084/) [[RNNoise]](https://github.com/xiph/rnnoise) [[RNNoise16k]](https://github.com/YongyuG/rnnoise_16k)
  * 2020, A Perceptually-Motivated Approach for Low-Complexity, Real-Time Enhancement of Fullband Speech, [Valin](https://github.com/jmvalin). [Paper](https://arxiv.org/abs/2008.04259) [[PercepNet]](https://github.com/jzi040941/PercepNet)
  * 2020, Online Monaural Speech Enhancement using Delayed Subband LSTM, Li. [[Paper]](https://arxiv.org/abs/2005.05037)
  * 2020, FullSubNet: A Full-Band and Sub-Band Fusion Model for Real-Time Single-Channel Speech Enhancement, [Hao](https://github.com/haoxiangsnr). [[Paper]](https://arxiv.org/pdf/2010.15508.pdf) [[FullSubNet]](https://github.com/haoxiangsnr/FullSubNet)
  * 2020， Weighted Speech Distortion Losses for Neural-network-based Real-time Speech Enhancement, Xia. [[Paper]](https://www.microsoft.com/en-us/research/uploads/prod/2020/05/0000871.pdf) [[NSNet]](https://github.com/GuillaumeVW/NSNet)
  * 2020, RNNoise-like fixed-point model deployed on Microcontroller using NNoM inference framework [[example]](https://github.com/majianjia/nnom/tree/master/examples/rnn-denoise) [[NNoM]](https://github.com/majianjia/nnom)
  * 2021, RNNoise-Ex: Hybrid Speech Enhancement System based on RNN and Spectral Features. [[Paper]](https://arxiv.org/abs/2105.11813) [[RNNoise-Ex]](https://github.com/CedArctic/rnnoise-ex) 
  * Other IRM-based SE repositories: [[IRM-SE-LSTM]](https://github.com/haoxiangsnr/IRM-based-Speech-Enhancement-using-LSTM) [[nn-irm]](https://github.com/zhaoforever/nn-irm) [[rnn-se]](https://github.com/amaas/rnn-speech-denoising) [[DL4SE]](https://github.com/miralv/Deep-Learning-for-Speech-Enhancement)

  #### spectral mapping
  * 2014, An Experimental Study on Speech Enhancement Based on Deep Neural Networks, [Xu](https://github.com/yongxuUSTC). [[Paper]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6665000)

  * 2014, A Regression Approach to Speech Enhancement Based on Deep Neural Networks, [Xu](https://github.com/yongxuUSTC). [[Paper]](https://ieeexplore.ieee.org/document/6932438) [[sednn]](https://github.com/yongxuUSTC/sednn) [[DNN-SE-Xu]](https://github.com/yongxuUSTC/DNN-Speech-enhancement-demo-tool) [[DNN-SE-Li]](https://github.com/hyli666/DNN-SpeechEnhancement) 

  * Other DNN magnitude spectrum mapping-based SE repositories: [[SE toolkit]](https://github.com/jtkim-kaist/Speech-enhancement) [[TensorFlow-SE]](https://github.com/linan2/TensorFlow-speech-enhancement-Chinese) [[UNetSE]](https://github.com/vbelz/Speech-enhancement)

  * 2015, Speech enhancement with LSTM recurrent neural networks and its application to noise-robust ASR, Weninger. [[Paper]](https://hal.inria.fr/hal-01163493/file/weninger_LVA15.pdf)

  * 2016, A Fully Convolutional Neural Network for Speech Enhancement, Park. [[Paper]](https://arxiv.org/abs/1609.07132) [[CNN4SE]](https://github.com/zhr1201/CNN-for-single-channel-speech-enhancement)

  * 2017, Long short-term memory for speaker generalizationin supervised speech separation, Chen. [[Paper]](http://web.cse.ohio-state.edu/~wang.77/papers/Chen-Wang.jasa17.pdf)

  * 2018, A Convolutional Recurrent Neural Network for Real-Time Speech Enhancement, [Tan](https://github.com/JupiterEthan). [[Paper]](https://web.cse.ohio-state.edu/~wang.77/papers/Tan-Wang1.interspeech18.pdf) [[CRN-Tan]](https://github.com/JupiterEthan/CRN-causal)

  * 2018, Convolutional-Recurrent Neural Networks for Speech Enhancement, Zhao. [[Paper]](https://arxiv.org/pdf/1805.00579.pdf) [[CRN-Hao]](https://github.com/haoxiangsnr/A-Convolutional-Recurrent-Neural-Network-for-Real-Time-Speech-Enhancement)

    

  ### Complex domain
  * 2017, Complex spectrogram enhancement by convolutional neural network with multi-metrics learning, [Fu](https://github.com/JasonSWFu). [[Paper]](https://arxiv.org/pdf/1704.08504.pdf)

  * 2017, Time-Frequency Masking in the Complex Domain for Speech Dereverberation and Denoising, Williamson. [[Paper]](https://ieeexplore.ieee.org/abstract/document/7906509)

  * 2019, PHASEN: A Phase-and-Harmonics-Aware Speech Enhancement Network, Yin. [[Paper]](https://arxiv.org/abs/1911.04697) [[PHASEN]](https://github.com/huyanxin/phasen)

  * 2019, Phase-aware Speech Enhancement with Deep Complex U-Net, Choi. [[Paper]](https://arxiv.org/abs/1903.03107) [[DC-UNet]](https://github.com/chanil1218/DCUnet.pytorch)

  * 2020, Learning Complex Spectral Mapping With GatedConvolutional Recurrent Networks forMonaural Speech Enhancement, [Tan](https://github.com/JupiterEthan). [[Paper]](https://web.cse.ohio-state.edu/~wang.77/papers/Tan-Wang.taslp20.pdf) [[GCRN]](https://github.com/JupiterEthan/GCRN-complex)

  * 2020, DCCRN: Deep Complex Convolution Recurrent Network for Phase-AwareSpeech Enhancement, [Hu](https://github.com/huyanxin). [[Paper]](https://isca-speech.org/archive/Interspeech_2020/pdfs/2537.pdf) [[DCCRN]](https://github.com/huyanxin/DeepComplexCRN)

  * 2020, T-GSA: Transformer with Gaussian-Weighted Self-Attention for Speech Enhancement, Kim. [[Paper]](https://ieeexplore.ieee.org/document/9053591) 

  * 2020, Phase-aware Single-stage Speech Denoising and Dereverberation with U-Net, Choi. [[Paper]](https://arxiv.org/abs/2006.00687)

  * 2021, DPCRN: Dual-Path Convolution Recurrent Network for Single Channel Speech Enhancement, [Le](https://github.com/Le-Xiaohuai-speech). [[Paper]](https://www.isca-speech.org/archive/pdfs/interspeech_2021/le21b_interspeech.pdf) [[DPCRN]](https://github.com/Le-Xiaohuai-speech/DPCRN_DNS3)

  * 2021, Real-time denoising and dereverberation with tiny recurrent u-net, Choi. [[Paper]](https://arxiv.org/pdf/2102.03207.pdf)

  * 2021, DCCRN+: Channel-wise Subband DCCRN with SNR Estimation for Speech Enhancement, [Lv](https://github.com/IMYBo/)  [[Paper]](https://arxiv.org/abs/2106.08672)

  * 2022, FullSubNet+: Channel Attention FullSubNet with Complex Spectrograms for Speech Enhancement, [Chen](https://github.com/hit-thusz-RookieCJ)  [[Paper]](https://arxiv.org/abs/2203.12188) [[FullSubNet+]](https://github.com/hit-thusz-RookieCJ/FullSubNet-plus)

  * 2022, Dual-branch Attention-In-Attention Transformer for single-channel speech enhancement, [Yu](https://github.com/yuguochencuc)  [[Paper]](https://arxiv.org/abs/2110.06467)

    

### Time domain

  * 2018, Improved Speech Enhancement with the Wave-U-Net, Macartney. [[Paper]](https://arxiv.org/pdf/1811.11307.pdf) [[WaveUNet]](https://github.com/YosukeSugiura/Wave-U-Net-for-Speech-Enhancement-NNabla) 
  * 2019, A New Framework for CNN-Based Speech Enhancement in the Time Domain, [Pandey](https://github.com/ashutosh620). [[Paper]](https://ieeexplore.ieee.org/document/8701652) 
  * 2019, TCNN: Temporal Convolutional Neural Network for Real-time Speech Enhancement in the Time Domain, [Pandey](https://github.com/ashutosh620). [[Paper]](https://ieeexplore.ieee.org/document/8683634)
  * 2020, Real Time Speech Enhancement in the Waveform Domain, Defossez. [[Paper]](https://arxiv.org/abs/2006.12847) [[facebookDenoiser]](https://github.com/facebookresearch/denoiser)
  * 2020, Monaural speech enhancement through deep wave-U-net, Guimarães. [[Paper]](https://www.sciencedirect.com/science/article/pii/S0957417420304061) [[SEWUNet]](https://github.com/Hguimaraes/SEWUNet) 
  * 2020, Speech Enhancement Using Dilated Wave-U-Net: an Experimental Analysis, Ali. [[Paper]](https://ieeexplore.ieee.org/document/9211072)
  * 2020, Densely Connected Neural Network with Dilated Convolutions for Real-Time Speech Enhancement in the Time Domain, [Pandey](https://github.com/ashutosh620). [[Paper]](https://ashutosh620.github.io/files/DDAEC_ICASSP_2020.pdf) [[DDAEC]](https://github.com/ashutosh620/DDAEC)
  * 2021, Dense CNN With Self-Attention for Time-Domain Speech Enhancement, [Pandey](https://github.com/ashutosh620). [[Paper]](https://ieeexplore.ieee.org/document/9372863)
  * 2021, Dual-path Self-Attention RNN for Real-Time Speech Enhancement, [Pandey](https://github.com/ashutosh620). [[Paper]](https://arxiv.org/abs/2010.12713)
  * 2022, Speech Denoising in the Waveform Domain with Self-Attention, Kong. [[Paper]](https://arxiv.org/abs/2202.07790)

  ### GAN
  * 2017, SEGAN: Speech Enhancement Generative Adversarial Network, Pascual. [[Paper]](https://arxiv.org/pdf/1703.09452.pdfsegan_pytorch) [[SEGAN]](https://github.com/santi-pdp/segan_pytorch)
  * 2019, SERGAN: Speech enhancement using relativistic generative adversarial networks with gradient penalty, [Deepak Baby]((https://github.com/deepakbaby)). [[Paper]](https://biblio.ugent.be/publication/8613639/file/8646769.pdf) [[SERGAN]](https://github.com/deepakbaby/se_relativisticgan)
  * 2019, MetricGAN: Generative Adversarial Networks based Black-box Metric Scores Optimization for Speech Enhancement, [Fu](https://github.com/JasonSWFu). [[Paper]](https://arxiv.org/pdf/1905.04874.pdf) [[MetricGAN]](https://github.com/JasonSWFu/MetricGAN)
  * 2019, MetricGAN+: An Improved Version of MetricGAN for Speech Enhancement, [Fu](https://github.com/JasonSWFu). [[Paper]](https://arxiv.org/abs/2104.03538) [[MetricGAN+]](https://github.com/speechbrain/speechbrain/tree/develop/recipes/Voicebank/enhance/MetricGAN)
  * 2020, HiFi-GAN: High-Fidelity Denoising and Dereverberation Based on Speech Deep Features in Adversarial Networks, Su. [[Paper]](https://arxiv.org/abs/2006.05694) [[HifiGAN]](https://github.com/rishikksh20/hifigan-denoiser)

  ### Hybrid SE 
  * 2019, Deep Xi as a Front-End for Robust Automatic Speech Recognition, [Nicolson](https://github.com/anicolson). [[Paper]](https://arxiv.org/abs/1906.07319) [[DeepXi]](https://github.com/anicolson/DeepXi)
  * 2019, Using Generalized Gaussian Distributions to Improve Regression Error Modeling for Deep-Learning-Based Speech Enhancement, [Li](https://github.com/LiChaiUSTC). [[Paper]](http://staff.ustc.edu.cn/~jundu/Publications/publications/chaili2019trans.pdf) [[SE-MLC]](https://github.com/LiChaiUSTC/Speech-enhancement-based-on-a-maximum-likelihood-criterion)
  * 2020, Deep Residual-Dense Lattice Network for Speech Enhancement, [Nikzad](https://github.com/nick-nikzad). [[Paper]](https://arxiv.org/pdf/2002.12794.pdf) [[RDL-SE]](https://github.com/nick-nikzad/RDL-SE)
  * 2020, DeepMMSE: A Deep Learning Approach to MMSE-based Noise Power Spectral Density Estimation, [Zhang](https://github.com/yunzqq). [[Paper]](https://ieeexplore.ieee.org/document/9066933)
  * 2020, Speech Enhancement Using a DNN-Augmented Colored-Noise Kalman Filter, [Yu](https://github.com/Hongjiang-Yu). [[Paper]](https://www.sciencedirect.com/science/article/pii/S0167639320302831) [[DNN-Kalman]](https://github.com/Hongjiang-Yu/DNN_Kalman_Filter)

  <!--### NMF
  * Speech_Enhancement_DNN_NMF 
    [[Code]](https://github.com/eesungkim/Speech_Enhancement_DNN_NMF)
  * gcc-nmf:Real-time GCC-NMF Blind Speech Separation and Enhancement 
    [[Code]](https://github.com/seanwood/gcc-nmf)
  * https://github.com/Jerry-jwz/Audio-Enhancement-via-ONMF-->

  ### Decoupling-style
  * 2020, A Recursive Network with Dynamic Attention for Monaural Speech Enhancement, [Li](https://github.com/Andong-Li-speech). [[Paper]](https://arxiv.org/abs/2003.12973) [[DARCN]](https://github.com/Andong-Li-speech/DARCN)
  * 2020, Masking and Inpainting: A Two-Stage Speech Enhancement Approach for Low SNR and Non-Stationary Noise, [Hao](https://github.com/haoxiangsnr). [[Paper]](https://ieeexplore.ieee.org/document/9053188/)
  * 2020, A Joint Framework of Denoising Autoencoder and Generative Vocoder for Monaural Speech Enhancement, Du. [[Paper]](https://ieeexplore.ieee.org/document/9082858)
  * 2020, Dual-Signal Transformation LSTM Network for Real-Time Noise Suppression, [Westhausen](https://github.com/breizhn). [[Paper]](https://www.isca-speech.org/archive/Interspeech_2020/pdfs/2631.pdf) [[DTLN]](https://github.com/breizhn/DTLN)
  * 2020, Listening to Sounds of Silence for Speech Denoising, [Xu](https://github.com/henryxrl). [[Paper]](http://www.cs.columbia.edu/cg/listen_to_the_silence/paper.pdf) [[LSS]](https://github.com/henryxrl/Listening-to-Sound-of-Silence-for-Speech-Denoising)
  * 2021, ICASSP 2021 Deep Noise Suppression Challenge: Decoupling Magnitude and Phase Optimization with a Two-Stage Deep Network, [Li](https://github.com/Andong-Li-speech). [[Paper]](https://arxiv.org/abs/2102.04198)
  * 2022, Glance and Gaze: A Collaborative Learning Framework for Single-channel Speech Enhancement, [Li](https://github.com/Andong-Li-speech/GaGNet) [[Paper]](https://www.sciencedirect.com/science/article/pii/S0003682X21005934)
  * 2022, HGCN : harmonic gated compensation network for speech enhancement, [Wang](https://github.com/wangtianrui/HGCN). [[Paper]](https://arxiv.org/pdf/2201.12755.pdf)
  * 2022, Uformer: A Unet based dilated complex & real dual-path conformer network for simultaneous speech enhancement and dereverberation, [Fu](https://github.com/felixfuyihui). [[Paper]](https://arxiv.org/abs/2111.06015) [[Uformer]](https://github.com/felixfuyihui/Uformer)
  * 2022， DeepFilterNet2: Towards Real-Time Speech Enhancement on Embedded Devices for Full-Band Audio, [Schröter](https://github.com/Rikorose). [[Paper]](https://arxiv.org/abs/2205.05474) [[DeepFilterNet]](https://github.com/Rikorose/DeepFilterNet)
  * 2021, Multi-Task Audio Source Separation, [Zhang](https://github.com/Windstudent). [[Paper]](https://arxiv.org/abs/2107.06467) [[Code]](https://github.com/Windstudent/Complex-MTASSNet)

  ### Data collection
  * [Kashyap](https://arxiv.org/pdf/2104.03838.pdf)([[Noise2Noise]](https://github.com/madhavmk/Noise2Noise-audio_denoising_without_clean_training_data))
  ### Loss 
  * [[Quality-Net]](https://github.com/JasonSWFu/Quality-Net)
  ### Challenge
  * DNS Challenge [[DNS Interspeech2020]](https://www.microsoft.com/en-us/research/academic-program/deep-noise-suppression-challenge-interspeech-2020/) [[DNS ICASSP2021]](https://www.microsoft.com/en-us/research/academic-program/deep-noise-suppression-challenge-icassp-2021/) [[DNS Interspeech2021]](https://www.microsoft.com/en-us/research/academic-program/deep-noise-suppression-challenge-interspeech-2021/)

  ### Other repositories
  * Collection of papers, datasets and tools on the topic of Speech Dereverberation and Speech Enhancement 
    [[Link]](https://github.com/jonashaag/speech-enhancement)
  * nanahou's awesome speech enhancement [[Link]](https://github.com/nanahou/Awesome-Speech-Enhancement)

## Dereverberation
### Traditional method
  * SPENDRED [[Paper]](https://ieeexplore.ieee.org/document/7795155)
    [[SPENDRED]](https://github.com/csd111/dereverberation)
  * WPE(MCLP) [[Paper]](https://ieeexplore.ieee.org/document/6255769)[[nara-WPE]](https://github.com/fgnt/nara_wpe)
  * GWPE [[Code]](https://github.com/snsun/gwpe-speech-dereverb)
  * LP Residual [[Paper]](https://ieeexplore.ieee.org/abstract/document/1621193) [[LP_residual]](https://github.com/shamim-hussain/speech_dereverbaration_using_lp_residual)
  * dereverberate [[Paper]](https://www.aes.org/e-lib/browse.cfm?elib=15675) [[Code]](https://github.com/matangover/dereverberate)
  * NMF [[Paper]](https://ieeexplore.ieee.org/document/7471656/) [[NMF]](https://github.com/deepakbaby/dereverberation-and-denoising)
### Hybrid method
  * DNN_WPE [[Paper]](https://ieeexplore.ieee.org/document/7471656/) [[Code]](https://github.com/nttcslab-sp/dnn_wpe)
### NN-based Derev
  * Dereverberation-toolkit-for-REVERB-challenge [[Code]](https://github.com/hshi-speech/Dereverberation-toolkit-for-REVERB-challenge)
  * SkipConvNet [[Paper]](https://arxiv.org/pdf/2007.09131.pdf) [[Code]](https://github.com/zehuachenImperial/SkipConvNet)

## Speech Separation (single channel)
* Tutorial speech separation, like awesome series [[Link]](https://github.com/gemengtju/Tutorial_Separation)
### NN-based separation
* 2015, Deep-Clustering:Discriminative embeddings for segmentation and separation, Hershey and Chen.[[Paper]](https://arxiv.org/abs/1508.04306)
  [[Code]](https://github.com/JusperLee/Deep-Clustering-for-Speech-Separation)
  [[Code]](https://github.com/simonsuthers/Speech-Separation)
  [[Code]](https://github.com/funcwj/deep-clustering)
* 2016, DANet:Deep Attractor Network (DANet) for single-channel speech separation, Chen.[[Paper]](https://arxiv.org/abs/1611.08930)
  [[Code]](https://github.com/naplab/DANet)
* 2017, Multitalker speech separation with utterance-level permutation invariant training of deep recurrent, Yu.[[Paper]](https://ai.tencent.com/ailab/media/publications/Multi-talker_Speech_Separation_with_Utterance-level.pdf)
  [[Code]](https://github.com/funcwj/uPIT-for-speech-separation)
* 2018, LSTM_PIT_Speech_Separation 
  [[Code]](https://github.com/pchao6/LSTM_PIT_Speech_Separation)
* 2018, Tasnet: time-domain audio separation network for real-time, single-channel speech separation, Luo.[[Paper]](https://arxiv.org/abs/1711.00541v2)
  [[Code]](https://github.com/mpariente/asteroid/blob/master/egs/whamr/TasNet)
* 2019, Conv-TasNet: Surpassing Ideal Time-Frequency Masking for Speech Separation, Luo.[(Paper)](https://arxiv.org/pdf/1809.07454.pdf)
  [[Code]](https://github.com/kaituoxu/Conv-TasNet)
* 2019, Dual-path RNN: efficient long sequence modeling for time-domain single-channel speech separation, Luo.[[Paper]](https://arxiv.org/abs/1910.06379v1)
  [[Code1]](https://github.com/ShiZiqiang/dual-path-RNNs-DPRNNs-based-speech-separation) 
  [[Code2]](https://github.com/JusperLee/Dual-Path-RNN-Pytorch)
* 2019, TAC end-to-end microphone permutation and number invariant multi-channel speech separation, Luo.[[Paper]](https://arxiv.org/abs/1910.14104) 
  [[Code]](https://github.com/yluo42/TAC)
* 2020, Continuous Speech Separation with Conformer, Chen.[[Paper]](https://arxiv.org/abs/2008.05773) [[Code]](https://github.com/Sanyuan-Chen/CSS_with_Conformer)
* 2020, Dual-Path Transformer Network: Direct Context-Aware Modeling for End-to-End Monaural Speech Separation, Chen.[[Paper]](https://arxiv.org/abs/2007.13975) [[Code]](https://github.com/ujscjj/DPTNet)
* 2020, Wavesplit: End-to-End Speech Separation by Speaker Clustering, Zeghidour.[[Paper]](https://arxiv.org/abs/2002.08933)
* 2021, Attention is All You Need in Speech Separation, Subakan.[[Paper]](https://arxiv.org/abs/2010.13154) [[Code]](https://github.com/speechbrain/speechbrain/tree/develop/recipes/WSJ0Mix/separation)
* 2021, Ultra Fast Speech Separation Model with Teacher Student Learning, Chen.[[Paper]](https://www.isca-speech.org/archive/pdfs/interspeech_2021/chen21l_interspeech.pdf) 
* sound separation(Google) [[Code]](https://github.com/google-research/sound-separation)
* sound separation: Deep learning based speech source separation using Pytorch [[Code]](https://github.com/AppleHolic/source_separation)
* music-source-separation 
  [[Code]](https://github.com/andabi/music-source-separation)
* Singing-Voice-Separation 
  [[Code]](https://github.com/Jeongseungwoo/Singing-Voice-Separation)
* Comparison-of-Blind-Source-Separation-techniques[[Code]](https://github.com/TUIlmenauAMS/Comparison-of-Blind-Source-Separation-techniques)
### BSS/ICA method
* FastICA[[Code]](https://github.com/ShubhamAgarwal1616/FastICA)
* A localisation- and precedence-based binaural separation algorithm[[Download]](http://iosr.uk/software/downloads/PrecSep_toolbox.zip)
* Convolutive Transfer Function Invariant SDR [[Code]](https://github.com/fgnt/ci_sdr)
* 
## Array Signal Processing
* MASP:Microphone Array Speech Processing [[Code]](https://github.com/ZitengWang/MASP)
* BeamformingSpeechEnhancer 
[[Code]](https://github.com/hkmogul/BeamformingSpeechEnhancer)
* TSENet [[Code]](https://github.com/felixfuyihui/felixfuyihui.github.io)
* steernet [[Code]](https://github.com/FrancoisGrondin/steernet)
* DNN_Localization_And_Separation [[Code]](https://github.com/shaharhoch/DNN_Localization_And_Separation)
* nn-gev:Neural network supported GEV beamformer CHiME3 [[Code]](https://github.com/fgnt/nn-gev)
* chime4-nn-mask:Implementation of NN based mask estimator in pytorch（reuse some programming from nn-gev）[[Code]](https://github.com/funcwj/chime4-nn-mask)
* beamformit_matlab:A MATLAB implementation of CHiME4 baseline Beamformit  [[Code]](https://github.com/gogyzzz/beamformit_matlab)
* pb_chime5:Speech enhancement system for the CHiME-5 dinner party scenario [[Code]](https://github.com/fgnt/pb_chime5)
* beamformit:麦克风阵列算法 [[Code]](https://github.com/592595/beamformit)
* Beamforming-for-speech-enhancement [[Code]](https://github.com/AkojimaSLP/Beamforming-for-speech-enhancement)
* deepBeam [[Code]](https://github.com/auspicious3000/deepbeam)
* NN_MASK [[Code]](https://github.com/ZitengWang/nn_mask)
* Cone-of-Silence [[Code]](https://github.com/vivjay30/Cone-of-Silence)

## Tools
* APS:A workspace for single/multi-channel speech recognition & enhancement & separation.  [[Code]](https://github.com/funcwj/aps)
* AKtools:the open software toolbox for signal acquisition, processing, and inspection in acoustics [[SVN Code]](https://svn.ak.tu-berlin.de/svn/AKtools)(username: aktools; password: ak)
* espnet [[Code]](https://github.com/espnet/espnet)
* asteroid:The PyTorch-based audio source separation toolkit for researchers[[PDF]](https://arxiv.org/pdf/2005.04132.pdf)[[Code]](https://github.com/mpariente/asteroid)
* pytorch_complex [[Code]](https://github.com/kamo-naoyuki/pytorch_complex)
* ONSSEN: An Open-source Speech Separation and Enhancement Library 
[[Code]](https://github.com/speechLabBcCuny/onssen)
* separation_data_preparation[[Code]](https://github.com/YongyuG/separation_data_preparation)
* MatlabToolbox [[Code]](https://github.com/IoSR-Surrey/MatlabToolbox)
* athena-signal [[Code]](https://github.com/athena-team/athena-signal）
* python_speech_features [[Code]](https://github.com/jameslyons/python_speech_features)
* speechFeatures [[Code]](https://github.com/SusannaWull/speechFeatures)
* sap-voicebox [[Code]](https://github.com/ImperialCollegeLondon/sap-voicebox)
* Calculate-SNR-SDR [[Code]](https://github.com/JusperLee/Calculate-SNR-SDR)
* RIR-Generator [[Code]](https://github.com/ehabets/RIR-Generator)
* Signal-Generator (for moving sources or a moving array) [[Code]](https://github.com/ehabets/Signal-Generator)
* Python library for Room Impulse Response (RIR) simulation with GPU acceleration [[Code]](https://github.com/DavidDiazGuerra/gpuRIR)
* ROOMSIM:binaural image source simulation [[Code]](https://github.com/Wenzhe-Liu/ROOMSIM)
* binaural-image-source-model [[Code]](https://github.com/iCorv/binaural-image-source-model)
* PESQ [[Code]](https://github.com/vBaiCai/python-pesq)
* SETK: Speech Enhancement Tools integrated with Kaldi 
[[Code]](https://github.com/funcwj/setk)
* pb_chime5:Speech enhancement system for the CHiME-5 dinner party scenario [[Code]](https://github.com/fgnt/pb_chime5)

## Books
* P. C.Loizou: Speech Enhancement: Theory and Practice
* J. Benesty, Y. Huang: Adaptive Signal Processing: Applications to Real-World Problems
* S. Haykin: Adaptive Filter Theory
* Eberhard Hansler, Gerhard Schmidt: Single-Channel Acoustic Echo Cancellation 和 Topics in Acoustic Echo and Noise Control
* J. Benesty, S. Makino, J. Chen: Speech Enhancement
* J. Benesty, M. M. Sondhi, Y. Huang: Handbook Of Speech Processing
* Ivan J. Tashev: Sound Capture and Processing: Practical Approaches
* I. Cohen, J. Benesty, S. Gannot: Speech Processing in Modern Communication
* E. Vincent, T. Virtanen, S. Gannot: Audio Source Separation and Speech Enhancement
* J. Benesty 等: A Perspective on Stereophonic Acoustic Echo Cancellation
* J. Benesty 等: Advances in Network and Acoustic Echo Cancellation
* T. F.Quatieri: Discrete-time speech signal processing: principles and practice
* 宋知用: MATLAB在语音信号分析与合成中的应用
* Harry L.Van Trees: Optimum Array Processing
* 王永良: 空间谱估计理论与算法
* 鄢社锋: 优化阵列信号处理
* 张小飞: 阵列信号处理及matlab实现
* 赵拥军: 宽带阵列信号波达方向估计理论与方法


## Resources
* Speech Signal Processing Course(ZH) [[Link]](https://github.com/veenveenveen/SpeechSignalProcessingCourse)
* Speech Algorithms(ZH) [[Link]](https://github.com/Ryuk17/SpeechAlgorithms)
* Speech Resources[[Link]](https://github.com/ddlBoJack/Speech-Resources)
* Sound capture and speech enhancement for speech-enabled devices [[Link]](https://www.microsoft.com/en-us/research/uploads/prod/2022/01/Sound-capture-and-speech-enhancement-for-speech-enabled-devices-ASA-181.pdf)
* CCF语音对话与听觉专业组语音对话与听觉前沿研讨会(ZH) [[Link]](https://www.bilibili.com/video/BV1MV411k7iJ)
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


