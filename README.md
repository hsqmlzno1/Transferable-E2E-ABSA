# Transferable-E2E-ABSA

Data and source code for our EMNLP'19 Long paper, oral, "[Transferable End-to-End Aspect-based Sentiment Analysis with Selective Adversarial Learning](https://hsqmlzno1.github.io/assets/publications/SAL2019.pdf)".


## Update:
Oct 6th, 2019: The experimental code (not fully clean version) has been released.

Oct 31th, 2019: The paper has been released in the arkiv.

# Introduction

**1) E2E-ABSA**: This task aims to jointly learn aspects as well as their sentiments from user reviews, whch can be effectively formulated as an end-to-end sequence labeling problem based on the unified tagging scheme.

The unified tagging is similar to the NER tagging.

**unified tag** = **aspect boundary tag** + **sentiment tag**

**NER tag** = **entity boundary tag** + **entity type tag**

As we all know, labeling sequence data behaves much more expensive and time-comsuming. 

**2) Transferable-E2E-ABSA**: we firstly explore an unsupervised domain adaptation (UDA) setting for cross-domain E2E-ABSA. Unlike the traditional UDA in classification problems, this task aims to leverage knowledge from a labeled source domain to improve the ***sequence learning*** in an unlabeled target domain.

# Requirements
+ Python 2.7.12

+ Tensorflow-gpu 1.4.1

+ numpy 1.15.4


# Environment
+ OS: CentOS Linux release 7.5.1804
+ GPU: NVIDIA TITAN Xp
+ CUDA: 8.0

# Running

### AD-SAL (full Model): 
selective adversairal learning on the low-level AD task.
```
python main.py --train --test -s rest -t service -model_name AD-SAL --selective
```

### AD-AL (ablation Model): 
pure adversairal learning without selectivity on the low-level AD task.
```
python main.py --train --test -s rest -t service -model_name AD-AL
```

### Training over all transfer pairs:
```
./train_AD-AL.sh
./train_AD-SAL.sh
```

# Citation

If the source code and data are useful for your research, please be kindly to give us stars and cite our paper as follows:

```
@article{li2019sal,
  title={Transferable End-to-End Aspect-based Sentiment Analysis with Selective Adversarial Learning},
  author={Li, Zheng and Li, Xin and Wei Ying and Bing Lidong and Zhang Yu and Yang, Qiang},
  conference={EMNLP},
  year={2019}
}
```
