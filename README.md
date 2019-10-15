# Transferable-E2E-ABSA

Data and source code for our EMNLP'19 Long paper "Transferable End-to-End Aspect-based Sentiment Analysis with Selective Adversarial Learning".

# Introduction

**1) E2E-ABSA**: This task aims to jointly learn aspects as well as their sentiments from user reviews, whch can be effectively formulated as an end-to-end sequence labeling problem based on the unified tagging scheme.

The unified tagging is similar to the NER tagging.

unified tag = aspect boundary tag + sentiment tag

NER tag = entity boundary tag + entity type tag

As we all know, labeling sequence data behaves much more expensive and time-comsuming. 

**2) Transferable-E2E-ABSA**: we firstly explore an unsupervised domain adaptation (UDA) setting for cross-domain E2E-ABSA. Unlike the traditional UDA in classification problems, this task aims to leverage knowledge from a labeled source domain to improve the ***sequence learning*** in an unlabeled target domain.





The code and paper are expected to be released in the late Oct.
