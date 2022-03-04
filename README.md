# Reducing the Noise-to-Signal Ratio to Improve the Robustness of DNN

This work was conducted at Department of Computer Science, University of Miami 
If you find our codes useful, we kindly ask you to cite our work:

Ma, Linhai, and Liang Liang. "Improve robustness of DNN for ECG signal classification: a noise-to-signal ratio perspective." arXiv preprint arXiv:2005.09134 (2020).

L. Ma and L. Liang, "Enhance CNN Robustness Against Noises for Classification of 12-Lead ECG with Variable Length," 2020 19th IEEE International Conference on Machine Learning and Applications (ICMLA), 2020, pp. 839-846, doi: 10.1109/ICMLA51294.2020.00137.

## Keywords: ECG, DNN, robustness, adversarial noises

# Environment
Python version==3.8.3; Pytorch version==1.5.0; Operation system: CentOS 7. Kernel version: is 3.10.0-1062.1.2.el7.x86_64.

# For PhysioNet's MIT-BIH: 
Download data from https://www.kaggle.com/shayanfazeli/heartbeat. Put the csv files at /ecg/ and run preprocess.py.

# For CPSC2018:
Prepare dataset: 

Download data from http://2018.icbeb.org/Challenge.html.
Put the *.mat and *.csv files at "data/CPSC2018/train/" and run preprocess.py

Training and evaluation:

Run "CPSC2018_CNN_NSR.py" for the result of "NSR" in the paper.
Run "CPSC2018_CNN_jacob.py" for the result of "jacob" in the paper.
Run "CPSC2018_CNN_CE.py" for the result of "CE" in the paper.
Run "CPSC2018_CNN_ce_adv_pgd_ls.py" for the result of "advls_$\epsilon$" in the paper.

The parameters can be set in the corresponding .py files and the detailed explaination can be found in the paper.


# Questions
If you have any question, please contact the authors (l.ma@miami.edu or liang.liang@miami.edu).
