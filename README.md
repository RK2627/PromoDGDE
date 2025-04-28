# PromoDGDE

This is the repository for the manuscript: De novo promoter design method based on deep generative and dynamic evolution algorithm.

# Requirements

You need to configure the following environment before running the PromoDGDE model. 
We  provide *environment.yml* for use by our peer.

* PyCharm 2020

* python = 3.6.3

* cuda-cudart=11.7.99=0

* cuda-cupti=11.7.101=0

* cuda-libraries=11.7.1=0

* cuda-nvrtc=11.7.99=0

* cuda-runtime=11.7.1=0

* cudatoolkit=11.3.1=ha36c431_9

* pytorch=1.10.2=py3.6_cuda11.3_cudnn8.2.0_0

* pytorch-cuda=11.7=h778d358_3

* pytorch-mutex=1.0=cuda

* numpy = 1.19.5

* pandas = 1.1.5

* scikit-learn = 0.19.1

* scipy = 1.1.0

# Datasets
URL: https://zenodo.org/records/15295470

content:
* *E.coli*: ecoli_exp.csv, ecoli_mpra_seq.fa
* *S. cerevisiae*: SC_exp_short.csv, SC_seq_short.txt

Reference: 
* Johns, N.I., Gomes, A.L.C., Yim, S.S., Yang, A., Blazejewski, T., Smillie, C.S., Smith, M.B., Alm, E.J., Kosuri, S. and Wang, H.H. Metagenomic mining of regulatory elements enables programmable species-selective gene expression. Nat Methods, 2018; 15: 323-329. http://dx.doi.org/10.1038/nmeth.4633

* Vaishnav, E.D., de Boer, C.G., Molinet, J., Yassour, M., Fan, L., Adiconis, X., Thompson, D.A., Levin, J.Z., Cubillos, F.A. and Regev, A. The evolution, evolvability and engineering of gene regulatory DNA. Nature, 2022; 603: 455-463. http://dx.doi.org/10.1038/s41586-022-04506-6

# model training

The training of the model consists of three parts: generation, prediction, and optimization.

### generation

run *./PromoDGDE/Generator/diffusion_gan.py*. 

PromoDGDE learns sequence features of natural promoters and generates new promoter sequences by Diffusion-GAN.

### prediction

run *./PromoDGDE/Predictor/predictor_train.py*. 

PromoDGDE learns local features and long-range dependencies of sequences by combining multiscale CNNs and LSTMs to predict promoter expression level values.

### optimization

run *./PromoDGDE/Optimizer/opt_RL.py*. 

PromoDGDE combines reinforcement learning with evolutionary algorithms to achieve functional optimization of promoter sequences.

















