The folder contains
1. Sample dictionary and Weight matrix in D.txt and W.txt respectively
2. Dictionary_Learning.m that saves D.txt and W.txt.  Dictionary_learning.m file requires KSVD, OMP and LCKSVD toolboxes that can be downloaded from 
%% requires OMP, KSVD and LCKSVD toolboxes to run this code
%% OMP and KSVD code can be downloaded from http://www.cs.technion.ac.il/~ronrubin/software.html
%% LC-KSVD code can be downloaded from http://www.umiacs.umd.edu/~zhuolin/projectlcksvd.html
3. DLTracing.exe and options_mnt_DL- to run the tracing type the following command 
DLTracing.exe Sample_Image1.tiff Sample_Centroid1.txt Sample_Soma1.tiff options_mnt_DL 0
	#requires ITK library
4. There are two sample images along with the soma centroid, soma image and golden truth.

The  source code and instructions to build farsight can be found at http://www.farsight-toolkit.org/wiki/Main_Page