%% requires OMP, KSVD and LCKSVD toolboxes to run this code
%% OMP and KSVD code can be downloaded from http://www.cs.technion.ac.il/~ronrubin/software.html
%% LC-KSVD code can be downloaded from http://www.umiacs.umd.edu/~zhuolin/projectlcksvd.html
%% change the dir, imfname(input image file name), labfnmae(ground truth traces), testfname(test file name)
clear all
close all
clc
%% read image files %%%
dir = 'C:\directory_name\';
imfname = 'input_Image.tif'; %% input microglia image
labfnmae = 'label_Image.tif'; %% corresponding manual traces for learning dictionary
testfname = 'test_Image.tif'; %% test image
blocksize = [15 15 3]; % patch size-should be odd number
info = imfinfo([dir,imfname]);
numRows = info(1).Width;
numColumns = info(1).Height;
numZsclices = size(info,1);
centerSlice = 3:15;
for z=1:numZsclices
     imdata(:,:,z) = double(imread([dir,imfname],z));
     labdata(:,:,z) = double(imread([dir,labfnmae],z));
     testdata(:,:,z) = double(imread([dir,testfname],z));
end
xc = [];
yc = [];
zc = [];
[xtmp,ytmp] = find(labdata(:,:,:)>=255);
xc = [xc;xtmp];yc = [yc;ytmp];
zc =[zc;ones(length(xc),1)*centerSlice];
numtrain = 50000;
if(length(xtmp) < numtrain)
    numtrain = length(xtmp);
end
ids = floor(1+length(xc)*rand(numtrain,1));
xc = xc(ids);yc = yc(ids);
zc = zc(ids);
mtrain_feats1 = [];
%% collect training set for class 1 (trace points)%
for i = 1:length(xc)
    
    xini = xc(i) - floor(blocksize(1)/2);
    xfin = xc(i) + floor(blocksize(1)/2);
    if (xini<1 ||  xfin>numColumns);continue;end
    xrange = xini:xfin;
    
    yini = yc(i) - floor(blocksize(2)/2);
    yfin = yc(i) + floor(blocksize(2)/2);
    if (yini<1 ||  yfin>numRows);continue;end
    yrange = yini:yfin;    
    
	zini = zc(i) - floor(blocksize(3)/2);
    zfin = zc(i) + floor(blocksize(3)/2);
    if (zini<1 ||  zfin>numRows);continue;end
    zrange = zini:zfin;  

    tmp = imdata(xrange,yrange,zrange);
    mtrain_feats1 = [mtrain_feats1 tmp(:)];
end

mH_train = zeros(2,size(mtrain_feats1,2));
mH_train(1,:)=1;

%% collect training set for class 2 (background)%
xc = [];
yc = [];
[xtmp,ytmp] = find(labdata(:,:,centerSlice)==0);
ztmp =[zc;ones(length(xtmp),1)*centerSlice];
mtrain_feats2 = [];
counter = 0;
isTrue = 1;
while isTrue  %% collect data till we have equal number of examples as class 1
    xc = xtmp(floor(1+length(xtmp)*rand));
    xini = xc - floor(blocksize(1)/2);
    xfin = xc + floor(blocksize(1)/2);
    if (xini<1 ||  xfin>numColumns);continue;end
    xrange = xini:xfin;
%     yc =  floor(1+numRows*rand);
    yc = ytmp(floor(1+length(ytmp)*rand));
    yini = yc - floor(blocksize(2)/2);
    yfin = yc + floor(blocksize(2)/2);
    if (yini<1 ||  yfin>numRows);continue;end
    yrange = yini:yfin;    

    zc = ztmp(floor(1+length(ytmp)*rand));
    zfin = zc + floor(blocksize(3)/2);
    if (zini<1 ||  zfin>numRows);continue;end
    zrange = zini:zfin;

    tmp = imdata(xrange,yrange,zrange);
    mtrain_feats2 = [mtrain_feats2 tmp(:)];
    counter = counter+1;
    if counter == size(mtrain_feats1,2)
        isTrue = 0;
    end
end

tmp = zeros(2,size(mtrain_feats2,2));
tmp(2,:)=1;
mH_train = [mH_train tmp];
mtrain_feats = [mtrain_feats1 mtrain_feats2];

%%% extract patches for testing %%%%%%%%%%%%%%%%%%%%%%%%%%%%
mtest_feats = [];

xini = floor(blocksize(1)/2)+1;
yini = floor(blocksize(2)/2)+1;
zini = floor(blocksize(3)/2)+1;
XYcoord = [];
fblocksize = floor(blocksize/2);
zrange = centerSlice-fblocksize(3):centerSlice+fblocksize(3);
for x = xini:2:size(testdata,1)-(fblocksize(1)+1) 
    xrange = x-fblocksize(1):x+fblocksize(1);
    for y = yini:2:size(testdata,2)-(fblocksize(2)+1)
        yrange = y-fblocksize(2):y+fblocksize(2);
            XYcoord = [XYcoord; [x, y,3]];
            tmp = testdata(xrange,yrange,zrange);
            mtest_feats = [mtest_feats tmp(:)];
    end
end

mtest_feats = mtrain_feats;
%%% run classification code %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
training_feats = mtrain_feats;
H_train = mH_train;
testing_feats = mtest_feats;
H_test = H_train;
zeros(2,size(testing_feats,2));
H_test = zeros(2,size(testing_feats,2));
%% parameters for dictionary learning.
sparsitythres = 5; % sparsity prior
dictsize = 1350; % dictionary size
sqrt_alpha = 25; % weights for label constraint term
sqrt_beta = 64; % weights for classification err term
iterations = 30; % iteration number
iterations4ini = 20; % iteration number for initialization
savedir = [pwd]

%% dictionary learning process
% get initial dictionary Dinit and Winit
fprintf('\nLC-KSVD initialization... ');
[Dinit,Tinit,Winit,Q_train] = initialization4LCKSVD(training_feats,H_train,dictsize,iterations4ini,sparsitythres);
fprintf('done!');

% run LC k-svd training (reconstruction err + class penalty + classifier err)
fprintf('\nDictionary and classifier learning by LC-KSVD2...')
[D2,X2,T2,W2] = labelconsistentksvd2(training_feats,Dinit,Q_train,Tinit,H_train,Winit,iterations,sparsitythres,sqrt_alpha,sqrt_beta);
save([dir,'dictionarydata2.mat'],'D2','X2','W2','T2');
fprintf('done!');

%% classification process
%% classification process
[prediction2,accuracy2] = classification(D2, W2, testing_feats, H_test, sparsitythres);
fprintf('\nFinal recognition rate for LC-KSVD2 is : %.03f ', accuracy2);

close all
%%%% result image:
% result_image = zeros(size(testdata(:,:)));
% matlabpool open 8
result_image = [];
for  i = 1:size(XYcoord,1)
    x = XYcoord(i,1); 
    y = XYcoord(i,2);
    if(prediction2(i)==1)
        tmp = 1;
    elseif(prediction2(i)==2) 
        tmp = 0;
    else
        tmp = 2;
    end
    result_image(x,y) = tmp;
end
matlabpool close

% % % 
% % % figure;imshow(result_image,[]);title('Result Image');
% % % figure;imshow(testdata(:,:,3),[]);title('Test Image'); 
% % % figure;imshow(imdata(:,:,3),[]);title('Training Image');
% % % figure;imshow(labdata(:,:,3));title('Training Label Image');
