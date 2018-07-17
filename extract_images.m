pkg load image
addpath('/projects/pcsri/PetaVision/OpenPV/mlab/util/')

arg_list = argv ();
if nargin != 3
    printf("Usage: %s <output dir> <image name> <num images>\n", program_name());
    exit(0);
endif

outputDir = arg_list{1};
imgDir   = arg_list{2};
numImages = str2num(arg_list{3});

%%Input        = readpvpfile([outputDir '/Input.pvp']);
%%NoiseLayer   = readpvpfile([outputDir '/NoiseLayer.pvp']);
InputRecon   = readpvpfile([outputDir '/InputRecon_V.pvp']);

function img = getImg(file, index)
    img = permute(file{index}.values, [2, 1, 3]);
    minVal = min(img(:));
    maxVal = max(img(:));
    img = img .- minVal;
    if maxVal - minVal > 1
        img = img ./ (maxVal - minVal);
    endif
end

for i = [1:numImages]
    %%    InputImg = getImg(Input, i);
    %%imwrite(InputImg, [imgDir "/" num2str(i) "/Input.png"]);

    %%    NoiseLayerImg = getImg(NoiseLayer, i);
    %%imwrite(NoiseLayerImg, [imgDir "/" num2str(i) "/NoiseLayer.png"]);
     
    InputReconImg = getImg(InputRecon, i);
    imwrite(InputReconImg, [imgDir "/" num2str(i) "/InputRecon.png"]);

    %%DenoiseErrorImg = abs(InputImg - InputReconImg);
    %%imwrite(DenoiseErrorImg, [imgDir "/" num2str(i) "/DenoiseError.png"]);
endfor
