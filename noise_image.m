addpath('/home/jacob/PetaVision_Data/PetaVision_Files/util')
pkg load image



arg_list = argv();

if(isempty(arg_list))
	printf("\nPlease pass me the path to the output directory.\n\n");
	return;
endif

FileName=arg_list{1};

		
[noiseData, noiseHdr] = readpvpfile(FileName);

pictureIndex=1;

# Noisy Picture
NoiseMax = max(noiseData{pictureIndex}.values(:));
NoiseMin = min(noiseData{pictureIndex}.values(:));
NoiseImage = (noiseData{pictureIndex}.values .- NoiseMin);
NoiseImage = NoiseImage./max(NoiseImage(:));
NoiseImage = (permute(NoiseImage, [2,1,3]));

concatenated = imresize(horzcat(NoiseImage), 1);

figure;
imwrite(concatenated, "test2.png")

