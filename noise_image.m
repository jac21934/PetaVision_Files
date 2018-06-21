addpath('/home/jcarroll/OpenPV/mlab/util/')
pkg load image


#BaseString = '/home/jcarroll/runs/CIFAR_noise/nf128/VThresh_0.01/output/';

arg_list = argv();

if(isempty(arg_list))
	printf("\nPlease pass me the path to the output directory.\n\n");
	return;
endif

BaseString=arg_list{1};

InputString = [ BaseString "Input.pvp"];
ReconString = [ BaseString "InputRecon.pvp"];
NoiseString = [ BaseString "Noise.pvp"];
ErrorString = [ BaseString "InputError.pvp"];
		
[inputData, inputHdr] = readpvpfile(InputString);
[reconData, reconHdr] = readpvpfile(ReconString);
[noiseData, noiseHdr] = readpvpfile(NoiseString);
[errorData, errorHdr] = readpvpfile(ErrorString);

pictureIndex=100;

# Original Picture
InputMax = max(inputData{pictureIndex}.values(:));
InputMin = min(inputData{pictureIndex}.values(:));
InputImage = (inputData{pictureIndex}.values .- InputMin);
InputImage = InputImage./max(InputImage(:));
InputImage = (permute(InputImage, [2,1,3]));
# Noisy Picture
NoiseMax = max(noiseData{pictureIndex}.values(:));
NoiseMin = min(noiseData{pictureIndex}.values(:));
NoiseImage = (noiseData{pictureIndex}.values .- NoiseMin);
NoiseImage = NoiseImage./max(NoiseImage(:));
NoiseImage = (permute(NoiseImage, [2,1,3]));
# Input Error Picture
ErrorMax = max(errorData{pictureIndex}.values(:));
ErrorMin = min(errorData{pictureIndex}.values(:));
ErrorImage = (errorData{pictureIndex}.values .- ErrorMin);
ErrorImage = ErrorImage./max(ErrorImage(:));
ErrorImage = (permute(ErrorImage, [2,1,3]));


# Recostruction
ReconMax = max(reconData{pictureIndex}.values(:));
ReconMin = min(reconData{pictureIndex}.values(:));
ReconImage = (reconData{pictureIndex}.values .- ReconMin);
ReconImage = ReconImage./max(ReconImage(:));
ReconImage = (permute(ReconImage, [2,1,3]));

concatenated = imresize(horzcat(InputImage,NoiseImage,ErrorImage, ReconImage), 4);

figure;
imwrite(concatenated, "test.png")



							 
