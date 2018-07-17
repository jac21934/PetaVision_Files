pkg load image
addpath('/home/jacob/Code/OpenPV/mlab/util/')

		arg_list = argv ();
 

InputRecon   = readpvpfile('./InputRecon_V.pvp');

function img = getImg(file, index)
		img = permute(file{index}.values, [2, 1, 3]);
		minVal = min(img(:));
		maxVal = max(img(:));
		img = img .- minVal;
		if maxVal - minVal > 1
			img = img ./ (maxVal - minVal);
endif
end

for i = [1:100]
	InputReconImg = getImg(InputRecon, i);
	imwrite(InputReconImg, ["./figures/InputRecon_" num2str(i) ".png"]);
endfor
