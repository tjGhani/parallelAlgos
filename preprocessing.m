function [original lungWindow mediastinal] = preprocessing(original)

	minVal = double(min(min(original)));
	maxVal = double(max(max(original)));

	% lung window view parameters
	windowWidth = 1500; 		%D&D: 1500
	windowLevel = -400;			%D&D: -400
	desiredMinVal = windowLevel - (windowWidth/2);
	desiredMaxVal = windowLevel + (windowWidth/2);
	lungWindow = imadjust(original, [(desiredMinVal-(-32768))/65535 (desiredMaxVal-(-32768))/65535], [(minVal+32768)/65535 (maxVal+32768)/65535]);
	lungWindow = mat2gray(lungWindow);

	% mediastinal view parameters
    windowWidth = 380;
    windowLevel = 40;
    desiredMinVal = windowLevel - (windowWidth/2);
	desiredMaxVal = windowLevel + (windowWidth/2);
    mediastinal = imadjust(original, [(desiredMinVal-(-32768))/65535 (desiredMaxVal-(-32768))/65535], [(minVal+32768)/65535 (maxVal+32768)/65535]);
	mediastinal = mat2gray(mediastinal);

	medSegment = adapthisteq(mediastinal);

	original = mat2gray(original);
	
end