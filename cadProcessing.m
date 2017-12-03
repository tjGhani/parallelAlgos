function arrSlice =  cadProcessing(workingDir, first, last)
  
	tic

  	imagesDir = 'images';

  	imageNames = dir(fullfile(workingDir, imagesDir, '*.dcm'));
  	imageNames = {imageNames.name};

	f1 = 'images';
	f2 = 'detected';
	f3 = 'metrics';
	f4 = 'textureFeatures';
	f5 = 'depth';
    f6 = 'rank';

    img = dicomread(fullfile(workingDir, imagesDir, imageNames{1}));
    sizeofscan = last-first;

    for i = first:last

        original = dicomread(fullfile(workingDir, imagesDir, imageNames{i}));
		[original lungWindow mediastinal] = preprocessing(original);

		depth = i/length(imageNames);
        [medSegment bwLungMed medLungsOnBlack] = detectWindPipe(mediastinal);

		originalOnWhite = double(bwLungMed==0) + lungWindow;
		originalOnBlack = double(bwLungMed) .* lungWindow;

		[ratio lung abnormality] = findArea(bwLungMed, medLungsOnBlack);
		
        [detectedBH blackHole blackBoundary blackArea] = detectBlackHoles(lungWindow, originalOnWhite);
		[detectedWM whiteMass whiteBoundary whiteArea] = whiteMassDetection(lungWindow, originalOnBlack);
        %textureFeatures = textureAnalysis(bwLungMed, lungWindow, depth);
        %textureFeatures = textureAnalysisGPU(bwLungMed, lungWindow, depth);
        textureFeatures = textureAnalysisSerial(bwLungMed, lungWindow, depth);

		imagesCat = cat(3, lungWindow, mediastinal);
        %index = i - first + 1;
        %tempSlice = struct(f1, imagesCat, f2, [detectedBH blackArea/lung detectedWM whiteArea/lung], f3, [ratio lung abnormality], f4, textureFeatures, f5, depth, f6, i-first+1);
		arrSlice{i} = struct(f1, imagesCat, f2, [detectedBH blackArea/lung detectedWM whiteArea/lung], f3, [ratio lung abnormality], f4, textureFeatures, f5, depth, f6, i-first+1);

    end

    toc

end
