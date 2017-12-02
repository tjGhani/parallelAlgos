function compiledFeatures = textureAnalysisGPU(bwLung, original, depth)

	patchSize = 37;
	patchMid = round(patchSize/2);
    
    lungs = bwLung==0;
    lungs = lungs + (double(bwLung) .* (original));   		% textured lungs on white background
    lungGPU = gpuArray(lungs);
    bwLungGPU = gpuArray(bwLung);
	%finalMask = zeros(size(lungs,1), size(lungs,2));
    
    x1 = find(sum(bwLungGPU,1), 1, 'first');
    x2 = find(sum(bwLungGPU,1), 1, 'last');
    y1 = find(sum(bwLungGPU,2), 1, 'first');
    y2 = find(sum(bwLungGPU,2), 1, 'last');
    
    lungsGPU = lungsGPU(y1-1:y2+1, x1-1:x2+1);					% cropped lungs
    %imshow(lungs)
    allPatches = im2col(lungs, [patchSize patchSize], 'sliding');
    allPatches(allPatches==1) = NaN;
	
	coordinates = zeros(size(allPatches,2),2);
	m=1;
	for j=patchMid:size(lungs,2)-patchMid+1
		for i=patchMid:size(lungs,1)-patchMid+1
			coordinates(m,:) = [i j];
			m=m+1;
		end
	end
	
	[r c] = find(isnan(allPatches));
	patches = allPatches;
	patches(:,c) = [];										% remove patches with non-lung white parts
	%size(patches,2)
	coordinates(c,:) = [];
	%size(coordinates,1)

	f1 = 'image';
	f2 = 'depth';
	f3 = 'glcmStats';
	f4 = 'dct';
	f5 = 'runlength';
	f6 = 'class';
	
	if (size(patches,2)>0)
		compiledFeatures(1, size(patches,2)) = struct(f1, zeros(patchSize), f2, NaN, f3, NaN, f4, NaN, f5, NaN);
		
		%statistics = zeros(1, 5);
		
		for i=1:size(patches,2)
			reshapedPatch = reshape(patches(:,i),patchSize,patchSize);

			glcm = graycomatrix(uint8(reshapedPatch*255), 'NumLevels', 16);
			stats = graycoprops(glcm, {'Contrast', 'Correlation', 'Energy', 'Homogeneity'});
			%statistics(:) = [stats.Energy, stats.Contrast, stats.Correlation, stats.Homogeneity, entropy(uint8(reshapedPatch*255))];

			%[rl(1),rl(2),rl(3),rl(4),rl(5),rl(6),rl(7)] = glrlm(reshapedPatch,16,ones(patchSize));
            [a b c d e f g] = glrlm(reshapedPatch,16,ones(patchSize));
			%textures{i} = patch;
			compiledFeatures(i) = struct(f1, reshapedPatch, f2, depth, f3, [stats.Energy, stats.Contrast, stats.Correlation, stats.Homogeneity, entropy(uint8(reshapedPatch*255))], f4, NaN, f5, [a b c d e f g]);
			%features(i,:) = cat(2, statistics(1), statistics(2), statistics(3), statistics(4), statistics(5), rl(1), rl(2), rl(3), rl(4), rl(5), rl(6), rl(7));
			%features(i,:) = cat(2, statistics(1), statistics(2), statistics(3), statistics(4), statistics(5));
		end
	else
		compiledFeatures(1) = struct(f1, zeros(patchSize), f2, NaN, f3, NaN, f4, NaN, f5, NaN);
	end

end