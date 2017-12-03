function compiledFeatures = textureAnalysisGPU(bwLung, original, depth)

	patchSize = 37;
	patchMid = round(patchSize/2);
    
    lungs = bwLung==0;
    lungs = lungs + (double(bwLung) .* (original));   		% textured lungs on white background
    %lungGPU = gpuArray(lungs);
    bwLungGPU = gpuArray(bwLung);
	%finalMask = zeros(size(lungs,1), size(lungs,2));
    
    x1 = find(sum(bwLungGPU,1), 1, 'first');
    x2 = find(sum(bwLungGPU,1), 1, 'last');
    y1 = find(sum(bwLungGPU,2), 1, 'first');
    y2 = find(sum(bwLungGPU,2), 1, 'last');
    bwLungs = gather(bwLungGPU);
    clear('bwLungs');
    
    lungs = lungs(y1-1:y2+1, x1-1:x2+1);					% cropped lungs
    %lungs = lungsGPU;
    jEnd = size(lungs,2)-patchMid+1;
    iEnd = size(lungs,2)-patchMid+1;
    
    %imshow(lungs)
    allPatches = im2col(lungs, [patchSize patchSize], 'sliding');
    allPatches(allPatches==1) = NaN;
    %allPatchesGPU = gpuArray(allPatches);
	
	
	[r c] = find(isnan(allPatches));
	%patches = gather(allPatchesGPU);
    %clear('allPatchesGPU');
    patches = allPatches;
	patches(:,c) = [];										% remove patches with non-lung white parts
	%size(patches,2)
	%coordinates(c,:) = [];
	%size(coordinates,1)
    patchesEnd = size(patches,2);

	f1 = 'image';
	f2 = 'depth';
	f3 = 'svd';
	f4 = 'dct';
	f5 = 'runlength';
	f6 = 'class';
	
	if (patchesEnd>0)
		compiledFeatures(1, patchesEnd) = struct(f1, zeros(patchSize), f2, NaN, f3, NaN, f4, NaN, f5, NaN);
		
		%statistics = zeros(1, 5);
		
		for i=1:size(patches,2)
			reshapedPatchGPU = gpuArray(reshape(patches(:,i),patchSize,patchSize));
            %reshapedPatchGPU = reshape(patches(:,i),patchSize,patchSize);
			%glcm = graycomatrix(uint8(reshapedPatchGPU*255), 'NumLevels', 16);
			%stats = graycoprops(glcm, {'Contrast', 'Correlation', 'Energy', 'Homogeneity'});
			%statistics(:) = [stats.Energy, stats.Contrast, stats.Correlation, stats.Homogeneity, entropy(uint8(reshapedPatch*255))];
			rpHist = histogram(reshapedPatchGPU,356);
			rpGausHist = histogram(imgaussfilt(reshapedPatchGPU),256);
			rpLaplaceHist = histogram(imfilter(reshapedPatchGPU, fspecial('laplacian', 0.8)),256);
			[rpFirstDerX rpFirstDerY] = gradient(reshapedPatchGPU);
			[rpSecDerX rpDerYX] = gradient(rpFirstDerX);
			[rpDerXY rpSecDerY] = gradient(rpFirstDerY);
            rpFirstXHist = histogram(rpFirstDerX,256);
            rpFirstYHist = histogram(rpFirstDerY,256);
            rpSecXHist = histogram(rpSecDerX,256);
            rpSecYHist = histogram(rpSecDerY,256);
            filteredTextureHist = cat(1, rpHist, rpGausHist, rpLaplaceHist, rpFirstXHist, rpFirstYHist, rpSecXHist, rpSecYHist);
            
            stats = zeros(4,size(filteredTextureHist,1));
            
            for j=1:4           %iterating through stats: mean, std2, skew, kurtosis, entropy
                for k=1:size(filteredTextureHist,1)
                    switch j
                        case 1
                            meanBin = zeros(1,256);
                            for m=1:256
                                meanBin(m) = mean(filteredTextureHist(k).Values(m)*(filteredTextureHist(k).BinWidth*m));
                            end
                            stats(j, k) = mean(meanBin);
                        case 2
                            stdBin = zeros(1,256);
                            for m=1:256
                                stdBin(m) = std2(filteredTextureHist(k).Values(m)*(filteredTextureHist(k).BinWidth*m));
                            end
                            stats(j, k) = std2(stdBin);
                        case 3
                            skewBin = zeros(1,256);
                            for m=1:256
                                skewBin(m) = (((filteredTextureHist(k).Values(m)*(filteredTextureHist(k).BinWidth*m))-stats(1,k))/stats(2,k))^3;
                            end
                            stats(j, k) = sum(skewBin)/patchSize^2;
                        case 4
                            kurtosisBin = zeros(1,256);
                            for m=1:256
                                kurtosisBin(m) = (((filteredTextureHist(k).Values(m)*(filteredTextureHist(k).BinWidth*m))-stats(1,k))/stats(2,k))^4 - 3;
                            end
                            stats(j, k) = sum(kurtosisBin)/patchSize^2;
                        %case 5
                            %stats(j, k) = 'entropy';
                    end
                end
            end
			%[U S V] = svd(reshapedPatchGPU);
            
            
            %gather(reshapedPatchGPU);
            %clear('reshapedPatchGPU');

			%[rl(1),rl(2),rl(3),rl(4),rl(5),rl(6),rl(7)] = glrlm(reshapedPatch,16,ones(patchSize));
            %[a b c d e f g] = glrlm(reshapedPatchGPU,16,ones(patchSize));
            %[a b c d e f g] = [ 0 0 0 0 0 0 0 ];
			%textures{i} = patch;
			compiledFeatures(i) = struct(f1, reshapedPatchGPU, f2, depth, f3, [0 0 0], f4, NaN, f5, 0);
			%features(i,:) = cat(2, statistics(1), statistics(2), statistics(3), statistics(4), statistics(5), rl(1), rl(2), rl(3), rl(4), rl(5), rl(6), rl(7));
			%features(i,:) = cat(2, statistics(1), statistics(2), statistics(3), statistics(4), statistics(5));
		end
	else
		compiledFeatures(1) = struct(f1, zeros(patchSize), f2, NaN, f3, NaN, f4, NaN, f5, NaN);
	end

end