function compiledFeatures = textureAnalysisSerial(bwLung, original, depth)

	patchSize = 37;
	patchMid = round(patchSize/2);
    
    lungs = bwLung==0;
    lungs = lungs + (double(bwLung) .* (original));   		% textured lungs on white background
	%finalMask = zeros(size(lungs,1), size(lungs,2));
    
    x1 = find(sum(bwLung,1), 1, 'first');
    x2 = find(sum(bwLung,1), 1, 'last');
    y1 = find(sum(bwLung,2), 1, 'first');
    y2 = find(sum(bwLung,2), 1, 'last');
    
    lungs = lungs(y1-1:y2+1, x1-1:x2+1);					% cropped lungs

    jEnd = size(lungs,2)-patchMid+1;
    iEnd = size(lungs,2)-patchMid+1;
    
    %imshow(lungs)
    allPatches = im2col(lungs, [patchSize patchSize], 'sliding');
    allPatches(allPatches==1) = NaN;
	
	
	[r c] = find(isnan(allPatches));
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
    
    g1 = 'name';
    g2 = 'values';
    g3 = 'binEdges';
	
	if (patchesEnd>0)
		compiledFeatures(1, patchesEnd) = struct(f1, zeros(patchSize), f2, NaN, f3, NaN, f4, NaN, f5, NaN);
		
		%statistics = zeros(1, 5);
		
		for i=1:size(patches,2)
			reshapedPatch = reshape(patches(:,i),patchSize,patchSize);
            textureHistos(1,7) = struct(g1, ' ', g2, NaN, g3, NaN);
            %reshapedPatchGPU = reshape(patches(:,i),patchSize,patchSize);
			%glcm = graycomatrix(uint8(reshapedPatchGPU*255), 'NumLevels', 16);
			%stats = graycoprops(glcm, {'Contrast', 'Correlation', 'Energy', 'Homogeneity'});
			%statistics(:) = [stats.Energy, stats.Contrast, stats.Correlation, stats.Homogeneity, entropy(uint8(reshapedPatch*255))];
            textureHistos(1).name = 'rpHist';
			[textureHistos(1).values textureHistos(1).binEdges] = histcounts(reshapedPatch);
            textureHistos(2).name = 'rpGausHist';
			[textureHistos(2).values textureHistos(2).binEdges] = histcounts(imgaussfilt(reshapedPatch));
            textureHistos(3).name = 'rpLaplaceHist';
            [textureHistos(3).values texureHistos(3).binEdges] = histcounts(imfilter(reshapedPatch, fspecial('laplacian', 0.8)));
			
            [rpFirstDerX rpFirstDerY] = gradient(reshapedPatch);
			[rpSecDerX rpDerYX] = gradient(rpFirstDerX);
			[rpDerXY rpSecDerY] = gradient(rpFirstDerY);
            
            textureHistos(4).name = 'rpFirstXHist';
            [textureHistos(4).values texureHistos(4).binEdges] = histogram(rpFirstDerX,256);
            textureHistos(5).name = 'rpFirstYHist';
            [textureHistos(5).values texureHistos(5).binEdges] = histogram(rpFirstDerY,256);
            textureHistos(6).name = 'rpSecXHist';
            [textureHistos(6).values texureHistos(6).binEdges] = histogram(rpSecDerX,256);
            textureHistos(7).name = 'rpSecYHist';
            [textureHistos(7).values texureHistos(7).binEdges] = histogram(rpSecDerY,256);
            %filteredTextureHist = cat(1, rpHist, rpGausHist, rpLaplaceHist, rpFirstXHist, rpFirstYHist, rpSecXHist, rpSecYHist);
            %filteredTextureHist = {rpHist, rpGausHist, rpLaplaceHist, rpFirstXHist, rpFirstYHist, rpSecXHist, rpSecYHist};
            
            stats = zeros(4,size(textureHistos,1))
            
            for j=1:4           %iterating through stats: mean, std2, skew, kurtosis, entropy
                for k=1:size(textureHistos,1)
                    switch j
                        case 1
                            meanBin = zeros(1,256);
                            for m=1:256
                                meanBin(m) = mean(textureHistos(k).values(m)*(textureHistos(k).binEdges(m+1)));
                            end
                            stats(j, k) = mean(meanBin);
                        case 2
                            stdBin = zeros(1,256);
                            for m=1:256
                                stdBin(m) = std2(textureHistos(k).values(m)*(textureHistos(k).binEdges(m+1)));
                            end
                            stats(j, k) = std2(stdBin);
                        case 3
                            skewBin = zeros(1,256);
                            for m=1:256
                                skewBin(m) = (((textureHistos(k).values(m)*(textureHistos(k).binEdges(m+1)))-stats(1,k))/stats(2,k))^3;
                            end
                            stats(j, k) = sum(skewBin)/patchSize^2;
                        case 4
                            kurtosisBin = zeros(1,256);
                            for m=1:256
                                kurtosisBin(m) = (((textureHistos(k).values(m)*(textureHistos(k).binEdges(m+1)))-stats(1,k))/stats(2,k))^4 - 3;
                            end
                            stats(j, k) = sum(kurtosisBin)/patchSize^2;
                        %case 5
                            %stats(i, k) = 'entropy';
                    end
                end
            end

			%[U S V] = svd(reshapedPatch);

			%[rl(1),rl(2),rl(3),rl(4),rl(5),rl(6),rl(7)] = glrlm(reshapedPatch,16,ones(patchSize));
            %[a b c d e f g] = glrlm(reshapedPatchGPU,16,ones(patchSize));
            %[a b c d e f g] = [ 0 0 0 0 0 0 0 ];
			%textures{i} = patch;
			compiledFeatures(i) = struct(f1, reshapedPatch, f2, depth, f3, [0 0 0], f4, NaN, f5, 0);
			%features(i,:) = cat(2, statistics(1), statistics(2), statistics(3), statistics(4), statistics(5), rl(1), rl(2), rl(3), rl(4), rl(5), rl(6), rl(7));
			%features(i,:) = cat(2, statistics(1), statistics(2), statistics(3), statistics(4), statistics(5));
		end
	else
		compiledFeatures(1) = struct(f1, zeros(patchSize), f2, NaN, f3, NaN, f4, NaN, f5, NaN);
	end

end