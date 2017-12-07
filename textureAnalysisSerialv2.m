function compiledFeatures = textureAnalysisSerialv2(bwLung, original, depth)

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
	f3 = 'stats';
	f4 = 'svd';
	f5 = 'corrCoef';
	f6 = 'class';
	
	if (patchesEnd>0)
		compiledFeatures(1, patchesEnd) = struct(f1, zeros(patchSize), f2, NaN, f3, NaN, f4, NaN, f5, NaN);
		

		
		for i=1:size(patches,2)
			reshapedPatch = reshape(patches(:,i),patchSize,patchSize);
            rpGaus = imgaussfilt(reshapedPatch);
            rpLaplace = imfilter(reshapedPatch, fspecial('laplacian',0.8));
            [rpFirstDerX rpFirstDerY] = gradient(reshapedPatch);
            [rpSecDerX rpDerYX] = gradient(rpFirstDerX);
            [rpDerXY rpSecDerY] = gradient(rpFirstDerY);
            
            filteredPatches = cat(3, reshapedPatch, rpGaus, rpLaplace, rpFirstDerX, rpFirstDerY, rpSecDerX, rpSecDerY);
            stats = zeros(4,size(filteredPatches,3));

            for j=1:7
                stats(1,j) = mean(mean(filteredPatches(:,:,j)));
                stats(2,j) = std2(filteredPatches(:,:,j));
                stats(3,j) = sum(sum(((filteredPatches(:,:,j)-stats(1,j))/stats(2,j)).^3))/patchSize^2;
                stats(4,j) = (sum(sum(((filteredPatches(:,:,j)-stats(1,j))/stats(2,j)).^4))/patchSize^2) - 3;
            end

			[U S V] = svd(reshapedPatch);
            R = corrcoef(reshapedPatch);

			%[rl(1),rl(2),rl(3),rl(4),rl(5),rl(6),rl(7)] = glrlm(reshapedPatch,16,ones(patchSize));
            %[a b c d e f g] = glrlm(reshapedPatchGPU,16,ones(patchSize));
            %[a b c d e f g] = [ 0 0 0 0 0 0 0 ];
			%textures{i} = patch;
			compiledFeatures(i) = struct(f1, reshapedPatch, f2, depth, f3, stats, f4, [U S V], f5, R);
			%features(i,:) = cat(2, statistics(1), statistics(2), statistics(3), statistics(4), statistics(5), rl(1), rl(2), rl(3), rl(4), rl(5), rl(6), rl(7));
			%features(i,:) = cat(2, statistics(1), statistics(2), statistics(3), statistics(4), statistics(5));
		end
	else
		compiledFeatures(1) = struct(f1, zeros(patchSize), f2, NaN, f3, NaN, f4, NaN, f5, NaN);
	end

end