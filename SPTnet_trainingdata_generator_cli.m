% CLI version of SPTnet_trainingdata_generator

% Add PSF-toolbox path
addpath(genpath(fullfile(pwd, 'PSF-toolbox')));

% Try to initialize DIPimage
try
    dip_initialise;
catch
    warning('dip_initialise failed or not found. Make sure DIPimage is installed and in your MATLAB path.');
end

if exist('newim', 'file') ~= 2
    error('DIPimage is not available. Please install DIPimage and add it to your MATLAB path on CSD3.');
end

% Set Simulation Parameters
simParams.Num_file = 1000;           % Number of files to generate
simParams.Videos_per_file = 100;    % Number of videos per file
simParams.Frames = 30;              % Frames per video
simParams.Image_dims = 64;          % Dimensions
simParams.p_num_min = 1;
simParams.p_num_max = 10;
simParams.Hurst_min = 0.0001;
simParams.Hurst_max = 0.9999;
simParams.D_min = 0.001;
simParams.D_max = 0.5;

% Set PSF Parameters
psfParams.NA = 1.49;
psfParams.Lambda = 0.69;
psfParams.RefractiveIndex = 1.518;
psfParams.OTFscale_SigmaX = 0.95;
psfParams.OTFscale_SigmaY = 0.95;
psfParams.Pixelsize = 0.157;
psfParams.PSFsize = 128;
psfParams.nMed = 1.33;
psfParams.Photon_min = 300;
psfParams.Photon_max = 5000;
psfParams.Bg_min = 1;
psfParams.Bg_max = 25;
psfParams.Perlin_Bg_min = 0;
psfParams.Perlin_Bg_max = 10;

enableMotionBlur = false;

% Zernike parameters (default piston=1)
zernikeCoefficients = zeros(1, 25);
magnitudeCoefficients = zeros(1, 25);
magnitudeCoefficients(1) = 1;

% Output folder
folder = fullfile(pwd, 'TestData', 'training_data');
if ~exist(folder, 'dir')
    mkdir(folder);
end

fprintf('Starting simulation of %d files, %d videos each...\n', simParams.Num_file, simParams.Videos_per_file);
fprintf('Saving to %s\n', folder);

total_files = simParams.Num_file; 
numvideos = simParams.Videos_per_file;

imageSize = 128;          
PRstruct.Zernike_phase = zernikeCoefficients;
PRstruct.Zernike_mag = magnitudeCoefficients;
PRstruct.NA = psfParams.NA;           
PRstruct.Lambda = psfParams.Lambda;          
PRstruct.RefractiveIndex = psfParams.RefractiveIndex;     
PRstruct.Pupil.phase = zeros(imageSize, imageSize);
PRstruct.Pupil.mag = zeros(imageSize, imageSize);
PRstruct.SigmaX = psfParams.OTFscale_SigmaX;       
PRstruct.SigmaY = psfParams.OTFscale_SigmaY;       

psfobj = PSF_zernike(PRstruct);
psfobj.Boxsize = simParams.Image_dims;                     
psfobj.Pixelsize = psfParams.Pixelsize;                
psfobj.PSFsize = psfParams.PSFsize;              
psfobj.nMed = psfParams.nMed;                      

for fileindex = 1:total_files
    fprintf('Generating file %d of %d...\n', fileindex, total_files);
    timelapsedata = single(zeros(simParams.Image_dims, simParams.Image_dims, simParams.Frames, numvideos));
    bglabel = zeros(1, numvideos, 'single');
    Perlinbglabel = zeros(1, numvideos, 'single');
    traceposition = cell(numvideos, simParams.p_num_max); 
    Hlabel = cell(numvideos, simParams.p_num_max);
    Clabel = cell(numvideos, simParams.p_num_max);
    photonlabel = cell(numvideos, simParams.p_num_max);
    moleculeid = cell(numvideos, simParams.p_num_max);
    duration = cell(numvideos,simParams.p_num_max);
    oversampling = 10;
    
    for datasetIndex = 1:numvideos
        bgLevel = unifrnd(psfParams.Bg_min, psfParams.Bg_max);
        if psfParams.Perlin_Bg_min<1
            Perlin_bg = 0;
        else
            Perlin_bg = unifrnd(psfParams.Perlin_Bg_min, psfParams.Perlin_Bg_max);
        end
        if enableMotionBlur
            psf_all = single(zeros(simParams.Image_dims, simParams.Image_dims, simParams.Frames*oversampling));
        else
            psf_all = single(zeros(simParams.Image_dims, simParams.Image_dims, simParams.Frames));
        end
        numParticles = randi([simParams.p_num_min, simParams.p_num_max]); 
        moleculeCounter = 0;
        
        T = simParams.Frames;
        Nmin = simParams.p_num_min;  
        Nmax = simParams.p_num_max;
        Lmin = 2;        
        Lmax = T;   
        tracks = uniform_duration_and_present(T, numParticles, Lmin, Lmax);
        
        for particleIndex = 1:numParticles
            start_t = tracks(particleIndex).t0;
            duration{datasetIndex,particleIndex} = tracks(particleIndex).t1 - tracks(particleIndex).t0 + 1;
            
            hurstExponent = unifrnd(simParams.Hurst_min, simParams.Hurst_max);
            diffusionCoefficient = unifrnd(simParams.D_min, simParams.D_max);
            
            if enableMotionBlur
                osduration{datasetIndex,particleIndex} = duration{datasetIndex,particleIndex}*oversampling;
                osstart_t = 1+(start_t-1)*oversampling;
                [trajectoryX, trajectoryY] = fractional_brownian_motion_generator_2D(hurstExponent, osduration{datasetIndex,particleIndex}, diffusionCoefficient);
                FBMx = (1/oversampling)^(hurstExponent)*trajectoryX;
                FBMy = (1/oversampling)^(hurstExponent)*trajectoryY;
                photons = unifrnd(psfParams.Photon_min,psfParams.Photon_max);
                xOffset = unifrnd(-(simParams.Image_dims/2)+4, (simParams.Image_dims/2)-4); 
                yOffset = unifrnd(-(simParams.Image_dims/2)+4, (simParams.Image_dims/2)-4); 
                overstep = simParams.Frames*oversampling;
                psfobj.Zpos = zeros(overstep,1);
                psfobj.Xpos = nan(overstep,1);
                psfobj.Ypos = nan(overstep,1);
                psfobj.Xpos(osstart_t:(osstart_t+osduration{datasetIndex,particleIndex} -1)) = FBMx + xOffset;   
                psfobj.Ypos(osstart_t:(osstart_t+osduration{datasetIndex,particleIndex} -1)) = FBMy + yOffset;   

                traceposition{datasetIndex,particleIndex} = single([mean(reshape(psfobj.Xpos,[oversampling,simParams.Frames]))', mean(reshape(psfobj.Ypos,[oversampling,simParams.Frames]))']);
                psfobj.precomputeParam();               
                psfobj.genPupil();                      
                psfobj.genPSF();                        
                norm_parameter = sum(sum(psfobj.Pupil.mag));
                psfobj.scalePSF('normal');              
                psf = psfobj.ScaledPSFs;    
                psf_blur = psf./norm_parameter.*(photons/oversampling);
                psf_blur(isnan(psf_blur)) = 0;
                psf_all(:,:,:) = psf_all(:,:,:) + psf_blur(:,:,:);

                temp = reshape(psf_all, simParams.Image_dims, simParams.Image_dims, oversampling, simParams.Frames);
                blurpsf = squeeze(sum(temp, 3));
                Hlabel{datasetIndex, particleIndex} = single(hurstExponent);
                Clabel{datasetIndex, particleIndex} = single(diffusionCoefficient);
                photonlabel{datasetIndex, particleIndex} = single(photons);
                moleculeCounter = moleculeCounter + 1;
                moleculeid{datasetIndex, particleIndex} = moleculeCounter;
            else 
                [trajectoryX, trajectoryY] = fractional_brownian_motion_generator_2D(hurstExponent, duration{datasetIndex,particleIndex}, diffusionCoefficient);
                xOffset = unifrnd(-(simParams.Image_dims/2)+4, (simParams.Image_dims/2)-4); 
                yOffset = unifrnd(-(simParams.Image_dims/2)+4, (simParams.Image_dims/2)-4); 
                psfobj.Xpos = nan(simParams.Frames,1);
                psfobj.Ypos = nan(simParams.Frames,1);
                psfobj.Zpos = zeros(simParams.Frames,1); 
                psfobj.Xpos(start_t:(start_t+duration{datasetIndex,particleIndex} -1)) = trajectoryX + xOffset; 
                psfobj.Ypos(start_t:(start_t+duration{datasetIndex,particleIndex} -1)) = trajectoryY + yOffset; 
                photons = unifrnd(psfParams.Photon_min,psfParams.Photon_max); 

                traceposition{datasetIndex, particleIndex} = single([psfobj.Xpos, psfobj.Ypos]);

                psfobj.precomputeParam();
                psfobj.genPupil();
                psfobj.genPSF();
                psfobj.scalePSF('normal');
                norm_parameter = sum(sum(psfobj.Pupil.mag));
                psfobj.scalePSF('normal');              
                psf = psfobj.ScaledPSFs;    
                psf = psf./norm_parameter.*photons;
                psf(isnan(psf)) = 0;
                psf_all(:,:,:) = psf_all(:,:,:) + psf(:,:,:);
                
                Hlabel{datasetIndex, particleIndex} = single(hurstExponent);
                Clabel{datasetIndex, particleIndex} = single(diffusionCoefficient);
                photonlabel{datasetIndex, particleIndex} = single(photons);
                moleculeCounter = moleculeCounter + 1;
                moleculeid{datasetIndex, particleIndex} = moleculeCounter;
            end
        end
        bglabel(datasetIndex) = single(bgLevel);
        Perlinbglabel(datasetIndex) = single(Perlin_bg);
        if enableMotionBlur
            timelapsedata(:,:,:,datasetIndex) = single(noise(blurpsf(:,:,:)+ bgLevel + Perlin_bg * perlin_noise(simParams.Image_dims),'poisson'));
        else
            timelapsedata(:,:,:,datasetIndex) = single(noise(psf_all + bgLevel + Perlin_bg * perlin_noise(simParams.Image_dims), 'poisson'));
        end
    end
    % Windows path fix ('\' to '/')
    saveFileName = fullfile(folder, sprintf('trainingvideos_%d.mat', fileindex));
    save(saveFileName, 'timelapsedata', 'Hlabel', 'Clabel', 'photonlabel', 'bglabel', 'traceposition', 'moleculeid', 'Perlinbglabel','duration', '-v7.3');
end

fprintf('Simulation Completed!\n');

%% Functions (Copied from original script)
function [x,y] = fractional_brownian_motion_generator_2D(H,steps,C)
    covariancematrix = zeros(steps,steps);
    for t = 1:steps
        for s = 1:steps
            covariancematrix(t,s) = 0.5*((t^(2*H)+s^(2*H)-abs(t-s)^(2*H)));
        end
    end
    [L] = chol(covariancematrix,'lower'); 
    X = sqrt(2*C)*randn(steps,1);
    Y = sqrt(2*C)*randn(steps,1);
    x = L*X;
    y = L*Y;
end

function tracks = uniform_duration_and_present(T, numParticles, Lmin, Lmax)
    occ = zeros(1, T);
    tracks = struct('t0', cell(1,numParticles), 't1', cell(1,numParticles), 'L', cell(1,numParticles));
    if numParticles == 0
        return;
    end
    Ls = randi([Lmin Lmax], 1, numParticles);
    Ls(Ls > T) = T;
    [Ls, order] = sort(Ls, 'descend');

    for k = 1:numParticles
        L = Ls(k);
        bestStarts = [];
        bestScore = inf;
        
        for t0 = 1:(T - L + 1)
            t1 = t0 + L - 1;
            occ2 = occ;
            occ2(t0:t1) = occ2(t0:t1) + 1;
            score = sum((occ2 - mean(occ2)).^2);     
            if score < bestScore - 1e-12
                bestScore = score;
                bestStarts = t0;
            elseif abs(score - bestScore) < 1e-12
                bestStarts(end+1) = t0;
            end
        end
        t0 = bestStarts(randi(numel(bestStarts)));
        t1 = t0 + L - 1;
        occ(t0:t1) = occ(t0:t1) + 1;

        tracks(k).t0 = t0;
        tracks(k).t1 = t1;
        tracks(k).L  = L;
    end
    tmp = tracks;
    tracks = struct('t0',{},'t1',{},'L',{});
    tracks(order) = tmp; 
end
