function SPTnet_trainingdata_generator_csd3()
% CSD3/headless training data generator for SPTnet.
%
% This version intentionally avoids DIPimage. It uses a local Poisson noise
% helper instead of DIPimage's noise(..., 'poisson') and avoids unifrnd.
%
% Useful environment variables for SLURM jobs:
%   SPT_NUM_FILES         default 10
%   SPT_VIDEOS_PER_FILE  default 100
%   SPT_FRAMES           default 30
%   SPT_IMAGE_DIMS       default 64
%   SPT_OUTPUT_DIR       default ./TestData/training_data
%   SPT_FILE_START       default 1
%   SPT_SEED             default shuffle
%   SPT_MOTION_BLUR      default false

addpath(genpath(fullfile(pwd, 'PSF-toolbox')));

if exist('PSF_zernike', 'file') ~= 2
    error('PSF_zernike is not available. Run from the SPTnet repo root so PSF-toolbox can be added.');
end

seed = getenv('SPT_SEED');
if isempty(seed)
    rng('shuffle');
else
    rng(str2double(seed));
end

% Set Simulation Parameters. Defaults are intentionally smaller than the
% original 1000 x 100 for quick cluster smoke tests.
simParams.Num_file = env_number('SPT_NUM_FILES', 10);
simParams.Videos_per_file = env_number('SPT_VIDEOS_PER_FILE', 100);
simParams.Frames = env_number('SPT_FRAMES', 30);
simParams.Image_dims = env_number('SPT_IMAGE_DIMS', 64);
simParams.p_num_min = env_number('SPT_P_NUM_MIN', 1);
simParams.p_num_max = env_number('SPT_P_NUM_MAX', 10);
simParams.Hurst_min = env_number('SPT_HURST_MIN', 0.0001);
simParams.Hurst_max = env_number('SPT_HURST_MAX', 0.9999);
simParams.D_min = env_number('SPT_D_MIN', 0.001);
simParams.D_max = env_number('SPT_D_MAX', 0.5);

% Set PSF Parameters.
psfParams.NA = env_number('SPT_NA', 1.49);
psfParams.Lambda = env_number('SPT_LAMBDA', 0.69);
psfParams.RefractiveIndex = env_number('SPT_REFRACTIVE_INDEX', 1.518);
psfParams.OTFscale_SigmaX = env_number('SPT_OTF_SIGMA_X', 0.95);
psfParams.OTFscale_SigmaY = env_number('SPT_OTF_SIGMA_Y', 0.95);
psfParams.Pixelsize = env_number('SPT_PIXELSIZE', 0.157);
psfParams.PSFsize = env_number('SPT_PSF_SIZE', 128);
psfParams.nMed = env_number('SPT_N_MED', 1.33);
psfParams.Photon_min = env_number('SPT_PHOTON_MIN', 300);
psfParams.Photon_max = env_number('SPT_PHOTON_MAX', 5000);
psfParams.Bg_min = env_number('SPT_BG_MIN', 1);
psfParams.Bg_max = env_number('SPT_BG_MAX', 25);
psfParams.Perlin_Bg_min = env_number('SPT_PERLIN_BG_MIN', 0);
psfParams.Perlin_Bg_max = env_number('SPT_PERLIN_BG_MAX', 10);

enableMotionBlur = env_bool('SPT_MOTION_BLUR', false);
fileStart = env_number('SPT_FILE_START', 1);

% Zernike parameters (default piston=1).
zernikeCoefficients = zeros(1, 25);
magnitudeCoefficients = zeros(1, 25);
magnitudeCoefficients(1) = 1;

folder = getenv('SPT_OUTPUT_DIR');
if isempty(folder)
    folder = fullfile(pwd, 'TestData', 'training_data');
end
if ~exist(folder, 'dir')
    mkdir(folder);
end

fprintf('SPTnet CSD3 generator starting.\n');
fprintf('DIPimage dependency: disabled.\n');
fprintf('Files: %d, videos/file: %d, frames: %d, image dims: %d.\n', ...
    simParams.Num_file, simParams.Videos_per_file, simParams.Frames, simParams.Image_dims);
fprintf('Saving to %s\n', folder);

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

for fileOffset = 0:(simParams.Num_file - 1)
    fileindex = fileStart + fileOffset;
    fprintf('Generating file %d (%d of %d)...\n', fileindex, fileOffset + 1, simParams.Num_file);
    tic;

    numvideos = simParams.Videos_per_file;
    timelapsedata = single(zeros(simParams.Image_dims, simParams.Image_dims, simParams.Frames, numvideos));
    bglabel = zeros(1, numvideos, 'single');
    Perlinbglabel = zeros(1, numvideos, 'single');
    traceposition = cell(numvideos, simParams.p_num_max);
    Hlabel = cell(numvideos, simParams.p_num_max);
    Clabel = cell(numvideos, simParams.p_num_max);
    photonlabel = cell(numvideos, simParams.p_num_max);
    moleculeid = cell(numvideos, simParams.p_num_max);
    duration = cell(numvideos, simParams.p_num_max);
    oversampling = 10;

    for datasetIndex = 1:numvideos
        bgLevel = rand_range(psfParams.Bg_min, psfParams.Bg_max);
        if psfParams.Perlin_Bg_min < 1
            Perlin_bg = 0;
        else
            Perlin_bg = rand_range(psfParams.Perlin_Bg_min, psfParams.Perlin_Bg_max);
        end

        if enableMotionBlur
            psf_all = single(zeros(simParams.Image_dims, simParams.Image_dims, simParams.Frames * oversampling));
        else
            psf_all = single(zeros(simParams.Image_dims, simParams.Image_dims, simParams.Frames));
        end

        numParticles = randi([simParams.p_num_min, simParams.p_num_max]);
        moleculeCounter = 0;

        T = simParams.Frames;
        Lmin = 2;
        Lmax = T;
        tracks = uniform_duration_and_present(T, numParticles, Lmin, Lmax);

        for particleIndex = 1:numParticles
            start_t = tracks(particleIndex).t0;
            duration{datasetIndex, particleIndex} = tracks(particleIndex).t1 - tracks(particleIndex).t0 + 1;

            hurstExponent = rand_range(simParams.Hurst_min, simParams.Hurst_max);
            diffusionCoefficient = rand_range(simParams.D_min, simParams.D_max);

            if enableMotionBlur
                osduration = duration{datasetIndex, particleIndex} * oversampling;
                osstart_t = 1 + (start_t - 1) * oversampling;
                [trajectoryX, trajectoryY] = fractional_brownian_motion_generator_2D(hurstExponent, osduration, diffusionCoefficient);
                FBMx = (1 / oversampling)^(hurstExponent) * trajectoryX;
                FBMy = (1 / oversampling)^(hurstExponent) * trajectoryY;
                photons = rand_range(psfParams.Photon_min, psfParams.Photon_max);
                xOffset = rand_range(-(simParams.Image_dims / 2) + 4, (simParams.Image_dims / 2) - 4);
                yOffset = rand_range(-(simParams.Image_dims / 2) + 4, (simParams.Image_dims / 2) - 4);
                overstep = simParams.Frames * oversampling;
                psfobj.Zpos = zeros(overstep, 1);
                psfobj.Xpos = nan(overstep, 1);
                psfobj.Ypos = nan(overstep, 1);
                psfobj.Xpos(osstart_t:(osstart_t + osduration - 1)) = FBMx + xOffset;
                psfobj.Ypos(osstart_t:(osstart_t + osduration - 1)) = FBMy + yOffset;

                traceposition{datasetIndex, particleIndex} = single([ ...
                    mean(reshape(psfobj.Xpos, [oversampling, simParams.Frames]))', ...
                    mean(reshape(psfobj.Ypos, [oversampling, simParams.Frames]))']);
                psfobj.precomputeParam();
                psfobj.genPupil();
                psfobj.genPSF();
                norm_parameter = sum(sum(psfobj.Pupil.mag));
                psfobj.scalePSF('normal');
                psf = psfobj.ScaledPSFs;
                psf_blur = psf ./ norm_parameter .* (photons / oversampling);
                psf_blur(isnan(psf_blur)) = 0;
                psf_all(:,:,:) = psf_all(:,:,:) + psf_blur(:,:,:);

                temp = reshape(psf_all, simParams.Image_dims, simParams.Image_dims, oversampling, simParams.Frames);
                blurpsf = squeeze(sum(temp, 3));
            else
                [trajectoryX, trajectoryY] = fractional_brownian_motion_generator_2D( ...
                    hurstExponent, duration{datasetIndex, particleIndex}, diffusionCoefficient);
                xOffset = rand_range(-(simParams.Image_dims / 2) + 4, (simParams.Image_dims / 2) - 4);
                yOffset = rand_range(-(simParams.Image_dims / 2) + 4, (simParams.Image_dims / 2) - 4);
                psfobj.Xpos = nan(simParams.Frames, 1);
                psfobj.Ypos = nan(simParams.Frames, 1);
                psfobj.Zpos = zeros(simParams.Frames, 1);
                psfobj.Xpos(start_t:(start_t + duration{datasetIndex, particleIndex} - 1)) = trajectoryX + xOffset;
                psfobj.Ypos(start_t:(start_t + duration{datasetIndex, particleIndex} - 1)) = trajectoryY + yOffset;
                photons = rand_range(psfParams.Photon_min, psfParams.Photon_max);

                traceposition{datasetIndex, particleIndex} = single([psfobj.Xpos, psfobj.Ypos]);

                psfobj.precomputeParam();
                psfobj.genPupil();
                psfobj.genPSF();
                psfobj.scalePSF('normal');
                norm_parameter = sum(sum(psfobj.Pupil.mag));
                psf = psfobj.ScaledPSFs;
                psf = psf ./ norm_parameter .* photons;
                psf(isnan(psf)) = 0;
                psf_all(:,:,:) = psf_all(:,:,:) + psf(:,:,:);
            end

            Hlabel{datasetIndex, particleIndex} = single(hurstExponent);
            Clabel{datasetIndex, particleIndex} = single(diffusionCoefficient);
            photonlabel{datasetIndex, particleIndex} = single(photons);
            moleculeCounter = moleculeCounter + 1;
            moleculeid{datasetIndex, particleIndex} = moleculeCounter;
        end

        bglabel(datasetIndex) = single(bgLevel);
        Perlinbglabel(datasetIndex) = single(Perlin_bg);
        perlin = perlin_noise(simParams.Image_dims);
        if enableMotionBlur
            timelapsedata(:,:,:,datasetIndex) = poisson_noise(blurpsf(:,:,:) + bgLevel + Perlin_bg * perlin);
        else
            timelapsedata(:,:,:,datasetIndex) = poisson_noise(psf_all + bgLevel + Perlin_bg * perlin);
        end
    end

    saveFileName = fullfile(folder, sprintf('trainingvideos_%d.mat', fileindex));
    save(saveFileName, 'timelapsedata', 'Hlabel', 'Clabel', 'photonlabel', ...
        'bglabel', 'traceposition', 'moleculeid', 'Perlinbglabel', 'duration', '-v7.3');
    fprintf('Saved %s in %.1f seconds.\n', saveFileName, toc);
end

fprintf('Simulation Completed!\n');
end

function value = env_number(name, defaultValue)
    raw = getenv(name);
    if isempty(raw)
        value = defaultValue;
        return;
    end
    value = str2double(raw);
    if isnan(value)
        error('Environment variable %s must be numeric, got "%s".', name, raw);
    end
end

function value = env_bool(name, defaultValue)
    raw = lower(strtrim(getenv(name)));
    if isempty(raw)
        value = defaultValue;
        return;
    end
    value = any(strcmp(raw, {'1', 'true', 'yes', 'on'}));
end

function value = rand_range(minValue, maxValue)
    value = minValue + (maxValue - minValue) * rand();
end

function noisy = poisson_noise(lambda)
    lambda = max(double(lambda), 0);
    if exist('poissrnd', 'file') == 2
        try
            noisy = single(poissrnd(lambda));
            return;
        catch ME
            persistent warnedPoissrndFallback;
            if isempty(warnedPoissrndFallback)
                warning('poissrnd exists but failed (%s). Falling back to local Poisson sampler.', ME.message);
                warnedPoissrndFallback = true;
            end
        end
    end

    noisy = zeros(size(lambda), 'double');

    highMask = lambda >= 30;
    if any(highMask(:))
        highLambda = lambda(highMask);
        noisy(highMask) = round(highLambda + sqrt(highLambda) .* randn(size(highLambda)));
        noisy(highMask) = max(noisy(highMask), 0);
    end

    lowMask = ~highMask;
    if any(lowMask(:))
        lowLambda = lambda(lowMask);
        L = exp(-lowLambda(:));
        p = ones(numel(lowLambda), 1);
        k = zeros(numel(lowLambda), 1);
        active = true(numel(lowLambda), 1);

        while any(active)
            nActive = nnz(active);
            k(active) = k(active) + 1;
            p(active) = p(active) .* rand(nActive, 1);
            active = p > L;
        end

        noisy(lowMask) = reshape(k - 1, size(lowLambda));
    end

    noisy = single(noisy);
end

function [x, y] = fractional_brownian_motion_generator_2D(H, steps, C)
    covariancematrix = zeros(steps, steps);
    for t = 1:steps
        for s = 1:steps
            covariancematrix(t, s) = 0.5 * ((t^(2 * H) + s^(2 * H) - abs(t - s)^(2 * H)));
        end
    end
    L = chol(covariancematrix, 'lower');
    X = sqrt(2 * C) * randn(steps, 1);
    Y = sqrt(2 * C) * randn(steps, 1);
    x = L * X;
    y = L * Y;
end

function tracks = uniform_duration_and_present(T, numParticles, Lmin, Lmax)
    occ = zeros(1, T);
    tracks = struct('t0', cell(1, numParticles), 't1', cell(1, numParticles), 'L', cell(1, numParticles));
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
                bestStarts(end + 1) = t0;
            end
        end
        t0 = bestStarts(randi(numel(bestStarts)));
        t1 = t0 + L - 1;
        occ(t0:t1) = occ(t0:t1) + 1;

        tracks(k).t0 = t0;
        tracks(k).t1 = t1;
        tracks(k).L = L;
    end
    tmp = tracks;
    tracks = struct('t0', {}, 't1', {}, 'L', {});
    tracks(order) = tmp;
end
