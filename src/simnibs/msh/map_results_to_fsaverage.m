function map_results_to_fsaverage(infiles, outfiles, headreco_dir)
%
%   map_results_to_fsaverage(infiles, outfiles, headreco_dir)
%
%   map results from the indiviudal surfaces to the fsaverage template
%   using CAT12 functions
%
%   infiles: cell-array of the to-be-mapped data files (stored in freesurfer
%   curvature format)
%   outfiles: cell-array of the names of the mapped files
%   headreco_dir: directory containing the results of the headreco-run
%
%   A. Thielscher, 02-Mar-2018


% try to get CAT defaults
try
    cat = cat_get_defaults();
catch
    try
        addpath(fullfile(spm('dir'),'toolbox','cat12'))
        cat = cat_get_defaults();
    catch ME
        disp('Could not get CAT12 defaults. Is CAT12 in toolbox/cat12?')
        exit(2);
    end
end

cat_res_dir= [headreco_dir filesep 'segment' filesep 'cat' filesep 'surf'];
if ~exist(cat_res_dir,'dir')
     disp('Could not locate CAT12 results. Did you set --cat when running headreco?')
     exit(2);
end

% CAT12 wants to have the files in the directory with the individual
% surfaces; we therefore create temporary copies of the input files
% in that directory, following the naming convention of CAT12
for i=1:length(infiles)
    [~,hemi,~] = fileparts(infiles{i});
    hemi=hemi(1:2);
    if ~(strcmpi(hemi,'rh')||strcmpi(hemi,'lh'))
        disp('input surface data has to have rh or lh at start of filename')
        disp(infiles{i})
        exit(2);
    end;
        
    hlpstr = tempname(cat_res_dir);
    [pathhlp,namehlp,~] = fileparts(hlpstr);
    
    tmpfiles{i}=[pathhlp filesep hemi '.' namehlp '.T1fs_conform'];
    copyfile(infiles{i},tmpfiles{i}) 
end;
disp(tmpfiles{1})
disp(tmpfiles{2})
% run CAT12 resampling on temporary files
s_resamp.data_surf=tmpfiles;
s_resamp.fwhm=0;
s_resamp.fwhm_surf=0;
s_resamp.merge_hemi=0;
s_resamp.nproc=0;
s_resamp.mesh32k=0;
P = cat_surf_resamp(s_resamp);

% delete temporary files
for i=1:length(tmpfiles)
    delete(tmpfiles{i});
end;

% newest CAT12 release assigns output names to lh and rh simultaneously
if isfield(P,'lPsdata')
    Phlp=P;
    P={};
    for i=1:length(Phlp.lPsdata)
        if ~isempty(Phlp.lPsdata{i})
            P{i}=Phlp.lPsdata{i};
        else
            P{i}=Phlp.rPsdata{i};
        end
    end
end

% extract data from resampled files and write data to outfiles
for i=1:length(P)
    resampled_data=gifti(P{i});
    cat_io_FreeSurfer('write_surf_data',outfiles{i},resampled_data.cdata);
    delete(P{i});
end;