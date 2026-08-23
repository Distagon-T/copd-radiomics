ct_dir = 'D:\copd-radiomics\ct_source\';
ct_files = dir(fullfile(ct_dir, 'patient_10_ct.nii.gz'));


    filepath = fullfile(ct_files.folder, ct_files.name);
    outpath = 'D:\copd-radiomics\ct_sourceX\patient_10_ct.nii.gz';
    img = niftiread(filepath);
    info = niftiinfo(filepath);

    img(img < -1024) = -1024;  % FOV 外区域设为空气值

    niftiwrite(img, outpath, info);  % 覆盖原文件
    fprintf('✅ 已修复: %s\n', ct_files.name);
