% 定义你的文件路径
f_inner  = 'D:\copd-radiomics\Airway_out\patient_00_airway.nii.gz';
f_outer  = 'D:\copd-radiomics\Airway_out\patient_00_airway_OuterWall.nii.gz';
f_output = 'D:\copd-radiomics\Airway_out\patient_00_Combined_Mask.nii.gz';

% 调用函数并获取执行结果
is_success = mergeAirwayMasks(f_inner, f_outer, f_output);

if is_success
    disp('准备在 ITK-SNAP 中打开查看！');
else
    disp('合并失败，请检查文件路径。');
end