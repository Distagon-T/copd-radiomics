%% 极简输入终极管线 (仅需 CT + 掩膜，内部动态生成骨架)
clear; clc;

% =========================================================================
% 1. 定义专属路径 (只需提供这两个文件！)
% =========================================================================
CT_name    = 'D:\copd-radiomics\ct_sourceX\patient_00.nii.gz';  

seg_name   = 'D:\copd-radiomics\Airway_out\patient_00_airway.nii.gz'; 

% 自动生成输出路径
output_csv  = strrep(seg_name, '.nii.gz', '_metrics.csv');
output_wall = strrep(seg_name, '.nii.gz', '_OuterWall.nii.gz');

% =========================================================================
% 2. 加载数据与物理参数
% =========================================================================
disp('-> 正在加载 CT 与掩膜数据...');
meta_CT = niftiinfo(CT_name);
source  = double(niftiread(meta_CT));
meta_seg = niftiinfo(seg_name);
seg_raw  = logical(niftiread(meta_seg));

spacing = meta_CT.PixelDimensions;
mean_spacing = mean(spacing); 
voxel_vol = prod(spacing);

% 基础清洗：填补绝对封闭的内部气泡，保留最大主树
seg_base = imfill(seg_raw, 'holes');
CC = bwconncomp(seg_base, 26);
numPixels = cellfun(@numel, CC.PixelIdxList);
[~, idx] = max(numPixels);
seg_clean = false(size(seg_base));
seg_clean(CC.PixelIdxList{idx}) = true;
seg_base = seg_clean;

% =========================================================================
% 3. 内部动态骨架生成与拓扑自愈合 (取代外部 PTK 骨架)
% =========================================================================
kernel_sizes = [0, 3, 5, 7]; 
success = false;

for k = kernel_sizes
    fprintf('\n======================================================\n');
    fprintf('🔄 正在尝试骨架净化等级: %d\n', k);
    
    if k == 0
        seg_for_skel = seg_base;
    else
        se = strel('cube', k);
        seg_for_skel = imopen(seg_base, se);
        
        CC_skel = bwconncomp(seg_for_skel, 26);
        numPixels_skel = cellfun(@numel, CC_skel.PixelIdxList);
        [~, idx_skel] = max(numPixels_skel);
        temp = false(size(seg_for_skel));
        temp(CC_skel.PixelIdxList{idx_skel}) = true;
        seg_for_skel = temp;
    end
    
    disp('   -> 正在使用 MATLAB 原生引擎动态生成 3D 中心线骨架...');
    skel = bwskel(seg_for_skel, 'MinBranchLength', 15); % 原生生成骨架！
    
    disp('   -> 正在将骨架送入 AirQuant 进行图网络编译...');
    try
        AQnet = ClinicalAirways(skel, ...
            'source', source, ...
            'header', meta_CT, ...
            'seg', seg_base, ... 
            'fillholes', 0, ...  
            'largestCC', 1, ...
            'plane_sample_sz', 0.5, ...
            'spline_sample_sz', 0.5);
            
        disp('   -> 图网络编译成功！');
        success = true;
        break;
    catch ME
        fprintf('   ❌ 失败。错误信息: %s\n', ME.message);
    end
end

if ~success
    error('⚠️ 拓扑自愈合失败，掩膜存在严重粘连。');
end

% =========================================================================
% 4. 分支标签映射：直接线性索引 (与 AirQuant 官方 ClassifySegmentationTubes 一致)
% =========================================================================
disp('-> 正在为每根气管分支分配标签 (直接线性索引映射)...');
label_matrix = zeros(size(skel), 'uint16');
for i = 1:length(AQnet.tubes)
    label_matrix(AQnet.tubes(i).skelpoints) = i;
end
fprintf('   ✅ 已标记 %d 个骨架点，共 %d 根分支。\n', nnz(label_matrix > 0), length(AQnet.tubes));

branches = skel; 
dist_from_lumen = bwdist(seg_base) * mean_spacing;
results = zeros(0, 9); 
pad = ceil(6 / mean_spacing); 

% 预分配 3D 外壁矩阵
outer_wall_mask = false(size(seg_base));

% =========================================================================
% 5. 局部 Bounding Box 裁切与 FWHM 厚度提取 (含 3D 外壁重构)
% =========================================================================
for k = 1:length(AQnet.tubes)
    idx = find(label_matrix == k);
    if length(idx) < 3, continue; end
    
    [d1, d2, d3] = ind2sub(size(seg_base), idx);
    min1 = max(1, min(d1) - pad); max1 = min(size(seg_base,1), max(d1) + pad);
    min2 = max(1, min(d2) - pad); max2 = min(size(seg_base,2), max(d2) + pad);
    min3 = max(1, min(d3) - pad); max3 = min(size(seg_base,3), max(d3) + pad);
    
    local_branches = branches(min1:max1, min2:max2, min3:max3);
    local_label    = label_matrix(min1:max1, min2:max2, min3:max3);
    local_seg      = seg_base(min1:max1, min2:max2, min3:max3);
    local_dist     = dist_from_lumen(min1:max1, min2:max2, min3:max3);
    local_source   = source(min1:max1, min2:max2, min3:max3);
    
    branch_k_mask = (local_label == k);
    other_branches_mask = (local_branches & (local_label ~= k) & (local_label > 0));
    
    D_k = bwdist(branch_k_mask);
    if any(other_branches_mask(:))
        D_other = bwdist(other_branches_mask);
        local_M = (D_k < D_other); 
    else
        local_M = true(size(local_branches)); 
    end
    
    L = AQnet.tubes(k).stats.arclength; 
    lumen_voxels = sum(local_M(:) & local_seg(:));
    LA = (lumen_voxels * voxel_vol) / L; 
    D_in = 2 * sqrt(LA / pi); 
    Pi_val = pi * D_in; 
    
    wall_zone = local_M & ~local_seg & (local_dist <= 5);
    d_vals = local_dist(wall_zone);
    hu_vals = local_source(wall_zone);
    
    if isempty(d_vals), continue; end
    
    edges = 0:0.2:5;
    bin_centers = edges(1:end-1) + 0.1;
    [~, ~, bin_idx] = histcounts(d_vals, edges);
    
    mean_hu = zeros(length(bin_centers), 1);
    for b = 1:length(bin_centers)
        if any(bin_idx == b)
            mean_hu(b) = mean(hu_vals(bin_idx == b)); 
        else
            mean_hu(b) = NaN;
        end
    end
    mean_hu = fillmissing(mean_hu, 'linear'); 
    
    [peak_hu, peak_idx] = max(mean_hu);
    lung_hu = min(mean_hu(peak_idx:end));
    if isempty(lung_hu) || isnan(lung_hu), lung_hu = -850; end
    
    half_max = (peak_hu + lung_hu) / 2;
    drop_idx = find(mean_hu(peak_idx:end) < half_max, 1);
    
    if isempty(drop_idx)
        WT = NaN;
    else
        WT = bin_centers(peak_idx + drop_idx - 1); 
        
        % -------------------------------------------------------------
        % 【核心修复】：解决离散网格的"亚像素吞噬"现象
        % 保证无论物理厚度多薄，都至少向外渲染一层像素，或者加上半个像素宽容度
        % -------------------------------------------------------------
        render_WT = max(WT, mean_spacing); % 保底渲染机制
        
        local_wall_reconstruction = local_M & ~local_seg & (local_dist <= render_WT);
        
        % 写回全图矩阵
        outer_wall_mask(min1:max1, min2:max2, min3:max3) = ...
            outer_wall_mask(min1:max1, min2:max2, min3:max3) | local_wall_reconstruction;
    end
    
    D_out = D_in + 2 * WT; 
    WA = pi * (D_out/2)^2 - pi * (D_in/2)^2; 
    sqrt_WA = sqrt(WA);
    WA_pct = (WA / (WA + LA)) * 100; 
    
    results = [results; k, LA, WA, WA_pct, D_in, D_out, WT, Pi_val, sqrt_WA];
end

% =========================================================================
% 6. 输出与金标准计算
% =========================================================================
disp('-> 正在导出特征与 3D 掩膜...');
AQnet.ExportCSV(output_csv);
topo_table = readtable(output_csv);
VariableNames = {'ID', 'LumenArea_mm2', 'WallArea_mm2', 'WA_pct', 'Inner_Diameter_mm', 'Outer_Diameter_mm', 'Wall_Thickness_mm', 'Pi_Perimeter_mm', 'Sqrt_WallArea'};
if isempty(results)
    geom_table = array2table(zeros(0, 9), 'VariableNames', VariableNames);
else
    geom_table = array2table(results, 'VariableNames', VariableNames);
end
final_table = outerjoin(topo_table, geom_table, 'Keys', 'ID', 'MergeKeys', true);
writetable(final_table, output_csv);

% 导出多标签 3D NIfTI 掩膜 (1=内腔, 2=外壁)
final_multi_label_mask = uint8(seg_base); % 1为内腔
final_multi_label_mask(outer_wall_mask) = 2; % 2为外壁强行覆盖

meta_wall = meta_seg;
meta_wall.Datatype = 'uint8';

% 【核心修复 2】：抹除可能的旧缩放系数，防止 ITK-SNAP 标签错乱
if isfield(meta_wall, 'MultiplicativeScaling')
    meta_wall.MultiplicativeScaling = 1;
end
if isfield(meta_wall, 'AdditiveOffset')
    meta_wall.AdditiveOffset = 0;
end

niftiwrite(final_multi_label_mask, output_wall, meta_wall, 'Compressed', true);
fprintf('🎉 外壁 3D 掩膜已保存至: %s\n', output_wall);

% Pi10 计算
valid_idx = ~isnan(final_table.Pi_Perimeter_mm) & ~isnan(final_table.Sqrt_WallArea) & (final_table.Pi_Perimeter_mm > 0);
X_Pi = final_table.Pi_Perimeter_mm(valid_idx);       
Y_sqrtWA = final_table.Sqrt_WallArea(valid_idx);  
if length(X_Pi) > 5
    p = polyfit(X_Pi, Y_sqrtWA, 1);
    Pi10 = polyval(p, 10);
    fprintf('\n🔥 本例患者的 Pi10 计算结果为: %.4f\n', Pi10);
    
    figure('Name', 'Pi10 Regression Model', 'Color', 'w');
    scatter(X_Pi, Y_sqrtWA, 30, 'filled', 'MarkerFaceColor', '#0072BD', 'MarkerEdgeColor', 'w');
    hold on;
    x_fit = linspace(min(X_Pi)*0.8, max(X_Pi)*1.2, 100);
    plot(x_fit, polyval(p, x_fit), 'r-', 'LineWidth', 2);
    plot([10 10], [0 Pi10], 'k--', 'LineWidth', 1.5);
    plot([0 10], [Pi10 Pi10], 'k--', 'LineWidth', 1.5);
    scatter(10, Pi10, 80, 'rp', 'filled', 'MarkerEdgeColor', 'k'); 
    title(sprintf('Airway Remodeling Regression (Pi10 = %.3f)', Pi10), 'FontSize', 14, 'FontWeight', 'bold');
    xlabel('Internal Perimeter (Pi, mm)', 'FontSize', 12, 'FontWeight', 'bold');
    ylabel('Square Root of Wall Area (\surdWA, mm)', 'FontSize', 12, 'FontWeight', 'bold');
    grid on; set(gca, 'FontSize', 11, 'LineWidth', 1.2); hold off;
end
disp('🎉 流程全部完成！');