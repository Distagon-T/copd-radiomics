function success = mergeAirwayMasks(inner_mask_file, outer_mask_file, output_file)
% mergeAirwayMasks 合并气道内腔与外壁掩膜 (适配 ITK-SNAP 多标签展示)
%
% 输入参数:
%   inner_mask_file - 气道内腔掩膜的 NIfTI 文件路径 (Label 1)
%   outer_mask_file - 气管外壁掩膜的 NIfTI 文件路径 (Label 2)
%   output_file     - 导出的合并多标签 NIfTI 文件路径
%
% 输出参数:
%   success         - 逻辑值，true 表示成功，false 表示失败

success = false;

% =====================================================================
% 1. 输入文件校验
% =====================================================================
if ~isfile(inner_mask_file)
    error('❌ 找不到内腔掩膜文件: %s', inner_mask_file);
end
if ~isfile(outer_mask_file)
    error('❌ 找不到外壁掩膜文件: %s', outer_mask_file);
end

try
    % =================================================================
    % 2. 读取 NIfTI 数据
    % =================================================================
    disp('-> 正在读取内腔与外壁掩膜...');
    meta_info  = niftiinfo(inner_mask_file);
    inner_mask = logical(niftiread(meta_info));
    outer_mask = logical(niftiread(outer_mask_file));

    % =================================================================
    % 3. 合并逻辑与多标签赋值
    % =================================================================
    disp('-> 正在执行多标签融合 (Background=0, Lumen=1, Wall=2)...');

    % 初始化一个全 0 的背景矩阵，强制使用 'uint8' 类型
    combined_mask = zeros(size(inner_mask), 'uint8');

    % 【覆盖逻辑】：先涂底色（外圈），再涂核心（内圈）
    combined_mask(outer_mask > 0) = 2;
    combined_mask(inner_mask > 0) = 1;

    % =================================================================
    % 4. 导出 NIfTI 文件
    % =================================================================
    disp('-> 正在导出至 ITK-SNAP 兼容格式...');

    % 修改 Header 中的 Datatype 以匹配 uint8 矩阵，避免底层报错
    meta_info.Datatype = 'uint8';

    niftiwrite(combined_mask, output_file, meta_info, 'Compressed', true);

    fprintf('🎉 合并完成！多标签掩膜已保存至: %s\n', output_file);
    success = true;

catch ME
    fprintf('❌ 合并过程中发生错误: %s\n', ME.message);
end
end