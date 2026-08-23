% 读取 CT 图像
nii_file = 'ct_source/patient_02_ct.nii.gz';
tmp = niftiread(nii_file);

% 获取图像尺寸
[rows, cols, slices] = size(tmp);
fprintf('图像尺寸: %d × %d × %d (层数)\n', rows, cols, slices);

% 找到所有等于 -3000 的位置
[bad_rows, bad_cols, bad_slices] = ind2sub(size(tmp), find(tmp < -1000));

% 统计每层有多少个 -3000 异常值
if isempty(bad_slices)
    fprintf('✅ 没有找到 -3000 的异常值。\n');
else
    slice_counts = histcounts(bad_slices, 1:slices+1);
    
    % 找出有异常值的层
    abnormal_slices = find(slice_counts > 0);
    
    fprintf('\n========== 异常值检测结果 ==========\n');
    fprintf('总共发现 %d 个值为 -3000 的像素\n', length(bad_rows));
    fprintf('分布在 %d 个不同的层中\n', length(abnormal_slices));
    
    % 列出每层异常值数量（只显示有异常值的层）
    fprintf('\n--- 各层异常值数量 ---\n');
    for i = 1:length(abnormal_slices)
        s = abnormal_slices(i);
        fprintf('第 %d 层: %d 个异常像素 (占总像素 %.2f%%)\n', ...
            s, slice_counts(s), 100 * slice_counts(s) / (rows * cols));
    end
    
    % 找出异常值最多的层（最值得检查）
    [max_count, worst_slice] = max(slice_counts);
    fprintf('\n⚠️  异常值最严重的层: 第 %d 层 (%d 个像素)\n', ...
        worst_slice, max_count);
    
    % 可视化：异常值在各层的分布
    figure;
    bar(abnormal_slices, slice_counts(abnormal_slices));
    xlabel('层号 (Slice Index)');
    ylabel('-3000 像素数量');
    title(sprintf('CT 图像中 -3000 异常值分布 (%s)', ...
        strrep(nii_file, '_', '\_')));
    grid on;
    
    % 可选：显示第一层有异常值的图像，帮助判断位置
    first_abnormal_slice = abnormal_slices(200);
    figure;
    subplot(1,2,1);
    imagesc(tmp(:,:,first_abnormal_slice));
    colormap gray; colorbar;
    title(sprintf('第 %d 层原始图像', first_abnormal_slice));
    
    subplot(1,2,2);
    mask_abnormal = (tmp(:,:,first_abnormal_slice) < -1000);
    imshow(mask_abnormal);
    title(sprintf('第 %d 层 -3000 像素位置 (红色)', first_abnormal_slice));
end

% 另外，也检查一下其他可能的异常值（如极小值）
fprintf('\n========== 其他潜在异常值 ==========\n');
fprintf('图像最小值: %.1f\n', min(tmp(:)));
fprintf('图像最大值: %.1f\n', max(tmp(:)));
fprintf('值 < -2000 的像素总数: %d\n', sum(tmp(:) < -2000));
fprintf('值 < -1000 且 > -1024 的像素总数(空气): %d\n', ...
    sum(tmp(:) < -1000 & tmp(:) > -1024));