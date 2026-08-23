disp('-> 正在启动高级几何测量引擎 (计算管壁内外径与壁厚)...');
% 注意：这一步运算量极大！算法会沿着数百条骨架向四周发射射线读取 CT 灰度
try
    % 实例化 FWHM (半高全宽) 射线测量器，传入原始 CT 和构建好的网络树
    measurer = measure.AirwayFWHMesl('source', source, ...
        'header', meta_CT, ...
        'net', AQnet);

    disp('   -> 射线发射与边界拟合中 (可能需要几分钟)...');
    measurer.Measure(); 
    disp('   🎉 壁厚测量完成！');

    % 将计算出的厚度数据强行绑定回网络节点中
    AQnet.measurements = measurer;
    is_measure_success = true;
catch ME
    warning('射线测量引擎崩溃，可能由于内存不足或管腔太细。跳过厚度计算。错误信息: %s', ME.message);
    is_measure_success = false;
end