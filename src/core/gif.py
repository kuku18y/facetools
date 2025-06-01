import imageio
import numpy as np
from PIL import Image, ImageSequence, ImagePalette
import os
import tempfile
import cv2

from ..utils import config
from .image import ImageOptimizer # 引入ImageOptimizer

class GifGenerator:
    """负责将处理后的帧序列编码为GIF动画，并进行文件大小优化。"""

    @staticmethod
    def estimate_frame_size_pil(pil_frame):
        """估算单帧Pillow图像（已量化）在GIF中的大致大小（字节）"""
        # 这是一个非常粗略的估计，实际大小受LZW压缩等因素影响很大
        # 假设每个像素索引1字节，加上一些GIF结构开销
        if pil_frame.mode != 'P':
            # 如果不是调色板模式，先尝试转换为调色板模式来估计
            temp_quantized = pil_frame.convert('P', palette=Image.ADAPTIVE, colors=config.MAX_GIF_COLORS)
            return temp_quantized.size[0] * temp_quantized.size[1] + 1024 # 1KB 开销
        return pil_frame.size[0] * pil_frame.size[1] + 1024

    @staticmethod
    def calculate_optimal_fps(pil_frames, target_size_kb=config.WECHAT_STICKER_MAX_SIZE_KB, initial_fps=config.DEFAULT_GIF_FPS, min_fps=5):
        """
        根据目标文件大小估算最佳FPS。
        这是一个迭代过程，实际生成GIF来检查大小，比较耗时。
        也可以使用更简单的基于平均帧大小的估计。

        Args:
            pil_frames (list[Image.Image]): Pillow图像帧列表 (模式 'P')。
            target_size_kb (int): 目标文件大小 (KB)。
            initial_fps (int): 初始尝试的FPS。
            min_fps (int): 允许的最小FPS。

        Returns:
            int: 估算的最佳FPS。
        """
        if not pil_frames:
            return initial_fps

        current_fps = initial_fps
        
        # 尝试使用更快的估算方法：
        # 1. 估算单帧的平均大小
        avg_frame_bytes = np.mean([GifGenerator.estimate_frame_size_pil(f) for f in pil_frames])
        # 2. 根据总帧数和目标大小反推FPS
        # total_bytes = num_frames * avg_frame_bytes
        # target_total_bytes = target_size_kb * 1024
        # desired_num_frames = target_total_bytes / avg_frame_bytes
        # video_duration = len(pil_frames) / initial_fps # 假设原始帧是以initial_fps采样的
        # estimated_fps = desired_num_frames / video_duration
        
        # 更精确（但更慢）的方法：实际生成GIF来测试大小
        # 这里我们先用一个简化的迭代逼近
        # 注意：这个循环可能比较耗时，实际应用中可能需要异步执行或提供进度反馈
        for test_fps in range(initial_fps, min_fps -1, -1):
            if test_fps <= 0: continue
            duration_ms = int(1000 / test_fps)
            try:
                with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as tmpfile:
                    output_path = tmpfile.name
                
                # 使用 imageio 保存，它通常有较好的优化
                imageio.mimsave(output_path, pil_frames, format='GIF', duration=duration_ms, subrectangles=True, palettesize=config.MAX_GIF_COLORS)
                # Pillow 保存方法：
                # pil_frames[0].save(output_path, save_all=True, append_images=pil_frames[1:], optimize=True, 
                #                    duration=duration_ms, loop=0, palette=pil_frames[0].palette) # 假设所有帧共享调色板

                file_size_kb = os.path.getsize(output_path) / 1024
                os.remove(output_path)

                if file_size_kb <= target_size_kb:
                    current_fps = test_fps
                    break # 找到一个可接受的FPS
                else:
                    current_fps = test_fps # 记录当前的，即使它超了，也是最接近的

            except Exception as e:
                # print(f"估算FPS时GIF生成失败 ({test_fps}fps): {e}")
                if os.path.exists(output_path):
                    os.remove(output_path)
                continue # 尝试下一个FPS
        
        return max(current_fps, min_fps) #确保不低于最小FPS

    @staticmethod
    def apply_delta_encoding(frames):
        """
        应用差分编码优化，减少帧间冗余，特别适合变化区域较小的GIF。
        该方法识别相邻帧之间的变化区域，只保留变化部分，从而提高GIF压缩效率。
        
        Args:
            frames (list): PIL图像帧列表或RGB帧列表 (numpy.ndarray)
            
        Returns:
            list: 处理后的PIL图像帧列表
        """
        if not frames or len(frames) < 2:
            return frames
            
        # 确保转换为numpy数组处理
        numpy_frames = []
        for frame in frames:
            if isinstance(frame, Image.Image):
                # 转换PIL图像为numpy数组
                numpy_frames.append(np.array(frame))
            else:
                numpy_frames.append(frame)
        
        # 第一帧保持不变
        result = [numpy_frames[0]]
        prev_frame = numpy_frames[0]
        
        # 处理后续帧
        for i in range(1, len(numpy_frames)):
            current_frame = numpy_frames[i]
            
            # 计算差异
            if len(current_frame.shape) == 3:  # 彩色图像
                delta = cv2.absdiff(current_frame, prev_frame)
                gray_delta = cv2.cvtColor(delta, cv2.COLOR_RGB2GRAY)
            else:  # 灰度图像或索引图像
                gray_delta = cv2.absdiff(current_frame, prev_frame)
                
            # 创建差异掩码（二值化处理）
            _, mask = cv2.threshold(gray_delta, 15, 255, cv2.THRESH_BINARY)
            
            # 腐蚀和膨胀操作，移除小的噪点并连接相近区域
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=1)
            
            # 转换掩码为3通道（如果输入是彩色图像）
            if len(current_frame.shape) == 3:
                mask_3d = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
            else:
                mask_3d = mask
                
            # 创建透明背景（与前一帧相同的区域变透明）
            # 注意：实际上GIF不支持真正的透明度，这里是一种模拟
            # 最终GIF中，这些区域会使用调色板的第一个颜色（通常是背景色）
            transparent_frame = np.zeros_like(current_frame)
            
            # 只保留变化区域
            if len(current_frame.shape) == 3:  # 彩色图像
                for c in range(3):  # 处理每个颜色通道
                    transparent_frame[:,:,c] = np.where(mask_3d[:,:,c] > 0, current_frame[:,:,c], prev_frame[:,:,c])
            else:  # 灰度图像
                transparent_frame = np.where(mask > 0, current_frame, prev_frame)
                
            # 添加处理后的帧
            result.append(transparent_frame)
            
            # 更新前一帧
            prev_frame = current_frame
            
        # 将numpy数组转回PIL图像
        pil_result = []
        for frame in result:
            pil_result.append(Image.fromarray(frame))
            
        return pil_result

    @staticmethod
    def create_optimized_gif(frames, output_path, fps=config.DEFAULT_GIF_FPS, max_colors=config.MAX_GIF_COLORS, 
                            target_size_kb=config.WECHAT_STICKER_MAX_SIZE_KB, dither=True, optimize_palette_globally=True):
        """
        创建优化的GIF文件，控制文件大小在目标范围内。
        
        Args:
            frames (list): RGB帧列表 (numpy.ndarray)
            output_path (str): 输出GIF文件路径
            fps (float): 目标帧率
            max_colors (int): 最大颜色数量
            target_size_kb (float): 目标文件大小 (KB)
            dither (bool): 是否使用抖动算法
            optimize_palette_globally (bool): 是否使用全局调色板
        
        Returns:
            float: 生成的GIF文件大小 (KB)
        """
        if not frames:
            print("错误：帧列表为空，无法生成GIF")
            return 0
            
        try:
            # 转换为PIL图像
            pil_frames = []
            for frame in frames:
                # 如果是numpy数组，转换为PIL
                if isinstance(frame, np.ndarray):
                    pil_img = Image.fromarray(frame)
                else:
                    pil_img = frame
                pil_frames.append(pil_img)
                
            # 1. 首先尝试使用原始帧率和颜色生成GIF
            duration_ms = int(1000 / fps)
            
            # 2. 如果需要，优化调色板
            if optimize_palette_globally:
                try:
                    # 使用全局调色板，所有帧共用一个调色板
                    # 从所有帧中采样颜色
                    palette = GifGenerator.optimize_color_palette(frames, max_colors)
                    
                    # 应用调色板到所有帧
                    for i in range(len(pil_frames)):
                        # 在PIL中，使用quantize而不是直接设置调色板
                        dither_method = 1 if dither else 0  # 1=FLOYDSTEINBERG, 0=NONE
                        pil_frames[i] = pil_frames[i].quantize(colors=max_colors, dither=dither_method)
                except Exception as e:
                    print(f"全局调色板优化失败，使用每帧独立调色板: {e}")
                    # 如果全局调色板失败，退回到独立调色板
                    for i in range(len(pil_frames)):
                        dither_method = 1 if dither else 0
                        pil_frames[i] = pil_frames[i].quantize(colors=max_colors, dither=dither_method)
            else:
                # 每帧使用独立调色板
                for i in range(len(pil_frames)):
                    dither_method = 1 if dither else 0
                    pil_frames[i] = pil_frames[i].quantize(colors=max_colors, dither=dither_method)
            
            # 3. 保存GIF
            pil_frames[0].save(
                output_path,
                save_all=True,
                append_images=pil_frames[1:],
                optimize=True,
                duration=duration_ms,
                loop=0  # 0表示无限循环
            )
            
            # 4. 检查大小，如果超出限制，尝试优化
            file_size_kb = os.path.getsize(output_path) / 1024
            
            # 如果文件太大，尝试降低帧率和颜色数量
            if file_size_kb > target_size_kb:
                current_fps = fps
                current_colors = max_colors
                # 初始降低参数的幅度
                fps_factor = 0.8
                
                # 增加优化迭代次数，从3次增加到5次
                for attempt in range(5):
                    # 更激进的参数调整策略
                    if current_fps > 6:  # 降低最低FPS限制到6
                        current_fps = max(6, current_fps * fps_factor)
                        # 随着迭代次数增加，更激进地降低帧率
                        fps_factor = max(0.6, fps_factor - 0.05)
                    else:
                        # 更激进地降低颜色数量
                        current_colors = max(32, current_colors // 2)  # 降低最低颜色到32
                    
                    print(f"GIF大小 {file_size_kb:.1f}KB 超过目标 {target_size_kb}KB，"
                          f"尝试优化 (第{attempt+1}次)：帧率={current_fps:.1f}fps，颜色={current_colors}")
                    
                    # 使用新参数重新生成
                    duration_ms = int(1000 / current_fps)
                    
                    # 如果需要降低帧数，每隔几帧采样一次
                    if current_fps < fps:
                        frame_interval = max(1, int(fps / current_fps))
                        optimized_frames = [pil_frames[i] for i in range(0, len(pil_frames), frame_interval)]
                    else:
                        optimized_frames = pil_frames
                    
                    # 在最后两次尝试中使用更小的颜色数量
                    if attempt >= 3:
                        current_colors = max(16, current_colors)  # 允许更低的颜色数量
                    
                    # 重新量化颜色
                    if current_colors < max_colors:
                        for i in range(len(optimized_frames)):
                            # 最后一次尝试时禁用抖动以减小文件大小
                            dither_method = 0 if attempt >= 4 else (1 if dither else 0)
                            optimized_frames[i] = optimized_frames[i].quantize(colors=current_colors, dither=dither_method)
                    
                    # 重新保存
                    optimized_frames[0].save(
                        output_path,
                        save_all=True,
                        append_images=optimized_frames[1:],
                        optimize=True,
                        duration=duration_ms,
                        loop=0
                    )
                    
                    # 检查新大小
                    file_size_kb = os.path.getsize(output_path) / 1024
                    if file_size_kb <= target_size_kb:
                        print(f"优化成功：{file_size_kb:.1f}KB")
                        break
                
                # 如果常规优化后仍然超过大小限制，尝试使用关键帧检测
                if file_size_kb > target_size_kb:
                    print(f"标准优化后GIF仍然过大 ({file_size_kb:.1f}KB)，尝试关键帧检测...")
                    
                    # 从numpy帧中检测关键帧
                    original_frames = frames
                    keyframe_indices = GifGenerator.detect_keyframes(original_frames, threshold=0.03)  # 使用更小的阈值捕获更多细微变化
                    
                    if len(keyframe_indices) < len(original_frames) * 0.8:  # 如果至少减少了20%的帧
                        keyframes = [original_frames[i] for i in keyframe_indices]
                        
                        # 使用关键帧重新生成GIF，使用最低设置
                        keyframe_pil_frames = [Image.fromarray(frame) for frame in keyframes]
                        for i in range(len(keyframe_pil_frames)):
                            keyframe_pil_frames[i] = keyframe_pil_frames[i].quantize(colors=16, dither=0)
                        
                        # 降低帧率
                        keyframe_fps = max(5, current_fps * 0.8)
                        duration_ms = int(1000 / keyframe_fps)
                        
                        # 保存关键帧GIF
                        keyframe_pil_frames[0].save(
                            output_path,
                            save_all=True,
                            append_images=keyframe_pil_frames[1:],
                            optimize=True,
                            duration=duration_ms,
                            loop=0
                        )
                        
                        file_size_kb = os.path.getsize(output_path) / 1024
                        print(f"关键帧优化后：{file_size_kb:.1f}KB，使用了{len(keyframes)}帧 (原{len(original_frames)}帧)")
                    else:
                        print(f"关键帧检测无法有效减少帧数 ({len(keyframe_indices)} vs {len(original_frames)})")
                
                # 尝试差分编码优化
                if file_size_kb > target_size_kb:
                    print(f"尝试差分编码优化...")
                    
                    # 应用差分编码处理
                    delta_frames = GifGenerator.apply_delta_encoding(
                        keyframe_pil_frames if 'keyframe_pil_frames' in locals() else pil_frames
                    )
                    
                    # 重新量化处理后的帧
                    for i in range(len(delta_frames)):
                        delta_frames[i] = delta_frames[i].quantize(colors=16, dither=0)
                    
                    # 保存差分编码优化后的GIF
                    delta_frames[0].save(
                        output_path,
                        save_all=True,
                        append_images=delta_frames[1:],
                        optimize=True,
                        duration=duration_ms,
                        loop=0
                    )
                    
                    file_size_kb = os.path.getsize(output_path) / 1024
                    print(f"差分编码优化后：{file_size_kb:.1f}KB")
                
                # 最后的保障措施：如果所有尝试都失败，强制缩小图像尺寸
                if file_size_kb > target_size_kb:
                    print(f"所有优化方法都失败，尝试最后的缩小尺寸方法...")
                    
                    # 计算缩小比例
                    scale_factor = 0.9  # 初始缩小到90%
                    max_attempts = 3
                    
                    for i in range(max_attempts):
                        new_size = (
                            int(config.WECHAT_STICKER_WIDTH * scale_factor),
                            int(config.WECHAT_STICKER_HEIGHT * scale_factor)
                        )
                        
                        # 选择最优化后的帧列表
                        source_frames = (delta_frames if 'delta_frames' in locals() else 
                                       (keyframe_pil_frames if 'keyframe_pil_frames' in locals() else pil_frames))
                        
                        # 调整所有帧的尺寸
                        resized_frames = []
                        for frame in source_frames:
                            # 保留调色板模式
                            resized = frame.resize(new_size, Image.LANCZOS)
                            resized_frames.append(resized)
                        
                        # 重新保存
                        resized_frames[0].save(
                            output_path,
                            save_all=True,
                            append_images=resized_frames[1:],
                            optimize=True,
                            duration=duration_ms,
                            loop=0
                        )
                        
                        file_size_kb = os.path.getsize(output_path) / 1024
                        print(f"尺寸缩小到{new_size}后：{file_size_kb:.1f}KB")
                        
                        if file_size_kb <= target_size_kb:
                            break
                            
                        # 继续缩小
                        scale_factor *= 0.9
                
                # 最后的尝试：使用gifsicle工具进行终极压缩
                if file_size_kb > target_size_kb:
                    print(f"尝试使用gifsicle进行终极压缩...")
                    
                    # 创建一个临时文件用于备份当前最佳结果
                    with tempfile.NamedTemporaryFile(suffix='.gif', delete=False) as temp_file:
                        temp_backup = temp_file.name
                        
                    # 复制当前最佳GIF作为备份
                    import shutil
                    shutil.copy2(output_path, temp_backup)
                    
                    # 尝试使用gifsicle进行压缩
                    file_size_kb = GifGenerator.compress_with_gifsicle(output_path)
                    
                    # 检查结果
                    if file_size_kb <= target_size_kb:
                        print(f"gifsicle压缩成功：{file_size_kb:.1f}KB")
                        os.remove(temp_backup)  # 删除备份
                    else:
                        # 如果还是超过，恢复备份
                        print(f"gifsicle压缩后仍然超出大小限制：{file_size_kb:.1f}KB，恢复到最佳结果")
                        shutil.copy2(temp_backup, output_path)
                        file_size_kb = os.path.getsize(output_path) / 1024
                        os.remove(temp_backup)  # 删除备份
            
            return file_size_kb
            
        except Exception as e:
            print(f"GIF生成错误: {e}")
            import traceback
            traceback.print_exc()
            return 0

    @staticmethod
    def optimize_color_palette(frames, max_colors=256):
        """为一组帧优化颜色调色板"""
        # 收集所有帧的颜色样本
        color_samples = []
        sample_rate = max(1, len(frames) // 10)  # 采样率
        
        for i in range(0, len(frames), sample_rate):
            # 降低分辨率以加速处理
            frame = frames[i]
            if isinstance(frame, np.ndarray):
                small_frame = cv2.resize(frame, (40, 40))
                pixels = small_frame.reshape(-1, 3)
                color_samples.extend(pixels)
            elif isinstance(frame, Image.Image):
                small_frame = frame.resize((40, 40))
                pixels = np.array(small_frame).reshape(-1, 3)
                color_samples.extend(pixels)
        
        # 限制样本数量以提高性能
        if len(color_samples) > 1000:
            import random
            color_samples = random.sample(color_samples, 1000)
            
        color_samples = np.array(color_samples, dtype=np.float32)
        
        # 使用K-means聚类找到最佳颜色
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        _, labels, palette = cv2.kmeans(color_samples, max_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        # 将调色板转换为整数
        palette = np.uint8(palette)
        
        return palette

    @staticmethod
    def detect_keyframes(frames_rgb, threshold=0.05):
        """
        检测关键帧，用于减少GIF中的冗余帧 (增强版)。
        比较相邻帧的平均像素差异，并使用动态阈值和区域检测。

        Args:
            frames_rgb (list[np.ndarray]): RGB帧列表。
            threshold (float): 基础差异阈值 (0到1之间)。越小则越多关键帧。

        Returns:
            list[int]: 关键帧的索引列表。
        """
        if not frames_rgb or len(frames_rgb) < 2:
            return list(range(len(frames_rgb)))

        keyframes_indices = [0]  # 第一帧总是关键帧
        prev_frame = frames_rgb[0]
        
        # 计算整个序列的平均变化程度，用于自适应阈值
        all_diffs = []
        for i in range(1, len(frames_rgb)):
            current_frame = frames_rgb[i]
            # 转换为灰度来计算差异
            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_RGB2GRAY).astype(np.float32)
            curr_gray = cv2.cvtColor(current_frame, cv2.COLOR_RGB2GRAY).astype(np.float32)
            
            # 计算全局差异
            frame_diff = np.mean(np.abs(curr_gray - prev_gray)) / 255.0
            all_diffs.append(frame_diff)
            prev_frame = current_frame
        
        # 根据视频整体变化特性设定动态阈值
        if all_diffs:
            avg_diff = np.mean(all_diffs)
            std_diff = np.std(all_diffs)
            
            # 如果视频变化剧烈，提高阈值；如果变化微小，降低阈值
            if avg_diff > 0.1:  # 高变化视频
                dynamic_threshold = threshold * 1.5
            elif avg_diff < 0.02:  # 低变化视频
                dynamic_threshold = threshold * 0.5
            else:
                dynamic_threshold = threshold
            
            # 确保阈值在合理范围内
            dynamic_threshold = max(0.01, min(0.2, dynamic_threshold))
        else:
            dynamic_threshold = threshold
            
        # 重置用于实际关键帧检测
        prev_frame = frames_rgb[0]
        last_keyframe_idx = 0
        min_frames_between_keyframes = 2  # 至少间隔几帧才能有新关键帧
        
        for i in range(1, len(frames_rgb)):
            current_frame = frames_rgb[i]
            
            # 1. 全局差异检测
            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_RGB2GRAY).astype(np.float32)
            curr_gray = cv2.cvtColor(current_frame, cv2.COLOR_RGB2GRAY).astype(np.float32)
            global_diff = np.mean(np.abs(curr_gray - prev_gray)) / 255.0
            
            # 2. 运动区域检测
            motion_detected = False
            if global_diff > dynamic_threshold * 0.7:  # 接近阈值时进行运动区域分析
                # 计算帧差
                frame_diff = cv2.absdiff(prev_gray.astype(np.uint8), curr_gray.astype(np.uint8))
                # 阈值处理
                _, motion_mask = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)
                # 腐蚀和膨胀来去除噪点
                kernel = np.ones((3, 3), np.uint8)
                motion_mask = cv2.erode(motion_mask, kernel, iterations=1)
                motion_mask = cv2.dilate(motion_mask, kernel, iterations=2)
                # 计算运动区域百分比
                motion_percent = np.sum(motion_mask) / (motion_mask.shape[0] * motion_mask.shape[1] * 255)
                motion_detected = motion_percent > 0.05  # 如果超过5%的区域有运动
            
            # 3. 与最后一个关键帧比较，检测长期变化
            if i - last_keyframe_idx >= 10:  # 每隔一定帧数
                last_keyframe_gray = cv2.cvtColor(frames_rgb[last_keyframe_idx], cv2.COLOR_RGB2GRAY).astype(np.float32)
                long_term_diff = np.mean(np.abs(curr_gray - last_keyframe_gray)) / 255.0
                long_term_change = long_term_diff > dynamic_threshold
            else:
                long_term_change = False
            
            # 综合判断是否为关键帧
            is_keyframe = ((global_diff > dynamic_threshold or motion_detected or long_term_change) and 
                           (i - last_keyframe_idx >= min_frames_between_keyframes))
            
            # 强制关键帧：每隔一定帧数，即使变化不大也标记为关键帧，避免丢失累积的小变化
            if i - last_keyframe_idx >= max(15, len(frames_rgb) // 10):
                is_keyframe = True
                
            if is_keyframe:
                keyframes_indices.append(i)
                last_keyframe_idx = i
            
            # 更新前一帧
            prev_frame = current_frame
        
        # 确保最后一帧被包含
        if keyframes_indices[-1] != len(frames_rgb) - 1:
            keyframes_indices.append(len(frames_rgb) - 1)
        
        # 如果关键帧太少，增加一些中间帧
        if len(keyframes_indices) < max(3, len(frames_rgb) // 20):
            additional_frames = []
            for i in range(1, len(keyframes_indices)):
                start_idx = keyframes_indices[i-1]
                end_idx = keyframes_indices[i]
                if end_idx - start_idx > 10:  # 如果两个关键帧间隔太大
                    middle_idx = start_idx + (end_idx - start_idx) // 2
                    additional_frames.append(middle_idx)
            
            # 合并并排序
            keyframes_indices.extend(additional_frames)
            keyframes_indices.sort()
        
        return keyframes_indices

    @staticmethod
    def compress_with_gifsicle(input_gif_path, output_gif_path=None, optimization_level=3):
        """
        使用gifsicle命令行工具对GIF进行压缩优化（如果可用）。
        这是最后一道压缩防线，当其他所有方法都失败时使用。
        
        Args:
            input_gif_path (str): 输入GIF路径
            output_gif_path (str, optional): 输出GIF路径。如果为None，则覆盖输入文件。
            optimization_level (int): 优化级别，1-3，3为最高。
            
        Returns:
            float: 压缩后的文件大小(KB)，如果失败则返回0
        """
        if output_gif_path is None:
            output_gif_path = input_gif_path
            
        try:
            # 检查gifsicle是否可用
            import shutil
            import subprocess
            
            if shutil.which("gifsicle") is None:
                # print("gifsicle未安装，跳过额外压缩")
                return os.path.getsize(input_gif_path) / 1024
                
            # 准备命令行参数
            cmd = [
                "gifsicle",
                f"--optimize={optimization_level}",  # 优化级别
                "--colors=64",                       # 减少颜色数量
                "--lossy=80",                        # 有损压缩级别
                "--no-comments",                     # 删除注释
                "--no-names",                        # 删除名称
                "--no-extensions",                   # 删除扩展块
                "--careful",                         # 小心处理，避免损坏
                "-o", output_gif_path,               # 输出文件
                input_gif_path                       # 输入文件
            ]
            
            # 执行命令
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # 检查结果
            if os.path.exists(output_gif_path):
                return os.path.getsize(output_gif_path) / 1024
            return 0
            
        except Exception as e:
            print(f"使用gifsicle压缩失败: {e}")
            return os.path.getsize(input_gif_path) / 1024

# 示例用法 (用于测试)
if __name__ == '__main__':
    # 创建一些虚拟RGB帧 (np.ndarray)
    dummy_frames_rgb = []
    for i in range(20): # 20帧
        frame = np.random.randint(0, 256, size=(config.WECHAT_STICKER_HEIGHT, config.WECHAT_STICKER_WIDTH, 3), dtype=np.uint8)
        # 添加一些变化以测试关键帧检测和subrectangles
        if i % 5 == 0:
            frame[50:150, 50:150, :] = np.random.randint(0,256, (100,100,3), dtype=np.uint8)
        dummy_frames_rgb.append(frame)

    output_gif_path = "test_optimized.gif"
    output_keyframe_gif_path = "test_keyframe_optimized.gif"

    print("测试标准GIF生成...")
    file_size = GifGenerator.create_optimized_gif(dummy_frames_rgb, output_gif_path, fps=10, max_colors=128, target_size_kb=None)
    if file_size > 0:
        print(f"标准GIF已生成: {output_gif_path}, 大小: {file_size:.2f} KB")
        assert os.path.exists(output_gif_path)
    else:
        print("标准GIF生成失败")

    print("\n测试关键帧检测...")
    keyframe_indices = GifGenerator.detect_keyframes(dummy_frames_rgb, threshold=0.1)
    print(f"检测到 {len(keyframe_indices)} 个关键帧索引: {keyframe_indices}")
    assert 0 in keyframe_indices
    
    selected_frames_for_gif = [dummy_frames_rgb[i] for i in keyframe_indices]
    if not selected_frames_for_gif: # 如果关键帧列表为空，至少用第一帧
        selected_frames_for_gif = [dummy_frames_rgb[0]] if dummy_frames_rgb else []

    print("\n测试基于关键帧的GIF生成...")
    if selected_frames_for_gif:
        keyframe_file_size = GifGenerator.create_optimized_gif(selected_frames_for_gif, output_keyframe_gif_path, fps=5, max_colors=128, target_size_kb=None)
        if keyframe_file_size > 0:
            print(f"基于关键帧的GIF已生成: {output_keyframe_gif_path}, 大小: {keyframe_file_size:.2f} KB")
            assert os.path.exists(output_keyframe_gif_path)
            # 通常，基于关键帧的GIF（如果帧间变化不大）会比包含所有帧的GIF小，但帧率和内容高度相关
        else:
            print("基于关键帧的GIF生成失败")
    else:
        print("没有提取到关键帧用于生成GIF")

    print("\n测试有目标大小限制的GIF生成 (可能会降低FPS)...")
    target_kb = 30 # 设置一个较小的目标大小以触发优化
    # 使用原始帧列表
    file_size_limited = GifGenerator.create_optimized_gif(dummy_frames_rgb, "test_limited_size.gif", fps=15, max_colors=64, target_size_kb=target_kb)
    if file_size_limited > 0:
        print(f"大小限制GIF已生成: test_limited_size.gif, 大小: {file_size_limited:.2f} KB (目标: <{target_kb}KB)")
        assert os.path.exists("test_limited_size.gif")
        # if file_size_limited > target_kb * 1.1: # 允许10%的误差
            # print(f"警告: 文件大小优化后仍可能略超出目标。实际大小: {file_size_limited:.2f}KB")
        os.remove("test_limited_size.gif")
    else:
        print("大小限制GIF生成失败")

    # 清理测试文件
    if os.path.exists(output_gif_path): os.remove(output_gif_path)
    if os.path.exists(output_keyframe_gif_path): os.remove(output_keyframe_gif_path)

    print("\nGifGenerator 测试完成.") 