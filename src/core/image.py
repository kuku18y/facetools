import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageOps
import random
from ..utils import config

class ImageOptimizer:
    """提供高级图像处理功能，用于优化GIF的视觉效果和文件大小。"""

    @staticmethod
    def optimize_color_palette(frames, max_colors=config.MAX_GIF_COLORS, sample_rate=0.1):
        """
        从一系列帧中计算优化的全局调色板。

        Args:
            frames (list[np.ndarray]): RGB帧列表。
            max_colors (int): 调色板中的最大颜色数。
            sample_rate (float): 用于颜色采样的帧的比例 (0.0到1.0)。

        Returns:
            Image.Palette: PIL调色板对象，如果无法生成则返回None。
        """
        if not frames:
            return None

        color_samples = []
        num_frames_to_sample = max(1, int(len(frames) * sample_rate))
        sampled_frame_indices = random.sample(range(len(frames)), num_frames_to_sample)

        for i in sampled_frame_indices:
            frame = frames[i]
            # 为了加速，可以先缩小图像再提取颜色
            pil_img = Image.fromarray(frame)
            # 降低采样图像的分辨率以加速颜色提取，但保留足够的颜色信息
            pil_img_small = pil_img.resize((max(1, frame.shape[1] // 4), max(1, frame.shape[0] // 4)), Image.LANCZOS)
            colors = pil_img_small.getcolors(pil_img_small.size[0] * pil_img_small.size[1])
            if colors:
                # getcolors 返回 (count, color_tuple_or_int)列表
                for count, color in colors:
                    if isinstance(color, tuple): # RGB
                        color_samples.append(color)
                    # else: # 索引颜色，不直接使用
        
        if not color_samples:
             # 如果无法从采样中获取颜色（例如，视频非常短或单色），则从第一帧中强行获取
            pil_img = Image.fromarray(frames[0])
            quantized_img = pil_img.quantize(colors=max_colors, method=Image.MAXCOVERAGE)
            return quantized_img.palette

        # 将RGB样本转换为Pillow可以处理的格式 (一维数组 R1,G1,B1,R2,G2,B2...)
        flat_samples = [c for rgb_tuple in color_samples for c in rgb_tuple]
        if not flat_samples: return None
        
        # 创建一个包含所有采样颜色的临时图像
        # 图像大小至少为1x1，宽度任意，高度为采样颜色数
        # 确保图像数据长度是宽度的倍数
        temp_img_width = 256 # 任意宽度
        num_pixels = len(color_samples)
        temp_img_height = (num_pixels + temp_img_width -1) // temp_img_width
        
        # 填充数据以匹配图像尺寸 (width * height * 3 for RGB)
        expected_data_len = temp_img_width * temp_img_height * 3
        current_data_len = len(flat_samples)
        if current_data_len < expected_data_len:
            flat_samples.extend([0] * (expected_data_len - current_data_len))
        elif current_data_len > expected_data_len:
            flat_samples = flat_samples[:expected_data_len]

        try:
            # 创建一个足够大的临时图像来容纳所有采样颜色
            temp_image_for_palette = Image.frombytes("RGB", (temp_img_width, temp_img_height), bytes(flat_samples))
            # 从这个临时图像生成调色板
            # Image.MEDIANCUT, Image.MAXCOVERAGE, Image.FASTOCTREE
            quantized_image = temp_image_for_palette.quantize(colors=max_colors, method=Image.MAXCOVERAGE)
            return quantized_image.palette
        except Exception as e:
            print(f"调色板优化错误: {e}")
            # Fallback: 使用第一帧的调色板
            pil_img = Image.fromarray(frames[0])
            quantized_img = pil_img.quantize(colors=max_colors)
            return quantized_img.palette

    @staticmethod
    def apply_palette_to_frame(frame_rgb, palette):
        """
        将优化后的调色板应用到单个帧。

        Args:
            frame_rgb (np.ndarray): RGB格式的输入帧。
            palette (Image.Palette): PIL调色板对象。

        Returns:
            Image.Image: 应用调色板后的Pillow图像对象 (模式 'P')。
        """
        pil_img = Image.fromarray(frame_rgb, 'RGB')
        # 使用dither=Image.FLOYDSTEINBERG 可以改善颜色过渡，但可能增加噪点
        quantized_img = pil_img.quantize(palette=palette, dither=Image.FLOYDSTEINBERG)
        return quantized_img

    @staticmethod
    def content_aware_crop(frame_rgb, target_width, target_height):
        """
        内容感知裁剪，尝试保留图像中最重要的部分。
        使用OpenCV的显著性检测 (如果可用且效果好) 或简单的中心裁剪。
        注意: OpenCV的显著性模块可能需要额外安装 `opencv-contrib-python`。
              这里提供一个基于Pillow的简化版本或中心裁剪作为备选。
        """
        try:
            # 尝试使用OpenCV的显著性检测 (如果高级功能需要)
            saliency = cv2.saliency.StaticSaliencyFineGrained_create()
            _, saliency_map = saliency.computeSaliency(frame_rgb)
            saliency_map = (saliency_map * 255).astype(np.uint8)
            
            moments = cv2.moments(saliency_map)
            if moments["m00"] != 0:
                center_x = int(moments["m10"] / moments["m00"])
                center_y = int(moments["m01"] / moments["m00"])
            else:
                center_x, center_y = frame_rgb.shape[1] // 2, frame_rgb.shape[0] // 2

            x1 = max(0, center_x - target_width // 2)
            y1 = max(0, center_y - target_height // 2)
            # 保证裁剪区域不超过原图边界
            x1 = min(x1, frame_rgb.shape[1] - target_width)
            y1 = min(y1, frame_rgb.shape[0] - target_height)
            x2 = x1 + target_width
            y2 = y1 + target_height
            
            cropped_cv = frame_rgb[y1:y2, x1:x2]
            return cropped_cv
        except AttributeError:
            # OpenCV显著性模块不可用，回退到Pillow的熵裁剪或中心裁剪
            pil_img = Image.fromarray(frame_rgb)
            # 简单的中心裁剪
            width, height = pil_img.size
            left = (width - target_width) / 2
            top = (height - target_height) / 2
            right = (width + target_width) / 2
            bottom = (height + target_height) / 2
            cropped_pil = pil_img.crop((left, top, right, bottom))
            return np.array(cropped_pil)
        except Exception as e:
            # 其他异常，回退到中心裁剪
            # print(f"内容感知裁剪错误: {e}, 回退到中心裁剪")
            pil_img = Image.fromarray(frame_rgb)
            width, height = pil_img.size
            left = (width - target_width) / 2
            top = (height - target_height) / 2
            right = (width + target_width) / 2
            bottom = (height + target_height) / 2
            cropped_pil = pil_img.crop((left, top, right, bottom))
            return np.array(cropped_pil)

    @staticmethod
    def enhance_edges(frame_rgb, strength=0.5):
        """
        使用Pillow的ImageFilter增强边缘。
        strength: 0.0 (无效果) to 1.0+ (更强效果)
        """
        pil_img = Image.fromarray(frame_rgb)
        # UnsharpMaskFilter: radius, percent, threshold
        # percent 控制锐化程度，可以映射 strength
        sharpened_img = pil_img.filter(ImageFilter.UnsharpMask(radius=1, percent=int(strength * 150), threshold=3))
        # 或者简单的锐化
        # sharpened_img = pil_img.filter(ImageFilter.SHARPEN)
        return np.array(sharpened_img)

    @staticmethod
    def auto_adjust_brightness_contrast(frame_rgb, brightness_factor=1.0, contrast_factor=1.0):
        """
        使用Pillow调整亮度和对比度。
        brightness_factor: 1.0 表示原始亮度, <1 更暗, >1 更亮。
        contrast_factor: 1.0 表示原始对比度, <1 更低对比度, >1 更高对比度。
        """
        pil_img = Image.fromarray(frame_rgb)
        
        # 调整亮度
        if brightness_factor != 1.0:
            enhancer = ImageEnhance.Brightness(pil_img)
            pil_img = enhancer.enhance(brightness_factor)

        # 调整对比度
        if contrast_factor != 1.0:
            enhancer = ImageEnhance.Contrast(pil_img)
            pil_img = enhancer.enhance(contrast_factor)
            
        return np.array(pil_img)
    
    @staticmethod
    def create_preview_image(frame_rgb, target_size=(config.WECHAT_PREVIEW_WIDTH, config.WECHAT_PREVIEW_HEIGHT)):
        """
        从单帧创建预览图 (PNG格式)。
        """
        pil_img = Image.fromarray(frame_rgb)
        pil_img = pil_img.resize(target_size, Image.LANCZOS)
        return pil_img # 返回PIL Image对象，以便保存为PNG

    @staticmethod
    def create_thumbnail_image(frame_rgb, target_size=(config.WECHAT_THUMBNAIL_WIDTH, config.WECHAT_THUMBNAIL_HEIGHT)):
        """
        从单帧创建缩略图 (PNG格式)。
        """
        pil_img = Image.fromarray(frame_rgb)
        pil_img = pil_img.resize(target_size, Image.LANCZOS)
        return pil_img # 返回PIL Image对象，以便保存为PNG


# 示例用法 (用于测试)
if __name__ == '__main__':
    # 创建一个虚拟的彩色帧 (np.ndarray, RGB)
    dummy_frame1 = np.random.randint(0, 256, size=(100, 100, 3), dtype=np.uint8)
    dummy_frame2 = np.random.randint(0, 256, size=(120, 80, 3), dtype=np.uint8)
    for _ in range(30):
        dummy_frame1[:,:,0] = (_/30) * 255 # R通道渐变
        dummy_frame2[:,:,1] = (_/30) * 255 # G通道渐变
    
    dummy_frames = [dummy_frame1.copy() for _ in range(5)] + [dummy_frame2.copy() for _ in range(5)]
    for i in range(len(dummy_frames)):
        dummy_frames[i][:,:, i % 3] = np.clip(dummy_frames[i][:,:, i % 3] + np.random.randint(-20,20, dummy_frames[i][:,:,i%3].shape), 0, 255).astype(np.uint8)

    print("测试颜色调色板优化...")
    palette = ImageOptimizer.optimize_color_palette(dummy_frames, max_colors=64)
    if palette:
        print(f"成功生成调色板，包含 {len(palette.palette)//3} 种颜色 (模式: {palette.mode})")
        # print(palette.getpalette()[:15]) # 打印前几种颜色
        quantized_pil_frame = ImageOptimizer.apply_palette_to_frame(dummy_frames[0], palette)
        # quantized_pil_frame.show() # 在某些系统上可能无法显示
        print(f"应用调色板后的帧模式: {quantized_pil_frame.mode}")
        assert quantized_pil_frame.mode == 'P'
    else:
        print("调色板生成失败")

    print("\n测试内容感知裁剪...")
    # 创建一个有明显特征的帧
    feature_frame = np.zeros((200, 300, 3), dtype=np.uint8)
    feature_frame[50:150, 100:200, :] = [255, 0, 0] # 红色方块作为特征
    cropped_frame = ImageOptimizer.content_aware_crop(feature_frame, 100, 100)
    print(f"内容感知裁剪后的尺寸: {cropped_frame.shape}")
    assert cropped_frame.shape[:2] == (100,100)
    # 理想情况下，裁剪结果应包含红色方块的中心部分，但这依赖OpenCV显著性算法

    print("\n测试边缘增强...")
    enhanced_frame = ImageOptimizer.enhance_edges(dummy_frames[1], strength=0.8)
    print(f"边缘增强后的帧尺寸: {enhanced_frame.shape}")
    assert enhanced_frame.shape == dummy_frames[1].shape
    # Image.fromarray(enhanced_frame).show()

    # Pillow的ImageEnhance需要在ImageOptimizer类的方法内使用，或者单独导入
    from PIL import ImageEnhance 
    print("\n测试亮度/对比度调整...")
    adjusted_frame = ImageOptimizer.auto_adjust_brightness_contrast(dummy_frames[2], brightness_factor=1.2, contrast_factor=1.1)
    print(f"亮度/对比度调整后的帧尺寸: {adjusted_frame.shape}")
    assert adjusted_frame.shape == dummy_frames[2].shape
    # Image.fromarray(adjusted_frame).show()

    print("\n测试预览图生成...")
    preview_img = ImageOptimizer.create_preview_image(dummy_frames[3])
    print(f"预览图尺寸: {preview_img.size}, 模式: {preview_img.mode}")
    assert preview_img.size == (config.WECHAT_PREVIEW_WIDTH, config.WECHAT_PREVIEW_HEIGHT)

    print("\n测试缩略图生成...")
    thumbnail_img = ImageOptimizer.create_thumbnail_image(dummy_frames[4])
    print(f"缩略图尺寸: {thumbnail_img.size}, 模式: {thumbnail_img.mode}")
    assert thumbnail_img.size == (config.WECHAT_THUMBNAIL_WIDTH, config.WECHAT_THUMBNAIL_HEIGHT)

    print("\nImageOptimizer 测试完成.") 