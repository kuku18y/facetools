import cv2
import numpy as np
from PyQt5.QtGui import QImage, QPixmap
from ..utils import config # 修正导入路径

class VideoProcessor:
    """处理视频解码、帧提取和基础图像操作"""

    def __init__(self, video_path):
        """
        初始化 VideoProcessor。

        Args:
            video_path (str): 视频文件路径。
        Raises:
            FileNotFoundError: 如果视频文件不存在。
            Exception: 如果视频文件无法打开或读取。
        """
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise FileNotFoundError(f"错误：无法打开视频文件 {video_path}")

        self.original_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.duration = self.total_frames / self.original_fps if self.original_fps > 0 else 0
        
        if self.total_frames == 0:
            self.cap.release()
            raise Exception(f"错误：视频文件 {video_path} 不包含任何帧或无法读取。")

        self.current_frame_index = 0
        self.frames_cache = {} # 用于缓存已提取的帧

    def extract_frames(self, target_fps=None, start_time=0, end_time=None):
        """
        从视频中提取指定时间段和帧率的帧。

        Args:
            target_fps (float, optional): 目标帧率。如果为None，则使用原始帧率。
            start_time (float, optional): 开始提取的时间点（秒）。默认为0。
            end_time (float, optional): 结束提取的时间点（秒）。如果为None，则提取到视频末尾。

        Returns:
            tuple: (list[np.ndarray], float): 提取的帧列表 (RGB格式) 和实际使用的帧率。
        """
        if not self.cap.isOpened():
            self.cap.open(self.video_path)
            if not self.cap.isOpened():
                raise Exception("无法重新打开视频文件进行帧提取。")

        actual_fps = target_fps if target_fps and target_fps > 0 else self.original_fps
        if actual_fps <= 0: actual_fps = config.DEFAULT_GIF_FPS # 防止除零或无效帧率

        frame_interval_time = 1.0 / actual_fps
        current_time = start_time
        
        start_frame_index = int(start_time * self.original_fps)
        end_frame_index = self.total_frames
        if end_time is not None:
            end_frame_index = min(self.total_frames, int(end_time * self.original_fps))

        extracted_frames = []
        last_extracted_frame_time = -1 # 确保第一帧能被提取

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_index)

        processed_frames_count = 0
        for i in range(start_frame_index, end_frame_index):
            # 检查是否到达当前提取点的时间
            # CAP_PROP_POS_MSEC 在某些视频格式下不准确，因此我们自行计算
            video_current_time = i / self.original_fps 
            if video_current_time < current_time:
                ret, frame = self.cap.read() # 读取并丢弃，直到到达正确的时间点
                if not ret: break
                continue

            ret, frame = self.cap.read()
            if not ret:
                break

            # 缓存机制可以考虑在这里加入，如果同一帧可能被多次请求
            # if i in self.frames_cache: 
            #    frame_rgb = self.frames_cache[i]
            # else: 
            #    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            #    self.frames_cache[i] = frame_rgb 
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            extracted_frames.append(frame_rgb)
            
            current_time += frame_interval_time
            processed_frames_count +=1

        # 如果目标帧率导致没有提取到任何帧，至少提取第一帧
        if not extracted_frames and self.total_frames > 0:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.cap.read()
            if ret:
                extracted_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            actual_fps = 1.0 / self.duration if self.duration > 0 else 1

        return extracted_frames, actual_fps

    def get_frame_at_time(self, time_sec):
        """获取指定时间的视频帧"""
        if not self.cap.isOpened(): return None
        frame_index = int(time_sec * self.original_fps)
        return self.get_frame_at_index(frame_index)

    def get_frame_at_index(self, frame_index):
        """获取指定索引的视频帧"""
        if not self.cap.isOpened(): return None
        if frame_index < 0 or frame_index >= self.total_frames:
            return None
        
        if frame_index in self.frames_cache:
            return self.frames_cache[frame_index]

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = self.cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.frames_cache[frame_index] = frame_rgb
            return frame_rgb
        return None

    def create_grid_positions(self, rows=config.DEFAULT_GRID_ROWS, cols=config.DEFAULT_GRID_COLS):
        """
        创建九宫格区块的初始位置信息。
        在修改后的版本中，每个区块初始时都指向完整的视频帧，
        而不是九宫格分割的小部分。用户可以通过缩放和拖动来选择想要的部分。
        """
        grid_positions = []
        for r_idx in range(rows):
            for c_idx in range(cols):
                grid_id = r_idx * cols + c_idx
                grid_positions.append({
                    'id': grid_id,
                    'x': 0,  # 初始时指向视频左上角
                    'y': 0,
                    'width': self.width,  # 使用完整视频宽度
                    'height': self.height,  # 使用完整视频高度
                    'source_x': 0,
                    'source_y': 0,
                    'source_width': self.width,
                    'source_height': self.height,
                    'scale': 1.0,  # 初始缩放比例
                    'offset_x': 0,  # 初始无偏移
                    'offset_y': 0,
                    'grid_row': r_idx,  # 记录网格位置，便于后续参考
                    'grid_col': c_idx
                })
        return grid_positions

    @staticmethod
    def crop_frame(frame, crop_rect, target_size=(config.WECHAT_STICKER_WIDTH, config.WECHAT_STICKER_HEIGHT)):
        """
        裁剪帧并缩放到目标尺寸。

        Args:
            frame (np.ndarray): 输入帧 (RGB)。
            crop_rect (dict): 裁剪区域，包含 'x', 'y', 'width', 'height' 或 'source_x', 'source_y', 'source_width', 'source_height'。
                               这些坐标是相对于原始帧的。
            target_size (tuple): (width, height) 目标输出尺寸，默认为240x240。

        Returns:
            np.ndarray: 裁剪并缩放后的帧 (RGB)，如果裁剪无效则返回None。
        """
        # 尝试从crop_rect中获取裁剪坐标，支持两种键名格式
        x = int(crop_rect.get('source_x', crop_rect.get('x', 0)))
        y = int(crop_rect.get('source_y', crop_rect.get('y', 0)))
        w = int(crop_rect.get('source_width', crop_rect.get('width', frame.shape[1])))
        h = int(crop_rect.get('source_height', crop_rect.get('height', frame.shape[0])))

        # 校验裁剪参数
        frame_height, frame_width = frame.shape[:2]
        
        # 确保坐标不超出边界
        x = max(0, min(x, frame_width - 1))
        y = max(0, min(y, frame_height - 1))
        
        # 确保宽高不超出边界
        w = max(1, min(w, frame_width - x))
        h = max(1, min(h, frame_height - y))
        
        # 裁剪
        try:
            cropped = frame[y:y+h, x:x+w]
            
            # 确保裁剪后的图像非空
            if cropped.size == 0:
                print(f"警告: 裁剪结果为空 (坐标: {x},{y},{w},{h}, 帧尺寸: {frame_width}x{frame_height})")
                # 如果裁剪失败，返回原始帧的中心区域
                center_x = frame_width // 2
                center_y = frame_height // 2
                half_size = min(frame_width, frame_height) // 2
                x1 = max(0, center_x - half_size)
                y1 = max(0, center_y - half_size)
                x2 = min(frame_width, center_x + half_size)
                y2 = min(frame_height, center_y + half_size)
                cropped = frame[y1:y2, x1:x2]
            
            # 缩放到目标尺寸
            resized = cv2.resize(cropped, target_size, interpolation=cv2.INTER_LANCZOS4)
            return resized
            
        except Exception as e:
            print(f"裁剪错误: {e} (坐标: {x},{y},{w},{h}, 帧尺寸: {frame_width}x{frame_height})")
            return None

    @staticmethod
    def frame_to_qpixmap(frame):
        """将OpenCV帧 (BGR或RGB) 转换为 QPixmap"""
        if frame is None: return QPixmap()
        
        # 检查 frame 是否已经是RGB
        if frame.ndim == 3 and frame.shape[2] == 3: # 通常是 (height, width, channels)
            height, width, channel = frame.shape
            bytes_per_line = 3 * width
            # QImage需要RGB格式
            # 如果输入是BGR，需要转换，但我们假设VideoProcessor内部处理为RGB
            q_img = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
            return QPixmap.fromImage(q_img)
        elif frame.ndim == 2: # 灰度图
            height, width = frame.shape
            bytes_per_line = width
            q_img = QImage(frame.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
            return QPixmap.fromImage(q_img)
        return QPixmap()

    def release(self):
        """释放视频捕捉对象"""
        if self.cap.isOpened():
            self.cap.release()
        self.frames_cache.clear()

    def __del__(self):
        self.release()

# 示例用法 (用于测试)
if __name__ == '__main__':
    # 创建一个虚拟的MP4文件用于测试 (需要ffmpeg安装在系统路径中)
    try:
        import subprocess
        dummy_video_path = "dummy_video.mp4"
        # 命令来生成一个1秒，10fps，320x240的视频
        # ffmpeg -f lavfi -i testsrc=duration=1:size=320x240:rate=10 -c:v libx264 -pix_fmt yuv420p dummy_video.mp4 -y
        subprocess.run([
            "ffmpeg", "-f", "lavfi", "-i", "testsrc=duration=1:size=320x240:rate=10", 
            "-c:v", "libx264", "-pix_fmt", "yuv420p", dummy_video_path, "-y"], check=True, capture_output=True)
        
        processor = VideoProcessor(dummy_video_path)
        print(f"视频信息: {processor.width}x{processor.height}, FPS: {processor.original_fps}, 时长: {processor.duration:.2f}s")

        frames, actual_fps = processor.extract_frames(target_fps=5)
        print(f"提取到 {len(frames)} 帧 @ {actual_fps:.2f} FPS")
        assert len(frames) == 5 or (len(frames) == 1 and processor.duration > 0) # 根据视频长度和fps可能只有1帧

        grid = processor.create_grid_positions()
        print(f"创建了 {len(grid)} 个网格单元")
        assert len(grid) == config.DEFAULT_GRID_ROWS * config.DEFAULT_GRID_COLS

        if frames:
            first_frame = frames[0]
            # 假设裁剪第一个网格单元
            cropped_frame = VideoProcessor.crop_frame(first_frame, grid[0])
            if cropped_frame is not None:
                print(f"裁剪后的帧尺寸: {cropped_frame.shape}")
                assert cropped_frame.shape[:2] == (config.WECHAT_STICKER_HEIGHT, config.WECHAT_STICKER_WIDTH)
            else:
                print("裁剪失败或裁剪区域无效")
        
        processor.release()
        print("VideoProcessor 测试完成.")
        subprocess.run(["rm", dummy_video_path]) # 清理

    except FileNotFoundError as e:
        print(f"测试失败: {e}. 请确保ffmpeg已安装并添加到PATH，或者提供一个有效的视频文件进行测试。")
    except subprocess.CalledProcessError as e:
        print(f"创建虚拟视频失败: {e}")
        print(e.stderr.decode())
    except Exception as e:
        print(f"VideoProcessor 测试时发生意外错误: {e}") 