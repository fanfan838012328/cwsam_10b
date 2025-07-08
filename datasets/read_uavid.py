from PIL import Image
import numpy as np

def analyze_image_colors(image_path):
    # 打开图像
    image = Image.open(image_path)
    
    # 将图像转换为RGB模式，以防它不是
    image = image.convert("RGB")
    
    # 将图像数据转换为numpy数组
    image_data = np.array(image)
    
    # 获取图像中所有独特的颜色值，并计算每种颜色的数量
    unique_colors, counts = np.unique(image_data.reshape(-1, 3), axis=0, return_counts=True)
    
    # 打印每种颜色及其数量
    print(f"在图像中找到 {len(unique_colors)} 个独特的颜色。")
    for color in unique_colors:
        print(f"颜色RGB值: {color}")

if __name__ == "__main__":
    image_path = '/public/home/hatmore/dataset/sam_uavid/train/Labels/CGDZ_1_offset-0.png'  # 替换为你的图像文件路径
    analyze_image_colors(image_path)