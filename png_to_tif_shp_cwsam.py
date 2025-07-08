import argparse
import os
import yaml
import torch
from PIL import Image
import numpy as np
from torchvision import transforms
from tqdm import tqdm

import models
# from test import de_normalize, onehot_to_mask

# 添加新的导入
import rasterio
import geopandas as gpd
from rasterio import features
from rasterio.transform import from_origin
import shapely.geometry as geometry

def png_to_tif(png_path, output_tif_path):
    """
    将PNG图片转换为TIF图片，使用WGS84坐标系和随机原点坐标
    
    参数:
        png_path: PNG图片路径
        output_tif_path: 输出TIF图片路径
    """
    # 读取PNG图片但不调整大小
    img = Image.open(png_path)
    width, height = img.size
    img_array = np.array(img)
    
    # 如果是灰度图，转换为RGB
    if len(img_array.shape) == 2:
        img_array = np.stack([img_array, img_array, img_array], axis=0)
    else:
        # 将HWC转换为CHW格式
        img_array = img_array.transpose(2, 0, 1)
        
    # 确保有3个通道(RGB)
    if img_array.shape[0] > 3:
        img_array = img_array[:3]
        
    # 生成随机WGS84坐标作为左上角原点 (经度在-180到180之间，纬度在-90到90之间)
    lon = np.random.uniform(-180, 180)
    lat = np.random.uniform(-90, 90)
    lon=117
    lat=31
    # 设置像素分辨率 (根据WGS84坐标系单位为度)
    pixel_size = 5.364418e-06  # 约0.6米在赤道处
    
    # 创建坐标变换矩阵
    transform = from_origin(lon, lat, pixel_size, pixel_size)
    
    # 输出坐标系信息
    print(f"TIF坐标系信息:")
    print(f"  左上角坐标: 经度={lon}, 纬度={lat}")
    print(f"  像元大小: {pixel_size}度 (约{pixel_size*111320}米在赤道处)")
    print(f"  图像尺寸: {width}x{height}像素")
    
    # 写入TIF文件
    with rasterio.open(
        output_tif_path,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=img_array.shape[0],
        dtype=img_array.dtype,
        crs='EPSG:4326',  # 使用标准EPSG:4326代码表示WGS84坐标系
        transform=transform,
        compress='lzw'  # 使用LZW无损压缩
    ) as dst:
        dst.write(img_array)
    
    return output_tif_path, transform, 'EPSG:4326'

def prepare_image(image_path, input_size=1024):
    """准备输入图像，支持tif和png格式"""
    transform = transforms.Compose([
        transforms.Resize((input_size, input_size), interpolation=Image.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # 检查文件扩展名
    ext = os.path.splitext(image_path)[1].lower()
    
    if ext in ['.png', '.jpg', '.jpeg']:
        # 如果是PNG或其他普通图像格式，先转换为TIF
        tif_path = os.path.splitext(image_path)[0] + '.tif'
        image_path, transform_matrix, crs = png_to_tif(image_path, tif_path)
        print(f"图像已转换为TIF格式: {tif_path}")
    
    # 使用rasterio读取tif文件
    with rasterio.open(image_path) as src:
        image = src.read()  # 读取所有波段
        # 如果是多波段，只取RGB波段
        if image.shape[0] > 3:
            image = image[:3]
        # 转换为PIL图像格式
        image = Image.fromarray(np.transpose(image, (1, 2, 0)).astype(np.uint8))
        image_tensor = transform(image)
        # 保存地理信息用于后续转换
        transform_matrix = src.transform
        crs = src.crs
    
    return image_tensor.unsqueeze(0), transform_matrix, crs

def color_to_list(
    mask, palette=[[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [0, 0, 0]]
):
    """
    Converts a segmentation mask (H, W, C) to (H, W, K) where the last dim is a one
    hot encoding vector, C is usually 1 or 3, and K is the number of class.
    """
    # mask = mask.permute(1,2,0)
    mask = mask * 255
    mask.int()
    semantic_map = np.zeros([1024, 1024], dtype=np.int8)
    for i, colour in enumerate(palette):
        equality = np.equal(mask, colour)
        class_map = np.all(equality, axis=-1)
        semantic_map += class_map * int(i)


def onehot_to_mask(
    mask, palette=[[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [0, 0, 0]]
):
    """
    Converts a mask (H, W, K) to (H, W, C)
    """
    mask = mask.permute(1, 2, 0).numpy()
    x = np.argmax(mask, axis=-1)
    # x = np.squeeze(mask, axis=-1)
    colour_codes = np.array(palette)
    x = np.uint8(colour_codes[x.astype(np.uint8)])
    # x=x.permute(2,0,1)
    # x=x.numpy()
    # x = np.around
    return x


def onehot_to_index_label(mask):
    """
    Converts a mask (H, W, K) to (H, W, C)
    """
    mask = mask.permute(1, 2, 0).numpy()
    x = np.argmax(mask, axis=-1)
    # colour_codes = np.array(palette)
    # x = np.uint8(colour_codes[x.astype(np.uint8)])*255
    # x=x.permute(2,0,1)
    # x=x.numpy()
    # x = np.around
    return x


def de_normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    mean = torch.as_tensor(mean)
    std = torch.as_tensor(std)
    if mean.ndim == 1:
        mean = mean.view(-1, 1, 1)
    if std.ndim == 1:
        std = std.view(-1, 1, 1)
    image = image * std + mean  # 由image=(x-mean)/std可知，x=image*std+mean
    image = image.numpy().transpose(1, 2, 0)  # 从tensor转为numpy,并由（C,H,W）变为（H,W,C）
    image = np.around(image * 255)  ##对数据恢复并进行取整
    image = np.array(image, dtype=np.uint8)  # 矩阵元素类型由浮点数转为整数
    return image
def mask_to_shapefile(mask, transform_matrix, crs, class_names, output_path):
    """将掩码转换为shapefile"""
    shapes = []
    properties = []
    
    # 对每个类别进行处理
    for class_idx in range(len(class_names)):
        # 提取当前类别的掩码
        binary_mask = (mask == class_idx).astype(np.uint8)
        
        # 使用rasterio.features提取多边形
        for geom, value in features.shapes(binary_mask, transform=transform_matrix):
            if value == 1:  # 只处理前景区域
                # 将GeoJSON格式的几何对象转换为shapely几何对象
                polygon = geometry.shape(geom)
                if not polygon.is_empty:  # 确保多边形有效
                    shapes.append(polygon)
                    # 确保类别名称是UTF-8编码的字符串
                    class_name = class_names[class_idx]
                    if isinstance(class_name, bytes):
                        class_name = class_name.decode('utf-8')
                    properties.append({'classname': class_name})
    
    if not shapes:  # 如果没有有效的形状，添加一个空的多边形
        print(f"Warning: No valid shapes found in the mask")
        return
    
    # 创建GeoDataFrame
    gdf = gpd.GeoDataFrame(
        properties,
        geometry=shapes,
        crs=crs
    )
    
    # 去除文件扩展名
    output_base = os.path.splitext(output_path)[0]
    
    # 保存为shapefile，指定编码为UTF-8
    gdf.to_file(output_base, encoding='utf-8')
    
    # 创建CPG文件指定字符编码
    with open(output_base + '.cpg', 'w') as f:
        f.write('UTF-8')

def prepare_image_tiles(image_path, tile_size=1024, overlap=128):
    """使用滑动窗口准备大尺寸图像的分块，支持tif和png格式"""
    transform_to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    with rasterio.open(image_path) as src:
        image_data = src.read()  # 读取所有波段，格式为CHW
        if image_data.shape[0] > 3: # 如果通道数多于3，只取RGB
            image_data = image_data[:3, :, :]
        
        img_height, img_width = src.height, src.width # 从rasterio元数据获取高宽
        transform_matrix = src.transform
        crs = src.crs

        # 检查是否为普通图像格式，并且地理参考信息可能缺失或为默认
        file_ext = os.path.splitext(image_path)[1].lower()
        is_common_image_format = file_ext in ['.png', '.jpg', '.jpeg']

        # 如果是常见图像格式，且CRS为空或变换矩阵是单位矩阵（表明可能是像素坐标）
        # 则应用默认的地理参考信息
        # transform_matrix 可能为 None，所以需要检查
        is_identity_transform = False
        if transform_matrix:
            is_identity_transform = transform_matrix.is_identity

        if is_common_image_format and (crs is None or is_identity_transform):
            print(f"警告: 图像 {image_path} 可能没有有效的地理参考信息或使用的是像素坐标。将应用默认地理参考。")
            # 使用与 png_to_tif 中类似的默认值
            lon = 117  # 默认经度
            lat = 31   # 默认纬度
            pixel_size = 5.364418e-06  # 默认像素大小 (度)
            
            transform_matrix = from_origin(lon, lat, pixel_size, pixel_size)
            crs = 'EPSG:4326'  # 默认为WGS84
            print(f"  应用默认地理参考: 左上角经度={lon}, 纬度={lat}, 像元大小={pixel_size}度, CRS=EPSG:4326")

        tiles = []
        positions = []
        
        stride = tile_size - overlap
        
        for y_coord in range(0, img_height, stride): # 变量名更改以避免与模块名冲突
            for x_coord in range(0, img_width, stride):
                end_y = min(y_coord + tile_size, img_height)
                end_x = min(x_coord + tile_size, img_width)
                start_y = max(0, end_y - tile_size)
                start_x = max(0, end_x - tile_size)
                
                # 从 image_data (CHW格式) 提取块
                tile_array_chw = image_data[:, start_y:end_y, start_x:end_x]
                
                if tile_array_chw.shape[1] < tile_size or tile_array_chw.shape[2] < tile_size:
                    padded_tile_chw = np.zeros((image_data.shape[0], tile_size, tile_size), dtype=image_data.dtype)
                    padded_tile_chw[:, :tile_array_chw.shape[1], :tile_array_chw.shape[2]] = tile_array_chw
                    tile_array_chw = padded_tile_chw
                
                # 转换为PIL图像 (需要 HWC 格式) 并进行预处理
                tile_pil = Image.fromarray(np.transpose(tile_array_chw, (1, 2, 0)).astype(np.uint8))
                tile_tensor = transform_to_tensor(tile_pil) # 使用更新后的变量名
                
                tiles.append(tile_tensor)
                positions.append((start_x, start_y, end_x, end_y))
        
        return tiles, positions, (img_height, img_width), transform_matrix, crs

def merge_predictions_with_confidence(predictions, positions, original_size, model_outputs):
    """使用置信度合并分块预测结果"""
    height, width = original_size
    final_mask = np.zeros((height, width), dtype=np.int64)
    confidence_map = np.zeros((height, width), dtype=np.float32)
    
    for pred, conf_matrix, (start_x, start_y, end_x, end_y) in zip(predictions, model_outputs, positions):
        pred = pred.numpy()
        # 获取每个像素的最大置信度
        confidence = torch.max(torch.softmax(conf_matrix, dim=0), dim=0)[0].cpu().numpy()
        
        # 更新预测结果，只在置信度更高的位置更新
        mask = confidence > confidence_map[start_y:end_y, start_x:end_x]
        final_mask[start_y:end_y, start_x:end_x][mask] = pred[mask]
        confidence_map[start_y:end_y, start_x:end_x][mask] = confidence[mask]
    
    return final_mask

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', 
                       default='/mnt/fanfq/data/fan/cwsam_1.5b/configs/XinTong/XinTong_sam_vit_h_moe_3b.yaml', 
                       help='配置文件路径')
    parser.add_argument('--model', 
                       default='/mnt/fanfq/data/fan/cwsam/save/XinTong_sam_vit_h_moe_3b/model_epoch_121.pth',
                       help='模型权重路径')
    parser.add_argument('--input_dir', 
                       default='/mnt/fanfq/project/data/uav_seg/1080-1-rgb', 
                       help='输入图像文件夹')
    parser.add_argument('--output_dir', 
                       default='/mnt/fanfq/project/data/uav_seg/1080-1-rgb/shp_moe_121',
                       help='输出结果文件夹')
    parser.add_argument('--device', 
                       default='cuda:0',
                       help='选择推理设备 (cuda/cpu)')
    parser.add_argument('--skip_png_to_tif_conversion', action='store_true',
                       help='如果设置此项，则不将PNG/JPG/JPEG转换为TIF格式，直接处理原始文件')
    args = parser.parse_args()

    # 加载配置
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    # 获取调色板和类别数
    color_palette = config['test_dataset']['dataset']['args']['palette']
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    # 创建TIF输出目录
    tif_output_dir = os.path.join(args.output_dir, 'tif')
    os.makedirs(tif_output_dir, exist_ok=True)

    # 加载模型
    device = torch.device(args.device if torch.cuda.is_available() and 'cuda' in args.device else 'cpu')
    model = models.make(config['model']).to(device)
    checkpoint = torch.load(args.model, map_location=device)
    model.load_state_dict(checkpoint, strict=True)
    model.eval()

    # 获取所有tif和png文件
    image_files = [f for f in os.listdir(args.input_dir) 
                  if f.lower().endswith(('.tif', '.tiff', '.png', '.jpg', '.jpeg'))]

    # 仅在使用CUDA时启用优化
    if 'cuda' in device.type:
        torch.backends.cudnn.benchmark = True
    
    # 根据设备类型调整批处理大小
    BATCH_SIZE = 1 if device.type == 'cuda' else 1  # CPU时使用较小的批量
    
    # 获取类别名称
    class_names = config['test_dataset']['dataset']['args']['classes']
    
    # 处理每张图像
    for image_file in tqdm(image_files, desc="处理图像"):
        original_image_path = os.path.join(args.input_dir, image_file)
        file_name_no_ext = os.path.splitext(image_file)[0]
        original_file_ext = os.path.splitext(image_file)[1].lower()

        path_for_tiles = original_image_path  # 默认使用原始路径
        
        # 如果是PNG/JPG/JPEG且用户未指定跳过转换，则执行转换
        if original_file_ext in ['.png', '.jpg', '.jpeg'] and not args.skip_png_to_tif_conversion:
            tif_path = os.path.join(tif_output_dir, f'{file_name_no_ext}.tif')
            # png_to_tif 会返回转换后的路径、变换矩阵和CRS
            # 我们在这里主要需要转换后的路径，地理信息会嵌入TIF中
            converted_tif_path, _, _ = png_to_tif(original_image_path, tif_path)
            path_for_tiles = converted_tif_path  # 更新路径为转换后的TIF文件
            print(f"已将 {image_file} 转换为TIF格式: {path_for_tiles}")
        elif original_file_ext in ['.png', '.jpg', '.jpeg'] and args.skip_png_to_tif_conversion:
            print(f"跳过PNG到TIF的转换，直接处理: {original_image_path}")
        
        # 获取图像分块
        tiles, positions, original_size, transform_matrix, crs = prepare_image_tiles(
            path_for_tiles, tile_size=1024, overlap=256
        )
        
        predictions = []
        model_outputs = []  # 存储模型输出的概率分布
        
        # 分批处理图像块
        for i in range(0, len(tiles), BATCH_SIZE):
            batch_tiles = tiles[i:i + BATCH_SIZE]
            batch_tensor = torch.stack(batch_tiles).to(device)
            
            with torch.no_grad():
                batch_outputs = model.infer(batch_tensor)
                batch_preds = torch.argmax(batch_outputs, dim=1).cpu()
                predictions.extend([pred for pred in batch_preds])
                model_outputs.extend([output.cpu() for output in batch_outputs])
            
            # 清理显存
            torch.cuda.empty_cache()
        
        # 使用置信度合并预测结果
        final_mask = merge_predictions_with_confidence(
            predictions, positions, original_size, model_outputs
        )
        
        # 保存为shapefile
        output_shp = os.path.join(args.output_dir, f'{file_name_no_ext}.shp')
        mask_to_shapefile(final_mask, transform_matrix, crs, class_names, output_shp)
        print(f"生成Shapefile: {output_shp}")

if __name__ == '__main__':
    main()
