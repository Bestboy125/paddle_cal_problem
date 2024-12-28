import os
from paddleocr import PaddleOCR, draw_ocr
from PIL import Image

# --------------- 配置部分 ---------------

# 1. 定义测试集目录和结果存放目录
test_dir = "paddle_test_data/jisuanti"          # 你的测试图片所在文件夹
results_dir = "paddle_ocr_result/jisuanti"                # 用于存放结果的文件夹

# 2. 定义测试图片数量上限
max_images = 100  # 设置你想要处理的最大图片数量

# 3. 创建结果目录（如果不存在则自动创建）
os.makedirs(results_dir, exist_ok=True)

# 4. 定义多个模型的配置信息（此处仅示例 1 个，按需求可添加更多）
model_configs = {
    "model_v4_server_train": {
        "rec_model_dir": "paddle_model/ch_PP-OCRv4_rec_server_train",
        "description": "高精度训练模型"
    }
    # 你可以取消注释并添加更多模型配置
    # "model_v4_server_infer": {
    #     "rec_model_dir": "paddle_model/ch_PP-OCRv4_rec_server_infer", 
    #     "description": "高精度推理模型"
    # },
    # "model_v4_train": {
    #     "rec_model_dir": "paddle_model/ch_PP-OCRv4_rec_train", 
    #     "description": "轻量化训练模型"
    # },
    # "model_v4_infer": {
    #     "rec_model_dir": "paddle_model/ch_PP-OCRv4_rec_infer", 
    #     "description": "轻量化推理模型"
    # }
}

# --------------- 主体部分 ---------------

def get_image_files(directory, extensions=('.png', '.jpg', '.jpeg', '.bmp')):
    """
    获取指定目录下所有符合扩展名的图片文件列表。
    """
    return [f for f in os.listdir(directory) if f.lower().endswith(extensions)]

def main():
    # 获取所有符合条件的图片文件
    all_images = get_image_files(test_dir)
    
    # 应用数量上限
    selected_images = all_images[:max_images]
    
    print(f"共找到 {len(all_images)} 张图片，开始处理前 {len(selected_images)} 张图片。")

    # 遍历选定的图片
    for idx, img_name in enumerate(selected_images, start=1):
        img_path = os.path.join(test_dir, img_name)
        print(f"\n=============================")
        print(f"处理图片 {idx}/{len(selected_images)}：{img_path}")
        
        # 针对每个模型进行OCR识别
        for model_name, config in model_configs.items():
            print(f"\n-- 使用模型: {model_name} ({config['description']}) --")
            try:
                # 初始化 PaddleOCR
                ocr = PaddleOCR(
                    use_angle_cls=True, 
                    lang='ch', 
                    rec_model_dir=config["rec_model_dir"]
                )
                
                # 执行 OCR
                result = ocr.ocr(img_path, cls=True)
                
                # 检查 result 是否为空
                if not result:
                    print(f"[Warning] 模型 [{model_name}] 对图片 [{img_name}] 没有检测到任何文本。")
                    # 创建一个空的 .txt 文件以记录无结果
                    txt_file_name = f"{os.path.splitext(img_name)[0]}_{model_name}.txt"
                    txt_save_path = os.path.join(results_dir, txt_file_name)
                    with open(txt_save_path, 'w', encoding='utf-8') as f:
                        f.write("")  # 写入空内容
                    print(f"已创建空的文本文件：{txt_save_path}")
                    continue  # 跳过当前模型的后续步骤
                
                # 打印识别结果到控制台（可选）
                for res_idx, res in enumerate(result):
                    if res is None:
                        print(f"[Warning] 模型 [{model_name}] 返回的结果中第 {res_idx} 项为 None。")
                        continue
                    for line in res:
                        print(line)
                
                # 将识别文本结果保存到 .txt 文件
                txt_file_name = f"{os.path.splitext(img_name)[0]}_{model_name}.txt"
                txt_save_path = os.path.join(results_dir, txt_file_name)
                with open(txt_save_path, 'w', encoding='utf-8') as f:
                    for res in result:
                        if res is None:
                            continue  # 跳过 None 项
                        for line in res:
                            text = line[1][0]  # 识别到的文字
                            f.write(text + '\n')
                print(f"文本结果已保存：{txt_save_path}")
                
                # 绘制 OCR 结果并将图片保存到结果目录
                result_img_name = f"{os.path.splitext(img_name)[0]}_{model_name}_result.jpg"
                result_img_path = os.path.join(results_dir, result_img_name)
                
                # 检查是否有可绘制的数据
                has_draw_data = any(res for res in result if res)
                
                if has_draw_data:
                    # 假设只绘制第一组结果进行可视化
                    draw_data = result[0]
                    if draw_data is None:
                        print(f"[Warning] 模型 [{model_name}] 的第一组结果为 None，跳过绘制。")
                        continue
                    image = Image.open(img_path).convert('RGB')
                    # 提取绘制所需的数据
                    boxes = [line[0] for line in draw_data]
                    txts = [line[1][0] for line in draw_data]
                    scores = [line[1][1] for line in draw_data]
                    
                    # 检查绘制数据是否完整
                    if not boxes or not txts or not scores:
                        print(f"[Warning] 模型 [{model_name}] 的绘制数据不完整，跳过绘制。")
                    else:
                        try:
                            im_show = draw_ocr(
                                image, 
                                boxes, 
                                txts, 
                                scores, 
                                font_path='doc/fonts/simfang.ttf'  # 字体路径可换成自己本地中文字体
                            )
                            im_show = Image.fromarray(im_show)
                            im_show.save(result_img_path)
                            print(f"可视化结果已保存：{result_img_path}")
                        except Exception as e:
                            print(f"[Error] 绘制 OCR 结果时发生错误：{e}")
                else:
                    print(f"[Info] 模型 [{model_name}] 对图片 [{img_name}] 没有可绘制的文本框。")
                
                # 完成当前模型的处理
                print(f"完成模型[{model_name}]识别：{img_name}\n文本结果已保存：{txt_save_path}")
            
            except Exception as e:
                print(f"[Error] 模型 [{model_name}] 处理图片 [{img_name}] 时发生异常：{e}")
        
        print(f"图片 {img_name} 全部模型处理完成。")
    
    print(f"\n所有选定的图片（{len(selected_images)} 张）已处理完成。")

if __name__ == "__main__":
    main()
