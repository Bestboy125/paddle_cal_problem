import os
import sys
from paddleocr import draw_ocr
from PIL import Image
from paddle_ocr import perform_ocr_single_model
from associate_model import associate_ocr_with_questions
from associate_model import associate_hand_with_kongbai
from math_cal_model import math_problem_solve
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# --------------- 配置部分 ---------------

# 1. 定义多个模型的配置信息（此处仅示例 1 个，按需求可添加更多）
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

def main():
    # 定义图像路径
    hand_img_path = r"paddle_test_data\math\math_hand_1.jpg"
    problem_img_path = r"paddle_test_data\math\math_kongbai_1.jpg"
    original_img_path = r"paddle_test_data\math\math_origin_1.jpg"
    
    # 执行空白卷OCR
    problem_ocr_result = perform_ocr_single_model(problem_img_path, model_configs['model_v4_server_train']['rec_model_dir'])
    
    # 执行手写体OCR
    hand_ocr_result = perform_ocr_single_model(hand_img_path, model_configs['model_v4_server_train']['rec_model_dir'])
    
    # 打印手写体OCR结果
    print("手写体OCR结果:")
    print(hand_ocr_result)
    
    # 关联手写体答案 OCR 结果和空白卷 OCR 结果
    associations = associate_hand_with_kongbai.associate_hand_with_kongbai(hand_ocr_result, problem_ocr_result, threshold=10000)
    
    # 数学计算
    math_problem_solve.append_math_answers(associations)
    
    # 判断正误
    for item in associations:
        # 尝试将正确答案和OCR文本转换为相同类型以进行比较
        try:
            correct_answer = float(item['correct_answer'])
            ocr_text = float(item['ocr_text'])
            # 使用一定的容差范围处理小数精度误差
            tolerance = 1e-6
            item['correct'] = abs(correct_answer - ocr_text) < tolerance
        except ValueError:
            # 如果转换失败，默认标记为错误
            item['correct'] = False
    
    print("关联结果及正误标注:")
    print(associations)
    
    # 加载原始图像
    original_img = cv2.imread(original_img_path)
    if original_img is None:
        print(f"无法加载图像: {original_img_path}")
        return
    
    # 定义颜色和线条厚度
    box_color = (0, 255, 0)         # 绿色用于边界框
    correct_color = (0, 255, 0)     # 绿色 'V'
    incorrect_color = (0, 0, 255)   # 红色 'X'
    box_thickness = 2
    text_font = cv2.FONT_HERSHEY_SIMPLEX
    text_scale = 1
    text_thickness = 2
    text_color_correct = correct_color
    text_color_incorrect = incorrect_color
    
    for idx, item in enumerate(associations):
        question_bbox = item.get('question_bbox', [])
        ocr_bbox = item.get('ocr_bbox', [])
        
        # 打印调试信息
        print(f"关联项 {idx + 1}: question_bbox = {question_bbox}, ocr_bbox = {ocr_bbox}")
        
        # 验证边界框结构
        valid_question_bbox = (
            isinstance(question_bbox, list) and 
            len(question_bbox) == 4 and 
            all(isinstance(point, (list, tuple)) and len(point) == 2 for point in question_bbox)
        )
        valid_ocr_bbox = (
            isinstance(ocr_bbox, list) and 
            len(ocr_bbox) == 4 and 
            all(isinstance(point, (list, tuple)) and len(point) == 2 for point in ocr_bbox)
        )
        
        if not (valid_question_bbox and valid_ocr_bbox):
            print(f"警告: 关联项 {idx + 1} 的边界框结构不正确。")
            continue  # 跳过无效的边界框
        
        # 合并两个边界框的所有点
        combined_points = question_bbox + ocr_bbox  # 8个点
        combined_points_np = np.array(combined_points, dtype=np.int32)
        
        # 计算最小外接矩形的坐标
        x, y, w, h = cv2.boundingRect(combined_points_np)
        top_left = (x, y)
        bottom_right = (x + w, y + h)
        
        # 绘制最小外接矩形
        cv2.rectangle(original_img, top_left, bottom_right, color=box_color, thickness=box_thickness)
        
        # 计算标注位置（矩形的右上角附近）
        mark_x = x + w
        mark_y = y - 10  # 向上偏移10个像素
        
        # 防止标注位置超出图像边界
        mark_x = min(mark_x, original_img.shape[1] - 20)  # 保留空间放置符号
        mark_y = max(mark_y, 20)  # 确保不会超出顶部，并留出空间
        
        # 选择标注符号和颜色
        if item.get('correct', False):
            # 绘制 'V' 作为对号的替代
            cv2.putText(
                original_img, 
                'V', 
                (mark_x, mark_y), 
                text_font, 
                text_scale, 
                text_color_correct, 
                text_thickness, 
                cv2.LINE_AA
            )
        else:
            # 绘制 'X' 作为错号
            cv2.putText(
                original_img, 
                'X', 
                (mark_x, mark_y), 
                text_font, 
                text_scale, 
                text_color_incorrect, 
                text_thickness, 
                cv2.LINE_AA
            )
    
    # 显示标注后的图像
    # 注意：在某些环境（如服务器）中，cv2.imshow 可能无法正常工作
    # 可以选择仅保存图像
    try:
        cv2.imshow('Annotated Image', original_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except:
        print("无法显示图像窗口，正在保存标注后的图像。")
    
    # 保存标注后的图像
    annotated_img_path = r"paddle_test_data\math\math_origin_1_annotated.jpg"
    cv2.imwrite(annotated_img_path, original_img)
    print(f"标注后的图像已保存至: {annotated_img_path}")

if __name__ == "__main__":
    main()
