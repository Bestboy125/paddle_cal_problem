import os
from paddleocr import PaddleOCR, draw_ocr
from PIL import Image

def perform_ocr_single_model(image_path, rec_model_dir):
    """
    使用单一识别模型对图片执行 OCR。

    参数:
        image_path (str): 图片路径
        rec_model_dir (str): 模型目录

    返回:
        dict: 包含坐标框、检测文本、置信度的字典
              {
                  "boxes": [[x1, y1], [x2, y2], ...],
                  "texts": ["text1", "text2", ...],
                  "scores": [0.99, 0.85, ...]
              }
    """
    # 初始化 PaddleOCR
    ocr = PaddleOCR(
        use_angle_cls=True,
        lang='ch',
        rec_model_dir=rec_model_dir
    )

    # 执行 OCR
    result = ocr.ocr(image_path, cls=True)

    # 若没有检测到文本，返回空结果
    if not result:
        print(f"[Warning] 模型对图片 [{image_path}] 没有检测到任何文本。")
        return {
            "boxes": [],
            "texts": [],
            "scores": []
        }

    boxes = []
    texts = []
    scores = []

    # PaddleOCR 默认输出格式是一张图片可能包含多行结果，每行一个 [ [x1, y1], [x2, y2], ... ], ["文本", 置信度 ]
    # 这里将所有行结果依次添加到对应列表中
    for res in result:
        if res is None:
            continue
        for line in res:
            boxes.append(line[0])      # 坐标框
            texts.append(line[1][0])   # 识别文本
            scores.append(line[1][1])  # 置信度

    return {
        "boxes": boxes,
        "texts": texts,
        "scores": scores
    }
