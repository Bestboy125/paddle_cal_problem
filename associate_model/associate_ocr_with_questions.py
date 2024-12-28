import math

def get_center(bbox):
    """
    根据四点坐标计算题干框的中心点。
    bbox: [x1, y1, x2, y2, x3, y3, x4, y4]
    返回值: (cx, cy)
    """
    x_coords = [bbox[0], bbox[2], bbox[4], bbox[6]]
    y_coords = [bbox[1], bbox[3], bbox[5], bbox[7]]
    cx = sum(x_coords) / 4.0
    cy = sum(y_coords) / 4.0
    return (cx, cy)

def get_polygon_center(polygon):
    """
    polygon: [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]  (4 个顶点坐标)
    返回值: (cx, cy)
    """
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    cx = sum(xs) / 4.0
    cy = sum(ys) / 4.0
    return (cx, cy)



def euclidean_distance(p1, p2):
    """
    计算两点间的欧几里得距离。
    p1, p2: (x, y)
    """
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def associate_ocr_with_questions(ocr_result, question_boxes, threshold=50):
    """
    将单模型 OCR 结果与题干坐标框按照“距离阈值 + 最近 + 优先上 + 优先左”策略进行关联。

    参数：
    - ocr_result: {
          "boxes": [  # 每个元素都是4个点: [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
              [...],
              ...
          ],
          "texts": ["text1", "text2", ...],
          "scores": [0.99, 0.85, ...]
      }
    - question_boxes: [
          {
              'bbox': [x1, y1, x2, y2, x3, y3, x4, y4],
              'text': '题干文本'
          },
          ...
      ]
    - threshold: 距离阈值，只有在该阈值内的题干才会进入候选进行优先级比较

    返回：
    [
        {
            'question_text': 题干文本,
            'question_bbox': 题干 bbox,
            'ocr_text': OCR 识别文本,
            'ocr_bbox': OCR 四点坐标
        },
        ...
    ]
    """
    associations = []

    # OCR 结果有多少个文本，就遍历多少次
    for i, box_points in enumerate(ocr_result["boxes"]):
        ocr_text = ocr_result["texts"][i]
        # ocr_score = ocr_result["scores"][i]  # 如果需要用到置信度，可额外处理

        # 计算这个 OCR 框的中心
        ocr_center = get_polygon_center(box_points)

        # 1. 找出与该 OCR 框距离 <= threshold 的所有题干候选
        candidates = []
        for q_item in question_boxes:
            q_center = get_center(q_item['bbox'])
            dist = euclidean_distance(ocr_center, q_center)
            if dist <= threshold:
                candidates.append(q_item)

        # 若没有题干进入阈值范围，按业务逻辑看是否跳过
        if not candidates:
            continue

        # 2. 按照距离 + 上 + 左进行排序
        def sort_key(q_item):
            q_center = get_center(q_item['bbox'])
            dist = euclidean_distance(ocr_center, q_center)
            return (
                dist,         # 距离越小越好
                q_center[1],  # y 坐标越小越“上”
                q_center[0],  # x 坐标越小越“左”
            )

        sorted_candidates = sorted(candidates, key=sort_key)
        best_match = sorted_candidates[0]

        associations.append({
            'question_text': best_match.get('text', ''),
            'question_bbox': best_match['bbox'],
            'ocr_text': ocr_text,
            'ocr_bbox': box_points
        })

    return associations
