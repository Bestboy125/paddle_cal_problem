import math

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

def associate_hand_with_kongbai(ocr_result, question_boxes, threshold=50):
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
    - question_boxes: {
          "boxes": [  # 每个元素都是4个点: [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
              [...],
              ...
          ],
          "texts": ["question1", "question2", ...],
          "scores": [0.99, 0.95, ...]
      }
    - threshold: 距离阈值，只有在该阈值内的题干才会进入候选进行优先级比较

    返回：
    [
        {
            'question_text': '题干文本',
            'question_bbox': [x1, y1, x2, y2, x3, y3, x4, y4],
            'ocr_text': 'OCR识别文本',
            'ocr_bbox': [[x1, y1], [x2, y2], [x3, y3], [x4, y4]],
            'association_type': 'question_association'
        },
        ...
    ]
    """
    associations = []

    # 提取题干框信息
    question_boxes_list = question_boxes.get("boxes", [])
    question_texts = question_boxes.get("texts", [])
    question_scores = question_boxes.get("scores", [])

    # 遍历每个 OCR 结果
    for i, ocr_box_points in enumerate(ocr_result.get("boxes", [])):
        ocr_text = ocr_result.get("texts", [])[i]
        # ocr_score = ocr_result.get("scores", [])[i]  # 如果需要用到置信度，可额外处理

        # 计算这个 OCR 框的中心
        ocr_center = get_polygon_center(ocr_box_points)

        # 找出与该 OCR 框距离 <= threshold 的所有题干候选
        candidates = []
        for q_idx, q_box_points in enumerate(question_boxes_list):
            q_text = question_texts[q_idx]
            q_score = question_scores[q_idx]
            q_center = get_polygon_center(q_box_points)
            dist = euclidean_distance(ocr_center, q_center)
            if dist <= threshold:
                candidates.append({
                    'text': q_text,
                    'bbox': q_box_points,
                    'score': q_score,
                    'distance': dist
                })

        # 若没有题干进入阈值范围，按业务逻辑看是否跳过
        if not candidates:
            # 你可以选择将这些 OCR 结果标记为未关联
            associations.append({
                'question_text': None,
                'question_bbox': None,
                'ocr_text': ocr_text,
                'ocr_bbox': ocr_box_points,
                'association_type': 'unassociated'
            })
            continue

        # 按照距离 + 上 + 左进行排序
        sorted_candidates = sorted(candidates, key=lambda q: (q['distance'], q['bbox'][1][1], q['bbox'][0][0]))
        best_match = sorted_candidates[0]

        associations.append({
            'question_text': best_match['text'],
            'question_bbox': best_match['bbox'],
            'ocr_text': ocr_text,
            'ocr_bbox': ocr_box_points,
            'association_type': 'question_association'
        })

    return associations
