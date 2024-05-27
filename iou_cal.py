def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.
    
    Parameters:
    box1 (list): Coordinates of the first bounding box [topleft_x, topleft_y, width, height].
    box2 (list): Coordinates of the second bounding box [topleft_x, topleft_y, width, height].
    
    Returns:
    float: Intersection over Union (IoU) ratio.
    """
    # Extract coordinates of the first bounding box
    x1_tl, y1_tl, w1, h1 = box1
    x1_br, y1_br = x1_tl + w1, y1_tl + h1
    
    # Extract coordinates of the second bounding box
    x2_tl, y2_tl, w2, h2 = box2
    x2_br, y2_br = x2_tl + w2, y2_tl + h2
    
    # Calculate coordinates of the intersection rectangle
    x_tl = max(x1_tl, x2_tl)
    y_tl = max(y1_tl, y2_tl)
    x_br = min(x1_br, x2_br)
    y_br = min(y1_br, y2_br)
    
    # Calculate area of intersection rectangle
    intersection_area = max(0, x_br - x_tl + 1) * max(0, y_br - y_tl + 1)
    
    # Calculate area of both bounding boxes
    box1_area = (w1 + 1) * (h1 + 1)
    box2_area = (w2 + 1) * (h2 + 1)
    
    # Calculate Intersection over Union (IoU) ratio
    if(float(box1_area + box2_area - intersection_area) > 0):
        iou = intersection_area / float(box1_area + box2_area - intersection_area)
    else:
        iou = 0
        print("shittt_division_by_zero")
    return iou

def iou_2d_cal(list1_2d,list2_2d):
    results = []
    for box1,box2 in zip(list1_2d,list2_2d):
        if((None in box1) or (None in box2)):
            continue
        else:
            results.append(calculate_iou(box1,box2))
    if(len(results) != 0):
        return sum(results)/len(results)
    else:
        #print("some error occured: iou_cal.py")
        return None
    

if __name__ == "main":
    # Example usage
    box1 = [50, 50, 100, 100]  # Format: [topleft_x, topleft_y, width, height]
    box2 = [60, 60, 100, 100]  # Format: [topleft_x, topleft_y, width, height]
    iou_ratio = calculate_iou(box1, box2)
    print("Overlap ratio (IoU):", iou_ratio)
