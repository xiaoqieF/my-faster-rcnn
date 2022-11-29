from PIL.Image import Image
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont
from PIL import ImageColor
import numpy as np


def draw_objs(image: Image,
              boxes: np.ndarray = None,
              classes: np.ndarray = None,
              scores: np.ndarray = None,
              category_index: dict = None,
              box_thresh: float = 0.1,
              line_thick: int = 8,
              font: str = 'nakula.ttf',
              font_size: int = 12,
              draw_boxes_on_image: bool = True):
    idxs = np.greater(scores, box_thresh)
    boxes = boxes[idxs]
    classes = classes[idxs]
    scores = scores[idxs]
    if len(boxes) == 0:
        return image
    
    if draw_boxes_on_image:
        draw = ImageDraw.Draw(image)
        font = ImageFont.truetype(font, font_size)
        for box, cls, score in zip(boxes, classes, scores):
            left, top, right, bottom = box
            draw.rectangle([(left, top), (right, bottom)], width=line_thick, outline="red")
            
            text = f"{category_index[str(cls)]}: {str(score):.3}"
            text_width, text_height = font.getsize(text)
            margin = np.ceil(0.05 * text_width)
            draw.rectangle([(left, top), (left + text_width + 2 * margin, top + text_height)], fill='blue')
            draw.text((left + margin, top), text, fill="white", font=font)
    return image