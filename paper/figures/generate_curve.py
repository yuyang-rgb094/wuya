#!/usr/bin/env python3
"""Generate only self-improvement-curve.png with larger fonts"""

import json
import os
import math
from PIL import Image, ImageDraw, ImageFont

NODE_COLORS = {
    "1": {"bg": "#FFE4E4", "border": "#FF6B6B"},
    "2": {"bg": "#E4F7F5", "border": "#4ECDC4"},
    "3": {"bg": "#E4F0F7", "border": "#45B7D1"},
    "4": {"bg": "#E8F5E8", "border": "#96CEB4"},
    "5": {"bg": "#FFF8E4", "border": "#FFD93D"},
    "6": {"bg": "#F5E4F7", "border": "#DDA0DD"},
}

EDGE_COLOR = "#34495E"
FEEDBACK_COLOR = "#95A5A6"

def get_node_color(color_id, part="bg"):
    return NODE_COLORS.get(str(color_id), {}).get(part, "#F5F5F5" if part == "bg" else "#CCCCCC")

def parse_markdown_text(text):
    lines = text.strip().split('\n')
    parsed = []
    for line in lines:
        line = line.strip()
        if not line:
            parsed.append(("", "empty"))
        elif line.startswith('# '):
            parsed.append((line[2:], "h1"))
        elif line.startswith('## '):
            parsed.append((line[3:], "h2"))
        elif line.startswith('- '):
            parsed.append((line[2:], "bullet"))
        else:
            parsed.append((line, "normal"))
    return parsed

def get_edge_point(node, side):
    x, y = node['x'], node['y']
    w, h = node.get('width', 100), node.get('height', 100)
    if side == 'top':    return (x + w // 2, y)
    elif side == 'bottom': return (x + w // 2, y + h)
    elif side == 'left':   return (x, y + h // 2)
    elif side == 'right':  return (x + w, y + h // 2)
    return (x + w // 2, y + h // 2)

def draw_edge_label(draw, pos, text, font):
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    px, py = pos
    padding = 5
    r = 5
    x1, y1 = px - tw//2 - padding, py - th//2 - padding
    x2, y2 = px + tw//2 + padding, py + th//2 + padding
    draw.rounded_rectangle([x1+2, y1+2, x2+2, y2+2], radius=r, fill='#E0E0E0')
    draw.rounded_rectangle([x1, y1, x2, y2], radius=r, fill='white', outline='#AAAAAA', width=1)
    draw.text((px - tw//2, py - th//2), text, fill='#2C3E50', font=font)

def route_edge(start, end, from_side, to_side):
    exit_off = 30
    entry_off = 30
    if from_side == 'top':    dx, dy = 0, -1
    elif from_side == 'bottom': dx, dy = 0, 1
    elif from_side == 'left':   dx, dy = -1, 0
    elif from_side == 'right':  dx, dy = 1, 0
    else: dx, dy = 0, 0
    if to_side == 'top':    ex, ey = 0, -1
    elif to_side == 'bottom': ex, ey = 0, 1
    elif to_side == 'left':   ex, ey = -1, 0
    elif to_side == 'right':  ex, ey = 1, 0
    else: ex, ey = 0, 0
    p1 = (start[0] + dx * exit_off, start[1] + dy * exit_off)
    p3 = (end[0] + ex * entry_off, end[1] + ey * entry_off)
    if from_side in ['left', 'right'] and to_side in ['left', 'right']:
        mid = (p1[0] + p3[0]) // 2
        if dx > 0:
            points = [start, p1, (mid - 40, p1[1]), (mid - 40, p3[1]), p3, end]
        else:
            points = [start, p1, (mid + 40, p1[1]), (mid + 40, p3[1]), p3, end]
    elif from_side in ['top', 'bottom'] and to_side in ['top', 'bottom']:
        mid = (p1[1] + p3[1]) // 2
        if dy > 0:
            points = [start, p1, (p1[0], mid - 40), (p3[0], mid - 40), p3, end]
        else:
            points = [start, p1, (p1[0], mid + 40), (p3[0], mid + 40), p3, end]
    elif from_side == 'right' and to_side == 'left':
        mid = (p1[0] + p3[0]) // 2
        points = [start, p1, (mid, p1[1]), (mid, p3[1]), p3, end]
    elif from_side == 'left' and to_side == 'right':
        mid = (p1[0] + p3[0]) // 2
        points = [start, p1, (mid, p1[1]), (mid, p3[1]), p3, end]
    elif from_side == 'bottom' and to_side == 'top':
        mid = (p1[1] + p3[1]) // 2
        points = [start, p1, (p1[0], mid), (p3[0], mid), p3, end]
    elif from_side == 'top' and to_side == 'bottom':
        mid = (p1[1] + p3[1]) // 2
        points = [start, p1, (p1[0], mid), (p3[0], mid), p3, end]
    elif from_side == 'right' and to_side in ['top', 'bottom']:
        points = [start, p1, (p3[0], p1[1]), p3, end]
    elif from_side == 'left' and to_side in ['top', 'bottom']:
        points = [start, p1, (p3[0], p1[1]), p3, end]
    elif from_side in ['top', 'bottom'] and to_side in ['left', 'right']:
        points = [start, p1, (p1[0], p3[1]), p3, end]
    else:
        points = [start, p1, p3, end]
    simplified = [points[0]]
    for i in range(1, len(points) - 1):
        if points[i] != simplified[-1]:
            simplified.append(points[i])
    simplified.append(points[-1])
    return simplified

def find_best_label_position(points, from_side, to_side):
    for i in range(len(points) - 1):
        p1, p2 = points[i], points[i + 1]
        if abs(p2[1] - p1[1]) < 2:
            mid = ((p1[0] + p2[0]) // 2, p1[1])
            return (mid[0], mid[1] - 15)
    return ((points[0][0] + points[-1][0]) // 2, (points[0][1] + points[-1][1]) // 2 - 15)

def draw_polyline_edge(draw, points, color, label, label_pos, font):
    width = 2
    for i in range(len(points) - 1):
        draw.line([points[i], points[i + 1]], fill=color, width=width)
    for i in range(1, len(points) - 1):
        cx, cy = points[i]
        r = width + 2
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color)
    if len(points) >= 2:
        p1, p2 = points[-2], points[-1]
        angle = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
        arrow_len = 12
        ax1 = p2[0] - arrow_len * math.cos(angle - math.pi/6)
        ay1 = p2[1] - arrow_len * math.sin(angle - math.pi/6)
        ax2 = p2[0] - arrow_len * math.cos(angle + math.pi/6)
        ay2 = p2[1] - arrow_len * math.sin(angle + math.pi/6)
        draw.polygon([p2, (ax1, ay1), (ax2, ay2)], fill=color)
    if label and font:
        draw_edge_label(draw, label_pos, label, font)

def render_curve():
    with open('self-improvement-curve.canvas', 'r') as f:
        data = json.load(f)
    
    nodes = data.get('nodes', [])
    edges = data.get('edges', [])
    
    all_x = [n['x'] for n in nodes] + [n['x'] + n.get('width', 100) for n in nodes]
    all_y = [n['y'] for n in nodes] + [n['y'] + n.get('height', 100) for n in nodes]
    
    margin = 120
    min_x, min_y = min(all_x) - margin, min(all_y) - margin
    max_x, max_y = max(all_x) + margin, max(all_y) + margin
    
    width, height = max_x - min_x, max_y - min_y
    
    img = Image.new('RGB', (width, height), '#FFFFFF')
    draw = ImageDraw.Draw(img)
    
    # Larger fonts for curve
    try:
        font_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
        font_header = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
        font_label = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    except:
        font_title = font_header = font_small = font_label = ImageFont.load_default()
    
    node_map = {n['id']: n for n in nodes}
    
    for edge in edges:
        from_node = node_map.get(edge['fromNode'])
        to_node = node_map.get(edge['toNode'])
        if not from_node or not to_node:
            continue
        from_side = edge.get('fromSide', 'center')
        to_side = edge.get('toSide', 'center')
        label = edge.get('label', '')
        color = FEEDBACK_COLOR if (edge.get('color') and '888' in str(edge['color'])) else EDGE_COLOR
        start = get_edge_point(from_node, from_side)
        end = get_edge_point(to_node, to_side)
        if from_side != 'center' or to_side != 'center':
            points = route_edge(start, end, from_side, to_side)
            label_pos = find_best_label_position(points, from_side, to_side)
        else:
            points = [start, end]
            label_pos = ((start[0] + end[0]) // 2, (start[1] + end[1]) // 2 - 15)
        points_img = [(p[0] - min_x, p[1] - min_y) for p in points]
        label_pos_img = (label_pos[0] - min_x, label_pos[1] - min_y)
        draw_polyline_edge(draw, points_img, color, label, label_pos_img, font_label)
    
    for node in nodes:
        x = node['x'] - min_x
        y = node['y'] - min_y
        w = node.get('width', 100)
        h = node.get('height', 100)
        node_type = node.get('type', 'text')
        color_id = node.get('color', '')
        bg_color = get_node_color(color_id, 'bg')
        border_color = get_node_color(color_id, 'border')
        
        if node_type == 'group':
            draw.rounded_rectangle([x, y, x+w, y+h], radius=12, fill=bg_color, outline=border_color, width=3)
            label = node.get('label', '')
            if label:
                draw.text((x+15, y+12), label, fill='#2C3E50', font=font_header)
        else:
            draw.rounded_rectangle([x, y, x+w, y+h], radius=6, fill=bg_color, outline=border_color, width=2)
            text = node.get('text', '')
            parsed = parse_markdown_text(text)
            text_y = y + 10
            max_y_text = y + h - 8
            for line, style in parsed:
                if not line or style == 'empty':
                    text_y += 5
                    continue
                if text_y > max_y_text:
                    break
                if style == 'h1':
                    draw.text((x+12, text_y), line, fill='#1A1A2E', font=font_title)
                    text_y += 28
                elif style == 'h2':
                    draw.text((x+12, text_y), line, fill='#2C3E50', font=font_header)
                    text_y += 20
                elif style == 'bullet':
                    draw.text((x+16, text_y), '• ' + line, fill='#34495E', font=font_small)
                    text_y += 17
                else:
                    draw.text((x+12, text_y), line, fill='#34495E', font=font_small)
                    text_y += 16
    
    img.save('self-improvement-curve.png', 'PNG', dpi=(150, 150), optimize=True)
    print("Saved: self-improvement-curve.png")

if __name__ == '__main__':
    os.chdir('/sessions/6a0d8a41926ecfc2b575988c/workspace/paper/figures')
    render_curve()
