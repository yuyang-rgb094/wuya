#!/usr/bin/env python3
"""
Convert JSON Canvas files to PNG images
Optimized: better label positioning, avoid occlusion
"""

import json
import os
import math
from PIL import Image, ImageDraw, ImageFont

# Enhanced color palette
NODE_COLORS = {
    "1": {"bg": "#FFE4E4", "border": "#FF6B6B"},
    "2": {"bg": "#E4F7F5", "border": "#4ECDC4"},
    "3": {"bg": "#E4F0F7", "border": "#45B7D1"},
    "4": {"bg": "#E8F5E8", "border": "#96CEB4"},
    "5": {"bg": "#FFF8E4", "border": "#FFD93D"},
    "6": {"bg": "#F5E4F7", "border": "#DDA0DD"},
}

EDGE_COLOR = "#34495E"  # Professional dark blue-gray
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

def get_segment_midpoint(points, segment_idx):
    """Get midpoint of a specific segment, not the whole edge"""
    if segment_idx < 0 or segment_idx >= len(points) - 1:
        # Fallback to overall midpoint
        p1, p2 = points[0], points[-1]
        return ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
    
    p1, p2 = points[segment_idx], points[segment_idx + 1]
    return ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)

def get_horizontal_segment_midpoint(points, preferred_y_offset=0):
    """Find a horizontal segment and get its midpoint, with offset"""
    for i in range(len(points) - 1):
        p1, p2 = points[i], points[i + 1]
        if abs(p2[1] - p1[1]) < 2:  # Horizontal segment
            mid = ((p1[0] + p2[0]) // 2, p1[1] + preferred_y_offset)
            return mid, i
    # No horizontal segment, use overall midpoint
    return get_segment_midpoint(points, 0), 0

def get_vertical_segment_midpoint(points, preferred_x_offset=0):
    """Find a vertical segment and get its midpoint, with offset"""
    for i in range(len(points) - 1):
        p1, p2 = points[i], points[i + 1]
        if abs(p2[0] - p1[0]) < 2:  # Vertical segment
            mid = (p1[0] + preferred_x_offset, (p1[1] + p2[1]) // 2)
            return mid, i
    return get_segment_midpoint(points, 0), 0

def draw_edge_label(draw, pos, text, font):
    """Draw a label with background at specified position"""
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    px, py = pos
    
    padding = 5
    r = 5
    x1, y1 = px - tw//2 - padding, py - th//2 - padding
    x2, y2 = px + tw//2 + padding, py + th//2 + padding
    
    # Draw shadow
    draw.rounded_rectangle([x1+2, y1+2, x2+2, y2+2], radius=r, fill='#E0E0E0')
    # Draw background
    draw.rounded_rectangle([x1, y1, x2, y2], radius=r, fill='white', outline='#AAAAAA', width=1)
    # Draw text
    draw.text((px - tw//2, py - th//2), text, fill='#2C3E50', font=font)

def route_edge(start, end, from_side, to_side):
    """Route edge with longer segments for better label placement"""
    exit_off = 30
    entry_off = 30
    
    # Exit direction
    if from_side == 'top':    dx, dy = 0, -1
    elif from_side == 'bottom': dx, dy = 0, 1
    elif from_side == 'left':   dx, dy = -1, 0
    elif from_side == 'right':  dx, dy = 1, 0
    else: dx, dy = 0, 0
    
    # Entry direction
    if to_side == 'top':    ex, ey = 0, -1
    elif to_side == 'bottom': ex, ey = 0, 1
    elif to_side == 'left':   ex, ey = -1, 0
    elif to_side == 'right':  ex, ey = 1, 0
    else: ex, ey = 0, 0
    
    p1 = (start[0] + dx * exit_off, start[1] + dy * exit_off)
    p3 = (end[0] + ex * entry_off, end[1] + ey * entry_off)
    
    # Route based on connection type
    if from_side in ['left', 'right'] and to_side in ['left', 'right']:
        # Horizontal to horizontal
        mid = (p1[0] + p3[0]) // 2
        if dx > 0:  # Going right
            points = [start, p1, (mid - 40, p1[1]), (mid - 40, p3[1]), p3, end]
        else:
            points = [start, p1, (mid + 40, p1[1]), (mid + 40, p3[1]), p3, end]
    elif from_side in ['top', 'bottom'] and to_side in ['top', 'bottom']:
        # Vertical to vertical
        mid = (p1[1] + p3[1]) // 2
        if dy > 0:  # Going down
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
    
    # Simplify
    simplified = [points[0]]
    for i in range(1, len(points) - 1):
        if points[i] != simplified[-1]:
            simplified.append(points[i])
    simplified.append(points[-1])
    
    return simplified

def find_best_label_position(points, from_side, to_side):
    """Find the best position for label, avoiding corners"""
    # Prefer horizontal segments for labels (easier to read)
    for i in range(len(points) - 1):
        p1, p2 = points[i], points[i + 1]
        if abs(p2[1] - p1[1]) < 2:  # Horizontal
            mid = ((p1[0] + p2[0]) // 2, p1[1])
            # Offset label away from the line
            if from_side == 'bottom' or to_side == 'top':
                return (mid[0], mid[1] - 15)
            else:
                return (mid[0], mid[1] - 15)
    
    # Fallback to vertical segment
    for i in range(len(points) - 1):
        p1, p2 = points[i], points[i + 1]
        if abs(p2[0] - p1[0]) < 2:  # Vertical
            mid = (p1[0], (p1[1] + p2[1]) // 2)
            if from_side == 'right' or to_side == 'left':
                return (mid[0] + 15, mid[1])
            else:
                return (mid[0] - 50, mid[1])
    
    # Ultimate fallback
    return ((points[0][0] + points[-1][0]) // 2, (points[0][1] + points[-1][1]) // 2 - 15)

def draw_polyline_edge(draw, points, color, label, label_pos, font):
    """Draw a polyline edge with label"""
    width = 2
    
    # Draw line segments
    for i in range(len(points) - 1):
        draw.line([points[i], points[i + 1]], fill=color, width=width)
    
    # Draw corner circles
    for i in range(1, len(points) - 1):
        cx, cy = points[i]
        r = width + 2
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color)
    
    # Draw arrowhead
    if len(points) >= 2:
        p1, p2 = points[-2], points[-1]
        angle = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
        arrow_len = 12
        ax1 = p2[0] - arrow_len * math.cos(angle - math.pi/6)
        ay1 = p2[1] - arrow_len * math.sin(angle - math.pi/6)
        ax2 = p2[0] - arrow_len * math.cos(angle + math.pi/6)
        ay2 = p2[1] - arrow_len * math.sin(angle + math.pi/6)
        draw.polygon([p2, (ax1, ay1), (ax2, ay2)], fill=color)
    
    # Draw label
    if label and font:
        draw_edge_label(draw, label_pos, label, font)

def render_canvas(canvas_file, output_file):
    with open(canvas_file, 'r') as f:
        data = json.load(f)
    
    nodes = data.get('nodes', [])
    edges = data.get('edges', [])
    
    if not nodes:
        return
    
    # Calculate bounds with more margin
    all_x = [n['x'] for n in nodes] + [n['x'] + n.get('width', 100) for n in nodes]
    all_y = [n['y'] for n in nodes] + [n['y'] + n.get('height', 100) for n in nodes]
    
    margin = 120
    min_x, min_y = min(all_x) - margin, min(all_y) - margin
    max_x, max_y = max(all_x) + margin, max(all_y) + margin
    
    width, height = max_x - min_x, max_y - min_y
    
    img = Image.new('RGB', (width, height), '#FFFFFF')
    draw = ImageDraw.Draw(img)
    
    # Load fonts - use larger fonts for self-improvement-curve
    is_curve = 'curve' in canvas_file.lower()
    try:
        if is_curve:
            font_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
            font_header = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
            font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
            font_label = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 10)
        else:
            font_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
            font_header = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 11)
            font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 9)
            font_label = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 8)
    except:
        font_title = font_header = font_small = font_label = ImageFont.load_default()
    
    node_map = {n['id']: n for n in nodes}
    
    # Draw edges
    for edge in edges:
        from_node = node_map.get(edge['fromNode'])
        to_node = node_map.get(edge['toNode'])
        if not from_node or not to_node:
            continue
        
        from_side = edge.get('fromSide', 'center')
        to_side = edge.get('toSide', 'center')
        label = edge.get('label', '')
        
        # Color based on feedback loop
        if edge.get('color') and '888' in str(edge['color']):
            color = FEEDBACK_COLOR
        else:
            color = EDGE_COLOR
        
        start = get_edge_point(from_node, from_side)
        end = get_edge_point(to_node, to_side)
        
        if from_side != 'center' or to_side != 'center':
            points = route_edge(start, end, from_side, to_side)
            label_pos = find_best_label_position(points, from_side, to_side)
        else:
            points = [start, end]
            label_pos = ((start[0] + end[0]) // 2, (start[1] + end[1]) // 2 - 15)
        
        # Convert to image coords
        points_img = [(p[0] - min_x, p[1] - min_y) for p in points]
        label_pos_img = (label_pos[0] - min_x, label_pos[1] - min_y)
        
        draw_polyline_edge(draw, points_img, color, label, label_pos_img, font_label)
    
    # Draw nodes
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
                    text_y += 4
                    continue
                if text_y > max_y_text:
                    break
                
                if style == 'h1':
                    draw.text((x+12, text_y), line, fill='#1A1A2E', font=font_title)
                    text_y += 20
                elif style == 'h2':
                    draw.text((x+12, text_y), line, fill='#2C3E50', font=font_header)
                    text_y += 16
                elif style == 'bullet':
                    draw.text((x+16, text_y), '• ' + line, fill='#34495E', font=font_small)
                    text_y += 13
                else:
                    draw.text((x+12, text_y), line, fill='#34495E', font=font_small)
                    text_y += 12
    
    img.save(output_file, 'PNG', dpi=(150, 150), optimize=True)
    print(f"Saved: {output_file}")

if __name__ == '__main__':
    os.chdir('/sessions/6a0d8a41926ecfc2b575988c/workspace/paper/figures')
    for cf in ['architecture.canvas', 'hybrid-knowledge.canvas', 'self-improvement-loop.canvas', 'self-improvement-curve.canvas']:
        if os.path.exists(cf):
            render_canvas(cf, cf.replace('.canvas', '.png'))
