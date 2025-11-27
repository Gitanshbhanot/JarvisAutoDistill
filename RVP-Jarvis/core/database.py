import os
import yaml
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from collections import Counter
import hashlib
import textwrap
import tempfile
import json

def get_available_annotated_datasets_for_viewing():
    """Get list of available annotated datasets for viewing"""
    annotated_data_path = Path("data/annotated_data")
    available_datasets = []
    if annotated_data_path.exists():
        for folder in annotated_data_path.iterdir():
            if folder.is_dir() and (folder / "dataset.yaml").exists():
                available_datasets.append(folder.name)
    return available_datasets

def get_dataset_images(dataset_name):
    """Get list of images in the selected dataset"""
    if not dataset_name:
        return []
    
    dataset_path = Path(f"data/annotated_data/{dataset_name}")
    if not dataset_path.exists():
        return []
    
    images = []
    
    # Check train images
    train_path = dataset_path / "images" / "train"
    if train_path.exists():
        for img_file in train_path.glob("*.jpg"):
            images.append(f"train/{img_file.name}")
        for img_file in train_path.glob("*.png"):
            images.append(f"train/{img_file.name}")
        for img_file in train_path.glob("*.jpeg"):
            images.append(f"train/{img_file.name}")
    
    # Check val images
    val_path = dataset_path / "images" / "val"
    if val_path.exists():
        for img_file in val_path.glob("*.jpg"):
            images.append(f"val/{img_file.name}")
        for img_file in val_path.glob("*.png"):
            images.append(f"val/{img_file.name}")
        for img_file in val_path.glob("*.jpeg"):
            images.append(f"val/{img_file.name}")
    
    return images

def load_dataset_info(dataset_name):
    """Load dataset information including class names"""
    if not dataset_name:
        return [], []
    
    dataset_path = Path(f"data/annotated_data/{dataset_name}")
    yaml_path = dataset_path / "dataset.yaml"
    
    if not yaml_path.exists():
        return [], []
    
    try:
        with open(yaml_path, 'r') as f:
            dataset_info = yaml.safe_load(f)
        
        class_names = dataset_info.get('names', [])
        images = get_dataset_images(dataset_name)
        
        return images, class_names
    except Exception as e:
        print(f"Error loading dataset info: {e}")
        return [], []

def draw_bounding_boxes(
    image_path,
    label_path,
    class_names,
    *,
    show_index=False,
    show_confidence=True,
    font_scale=0.01,       # ~3% of image width
    min_font_px=12,
    line_ratio=0.25,       # stroke ≈ font_size * 0.25
    chip_pad_px=4,
    max_label_chars=28,
    min_conf=0.0,
    include_class_ids=None,
):
    """Draw bounding boxes on image using YOLO labels (train: class x y w h; pred: class x y w h conf).
    Everything is self-contained in this function: imports, helpers, and logic.
    Improvements: sort boxes by confidence or area (descending), try multiple label positions to avoid overlaps,
    use rounded rectangles for chips, filter by min_conf, clamp improvements."""

    # ---- imports (scoped inside for self-contained) ----
    import os
    import hashlib
    import textwrap
    from PIL import Image, ImageDraw, ImageFont

    # ---- helpers ----
    def _load_font(px):
        candidates = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/System/Library/Fonts/SFNS.ttf",
            "C:\\Windows\\Fonts\\arial.ttf",
            "arial.ttf",
        ]
        for p in candidates:
            try:
                return ImageFont.truetype(p, size=int(px))
            except Exception:
                pass
        return ImageFont.load_default()

    def _class_color(cid):
        # pleasant-ish stable RGB from class id hash
        h = int(hashlib.md5(str(cid).encode()).hexdigest()[:6], 16)
        r = 64 + (h >> 16) % 160
        g = 64 + (h >> 8) % 160
        b = 64 + h % 160
        return (r, g, b)

    def _clamp(v, lo, hi):
        return max(lo, min(hi, v))

    def _rect_overlaps(r1, r2):
        x1, y1, x2, y2 = r1
        ex1, ey1, ex2, ey2 = r2
        return not (x2 <= ex1 or x1 >= ex2 or y2 <= ey1 or y1 >= ey2)

    # ---- load image ----
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as e:
        raise RuntimeError(f"Could not open image: {e}")

    W, H = img.size

    # ---- auto-scale font and stroke ----
    font_px = max(int(W * float(font_scale)), int(min_font_px))
    font = _load_font(font_px)
    stroke = max(1, int(round(font_px * float(line_ratio))))

    # ---- read labels (if missing, return original) ----
    if not os.path.exists(label_path):
        return img
    try:
        with open(label_path, "r") as f:
            lines = [ln.strip() for ln in f if ln.strip()]
    except Exception as e:
        print(f"Error reading label file: {e}")
        return img

    # ---- parse lines and filter by class_ids if specified ----
    parsed = []
    for line in lines:
        parts = line.split()
        if len(parts) < 5: 
            continue
        cid = int(parts[0])
        xc, yc, w, h = map(float, parts[1:5])
        conf = float(parts[5]) if len(parts) >= 6 else 1.0
        parsed.append((cid, xc, yc, w, h, conf))

    # Filter by include_class_ids if specified
    if include_class_ids:
        parsed = [p for p in parsed if p[0] in include_class_ids]

    # Sort by conf desc (per image)
    parsed.sort(key=lambda x: x[5], reverse=True)

    # Convert back to lines for existing logic
    lines = []
    for cid, xc, yc, w, h, conf in parsed:
        if conf < min_conf:
            continue
        lines.append(f"{cid} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f} {conf:.6f}")

    # Draw on an RGBA overlay for nice alpha chips
    overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    occupied_chips = []  # list of (x1,y1,x2,y2) for placed chips

    possible_positions = ['top', 'inside_top', 'bottom', 'inside_bottom', 'left', 'right']

    for idx, line in enumerate(lines):
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            class_id = int(float(parts[0]))
            cx = float(parts[1]) * W
            cy = float(parts[2]) * H
            bw = float(parts[3]) * W
            bh = float(parts[4]) * H
            conf = float(parts[5]) if (show_confidence and len(parts) >= 6) else None
        except Exception:
            continue

        # Skip low conf or degenerate/tiny boxes
        if conf is not None and conf < min_conf:
            continue
        if bw <= 1 or bh <= 1:
            continue

        # corners + clamp
        x1 = _clamp(cx - bw / 2.0, 0, W - 1)
        y1 = _clamp(cy - bh / 2.0, 0, H - 1)
        x2 = _clamp(cx + bw / 2.0, 0, W - 1)
        y2 = _clamp(cy + bh / 2.0, 0, H - 1)

        # color for class
        color = _class_color(class_id)  # (r,g,b)
        outline_color = color + (255,)

        # draw rectangle (do this first, labels later)
        draw.rectangle([x1, y1, x2, y2], outline=outline_color, width=stroke)

        # label text
        class_name = class_names[class_id] if 0 <= class_id < len(class_names) else f"Class {class_id}"
        if max_label_chars and len(class_name) > max_label_chars:
            class_name = textwrap.shorten(class_name, width=max_label_chars, placeholder="…")
        pieces = []
        if show_index:
            pieces.append(f"#{idx}")
        pieces.append(class_name)
        if conf is not None:
            pieces.append(f"{conf:.2f}")
        label_text = " ".join(pieces)

        # text size (use textbbox for precision)
        tb = draw.textbbox((0, 0), label_text, font=font, stroke_width=0)
        text_w = tb[2] - tb[0]
        text_h = tb[3] - tb[1]

        # chip size
        chip_w = text_w + chip_pad_px * 2
        chip_h = text_h + chip_pad_px * 2

        # find best position
        best_chip_rect = None
        for pos in possible_positions:
            if pos == 'top':
                chip_y1 = y1 - chip_h - max(1, stroke)
                if chip_y1 < 0:
                    continue
                chip_x1 = x1
            elif pos == 'bottom':
                chip_y1 = y2 + max(1, stroke)
                if chip_y1 + chip_h > H:
                    continue
                chip_x1 = x1
            elif pos == 'left':
                chip_x1 = x1 - chip_w - max(1, stroke)
                if chip_x1 < 0:
                    continue
                chip_y1 = _clamp(cy - chip_h / 2, 0, H - chip_h)
                if chip_y1 + chip_h > H or chip_y1 < 0:
                    continue
            elif pos == 'right':
                chip_x1 = x2 + max(1, stroke)
                if chip_x1 + chip_w > W:
                    continue
                chip_y1 = _clamp(cy - chip_h / 2, 0, H - chip_h)
                if chip_y1 + chip_h > H or chip_y1 < 0:
                    continue
            elif pos == 'inside_top':
                chip_y1 = y1 + max(1, stroke // 2)
                if chip_y1 + chip_h > y2:
                    continue
                chip_x1 = x1
            elif pos == 'inside_bottom':
                chip_y1 = y2 - chip_h - max(1, stroke // 2)
                if chip_y1 < y1:
                    continue
                chip_x1 = x1

            # horizontal shift to fit in image
            if chip_x1 + chip_w > W:
                chip_x1 = _clamp(W - chip_w, 0, W - 1)
            if chip_x1 < 0:
                continue

            chip_x2 = chip_x1 + chip_w
            chip_y2 = chip_y1 + chip_h

            # skip if still out (rare)
            if chip_y2 > H or chip_x2 > W or chip_y1 < 0 or chip_x1 < 0:
                continue

            # check overlaps
            overlaps = any(_rect_overlaps((chip_x1, chip_y1, chip_x2, chip_y2), occ) for occ in occupied_chips)
            if not overlaps:
                best_chip_rect = (chip_x1, chip_y1, chip_x2, chip_y2)
                break

        if best_chip_rect is None:
            # if no non-overlapping, fall back to inside_top if possible, else skip label
            # quick calc for inside_top
            chip_y1 = y1 + max(1, stroke // 2)
            if chip_y1 + chip_h > y2:
                continue  # skip label
            chip_x1 = x1
            if chip_x1 + chip_w > W:
                chip_x1 = _clamp(W - chip_w, 0, W - 1)
            if chip_x1 < 0:
                continue
            chip_x2 = chip_x1 + chip_w
            chip_y2 = chip_y1 + chip_h
            best_chip_rect = (chip_x1, chip_y1, chip_x2, chip_y2)

        chip_x1, chip_y1, chip_x2, chip_y2 = best_chip_rect

        # add to occupied (even if fallback)
        occupied_chips.append((chip_x1, chip_y1, chip_x2, chip_y2))

        # chip bg: semi-transparent fill + solid outline, rounded
        fill_rgba = color + (180,)  # translucent
        radius = min(8, chip_h // 3)
        draw.rounded_rectangle([chip_x1, chip_y1, chip_x2, chip_y2], radius=radius, fill=fill_rgba, outline=outline_color, width=max(1, stroke//2))

        # draw text (white for contrast)
        text_x = chip_x1 + chip_pad_px
        text_y = chip_y1 + chip_pad_px
        draw.text((text_x, text_y), label_text, font=font, fill=(255, 255, 255, 255))

    # composite overlay onto base
    out = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
    return out


def view_annotated_image(dataset_name, image_choice, class_names, include_classes=None):
    """
    View annotated image with bounding boxes.

    include_classes: list[str] | None
        If provided, only these class names will be drawn.
        If None or empty, all classes are drawn (current behavior).
    """
    if not dataset_name or not image_choice:
        return None, "Please select a dataset and image first."
    
    try:
        dataset_path = Path(f"data/annotated_data/{dataset_name}")

        # Determine if it's train or val
        if image_choice.startswith("train/"):
            img_rel = image_choice.split("/")[1]
            image_path = dataset_path / "images" / "train" / img_rel
            label_path = dataset_path / "labels" / "train" / f"{img_rel.rsplit('.', 1)[0]}.txt"
        elif image_choice.startswith("val/"):
            img_rel = image_choice.split("/")[1]
            image_path = dataset_path / "images" / "val" / img_rel
            label_path = dataset_path / "labels" / "val" / f"{img_rel.rsplit('.', 1)[0]}.txt"
        else:
            return None, "Invalid image choice."

        if not image_path.exists():
            return None, f"Image file not found: {image_path}"

        # Build class-name -> id map
        name_to_id = {name: idx for idx, name in enumerate(class_names)}

        # Normalize/validate requested classes
        include_classes = include_classes or []
        include_classes = [c for c in include_classes if c in name_to_id]
        include_ids = {name_to_id[c] for c in include_classes}

        # Read labels (YOLO format: "<cls> x y w h [rest]")
        lines = []
        if label_path.exists():
            with open(label_path, "r") as f:
                lines = [ln.strip() for ln in f if ln.strip()]

        # Filter lines if a subset is requested
        from typing import Union
        def _line_class_id(ln: str) -> Union[int, None]:
            try:
                return int(ln.split()[0])
            except Exception:
                return None

        if include_ids:
            filtered_lines = [ln for ln in lines if _line_class_id(ln) in include_ids]
        else:
            filtered_lines = lines

        # Counts for info text (after filtering)
        counts = Counter(_line_class_id(ln) for ln in filtered_lines if _line_class_id(ln) is not None)

        # --- Render annotated image ---
        # Prefer calling a drawer that supports include_class_ids=...
        try:
            annotated_image = draw_bounding_boxes(
                image_path,
                label_path,
                class_names,
                include_class_ids=(sorted(include_ids) if include_ids else None),
            )
        except TypeError:
            # Fallback for legacy signature: write a temp filtered label file
            if include_ids:
                with tempfile.NamedTemporaryFile("w+", suffix=".txt", delete=False) as tf:
                    tf.write("\n".join(filtered_lines))
                    tf.flush()
                    temp_label_path = Path(tf.name)
                try:
                    annotated_image = draw_bounding_boxes(image_path, temp_label_path, class_names)
                finally:
                    try:
                        temp_label_path.unlink(missing_ok=True)
                    except Exception:
                        pass
            else:
                annotated_image = draw_bounding_boxes(image_path, label_path, class_names)

        annotated_image_np = np.array(annotated_image)
        img_width, img_height = annotated_image.size

        # Info text
        info_lines = [
            "📊 **Image Information:**",
            f"• **Dataset:** {dataset_name}",
            f"• **Image:** {image_choice}",
            f"• **Dimensions:** {img_width} × {img_height}",
            f"• **Classes (dataset):** {', '.join(class_names)}",
        ]
        if include_classes:
            info_lines.append(f"• **Shown classes:** {', '.join(include_classes)}")

        if lines:
            info_lines.append(f"• **Annotations (shown):** {sum(counts.values())} bounding boxes")
            # Per-class counts (only those present)
            if counts:
                per_class = ", ".join(
                    f"{class_names[cid]}: {cnt}" for cid, cnt in sorted(counts.items())
                )
                info_lines.append(f"• **Counts by class (shown):** {per_class}")
        else:
            info_lines.append("• **Annotations:** No label file found")

        info_text = "\n".join(info_lines)

        return annotated_image_np, info_text

    except Exception as e:
        return None, f"Error viewing annotated image: {str(e)}"


PROJECT_ROOT = Path(__file__).resolve().parent.parent

def save_golden_set(dataset_name: str, golden_data: dict):
    """Save golden set data for a dataset."""
    metadata_dir = PROJECT_ROOT / "data" / "raw_data" / dataset_name / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    with open(metadata_dir / "golden.json", "w") as f:
        json.dump(golden_data, f)

def load_golden_set(dataset_name: str) -> dict:
    """Load golden set data for a dataset."""
    metadata_dir = PROJECT_ROOT / "data" / "raw_data" / dataset_name / "metadata"
    golden_path = metadata_dir / "golden.json"
    if golden_path.exists():
        with open(golden_path) as f:
            return json.load(f)
    return {}
