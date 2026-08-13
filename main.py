import os
import cv2
import numpy as np

BANDAID_PATH = "assets/bandaid.png"
WOUND_FOLDER = "wound"
OUTPUT_FOLDER = "output"


def get_red_mask(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # red wraps around in HSV
    red_low = cv2.inRange(hsv, (0, 80, 40), (8, 255, 255))
    red_high = cv2.inRange(hsv, (172, 80, 40), (180, 255, 255))
    red = red_low | red_high
    red = cv2.morphologyEx(red, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

    # ignore
    saturation = hsv[:, :, 1]
    red_saturation_values = saturation[red > 0]
    if not len(red_saturation_values):
        return red

    vivid_cutoff = min(120, 0.6 * np.percentile(red_saturation_values, 90))
    is_vivid = (saturation >= vivid_cutoff).astype(np.uint8) * 255
    return cv2.bitwise_and(red, is_vivid)


def find_wounds(image):
    height, width = image.shape[:2]
    red = get_red_mask(image)

    close_kernel_size = max(7, (min(height, width) // 25) | 1)
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel_size, close_kernel_size))
    merged = cv2.morphologyEx(red, cv2.MORPH_CLOSE, close_kernel)

    halo_kernel_size = max(5, (min(height, width) // 40) | 1)
    halo_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (halo_kernel_size, halo_kernel_size))

    contours, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = 0.0002 * (height * width)

    wounds = []
    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            continue

        blob_mask = np.zeros((height, width), np.uint8)
        cv2.drawContours(blob_mask, [contour], -1, 255, -1)
        blob_mask = cv2.dilate(blob_mask, halo_kernel)

        ys, xs = np.nonzero(cv2.bitwise_and(red, blob_mask))
        points = np.column_stack([xs, ys]).astype(np.float32)
        wound = measure_wound(points)

        # pen lines / scratches
        if wound["width"] < 8 or wound["length"] / max(wound["width"], 1) > 8:
            continue

        wound["pixel_count"] = len(xs)
        wounds.append(wound)

    wounds.sort(key=lambda w: w["pixel_count"], reverse=True)
    return wounds


def measure_wound(points):
    vx, vy, x0, y0 = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01).flatten()

    angle = np.degrees(np.arctan2(vy, vx))
    if angle > 90:
        angle -= 180
    if angle < -90:
        angle += 180

    dx = points[:, 0] - x0
    dy = points[:, 1] - y0
    along = dx * vx + dy * vy
    across = -dx * vy + dy * vx

    along_min, along_max = np.percentile(along, [2, 98])
    across_min, across_max = np.percentile(across, [2, 98])

    mid_along = (along_min + along_max) / 2
    mid_across = (across_min + across_max) / 2
    center_x = x0 + mid_along * vx - mid_across * vy
    center_y = y0 + mid_along * vy + mid_across * vx

    return {
        "center": (int(center_x), int(center_y)),
        "length": float(along_max - along_min),
        "width": float(across_max - across_min),
        "angle": float(angle),
    }


def fit_inside_bounds(x, y, box_width, box_height, image_width, image_height):
    x1, y1 = max(x, 0), max(y, 0)
    x2, y2 = min(x + box_width, image_width), min(y + box_height, image_height)
    return x1, y1, x2, y2


def rotate_image(image, angle):
    height, width = image.shape[:2]
    diagonal = int(np.hypot(height, width)) + 2

    canvas = np.zeros((diagonal, diagonal, 4), np.uint8)
    top = (diagonal - height) // 2
    left = (diagonal - width) // 2
    canvas[top:top + height, left:left + width] = image

    center = (diagonal / 2, diagonal / 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(canvas, rotation_matrix, (diagonal, diagonal),
                           flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0, 0))


def overlay_image(background, overlay, center_x, center_y):
    bg_height, bg_width = background.shape[:2]
    ov_height, ov_width = overlay.shape[:2]

    top_left_x = center_x - ov_width // 2
    top_left_y = center_y - ov_height // 2

    x1, y1, x2, y2 = fit_inside_bounds(top_left_x, top_left_y, ov_width, ov_height, bg_width, bg_height)
    if x1 >= x2 or y1 >= y2:
        return background

    visible_part = overlay[y1 - top_left_y:y2 - top_left_y, x1 - top_left_x:x2 - top_left_x]

    alpha = visible_part[:, :, 3:4].astype(float) / 255.0
    background_region = background[y1:y2, x1:x2]
    blended = alpha * visible_part[:, :, :3] + (1 - alpha) * background_region
    background[y1:y2, x1:x2] = blended.astype(np.uint8)
    return background


def round_corners_mask(width, height):
    radius = int(height * 0.4)
    mask = np.zeros((height, width), np.uint8)

    cv2.rectangle(mask, (radius, 0), (width - radius, height), 255, -1)
    cv2.rectangle(mask, (0, radius), (width, height - radius), 255, -1)
    cv2.circle(mask, (radius, radius), radius, 255, -1)
    cv2.circle(mask, (width - radius, radius), radius, 255, -1)
    cv2.circle(mask, (radius, height - radius), radius, 255, -1)
    cv2.circle(mask, (width - radius, height - radius), radius, 255, -1)

    return cv2.GaussianBlur(mask, (5, 5), 0)


def shape_bandaid(bandaid, length, width):
    length, width = int(length), int(width)
    fabric = cv2.resize(bandaid, (length, width), interpolation=cv2.INTER_AREA)
    mask = round_corners_mask(length, width)
    fabric[:, :, 3] = np.minimum(fabric[:, :, 3], mask)
    return fabric


def place_bandaid(image, wound, bandaid):
    img_height, img_width = image.shape[:2]

    length = wound["length"] * 1.6
    length = min(length, 0.75 * min(img_height, img_width))

    width = wound["width"] * 1.3
    width = max(length / 6.0, min(width, length / 1.6))

    shaped = shape_bandaid(bandaid, length, width)
    rotated = rotate_image(shaped, -wound["angle"])

    center_x, center_y = wound["center"]
    return overlay_image(image, rotated, center_x, center_y)


def combine_before_after(before, after):
    height = before.shape[0]
    after_resized = cv2.resize(after, (before.shape[1], height))
    return np.hstack([before, after_resized])


def process_image(image_path, number):
    before = cv2.imread(image_path)
    after = before.copy()
    detected = before.copy()

    wounds = find_wounds(before)
    bandaid = cv2.imread(BANDAID_PATH, cv2.IMREAD_UNCHANGED) if wounds else None

    for wound in wounds:
        box = cv2.boxPoints((wound["center"], (wound["length"], wound["width"]), wound["angle"]))
        cv2.drawContours(detected, [box.astype(np.int32)], -1, (0, 255, 0), 2)
        after = place_bandaid(after, wound, bandaid)

    cv2.imwrite(os.path.join(OUTPUT_FOLDER, "final_%d.jpg" % number), after)
    cv2.imwrite(os.path.join(OUTPUT_FOLDER, "combined_%d.jpg" % number), combine_before_after(detected, after))


def main():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    image_names = [n for n in sorted(os.listdir(WOUND_FOLDER)) if n.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))]
    for number, name in enumerate(image_names, start=1):
        process_image(os.path.join(WOUND_FOLDER, name), number)


if __name__ == "__main__":
    main()
