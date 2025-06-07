import os
import glob
import numpy as np
from sklearn.cluster import KMeans

# Directory containing YOLO .txt label files
LABEL_DIR = os.path.join(os.path.dirname(__file__), '../kaist-rgbt/train/labels')
NUM_ANCHORS = 9
IGNORE_CLASS = 2  # 'People' class index

# Gather all (w, h) for non-people boxes
whs = []
for label_file in glob.glob(os.path.join(LABEL_DIR, '*.txt')):
    with open(label_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls, x, y, w, h = parts[:5]
            if int(cls) == IGNORE_CLASS:
                continue
            try:
                w = float(w)
                h = float(h)
                if w > 0 and h > 0:
                    whs.append([w, h])
            except Exception:
                continue

whs = np.array(whs)
print(f"Collected {len(whs)} boxes (excluding 'People') from {LABEL_DIR}")

# K-means clustering
kmeans = KMeans(n_clusters=NUM_ANCHORS, random_state=0, n_init=30).fit(whs)
anchors = kmeans.cluster_centers_

# Sort anchors by area (small to large)
anchors = anchors[np.argsort(anchors.prod(axis=1))]

# Convert normalized coordinates to pixel values (assuming 640x640 input)
pixel_anchors = (anchors * 640).astype(int)

# Split into 3 groups for P3/8, P4/16, P5/32
anchors_p3 = pixel_anchors[:3]  # Small objects
anchors_p4 = pixel_anchors[3:6]  # Medium objects
anchors_p5 = pixel_anchors[6:]  # Large objects

print("\nYOLO anchor (width,height) pairs in pixels (640x640 input):")
print("  # P3/8 (small objects)", anchors_p3.tolist())
print("  # P4/16 (medium objects)", anchors_p4.tolist())
print("  # P5/32 (large objects)", anchors_p5.tolist())

print("\nYOLO anchor string for model yaml:")
print('anchors:')
print('  # P3/8')
print('  -', str(anchors_p3.flatten().tolist())[1:-1])
print('  # P4/16')
print('  -', str(anchors_p4.flatten().tolist())[1:-1])
print('  # P5/32')
print('  -', str(anchors_p5.flatten().tolist())[1:-1])

# Optional: Print aspect ratios
print("\nAnchor aspect ratios:")
print([round(w/h, 2) for w, h in anchors])

# Optional: Print min/max
print(f"Min width: {whs[:,0].min():.5f}, Max width: {whs[:,0].max():.5f}")
print(f"Min height: {whs[:,1].min():.5f}, Max height: {whs[:,1].max():.5f}")
