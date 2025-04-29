import json, re, sys
from pathlib import Path
from itertools import groupby

SRC   = "TC_label-studio_input_file.json"    # your file
DEST  = "tasks_seq.json"                     # output for sequence tagging
HOST  = "http://localhost:8082"              # where your PNGs are served
LAYOUTLM_DEST = "layoutlm_data.json"         # output for LayoutLM format

def normalize_space(txt: str) -> str:
    """Collapse multiple spaces and strip."""
    return re.sub(r"\s+", " ", txt).strip()

def build_plain_text_with_bbox(results):
    """
    Pick textarea items, sort them by top-to-bottom (value['y']),
    then left-to-right (value['x']), turn into lines.
    Preserve bounding box information for each token.
    """
    # grab every textarea
    regions = [
        r for r in results
        if r["type"] == "textarea" and r["from_name"] == "transcription"
    ]
    # sort
    regions.sort(key=lambda r: (r["value"]["y"], r["value"]["x"]))
    
    # concatenate words into lines (simple y-bucket grouping)
    lines = []
    tokens_with_bbox = []
    
    for _, bucket in groupby(regions, key=lambda r: round(r["value"]["y"], 1)):
        line_words = []
        for reg in sorted(bucket, key=lambda r: r["value"]["x"]):
            text = reg["value"]["text"][0]
            words = text.split()
            line_words.extend(words)
            
            # For LayoutLM we need bbox for each token
            bbox = reg["value"]
            x = bbox["x"] / 100  # convert from percentage to normalized [0,1]
            y = bbox["y"] / 100
            width = bbox["width"] / 100
            height = bbox["height"] / 100
            
            # If this is a multi-word region, we'll simplify by using the same bbox for all tokens
            # For a more accurate approach, you would need to estimate individual token widths
            for word in words:
                tokens_with_bbox.append({
                    "text": word,
                    "bbox": [x, y, x + width, y + height]  # [x0, y0, x1, y1] format for LayoutLM
                })
                
        lines.append(" ".join(line_words))
    
    return "\n".join(normalize_space(l) for l in lines), tokens_with_bbox

def main():
    tasks_seq = []
    layoutlm_data = []
    data = json.loads(Path(SRC).read_text(encoding="utf-8"))

    for task in data:
        img_url = task["data"]["ocr"]              # already full URL
        # fallback if you later store only filename:
        if not img_url.startswith("http"):
            img_url = f"{HOST}/{img_url}"
            
        results = task["predictions"][0]["result"]
        plain_text, tokens_with_bbox = build_plain_text_with_bbox(results)

        # Regular sequence tagging format
        tasks_seq.append({
            "data": {
                "text": plain_text,
                "image": img_url        # keep image for visual aid (optional)
            }
        })
        
        # LayoutLM format
        layoutlm_data.append({
            "image": img_url,
            "text": plain_text,
            "tokens": tokens_with_bbox,
            "label": []  # Will be filled after annotation
        })

    Path(DEST).write_text(json.dumps(tasks_seq, ensure_ascii=False, indent=2), encoding="utf-8")
    Path(LAYOUTLM_DEST).write_text(json.dumps(layoutlm_data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✔  Wrote {len(tasks_seq)} tasks to {DEST}")
    print(f"✔  Wrote {len(layoutlm_data)} LayoutLM formatted documents to {LAYOUTLM_DEST}")

if __name__ == "__main__":
    main()
