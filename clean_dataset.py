import json
import re

NUM_RE = re.compile(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?")

def round_svg_numbers(svg_text, ndigits=2):
    def _r(m):
        v = round(float(m.group(0)), ndigits)
        return str(int(v)) if v == int(v) else str(v)
    return NUM_RE.sub(_r, svg_text)

def sort_svg_attributes(svg_text):
    def sort_attrs(match):
        tag = match.group(1)
        attrs_str = match.group(2)
        closing = match.group(3)
        
        attrs = re.findall(r'([a-zA-Z0-9\-:]+)="([^"]*)"', attrs_str)
        # Custom order: id, d, fill, stroke, transform, viewBox, everything else alphabetically
        priority = {"id": 0, "d": 1, "fill": 2, "stroke": 3, "transform": 4, "viewBox": 5}
        attrs.sort(key=lambda x: (priority.get(x[0], 100), x[0]))
        
        new_attrs_str = " ".join([f'{k}="{v}"' for k, v in attrs])
        if new_attrs_str:
            return f"<{tag} {new_attrs_str}{closing}"
        else:
            return f"<{tag}{closing}"
            
    return re.sub(r'<([a-zA-Z0-9\-:]+)\s+([^>]*?)(/?>)', sort_attrs, svg_text)

def clean_dataset(in_path, out_path):
    count = 0
    with open(in_path, 'r') as fin, open(out_path, 'w') as fout:
        for line in fin:
            if not line.strip(): continue
            try:
                data = json.loads(line)
                if "svg" in data:
                    svg = data["svg"]
                    # 1. Round to 2 decimal places
                    svg = round_svg_numbers(svg, 2)
                    # 2. Sort attributes
                    svg = sort_svg_attributes(svg)
                    data["svg"] = svg
                fout.write(json.dumps(data) + '\n')
                count += 1
            except Exception as e:
                print("Error on line:", e)
    print(f"Cleaned {count} SVGs into {out_path}")

if __name__ == "__main__":
    clean_dataset("dataset_raw.jsonl", "dataset_cleaned.jsonl")
