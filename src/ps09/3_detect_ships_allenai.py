#!/usr/bin/env python3
import argparse, csv, sys, time
import requests

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, help="Path to CSV with header 'image_name'")
    p.add_argument("--url", default="http://localhost:5557/detections")
    p.add_argument("--raw_path", default="/home/vessel_detection/data")
    p.add_argument("--output_dir", default="/home/vessel_detection/data/output")
    p.add_argument("--conf", type=float, default=0.9)
    p.add_argument("--nms_thresh", type=float, default=10.0)
    p.add_argument("--save_crops", type=lambda v: v.lower()=="true", default=True)
    p.add_argument("--window_size", type=int, default=2048)
    p.add_argument("--padding", type=int, default=400)
    p.add_argument("--overlap", type=int, default=20)
    p.add_argument("--detector_batch_size", type=int, default=4)
    p.add_argument("--postprocessor_batch_size", type=int, default=32)
    p.add_argument("--timeout", type=int, default=1800)  # seconds per scene
    p.add_argument("--sleep", type=float, default=0.5)   # pause between requests
    args = p.parse_args()

    successes, failures = [], []
    with open(args.csv, newline="") as f:
        reader = csv.DictReader(f)
        if "image_name" not in reader.fieldnames:
            print("CSV missing 'image_name' header.", file=sys.stderr)
            sys.exit(1)

        for row in reader:
            scene_id = row["image_name"].strip()
            if not scene_id:
                continue
            payload = {
                "scene_id": scene_id,
                "raw_path": args.raw_path,
                "output_dir": args.output_dir,
                "conf": args.conf,
                "nms_thresh": args.nms_thresh,
                "save_crops": args.save_crops,
                "window_size": args.window_size,
                "padding": args.padding,
                "overlap": args.overlap,
                "detector_batch_size": args.detector_batch_size,
                "postprocessor_batch_size": args.postprocessor_batch_size,
                "avoid": False,
                "remove_clouds": True,
                "force_cpu": False,
            }
            try:
                r = requests.post(args.url, json=payload, timeout=args.timeout)
                if r.ok:
                    successes.append(scene_id)
                    print(f"[OK] {scene_id}")
                else:
                    failures.append(scene_id)
                    print(f"[FAIL {r.status_code}] {scene_id}")
            except Exception as e:
                failures.append(scene_id)
                print(f"[ERROR] {scene_id}: {e}")
            time.sleep(args.sleep)

    print(f"\nDone. {len(successes)} succeeded, {len(failures)} failed.")
    if failures:
        print("Failed scenes:", ", ".join(failures))

if __name__ == "__main__":
    main()