"""Real CPU/offline smoke and portable class-map digests for service parity."""

from __future__ import annotations
import argparse, hashlib, json, os, socket, time
from pathlib import Path
import numpy as np


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    os.environ["DEVICE"] = "cpu"
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    from competition.service_bridge import describe_models, predict_chip
    from competition.data import discover_chips

    state = describe_models(args.model_dir)
    if not state["available"]:
        raise RuntimeError(state["reason"])

    def deny_network(*args, **kwargs):
        raise RuntimeError("Network blocked for offline smoke")

    socket.create_connection = deny_network
    socket.socket.connect = deny_network
    socket.socket.connect_ex = deny_network
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "bundle_id": state["bundle_id"],
        "device": "cpu",
        "network_connections_blocked": True,
        "masks": [],
    }
    for chip_id in sorted(discover_chips(args.data_dir)):
        start = time.monotonic()
        result = predict_chip(args.data_dir, chip_id, model_dir=args.model_dir)
        mask = np.ascontiguousarray(result["class_map"])
        expected = (256, 256) if chip_id.startswith("AF_") else (512, 512)
        classes = {0, 1} if chip_id.startswith("AF_") else {0, 1, 2, 3}
        if (
            mask.dtype != np.uint8
            or mask.shape != expected
            or not set(np.unique(mask)).issubset(classes)
        ):
            raise ValueError(chip_id)
        np.save(output / (chip_id + ".npy"), mask, allow_pickle=False)
        report["masks"].append(
            {
                "chip_id": chip_id,
                "shape": list(mask.shape),
                "dtype": str(mask.dtype),
                "sha256_c_order_uint8": hashlib.sha256(
                    mask.tobytes(order="C")
                ).hexdigest(),
                "counts": {str(c): int((mask == c).sum()) for c in sorted(classes)},
                "seconds": time.monotonic() - start,
                "provenance": result["provenance"],
            }
        )
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
