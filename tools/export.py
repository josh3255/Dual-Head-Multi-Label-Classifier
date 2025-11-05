# tools/export.py
import os
import sys
import argparse
import torch


def parse_args():
    parser = argparse.ArgumentParser(description="Export Dual-Head Classifier to ONNX format")

    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint (.pth)")
    parser.add_argument("--onnx-output", type=str, required=True,
                        help="Output ONNX file path")
    parser.add_argument("--num-classes", type=int, default=4,
                        help="Number of classes (default: 4)")
    parser.add_argument("--input-size", type=int, nargs=3, default=[3, 224, 224],
                        help="Input tensor shape: C H W (default: 3 224 224)")
    parser.add_argument("--opset", type=int, default=17,
                        help="ONNX opset version (default: 17)")

    return parser.parse_args()


def load_model(checkpoint_path: str, num_classes: int):
    # src/model.py import 경로 설정
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
    SRC_DIR = os.path.join(PROJECT_ROOT, "src")
    sys.path.append(SRC_DIR)

    from model import DualHeadClassifier  # noqa: E402 (import after sys.path.append)

    model = DualHeadClassifier(num_classes=num_classes)
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model.eval()

    print(f"✅ Model loaded from: {checkpoint_path}")
    return model


def export_to_onnx(model, onnx_path: str, input_size: list, opset_version: int):
    dummy_input = torch.randn(1, *input_size)
    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=["input"],
        output_names=["headA", "headB"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "headA": {0: "batch_size"},
            "headB": {0: "batch_size"},
        },
        opset_version=opset_version,
    )

    print(f"🎉 ONNX model successfully exported to: {onnx_path}")


def main():
    args = parse_args()

    # 모델 로드
    model = load_model(args.checkpoint, args.num_classes)

    # Export 수행
    export_to_onnx(model, args.onnx_output, args.input_size, args.opset)


if __name__ == "__main__":
    main()
