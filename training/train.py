"""Training entry point placeholder for the labelled VV/VH dataset phase."""
import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="training/data")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--output", default="backend/app/ml/weights/oil_unet.pt")
    parser.parse_args()
    raise SystemExit("Training scaffold is ready; add labelled VV/VH patches to run the configured U-Net pipeline.")


if __name__ == "__main__":
    main()
