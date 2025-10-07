import argparse
import jax

from naive.data import CIFAR100Dataset, DataLoader
from naive.model import UNet
from naive.trainer import Trainer


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser("Naive Jax Diffusion Training")

    parser.add_argument("--num-epochs", type=int, default=50)
    parser.add_argument("--dataset", type=str, default="datasets/normal/cifar-100-python")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--momentum", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    trainer = Trainer(
        num_epochs=args.num_epochs,
        lr=args.lr,
        momentum=args.momentum,
    )

    random_key = jax.random.PRNGKey(args.seed)
    model = UNet(
        random_key=random_key,
        embed_args={
            "num_embeds": args.timesteps,
            "embed_dim": 384,
        },
        down_args={
            "num_blocks": 3,
            "kernel_sizes": [7, 3, 3],
            "channels": [[3, 192], [192, 192], [192, 384]],
            "strides": [1, 1, 1],
            "paddings": "SAME",
        },
        middle_args={
            "num_blocks": 1,
            "kernel_size": [3],
            "channels": [[384, 384]],
            "strides": [1],
            "paddings": "SAME",
        },
        up_args={
            "num_blocks": 3,
            "kernel_size": [3, 3, 3],
            "channels": [[384, 384], [384, 192], [192, 3]],
            "strides": 1,
            "paddings": "SAME",
        },
        conv_args={
            "kernel_size": 3,
            "channels": [3, 3],
            "stride": 1,
            "padding": "SAME",
        },
    )

    dataset = CIFAR100Dataset(
        cifa100_dir=args.dataset,
        split="train",
        random_key=random_key,
    )
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
    )
    trainer.run(model, data_loader)


if __name__ == "__main__":

    args = parse_args()
    print(args)
    main(args)
