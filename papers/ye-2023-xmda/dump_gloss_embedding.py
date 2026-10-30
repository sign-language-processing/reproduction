"""Save a teacher's GlossEmbeddings state_dict as SignXmDA gloss_pre_embedding.

build_model_mixup() loads model.gloss_pre_embedding with
GlossEmbeddings.load_state_dict(torch.load(path)); the teacher checkpoint holds
the same module under the 'gloss_embed.' prefix of 'model_state'.
"""

import argparse

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("out")
    args = parser.parse_args()

    state = torch.load(args.checkpoint, map_location="cpu")["model_state"]
    prefix = "gloss_embed."
    embedding = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
    if not embedding:
        raise KeyError(f"no {prefix}* parameters in {args.checkpoint}")
    torch.save(embedding, args.out)
    print({k: list(v.shape) for k, v in embedding.items()})


if __name__ == "__main__":
    main()
