import shutil
from pathlib import Path

import requests
from sentencepiece import SentencePieceProcessor, SentencePieceTrainer
from tqdm import tqdm


def download(url: str, path: Path | str) -> None:
    response = requests.get(url, stream=True)
    bar = tqdm(
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
        miniters=1,
        total=int(response.headers.get('content-length', 0)),
    )
    with open(path, "wb") as file:
        for chunk in response.iter_content(chunk_size=4096):
            file.write(chunk)
            bar.update(len(chunk))
            bar.refresh()
    bar.close()


def tokenize(
    input: Path | str,
    output: Path | str,
    size: int,
    pad: bool = False,
) -> None:
    input, output = Path(input), Path(output)
    SentencePieceTrainer.Train(
        input=input,
        model_prefix=output.parent / "_",
        vocab_size=size,
        normalization_rule_name='nfkc_cf',
        pad_id=1 if pad else -1,
        bos_id=-1,
        eos_id=-1,
        split_digits=True,
        input_sentence_size=1000000,
        shuffle_input_sentence=True,
    )
    shutil.move(output.parent / "_.vocab", output.parent / "vocabulary")
    text = input.read_text().split("\n")
    sp = SentencePieceProcessor(str(output.parent / "_.model"))  # type: ignore
    ids = "\n".join(" ".join(map(str, line)) for line in sp.Encode(text))
    output.write_text(ids)
    (output.parent / "_.model").unlink()