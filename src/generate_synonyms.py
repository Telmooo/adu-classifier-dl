from typing import Optional, Tuple

import re

def parse_synonym(line : str) -> Optional[Tuple[str, str]]:
    matches = re.match("^(\[[A-Z]+\]) (.*) (SYNONYMOUS_OF) (.*)$", line)
    if matches is not None:
        _pos_tag, word, _, synonym = matches.groups()
        return word, synonym
    return None

if __name__ == "__main__":
    import os
    import json
    from tqdm import tqdm
    from collections import defaultdict

    OUT_DIR = "../out"
    os.makedirs(OUT_DIR, exist_ok=True)
    
    lines = []
    with open("../data/synonyms.txt", mode="r", encoding="utf-8") as f:
        lines = [line.strip("\n") for line in f.readlines()]

    synonym_dict = defaultdict(list)
    for line in tqdm(lines):
        ret = parse_synonym(line)
        if ret is not None:
            word, synonym = ret
            synonym_dict[word].append(synonym)

    json_obj = json.dumps(synonym_dict, indent=2, ensure_ascii=False)

    with open(f"{OUT_DIR}/synonyms.json", mode="w", encoding="utf-8") as f:
        f.write(json_obj)