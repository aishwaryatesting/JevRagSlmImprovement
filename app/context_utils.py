def trim_evidence(passages, max_chars=4000, top_k=3):
    selected = []
    used = 0
    for passage in passages[:top_k]:
        if used >= max_chars:
            break
        chunk = passage[: max_chars - used]
        if chunk:
            selected.append(chunk)
            used += len(chunk)
    return selected
