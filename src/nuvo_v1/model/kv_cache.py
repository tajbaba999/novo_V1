class KVCache:
    """Stores (k, v) tuples per transformer layer.

    Usage:
        cache = KVCache(n_layers)
        # Prefill whole prompt
        model(prompt_ids, cache=cache, start_pos=0)
        # Decode one token at a time
        model(new_token_id, cache=cache, start_pos=prompt_len)
    """

    def __init__(self, n_layers: int):
        self.cache: list[tuple[object, object] | None] = [None] * n_layers

    def get(self, layer_idx: int):
        return self.cache[layer_idx]

    def update(self, layer_idx: int, kv: tuple) -> None:
        self.cache[layer_idx] = kv

    def reset(self) -> None:
        for i in range(len(self.cache)):
            self.cache[i] = None

    def __len__(self) -> int:
        return len(self.cache)
