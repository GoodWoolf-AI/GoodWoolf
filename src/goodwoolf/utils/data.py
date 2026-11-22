"""Shared helpers for preparing and inspecting parquet datasets.

Every dataset is stored as parquet. NTP datasets store documents in a text
column; SFT and RL datasets store source rows and build conversations at load
time. Tokenization happens at training time.
"""

import sys
from itertools import islice
from pathlib import Path

import torch

from goodwoolf.utils.models import DATASET_BASE
from goodwoolf.utils.system import ensure_dir, import_file


def load_data(data_name, split=None):
    """Return the dataset exposed by ``data/<vendor>/<dataset>/dataset.py``."""
    module_path = DATASET_BASE / data_name / "dataset.py"
    if not module_path.exists():
        raise FileNotFoundError(f"Dataset module not found at {module_path}")

    module_name = f"goodwoolf.data.{data_name.replace('/', '.')}.dataset"
    module = sys.modules.get(module_name) or import_file(str(module_path), module_name)
    return module.load_data(split)


def split_files(data_dir, splits, split=None):
    """Return the prepared ``.parquet`` files for one split, or for every split.

    ``val`` maps to ``validation`` when available, otherwise to ``test``.
    """
    data_dir = Path(data_dir)
    if split is None:
        names = splits
    else:
        name = split
        if split == "val":
            if "validation" in splits:
                name = "validation"
            elif "test" in splits:
                name = "test"
        if name not in splits:
            raise ValueError(f"Unknown split: {split!r} (available: {', '.join(splits)})")
        names = (name,)

    paths = []
    for name in names:
        path = data_dir / f"{name}.parquet"
        if not path.exists():
            raise FileNotFoundError(
                f"Dataset split not found at {path}; run with --prepare first"
            )
        paths.append(path)
    return paths


def write_text_splits(data_dir, splits, column="text"):
    """Write ``{split: documents}`` to parquet files with one document per row."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    data_dir = ensure_dir(data_dir)
    schema = pa.schema([(column, pa.string())])
    paths = []
    for split, documents in splits.items():
        path = data_dir / f"{split}.parquet"
        count = 0
        batch = []
        with pq.ParquetWriter(path, schema, compression="zstd") as writer:
            for text in documents:
                if text is None:
                    continue
                batch.append({column: text})
                count += 1
                if len(batch) >= 1024:
                    writer.write_table(pa.Table.from_pylist(batch, schema=schema))
                    batch = []
            if batch:
                writer.write_table(pa.Table.from_pylist(batch, schema=schema))
        paths.append(path)
        print(f"  Wrote {count:,} documents -> {path}", file=sys.stderr, flush=True)
    return paths


def text_metadata(files, column="text"):
    """Print document, character, and size statistics for text parquet files."""
    import pyarrow.parquet as pq

    total_docs = 0
    total_chars = 0
    total_bytes = 0

    print("Splits:")
    for path in files:
        parquet = pq.ParquetFile(path)
        chars = 0
        for batch in parquet.iter_batches(columns=[column], batch_size=1024):
            for text in batch.column(0).to_pylist():
                if text:
                    chars += len(text)
        size = path.stat().st_size
        docs = parquet.metadata.num_rows
        total_chars += chars
        total_docs += docs
        total_bytes += size
        print(f"  {path.stem:<12} {docs:>10,} docs  {chars:>12,} chars  {size / 1e6:>8.2f} MB")

    print(f"  {'total':<12} {total_docs:>10,} docs  {total_chars:>12,} chars  {total_bytes / 1e6:>8.2f} MB")


def read_parquet_texts(paths, column="text"):
    """Yield the text column of parquet files."""
    import pyarrow.parquet as pq

    for path in paths:
        parquet = pq.ParquetFile(path)
        for row_group in range(parquet.num_row_groups):
            for text in parquet.read_row_group(row_group).column(column).to_pylist():
                if text:
                    yield text


def read_parquet_rows(paths):
    """Yield rows from parquet files as dictionaries."""
    import pyarrow.parquet as pq

    for path in paths:
        parquet = pq.ParquetFile(path)
        for row_group in range(parquet.num_row_groups):
            yield from parquet.read_row_group(row_group).to_pylist()


def count_parquet_rows(paths):
    """Return the total row count of parquet files without reading their data."""
    import pyarrow.parquet as pq

    return sum(pq.ParquetFile(path).metadata.num_rows for path in paths)


def count_lines(paths):
    """Count non-empty text lines without loading the files into memory."""
    total = 0
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            total += sum(1 for line in f if line.strip())
    return total


def split_source_texts(paths, reader, count, train_ratio=0.8):
    """Return lazy train/test text iterables for source files.

    Multiple files use the last file as the test split. A single file is split
    by document count, which is provided by ``count`` and streamed lazily.
    """
    paths = list(paths)
    if not paths:
        raise FileNotFoundError("No source files to split")

    if len(paths) > 1:
        return reader(paths[:-1]), reader(paths[-1:])

    total = count(paths)
    if total < 2:
        raise ValueError(f"Not enough documents to split: {total}")
    split_index = min(total - 1, max(1, int(total * train_ratio)))
    return (
        islice(reader(paths), 0, split_index),
        islice(reader(paths), split_index, None),
    )


def render_mc(question, letters, choices):
    """Render a multiple-choice question the way nanochat does.

    The letter follows the choice with no whitespace (``- choice=A``) so the
    assistant target ``A`` is the exact same token as in the prompt.
    """
    query = f"Multiple Choice question: {question}\n"
    query += "".join(f"- {choice}={letter}\n" for letter, choice in zip(letters, choices))
    query += "\nRespond only with the letter of the correct answer."
    return query


def get_bos_token_id(tokenizer):
    getter = getattr(tokenizer, "get_bos_token_id", None)
    if callable(getter):
        token_id = getter()
        if token_id is not None:
            return token_id
    for attribute in ("bos_token_id", "eos_token_id", "pad_token_id"):
        token_id = getattr(tokenizer, attribute, None)
        if token_id is not None:
            return token_id
    return 0


def _message_text(message):
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    return "\n".join(part.get("text", "") for part in content if isinstance(part, dict))


def _chat_messages(messages):
    return [{"role": message["role"], "content": _message_text(message)} for message in messages]


def _encode_chat_text(tokenizer, text):
    try:
        return tokenizer.encode(text, add_special_tokens=False)
    except TypeError:
        return tokenizer.encode(text)


def render_conversation(tokenizer, conversation, max_tokens=2048):
    renderer = getattr(tokenizer, "render_conversation", None)
    if callable(renderer):
        return renderer(conversation, max_tokens=max_tokens)
    apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
    if not callable(apply_chat_template):
        raise ValueError("Tokenizer must provide render_conversation() or apply_chat_template()")
    messages = _chat_messages(conversation["messages"])
    full_text = apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    prompt_messages = messages[:-1] if messages and messages[-1]["role"] == "assistant" else messages
    prompt_text = apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=bool(prompt_messages))
    ids = _encode_chat_text(tokenizer, full_text)
    prompt_ids = _encode_chat_text(tokenizer, prompt_text)
    bos_token_id = get_bos_token_id(tokenizer)
    if bos_token_id is not None and (not ids or ids[0] != bos_token_id):
        ids = [bos_token_id] + ids
        prompt_ids = [bos_token_id] + prompt_ids
    prompt_len = 0
    for prompt_id, full_id in zip(prompt_ids, ids):
        if prompt_id != full_id:
            break
        prompt_len += 1
    mask = [0] * prompt_len + [1] * max(0, len(ids) - prompt_len)
    return ids[:max_tokens], mask[:max_tokens]


def render_prompt(tokenizer, messages, max_tokens=2048):
    renderer = getattr(tokenizer, "render_prompt", None)
    if callable(renderer):
        return renderer(messages, max_tokens=max_tokens)
    apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
    if not callable(apply_chat_template):
        raise ValueError("Tokenizer must provide render_prompt() or apply_chat_template()")
    text = apply_chat_template(_chat_messages(messages), tokenize=False, add_generation_prompt=True)
    ids = _encode_chat_text(tokenizer, text)
    bos_token_id = get_bos_token_id(tokenizer)
    if bos_token_id is not None and (not ids or ids[0] != bos_token_id):
        ids = [bos_token_id] + ids
    return ids[:max_tokens]


def record_text(record):
    """Return the plain text of a conversation record (used for statistics)."""
    chunks = []
    for message in record["messages"]:
        content = message["content"]
        if isinstance(content, str):
            chunks.append(content)
        else:
            chunks.extend(part["text"] for part in content)
    return "\n".join(chunks)


def write_parquet_rows(path, rows):
    """Write row dictionaries to ``path`` while inferring the parquet schema."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    writer = None
    schema = None
    batch = []
    count = 0

    def flush():
        nonlocal writer, schema, batch
        if not batch:
            return
        if schema is None:
            table = pa.Table.from_pylist(batch)
            schema = table.schema
            writer = pq.ParquetWriter(path, schema, compression="zstd")
        else:
            table = pa.Table.from_pylist(batch, schema=schema)
        writer.write_table(table)
        batch = []

    for row in rows:
        batch.append(row)
        count += 1
        if len(batch) >= 1024:
            flush()

    flush()
    if writer is None:
        pq.write_table(pa.table({}), path)
    else:
        writer.close()
    return count


def prepare_json_splits(data_dir, sources, splits, source_dirname="raw"):
    """Copy raw parquet source rows into one ``<split>.parquet`` per split."""
    data_dir = ensure_dir(data_dir)
    source_dir = data_dir / source_dirname
    if not source_dir.exists():
        raise FileNotFoundError(f"{source_dir} not found; run with --fetch first")

    for split in splits:
        paths = []
        for config, source_split in sources:
            if source_split != split:
                continue
            paths.extend(sorted((source_dir / config / split).glob("*.parquet")))
        count = write_parquet_rows(
            data_dir / f"{split}.parquet",
            read_parquet_rows(paths),
        )
        print(f"  Wrote {count:,} rows -> {data_dir / f'{split}.parquet'}", file=sys.stderr, flush=True)


class TextDataset:
    """Iterable dataset of text documents from parquet files."""

    def __init__(self, files, column="text"):
        self.files = [Path(path) for path in files]
        self.column = column
        if not self.files:
            raise ValueError("TextDataset requires parquet files")

    def __iter__(self):
        return read_parquet_texts(self.files, self.column)

    def __len__(self):
        return count_parquet_rows(self.files)


class JsonDataset:
    """Iterable dataset of conversations built from parquet source rows."""

    def __init__(self, files, conversation):
        self.files = [Path(path) for path in files]
        self.conversation = conversation
        if not self.files:
            raise ValueError("JsonDataset requires parquet files")

    def records(self):
        for row in read_parquet_rows(self.files):
            record = self.conversation(row)
            if record:
                yield record

    def __iter__(self):
        return self.records()

    def __len__(self):
        return count_parquet_rows(self.files)


class NTPDataLoader:
    """BOS-aligned best-fit batches from an iterable text dataset."""

    def __init__(
        self,
        dataset,
        tokenizer,
        batch_size,
        block_size,
        device,
        device_type,
        buffer_size=1000,
        tokenizer_batch_size=128,
        num_epochs=None,
    ):
        if num_epochs is not None and num_epochs <= 0:
            raise ValueError("num_epochs must be positive")
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.row_capacity = block_size + 1
        self.device = device
        self.device_type = device_type
        self.buffer_size = buffer_size
        self.tokenizer_batch_size = tokenizer_batch_size
        self.num_epochs = num_epochs
        self.bos_token_id = self._bos_token_id()
        self.epoch = 0
        self.record_index = 0
        self.iterator = iter(dataset)
        self.epoch_finished = False
        self.exhausted = False
        self.doc_buffer = []

    def _bos_token_id(self):
        return get_bos_token_id(self.tokenizer)

    @property
    def done(self):
        return self.exhausted and not self.doc_buffer

    def _next_text_from_epoch(self):
        try:
            text = next(self.iterator)
        except StopIteration:
            self.epoch_finished = True
            return None
        self.record_index += 1
        return text

    def _advance_epoch(self):
        if self.num_epochs is None:
            self.epoch += 1
        elif self.epoch + 1 >= self.num_epochs:
            self.exhausted = True
            return
        else:
            self.epoch += 1
        self.record_index = 0
        self.iterator = iter(self.dataset)
        self.epoch_finished = False

    def _next_texts(self):
        texts = []
        while len(texts) < self.tokenizer_batch_size:
            text = self._next_text_from_epoch()
            if text is None:
                if not texts and self.record_index == 0 and not self.exhausted:
                    raise ValueError("Dataset contains no documents")
                if self.epoch_finished:
                    self._advance_epoch()
                if texts:
                    break
                if self.exhausted:
                    return None
                continue
            texts.append(text)
        return texts if texts else None

    def _ensure_docs(self):
        while len(self.doc_buffer) < self.buffer_size:
            texts = self._next_texts()
            if texts is None:
                return bool(self.doc_buffer)
            encoded = self.tokenizer(texts, add_special_tokens=False)["input_ids"]
            for ids in encoded:
                self.doc_buffer.append([self.bos_token_id] + list(ids))
        return True

    def _build_batch(self, max_examples=None, mask_padding=False):
        if max_examples is not None and max_examples <= 0:
            raise ValueError("max_examples must be positive")
        rows, row_lengths = [], []
        docs_used = 0

        for row_idx in range(self.batch_size):
            row, content_len = [], None
            pos = 0
            while pos < self.row_capacity:
                if max_examples is not None and docs_used >= max_examples:
                    content_len = pos
                    break
                if not self._ensure_docs():
                    content_len = pos
                    break
                if not self.doc_buffer:
                    content_len = pos
                    break
                remaining = self.row_capacity - pos
                best_idx = -1
                best_len = 0
                for idx, doc in enumerate(self.doc_buffer):
                    if len(doc) <= remaining and len(doc) > best_len:
                        best_idx = idx
                        best_len = len(doc)
                if best_idx >= 0:
                    doc = self.doc_buffer.pop(best_idx)
                    take = min(len(doc), remaining)
                    row.extend(doc[:take])
                    pos += take
                    docs_used += 1
                    continue
                shortest_idx = min(range(len(self.doc_buffer)), key=lambda i: len(self.doc_buffer[i]))
                doc = self.doc_buffer.pop(shortest_idx)
                take = min(len(doc), remaining)
                row.extend(doc[:take])
                pos += take
                docs_used += 1
                content_len = self.row_capacity
                break
            if pos < self.row_capacity:
                row.extend([self.bos_token_id] * (self.row_capacity - pos))
                if content_len is None:
                    content_len = pos
            row_lengths.append(self.row_capacity if content_len is None or content_len < 0 else content_len)
            rows.append(row[:self.row_capacity])

        if docs_used == 0:
            return None
        rows_tensor = torch.tensor(rows, dtype=torch.long)
        inputs = rows_tensor[:, :-1]
        targets = rows_tensor[:, 1:]
        if mask_padding:
            targets = targets.clone()
            for row_idx, content_len in enumerate(row_lengths):
                if content_len < self.row_capacity:
                    targets[row_idx, max(0, content_len - 1):] = -1
        if self.device_type == "cuda":
            inputs = inputs.pin_memory().to(self.device, non_blocking=True)
            targets = targets.pin_memory().to(self.device, non_blocking=True)
        else:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
        return inputs, targets, docs_used, self.done

    def next_batch(self):
        batch = self._build_batch()
        if batch is None:
            raise ValueError("Dataset contains no documents")
        return batch[0], batch[1]

    def next_update_batch(self, max_examples=None):
        return self._build_batch(max_examples=max_examples, mask_padding=True)

    def state_dict(self):
        return {
            "num_epochs": self.num_epochs,
            "epoch": self.epoch,
            "record_index": self.record_index,
            "epoch_finished": self.epoch_finished,
            "exhausted": self.exhausted,
            "doc_buffer": [list(doc) for doc in self.doc_buffer],
        }

    def load_state_dict(self, state):
        self.num_epochs = state.get("num_epochs", self.num_epochs)
        self.epoch = state.get("epoch", 0)
        self.record_index = state.get("record_index", 0)
        self.epoch_finished = state.get("epoch_finished", False)
        self.exhausted = state.get("exhausted", False)
        self.doc_buffer = [list(doc) for doc in state.get("doc_buffer", [])]
        self.iterator = iter(self.dataset)
        for _ in range(self.record_index):
            try:
                next(self.iterator)
            except StopIteration:
                break


class SFTDataLoader:
    """Finite or infinite stream of packed SFT batches."""

    def __init__(
        self,
        dataset,
        tokenizer,
        max_seq_len,
        batch_size,
        buffer_size,
        device,
        device_type,
        num_epochs=None,
    ):
        if num_epochs is not None and num_epochs <= 0:
            raise ValueError("num_epochs must be positive")
        if max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if buffer_size <= 0:
            raise ValueError("buffer_size must be positive")
        self.dataset = dataset
        self.dataset_size = len(dataset)
        if self.dataset_size == 0:
            raise ValueError("Dataset contains no records")
        self.tokenizer = tokenizer
        self.max_seq_len = int(max_seq_len)
        self.row_capacity = self.max_seq_len + 1
        self.batch_size = int(batch_size)
        self.buffer_size = int(buffer_size)
        self.device = device
        self.device_type = device_type
        self.bos_token_id = get_bos_token_id(tokenizer)
        self.num_epochs = num_epochs
        self.epoch = 0
        self.record_index = 0
        self.iterator = iter(dataset)
        self._finished = False
        self.buffer = []

    @property
    def done(self):
        return self._finished and not self.buffer

    def _next_record(self):
        while True:
            try:
                return next(self.iterator)
            except StopIteration:
                if self.num_epochs is None:
                    self.epoch += 1
                    self.record_index = 0
                    self.iterator = iter(self.dataset)
                    continue
                if self.epoch + 1 >= self.num_epochs:
                    self._finished = True
                    return None
                self.epoch += 1
                self.record_index = 0
                self.iterator = iter(self.dataset)

    def _next_sample(self):
        if self._finished:
            return None
        record = self._next_record()
        if record is None:
            return None
        self.record_index += 1
        return render_conversation(self.tokenizer, record, max_tokens=self.max_seq_len)

    def _ensure_buffer(self):
        while len(self.buffer) < self.buffer_size and not self._finished:
            sample = self._next_sample()
            if sample is None:
                break
            ids, mask = sample
            if ids:
                self.buffer.append((ids, mask))

    def next_batch(self, max_examples=None):
        if max_examples is not None and max_examples <= 0:
            raise ValueError("max_examples must be positive")
        rows, mask_rows, row_lengths = [], [], []
        items_used = 0

        for _ in range(self.batch_size):
            row, mask_row, content_len = [], [], None
            pos = 0
            while pos < self.row_capacity:
                if max_examples is not None and items_used >= max_examples:
                    content_len = pos
                    break
                self._ensure_buffer()
                if not self.buffer:
                    content_len = pos
                    break
                remaining = self.row_capacity - pos
                best_idx = -1
                best_len = 0
                for idx, (ids, _) in enumerate(self.buffer):
                    if len(ids) <= remaining and len(ids) > best_len:
                        best_idx = idx
                        best_len = len(ids)
                if best_idx >= 0:
                    ids, mask = self.buffer.pop(best_idx)
                    take = min(len(ids), remaining)
                    row.extend(ids[:take])
                    mask_row.extend(mask[:take])
                    pos += take
                    items_used += 1
                    continue
                shortest_idx = min(range(len(self.buffer)), key=lambda i: len(self.buffer[i][0]))
                ids, mask = self.buffer.pop(shortest_idx)
                take = min(len(ids), remaining)
                row.extend(ids[:take])
                mask_row.extend(mask[:take])
                pos += take
                items_used += 1
                content_len = self.row_capacity
                break
            if pos < self.row_capacity:
                pad = self.row_capacity - pos
                row.extend([self.bos_token_id] * pad)
                mask_row.extend([0] * pad)
                if content_len is None:
                    content_len = pos
            row_lengths.append(self.row_capacity if content_len is None or content_len < 0 else content_len)
            rows.append(row[:self.row_capacity])
            mask_rows.append(mask_row[:self.row_capacity])

        if items_used == 0:
            return None
        use_cuda = self.device_type == "cuda"
        batch_tensor = torch.tensor(rows, dtype=torch.long, pin_memory=use_cuda)
        inputs = batch_tensor[:, :-1].to(self.device, non_blocking=use_cuda)
        targets = batch_tensor[:, 1:].to(self.device, non_blocking=use_cuda).clone()
        mask_tensor = torch.tensor(mask_rows, dtype=torch.bool, device=self.device)
        targets[~mask_tensor[:, 1:]] = -1
        for row_idx, content_len in enumerate(row_lengths):
            if content_len < self.row_capacity:
                targets[row_idx, max(0, content_len - 1):] = -1
        return inputs, targets, items_used, self.done

    def state_dict(self):
        return {
            "num_epochs": self.num_epochs,
            "epoch": self.epoch,
            "record_index": self.record_index,
            "finished": self._finished,
            "buffer": [(list(ids), list(mask)) for ids, mask in self.buffer],
        }

    def load_state_dict(self, state):
        self.num_epochs = state.get("num_epochs", self.num_epochs)
        self.epoch = state.get("epoch", 0)
        self.record_index = state.get("record_index", 0)
        self._finished = state.get("finished", False)
        self.buffer = [(list(ids), list(mask)) for ids, mask in state.get("buffer", [])]
        self.iterator = iter(self.dataset)
        for _ in range(self.record_index):
            try:
                next(self.iterator)
            except StopIteration:
                break


class RLDataLoader:
    """Dataset iterator for RL prompts and reference answers."""

    def __init__(self, dataset, tokenizer):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.epoch = 0
        self.record_index = 0
        self.iterator = iter(dataset)

    def _next_record(self):
        while True:
            try:
                record = next(self.iterator)
            except StopIteration:
                self.epoch += 1
                self.record_index = 0
                self.iterator = iter(self.dataset)
                continue
            self.record_index += 1
            return record

    def next_batch(self):
        conversation = self._next_record()
        messages = list(conversation["messages"])
        if messages and messages[-1]["role"] == "assistant":
            messages.pop()
        prompt_ids = render_prompt(self.tokenizer, messages)
        answer = conversation["answer"]
        return prompt_ids, answer

    def state_dict(self):
        return {
            "epoch": self.epoch,
            "record_index": self.record_index,
        }

    def load_state_dict(self, state):
        self.epoch = state.get("epoch", 0)
        self.record_index = state.get("record_index", 0)
        self.iterator = iter(self.dataset)
        for _ in range(self.record_index):
            try:
                next(self.iterator)
            except StopIteration:
                break


