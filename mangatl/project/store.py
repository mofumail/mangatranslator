"""A project file: one SQLite database holding pages, their history and their images.

Share / back up / resume a project by copying / opening this one file. Tables:
- pages: the current document of each page (opening never replays history)
- blobs: images and masks by content hash (an image stored twice is stored once)
- settings: the settings documents pages were made with (secrets excluded)
- groups / ops: the operation log. A group is one action (by the user, the pipeline or the
  agent); undo works per group or per operation. A pending group is a proposal (the agent's),
  applied or rejected operation by operation.
- snapshots: whole documents at points in the log (the pipeline's capture, and every
  SNAPSHOT_EVERY operations), to restore an earlier state without replaying everything.

WAL mode and one connection per thread, so parallel batch pages can write at the same time.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Literal

from mangatl.project import ops as oplog
from mangatl.project.model import Page

Author = Literal["user", "pipeline", "agent"]
SNAPSHOT_EVERY = 100
SCHEMA = """
create table if not exists meta (key text primary key, value text not null);
create table if not exists blobs (sha text primary key, mime text not null, data blob not null);
create table if not exists settings (id text primary key, doc text not null);
create table if not exists pages (id text primary key, doc text not null, updated real not null);
create table if not exists groups (
    id integer primary key, page text not null, label text not null, author text not null,
    status text not null, created real not null);
create table if not exists ops (
    seq integer primary key, grp integer not null references groups(id), page text not null,
    path text not null, old text, new text, state text not null default 'applied');
create index if not exists ops_by_page on ops(page, seq);
create table if not exists snapshots (page text not null, seq integer not null, doc text not null,
    primary key (page, seq));
"""
FORMAT_VERSION = "1"


class Project:
    def __init__(self, path: str | Path, create: bool = True):
        self.path = Path(path)
        if not create and not self.path.is_file():
            raise FileNotFoundError(f"no project file at {self.path}")
        self._local = threading.local()
        self._db().executescript(SCHEMA)  # executescript commits on its own: outside a transaction
        with self._tx() as db:
            db.execute("insert or ignore into meta values ('format', ?)", (FORMAT_VERSION,))

    # --- connection ---------------------------------------------------------------------------
    def _db(self) -> sqlite3.Connection:
        db = getattr(self._local, "db", None)
        if db is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            db = sqlite3.connect(self.path, timeout=60, isolation_level=None)
            db.execute("pragma journal_mode=wal")
            db.execute("pragma foreign_keys=on")
            self._local.db = db
        return db

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Connection]:
        """One write transaction (taking the write lock up front, so parallel writers queue)."""
        db = self._db()
        db.execute("begin immediate")
        try:
            yield db
        except BaseException:
            db.execute("rollback")
            raise
        db.execute("commit")

    def checkpoint(self) -> None:
        """Fold the write-ahead log into the file (a copied project is then complete)."""
        self._db().execute("pragma wal_checkpoint(truncate)")

    def close(self) -> None:
        db = getattr(self._local, "db", None)
        if db is not None:
            db.close()
            self._local.db = None

    # --- blobs and settings ---------------------------------------------------------------------
    def put_blob(self, data: bytes, mime: str = "image/png") -> str:
        sha = hashlib.sha256(data).hexdigest()
        with self._tx() as db:
            db.execute("insert or ignore into blobs values (?, ?, ?)", (sha, mime, data))
        return sha

    def get_blob(self, sha: str) -> bytes:
        row = self._db().execute("select data from blobs where sha = ?", (sha,)).fetchone()
        if row is None:
            raise KeyError(f"blob {sha}")
        return row[0]

    def put_settings(self, doc: str) -> str:
        sid = hashlib.sha256(doc.encode()).hexdigest()[:16]
        with self._tx() as db:
            db.execute("insert or ignore into settings values (?, ?)", (sid, doc))
        return sid

    def get_settings(self, sid: str) -> str:
        row = self._db().execute("select doc from settings where id = ?", (sid,)).fetchone()
        if row is None:
            raise KeyError(f"settings {sid}")
        return row[0]

    # --- pages ----------------------------------------------------------------------------------
    def page_ids(self) -> list[str]:
        return [r[0] for r in self._db().execute("select id from pages order by id")]

    def load_page(self, page_id: str) -> Page:
        row = self._db().execute("select doc from pages where id = ?", (page_id,)).fetchone()
        if row is None:
            raise KeyError(f"page {page_id}")
        return Page.model_validate_json(row[0])

    def save_page(self, page: Page, author: Author = "pipeline", label: str = "translated") -> int:
        """Store a whole page (the pipeline's capture): replaces the document, logs one group and
        a snapshot. Returns the group id."""
        doc = page.model_dump_json()
        now = time.time()
        with self._tx() as db:
            db.execute("insert or replace into pages values (?, ?, ?)", (page.id, doc, now))
            gid = db.execute(
                "insert into groups (page, label, author, status, created) values (?, ?, ?, 'applied', ?)",
                (page.id, label, author, now),
            ).lastrowid
            seq = db.execute("select coalesce(max(seq), 0) from ops").fetchone()[0]
            db.execute("insert or replace into snapshots values (?, ?, ?)", (page.id, seq, doc))
        return int(gid or 0)

    def export_json(self, page_id: str) -> str:
        return self.load_page(page_id).model_dump_json(indent=1)

    # --- operations -----------------------------------------------------------------------------
    def apply(self, page_id: str, ops: list[oplog.Op], author: Author = "user", label: str = "edit") -> int:
        """Apply operations to a page as one group; returns the group id."""
        return self._record(page_id, ops, author, label, "applied")

    def propose(
        self, page_id: str, ops: list[oplog.Op], author: Author = "agent", label: str = "proposal"
    ) -> int:
        """Record operations as a pending group, not applied (e.g. the agent's suggestion)."""
        return self._record(page_id, ops, author, label, "pending")

    def _record(self, page_id: str, ops: list[oplog.Op], author: str, label: str, status: str) -> int:
        with self._tx() as db:
            doc = self._doc(db, page_id)
            if status == "applied":
                oplog.apply(doc, ops)
            now = time.time()
            gid = db.execute(
                "insert into groups (page, label, author, status, created) values (?, ?, ?, ?, ?)",
                (page_id, label, author, status, now),
            ).lastrowid
            state = "applied" if status == "applied" else "pending"
            for op in ops:
                db.execute(
                    "insert into ops (grp, page, path, old, new, state) values (?, ?, ?, ?, ?, ?)",
                    (gid, page_id, json.dumps(list(op.path)), json.dumps(op.old), json.dumps(op.new), state),
                )
            if status == "applied":
                self._write_doc(db, page_id, doc, now)
        return int(gid or 0)

    def group_ops(self, group_id: int, state: str | None = None) -> list[tuple[int, oplog.Op]]:
        q = (
            "select seq, path, old, new from ops where grp = ?"
            + (" and state = ?" if state else "")
            + " order by seq"
        )
        args: tuple[Any, ...] = (group_id, state) if state else (group_id,)
        return [
            (seq, oplog.Op(tuple(json.loads(p)), json.loads(o), json.loads(n)))
            for seq, p, o, n in self._db().execute(q, args)
        ]

    def history(self, page_id: str) -> list[dict]:
        rows = self._db().execute(
            "select id, label, author, status, created from groups where page = ? order by id", (page_id,)
        )
        return [dict(id=r[0], label=r[1], author=r[2], status=r[3], created=r[4]) for r in rows]

    def region_history(self, page_id: str, region_id: str) -> list[dict]:
        """Every applied change to one region, oldest first: when, by whom, the group's label,
        the field (None: the whole region, added or removed), the value before and after."""
        rows = self._db().execute(
            "select o.path, o.old, o.new, g.author, g.label, g.created from ops o join groups g on g.id = o.grp"
            " where o.page = ? and o.state = 'applied' order by o.seq",
            (page_id,),
        )
        out = []
        for path, old, new, author, label, created in rows:
            p = json.loads(path)
            if len(p) >= 2 and p[0] == "regions" and p[1] == region_id:
                field = ".".join(p[2:]) or None
                out.append(
                    dict(
                        t=created,
                        author=author,
                        label=label,
                        field=field,
                        old=json.loads(old),
                        new=json.loads(new),
                    )
                )
        return out

    def undo_group(self, group_id: int) -> None:
        """Undo every applied operation of a group (atomically; Conflict if a later change
        touched the same fields)."""
        self._undo(group_id, None)

    def undo_op(self, seq: int) -> None:
        """Undo one operation (Conflict if its field was changed afterwards)."""
        row = self._db().execute("select grp from ops where seq = ?", (seq,)).fetchone()
        if row is None:
            raise KeyError(f"op {seq}")
        self._undo(row[0], seq)

    def _undo(self, group_id: int, only_seq: int | None) -> None:
        with self._tx() as db:
            page_id = db.execute("select page from groups where id = ?", (group_id,)).fetchone()[0]
            items = [(s, op) for s, op in self.group_ops(group_id, "applied") if only_seq in (None, s)]
            doc = self._doc(db, page_id)
            oplog.undo(doc, [op for _, op in items])
            for s, _ in items:
                db.execute("update ops set state = 'undone' where seq = ?", (s,))
            self._write_doc(db, page_id, doc, time.time())

    def accept(self, group_id: int, seqs: list[int] | None = None) -> None:
        """Apply a pending group's operations (all, or the chosen ones); Conflict if the page
        changed under them."""
        with self._tx() as db:
            page_id = db.execute("select page from groups where id = ?", (group_id,)).fetchone()[0]
            items = [(s, op) for s, op in self.group_ops(group_id, "pending") if seqs is None or s in seqs]
            doc = self._doc(db, page_id)
            oplog.apply(doc, [op for _, op in items], check=True)
            for s, _ in items:
                db.execute("update ops set state = 'applied' where seq = ?", (s,))
            self._settle(db, group_id)
            self._write_doc(db, page_id, doc, time.time())

    def reject(self, group_id: int, seqs: list[int] | None = None) -> None:
        with self._tx() as db:
            for s, _ in self.group_ops(group_id, "pending"):
                if seqs is None or s in seqs:
                    db.execute("update ops set state = 'rejected' where seq = ?", (s,))
            self._settle(db, group_id)

    def _settle(self, db: sqlite3.Connection, group_id: int) -> None:
        """A proposal stays pending while any operation is undecided; then it is applied when
        any operation was accepted, else rejected."""
        states = {r[0] for r in db.execute("select state from ops where grp = ?", (group_id,))}
        status = "pending" if "pending" in states else "applied" if "applied" in states else "rejected"
        db.execute("update groups set status = ? where id = ?", (status, group_id))

    def _doc(self, db: sqlite3.Connection, page_id: str) -> dict:
        row = db.execute("select doc from pages where id = ?", (page_id,)).fetchone()
        if row is None:
            raise KeyError(f"page {page_id}")
        return json.loads(row[0])

    def _write_doc(self, db: sqlite3.Connection, page_id: str, doc: dict, now: float) -> None:
        text = Page.model_validate(doc).model_dump_json()  # stays a valid page
        db.execute("update pages set doc = ?, updated = ? where id = ?", (text, now, page_id))
        seq = db.execute("select coalesce(max(seq), 0) from ops").fetchone()[0]
        last = db.execute(
            "select coalesce(max(seq), -1) from snapshots where page = ?", (page_id,)
        ).fetchone()[0]
        if seq - last >= SNAPSHOT_EVERY:
            db.execute("insert or replace into snapshots values (?, ?, ?)", (page_id, seq, text))
