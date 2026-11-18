"""The node's weights (weights.py) and its --download-only and --no-download, without a GPU, the real weights or the
internet: the files, their checksums and sources, and the node's start up to the end of its weights step.

The download itself is the core's xblick.weights, which tests it in its own tests/test_weights.py. Downloads here go
to a local HTTP server: URL files directly, Hugging Face files through HF_ENDPOINT. Where network use must not
happen, any use fails the test. The part from "the same in every node's test_weights.py" on is the same in every
node."""

import contextlib
import hashlib
import http.server
import io
import ipaddress
import os
import signal
import socket
import subprocess
import sys
import threading
from pathlib import Path

import pytest
from xblick import paths as xblick_paths
from xblick import weights as xblick_weights
from xblick.session import open_session

from xblick_tapnext import node, weights

TERMS_LINE = "TAPNext++ is the work of Google DeepMind"  # the first line of weights.TERMS
OPTIONS = ["--checkpoint", "256"]  # the options of a default start that select the weights
START_NEEDS_OPTIONS = True  # a start without options is an argument error

WITHOUT_SHA256 = ()  # files whose source states no SHA-256


def start_options(tmp_path) -> list[str]:
    """A start's required options."""
    return ["--camera", "test/camera", "--no-segmenter"]


def default_weights():
    return weights.needed("256")


@pytest.fixture
def stand_in(home, server, monkeypatch):
    """The default checkpoint as a stand-in on the local server: {its path: its content}."""
    server.files["/a.pt"] = DATA
    monkeypatch.setitem(weights.CHECKPOINTS, "256", ("a.pt", 256, f"{server.url}/a.pt", DATA_SHA, len(DATA)))
    return {home / "a.pt": DATA}


def test_the_files_are_the_checkpoints_with_their_sha256():
    assert [(w.path.name, w.source.url, w.sha256) for c in ("256", "512") for w in weights.needed(c)] == [
        ("tapnextpp_ckpt.pt", "https://storage.googleapis.com/dm-tapnet/tapnextpp/tapnextpp_ckpt.pt",
         "cb96a43444ccb4fbdb25d800b88c7ba196179a526e78f01b021a16b1c1eff6da"),
        ("tapnextpp_512.ckpt", "https://storage.googleapis.com/gresearch/tapnextpp/tapnextpp_512.ckpt",
         "6cd0e793fdcface3063d63f8ed3819bcf74c2c0468fe1fef85acee4de2f3609f")]
    assert all(w.path == Path(w.path.name) and w.size > 2.5e9 for w in weights.needed("512"))  # below the root
    assert weights.needed("/some/tapnext.pt") == []  # a file of one's own: nothing to download


def test_download_only_takes_the_checkpoint_option(home, stand_in, monkeypatch, capsys):
    monkeypatch.setitem(weights.CHECKPOINTS, "512", ("b.pt", 512, weights.CHECKPOINTS["256"][2], DATA_SHA,
                                                     len(DATA)))
    assert node.main(["--download-only", "--checkpoint", "512"]) == 0
    assert (home / "b.pt").read_bytes() == DATA and not (home / "a.pt").exists()
    assert node.main(["--download-only", "--checkpoint", "/some/tapnext.pt"]) == 0
    assert "no weights to download" in capsys.readouterr().out



# ------------------------------------------------------------------------ the same in every node's test_weights.py

DATA = os.urandom(3 * 1024 * 1024 + 17)  # a stand-in weight file, larger than a few read chunks
DATA_SHA = hashlib.sha256(DATA).hexdigest()


class Server:
    """A local HTTP server of files: path -> bytes. ``/resolve/`` paths redirect to ``/cdn/`` (as Hugging Face's do);
    ``status``: every GET answered with it. Records each GET and its Authorization header."""

    def __init__(self):
        self.files: dict[str, bytes] = {}
        self.gets: list[tuple[str, str | None]] = []
        self.status: int | None = None
        owner = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                owner.gets.append((self.path, self.headers.get("Authorization")))
                if owner.status:
                    self.send_error(owner.status)
                    return
                if "/resolve/" in self.path:
                    self.send_response(302)
                    self.send_header("Location", "/cdn/" + self.path.rsplit("/", 1)[1])
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                data = owner.files.get(self.path.removeprefix("/cdn"))
                if data is None:
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                with contextlib.suppress(OSError):
                    self.wfile.write(data)

        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.httpd.daemon_threads = True
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def server():
    s = Server()
    yield s
    s.close()


@pytest.fixture
def no_network(monkeypatch):
    """Any network use of this process fails the test."""
    def refuse(*args, **kwargs):
        raise AssertionError(f"network use: {args}")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket.socket, "connect_ex", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    real_getaddrinfo = socket.getaddrinfo

    def numeric_only(host, *args, **kwargs):
        """An address given as numbers resolves without the network (the core's session looks up the hosts of
        --connect, to tell whether they are this PC); a name would need DNS."""
        try:
            ipaddress.ip_address(host)
        except ValueError:
            refuse(host, *args)
        return real_getaddrinfo(host, *args, **kwargs)

    monkeypatch.setattr(socket, "getaddrinfo", numeric_only)


def use_store(monkeypatch, required=None, former=None):
    """The node's STORE as it is, with other required paths (code from the setup) or former places."""
    store = weights.STORE
    monkeypatch.setattr(weights, "STORE", xblick_weights.Store(
        store.prog, store.root, store.terms, store.required if required is None else required, repo=store.repo,
        former=store.former if former is None else former))


@pytest.fixture
def home(tmp_path, monkeypatch):
    """The STORE's root (the lock goes there) in tmp_path, no former place, no code from the setup required (stand_in
    may); XBLICK_HOME in tmp_path too, so that nothing reaches this PC's cache; a token that is no login of this PC."""
    root = tmp_path / "weights"
    monkeypatch.setenv("XBLICK_HOME", str(tmp_path / "xblick-home"))
    monkeypatch.setattr(weights, "STORE", xblick_weights.Store(weights.PROG, root, weights.TERMS, (),
                                                               repo=weights.REPO))
    monkeypatch.setenv("HF_TOKEN", "secret")
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    return root


def command(*words) -> str:
    """The download command that the node's lines name."""
    return " ".join([weights.PROG, "--download-only", *OPTIONS, *words])


def place(files: dict[Path, bytes]):
    for path, data in files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)


def test_the_store_is_the_nodes_cache_directory_with_its_former_place():
    store = weights.STORE
    assert (store.prog, store.node, store.terms, store.required, store.repo) == (
        weights.PROG, weights.NODE, weights.TERMS, tuple(weights.REQUIRED), weights.REPO)
    assert store.prog == node.main.__module__.split(".")[0].replace("_", "-") == f"xblick-{weights.NODE}"
    assert store.root == xblick_paths.cache_dir(weights.NODE) and store.former == (weights.FORMER,)
    assert all(not w.path.is_absolute() for w in default_weights())  # below the root: they move with it


def test_xblick_home_moves_the_cache_directory_and_creates_nothing(tmp_path):
    env = {**os.environ, "XBLICK_HOME": str(tmp_path)}
    code = f"from {weights.__name__} import STORE; print(STORE.root)"
    root = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True)
    assert root.stdout.strip() == str(tmp_path / "cache" / weights.NODE) and not list(tmp_path.iterdir())


def test_weights_dir_is_where_the_weights_go(home, stand_in, tmp_path, capsys):
    there = tmp_path / "elsewhere"
    assert node.main(["--download-only", "--weights-dir", str(there)]) == 0
    assert all((there / path.relative_to(home)).read_bytes() == data for path, data in stand_in.items())
    assert not home.exists() and str(there) in capsys.readouterr().out
    with pytest.raises(SystemExit) as exit_:
        node.main(["--download-only", "--weights-dir", "${XBLICK_NO_SUCH_VARIABLE}/w"])
    assert exit_.value.code == 2 and "--weights-dir" in capsys.readouterr().err


def test_the_former_place_is_named_not_used_and_its_command_moves_it(home, stand_in, tmp_path, monkeypatch,
                                                                      no_network, capsys):
    old = tmp_path / "checkout" / "weights"
    place({old / path.relative_to(home): data for path, data in stand_in.items()})
    use_store(monkeypatch, former=[old])
    with pytest.raises(xblick_weights.WeightsError, match="--no-download"):
        weights.STORE.ensure(default_weights(), download=False)
    lines = capsys.readouterr().out.splitlines()
    assert "still at the node's former place. They are not used from there." in lines[0]
    assert lines[1].strip().startswith("mkdir -p ") and f"mv {old} {home}" in lines[1]
    subprocess.run(["bash", "-c", lines[1]], check=True)  # the command as printed
    assert weights.STORE.ensure(default_weights(), verify=True) == [] and not old.exists()
    assert all(path.read_bytes() == data for path, data in stand_in.items())


def test_every_file_of_the_node_has_its_exact_size_and_sha256():
    for w in default_weights():
        assert isinstance(w, xblick_weights.Weight) and w.size > 0, w.path.name
        assert (w.sha256 is None) == (w.path.name in WITHOUT_SHA256), w.path.name
        assert w.sha256 is None or len(w.sha256) == 64


def test_all_present_touches_neither_network_nor_disk(home, stand_in, no_network):
    place(stand_in)
    before = sorted(home.rglob("*"))
    out = io.StringIO()
    assert weights.STORE.ensure(default_weights(), out=out) == [] and out.getvalue() == ""
    assert sorted(home.rglob("*")) == before  # no lock file either


def test_download_only_with_all_present_prints_one_line(home, stand_in, no_network, capsys):
    place(stand_in)
    assert node.main(["--download-only"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 1 and "nothing to download" in lines[0]


def test_download_only_downloads_with_the_terms_and_exits_0(home, stand_in, capsys):
    assert node.main(["--download-only"]) == 0
    assert all(path.read_bytes() == data for path, data in stand_in.items())
    out = capsys.readouterr().out
    assert TERMS_LINE in out and f"weights complete: {len(stand_in)} downloaded" in out.splitlines()[-1]
    assert not list(home.rglob("*.part"))


def test_download_only_needs_no_start_options_and_excludes_no_download(capsys):
    with pytest.raises(SystemExit) as exit_:
        node.main(["--download-only", "--no-download"])
    assert exit_.value.code == 2 and "not allowed with" in capsys.readouterr().err
    if START_NEEDS_OPTIONS:
        with pytest.raises(SystemExit) as exit_:
            node.main(["--no-download"])  # a start needs its required options
        assert exit_.value.code == 2 and "required" in capsys.readouterr().err


class Started(Exception):
    """The node got past its weights: what follows (the model, the GPU) is not this test's."""


@pytest.fixture
def started(monkeypatch, tmp_path):
    """The STORE's ensure raising Started once it returned (the node's start stops there, before anything touches
    the GPU: the tests need none); a Zenoh session the node connects to; the node's parameters kept."""
    from xblick import node as core_node

    ensure = weights.STORE.ensure

    def ensure_then_stop(*args, **kwargs):
        ensure(*args, **kwargs)
        raise Started

    monkeypatch.setattr(weights.STORE, "ensure", ensure_then_stop)
    kept = []

    class Kept(core_node.Parameters):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            kept.append(self)

    monkeypatch.setattr(core_node, "Parameters", Kept)
    sessions = []
    opened = node.open_from_args

    def keep_session(args):
        sessions.append(opened(args))
        return sessions[-1]

    monkeypatch.setattr(node, "open_from_args", keep_session)
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    session = open_session(listen=[f"127.0.0.1:{port}"])
    sigterm = signal.getsignal(signal.SIGTERM)  # node.start sets its own
    yield [*start_options(tmp_path), "--connect", f"127.0.0.1:{port}"], kept
    for closable in (*kept, *sessions):  # a start that got past the weights left them open (Started)
        with contextlib.suppress(Exception):  # closed already by a node that ended
            closable.close()
    signal.signal(signal.SIGTERM, sigterm)
    session.close()


def test_no_download_ends_at_once_naming_the_missing_files(home, server, stand_in, started, capsys):
    start, kept = started
    assert node.main([*start, "--no-download"]) == 1
    err = capsys.readouterr().err.strip().splitlines()
    assert len(err) == 1 and all(path.name in err[0] for path in stand_in) and command() in err[0]
    assert server.gets == [] and not any(path.exists() for path in stand_in)
    # a read-only parameter, as every start option; --download-only is a mode of the command line, no parameter
    assert kept[0].descriptor("no_download").read_only and kept[0]["no_download"] is True
    assert "download_only" not in kept[0].names()
    assert kept[0].descriptor("weights_dir").read_only and kept[0]["weights_dir"] == ""  # --weights-dir: a parameter


def test_no_download_with_all_present_starts_as_usual(home, stand_in, started, no_network):
    start, _ = started
    place(stand_in)
    with pytest.raises(Started):
        node.main([*start, "--no-download"])


def test_a_start_downloads_what_is_missing_first(home, stand_in, started):
    start, _ = started
    with pytest.raises(Started):
        node.main(start)
    assert all(path.read_bytes() == data for path, data in stand_in.items())
