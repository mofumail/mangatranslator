"""The translator's local llama-server: start, health, restart, model switch.

The engine owns one; `ensure(server, vision)` returns once a server with exactly that
configuration answers (starting it, or restarting it when the model, the vision projector or
its flags changed, or when it died).
"""

from __future__ import annotations

import shlex
import signal
import subprocess
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path

from mangatl.settings import ROOT

BINARY = ROOT / "bin" / "llama.cpp" / "llama-server"
LOG_DIR = ROOT / "logs"


class LlmServerError(RuntimeError):
    pass


def server_healthy(base_url: str) -> bool:
    root = base_url.rstrip("/").removesuffix("/v1")
    try:
        with urllib.request.urlopen(f"{root}/health", timeout=2) as r:
            return r.status == 200
    except (urllib.error.URLError, OSError):
        return False


def command(server, vision: bool, alias: str, binary: Path = BINARY) -> list[str]:
    """llama-server's command line for ServerSettings `server`."""
    cmd = [
        str(binary),
        "-m",
        str(server.gguf),
        "--alias",
        alias,
        "--host",
        "127.0.0.1",
        "--port",
        str(server.port),
        "-c",
        str(server.ctx),
        "-np",
        "1",
        "-ngl",
        "99",
        "-fa",
        "on",
        "--reasoning",
        "off",  # thinking burns tokens without improving translations
        "--no-webui",
    ]
    if vision:
        # Gemma 4 vision needs the whole image in one ubatch
        cmd += ["--mmproj", str(server.mmproj), "-b", "2048", "-ub", "2048"]
    else:
        cmd += ["--no-mmproj"]
    if server.cpu_moe:  # MoE models: keep expert weights of the first N layers in system RAM
        cmd += ["--n-cpu-moe", str(server.cpu_moe)]
    extra = shlex.split(server.extra_args)
    if "--fit" in extra or "-fit" in extra:  # let llama.cpp choose the GPU layer count
        i = cmd.index("-ngl")
        del cmd[i : i + 2]
    return cmd + extra


class LlamaServer:
    def __init__(
        self,
        binary: Path = BINARY,
        log_dir: Path = LOG_DIR,
        log: Callable[[str], None] = print,
        start_timeout: float = 300,
    ):
        self.binary = binary
        self.log_dir = log_dir
        self.log = log
        self.start_timeout = start_timeout
        self._lock = threading.RLock()
        self._proc: subprocess.Popen | None = None
        self._cmd: list[str] | None = None
        self._url: str | None = None
        self._logf = None
        self.restarts = 0

    @property
    def url(self) -> str | None:
        """The OpenAI-compatible base URL of the running server."""
        return self._url

    def running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def ensure(self, server, vision: bool, repo: str = "") -> str:
        """A healthy server with this configuration; returns its base URL. `repo`: where to get
        a missing vision projector (for the error message)."""
        from mangatl import presets

        if not self.binary.exists():
            raise LlmServerError(f"llama-server not found at {self.binary}")
        if not Path(server.gguf).exists():
            raise LlmServerError(f"LLM weights not found: {server.gguf}")
        if vision and not Path(server.mmproj).exists():
            raise LlmServerError(
                f"Vision projector not found: {server.mmproj}\n"
                f"Download mmproj-F16.gguf from https://huggingface.co/{repo} to that path."
            )
        cmd = command(server, vision, presets.model_alias(server.gguf), self.binary)
        with self._lock:
            if self.running() and cmd == self._cmd and self._url and server_healthy(self._url):
                return self._url
            if self._proc is not None:  # another model / flags, or it died
                if not self.running():
                    self.log(f"llama-server exited (code {self._proc.returncode}); restarting")
                    self.restarts += 1
                self.stop()
            return self._start(cmd, f"http://127.0.0.1:{server.port}/v1", vision)

    def _start(self, cmd: list[str], url: str, vision: bool) -> str:
        if server_healthy(url):
            raise LlmServerError(
                f"Something is already listening on {url}. Pass --llm-url {url} to reuse it, or --port "
                "to pick another."
            )
        self.log_dir.mkdir(parents=True, exist_ok=True)
        log_path = self.log_dir / "llama-server.log"
        self.log(f"Starting llama-server ({'vision' if vision else 'text-only'}), log: {log_path}")
        self._logf = open(log_path, "w")  # noqa: SIM115
        self._proc = subprocess.Popen(
            cmd, stdout=self._logf, stderr=subprocess.STDOUT, start_new_session=True
        )
        self._cmd, self._url = cmd, None
        deadline = time.time() + self.start_timeout
        while time.time() < deadline:
            if self._proc.poll() is not None:
                code = self._proc.returncode
                self.stop()
                raise LlmServerError(f"llama-server exited early (code {code}); see {log_path}")
            if server_healthy(url):
                self.log("llama-server ready")
                self._url = url
                return url
            time.sleep(1)
        self.stop()
        raise LlmServerError(f"llama-server did not become healthy in time; see {log_path}")

    def stop(self) -> None:
        with self._lock:
            proc, self._proc, self._cmd, self._url = self._proc, None, None, None
            if proc is not None and proc.poll() is None:
                self.log("Stopping llama-server")
                proc.send_signal(signal.SIGINT)
                try:
                    proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
            if self._logf is not None:
                self._logf.close()
                self._logf = None

    def status(self) -> dict:
        with self._lock:
            return {
                "running": self.running(),
                "url": self._url,
                "healthy": bool(self._url and server_healthy(self._url)),
                "pid": self._proc.pid if self._proc is not None else None,
                "restarts": self.restarts,
            }
