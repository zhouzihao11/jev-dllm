"""Loopback-only, serial TypeSafe transport for the frozen inherited evaluator."""

import argparse
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from pathlib import Path
import sys
import time
from typing import cast

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import decode, option_count


class ServiceError(Exception):
    def __init__(self, status, code):
        self.status, self.code = status, code
        super().__init__(code)


def questions_for(payload, model_name):
    try:
        if not isinstance(payload, dict) or set(payload) != {"state", "model", "questions"}:
            raise ValueError
        if payload["model"] not in (model_name, "jev-latest"):
            raise ServiceError(400, "unknown_model")
        if not isinstance(payload["state"], (str, dict, list)):
            raise ValueError
        questions = payload["questions"]
        if not isinstance(questions, dict) or not questions:
            raise ValueError
        result = {}
        for name, original in questions.items():
            if not isinstance(name, str) or not name.strip() or not isinstance(original, dict):
                raise ValueError
            qdef = dict(original)
            if qdef.get("type") == "noul":
                qdef.setdefault("criteria", None)
            option_count(qdef)
            result[name] = qdef
        return result
    except (ValueError, TypeError, KeyError):
        raise ServiceError(400, "invalid_schema") from None


def answer_for(qdef, record):
    """Convert native index-ordered probabilities without rounding or renormalizing."""
    status = record["status"]
    if status in ("unsupported_source_mask", "unsupported_length"):
        raise ServiceError(422, status)
    if status != "ok":
        raise ServiceError(500, "inference_failed")
    probs, kind = record["probs"], qdef["type"]
    if kind == "noul":
        return {"type": kind, "noul": probs[1]}
    labels = list(qdef["criteria"]) if kind == "choice" else [str(i) for i in range(len(probs))]
    distribution = dict(zip(labels, probs))
    if kind == "choice":
        return {"type": kind, "choice": labels[record["pred_idx"]], "probabilities": distribution}
    return {"type": kind, "score": sum(i * p for i, p in enumerate(probs)),
            "probabilities": distribution}


class Engine:
    def __init__(self, args):
        from adapters.inherited import Adapter

        options = argparse.Namespace(**vars(args))
        options.backend, options.batch_size = "dllm", 1
        options.model_path = str(Path(options.model_path).expanduser().resolve())
        if not Path(options.model_path).is_dir():
            raise ValueError("model-path must be a local checkpoint directory")
        if not options.model_name.strip() or options.max_length < 1:
            raise ValueError("model-name and positive max-length are required")
        self.model_name = options.model_name
        self.adapter = Adapter(options)
        self.runtime = dict(self.adapter.runtime)
        self.runtime.update(protocol="jevbench::v1.4", backend=self.adapter.evaluator.backend,
                            probs_source="native", readout="shared Yes/No, T=1",
                            batching="serial independent questions", score_semantics="expected zero-based index")

    def health(self):
        return {"status": "ready", "model": self.model_name, "runtime": self.runtime}

    def predict(self, payload):
        questions = questions_for(payload, self.model_name)
        answers, tokens = {}, 0
        for index, (name, qdef) in enumerate(questions.items()):
            records, _ = self.adapter.predict_batch([{"state": payload["state"], "qdef": qdef}], index)
            record = records[0]
            answers[name] = answer_for(qdef, record)
            tokens += record["length"]
        return {"model": self.model_name, "answers": answers,
                "usage": {"input_tokens": tokens, "output_tokens": 0}, "runtime": self.runtime}


class Handler(BaseHTTPRequestHandler):
    @property
    def service(self):
        return cast("Server", self.server)

    def log_message(self, format, *args):
        pass

    def send_json(self, status, payload):
        body = json.dumps(payload, ensure_ascii=True, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True
        self.wfile.write(body)

    def send_error(self, code, message=None, explain=None):
        self.send_json(code, {"error": "http_error"})

    def do_GET(self):
        if self.path == "/health":
            self.send_json(200, self.service.engine.health())
        else:
            self.send_json(404, {"error": "not_found"})

    def do_POST(self):
        started, status, kinds = time.perf_counter(), 500, []
        try:
            if self.path != "/v1/systemone":
                raise ServiceError(404, "not_found")
            lengths = self.headers.get_all("Content-Length", [])
            if len(lengths) != 1 or not lengths[0].isascii() or not lengths[0].isdigit():
                raise ServiceError(400, "invalid_body_length")
            if self.headers.get("Transfer-Encoding"):
                raise ServiceError(400, "unsupported_transfer_encoding")
            try:
                length = int(lengths[0])
            except ValueError:
                raise ServiceError(400, "invalid_body_length") from None
            if length > self.service.max_body_bytes:
                raise ServiceError(413, "body_too_large")
            if self.headers.get_content_type() != "application/json":
                raise ServiceError(400, "expected_json")
            try:
                raw = self.rfile.read(length)
                if len(raw) != length:
                    raise ValueError
                payload = decode(raw.decode("utf-8"))
            except (ValueError, UnicodeError, RecursionError):
                raise ServiceError(400, "invalid_json") from None
            questions = questions_for(payload, self.service.engine.model_name)
            kinds = [q["type"] for q in questions.values()]
            response = self.service.engine.predict(payload)
            status = 200
        except ServiceError as exc:
            status, response = exc.status, {"error": exc.code}
        except Exception as exc:
            status, response = 500, {"error": "inference_failed"}
            print(json.dumps({"event": "inference_error", "exception_type": type(exc).__name__}), flush=True)
        try:
            self.send_json(status, response)
        finally:
            print(json.dumps({"event": "request", "types": kinds, "status": status,
                              "seconds": time.perf_counter() - started}), flush=True)


class Server(HTTPServer):
    engine: Engine
    max_body_bytes: int

    def handle_error(self, request, client_address):
        print('{"event":"transport_error"}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max-body-bytes", type=int, default=16 * 1024 * 1024)
    args = parser.parse_args()
    if not 0 <= args.port <= 65535 or args.max_body_bytes < 1 or args.max_length < 1:
        parser.error("invalid port, body limit, or context limit")
    print('{"event":"startup"}', flush=True)
    engine = Engine(args)
    with Server(("127.0.0.1", args.port), Handler) as server:
        server.engine, server.max_body_bytes = engine, args.max_body_bytes
        print(json.dumps({"event": "ready", "host": "127.0.0.1", "port": server.server_port,
                          **engine.health()}), flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
