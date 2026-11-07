"""Local, offline classroom demo. Run: python demo.py, then http://127.0.0.1:8765."""
import argparse
import json
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch
from src.artifacts import load_bundle, inference_loader
from src.engine import predict

ROOT = Path(__file__).resolve().parent
NAMES = {'mlp':'MLP (TF-IDF)', 'textcnn':'TextCNN', 'bilstm':'BiLSTM + Attention', 'distilbert':'DistilBERT'}
BUNDLES = {}
LOCK = threading.Lock()


def classify(text):
    output = []
    with LOCK:
        for name, label in NAMES.items():
            if name not in BUNDLES:
                BUNDLES[name] = load_bundle(ROOT/'results'/name, 'cpu')
            model, spec, info = BUNDLES[name]
            probabilities, _, _ = predict(model, inference_loader([text], spec, info), torch.device('cpu'), amp=False)
            probability = float(probabilities[0])
            output.append({'model':label, 'sentiment':'Positive' if probability >= .5 else 'Negative',
                           'probability_positive':probability})
    return output


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def json_response(self, value, status=200):
        payload = json.dumps(value).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(payload)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        if self.path == '/api/results':
            return self.json_response([json.loads((ROOT/'results'/name/'metrics.json').read_text()) for name in NAMES])
        if self.path == '/':
            self.path = '/demo/index.html'
        # Serve only demo pages, reports and figures, not downloaded data or model weights.
        if not self.path.startswith(('/demo/', '/docs/', '/results/')) or '..' in self.path:
            return self.send_error(404)
        if self.path.startswith('/results/') and not self.path.endswith(('.png', '.csv', '.json')):
            return self.send_error(404)
        return super().do_GET()

    def do_POST(self):
        if self.path != '/api/predict':
            return self.send_error(404)
        origin = self.headers.get('Origin')
        if origin and origin not in {f'http://127.0.0.1:{self.server.server_port}', f'http://localhost:{self.server.server_port}'}:
            return self.json_response({'error':'Open the demo on localhost.'}, 403)
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if not 0 < length <= 20000:
                return self.json_response({'error':'Enter a review of at most 4,000 characters.'}, 400)
            request = json.loads(self.rfile.read(length))
            text = request.get('text', '')
            if not isinstance(text, str) or not text.strip() or len(text) > 4000:
                return self.json_response({'error':'Enter a review of 1 to 4,000 characters.'}, 400)
            self.json_response({'predictions':classify(text)})
        except (ValueError, json.JSONDecodeError):
            self.json_response({'error':'Invalid request.'}, 400)
        except Exception as exc:
            print(f'Prediction error: {exc}', flush=True)
            self.json_response({'error':'Could not load a trained bundle. Check the server window and results folder.'}, 500)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8765)
    args = parser.parse_args()
    torch.set_num_threads(2)
    server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    print(f'IMDB demo ready: http://127.0.0.1:{args.port}', flush=True)
    print('Keep this window open. Press Ctrl+C to stop. No training or downloads occur.', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
