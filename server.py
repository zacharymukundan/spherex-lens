#!/usr/bin/env python3
"""Serve the SPHEREx Sky Flipbook and fetch real SPHEREx data on demand.

Setup:  pip install flask numpy scipy astropy astroquery
Run:    python server.py        then open http://localhost:8000
Keep index.html, spherex-flipbook.html, fetch_spherex.py and this file together.

Data requests run as background jobs so the page can show progress:
GET /api/cube/start?ra=&dec=&fov=   start or join a job -> {id, state, progress, stage, url}
GET /api/cube/status/<id>           poll it; state is queued, running, done or error
GET /api/cube/file/<id>             the finished data (JSON)
Jobs run one at a time in a queue so many visitors cannot overload IRSA or this machine.
Results are cached in ./cache for 24 hours, so new SPHEREx releases appear within a day.
"""
import os, json, math, time, threading
from concurrent.futures import ThreadPoolExecutor
from flask import Flask, request, jsonify, send_from_directory
import fetch_spherex as fs

app = Flask(__name__)
CACHE, MAX_AGE = "cache", 24 * 3600
os.makedirs(CACHE, exist_ok=True)
pool = ThreadPoolExecutor(max_workers=1)
JOBS, jl = {}, threading.Lock()
fresh = lambda p: os.path.exists(p) and time.time() - os.path.getmtime(p) < MAX_AGE


@app.route("/")
@app.route("/index.html")
def index():
    return send_from_directory(".", "index.html")


@app.route("/spherex-flipbook.html")
def flipbook():
    return send_from_directory(".", "spherex-flipbook.html")


def clean(o):
    """Browsers reject NaN and Infinity in JSON, so turn any that slip through into null."""
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, dict):
        return {k: clean(x) for k, x in o.items()}
    if isinstance(o, list):
        return [clean(x) for x in o]
    return o


def run_job(jid, path, ra, dec, fov):
    job = JOBS[jid]
    job["state"] = "running"
    tmp, marker = path + ".tmp", path + ".nogaia"
    try:
        if os.path.exists(path):  # images are already saved: only the Gaia stars failed last time
            with open(path) as fh:
                data = json.load(fh)
            job.update(progress=0.5, stage="Retrying the Gaia stars...")
            data["gaia"], data["gaia_error"] = fs.gaia(fs.parse(data["ra"], data["dec"]), data["fov"])
        else:
            data = fs.build(ra, dec, fov, progress=lambda frac, msg: job.update(progress=round(frac, 3), stage=msg))
            if not data["frames"]:
                raise RuntimeError("No SPHEREx images cover this position yet.")
        with open(tmp, "w") as fh:
            json.dump(clean(data), fh, allow_nan=False)
        os.replace(tmp, path)
        if data.get("gaia_error"):
            open(marker, "w").close()
        elif os.path.exists(marker):
            os.remove(marker)
        job.update(state="done", progress=1, stage="Done")
    except Exception as e:
        job.update(state="error", error=str(e))


def view(jid):
    j = JOBS[jid]
    return jsonify(id=jid, state=j["state"], progress=j.get("progress", 0), stage=j.get("stage", ""),
                   error=j.get("error"), url=f"api/cube/file/{jid}" if j["state"] == "done" else None)


@app.route("/api/cube/start")
def start():
    try:
        ra, dec = request.args["ra"], request.args["dec"]
        fov = max(1.0, min(float(request.args.get("fov", 5)), 10.0))
        c = fs.parse(ra, dec)
    except ValueError as e:
        return jsonify(error=str(e)), 400
    except Exception:
        return jsonify(error="Check the coordinates and field of view."), 400
    jid = f"v3_{c.ra.deg:.4f}_{c.dec.deg:.4f}_{fov:g}"  # v2: ignores files cached before the NaN fix
    path = os.path.join(CACHE, jid + ".json")
    with jl:
        j = JOBS.get(jid)
        marker = path + ".nogaia"  # present when the saved data is missing its Gaia stars
        gaia_retry = os.path.exists(marker) and time.time() - os.path.getmtime(marker) > 600
        if fresh(path) and not gaia_retry:
            JOBS[jid] = {"state": "done", "progress": 1, "stage": "Done"}
        elif not j or j["state"] in ("error", "done"):  # new, failed before, or cache expired
            JOBS[jid] = {"state": "queued", "progress": 0, "stage": "Waiting for another request to finish..."}
            pool.submit(run_job, jid, path, ra, dec, fov)
    return view(jid)


@app.route("/api/cube/status/<jid>")
def status(jid):
    return view(jid) if jid in JOBS else (jsonify(error="Unknown job."), 404)


@app.route("/api/cube/file/<jid>")
def file(jid):
    return send_from_directory(CACHE, os.path.basename(jid) + ".json")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8000)), threaded=True)
