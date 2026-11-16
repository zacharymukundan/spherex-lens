#!/usr/bin/env python3
"""Fetch real SPHEREx spectral-image cutouts for one sky position.

Command line:  python fetch_spherex.py 133.79513 -7.24515 --fov 5 --out wise0855.json
Also used by server.py (import fetch_spherex; fetch_spherex.build(ra, dec, fov)).

Setup:  pip install numpy scipy astropy astroquery
Follows the IRSA tutorial "Download a collection of SPHEREx Spectral Image cutouts".
Every cutout is resampled onto one north-up, east-left grid (6.15"/px) centred on the target.
"""
import argparse, base64, csv, io, json, re, time, concurrent.futures as cf
import numpy as np
import requests
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from astropy.utils.data import conf
from astroquery.ipac.irsa import Irsa
from scipy.ndimage import map_coordinates

conf.remote_timeout = 40
SC = 6.15  # arcsec per output pixel (SPHEREx native scale)


_SEX = re.compile(r"^([+-]?)(\d{1,3})[:hd]\s*(\d{1,2})(?:[:m]\s*(\d{1,2}(?:\.\d+)?)s?)?$", re.I)


def _angle(text, is_ra):
    """Degrees from decimal degrees or h:m:s / d:m:s. Raises ValueError with a readable message when out of range."""
    name, t = ("RA" if is_ra else "Dec"), str(text).strip().replace(" ", "")
    if not t:
        raise ValueError(f"{name} is empty.")
    m = _SEX.match(t)
    if m:
        sign, d, mi, sec = m.groups()
        d, mi, sec = int(d), int(mi), float(sec or 0)
        if mi >= 60 or sec >= 60:
            raise ValueError(f"{name}: minutes and seconds must be under 60.")
        v = d + mi / 60 + sec / 3600
        if is_ra and v >= 24:
            raise ValueError(f"RA must be under 24 hours (you entered {d}h).")
        if not is_ra and v > 90:
            raise ValueError("Dec must be between -90 and +90 degrees.")
        v = -v if sign == "-" else v
        return v * 15 if is_ra else v
    try:
        v = float(t)
    except ValueError:
        raise ValueError(f"Couldn't read {name}. Use degrees (133.795) or h:m:s (08:55:10.8).")
    if is_ra and not 0 <= v < 360:
        raise ValueError("RA in degrees must be from 0 to 360.")
    if not is_ra and abs(v) > 90:
        raise ValueError("Dec must be between -90 and +90 degrees.")
    return v


def parse(ra, dec):
    return SkyCoord(_angle(ra, True) * u.deg, _angle(dec, False) * u.deg)


def grid(c, n):
    j, i = np.meshgrid(np.arange(n), np.arange(n))
    k = (n - 1) / 2
    xi = -(j - k) * SC / 206264.806
    eta = -(i - k) * SC / 206264.806
    a0, d0 = c.ra.rad, c.dec.rad
    rho = np.hypot(xi, eta)
    cc = np.arctan(rho)
    with np.errstate(invalid="ignore", divide="ignore"):
        t = np.where(rho > 0, eta * np.sin(cc) * np.cos(d0) / rho, 0)
    dec = np.arcsin(np.cos(cc) * np.sin(d0) + t)
    ra = a0 + np.arctan2(xi * np.sin(cc), rho * np.cos(d0) * np.cos(cc) - eta * np.sin(d0) * np.sin(cc))
    return np.degrees(ra) % 360, np.degrees(dec)


def work(row, c, gra, gdec):
    try:
        with fits.open(row["uri"], cache=False) as h:
            hdr = h[1].header
            img = h[1].data.astype("f4")
            w = WCS(hdr)
            x, y = w.world_to_pixel(c)
            sw = WCS(hdr, fobj=h, key="W")
            sw.sip = None
            wl = float(sw.pixel_to_world(x, y)[0].to(u.um).value)
            if not np.isfinite(wl):  # target falls outside this cutout: use the cutout's centre instead
                wl = float(sw.pixel_to_world(img.shape[1] / 2, img.shape[0] / 2)[0].to(u.um).value)
            if not np.isfinite(wl):
                return None
            flagged = float((h["FLAGS"].data != 0).mean())
            if not np.isfinite(flagged):
                flagged = 0.0
            ver = str(h[0].header.get("VERSION", ""))
        px, py = w.all_world2pix(gra, gdec, 0)
        out = map_coordinates(img, [py, px], order=1, mode="constant", cval=np.nan).astype("f4")
        if not np.isfinite(out).any():
            return None
        return dict(mjd=float(row["mjd"]), det=int(str(row["band"])[-1]) - 1, wl=wl, ver=ver,
                    fl=round(flagged, 4), b=base64.b64encode(out.tobytes()).decode())
    except Exception as e:  # transient read errors: skip this cutout
        print("skip:", type(e).__name__, row["uri"][-60:])
        return None


# Gaia DR3 from three independent servers, asked directly over HTTP so failures show a real reason.
# IRSA's Gator catalogue service goes first: it is separate from IRSA's TAP service, which has been failing.
_TAP = [
    ("VizieR", "https://tapvizier.cds.unistra.fr/TAPVizieR/tap/sync", '"I/355/gaiadr3"',
     ["RA_ICRS", "DE_ICRS", "pmRA", "pmDE", "Gmag"]),
    ("ESA", "https://gea.esac.esa.int/tap-server/tap/sync", "gaiadr3.gaia_source",
     ["ra", "dec", "pmra", "pmdec", "phot_g_mean_mag"]),
]


def _num(v):
    try:
        x = float(v)
        return x if np.isfinite(x) else 0.0
    except (TypeError, ValueError):
        return 0.0


def _gator(c, radius_deg, timeout=(12, 60)):
    r = requests.get("https://irsa.ipac.caltech.edu/cgi-bin/Gator/nph-query", timeout=timeout, params={
        "catalog": "gaia_dr3_source", "spatial": "Cone", "objstr": f"{c.ra.deg} {c.dec.deg}",
        "radius": radius_deg * 3600, "radunits": "arcsec", "outfmt": 1,
        "selcols": "ra,dec,pmra,pmdec,phot_g_mean_mag", "constraints": "phot_g_mean_mag<20"})
    r.raise_for_status()
    if 'stat="ERROR"' in r.text[:500]:
        m = re.search(r'msg="([^"]+)"', r.text)
        raise RuntimeError(m.group(1)[:60] if m else "Gator error")
    t = Table.read(r.text, format="ascii.ipac")
    return [[float(x["ra"]), float(x["dec"]), _num(x["pmra"]), _num(x["pmdec"]), _num(x["phot_g_mean_mag"])] for x in t]


def _tap_csv(url, table, cols, c, radius_deg, timeout=(12, 45)):
    q = (f"SELECT {','.join(cols)} FROM {table} "
         f"WHERE 1=CONTAINS(POINT('ICRS',{cols[0]},{cols[1]}),CIRCLE('ICRS',{c.ra.deg},{c.dec.deg},{radius_deg})) "
         f"AND {cols[4]}<20")
    r = requests.post(url, data={"REQUEST": "doQuery", "LANG": "ADQL", "FORMAT": "csv", "QUERY": q}, timeout=timeout)
    r.raise_for_status()
    rows = []
    for rec in csv.DictReader(io.StringIO(r.text)):
        try:
            rows.append([float(rec[cols[0]]), float(rec[cols[1]]), _num(rec[cols[2]]), _num(rec[cols[3]]), _num(rec[cols[4]])])
        except (KeyError, ValueError, TypeError):
            continue
    return rows


def gaia(c, fov, tries=2):
    """Gaia DR3 stars (G < 20) in the field. Returns (rows, error); error is None on success, else a short reason."""
    radius = fov / 60 / 2 * 1.5
    sources = [("IRSA", lambda: _gator(c, radius))] + [
        (n, (lambda u=u, t=t, cl=cl: _tap_csv(u, t, cl, c, radius))) for n, u, t, cl in _TAP]
    errs = []
    for name, fn in sources:
        for attempt in range(tries):
            try:
                rows = fn()
                print(f"Gaia DR3 via {name}: {len(rows)} sources")
                return rows, None
            except requests.HTTPError as e:
                code = e.response.status_code if e.response is not None else "?"
                errs.append(f"{name}: HTTP {code}")
                print(f"Gaia DR3 via {name} failed: HTTP {code}")
                if isinstance(code, int) and code < 500:
                    break  # the server rejected the request itself; trying again will not help
            except requests.ConnectTimeout:
                errs.append(f"{name}: could not connect")
                print(f"Gaia DR3 via {name}: could not connect within 12 s")
                break  # the server is unreachable from here; retrying will not help
            except requests.Timeout:
                errs.append(f"{name}: no answer in time")
                print(f"Gaia DR3 via {name}: connected but no answer in time")
                break
            except Exception as e:
                errs.append(f"{name}: {type(e).__name__} {str(e)[:40]}".strip())
                print(f"Gaia DR3 via {name} failed: {type(e).__name__}: {e}")
                break
            time.sleep(2)
    return [], "; ".join(dict.fromkeys(errs))[:110]


def query_irsa(q, tries=6, wait=20):
    """Run the TAP query, retrying (and switching to an async job) when IRSA returns errors."""
    for attempt in range(tries):
        try:
            return Irsa.query_tap(q, async_job=attempt >= 1).to_table()
        except Exception as e:
            print(f"IRSA query failed ({type(e).__name__}); retry {attempt + 1}/{tries} in {wait} s")
            time.sleep(wait)
    raise RuntimeError("The SPHEREx archive (IRSA) is not answering right now. Try again later.")


def spherex_collections():
    """Every SPHEREx image collection IRSA offers right now, so new data releases are picked up automatically."""
    try:
        t = Irsa.list_collections()
        col = "collection" if "collection" in t.colnames else t.colnames[0]
        names = sorted({str(x) for x in t[col] if str(x).lower().startswith("spherex")})
        return sorted(set(names) | {"spherex_dr1"})  # always keep the collection known to work
    except Exception as e:
        print("Could not list collections, using the default:", e)
    return ["spherex_dr1"]


def sia_rows(c, fov, size, tries=3, wait=8):
    """Image-search route (works when the TAP table service is down). Returns cutout rows, or None on failure."""
    radius = fov / np.sqrt(2) * 1.1 * u.arcmin
    rows, seen, reached = [], set(), False
    for coll in spherex_collections():
        res = None
        for attempt in range(tries):
            try:
                res = Irsa.query_sia(pos=(c, radius), collection=coll)
                break
            except Exception as e:
                print(f"IRSA image search of {coll} failed ({type(e).__name__}); retry {attempt + 1}/{tries}")
                time.sleep(wait)
        if res is None:
            continue
        reached = True
        for r in res:
            try:
                band, url, mjd = str(r["energy_bandpassname"]), str(r["access_url"]), float(r["t_min"])
            except Exception:
                continue
            key = (round(mjd, 5), band)  # the same exposure can appear in two releases: keep it once
            if not band.startswith("SPHEREx-D") or not np.isfinite(mjd) or url in seen or key in seen:
                continue
            seen.update([url, key])
            sep = "&" if "?" in url else "?"
            rows.append(dict(uri=f"{url}{sep}center={c.ra.deg},{c.dec.deg}d&size={size}", mjd=mjd, band=band, base=url))
    return rows if reached else None


def overview(rows, c, fov, factor=3):
    """One wide view (factor x the field) of the earliest image, for the viewer's 'where am I' inset."""
    cand = [r for r in rows if isinstance(r, dict) and "base" in r]
    if not cand:
        return None
    r0 = min(cand, key=lambda r: r["mjd"])
    n2, size = int(round(fov * factor * 60 / SC)), fov * factor / 60 * 1.5
    sep = "&" if "?" in r0["base"] else "?"
    wide = dict(r0, uri=f"{r0['base']}{sep}center={c.ra.deg},{c.dec.deg}d&size={size}")
    gra, gdec = grid(c, n2)
    f = work(wide, c, gra, gdec)
    return dict(n=n2, mjd=f["mjd"], det=f["det"], wl=f["wl"], b=f["b"]) if f else None


def build(ra, dec, fov=5.0, use_gaia=True, tries=6, wait=20, progress=None):
    """Return the cube dict the web page loads. progress(fraction, message) is optional."""
    p = progress or (lambda f, m: None)
    c = parse(ra, dec)
    n = int(round(fov * 60 / SC))
    size = fov / 60 * 1.5  # degrees; margin so rotated frames still cover the grid
    q = f"""
    SELECT 'https://irsa.ipac.caltech.edu/' || a.uri || '?center={c.ra.deg},{c.dec.deg}d&size={size}' AS uri,
           p.time_bounds_lower AS mjd, p.energy_bandpassname AS band
    FROM spherex.artifact a JOIN spherex.plane p ON a.planeid = p.planeid
    WHERE 1 = CONTAINS(POINT('ICRS', {c.ra.deg}, {c.dec.deg}), p.poly)"""
    print("Searching the SPHEREx archive...")
    p(0.02, "Searching the SPHEREx archive...")
    res = sia_rows(c, fov, size)
    if res is None:  # image search unavailable: fall back to the TAP table query
        print("Trying the TAP table query instead...")
        res = query_irsa(q, tries, wait)
        res = res[[str(b).startswith("SPHEREx-D") for b in res["band"]]]
    print(f"{len(res)} spectral images overlap this position; downloading cutouts...")
    p(0.1, f"Downloading {len(res)} images...")
    gra, gdec = grid(c, n)
    frames, done = [], 0
    ex = cf.ThreadPoolExecutor(max_workers=10)
    futs = [ex.submit(work, r, c, gra, gdec) for r in res]
    try:
        for fut in cf.as_completed(futs):
            done += 1
            p(0.1 + 0.8 * done / len(futs), f"Downloaded {done} of {len(futs)} images...")
            f = fut.result()
            if f:
                frames.append(f)
            if done % 20 == 0:
                print(f"{done} of {len(futs)} cutouts processed ({len(frames)} usable)")
            if done >= 20 and not frames:  # nothing usable after 20 tries: the archive is failing, stop waiting
                raise RuntimeError("The archive is returning errors for the image downloads. Try again later.")
    finally:
        ex.shutdown(wait=False, cancel_futures=True)
    frames.sort(key=lambda f: f["mjd"])
    if len(res) and not frames:
        raise RuntimeError(f"Found {len(res)} images but could not read any cutouts. See the server log for the first error.")
    p(0.9, "Making the overview image...")
    try:
        ctx = overview(res, c, fov)
    except Exception as e:  # the overview is a nicety: never fail the whole load because of it
        print("Overview image failed:", e)
        ctx = None
    p(0.94, "Looking up Gaia stars...")
    g, gerr = gaia(c, fov) if use_gaia else ([], None)
    p(1.0, "Done")
    return dict(ra=c.ra.deg, dec=c.dec.deg, fov=fov, n=n, scale=SC, frames=frames, gaia=g, gaia_error=gerr, context=ctx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ra"); ap.add_argument("dec")
    ap.add_argument("--fov", type=float, default=5.0, help="field of view, arcmin (max 10)")
    ap.add_argument("--out", default="spherex_cube.json")
    ap.add_argument("--no-gaia", action="store_true")
    ap.add_argument("--tries", type=int, default=6, help="IRSA query attempts")
    a = ap.parse_args()
    out = build(a.ra, a.dec, min(a.fov, 10), not a.no_gaia, a.tries)
    with open(a.out, "w") as fh:
        json.dump(out, fh)
    print(f"Wrote {a.out}: {len(out['frames'])} frames, {out['n']}x{out['n']} px, {len(out['gaia'])} Gaia sources.")


if __name__ == "__main__":
    main()
