"""moex serve refuses callers it shouldn't serve: a foreign Host header (DNS rebinding), browser requests from other
sites (Origin), and never answers with Access-Control-Allow-Origin: *. Normal clients keep working.

  python tools/test_security.py [http://127.0.0.1:8765]
"""
import http.client, json, sys
from urllib.parse import urlparse

u = urlparse(sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8765")
fails = 0

def req(method, path, headers, body=None):
    c = http.client.HTTPConnection(u.hostname, u.port, timeout=120)
    c.putrequest(method, path, skip_host=True)
    for k, v in headers.items(): c.putheader(k, v)
    data = json.dumps(body).encode() if body is not None else b""
    if body is not None: c.putheader("Content-Length", str(len(data)))
    c.endheaders(data)
    r = c.getresponse()
    return r.status, dict((k.lower(), v) for k, v in r.getheaders()), r.read().decode(errors="replace")

def check(name, ok, detail=""):
    global fails
    fails += not ok
    print(f"  {name:62s} {'PASS' if ok else 'FAIL'}  {detail}")

chat = {"messages": [{"role": "user", "content": "Say hi."}], "max_tokens": 5}
local = f"127.0.0.1:{u.port}"
s, h, _ = req("GET", "/health", {"Host": local})
check("no Origin, Host 127.0.0.1 (curl, SDKs): allowed", s == 200 and "access-control-allow-origin" not in h, f"{s}")
s, h, _ = req("GET", "/health", {"Host": f"localhost:{u.port}"})
check("Host localhost: allowed", s == 200, f"{s}")
s, h, b = req("POST", "/v1/chat/completions", {"Host": f"evil.example:{u.port}", "Content-Type": "text/plain"}, chat)
check("Host evil.example (DNS rebinding): refused", s == 403, f"{s} {b[:90]}")
s, h, b = req("GET", "/health", {"Host": f"127.0.0.1.evil.com:{u.port}"})
check("Host 127.0.0.1.evil.com (looks like 127.x): refused", s == 403, f"{s} {b[:90]}")
s, h, _ = req("GET", "/health", {"Host": f"[::1]:{u.port}"})
check("Host [::1]: allowed", s == 200, f"{s}")
s, h, b = req("GET", "/health", {"Host": local, "Origin": "http://127.0.0.1.evil.com"})
check("Origin http://127.0.0.1.evil.com: refused", s == 403 and "access-control-allow-origin" not in h, f"{s} {b[:90]}")
s, h, b = req("GET", "/health", {"Host": local, "Origin": "http://127.0.0.2:3000"})
check("Origin http://127.0.0.2:3000 (loopback address): allowed", s == 200, f"{s}")
s, h, b = req("POST", "/v1/chat/completions", {"Host": local, "Origin": "https://evil.example", "Content-Type": "text/plain"}, chat)
check("Origin https://evil.example (cross-site POST): refused", s == 403, f"{s} {b[:90]}")
s, h, _ = req("OPTIONS", "/v1/chat/completions", {"Host": local, "Origin": "https://evil.example"})
check("preflight from https://evil.example: refused", s == 403, f"{s}")
s, h, b = req("POST", "/v1/chat/completions", {"Host": local, "Origin": "http://localhost:3000", "Content-Type": "application/json"}, chat)
acao = h.get("access-control-allow-origin")
check("Origin http://localhost:3000: allowed, origin echoed (not *)", s == 200 and acao == "http://localhost:3000", f"{s} ACAO={acao}")
print("ALL PASS" if not fails else f"{fails} FAILED")
sys.exit(1 if fails else 0)
