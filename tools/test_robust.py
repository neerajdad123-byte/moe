"""moex serve survives bad clients: a connection that never sends anything, JSON nested 100,000 deep, a huge
Content-Length, an object with 200,000 keys. After each, a normal request must still work.

  python tools/test_robust.py [http://127.0.0.1:8765]
"""
import json, socket, sys, time
from urllib.parse import urlparse
import urllib.request

u = urlparse(sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8765")
BASE = f"http://{u.hostname}:{u.port}"
fails = 0

def check(name, ok, detail=""):
    global fails
    fails += not ok
    print(f"  {name:58s} {'PASS' if ok else 'FAIL'}  {detail}")

def alive(timeout=60):
    try: return urllib.request.urlopen(BASE + "/health", timeout=timeout).status == 200
    except Exception: return False

def raw(data, read=True, timeout=60):
    s = socket.create_connection((u.hostname, u.port), timeout=timeout)
    s.sendall(data)
    out = b""
    if read:
        try:
            while chunk := s.recv(65536): out += chunk
        except socket.timeout: pass
    s.close()
    return out.decode(errors="replace").split("\r\n")[0]

def post(body):
    return raw(b"POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\nContent-Length: " +
               str(len(body)).encode() + b"\r\n\r\n" + body)

# 1. a client that connects and sends nothing must not block everyone else for long
idle = socket.create_connection((u.hostname, u.port))
t = time.time(); ok = alive(timeout=90); dt = time.time() - t
idle.close()
check("idle connection doesn't block the server for long", ok and dt < 40, f"other client served after {dt:.1f}s")

# 1b. a client that trickles its headers (a byte every 5 s, under the 15 s per-read timeout) must not hold the
# server forever: a request has a total deadline
import threading
stop = threading.Event()
def trickle():
    s = socket.create_connection((u.hostname, u.port))
    try:
        for ch in b"GET /health HTTP/1.1\r\nX-Slow: " + b"a" * 1000:
            s.send(bytes([ch]))
            if stop.wait(5): break
    except OSError: pass  # the server dropped us
    s.close()
th = threading.Thread(target=trickle); th.start()
time.sleep(1)
t = time.time(); ok = alive(timeout=120); dt = time.time() - t
stop.set(); th.join()
check("trickling client doesn't block the server for long", ok and dt < 45, f"other client served after {dt:.1f}s")

# 2. deeply nested JSON: refused, no crash
r = post(b"[" * 100000)
check("JSON nested 100,000 deep: refused, server alive", " 400 " in r and alive(), r)

# 3. a Content-Length far beyond the cap: refused at once
r = raw(b"POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Length: 99999999999\r\n\r\n{", timeout=20)
check("Content-Length 100 GB: refused, server alive", " 413 " in r and alive(), r)

# 3b. refused while the client is still sending (body over the cap, headers over 1 MB): the client must read the
# status, not a connection reset (closing a socket with unread data resets it)
def refused_while_sending(head, more):
    s = socket.create_connection((u.hostname, u.port), timeout=30)
    try:
        s.sendall(head)
        for _ in range(8): s.sendall(more)  # 2 MB more than the server reads before refusing
    except OSError: pass
    out = b""
    try:
        while chunk := s.recv(65536): out += chunk
    except OSError as e: out += f" [{type(e).__name__}]".encode()
    s.close()
    return out.decode(errors="replace").split("\r\n")[0]
r = refused_while_sending(b"POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Length: 100000000\r\n\r\n", b"x" * 262144)
check("body over 64 MB, still sending: 413 read", " 413 " in r and "Error" not in r and alive(), r)
r = refused_while_sending(b"GET /health HTTP/1.1\r\nHost: 127.0.0.1\r\nX-Big: " + b"a" * (1 << 20), b"a" * 262144)
check("headers over 1 MB, still sending: 431 read", " 431 " in r and "Error" not in r and alive(), r)

# 4. an object with 200,000 keys parses quickly (it was quadratic)
big = json.dumps({"messages": [{"role": "user", "content": "hi"}], "max_tokens": 1, **{f"k{i}": i for i in range(200000)}}).encode()
t = time.time(); r = post(big); dt = time.time() - t
check("object with 200,000 keys: answered in seconds", " 200 " in r and dt < 15, f"{dt:.1f}s {r}")

# 5. normal request still fine
r = post(json.dumps({"messages": [{"role": "user", "content": "Say hi."}], "max_tokens": 5}).encode())
check("normal request afterwards", " 200 " in r, r)
print("ALL PASS" if not fails else f"{fails} FAILED")
sys.exit(1 if fails else 0)
