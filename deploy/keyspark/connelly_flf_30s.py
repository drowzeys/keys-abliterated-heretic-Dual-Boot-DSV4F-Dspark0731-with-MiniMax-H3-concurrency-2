#!/usr/bin/env python3
"""Parallel dual-H3 ~30s Jennifer Connelly / Career Opportunities — FLF-correct.

User-locked spec (all requirements bundled):
  1. Real Jennifer Connelly likeness from H3_I0_REF (approved 1991 photo).
  2. Match that photo's real skin texture and lighting.
  3. Mechanical horse ROCKS BACK AND FORTH -- she is visibly riding it.
  4. NO areola visible (breasts shown, nipples/areolae kept hidden / out of frame).
  5. Show her BACK side with her LOOKING BACK over her shoulder at camera.
  6. 360-degree walkaround camera orbit around her.
  7. SLOW seductive motion, tight bosom->pubic framing.
  8. FLF-correct: consistent keys first -> parallel arms -> hard-cut (no xfade).
  9. 30 SECONDS (6 arms x ~5.17s = ~31s, 3 parallel waves).

Architecture (mirrors tonyd2wild factory keyframe_dual_flf.py):
  Phase 0  sequential keys K0..K6 on HEAD (.2) -- consistent, identity-chained.
  Phase 2  arms in 3 parallel waves across HEAD(.2)+WORKER(.3): (1,2)(3,4)(5,6).
  Phase 3  hard-cut concat (no xfade) -- seams share exact key frames.

Note: the 360 orbit is staged as an AROUND-THE-PONY arc across the 6 arms -- H3
cannot hold a literal camera orbit from one 5s clip to the next, so the choreography
is described turn-by-turn with matched keyframes, which FLF locks together.

Usage:
  H3_I0_REF=/path/to/face.png python3 connelly_flf_30s.py
Based on tonyd2wild/ds4-h3-video-gen-factory (keyspark dual-boot ext).
"""
from __future__ import annotations

import json, os, shutil, subprocess, sys, threading, time, urllib.error, urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from enhanced_graph import build_enhanced  # noqa: E402

HEAD = os.environ.get("H3_HEAD", "10.100.10.2:8188")
WORKER = os.environ.get("H3_WORKER", "10.100.10.3:8188")
W = int(os.environ.get("H3_W", "864")); H = int(os.environ.get("H3_H", "480"))
KF_FRAMES = int(os.environ.get("H3_LEN_KF", "49"))
ARM_FRAMES = int(os.environ.get("H3_LEN", "124"))   # ~5.17s each
STEPS = int(os.environ.get("STEPS", "20")); KF_STEPS = int(os.environ.get("H3_STEPS_KF", "20"))
SEED = int(os.environ.get("SEED", "33445"))
SPECTRUM = os.environ.get("H3_SPECTRUM", "1") not in ("0","false","no")
UPSCALE = os.environ.get("UPSCALE", "RealESRGAN_x2plus.pth")
if UPSCALE in ("0","none","None"): UPSCALE = None
OUT_DIR = Path(os.environ.get("OUT_DIR", str(Path.home() / "Videos" / "connelly_flf_30s")))
WORK = Path(os.environ.get("WORK_DIR", "/tmp/connelly_flf_30s"))
I0_REF = os.environ.get("H3_I0_REF", "").strip() or None
if I0_REF: I0_REF = str(Path(I0_REF).expanduser().resolve())

N_KEYS, N_ARMS = 7, 6   # 30s

LOOK = """integrated_multimodal_description: photorealistic live-action cinema, 35mm film grain, natural anamorphic lens, 24fps, 180 shutter, warm dim practical retail lighting of a closed department store at night, soft tungsten pools and cool blue moonlight through tall display windows. No cartoon, no anime, no CGI plastic, no airbrushed or retouched skin, no game-engine look, no text, no subtitles, no watermark. The bare SKIN MUST MATCH the reference photograph exactly in texture and tone: real visible pore texture, fine natural creases, soft subsurface scatter, a subtle natural sheen, delicate body hair, faint freckling, pale fair tones under the same warm+cool lighting as the photo. Natural curves and soft volume; a real woman's rounded breasts, waist and hips, NO flat mannequin. Motion discipline: EVERY gesture SLOW and languid, dreamlike, minimal held motion, gentle breathing.

MOTION REQUIREMENT: she is actively RIDING a life-size mechanical bucking horse. The horse visibly ROCKS BACK AND FORTH the whole time (hydraulic servo sway, seesawing rhythm), and she moves with it -- hips and torso gently riding the rocking motion, hand gripping. She never just sits still.

WARDROBE (Career Opportunities): she wears a loose WHITE TANK TOP over her bare torso, and loose low-slung jeans. During the scene she lifts the hem of the white tank top UP HALFWAY to bare her lower ribcage and the UNDER-SIDE / bottom curve of her breasts -- the gentle lower swell of the breasts visible with at most the very BOTTOM EDGE of the areola peeking at the raised hem. NO nipple ever shows -- the nipples and upper areola stay covered just above the tank-top hem, below which only the smooth bare under-swell and (at most) the lowest crescent of the areola is visible. She also rolls the waistband of her jeans DOWN low over her hips to bare the flat lower belly, the very top of her butt crack behind, and the clean top line of her dark pubic hair. NO zipper, NO full unzip -- the tank-top lift and jeans roll-down are the reveal.

Identity lock: the face and figure of a young Jennifer Connelly -- luminous green eyes, strikingly fair pale skin, youthful face, delicate jaw, full soft lips, long dark chestnut curls. She sits astride a chrome-and-fiberglass mechanical pony (sign reading SKIP) in a closed store at night."""

# Per-arm angle in the around-the-pony 360 arc
ANGLE = ["front-left three-quarter", "left profile / side", "back-left shoulder",
         "DIRECTLY FROM BEHIND -- her BACK, looking back over her shoulder",
         "back-right shoulder", "right profile returning to front"]

K_ANGLES = ["front-left three-quarter", "left side", "back-left back angle over shoulder",
            "DIRECTLY BEHIND -- full back view, head turned back over shoulder facing camera",
            "back-right back angle", "right side", "front three-quarter facing camera"]

def key_prompt(i, scene):
    return f"""{LOOK}

Key frame still (near-frozen, minimal motion, high quality), camera at {K_ANGLES[i]}: {scene} overall_soundscape: soft servo hum, her slow breath. No music. No dialogue.
non_diegetic_music: N/A"""

# Tank-top + lift reveal + rolled-down jeans + no-nipple choreography
K_PROMPTS = [
 key_prompt(0,"She sits astride the mechanical pony, half-turned to camera in front-left view, hand on the chrome pommel, the horse gently rocking; she has a knowing, warm half-smile, dark chestnut hair over her shoulder, wearing the loose white tank top (hem still down) and low-slung jeans."),
 key_prompt(1,"Left-side view on the pony: her profile and long hair, hips riding the rock; she has just lifted the white tank top hem UP HALFWAY, baring her smooth lower ribcage and the round UNDER-SIDE of her breasts with no nipple showing (the tank hem covers the nipples just above), fair skin in warm+cool light."),
 key_prompt(2,"Back-left view over her shoulder: she is turned mostly away, looking back at the camera over her shoulder with a sly warm smile, the white tank top lifted halfway revealing the lower curve of her breasts from the side (no nipple), the waistband of her jeans rolled down low showing the top of her butt crack, pony rocking."),
 key_prompt(3,"DIRECTLY from behind on the pony: the full back of her bare shoulders and spine, head turned all the way back to smile warmly at the camera, dark hair falling over one shoulder; the raised white tank top shows her bare lower back and ribs, and her jeans are rolled down low revealing the very TOP of her butt crack, fair pore-textured skin."),
 key_prompt(4,"Back-right view over the shoulder, looking back at camera with a playful smile, bare shoulder and arm, white tank top lifted showing the lower side of her breast (no nipple), jeans rolled down showing the top of her hips and butt crack, pony rocking."),
 key_prompt(5,"Right-side profile on the pony, facing forward again, hand gripping; the white tank top is lifted halfway showing the smooth bare under-swell of her breasts with only the lowest crescent of areola at the hem (no nipple), and her jeans rolled low showing the flat lower belly and the top line of her pubic hair."),
 key_prompt(6,"Front three-quarter facing camera, reclining slightly on the rocking pony, the white tank top lifted halfway baring the round under-side of both breasts with at most the bottom edge of the areola visible and NO nipple shown, and the jeans rolled down low over her hips revealing the flat lower belly and the clean top line of her dark pubic hair; she gives a slow warm memorable smile at camera."),
]

# Arms: orbit + riding + tank-top lift + jeans roll-down + back-look-smile + no nipple
ARM_SCRIPTS = [
 f"""{LOOK}

[Shot 1] ONE continuous slow shot, camera in front-left view orbiting a few degrees toward the side. FIRST FRAME matches key A (she on the pony, tank top and jeans still on, half-turned). LAST FRAME matches key B (left profile, tank top lifted halfway). She rides the pony's slow lazy rock the whole time, hips and torso moving with it. Slowly she grips the white tank top hem and lifts it UP HALFWAY over ~2 seconds, baring her smooth lower ribcage and the round under-side of her breasts with NO nipple showing, then glances to camera with a warm knowing look, biting her lower lip.
timestamps: [0-1.7s] riding the rock, camera drifts sideways; [1.7-3.4s] slow lift of the tank top hem halfway; [3.4-5.17s] lower under-swell of breasts revealed (no nipple), glance back, lip bite. No cuts.
overall_soundscape: hydraulic servo rock, soft cotton shift, her slow breath. No music.
non_diegetic_music: N/A""",
 f"""{LOOK}

[Shot 1] ONE continuous slow shot, camera orbiting to her LEFT profile / side. FIRST FRAME matches key B (tank top lifted). LAST FRAME matches key C (back-left over shoulder). She keeps riding, the pony rocking; the lifted tank top stays raised, showing the bare under-curve of her breasts from the side (no nipple), and she slowly rolls her jeans waistband DOWN low over her hips, revealing the flat lower belly and top of her hips. She begins to turn and glance back over her shoulder to the camera.
timestamps: [0-1.7s] ride + hold lift; [1.7-3.4s] slow jeans roll-down, bare lower belly; [3.4-5.17s] turn toward the back, look-back begins. No cuts.
overall_soundscape: servo rock, denim slide, her slow breath. No music.
non_diegetic_music: N/A""",
 f"""{LOOK}

[Shot 1] ONE continuous slow shot, camera now at her BACK-LEFT shoulder. FIRST FRAME matches key C (back-left). LAST FRAME matches key D (DIRECTLY behind, head turned back to camera). She rides steadily, horse rocking back and forth. The raised white tank top bares her lower back and ribs; her jeans are rolled down low showing the TOP OF HER BUTT CRACK; she turns her head fully back and SMILES WARM AND SEDUCTIVELY AT THE CAMERA, holding it, hair spilling.
timestamps: [0-1.7s] ride, back and butt-crack top revealed; [1.7-3.4s] she turns her head back to camera; [3.4-5.17s] warm seductive smile held at camera while she keeps riding. No cuts.
overall_soundscape: servo rock, denim creak, her slow breath near mic. No music.
non_diegetic_music: N/A""",
 f"""{LOOK}

[Shot 1] ONE continuous slow shot, camera DIRECTLY BEHIND her on the pony -- full back, bare spine and lower back (tank top raised), jeans rolled down revealing the top of her butt crack, head turned all the way back to smile at camera. FIRST FRAME matches key D (directly behind). LAST FRAME matches key E (back-right). The horse rocks back and forth beneath her and she rides with it, hips moving; the sides of her breasts just show around her ribs below the raised tank top with no nipple. She holds a slow, warm, flirty smile straight into the camera.
timestamps: [0-1.7s] hold the behind view, she rides and smiles back; [1.7-3.4s] she shifts, bare lower back and butt-crack top catch light; [3.4-5.17s] she begins to turn toward back-right, smile unchanged. No cuts.
overall_soundscape: servo rock, distant store hum, her slow breathing. No music.
non_diegetic_music: N/A""",
 f"""{LOOK}

[Shot 1] ONE continuous slow shot, camera at her BACK-RIGHT shoulder as she continues the turn. FIRST FRAME matches key E (back-right). LAST FRAME matches key F (right profile). Horse rocking; she rides. She glances back once more with a playful smile, then begins to turn forward, the raised white tank top showing the bare lower curve of her ribs and the side of one breast (no nipple) in the warm+cool light, jeans still rolled low.
timestamps: [0-1.7s] ride + glance back smile; [1.7-3.4s] begin to turn forward; [3.4-5.17s] facing side, lower breast under-curve and ribs in light. No cuts.
overall_soundscape: servo rock, soft cotton, her slow breath. No music.
non_diegetic_music: N/A""",
 f"""{LOOK}

[Shot 1] ONE continuous slow shot completing the orbit back to FRONT three-quarter facing camera, camera holding and barely drifting. FIRST FRAME matches key F (right profile). LAST FRAME matches key G (front three-quarter, reclining, warm smile). The pony rocks back and forth and she rides it. Facing camera, the white tank top stays lifted halfway, baring the round under-side of both breasts with at most the bottom edge of the areola visible and NO nipple, and her jeans rolled down low show the flat lower belly and the clean top line of her dark pubic hair. She bites her lower lip slowly and gives a warm, seductive, memorable smile straight at camera, green eyes locked, chest rising and falling with the rock.
timestamps: [0-1.7s] completing the turn, ride; [1.7-3.4s] tank hem at under-swell (no nipple), pubic-top line shows; [3.4-5.17s] slow lip bite, warm smile held at camera. No cuts.
overall_soundscape: servo rock, her slow breathing, faint store hum. No music.
non_diegetic_music: N/A""",
]

def http_json(url, data=None, timeout=120):
    body = None if data is None else json.dumps(data).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type":"application/json"} if body else {}, method="POST" if body is not None else "GET")
    with urllib.request.urlopen(req, timeout=timeout) as r: return json.loads(r.read().decode())

def upload_image(host, path, name):
    import mimetypes, uuid
    boundary = uuid.uuid4().hex
    raw = path.read_bytes()
    mime = mimetypes.guess_type(str(path))[0] or "image/png"
    parts=[f"--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"{name}\"\r\nContent-Type: {mime}\r\n\r\n".encode(), raw,
           f"\r\n--{boundary}\r\nContent-Disposition: form-data; name=\"overwrite\"\r\n\r\ntrue\r\n".encode(),
           f"--{boundary}\r\nContent-Disposition: form-data; name=\"type\"\r\n\r\ninput\r\n".encode(),
           f"--{boundary}--\r\n".encode()]
    req=urllib.request.Request(f"http://{host}/upload/image", data=b"".join(parts), headers={"Content-Type":f"multipart/form-data; boundary={boundary}"}, method="POST")
    return json.loads(urllib.request.urlopen(req, timeout=120).read().decode()).get("name", name)

def queue(host, graph, cid):
    try:
        r=http_json(f"http://{host}/prompt", {"prompt":graph,"client_id":cid}, timeout=60)
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"queue {host}: {e.read().decode()[:2500]}") from e
    if "error" in r: raise RuntimeError(r)
    return r["prompt_id"]

def wait_done(host, pid, label, timeout=9500):
    t0=time.time(); last=0
    while time.time()-t0<timeout:
        try: hist=http_json(f"http://{host}/history/{pid}", timeout=30)
        except Exception: time.sleep(5); continue
        if pid in hist:
            e=hist[pid]; st=e.get("status",{}); s=st.get("status_str","")
            if s=="success" or e.get("outputs"):
                print(f"  [{label}] SUCCESS {time.time()-t0:.0f}s", flush=True); return e
            if s=="error":
                for m in st.get("messages") or []:
                    if isinstance(m,list) and m and m[0]=="execution_error":
                        raise RuntimeError(f"{label}: {m[1].get('exception_message','')[:1200]}")
                raise RuntimeError(f"{label}: {json.dumps(st)[:1200]}")
        if time.time()-last>20: print(f"  [{label}] {time.time()-t0:.0f}s ...", flush=True); last=time.time()
        time.sleep(4)
    raise TimeoutError(label)

def find_video(entry):
    for _nid,out in (entry.get("outputs") or {}).items():
        for _k,items in out.items():
            if not isinstance(items,list): continue
            for it in items:
                if isinstance(it,dict) and any(it.get("filename","").lower().endswith(e) for e in (".mp4",".webm",".mkv")):
                    return it["filename"], it.get("subfolder",""), it.get("type","output")
    raise RuntimeError(f"no video: {json.dumps(entry.get('outputs'))[:1500]}")

def download(host, fn, sub, typ, dest):
    from urllib.parse import quote
    q=f"filename={quote(fn)}&subfolder={quote(sub)}&type={typ}"
    with urllib.request.urlopen(f"http://{host}/view?{q}", timeout=600) as r: dest.write_bytes(r.read())
    print(f"  saved {dest} ({dest.stat().st_size/1e6:.1f} MB)", flush=True)

def extract_frame(video, out, when="last"):
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd=["ffmpeg","-y","-sseof","-0.05","-i",str(video),"-frames:v","1","-q:v","2",str(out)] if when=="last" else ["ffmpeg","-y","-i",str(video),"-frames:v","1","-q:v","2",str(out)]
    subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"  frame[{when}] -> {out}", flush=True)

def render(host, label, prompt, seed, prefix, length, steps, first, last, upscale=UPSCALE):
    g=build_enhanced(prompt, seed=seed, width=W, height=H, length=length, steps=steps,
        prefix=prefix, first_frame=first or None, last_frame=last or None,
        sage="auto", spectrum=SPECTRUM, upscale=upscale)
    pid=queue(host, g, label)
    print(f"  [{label}] queued on {host} len={length} steps={steps} first={bool(first)} last={bool(last)}", flush=True)
    e=wait_done(host, pid, label)
    fn,sub,typ=find_video(e); dest=WORK/f"{label}.mp4"; download(host,fn,sub,typ,dest); return dest

def prepare_ref():
    if not I0_REF:
        print("  [warn] no H3_I0_REF -- weaker likeness", flush=True); return None
    src=Path(I0_REF)
    if not src.is_file(): raise FileNotFoundError(f"H3_I0_REF not found: {src}")
    norm=WORK/"face_ref.png"
    subprocess.check_call(["ffmpeg","-y","-i",str(src),"-vf","scale=640:960:force_original_aspect_ratio=decrease,pad=640:960:(ow-iw)/2:(oh-ih)/2","-frames:v","1",str(norm)],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    hn=upload_image(HEAD, norm, "face_ref.png"); upload_image(WORKER, norm, "face_ref.png")
    print(f"  identity ref {norm} seeded on BOTH hosts", flush=True)
    return hn

def run_parallel(jobs):
    results={}
    def run(i, fn):
        try: results[i]=fn()
        except BaseException as ex: results[i]=ex
    ts=[threading.Thread(target=run, args=(i,f)) for i,(_,_,f) in enumerate(jobs)]
    t0=time.time()
    for t in ts: t.start()
    for t in ts: t.join()
    wall=time.time()-t0
    for i in range(len(jobs)):
        if isinstance(results.get(i), BaseException): raise results[i]
    print(f"  parallel wave wall {wall:.0f}s", flush=True)
    return [results[i] for i in range(len(jobs))]

def main():
    t_all=time.time()
    WORK.mkdir(parents=True, exist_ok=True); OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"FLF 30s  spectrum={SPECTRUM} upscale={UPSCALE} keys={N_KEYS} arms={N_ARMS}", flush=True)
    for host,name in ((HEAD,".2"),(WORKER,".3")):
        try:
            s=http_json(f"http://{host}/system_stats", timeout=6)
            print(f"H3 {name} {host} up ram_free={s['system']['ram_free']/1e9:.1f}G", flush=True)
        except Exception as e:
            print(f"H3 {name} DOWN: {e}", file=sys.stderr); sys.exit(1)
    for host in (HEAD,WORKER):
        try: http_json(f"http://{host}/interrupt",{},timeout=10); http_json(f"http://{host}/queue",{"clear":True},timeout=10)
        except Exception: pass

    face=prepare_ref()
    print(f"\n=== PHASE 0: sequential keyframes K0..K{N_KEYS-1} on {HEAD} (each re-anchored to the locked face ref to stop identity drift) ===", flush=True)
    keys=[]
    for i,kp in enumerate(K_PROMPTS):
        label=f"K{i}"
        # CRITICAL: seed EVERY keyframe from the SAME approved face reference, not the
        # previous key's (possibly degraded) last frame. This locks identity at K0's
        # quality for all keys instead of letting the likeness drift down the chain.
        vid=render(HEAD,label,kp,SEED+i,f"video/kf_{label}",KF_FRAMES,KF_STEPS,face,None,None)
        key_png=WORK/f"K{i}.png"; extract_frame(vid,key_png,"last")
        shutil.copy2(key_png, OUT_DIR/f"key_{label}.png")
        kn=upload_image(HEAD,key_png,f"K{i}.png"); upload_image(WORKER,key_png,f"K{i}.png")
        keys.append(kn)
    print(f"  phase0 wall {time.time()-t_all:.0f}s -- {N_KEYS} identity-locked keys", flush=True)

    print(f"\n=== PHASE 2: parallel motion arms ({N_ARMS//2} waves x 2 boxes) ===", flush=True)
    arms=[]
    for i in range(N_ARMS):
        host=HEAD if i%2==0 else WORKER
        fn=i+1
        arms.append((host, f"arm{fn}", lambda i=i,host=host,fn=fn: render(
            host,f"arm{fn}",ARM_SCRIPTS[i],SEED+100+i,f"video/arm{fn}",ARM_FRAMES,STEPS,keys[i],keys[i+1],UPSCALE)))
    all_arms=[]
    for w in range((N_ARMS+1)//2):
        wave=arms[w*2:w*2+2]
        all_arms += run_parallel(wave)
    for label,p in zip([f"arm{i+1}" for i in range(N_ARMS)], all_arms):
        shutil.copy2(p, OUT_DIR/f"{label}.mp4")

    print("\n=== PHASE 3: hard-cut concat (no xfade) ===", flush=True)
    lst=WORK/"concat_all.txt"
    lst.write_text("\n".join(f"file '{p.resolve()}'" for p in all_arms))
    final=OUT_DIR/"connelly_flf_30s.mp4"
    subprocess.check_call(["ffmpeg","-y","-f","concat","-safe","0","-i",str(lst),
        "-c:v","libx264","-crf","16","-preset","medium","-c:a","aac","-b:a","192k","-movflags","+faststart",str(final)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"  FINAL {final}", flush=True)
    total=time.time()-t_all
    (OUT_DIR/"TIMING.txt").write_text(f"mode=FLF-30s orbit riding no-areola back-smile\ntotal_wall_s={total:.1f}\nkeys={N_KEYS} arms={N_ARMS}\nnote=3 parallel waves; hard-cut concat\n")
    dur=float(subprocess.check_output(["ffprobe","-v","error","-show_entries","format=duration","-of","default=noprint_wrappers=1:nokey=1",str(final)],text=True).strip())
    print(f"\nDone in {total:.0f}s wall (~{total/60:.1f} min)  final duration={dur:.2f}s", flush=True)
    print(f"  {final}", flush=True)

if __name__=="__main__":
    main()
