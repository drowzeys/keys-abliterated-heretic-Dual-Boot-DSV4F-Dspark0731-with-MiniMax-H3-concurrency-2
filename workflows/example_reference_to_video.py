#!/usr/bin/env python3
"""Queue the granny/cookies r2v clip on a ComfyUI lane.

Built from Tony's own working roko r2v graph (pulled live off the 3090 queue),
so the node wiring, sampler and loaders are his proven ones, not my guesses.
Only the reference image, prompt, seed and output name change.

Usage: crone_r2v.py <host:port> [length] [width] [height]
"""
import json
import sys
import urllib.request

HOST = sys.argv[1] if len(sys.argv) > 1 else "127.0.0.1:8188"
# H3 wants a 17n+5 frame grid. 17*21+5 = 362 frames = 15.08s at 24fps.
LENGTH = int(sys.argv[2]) if len(sys.argv) > 2 else 362
WIDTH = int(sys.argv[3]) if len(sys.argv) > 3 else 1280
HEIGHT = int(sys.argv[4]) if len(sys.argv) > 4 else 720

PROMPT = """integrated_multimodal_description: 8K IMAX. Photorealistic — no 3D render, no game engine, no game-cutscene aesthetic. Cinematography: Roger Deakins x Robert Eggers. Strictly practical light sources physically present on set — one bare failing bulb far down the hallway behind her, weak grey daylight bleeding under a distant door, zero added cinematic fills, no front fill, no rim light from outside frame. The camera is the POV of a person standing still in the hallway; she comes to us. Color 60:30:10, desaturated toward sick yellow-green wallpaper and brown shadow. Physical cine lens, 180° shutter motion blur, very slight handheld breathing in the frame as if the viewer is holding their breath. Realism: pore-level skin, vellus hair, liver spots, translucent papery eyelids, capillary flush, correct contact shadows, gravity and inertia fully respected, the tray has real weight. Continuity identical across cuts, no identity drift, same woman, same dress, same growth, same clouded eye in every shot. 24fps. 8K detail.

Setting: the long narrow hallway of an old country farmhouse. Peeling floral wallpaper, warped wainscoting, a threadbare runner rug over bare floorboards, closed doors on both sides, the ceiling low. The far end of the hallway is almost black. Empty, still, no other people. Backrooms unease — a normal domestic space that is too long, too quiet, and wrong.

Reference picture 1 is a character sheet showing the same elderly woman from three angles — full-body front, full-body back, and a close-up of her face. Frail, stooped, thin arms. White-grey hair pinned up in a loose bun. Deep-set wrinkled face with a large pendulous flesh-toned growth hanging from her right cheek and jaw. Her left eye is clouded white and blind, her right eye pale and watery. She wears a faded blue-grey long-sleeved period work dress with a lace collar, small buttons down the front, a stained cream apron tied at the back, ribbed wool stockings and scuffed black leather shoes. Use this exact face, growth, hair, dress and apron in every shot.

[Shot 1] A locked-off wide POV shot straight down the hallway. She stands at the far dark end, small in frame, holding a tin tray of chocolate chip cookies in both hands. She begins walking toward the camera, extremely slowly, one dragging step at a time, her stooped silhouette passing under the one failing bulb so the light crawls across the growth on her face and then loses her again. She never blinks. She tilts her head and calls down the hallway in a thin, sweet, papery old voice, warm on the surface and wrong underneath: <d>[English] Do you want some cookies?</d> She keeps coming, closing maybe a third of the hallway, her shoes scuffing the runner rug.

[Shot 2] At 00:05.000, hard cut to a tight close-up on the tray, low and near, her hands filling frame. The hands are grotesque — swollen knuckles, skin like wet paper over bone, long yellowed cracked nails packed with dirt, dark liver spots, a weeping split across one thumb, the fingers gripping the tray rim too tightly. The cookies are wrong up close, greasy and unevenly baked, one of them faintly moving. Her arms tremble with the weight. Her voice comes from just above the frame, harder now, the sweetness thinning: <d>[English] I got some chocolate chip cookies.</d>

[Shot 3] At 00:10.000, hard cut to an extreme close-up of her face, far too near the camera, filling the frame — the clouded blind eye, the pendulous growth, the papery skin, her mouth working. She is no longer performing warmth. Her jaw sets, the pale watery eye fixes on the lens, and her voice climbs into a cracked, furious rasp, spit at the teeth: <d>[English] You want Granny's cookies? I got some really tasty cookies for you.</d> She pushes the tray up into the bottom of the frame as she finishes, closing the last of the distance.

Camera: three fixed shots, hard cuts between them, no dissolves, no zooms within shots, no camera moves. The camera never retreats.

overall_soundscape: Environment SFX only, no music, no subtitles. Dead farmhouse room tone with a faint high ringing. The slow drag and scuff of her shoes on the runner rug, floorboards groaning under her weight one step at a time, the tin tray rattling faintly in her shaking hands, her wet shallow breathing between lines, a distant window frame ticking in the wind. Her voice is close-mic'd and dry even in the wide shot, as if she is already next to the listener.

non_diegetic_music: N/A"""


def build(ref_name="crone_ref.jpg", seed=8821, prefix="video/granny_cookies"):
    """Tony's roko graph with the reference/prompt/output swapped.

    Node ids are kept identical to his so the wiring reads the same if we ever
    diff the two. `ref_images` takes a list of LoadImage NODE ids (not the
    usual [id, slot] pairs) — that is how his working graph does it.
    """
    return {
        "201": {"class_type": "LoadImage",
                "inputs": {"image": ref_name, "upload": "image"}},
        "6": {"class_type": "UNETLoader",
              "inputs": {"unet_name": "minimax_h3_fl2va_pruned_int8_convrot.safetensors",
                         "weight_dtype": "default"}},
        "13": {"class_type": "CLIPLoader",
               "inputs": {"clip_name": "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
                          "type": "minimax", "device": "default"}},
        "11": {"class_type": "VAELoader",
               "inputs": {"vae_name": "minimax_h3_video_vae_fp16.safetensors"}},
        "24": {"class_type": "VAELoader",
               "inputs": {"vae_name": "minimax_h3_audio_vae_fp32.safetensors"}},
        "230": {"class_type": "MiniMaxH3ReferenceToVideo",
                "inputs": {"clip": ["13", 0], "vae": ["11", 0], "audio_vae": ["24", 0],
                           "ref_images": ["201"], "prompt": PROMPT,
                           "width": WIDTH, "height": HEIGHT, "length": LENGTH,
                           "ref_image_size": "max"}},
        "16": {"class_type": "BasicGuider",
               "inputs": {"model": ["6", 0], "conditioning": ["230", 0]}},
        "17": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "res_multistep"}},
        "9": {"class_type": "BasicScheduler",
              "inputs": {"model": ["6", 0], "scheduler": "simple",
                         "steps": 20, "denoise": 1.0}},
        "15": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "14": {"class_type": "SamplerCustomAdvanced",
               "inputs": {"noise": ["15", 0], "guider": ["16", 0],
                          "sampler": ["17", 0], "sigmas": ["9", 0],
                          "latent_image": ["230", 1]}},
        "10": {"class_type": "VAEDecode",
               "inputs": {"samples": ["14", 0], "vae": ["11", 0]}},
        # Audio comes off the SAME sampler latent as the video (node 14), not
        # off a third output of 230 — that node only returns CONDITIONING and
        # LATENT. The audio VAE decodes the audio channels out of that latent.
        "23": {"class_type": "VAEDecodeAudio",
               "inputs": {"samples": ["14", 0], "vae": ["24", 0]}},
        "91": {"class_type": "CreateVideo",
               "inputs": {"images": ["10", 0], "audio": ["23", 0], "fps": 24.0}},
        "92": {"class_type": "SaveVideo",
               "inputs": {"video": ["91", 0], "filename_prefix": prefix,
                          "format": "auto", "codec": "auto"}},
    }


if __name__ == "__main__":
    body = json.dumps({"prompt": build(), "client_id": "kai-crone"}).encode()
    req = urllib.request.Request(f"http://{HOST}/prompt", data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            print(json.dumps(json.loads(r.read().decode()), indent=1))
    except urllib.error.HTTPError as e:
        # ComfyUI puts the actual node validation failure in the body, not the status.
        print("HTTP", e.code)
        print(e.read().decode()[:3000])
